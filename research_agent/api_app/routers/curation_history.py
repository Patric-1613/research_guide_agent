from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.curation_loop import start_curation_turn
from research_agent.curation_session import (
    _session_to_dict,
    load_curation_session,
    reopen_curation_session,
    save_curation_session,
    select_paper_from_history,
)

router = APIRouter()


@router.post("/curation/{session_id}/select-from-history", response_model=api.CurationSelectFromHistoryResponse)
def curation_select_from_history(
    session_id: str, req: api.CurationSelectFromHistoryRequest, cp=Depends(api.get_curation_checkpointer),
) -> api.CurationSelectFromHistoryResponse:
    """curation-turn-history Phase 9c: retroactively add a paper seen in
    an earlier turn, without a new search -- lets a user unstuck from a
    curation that ended short of their target (e.g. the pool genuinely
    ran dry) by picking from what they already saw, even from Report/Chat
    mode. SYNTHESIZE-STAGE ONLY -- see select_paper_from_history()'s own
    docstring for exactly why (out-of-band writes while a real interrupt
    is pending corrupt its bookkeeping, the same failure mode
    curation-refinement-and-auto-offer Phase 6f already found once).
    Picking from history while still curating goes through
    /curation/{session_id}/picks instead (Phase 9d)."""
    session = load_curation_session(session_id, cp)
    if session is None:
        raise HTTPException(status_code=404, detail="session_id not found")
    try:
        select_paper_from_history(session, req.paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    save_curation_session(session, session_id, cp)
    return api.CurationSelectFromHistoryResponse(session_id=session_id, selected_paper_ids=session.selected_paper_ids)


@router.post("/curation/{session_id}/reopen", response_model=api.CurationTurnResponse)
def curation_reopen(session_id: str, cp=Depends(api.get_curation_checkpointer)) -> api.CurationTurnResponse:
    """curation-editable-until-locked Phase 10c: reopens a review that WAS
    explicitly stopped (stage=="synthesize") back into active curation --
    the counterpart to Phase 10b's routing change, which made stopping
    the ONLY thing that ever locks a review. Gated on reopen_curation_
    session()'s own rules (still curating / report already generated /
    chat already started, in that order) -- see its docstring for why. On
    success, re-invokes start_curation_turn() fresh on the same
    thread_id: not a Command(resume=...) (there's no pending interrupt
    to resume once a real stop has already run through END), just a
    plain graph.invoke() that picks up from wherever cursor/
    selected_paper_ids left off, verified empirically before this was
    implemented."""
    with api._upstream_error_guard("curation_reopen"):
        session = load_curation_session(session_id, cp)
        if session is None:
            raise HTTPException(status_code=404, detail="session_id not found")
        try:
            reopen_curation_session(session)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        target_count = session.target_count
        result = start_curation_turn(session_id, cp, _session_to_dict(session), config=api._curation_config())
        return api._turn_result_to_response(session_id, target_count, result)
