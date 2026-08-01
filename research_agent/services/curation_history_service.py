from __future__ import annotations

from research_agent.api_app.schemas import CurationSelectFromHistoryRequest, CurationSelectFromHistoryResponse, CurationTurnResponse
from research_agent.api_app.serializers import _turn_result_to_response
from research_agent.curation_loop import start_curation_turn
from research_agent.curation_session import (
    _session_to_dict,
    load_curation_session,
    reopen_curation_session,
    save_curation_session,
    select_paper_from_history,
)
from research_agent.services.curation_helpers import _curation_config
from research_agent.services.errors import ServiceError


def select_from_history(session_id: str, req: CurationSelectFromHistoryRequest, cp) -> CurationSelectFromHistoryResponse:
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
        raise ServiceError(404, "session_id not found")
    try:
        select_paper_from_history(session, req.paper_id)
    except ValueError as exc:
        raise ServiceError(400, str(exc)) from exc
    save_curation_session(session, session_id, cp)
    return CurationSelectFromHistoryResponse(session_id=session_id, selected_paper_ids=session.selected_paper_ids)


def reopen_curation(session_id: str, cp) -> CurationTurnResponse:
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
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    try:
        reopen_curation_session(session)
    except ValueError as exc:
        raise ServiceError(400, str(exc)) from exc

    target_count = session.target_count
    result = start_curation_turn(session_id, cp, _session_to_dict(session), config=_curation_config())
    return _turn_result_to_response(session_id, target_count, result)
