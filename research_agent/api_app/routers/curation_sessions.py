from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.curation_loop import get_curation_state
from research_agent.curation_session import delete_curation_session, list_curation_sessions, load_curation_session

router = APIRouter()


# Registered BEFORE GET /curation/{session_id} below — Starlette matches
# routes in registration order, so /curation/reviews must come first or a
# request for it would match {session_id}="reviews" instead of this route.
@router.get("/curation/reviews", response_model=list[api.CurationReviewSummary])
def curation_list_reviews(cp=Depends(api.get_curation_checkpointer)) -> list[api.CurationReviewSummary]:
    return [api.CurationReviewSummary(**s) for s in list_curation_sessions(cp)]


@router.get("/curation/{session_id}", response_model=api.CurationStateResponse)
def curation_get_state(session_id: str, cp=Depends(api.get_curation_checkpointer)) -> api.CurationStateResponse:
    """The refresh-persistence endpoint (Phase 6d's whole point): every
    field a client needs to fully redraw the UI comes from here, read
    straight from the checkpointer — nothing about this response depends
    on any prior request having happened in the same browser session."""
    state = get_curation_state(session_id, cp)
    if state is None:
        raise HTTPException(status_code=404, detail="session_id not found")

    session = state["session"]
    pending_batch = state["pending_batch"]
    return api.CurationStateResponse(
        session_id=session_id, topic=session.topic, display_title=session.display_title,
        stage=session.stage, target_count=session.target_count,
        selected_paper_ids=session.selected_paper_ids,
        selected_papers=[api._paper_to_out(p) for p in session.selected_papers],
        pending_batch=[api._paper_out_from_batch_entry(e) for e in pending_batch] if pending_batch is not None else None,
        refilled=state.get("refilled", False),
        reserve_remaining=max(0, session.remaining()),
        refinement_notes=list(session.refinement_notes),
        report=api._report_to_out(session.report) if session.report is not None else None,
        chat_history=[api.ChatTurn(**turn) for turn in session.chat_history],
        web_articles_added=[api._web_article_to_out(a) for a in session.web_articles_added],
        pending_web_offer=session.pending_web_offer,
        pending_report_update=session.pending_report_update,
        turn_history=api._turn_history_out(session.turn_history),
        stop_reason=session.stop_reason,
    )


@router.delete("/curation/{session_id}", response_model=api.CurationDeleteResponse)
def curation_delete(session_id: str, cp=Depends(api.get_curation_checkpointer)) -> api.CurationDeleteResponse:
    """curation-review-management Phase 8, item 1: permanently deletes a
    review -- confirmed no delete/abandon concept existed anywhere in the
    codebase before this. 404s on an unknown session_id first (same
    existence-check convention as report/chat below), rather than silently
    "succeeding" on a delete_thread() call that would have matched zero
    rows either way -- a caller should be told the id it tried to act on
    doesn't exist."""
    session = load_curation_session(session_id, cp)
    if session is None:
        raise HTTPException(status_code=404, detail="session_id not found")
    delete_curation_session(session_id, cp)
    return api.CurationDeleteResponse(session_id=session_id, deleted=True)
