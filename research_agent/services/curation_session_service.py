from __future__ import annotations

from research_agent.api_app.schemas import ChatTurn, CurationDeleteResponse, CurationReviewSummary, CurationStateResponse
from research_agent.api_app.serializers import (
    _paper_out_from_batch_entry,
    _paper_to_out,
    _report_to_out,
    _turn_history_out,
    _web_article_to_out,
)
from research_agent.curation_loop import get_curation_state
from research_agent.curation_session import delete_curation_session, list_curation_sessions, load_curation_session


def list_reviews(cp) -> list[CurationReviewSummary]:
    return [CurationReviewSummary(**s) for s in list_curation_sessions(cp)]


def get_state(session_id: str, cp) -> CurationStateResponse | None:
    """The refresh-persistence endpoint (Phase 6d's whole point): every
    field a client needs to fully redraw the UI comes from here, read
    straight from the checkpointer — nothing about this response depends
    on any prior request having happened in the same browser session."""
    state = get_curation_state(session_id, cp)
    if state is None:
        return None

    session = state["session"]
    pending_batch = state["pending_batch"]
    return CurationStateResponse(
        session_id=session_id, topic=session.topic, display_title=session.display_title,
        stage=session.stage, target_count=session.target_count,
        selected_paper_ids=session.selected_paper_ids,
        selected_papers=[_paper_to_out(p) for p in session.selected_papers],
        pending_batch=[_paper_out_from_batch_entry(e) for e in pending_batch] if pending_batch is not None else None,
        refilled=state.get("refilled", False),
        reserve_remaining=max(0, session.remaining()),
        refinement_notes=list(session.refinement_notes),
        report=_report_to_out(session.report) if session.report is not None else None,
        chat_history=[ChatTurn(**turn) for turn in session.chat_history],
        web_articles_added=[_web_article_to_out(a) for a in session.web_articles_added],
        pending_web_offer=session.pending_web_offer,
        pending_report_update=session.pending_report_update,
        turn_history=_turn_history_out(session.turn_history),
        stop_reason=session.stop_reason,
    )


def delete_session(session_id: str, cp) -> CurationDeleteResponse | None:
    """curation-review-management Phase 8, item 1: permanently deletes a
    review -- confirmed no delete/abandon concept existed anywhere in the
    codebase before this. 404s on an unknown session_id first (same
    existence-check convention as report/chat below), rather than silently
    "succeeding" on a delete_thread() call that would have matched zero
    rows either way -- a caller should be told the id it tried to act on
    doesn't exist."""
    session = load_curation_session(session_id, cp)
    if session is None:
        return None
    delete_curation_session(session_id, cp)
    return CurationDeleteResponse(session_id=session_id, deleted=True)
