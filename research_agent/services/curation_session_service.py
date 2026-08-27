from __future__ import annotations

from research_agent.api_app.schemas import (
    ChatTurn,
    CurationDeleteResponse,
    CurationReviewSummary,
    CurationStateResponse,
    ReferenceEntry,
)
from research_agent.api_app.serializers import (
    _lanes_out,
    _paper_out_from_batch_entry,
    _paper_to_out,
    _report_to_out,
    _report_version_to_summary,
    _turn_history_out,
    _web_article_to_out,
)
from research_agent.curation_chat import derive_chat_references
from research_agent.curation_loop import get_curation_state
from research_agent.curation_session import delete_curation_session, list_curation_sessions, load_curation_session
from research_agent.report import get_active_report_version


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
    # report-quality Phase R3.2 Chunk 2: derived fresh from session.
    # chat_history on every call, never persisted -- chat_history below
    # is this derivation's own rewritten copy (chat-local numeric [N]
    # markers), never session.chat_history directly, so a client always
    # sees resolved citations, not the raw [Paper N]/[Web N] the model
    # wrote. Independent of report.references entirely -- see derive_
    # chat_references' own docstring.
    chat = derive_chat_references(session)
    return CurationStateResponse(
        session_id=session_id, topic=session.topic, display_title=session.display_title,
        stage=session.stage, target_count=session.target_count,
        selected_paper_ids=session.selected_paper_ids,
        selected_papers=[_paper_to_out(p) for p in session.selected_papers],
        pending_batch=[_paper_out_from_batch_entry(e) for e in pending_batch] if pending_batch is not None else None,
        refilled=state.get("refilled", False),
        reserve_remaining=max(0, session.remaining()),
        refinement_notes=list(session.refinement_notes),
        report=_report_to_out(session.report, get_active_report_version(session)) if session.report is not None else None,
        chat_history=[ChatTurn(**turn) for turn in chat["chat_history"]],
        web_articles_added=[_web_article_to_out(a) for a in session.web_articles_added],
        pending_web_offer=session.pending_web_offer,
        pending_report_update=session.pending_report_update,
        turn_history=_turn_history_out(session.turn_history),
        stop_reason=session.stop_reason,
        # report-quality Phase R3
        report_versions=[
            _report_version_to_summary(v, session.active_report_version_id) for v in session.report_versions
        ],
        active_report_version_id=session.active_report_version_id,
        # report-quality Phase R3.2 Chunk 2
        chat_references=[ReferenceEntry(**r) for r in chat["references"]],
        # Research Lanes (RL4): frozen lane set + cumulative provenance +
        # recomputed counts. All empty for a single-query / pre-RL4
        # session (session.lanes deserializes to [] via RL1's .get()
        # default). turn_history entries already carry their own frozen
        # per-turn snapshot via _turn_history_out above.
        lanes=_lanes_out(session.lanes),
        paper_lane_ids={pid: list(lids) for pid, lids in session.paper_lane_ids.items()},
        lane_result_counts=dict(session.lane_result_counts),
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
