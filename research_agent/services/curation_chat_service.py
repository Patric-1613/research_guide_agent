from __future__ import annotations

from typing import AsyncIterator

from fastapi.responses import StreamingResponse

import research_agent.api as api
from research_agent.api_app.schemas import (
    ChatTurn,
    CitedPaperOut,
    CitedWebArticleOut,
    CurationChatAddToReportRequest,
    CurationChatAddToReportResponse,
    CurationChatDeleteRequest,
    CurationChatDeleteResponse,
    CurationChatEditRequest,
    CurationChatEditResponse,
    CurationChatRequest,
    CurationChatResponse,
)
import research_agent.telemetry as telemetry
from research_agent.api_app.serializers import _report_to_out
from research_agent.chat_streaming import build_done_event, build_error_event
from research_agent.config import get_usage_policy
from research_agent.curation_chat_streaming import HandledStreamFailure, stream_curation_chat_turn
from research_agent.curation_session import load_curation_session, save_curation_session
from research_agent.services.errors import ServiceError
from research_agent.session_limits import check_chat_turn_capacity
from research_agent.usage_guard import guard_paid_action, open_admission_and_lease_for_streaming


def answer_curation_chat(session_id: str, req: CurationChatRequest, cp) -> CurationChatResponse:
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    # Usage Protection M2.2C Part D: checked before admission/lease/
    # paid_action/any provider call -- a normal chat turn only ever
    # appends, never truncates, so the cap is checked against the
    # session's current chat_history as-is.
    check_chat_turn_capacity(session.chat_history, get_usage_policy())
    # A single "curation_chat" action covers the whole turn, including any
    # nested second ask (web-offer accept) or report-regeneration
    # (report-update-offer accept) chat_turn() triggers internally -- "first
    # active action wins" means neither ever opens its own top-level row.
    # Usage Protection M2.2A: session-scoped, so full hourly/daily/global
    # admission plus the shared expensive-action lease -- this single
    # guard covers the nested web-offer/report-update-offer work too,
    # since those never open their own top-level paid_action.
    with guard_paid_action("curation_chat", subject=("session", session_id), use_lease=True):
        try:
            result = api.chat_turn(session, req.message, client=api._state["client"])
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
    save_curation_session(session, session_id, cp)

    return CurationChatResponse(
        answer=result["answer"], answerable=result["answerable"],
        cited_papers=[CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in result["cited_papers"]],
        cited_web_articles=[CitedWebArticleOut(url=a.url, title=a.title) for a in result["cited_web_articles"]],
        web_offer_made=result.get("web_offer_made", False),
        web_offer_declined=result.get("web_offer_declined", False),
        web_search_used=result.get("web_search_used", False),
        new_web_articles_found=result.get("new_web_articles_found"),
        report_update_offer_made=result.get("report_update_offer_made", False),
        report_update_declined=result.get("report_update_declined", False),
        report_updated=result.get("report_updated", False),
        chat_history=[ChatTurn(**turn) for turn in session.chat_history],
    )


def stream_answer_curation_chat(session_id: str, req: CurationChatRequest, cp) -> StreamingResponse:
    """Usage Protection M4.2A Part E: the streaming counterpart to
    `answer_curation_chat` above -- `POST /curation/{session_id}/chat/
    stream`. Every pre-guard check below (session existence, chat-turn
    capacity, stage readiness) runs BEFORE the guard opens and BEFORE
    any `StreamingResponse` is constructed, exactly mirroring
    `answer_curation_chat`'s own ordering -- a streaming generator body
    only ever starts running once HTTP headers may already be
    committed, so every rejection that must become a clean 409/429/503/
    400 JSON response has to be resolved here, synchronously, first.

    Uses `usage_guard.open_admission_and_lease_for_streaming` (NOT
    `guard_paid_action`) -- admission + the lease are acquired here,
    before the response is built, and the lease is released later,
    inside the streaming generator's own `finally`, once the streamed
    work (model execution AND final session persistence) has fully
    finished. `telemetry.paid_action` is deliberately opened ENTIRELY
    inside `event_stream()` below, via a plain `with` statement --
    never split across this function and the generator -- see `open_
    admission_and_lease_for_streaming`'s own docstring for the real,
    empirically-confirmed `contextvars` hazard that would result from
    doing otherwise.

    Lifecycle-hardening checkpoint: `stream_curation_chat_turn`'s own
    `HandledStreamFailure` is caught HERE, OUTSIDE the `with telemetry.
    paid_action(...):` block below -- letting it propagate OUT of that
    block first is what makes `telemetry.paid_action`'s own `except
    Exception` branch record the top-level `paid_actions` row as
    `outcome="error"`, which simply yielding `error`/`done` and
    returning normally from inside that block never did. Only once that
    outcome has been recorded does this function convert the exception
    into the safe `error`+`done` SSE pair the client actually sees --
    never exception text, never a second telemetry write. `asyncio.
    CancelledError` is not caught here at all: it is left to propagate
    all the way out of `event_stream()`, so `telemetry.paid_action`
    records `outcome="cancelled"` (never `"error"`) and no `completed`/
    `error` event is ever emitted for it.
    """
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    check_chat_turn_capacity(session.chat_history, get_usage_policy())
    if session.stage != "synthesize":
        raise ServiceError(
            400,
            f"Session is not ready for chat (stage={session.stage!r}, expected 'synthesize') -- "
            "curation must finish (target met, user stopped, or topic exhausted) before chatting.",
        )

    lease_handle = open_admission_and_lease_for_streaming("curation_chat", subject=("session", session_id), use_lease=True)

    async def event_stream() -> AsyncIterator[str]:
        try:
            try:
                with telemetry.paid_action("curation_chat", subject_type="session", subject_id=session_id):
                    async for event in stream_curation_chat_turn(
                        session, req.message, session_id=session_id, checkpointer=cp,
                        sync_client=api._state["client"], async_client=api._state["async_client"],
                    ):
                        yield event.to_sse()
            except HandledStreamFailure as exc:
                # telemetry.paid_action's own __exit__ has ALREADY run
                # by this point (it wraps the `async for` above, not
                # this except clause) and already recorded outcome=
                # "error" -- this only ever produces the client-facing
                # SSE frames, never a second telemetry write, and never
                # raises again (no exception escapes event_stream after
                # this, so Starlette sees a clean, normal stream end).
                yield build_error_event(exc.reason_code, exc.message).to_sse()
                yield build_done_event().to_sse()
        finally:
            # Always runs after telemetry.paid_action's own __exit__
            # above (a `finally` on the OUTER try only executes once
            # the inner `with` block has already fully exited) -- the
            # lease is released only once the wrapped action's own
            # success/error/cancelled outcome has already been
            # recorded, never before.
            lease_handle.release()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


def delete_curation_chat_exchanges(session_id: str, req: CurationChatDeleteRequest, cp) -> CurationChatDeleteResponse:
    if not req.exchange_ids:
        raise ServiceError(400, "exchange_ids must not be empty")
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")

    deleted_exchange_ids, report_possibly_stale = api.delete_chat_exchanges(session, req.exchange_ids)
    save_curation_session(session, session_id, cp)

    return CurationChatDeleteResponse(
        chat_history=[ChatTurn(**turn) for turn in session.chat_history],
        deleted_exchange_ids=deleted_exchange_ids,
        report_possibly_stale=report_possibly_stale,
    )


def add_curation_chat_exchanges_to_report(
    session_id: str, req: CurationChatAddToReportRequest, cp,
) -> CurationChatAddToReportResponse:
    """curation-chat-add-to-report Phase 4. Mutates session.report_
    approved_web_article_urls and each eligible exchange's
    added_to_report ONLY after regenerate_report_with_approved_web_
    sources succeeds -- a raised exception (ValueError from a precondition,
    or anything else from the LLM call) propagates before either mutation,
    structurally guaranteeing a failed regeneration never marks anything
    approved/added.
    """
    if not req.exchange_ids:
        raise ServiceError(400, "exchange_ids must not be empty")
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    if session.report is None:
        raise ServiceError(400, "Generate a report first before adding web sources to it.")

    eligible_ids, skipped_ids = api.select_eligible_exchanges_for_report(session, req.exchange_ids)
    if not eligible_ids:
        raise ServiceError(400, "No eligible web-backed exchanges to add to the report.")

    newly_approved_urls = api.cited_web_article_urls_for_exchanges(session, eligible_ids)
    approved_web_articles = api.resolve_approved_web_articles_for_regeneration(session, newly_approved_urls)

    # Usage Protection M2.2A: chat-triggered report update -- a separate
    # top-level guarded action from curation_chat above (a distinct HTTP
    # endpoint, not nested inside a chat turn), same lease group.
    with guard_paid_action("report_regenerate", subject=("session", session_id), use_lease=True):
        try:
            new_report = api.regenerate_report_with_approved_web_sources(
                session, approved_web_articles, client=api._state["client"],
            )
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc

    # report-quality Phase R3: appended as a new version (generation_
    # reason=api.GENERATION_REASON_CHAT_ADD_TO_REPORT), not assigned to
    # session.report directly -- append_report_version keeps session.
    # report mirrored to it as a side effect, same as every other
    # report-mutation call site.
    api.append_report_version(session, new_report, api.GENERATION_REASON_CHAT_ADD_TO_REPORT)
    api.approve_web_article_urls(session, newly_approved_urls)
    api.mark_exchanges_added_to_report(session, eligible_ids)
    save_curation_session(session, session_id, cp)

    return CurationChatAddToReportResponse(
        report=_report_to_out(session.report, api.get_active_report_version(session)),
        chat_history=[ChatTurn(**turn) for turn in session.chat_history],
        added_exchange_ids=eligible_ids,
        skipped_exchange_ids=skipped_ids,
        source_count=len(newly_approved_urls),
    )


def edit_curation_chat_exchange(session_id: str, req: CurationChatEditRequest, cp) -> CurationChatEditResponse:
    if not req.exchange_id:
        raise ServiceError(400, "exchange_id must not be empty")
    if not req.question or not req.question.strip():
        raise ServiceError(400, "question must not be empty")
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")

    # Usage Protection M2.2C Part D: edit_chat_exchange (api.py/curation_
    # chat.py) is documented as truncate-and-regenerate, not an in-place
    # edit -- it removes the edited exchange and everything
    # chronologically after it, then appends exactly one fresh pair.
    # Checking the cap against the raw PRE-edit count would incorrectly
    # reject an edit to an OLD exchange that actually reduces the total
    # turn count; this simulates the same truncation (the identical,
    # simple, stable predicate edit_chat_exchange itself uses -- not a
    # fragile duplication of complex logic) and only rejects if even the
    # one fresh replacement pair would exceed the cap. An unknown
    # exchange_id (user_idx is None) fails validation inside
    # edit_chat_exchange itself right after with a 400, regardless of
    # this check's outcome.
    user_idx = next(
        (i for i, turn in enumerate(session.chat_history)
         if turn.get("role") == "user" and turn.get("exchange_id") == req.exchange_id),
        None,
    )
    history_after_truncation = session.chat_history[:user_idx] if user_idx is not None else session.chat_history
    check_chat_turn_capacity(history_after_truncation, get_usage_policy())

    # Usage Protection M2.2A: edit-and-rerun -- re-invokes the model for a
    # single exchange, same guard shape as answer_curation_chat above.
    with guard_paid_action("curation_chat", subject=("session", session_id), use_lease=True):
        try:
            result, report_possibly_stale = api.edit_chat_exchange(
                session, req.exchange_id, req.question, client=api._state["client"],
            )
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
    save_curation_session(session, session_id, cp)

    return CurationChatEditResponse(
        answer=result["answer"], answerable=result["answerable"],
        cited_papers=[CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in result["cited_papers"]],
        cited_web_articles=[CitedWebArticleOut(url=a.url, title=a.title) for a in result["cited_web_articles"]],
        web_offer_made=result.get("web_offer_made", False),
        web_offer_declined=result.get("web_offer_declined", False),
        web_search_used=result.get("web_search_used", False),
        new_web_articles_found=result.get("new_web_articles_found"),
        report_update_offer_made=result.get("report_update_offer_made", False),
        report_update_declined=result.get("report_update_declined", False),
        report_updated=result.get("report_updated", False),
        chat_history=[ChatTurn(**turn) for turn in session.chat_history],
        report_possibly_stale=report_possibly_stale,
    )
