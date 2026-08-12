"""Usage Protection M4.2A Part D: the production curation-chat
streaming domain service -- one async generator per turn, yielding the
shared M4.1 SSE event vocabulary (research_agent/chat_streaming.py) in
the fixed order `started -> phase* -> delta* -> completed -> done` on
success, or `started -> phase* -> delta* -> error -> done` on a
recognized, safely-reportable failure.

**No admission/lease/HTTP awareness here.** This module assumes the
caller (`services/curation_chat_service.py::stream_answer_curation_chat`)
has ALREADY: resolved `session_id` to a real session (404 otherwise),
checked chat-turn capacity, and acquired admission + the session lease
(`usage_guard.open_admission_and_lease_for_streaming`) -- exactly
mirroring `answer_curation_chat`'s own pre-guard checks for the
existing synchronous endpoint. This module's own `started` event is
the first thing yielded specifically because everything before it
(validation, admission, the lease) has already resolved successfully
by the time it runs.

This function is also expected to run entirely INSIDE its caller's own
`with telemetry.paid_action("curation_chat", ...):` block -- opened
and closed by the caller itself, around the whole `async for event in
stream_curation_chat_turn(...):` loop, never split across the caller
and this generator (see `usage_guard.open_admission_and_lease_for_
streaming`'s own docstring for the real `contextvars` hazard that
motivates this).

**Two execution strategies for one turn**, matching
`curation_chat._chat_turn_impl`'s own exact branching
(pending_web_offer/pending_report_update accept/decline/other, plain
fallback) with ZERO reimplementation of its node/retrieval/citation/
filtering logic:

1. **True incremental streaming** (`_stream_fresh_answer`) -- every
   branch that would call `curation_chat.ask_in_session` for a FRESH
   answer (the plain fallback, and the decline/other replies to either
   pending offer): runs `research_agent.qa.prepare_qa_turn` (M4.2A
   Part C, a shared orchestration helper built from the SAME node/
   routing functions `research_agent.qa`'s own compiled graph uses) to
   reach the exact point real answer generation would begin, then
   streams the model's own answer via M4.1's
   `chat_streaming.stream_chat_answer` adapter.

2. **Whole-turn synchronous fallback** (`_stream_via_sync_fallback`)
   -- the two "accept" branches (`_accept_web_offer`,
   `_accept_report_update`) do real work this phase is explicitly out
   of scope for streaming progress on (a live web search + insertion-
   time relevance filtering; a full report regeneration -- M4.2A's own
   scope exclusions rule out "report-generation progress"/"report
   prose streaming" and "new refinement behavior"). Rather than
   partially reimplementing either, this reuses
   `curation_chat.chat_turn()` completely unmodified, off the event
   loop via `asyncio.to_thread`, and reports its one final answer as a
   single `delta` before `completed` -- still honest, still
   "phase-first, answer may arrive as one large delta" per this
   phase's own product contract, just never claiming mid-flight
   progress for operations this module does not instrument.

**Persistence commit point.** `save_curation_session` (a LangGraph
checkpointer write) is the one and only mutation of durable state this
module ever performs, always immediately before the `completed` event,
always wrapped in `asyncio.shield(...)` (see `_persist_and_complete`)
-- once that call has actually been scheduled onto its worker thread,
the write itself runs to completion regardless of whether the
awaiting coroutine is later cancelled (Python cannot forcibly stop a
running thread; `asyncio.shield` only protects the AWAITING coroutine
from having the inner call torn down early, so it still gets a chance
to react normally in the common case). This module never claims a
cancellation can abort in-flight threadpool work -- it only guarantees
the write itself is a single, already-atomic checkpoint operation, and
that `completed` is never emitted before that write has genuinely
succeeded.

**Cancellation.** A genuine `asyncio.CancelledError` from request
disconnection propagates through this module's `await`/`async for`
points completely untouched wherever this module itself does not
intercept it, EXCEPT for one documented case: M4.1's own
`stream_chat_answer` catches `CancelledError` internally and re-raises
it as an ordinary `ChatAnswerStreamError(reason_code="cancelled", ...)`
(a deliberate, documented M4.1 design -- see that function's own
docstring). This module detects exactly that reason code and raises a
FRESH `asyncio.CancelledError` in response, so real task cancellation
still propagates to the caller's own enclosing `telemetry.paid_action`
block (which records the "cancelled" outcome) and, after that, to
`usage_guard.StreamingLeaseHandle.release` (which frees the lease)
rather than being silently normalized into an ordinary SSE `error`
event.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI, OpenAI

import research_agent.telemetry as telemetry
from research_agent import chat_summarization, qa
from research_agent.chat_streaming import (
    AnswerDelta,
    ChatAnswerStreamError,
    ChatStreamEvent,
    StreamedChatAnswer,
    build_completed_event,
    build_delta_event,
    build_done_event,
    build_error_event,
    build_phase_event,
    build_started_event,
    stream_chat_answer,
)
from research_agent.config import get_usage_policy
from research_agent.curation_chat import (
    _attach_exchange_metadata,
    _build_chat_session,
    _classify_offer_response,
    _maybe_set_web_offer,
    chat_turn,
)
from research_agent.curation_session import save_curation_session
from research_agent.query_expansion import PaperPoolSession

logger = logging.getLogger(__name__)


def _pending_offer_kind_and_description(session: PaperPoolSession) -> tuple[str | None, str | None]:
    """Duplicates only the two small, static description strings
    `curation_chat._chat_turn_impl` itself builds inline -- orchestration
    dispatch text, not model/retrieval/citation/filtering logic -- so
    this module's async dispatch reaches `_classify_offer_response`
    (reused completely unchanged) with an identical prompt to what the
    synchronous path already sends it."""
    if session.pending_web_offer is not None:
        return "web", f"search the web for more on: {session.pending_web_offer['question']!r}"
    if session.pending_report_update is not None:
        new_count = session.pending_report_update.get("new_article_count", 1)
        return "report", f"regenerate the literature review report to include {new_count} newly approved web source(s)"
    return None, None


def _apply_offer_dispatch(session: PaperPoolSession, question: str, offer_kind: str | None, offer_intent: str | None, result: dict) -> dict:
    """Mirrors `curation_chat._chat_turn_impl`'s own exact
    post-`ask_in_session` dispatch table for the branches this module
    streams a fresh answer for (decline/other/plain): a decline never
    re-offers a web search built from its own decline text; other/plain
    both do, via the SAME `curation_chat._maybe_set_web_offer` helper,
    reused unchanged."""
    if offer_kind == "web" and offer_intent == "decline":
        return {**result, "web_offer_declined": True}
    if offer_kind == "report" and offer_intent == "decline":
        return {**result, "report_update_declined": True}
    return _maybe_set_web_offer(session, question, result)


async def _drain_phase_queue_until(task: "asyncio.Task[Any]", phase_queue: "asyncio.Queue[str]") -> AsyncIterator[ChatStreamEvent]:
    """Bridges a background thread's synchronous `on_phase(...)` calls
    (see `_stream_fresh_answer`'s own `on_phase`, which pushes onto
    `phase_queue` via `loop.call_soon_threadsafe`) into real-time
    `phase` SSE events on the async side, while `task` (the thread
    doing the actual classify/condense/retrieve/filter work) is still
    running -- never buffers a phase until the thread finishes."""
    get_task: "asyncio.Task[str] | None" = None
    try:
        while not task.done():
            if get_task is None:
                get_task = asyncio.create_task(phase_queue.get())
            done, _pending = await asyncio.wait({task, get_task}, return_when=asyncio.FIRST_COMPLETED)
            if get_task in done:
                yield build_phase_event(get_task.result())
                get_task = None
    finally:
        if get_task is not None and not get_task.done():
            get_task.cancel()
    while not phase_queue.empty():
        yield build_phase_event(phase_queue.get_nowait())


async def _persist_and_complete(
    session: PaperPoolSession, session_id: str, checkpointer: Any, streamed: StreamedChatAnswer,
) -> AsyncIterator[ChatStreamEvent]:
    """The ONE place this module ever calls `save_curation_session` --
    see this module's own docstring for why the call is wrapped in
    `asyncio.shield`. `completed` is yielded if, and only if, this save
    genuinely succeeds; a save failure yields a safe `error` (a new,
    M4.2A-specific reason code -- M4.1 never covered persistence, so
    there was no existing safe code for this) then `done`, never
    `completed`."""
    yield build_phase_event("saving")
    try:
        await asyncio.shield(asyncio.to_thread(save_curation_session, session, session_id, checkpointer))
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("curation_chat_streaming: failed to persist session after a streamed turn")
        yield build_error_event("persistence_failed", "Failed to save the conversation.")
        yield build_done_event()
        return
    yield build_completed_event(streamed)
    yield build_done_event()


async def _stream_via_sync_fallback(
    session: PaperPoolSession, message: str, sync_client: OpenAI, session_id: str, checkpointer: Any, top_k: int,
    *, phases: tuple[str, ...],
) -> AsyncIterator[ChatStreamEvent]:
    """The web-search-offer-accept / report-update-offer-accept
    branches -- see this module's own docstring for why these reuse
    `curation_chat.chat_turn()` end to end rather than attempting
    incremental progress. `chat_turn()` already performs its own
    history mutation + `_attach_exchange_metadata` internally (unlike
    the fresh-answer path below), so there is nothing left to do here
    except report and persist its result."""
    for phase in phases:
        yield build_phase_event(phase)

    result = await asyncio.to_thread(chat_turn, session, message, sync_client, top_k)

    streamed = StreamedChatAnswer(
        answer=result["answer"], answerable=result["answerable"],
        cited_papers=result["cited_papers"], cited_web_articles=result["cited_web_articles"],
    )
    if streamed.answer:
        yield build_delta_event(streamed.answer)

    async for event in _persist_and_complete(session, session_id, checkpointer, streamed):
        yield event


async def _stream_fresh_answer(
    session: PaperPoolSession, message: str, offer_kind: str | None, offer_intent: str | None,
    sync_client: OpenAI, async_client: AsyncOpenAI, top_k: int, session_id: str, checkpointer: Any,
) -> AsyncIterator[ChatStreamEvent]:
    """The plain fallback / offer-decline / offer-other branches --
    true incremental token-level streaming via `qa.prepare_qa_turn` +
    M4.1's `chat_streaming.stream_chat_answer`. Mirrors `curation_chat.
    ask_in_session`'s own bookkeeping (building a `qa.ChatSession`,
    copying summary state back afterward) plus `_chat_turn_impl`'s own
    offer-clearing/dispatch discipline -- see `_apply_offer_dispatch`.
    """
    if offer_kind == "web":
        session.pending_web_offer = None
    elif offer_kind == "report":
        session.pending_report_update = None

    chat_session = _build_chat_session(session)

    loop = asyncio.get_running_loop()
    phase_queue: "asyncio.Queue[str]" = asyncio.Queue()

    def on_phase(phase: str) -> None:
        loop.call_soon_threadsafe(phase_queue.put_nowait, phase)

    def _prepare() -> qa.QATurnPreparation:
        # Usage Protection M3.2's own qa._prepare_bounded_recent_history
        # is reused completely unchanged (never reimplemented) -- this
        # calls chat_summarization.build_chat_context ONE extra,
        # deliberately redundant time first (cheap, deterministic, no
        # provider call -- the exact same function _prepare_bounded_
        # recent_history itself calls as ITS OWN first step) purely so
        # "summarizing_history" can be reported as its own real phase
        # BEFORE the (possibly real LLM-call) summarization pass that
        # may follow, rather than only after the fact.
        policy = get_usage_policy()
        context = chat_summarization.build_chat_context(
            chat_session.history, chat_session.chat_summary, chat_session.chat_summary_covers_history_count,
            policy, qa.ANSWER_SYSTEM_PROMPT, selected_papers=chat_session.papers, web_articles_added=chat_session.web_articles,
        )
        if context.should_summarize:
            on_phase("summarizing_history")
        recent_history = qa._prepare_bounded_recent_history(chat_session, sync_client)
        return qa.prepare_qa_turn(chat_session, message, sync_client, top_k, recent_history, on_phase=on_phase)

    prep_task: "asyncio.Task[qa.QATurnPreparation]" = asyncio.create_task(asyncio.to_thread(_prepare))
    async for event in _drain_phase_queue_until(prep_task, phase_queue):
        yield event
    preparation = await prep_task

    # Usage Protection M3.2 parity with ask_in_session: copy whatever
    # _prepare_bounded_recent_history updated in memory back onto the
    # real PaperPoolSession, explicitly -- not relying solely on
    # chat_session.history/session.chat_history aliasing (same
    # "explicit, not just implicit" posture ask_in_session's own
    # docstring documents).
    session.chat_history = chat_session.history
    session.chat_summary = chat_session.chat_summary
    session.chat_summary_covers_history_count = chat_session.chat_summary_covers_history_count
    session.chat_summary_updated_at = chat_session.chat_summary_updated_at

    if preparation.early_result is not None:
        result = _apply_offer_dispatch(session, message, offer_kind, offer_intent, preparation.early_result)
        _attach_exchange_metadata(session, result)
        streamed = StreamedChatAnswer(
            answer=result["answer"], answerable=result["answerable"],
            cited_papers=result.get("cited_papers") or [], cited_web_articles=result.get("cited_web_articles") or [],
        )
        async for event in _persist_and_complete(session, session_id, checkpointer, streamed):
            yield event
        return

    yield build_phase_event("generating")
    prepared = preparation.prepared
    assert prepared is not None  # QATurnPreparation guarantees exactly one of the two is set

    final_result_holder: dict[str, Any] = {}
    with telemetry.timed_child_call("generate_answer_streaming", "openai", model=qa.ANSWER_MODEL) as call:
        try:
            async for item in stream_chat_answer(async_client, prepared.messages, prepared.schema, prepared.papers_by_id, prepared.web_by_url):
                if isinstance(item, AnswerDelta):
                    if item.text:
                        yield build_delta_event(item.text)
                else:
                    final_result_holder["value"] = item.result
        except ChatAnswerStreamError as exc:
            if exc.reason_code == "cancelled":
                call.outcome = "cancelled"
                raise asyncio.CancelledError() from None
            call.outcome = "error"
            call.error_type = "ChatAnswerStreamError"
            yield build_error_event(exc.reason_code, exc.message)
            yield build_done_event()
            return

    streamed_raw = final_result_holder["value"]
    # chat-ux-fixes bug 5 parity: _generate_node applies this exact
    # renumbering to the terminal answer before persisting/returning it;
    # M4.1's adapter deliberately does NOT (see StreamedChatAnswer's own
    # field docstring) so this is the one place it happens on the
    # streaming path -- applied once, to the complete final answer,
    # never to any individual delta already sent.
    answer = qa._renumber_citation_markers(streamed_raw.answer)
    chat_session.history.append({"role": "user", "content": message})
    chat_session.history.append({"role": "assistant", "content": answer})
    session.chat_history = chat_session.history

    result = {
        "answer": answer, "answerable": streamed_raw.answerable,
        "cited_papers": streamed_raw.cited_papers, "cited_web_articles": streamed_raw.cited_web_articles,
        "web_relevance_verified": preparation.web_relevance_verified,
    }
    result = _apply_offer_dispatch(session, message, offer_kind, offer_intent, result)
    _attach_exchange_metadata(session, result)

    if result["answer"] != answer:
        # _maybe_set_web_offer appended a suffix (e.g. the web-search-
        # offer prompt) onto the just-generated answer -- reported as one
        # additional safe delta, still within the "zero or more deltas"
        # contract, so the final `completed` payload's own answer text
        # was fully covered by what was streamed.
        yield build_delta_event(result["answer"][len(answer):])

    final_streamed = StreamedChatAnswer(
        answer=result["answer"], answerable=result["answerable"],
        cited_papers=result.get("cited_papers") or [], cited_web_articles=result.get("cited_web_articles") or [],
    )
    async for event in _persist_and_complete(session, session_id, checkpointer, final_streamed):
        yield event


async def stream_curation_chat_turn(
    session: PaperPoolSession,
    message: str,
    *,
    session_id: str,
    checkpointer: Any,
    sync_client: OpenAI,
    async_client: AsyncOpenAI,
    top_k: int = qa.TOP_K_DEFAULT,
) -> AsyncIterator[ChatStreamEvent]:
    """One curation-chat turn, streamed. Assumes session existence,
    chat-turn capacity, and stage-readiness (`session.stage ==
    "synthesize"`) have ALL already been validated by the caller --
    mirroring `curation_chat_service.answer_curation_chat`'s own
    pre-guard checks -- and that admission + the session lease have
    already been acquired (see `usage_guard.open_paid_action_for_
    streaming`), BEFORE this generator is ever constructed. This
    function performs no request-shape validation of its own, only the
    actual guarded streaming work; the caller is responsible for
    releasing the guard handle in its own `finally`, regardless of how
    this generator ends.

    A handled, recognized failure (a `ChatAnswerStreamError` from the
    model adapter, a persistence failure, or any other unexpected
    exception from the underlying sync helpers this reuses) always
    yields a safe `error` event -- a stable, fixed reason code and
    message, never exception text -- followed by `done`, never
    `completed`. A genuine `asyncio.CancelledError` is always
    RE-RAISED after whatever cleanup already ran, never turned into an
    `error` event -- see this module's own docstring.
    """
    yield build_started_event()

    try:
        offer_kind, offer_description = _pending_offer_kind_and_description(session)
        offer_intent: str | None = None
        if offer_kind is not None and offer_description is not None:
            offer_intent = await asyncio.to_thread(_classify_offer_response, offer_description, message, sync_client)

        if offer_kind == "web" and offer_intent == "accept":
            branch = _stream_via_sync_fallback(
                session, message, sync_client, session_id, checkpointer, top_k, phases=("searching_web", "generating"),
            )
        elif offer_kind == "report" and offer_intent == "accept":
            branch = _stream_via_sync_fallback(
                session, message, sync_client, session_id, checkpointer, top_k, phases=("generating",),
            )
        else:
            branch = _stream_fresh_answer(
                session, message, offer_kind, offer_intent, sync_client, async_client, top_k, session_id, checkpointer,
            )

        async for event in branch:
            yield event
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("curation_chat_streaming: unhandled failure during a streamed turn")
        yield build_error_event("provider_error", "The model provider returned an error.")
        yield build_done_event()
