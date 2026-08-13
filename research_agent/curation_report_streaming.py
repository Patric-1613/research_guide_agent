"""Usage Protection M4.3A: the production report-generation/regeneration
streaming domain service -- one async generator per turn, yielding the
shared report-streaming SSE vocabulary (research_agent/report_streaming.py)
in the fixed order `started -> phase* -> completed -> done` on success,
matching the required sequences exactly:

  refinement off:         started -> generating -> saving -> completed -> done
  evaluation, no revision: started -> generating -> evaluating -> saving -> completed -> done
  evaluation + revision:  started -> generating -> evaluating -> revising -> saving -> completed -> done
  initial-gen cache hit:  started -> completed -> done   (no phase events at all)

**Lifecycle-hardening pattern, reused verbatim from M4.2A.** A handled,
recognized failure is raised here as this module's own typed
`HandledReportStreamFailure`, NOT self-converted into `error`/`done`
events -- that conversion happens one layer up, in
`services/curation_report_service.py`'s `stream_generate_report`/
`stream_regenerate_report`, OUTSIDE their own `with telemetry.paid_
action(...):` block, for the exact same reason M4.2A's own curation_
chat_streaming.py docstring documents: letting the exception propagate
out of that block first is what makes `telemetry.paid_action` record the
top-level `paid_actions` row as `outcome="error"` instead of `"success"`.
`asyncio.CancelledError` is unaffected -- always re-raised, never
converted to `HandledReportStreamFailure` or an `error` event, so
`telemetry.paid_action` records `outcome="cancelled"`.

**No admission/lease/HTTP awareness here.** This module assumes the
caller has already: resolved `session_id` to a real session (404
otherwise), checked the report precondition(s) (`report.require_
synthesize_stage_for_report` / `report.require_existing_report_for_
regeneration`, both reused unchanged from report.py -- see that
module's own docstrings for why they were extracted, verbatim, rather
than duplicated), and -- for the non-cached path -- acquired admission
and the session lease (`usage_guard.open_admission_and_lease_for_
streaming`). `telemetry.paid_action` is likewise the caller's own
responsibility to open, entirely inside its own generator body, never
split across a route/service function and this module -- the exact same
`contextvars` hazard `usage_guard.open_admission_and_lease_for_
streaming`'s own docstring documents for chat streaming applies
identically here.

**No second report-generation pipeline.** Every actual provider call in
this module goes through the SAME, completely unmodified report.py
functions the synchronous endpoints already use: `generate_report_for_
session`, `regenerate_report_with_new_sources`, `refine_report_if_
requested` (now with an optional `progress_callback`, still a pure
observability hook -- see that function's own docstring), `evaluate_
report`, `revise_report`, `append_report_version`. Nothing here
re-implements report generation, evaluation, revision, or versioning
logic.

**Synchronous work, threaded.** `generate_report_for_session`/
`regenerate_report_with_new_sources`/`refine_report_if_requested` are
all plain synchronous functions (real OpenAI SDK calls inside), so each
is dispatched via `asyncio.to_thread` and awaited through `asyncio.
shield` -- the same idiom M4.2A's own curation_chat_streaming.py
established for `save_curation_session`. `refine_report_if_requested`'s
own `progress_callback` runs INSIDE that worker thread; it is bridged
into real-time `phase` SSE events via a `loop.call_soon_threadsafe`-fed
`asyncio.Queue`, drained by `_drain_phase_queue_until` (a small, local,
protocol-agnostic helper deliberately NOT imported from research_agent/
curation_chat_streaming.py's own near-identical helper -- M4.2 is closed
and this phase intentionally does not import from, or modify, any M4.2
file; the ~15 lines of duplication is the accepted, disclosed cost of
that boundary).

**Cancellation.** Per this phase's own explicit lifecycle rule (broader
than M4.2A's own persistence-only version): a genuine `asyncio.
CancelledError` observed while ANY of the four phases' own worker thread
is active -- generating, evaluating/revising (both inside the one
refine-phase thread call), or saving -- is handled by `asyncio.wait`ing
for that SAME retained task to fully settle BEFORE the lease is released
and the original `CancelledError` is re-raised. This is deliberately
NOT limited to the persistence step the way M4.2A's own design was:
even though `generate_report_for_session`/`refine_report_if_requested`
never mutate `session` (both are pure functions returning a fresh dict --
confirmed directly by reading them), releasing the lease while an
orphaned OpenAI call is still running in the background would let a NEW
report action start concurrently against the SAME session/lease group,
which is exactly the concurrency the lease exists to prevent -- a
correctness concern independent of whether the orphaned thread happens
to touch shared session state. No later phase (evaluating after a
generating-phase cancellation; revising after an evaluating-phase
cancellation; saving after any earlier cancellation) is ever begun --
structurally guaranteed by control flow, not a flag: cancellation
propagates out of this generator (via a bare `raise` after `asyncio.
wait`) before the code that would start the next phase is ever reached.
Python cannot forcibly terminate a running OS thread; waiting for
settlement here may therefore take up to the provider's own configured
timeout (`config.get_usage_policy().provider_timeout_seconds`) to
actually resolve -- this is a real, disclosed limit of threaded
synchronous work, not a bug.

**Persistence commit point and rollback safety.** `_persist_and_
complete` is the ONE place `append_report_version`/`save_curation_
session` are ever called. Per this phase's own explicit mutation-safety
requirement: the commit-point mutations (`append_report_version`, the
`report_covered_web_article_count` update) run against a `copy.
deepcopy` of `session`, never the original object the caller loaded --
`completed` is built from, and only from, that copy, and ONLY after
`save_curation_session` genuinely succeeds. `research_agent/curation_
session.py::load_curation_session` already reconstructs a brand-new
`PaperPoolSession` on every call (never a shared/cached object), so
nothing else in this process could observe the original `session`
object's mutation either way -- this deep-copy is a deliberate,
low-cost, explicitly-requested defense-in-depth measure, not a fix for
an active cross-request bug. A failed save leaves the ORIGINAL `session`
object completely untouched (never mutated at all) and the throwaway
copy's own partial mutation simply becomes unreachable garbage -- it is
never persisted, never exposed in any event payload, and never read
again.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any, AsyncIterator, Callable

from openai import OpenAI

from research_agent import report
from research_agent.api_app.serializers import _report_to_out
from research_agent.curation_session import save_curation_session
from research_agent.query_expansion import PaperPoolSession
from research_agent.report_streaming import (
    ReportStreamEvent,
    build_completed_event,
    build_done_event,
    build_phase_event,
    build_started_event,
)

logger = logging.getLogger(__name__)


class HandledReportStreamFailure(Exception):
    """The ONE internal signal this module raises for a recognized,
    safely-reportable failure (a provider error from generate/evaluate/
    revise, or a persistence failure) -- deliberately an ordinary
    `Exception` (never `asyncio.CancelledError`), mirroring `research_
    agent.curation_chat_streaming.HandledStreamFailure`'s own contract
    exactly: makes `telemetry.paid_action`'s own `except Exception`
    branch record the wrapping action's outcome as `"error"`, and is
    exactly what the caller (one layer up, OUTSIDE that `with telemetry.
    paid_action(...):` block) catches to emit the safe `error`+`done`
    SSE frames. `reason_code`/`message` are always small, fixed, safe
    values -- never raw exception text."""

    def __init__(self, reason_code: str, message: str):
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.message = message


async def _drain_phase_queue_until(
    task: "asyncio.Task[Any]", phase_queue: "asyncio.Queue[str]",
) -> AsyncIterator[ReportStreamEvent]:
    """Bridges a background thread's synchronous `progress_callback(...)`
    calls (see `_run_generate_and_refine`'s own `on_phase`, which pushes
    onto `phase_queue` via `loop.call_soon_threadsafe`) into real-time
    `phase` SSE events on the async side, while `task` (the thread doing
    the actual generate/evaluate/revise work) is still running -- never
    buffers a phase until the thread finishes. Deliberately local to this
    module -- see this module's own top-level docstring for why it is
    not imported from research_agent/curation_chat_streaming.py's
    near-identical helper."""
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


async def stream_cached_report(session: PaperPoolSession) -> AsyncIterator[ReportStreamEvent]:
    """The initial-generation cache-hit path -- mirrors `services.
    curation_report_service.get_or_create_report`'s own cache branch
    exactly: `session.report` already exists, so this performs NO
    admission check, NO lease acquisition, NO `telemetry.paid_action`,
    NO provider call, NO mutation, and NO save. Just re-serializes the
    already-persisted report through the same `_report_to_out` every
    other report read uses. The caller (`services.curation_report_
    service.stream_generate_report`) is responsible for detecting the
    cache hit and routing here BEFORE opening admission/the lease at
    all -- this function has no way to enforce that itself, since by
    the time it runs the decision has already been made."""
    yield build_started_event()
    version = report.get_active_report_version(session)
    report_out = _report_to_out(session.report, version)
    yield build_completed_event(report_out.model_dump(mode="json"))
    yield build_done_event()


async def _run_generate_and_refine(
    session: PaperPoolSession, client: OpenAI, generate_fn: Callable[[], dict],
    report_template: str | None, refinement_mode: str | None, web_articles_for_refinement: list,
    result_holder: dict[str, Any],
) -> AsyncIterator[ReportStreamEvent]:
    """Shared body for the fresh-generation and regeneration streaming
    turns below -- identical generate-then-refine orchestration either
    way; the only difference between the two public callers is which
    report.py generate function `generate_fn` closes over and which
    `web_articles_for_refinement` list gets passed to `refine_report_
    if_requested`. Neither public function duplicates this body, so the
    threading/cancellation/phase-bridging logic can never drift between
    the two paths.

    Yields every SSE event to forward; on success, stores the fully
    generated-and-refined report dict at `result_holder["value"]` --
    same "mutable holder the caller already owns" convention M4.2A's
    own `_stream_fresh_answer` uses for exactly the same reason (an
    async generator's own `return` cannot carry a value a `yield`-driven
    caller can retrieve).
    """
    yield build_phase_event("generating")
    generate_task: "asyncio.Task[dict]" = asyncio.create_task(asyncio.to_thread(generate_fn))
    try:
        draft = await asyncio.shield(generate_task)
    except asyncio.CancelledError:
        await asyncio.wait([generate_task])
        raise
    except Exception:
        logger.exception("curation_report_streaming: failed to generate the report draft")
        raise HandledReportStreamFailure("provider_error", "The model provider returned an error.") from None

    loop = asyncio.get_running_loop()
    phase_queue: "asyncio.Queue[str]" = asyncio.Queue()

    def on_phase(phase: str) -> None:
        loop.call_soon_threadsafe(phase_queue.put_nowait, phase)

    refine_task: "asyncio.Task[dict]" = asyncio.create_task(
        asyncio.to_thread(
            report.refine_report_if_requested,
            draft, session.topic, session.selected_papers, web_articles_for_refinement,
            draft.get("report_template", report_template or "analytical"),
            refinement_mode, client, report.REPORT_MODEL, on_phase,
        )
    )
    try:
        async for event in _drain_phase_queue_until(refine_task, phase_queue):
            yield event
        result_holder["value"] = await asyncio.shield(refine_task)
    except asyncio.CancelledError:
        await asyncio.wait([refine_task])
        raise
    except Exception:
        logger.exception("curation_report_streaming: failed to evaluate/revise the report draft")
        raise HandledReportStreamFailure("provider_error", "The model provider returned an error.") from None


async def _persist_and_complete(
    session: PaperPoolSession, session_id: str, checkpointer: Any, generated_report: dict, generation_reason: str,
) -> AsyncIterator[ReportStreamEvent]:
    """The ONE place this module ever calls `append_report_version`/
    `save_curation_session`. `completed` is yielded if, and only if,
    this save genuinely succeeds; a save failure raises
    `HandledReportStreamFailure` instead of self-emitting `error`/
    `done` -- see this module's own docstring for why.

    Mutation/rollback safety: operates on a `copy.deepcopy` of `session`,
    never the original -- see this module's own top-level docstring for
    the full reasoning. The save runs as an explicitly retained
    `asyncio.Task`, awaited via `asyncio.shield` so an outer cancellation
    does not tear the task itself down (Python cannot forcibly stop a
    running thread regardless); on cancellation, this module explicitly
    `asyncio.wait`s for that SAME retained task before re-raising, so the
    save is always allowed to fully settle before this generator's own
    cleanup (and, one layer up, the caller's lease release) proceeds.
    `completed` is never emitted before the save has genuinely
    succeeded, and never at all once a cancellation has been observed
    here.
    """
    yield build_phase_event("saving")

    session_copy = copy.deepcopy(session)

    def _commit() -> PaperPoolSession:
        report.append_report_version(session_copy, generated_report, generation_reason)
        session_copy.report_covered_web_article_count = len(session_copy.web_articles_added)
        save_curation_session(session_copy, session_id, checkpointer)
        return session_copy

    save_task: "asyncio.Task[PaperPoolSession]" = asyncio.create_task(asyncio.to_thread(_commit))
    try:
        committed_session = await asyncio.shield(save_task)
    except asyncio.CancelledError:
        await asyncio.wait([save_task])
        raise
    except Exception:
        logger.exception("curation_report_streaming: failed to persist the report after a streamed turn")
        raise HandledReportStreamFailure("persistence_failed", "Failed to save the report.") from None

    version = report.get_active_report_version(committed_session)
    report_out = _report_to_out(committed_session.report, version)
    yield build_completed_event(report_out.model_dump(mode="json"))
    yield build_done_event()


async def stream_generate_report_turn(
    session: PaperPoolSession, session_id: str, checkpointer: Any, client: OpenAI,
    report_template: str | None, refinement_mode: str | None,
) -> AsyncIterator[ReportStreamEvent]:
    """One streamed FRESH-generation turn (the non-cached branch of
    `POST /curation/{session_id}/report/stream`). Assumes the cache-hit
    check, `report.require_synthesize_stage_for_report`, and admission +
    the session lease have ALL already resolved successfully by the
    caller, exactly mirroring `services.curation_report_service.get_or_
    create_report`'s own pre-guard structure. Mirrors that function's
    exact resolution rule for `report_template` (`None` means
    "analytical", the confirmed default) -- resolved at the same call
    site sync code resolves it, not earlier.
    """
    yield build_started_event()
    try:
        result_holder: dict[str, Any] = {}
        async for event in _run_generate_and_refine(
            session, client,
            generate_fn=lambda: report.generate_report_for_session(
                session, client=client, report_template=report_template or "analytical",
            ),
            report_template=report_template, refinement_mode=refinement_mode, web_articles_for_refinement=[],
            result_holder=result_holder,
        ):
            yield event
        report_out = result_holder["value"]

        async for event in _persist_and_complete(session, session_id, checkpointer, report_out, report.GENERATION_REASON_INITIAL):
            yield event
    except asyncio.CancelledError:
        raise
    except HandledReportStreamFailure:
        raise
    except Exception:
        logger.exception("curation_report_streaming: unhandled failure during a streamed generation turn")
        raise HandledReportStreamFailure("provider_error", "The model provider returned an error.") from None


async def stream_regenerate_report_turn(
    session: PaperPoolSession, session_id: str, checkpointer: Any, client: OpenAI,
    report_template: str | None, refinement_mode: str | None,
) -> AsyncIterator[ReportStreamEvent]:
    """One streamed regeneration turn (`POST /curation/{session_id}/
    report/regenerate/stream`) -- always real paid work, no cache
    branch (mirrors `services.curation_report_service.regenerate_
    report`'s own unconditional guard). Assumes `report.require_
    existing_report_for_regeneration`/`report.require_synthesize_stage_
    for_report` and admission + the session lease have ALL already
    resolved successfully by the caller.

    `web_articles_for_refinement` mirrors `regenerate_report`'s own
    exact filter -- session.web_articles_added minus anything currently
    revoked -- so a refinement pass evaluates against the SAME candidate
    set the draft was actually generated against, never a revoked
    source.
    """
    yield build_started_event()
    try:
        web_articles = [a for a in session.web_articles_added if a.url not in session.revoked_web_article_urls]
        result_holder: dict[str, Any] = {}
        async for event in _run_generate_and_refine(
            session, client,
            generate_fn=lambda: report.regenerate_report_with_new_sources(
                session, client=client, report_template=report_template,
            ),
            report_template=report_template, refinement_mode=refinement_mode, web_articles_for_refinement=web_articles,
            result_holder=result_holder,
        ):
            yield event
        report_out = result_holder["value"]

        async for event in _persist_and_complete(session, session_id, checkpointer, report_out, report.GENERATION_REASON_REGENERATE):
            yield event
    except asyncio.CancelledError:
        raise
    except HandledReportStreamFailure:
        raise
    except Exception:
        logger.exception("curation_report_streaming: unhandled failure during a streamed regeneration turn")
        raise HandledReportStreamFailure("provider_error", "The model provider returned an error.") from None
