from __future__ import annotations

from typing import AsyncIterator

from fastapi.responses import StreamingResponse

import research_agent.api as api
import research_agent.telemetry as telemetry
from research_agent import report
from research_agent.api_app.schemas import CurationGenerateReportRequest, CurationRegenerateReportRequest, ReportOut
from research_agent.api_app.serializers import _report_to_out
from research_agent.curation_report_streaming import (
    HandledReportStreamFailure,
    stream_cached_report,
    stream_generate_report_turn,
    stream_regenerate_report_turn,
)
from research_agent.curation_session import load_curation_session, save_curation_session
from research_agent.report_streaming import build_done_event, build_error_event
from research_agent.services.errors import ServiceError
from research_agent.usage_guard import guard_paid_action, open_admission_and_lease_for_streaming


def get_or_create_report(
    session_id: str, cp, report_template: str | None = None, refinement_mode: str | None = None,
) -> ReportOut:
    """Generate-or-get, same cache-then-generate convention /summarize
    already uses for its own report-like artifact — a second call for the
    same session_id doesn't re-bill the LLM.

    report-quality Phase R2C: report_template only ever matters on the
    FRESH-generation branch below -- if a report already exists, it's
    returned as-is regardless of what template this call requested,
    matching this endpoint's own already-established cache-first
    semantics (unchanged: a second call must not re-generate). Switching
    an EXISTING report's template is what /report/regenerate is for.
    None/omitted resolves to "analytical", the confirmed default.

    report-quality Phase R3: a freshly generated report is appended as
    this session's first report version (api.append_report_version,
    generation_reason=api.GENERATION_REASON_INITIAL) rather than
    assigned to session.report directly -- append_report_version keeps
    session.report mirrored to the new version as a side effect, so
    every line below this one is otherwise unchanged from before R3.

    report-quality Phase R4.1: refinement_mode follows the exact same
    "only matters on the fresh-generation branch" rule report_template
    already established just above -- a cache hit returns the existing
    report as-is regardless of what refinement_mode this call requested.
    web_articles is always [] here: generate_report_for_session (unlike
    regenerate) never offers web sources to the model at all, so a
    refinement pass over the initial report has nothing web-side to
    re-evaluate against.
    """
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    if session.report is None:
        # Fresh-generation branch only -- a cache hit (session.report
        # already exists) never enters this block at all, so it creates no
        # paid_actions row, matching /summarize's own cache-then-generate
        # telemetry posture. Usage Protection M2.2A: for the same reason,
        # a cache hit never reaches the guard either -- it bypasses
        # admission and the lease entirely, not just telemetry.
        with guard_paid_action("report_generate", subject=("session", session_id), use_lease=True):
            try:
                report = api.generate_report_for_session(
                    session, client=api._state["client"], report_template=report_template or "analytical",
                )
            except ValueError as exc:
                raise ServiceError(400, str(exc)) from exc
            report = api.refine_report_if_requested(
                report, session.topic, session.selected_papers, [],
                report.get("report_template", "analytical"),
                refinement_mode=refinement_mode, client=api._state["client"],
            )
        api.append_report_version(session, report, api.GENERATION_REASON_INITIAL)
        # curation-refinement-and-auto-offer Phase 6f-3: keeps the
        # auto-offer's staleness check accurate even when a report is
        # generated straight through this endpoint rather than via
        # chat's accept-web-offer path.
        session.report_covered_web_article_count = len(session.web_articles_added)
        save_curation_session(session, session_id, cp)
    return _report_to_out(session.report, api.get_active_report_version(session))


def stream_generate_report(session_id: str, req: CurationGenerateReportRequest, cp) -> StreamingResponse:
    """Usage Protection M4.3A: the streaming counterpart to get_or_
    create_report above -- POST /curation/{session_id}/report/stream.
    Every pre-guard check below (session existence, and -- for the
    non-cached branch only -- the synthesize-stage precondition, reused
    unchanged via report.require_synthesize_stage_for_report) runs
    BEFORE the guard opens and BEFORE any StreamingResponse is
    constructed, exactly mirroring get_or_create_report's own ordering.

    Cache-hit detection happens here, synchronously, BEFORE admission/
    the lease are ever touched -- an existing session.report means this
    performs NO admission check, NO lease acquisition, and (inside the
    generator) NO telemetry.paid_action either -- see curation_report_
    streaming.stream_cached_report's own docstring.

    Uses usage_guard.open_admission_and_lease_for_streaming (NOT
    guard_paid_action) for the non-cached branch -- admission + the
    lease are acquired here, before the response is built, and the
    lease is released later, inside the streaming generator's own
    finally, once the streamed work (provider calls AND final session
    persistence) has fully finished. telemetry.paid_action is
    deliberately opened ENTIRELY inside event_stream() below, via a
    plain with statement -- never split across this function and the
    generator -- see usage_guard.open_admission_and_lease_for_
    streaming's own docstring for the real, empirically-confirmed
    contextvars hazard that motivates this (identical reasoning to
    M4.2A's own chat-streaming design; M4.2 itself is unmodified here).

    Lifecycle-hardening pattern (also reused verbatim from M4.2A):
    `HandledReportStreamFailure` is caught HERE, OUTSIDE the `with
    telemetry.paid_action(...):` block -- letting it propagate out of
    that block first is what makes telemetry.paid_action's own `except
    Exception` branch record the top-level paid_actions row as
    `outcome="error"`. `asyncio.CancelledError` is not caught here at
    all: it propagates all the way out of event_stream(), so telemetry.
    paid_action records `outcome="cancelled"` and no `completed`/`error`
    event is ever emitted for it.
    """
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")

    is_cache_hit = session.report is not None
    lease_handle = None
    if not is_cache_hit:
        # Fresh-generation branch only -- a cache hit never reaches this
        # check (or the guard below) at all, mirroring get_or_create_
        # report's own cache-first semantics exactly.
        try:
            report.require_synthesize_stage_for_report(session)
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        lease_handle = open_admission_and_lease_for_streaming(
            "report_generate", subject=("session", session_id), use_lease=True,
        )

    async def event_stream() -> AsyncIterator[str]:
        try:
            if is_cache_hit:
                async for event in stream_cached_report(session):
                    yield event.to_sse()
                return
            try:
                with telemetry.paid_action("report_generate", subject_type="session", subject_id=session_id):
                    async for event in stream_generate_report_turn(
                        session, session_id, cp, api._state["client"], req.report_template, req.refinement_mode,
                    ):
                        yield event.to_sse()
            except HandledReportStreamFailure as exc:
                yield build_error_event(exc.reason_code, exc.message).to_sse()
                yield build_done_event().to_sse()
        finally:
            if lease_handle is not None:
                lease_handle.release()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


def regenerate_report(
    session_id: str, cp, report_template: str | None = None, refinement_mode: str | None = None,
) -> ReportOut:
    """report-quality Phase R2C: report_template=None (the default)
    preserves the existing report's own current template; an explicit
    value switches it -- see report.py's regenerate_report_with_new_
    sources for the exact resolution rule this simply forwards to.

    report-quality Phase R3: builds from session.report, which is
    always the currently ACTIVE version's own report body (append_
    report_version/activate_report_version keep the two in lockstep) --
    so this already regenerates from the active version, not always the
    latest one, with no extra plumbing needed here. The result is
    appended as a NEW version (generation_reason=api.GENERATION_REASON_
    REGENERATE) rather than overwriting the source version -- the prior
    active version stays exactly as it was, in report_versions, purely
    historical from this point on.

    report-quality Phase R4.1: refinement_mode="single" runs the
    draft -> evaluate -> revise -> finalize pass over the SAME web
    candidate set the draft was actually generated against -- session.
    web_articles_added minus anything currently revoked, mirroring the
    exact filter _regenerate_report_sections_with_sources itself
    already applies before the model ever sees a candidate. Passing
    the unfiltered pool here would let a revision re-cite a revoked
    source, reopening the bug R3.1's own revocation guarantee closed.
    This endpoint is whole-pool regeneration only -- chat-triggered
    regeneration (add-to-report, auto-update) never reaches this
    function, so refinement never applies there in R4.1.
    """
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    # Usage Protection M2.2A: whole-pool regeneration always does real
    # paid work (no cache branch), so this is unconditionally guarded.
    with guard_paid_action("report_regenerate", subject=("session", session_id), use_lease=True):
        try:
            report = api.regenerate_report_with_new_sources(
                session, client=api._state["client"], report_template=report_template,
            )
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        web_articles = [a for a in session.web_articles_added if a.url not in session.revoked_web_article_urls]
        report = api.refine_report_if_requested(
            report, session.topic, session.selected_papers, web_articles,
            report.get("report_template", "analytical"),
            refinement_mode=refinement_mode, client=api._state["client"],
        )
    api.append_report_version(session, report, api.GENERATION_REASON_REGENERATE)
    session.report_covered_web_article_count = len(session.web_articles_added)
    save_curation_session(session, session_id, cp)
    return _report_to_out(session.report, api.get_active_report_version(session))


def stream_regenerate_report(session_id: str, req: CurationRegenerateReportRequest, cp) -> StreamingResponse:
    """Usage Protection M4.3A: the streaming counterpart to regenerate_
    report above -- POST /curation/{session_id}/report/regenerate/
    stream. Always real paid work, no cache branch (mirrors regenerate_
    report's own unconditional guard) -- both preconditions (an existing
    report to regenerate from, and synthesize-stage readiness, both
    reused unchanged from report.py) are checked BEFORE admission/the
    lease/any StreamingResponse, same ordering discipline as stream_
    generate_report above. See that function's own docstring for the
    full admission/lease/telemetry/cancellation reasoning, identical
    here.
    """
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    try:
        report.require_existing_report_for_regeneration(session)
        report.require_synthesize_stage_for_report(session)
    except ValueError as exc:
        raise ServiceError(400, str(exc)) from exc

    lease_handle = open_admission_and_lease_for_streaming(
        "report_regenerate", subject=("session", session_id), use_lease=True,
    )

    async def event_stream() -> AsyncIterator[str]:
        try:
            try:
                with telemetry.paid_action("report_regenerate", subject_type="session", subject_id=session_id):
                    async for event in stream_regenerate_report_turn(
                        session, session_id, cp, api._state["client"], req.report_template, req.refinement_mode,
                    ):
                        yield event.to_sse()
            except HandledReportStreamFailure as exc:
                yield build_error_event(exc.reason_code, exc.message).to_sse()
                yield build_done_event().to_sse()
        finally:
            lease_handle.release()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


_EXPORT_FORMATS = ("markdown", "docx", "pdf")


def export_active_report(session_id: str, cp, format: str) -> tuple[str | bytes, str]:
    """report-quality Phase R5A/R5C.1/R5C.2: exports session.report --
    always the ACTIVE version's own body (append_report_version/
    activate_report_version keep session.report/get_active_report_
    version in lockstep, the same invariant every other report read in
    this module already relies on) -- as a rendered document. Returns
    (content, filename); content is `str` for markdown, `bytes` for
    docx/pdf -- the router picks the right Response type/media_type
    based on the same `format` it already received, so this function
    stays HTTP-unaware, same "session load + errors only" scope as
    get_or_create_report/regenerate_report above.

    format validation happens FIRST, before any session lookup -- an
    unsupported format is a pure request-shape problem, independent of
    whether session_id resolves to anything, so it's cheaper and more
    correct to reject it before touching the checkpointer at all.
    "markdown"/"docx"/"pdf" are the only supported values as of R5C.2.
    """
    if format not in _EXPORT_FORMATS:
        raise ServiceError(400, f"unsupported export format: {format!r}")
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    if session.report is None:
        raise ServiceError(404, "session has no report yet")
    version = api.get_active_report_version(session)
    if format == "docx":
        content: str | bytes = api.render_report_docx(session, version)
        filename = api.report_export_filename(session, version, "docx")
    elif format == "pdf":
        content = api.render_report_pdf(session, version)
        filename = api.report_export_filename(session, version, "pdf")
    else:
        content = api.render_report_markdown(session, version)
        filename = api.report_export_filename(session, version, "md")
    return content, filename


def activate_report_version(session_id: str, version_id: str, cp) -> ReportOut:
    """report-quality Phase R3: switches which report version is active
    for this session (report.py's own activate_report_version, which
    mirrors session.report to the requested version's body as a side
    effect) -- never regenerates, never mutates any version's content,
    purely a pointer switch. version_id not matching anything on this
    session is a 404 (an unknown/typo'd id, or one from a different
    session entirely), not a 400 -- the session_id itself IS valid, only
    the requested version isn't."""
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    version = api.activate_report_version(session, version_id)
    if version is None:
        raise ServiceError(404, f"version_id {version_id!r} not found for this session")
    save_curation_session(session, session_id, cp)
    return _report_to_out(session.report, version)
