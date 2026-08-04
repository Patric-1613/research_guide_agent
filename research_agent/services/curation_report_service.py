from __future__ import annotations

import research_agent.api as api
from research_agent.api_app.schemas import ReportOut
from research_agent.api_app.serializers import _report_to_out
from research_agent.curation_session import load_curation_session, save_curation_session
from research_agent.services.errors import ServiceError


def get_or_create_report(session_id: str, cp, report_template: str | None = None) -> ReportOut:
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
    """
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    if session.report is None:
        try:
            report = api.generate_report_for_session(
                session, client=api._state["client"], report_template=report_template or "analytical",
            )
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        api.append_report_version(session, report, api.GENERATION_REASON_INITIAL)
        # curation-refinement-and-auto-offer Phase 6f-3: keeps the
        # auto-offer's staleness check accurate even when a report is
        # generated straight through this endpoint rather than via
        # chat's accept-web-offer path.
        session.report_covered_web_article_count = len(session.web_articles_added)
        save_curation_session(session, session_id, cp)
    return _report_to_out(session.report, api.get_active_report_version(session))


def regenerate_report(session_id: str, cp, report_template: str | None = None) -> ReportOut:
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
    """
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    try:
        report = api.regenerate_report_with_new_sources(
            session, client=api._state["client"], report_template=report_template,
        )
    except ValueError as exc:
        raise ServiceError(400, str(exc)) from exc
    api.append_report_version(session, report, api.GENERATION_REASON_REGENERATE)
    session.report_covered_web_article_count = len(session.web_articles_added)
    save_curation_session(session, session_id, cp)
    return _report_to_out(session.report, api.get_active_report_version(session))


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
