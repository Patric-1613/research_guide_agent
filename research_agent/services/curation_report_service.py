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
    """
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    if session.report is None:
        try:
            session.report = api.generate_report_for_session(
                session, client=api._state["client"], report_template=report_template or "analytical",
            )
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
        # curation-refinement-and-auto-offer Phase 6f-3: keeps the
        # auto-offer's staleness check accurate even when a report is
        # generated straight through this endpoint rather than via
        # chat's accept-web-offer path.
        session.report_covered_web_article_count = len(session.web_articles_added)
        save_curation_session(session, session_id, cp)
    return _report_to_out(session.report)


def regenerate_report(session_id: str, cp, report_template: str | None = None) -> ReportOut:
    """report-quality Phase R2C: report_template=None (the default)
    preserves the existing report's own current template; an explicit
    value switches it -- see report.py's regenerate_report_with_new_
    sources for the exact resolution rule this simply forwards to."""
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    try:
        session.report = api.regenerate_report_with_new_sources(
            session, client=api._state["client"], report_template=report_template,
        )
    except ValueError as exc:
        raise ServiceError(400, str(exc)) from exc
    session.report_covered_web_article_count = len(session.web_articles_added)
    save_curation_session(session, session_id, cp)
    return _report_to_out(session.report)
