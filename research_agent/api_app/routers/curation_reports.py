from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.curation_session import load_curation_session, save_curation_session

router = APIRouter()


@router.post("/curation/{session_id}/report", response_model=api.ReportOut)
def curation_report(session_id: str, cp=Depends(api.get_curation_checkpointer)) -> api.ReportOut:
    """Generate-or-get, same cache-then-generate convention /summarize
    already uses for its own report-like artifact — a second call for the
    same session_id doesn't re-bill the LLM."""
    with api._upstream_error_guard("curation_report"):
        session = load_curation_session(session_id, cp)
        if session is None:
            raise HTTPException(status_code=404, detail="session_id not found")
        if session.report is None:
            try:
                session.report = api.generate_report_for_session(session, client=api._state["client"])
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            # curation-refinement-and-auto-offer Phase 6f-3: keeps the
            # auto-offer's staleness check accurate even when a report is
            # generated straight through this endpoint rather than via
            # chat's accept-web-offer path.
            session.report_covered_web_article_count = len(session.web_articles_added)
            save_curation_session(session, session_id, cp)
        return api._report_to_out(session.report)


@router.post("/curation/{session_id}/report/regenerate", response_model=api.ReportOut)
def curation_report_regenerate(session_id: str, cp=Depends(api.get_curation_checkpointer)) -> api.ReportOut:
    with api._upstream_error_guard("curation_report_regenerate"):
        session = load_curation_session(session_id, cp)
        if session is None:
            raise HTTPException(status_code=404, detail="session_id not found")
        try:
            session.report = api.regenerate_report_with_new_sources(session, client=api._state["client"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session.report_covered_web_article_count = len(session.web_articles_added)
        save_curation_session(session, session_id, cp)
        return api._report_to_out(session.report)
