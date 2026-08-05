from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

import research_agent.api as api
from research_agent.api_app.errors import _upstream_error_guard
from research_agent.api_app.schemas import CurationGenerateReportRequest, CurationRegenerateReportRequest, ReportOut
from research_agent.services.curation_report_service import (
    activate_report_version,
    export_active_report,
    get_or_create_report,
    regenerate_report,
)
from research_agent.services.errors import ServiceError

router = APIRouter()


@router.post("/curation/{session_id}/report", response_model=ReportOut)
def curation_report(
    session_id: str, req: CurationGenerateReportRequest = CurationGenerateReportRequest(),
    cp=Depends(api.get_curation_checkpointer),
) -> ReportOut:
    with _upstream_error_guard("curation_report"):
        try:
            return get_or_create_report(
                session_id, cp, report_template=req.report_template, refinement_mode=req.refinement_mode,
            )
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post("/curation/{session_id}/report/regenerate", response_model=ReportOut)
def curation_report_regenerate(
    session_id: str, req: CurationRegenerateReportRequest = CurationRegenerateReportRequest(),
    cp=Depends(api.get_curation_checkpointer),
) -> ReportOut:
    with _upstream_error_guard("curation_report_regenerate"):
        try:
            return regenerate_report(
                session_id, cp, report_template=req.report_template, refinement_mode=req.refinement_mode,
            )
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.get("/curation/{session_id}/report/export")
def curation_report_export(
    session_id: str, format: str = "markdown", cp=Depends(api.get_curation_checkpointer),
) -> PlainTextResponse:
    """report-quality Phase R5A: exports the session's ACTIVE report
    version -- never always-the-latest, same "session.report always
    mirrors the active version" invariant every other report read
    already relies on -- as a downloadable Markdown document.

    `format` defaults to "markdown" -- the deliberate, lower-risk
    choice over requiring it explicitly: markdown is the ONLY supported
    value in this phase, so defaulting means a client hitting this
    endpoint with no query string at all still gets a sensible result,
    rather than an error for omitting a parameter with just one valid
    value anyway. Any OTHER explicit value (a future PDF/DOCX request
    before those formats exist, or a typo) still 400s -- see export_
    active_report's own docstring for why that check runs before any
    session lookup at all.

    No response_model/response_class here -- returning a real
    PlainTextResponse instance directly is what lets this set a custom
    media_type (text/markdown, not the default text/plain) and a
    Content-Disposition header for the download filename; FastAPI uses
    a returned Response object as-is.
    """
    with _upstream_error_guard("curation_report_export"):
        try:
            content, filename = export_active_report(session_id, cp, format)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        return PlainTextResponse(
            content, media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )


@router.post("/curation/{session_id}/reports/{version_id}/activate", response_model=ReportOut)
def curation_report_activate_version(
    session_id: str, version_id: str, cp=Depends(api.get_curation_checkpointer),
) -> ReportOut:
    """report-quality Phase R3: switches which report version is
    active/current for this session -- a pure pointer switch, never a
    regeneration, never a mutation of any version's own content. An
    unknown version_id (or one from a different session) 404s, same as
    an unknown session_id -- both mean "this URL doesn't point at
    anything real," not a client error worth a 400."""
    with _upstream_error_guard("curation_report_activate_version"):
        try:
            return activate_report_version(session_id, version_id, cp)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
