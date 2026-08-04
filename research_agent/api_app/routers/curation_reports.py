from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.api_app.errors import _upstream_error_guard
from research_agent.api_app.schemas import CurationGenerateReportRequest, CurationRegenerateReportRequest, ReportOut
from research_agent.services.curation_report_service import activate_report_version, get_or_create_report, regenerate_report
from research_agent.services.errors import ServiceError

router = APIRouter()


@router.post("/curation/{session_id}/report", response_model=ReportOut)
def curation_report(
    session_id: str, req: CurationGenerateReportRequest = CurationGenerateReportRequest(),
    cp=Depends(api.get_curation_checkpointer),
) -> ReportOut:
    with _upstream_error_guard("curation_report"):
        try:
            return get_or_create_report(session_id, cp, report_template=req.report_template)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post("/curation/{session_id}/report/regenerate", response_model=ReportOut)
def curation_report_regenerate(
    session_id: str, req: CurationRegenerateReportRequest = CurationRegenerateReportRequest(),
    cp=Depends(api.get_curation_checkpointer),
) -> ReportOut:
    with _upstream_error_guard("curation_report_regenerate"):
        try:
            return regenerate_report(session_id, cp, report_template=req.report_template)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


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
