from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.api_app.errors import _upstream_error_guard
from research_agent.api_app.schemas import CurationGenerateReportRequest, CurationRegenerateReportRequest, ReportOut
from research_agent.services.curation_report_service import get_or_create_report, regenerate_report
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
