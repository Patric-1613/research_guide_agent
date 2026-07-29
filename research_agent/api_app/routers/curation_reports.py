from fastapi import APIRouter, Depends

import research_agent.api as api
from research_agent.services.curation_report_service import get_or_create_report, regenerate_report

router = APIRouter()


@router.post("/curation/{session_id}/report", response_model=api.ReportOut)
def curation_report(session_id: str, cp=Depends(api.get_curation_checkpointer)) -> api.ReportOut:
    with api._upstream_error_guard("curation_report"):
        return get_or_create_report(session_id, cp)


@router.post("/curation/{session_id}/report/regenerate", response_model=api.ReportOut)
def curation_report_regenerate(session_id: str, cp=Depends(api.get_curation_checkpointer)) -> api.ReportOut:
    with api._upstream_error_guard("curation_report_regenerate"):
        return regenerate_report(session_id, cp)
