from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.api_app.errors import _upstream_error_guard
from research_agent.api_app.schemas import CurationPicksRequest, CurationStartRequest, CurationTurnResponse
from research_agent.services.curation_core_service import start_curation, submit_picks
from research_agent.services.errors import ServiceError

router = APIRouter()


@router.post("/curation/start", response_model=CurationTurnResponse)
def curation_start(req: CurationStartRequest, cp=Depends(api.get_curation_checkpointer)) -> CurationTurnResponse:
    with _upstream_error_guard("curation_start"):
        try:
            return start_curation(req, cp)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post("/curation/{session_id}/picks", response_model=CurationTurnResponse)
def curation_picks(session_id: str, req: CurationPicksRequest, cp=Depends(api.get_curation_checkpointer)) -> CurationTurnResponse:
    with _upstream_error_guard("curation_picks"):
        try:
            return submit_picks(session_id, req, cp)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
