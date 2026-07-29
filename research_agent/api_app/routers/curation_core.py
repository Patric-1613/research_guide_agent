from fastapi import APIRouter, Depends

import research_agent.api as api
from research_agent.services.curation_core_service import start_curation, submit_picks

router = APIRouter()


@router.post("/curation/start", response_model=api.CurationTurnResponse)
def curation_start(req: api.CurationStartRequest, cp=Depends(api.get_curation_checkpointer)) -> api.CurationTurnResponse:
    with api._upstream_error_guard("curation_start"):
        return start_curation(req, cp)


@router.post("/curation/{session_id}/picks", response_model=api.CurationTurnResponse)
def curation_picks(session_id: str, req: api.CurationPicksRequest, cp=Depends(api.get_curation_checkpointer)) -> api.CurationTurnResponse:
    with api._upstream_error_guard("curation_picks"):
        return submit_picks(session_id, req, cp)
