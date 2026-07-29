from fastapi import APIRouter, Depends

import research_agent.api as api
from research_agent.api_app.schemas import (
    CurationSelectFromHistoryRequest,
    CurationSelectFromHistoryResponse,
    CurationTurnResponse,
)
from research_agent.services.curation_history_service import reopen_curation, select_from_history

router = APIRouter()


@router.post("/curation/{session_id}/select-from-history", response_model=CurationSelectFromHistoryResponse)
def curation_select_from_history(
    session_id: str, req: CurationSelectFromHistoryRequest, cp=Depends(api.get_curation_checkpointer),
) -> CurationSelectFromHistoryResponse:
    return select_from_history(session_id, req, cp)


@router.post("/curation/{session_id}/reopen", response_model=CurationTurnResponse)
def curation_reopen(session_id: str, cp=Depends(api.get_curation_checkpointer)) -> CurationTurnResponse:
    with api._upstream_error_guard("curation_reopen"):
        return reopen_curation(session_id, cp)
