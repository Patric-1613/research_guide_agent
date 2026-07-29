from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.api_app.errors import _upstream_error_guard
from research_agent.api_app.schemas import (
    CurationSelectFromHistoryRequest,
    CurationSelectFromHistoryResponse,
    CurationTurnResponse,
)
from research_agent.services.curation_history_service import reopen_curation, select_from_history
from research_agent.services.errors import ServiceError

router = APIRouter()


@router.post("/curation/{session_id}/select-from-history", response_model=CurationSelectFromHistoryResponse)
def curation_select_from_history(
    session_id: str, req: CurationSelectFromHistoryRequest, cp=Depends(api.get_curation_checkpointer),
) -> CurationSelectFromHistoryResponse:
    try:
        return select_from_history(session_id, req, cp)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post("/curation/{session_id}/reopen", response_model=CurationTurnResponse)
def curation_reopen(session_id: str, cp=Depends(api.get_curation_checkpointer)) -> CurationTurnResponse:
    with _upstream_error_guard("curation_reopen"):
        try:
            return reopen_curation(session_id, cp)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
