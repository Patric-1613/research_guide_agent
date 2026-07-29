from fastapi import APIRouter, Depends

import research_agent.api as api
from research_agent.services.curation_history_service import reopen_curation, select_from_history

router = APIRouter()


@router.post("/curation/{session_id}/select-from-history", response_model=api.CurationSelectFromHistoryResponse)
def curation_select_from_history(
    session_id: str, req: api.CurationSelectFromHistoryRequest, cp=Depends(api.get_curation_checkpointer),
) -> api.CurationSelectFromHistoryResponse:
    return select_from_history(session_id, req, cp)


@router.post("/curation/{session_id}/reopen", response_model=api.CurationTurnResponse)
def curation_reopen(session_id: str, cp=Depends(api.get_curation_checkpointer)) -> api.CurationTurnResponse:
    with api._upstream_error_guard("curation_reopen"):
        return reopen_curation(session_id, cp)
