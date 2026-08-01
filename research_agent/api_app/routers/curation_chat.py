from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.api_app.errors import _upstream_error_guard
from research_agent.api_app.schemas import CurationChatRequest, CurationChatResponse
from research_agent.services.curation_chat_service import answer_curation_chat
from research_agent.services.errors import ServiceError

router = APIRouter()


@router.post("/curation/{session_id}/chat", response_model=CurationChatResponse)
def curation_chat_turn(session_id: str, req: CurationChatRequest, cp=Depends(api.get_curation_checkpointer)) -> CurationChatResponse:
    with _upstream_error_guard("curation_chat"):
        try:
            return answer_curation_chat(session_id, req, cp)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
