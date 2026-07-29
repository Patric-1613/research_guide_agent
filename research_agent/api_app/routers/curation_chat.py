from fastapi import APIRouter, Depends

import research_agent.api as api
from research_agent.api_app.schemas import CurationChatRequest, CurationChatResponse
from research_agent.services.curation_chat_service import answer_curation_chat

router = APIRouter()


@router.post("/curation/{session_id}/chat", response_model=CurationChatResponse)
def curation_chat_turn(session_id: str, req: CurationChatRequest, cp=Depends(api.get_curation_checkpointer)) -> CurationChatResponse:
    with api._upstream_error_guard("curation_chat"):
        return answer_curation_chat(session_id, req, cp)
