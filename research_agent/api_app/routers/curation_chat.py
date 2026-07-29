from fastapi import APIRouter, Depends

import research_agent.api as api
from research_agent.services.curation_chat_service import answer_curation_chat

router = APIRouter()


@router.post("/curation/{session_id}/chat", response_model=api.CurationChatResponse)
def curation_chat_turn(session_id: str, req: api.CurationChatRequest, cp=Depends(api.get_curation_checkpointer)) -> api.CurationChatResponse:
    with api._upstream_error_guard("curation_chat"):
        return answer_curation_chat(session_id, req, cp)
