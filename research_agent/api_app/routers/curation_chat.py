from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.api_app.errors import _upstream_error_guard
from research_agent.api_app.schemas import CurationChatDeleteRequest, CurationChatDeleteResponse, CurationChatRequest, CurationChatResponse
from research_agent.services.curation_chat_service import answer_curation_chat, delete_curation_chat_exchanges
from research_agent.services.errors import ServiceError

router = APIRouter()


@router.post("/curation/{session_id}/chat", response_model=CurationChatResponse)
def curation_chat_turn(session_id: str, req: CurationChatRequest, cp=Depends(api.get_curation_checkpointer)) -> CurationChatResponse:
    with _upstream_error_guard("curation_chat"):
        try:
            return answer_curation_chat(session_id, req, cp)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


# curation-chat-delete Phase 3: POST, not DELETE-with-body -- the one
# existing DELETE route in this API (DELETE /curation/{session_id}) is
# deliberately bodyless (see curation_sessions.py and lib/api/client.ts's
# deleteRequest() helper, which has no body parameter at all), and every
# OTHER payload-carrying mutation here is already POST to an action-
# suffixed path (/picks, /select-from-history, /reopen, /report/
# regenerate) -- this matches that established convention instead of
# being the one DELETE-with-body exception. No _upstream_error_guard: this
# endpoint never calls an LLM/external API, same as curation_delete()
# above it in spirit (curation_sessions.py's plain review-delete route).
@router.post("/curation/{session_id}/chat/exchanges/delete", response_model=CurationChatDeleteResponse)
def curation_chat_delete_exchanges(
    session_id: str, req: CurationChatDeleteRequest, cp=Depends(api.get_curation_checkpointer),
) -> CurationChatDeleteResponse:
    try:
        return delete_curation_chat_exchanges(session_id, req, cp)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
