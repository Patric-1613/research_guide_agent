from fastapi import APIRouter, Depends

import research_agent.api as api
from research_agent.api_app.errors import _upstream_error_guard
from research_agent.api_app.schemas import (
    CurationChatAddToReportRequest,
    CurationChatAddToReportResponse,
    CurationChatDeleteRequest,
    CurationChatDeleteResponse,
    CurationChatEditRequest,
    CurationChatEditResponse,
    CurationChatRequest,
    CurationChatResponse,
)
from research_agent.services.curation_chat_service import (
    add_curation_chat_exchanges_to_report,
    answer_curation_chat,
    delete_curation_chat_exchanges,
    edit_curation_chat_exchange,
    stream_answer_curation_chat,
)

router = APIRouter()


@router.post("/curation/{session_id}/chat", response_model=CurationChatResponse)
def curation_chat_turn(session_id: str, req: CurationChatRequest, cp=Depends(api.get_curation_checkpointer)) -> CurationChatResponse:
    with _upstream_error_guard("curation_chat"):
        return answer_curation_chat(session_id, req, cp)


# Usage Protection M4.2A Part E: the streaming counterpart -- the
# existing non-streaming endpoint above is completely unchanged, still
# the default for any client that hasn't adopted streaming yet. No
# _upstream_error_guard here: unlike the endpoint above, nothing this
# route does BEFORE returning the StreamingResponse ever calls an
# LLM/external API synchronously (every provider call happens later,
# inside the streamed generator body, which reports its own safe error
# events instead -- see curation_chat_streaming.py's own docstring).
@router.post("/curation/{session_id}/chat/stream")
def curation_chat_turn_stream(session_id: str, req: CurationChatRequest, cp=Depends(api.get_curation_checkpointer)):
    return stream_answer_curation_chat(session_id, req, cp)


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
    return delete_curation_chat_exchanges(session_id, req, cp)


# curation-chat-add-to-report Phase 4: same POST-not-DELETE-with-body
# convention as the delete endpoint above. This one DOES call an LLM
# (report regeneration) so it needs _upstream_error_guard, unlike delete.
@router.post("/curation/{session_id}/chat/exchanges/add-to-report", response_model=CurationChatAddToReportResponse)
def curation_chat_add_to_report(
    session_id: str, req: CurationChatAddToReportRequest, cp=Depends(api.get_curation_checkpointer),
) -> CurationChatAddToReportResponse:
    with _upstream_error_guard("curation_chat_add_to_report"):
        return add_curation_chat_exchanges_to_report(session_id, req, cp)


# curation-chat-edit Phase 5: same POST-not-DELETE-with-body convention.
# Calls chat_turn() (an LLM call) for the fresh answer, so needs
# _upstream_error_guard, same as add-to-report above.
@router.post("/curation/{session_id}/chat/exchanges/edit", response_model=CurationChatEditResponse)
def curation_chat_edit_exchange(
    session_id: str, req: CurationChatEditRequest, cp=Depends(api.get_curation_checkpointer),
) -> CurationChatEditResponse:
    with _upstream_error_guard("curation_chat_edit_exchange"):
        return edit_curation_chat_exchange(session_id, req, cp)
