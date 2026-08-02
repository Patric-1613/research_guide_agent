from __future__ import annotations

import research_agent.api as api
from research_agent.api_app.schemas import (
    ChatTurn,
    CitedPaperOut,
    CitedWebArticleOut,
    CurationChatDeleteRequest,
    CurationChatDeleteResponse,
    CurationChatRequest,
    CurationChatResponse,
)
from research_agent.curation_session import load_curation_session, save_curation_session
from research_agent.services.errors import ServiceError


def answer_curation_chat(session_id: str, req: CurationChatRequest, cp) -> CurationChatResponse:
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    try:
        result = api.chat_turn(session, req.message, client=api._state["client"])
    except ValueError as exc:
        raise ServiceError(400, str(exc)) from exc
    save_curation_session(session, session_id, cp)

    return CurationChatResponse(
        answer=result["answer"], answerable=result["answerable"],
        cited_papers=[CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in result["cited_papers"]],
        cited_web_articles=[CitedWebArticleOut(url=a.url, title=a.title) for a in result["cited_web_articles"]],
        web_offer_made=result.get("web_offer_made", False),
        web_offer_declined=result.get("web_offer_declined", False),
        web_search_used=result.get("web_search_used", False),
        new_web_articles_found=result.get("new_web_articles_found"),
        report_update_offer_made=result.get("report_update_offer_made", False),
        report_update_declined=result.get("report_update_declined", False),
        report_updated=result.get("report_updated", False),
        chat_history=[ChatTurn(**turn) for turn in session.chat_history],
    )


def delete_curation_chat_exchanges(session_id: str, req: CurationChatDeleteRequest, cp) -> CurationChatDeleteResponse:
    if not req.exchange_ids:
        raise ServiceError(400, "exchange_ids must not be empty")
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")

    deleted_exchange_ids, report_possibly_stale = api.delete_chat_exchanges(session, req.exchange_ids)
    save_curation_session(session, session_id, cp)

    return CurationChatDeleteResponse(
        chat_history=[ChatTurn(**turn) for turn in session.chat_history],
        deleted_exchange_ids=deleted_exchange_ids,
        report_possibly_stale=report_possibly_stale,
    )
