from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.curation_session import load_curation_session, save_curation_session

router = APIRouter()


@router.post("/curation/{session_id}/chat", response_model=api.CurationChatResponse)
def curation_chat_turn(session_id: str, req: api.CurationChatRequest, cp=Depends(api.get_curation_checkpointer)) -> api.CurationChatResponse:
    with api._upstream_error_guard("curation_chat"):
        session = load_curation_session(session_id, cp)
        if session is None:
            raise HTTPException(status_code=404, detail="session_id not found")
        try:
            result = api.chat_turn(session, req.message, client=api._state["client"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        save_curation_session(session, session_id, cp)

        return api.CurationChatResponse(
            answer=result["answer"], answerable=result["answerable"],
            cited_papers=[api.CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in result["cited_papers"]],
            cited_web_articles=[api.CitedWebArticleOut(url=a.url, title=a.title) for a in result["cited_web_articles"]],
            web_offer_made=result.get("web_offer_made", False),
            web_offer_declined=result.get("web_offer_declined", False),
            web_search_used=result.get("web_search_used", False),
            new_web_articles_found=result.get("new_web_articles_found"),
            report_update_offer_made=result.get("report_update_offer_made", False),
            report_update_declined=result.get("report_update_declined", False),
            report_updated=result.get("report_updated", False),
            chat_history=[api.ChatTurn(**turn) for turn in session.chat_history],
        )
