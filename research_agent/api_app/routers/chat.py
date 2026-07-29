import sqlite3

from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.qa import ChatSession
from research_agent.storage import get_db_connection, get_search

router = APIRouter()


@router.post("/chat", response_model=api.ChatResponse)
def chat(req: api.ChatRequest, db: sqlite3.Connection = Depends(get_db_connection)) -> api.ChatResponse:
    with api._upstream_error_guard("chat"):
        saved = get_search(db, req.search_id)
        if saved is None:
            raise HTTPException(status_code=404, detail="search_id not found")

        papers = api.get_papers_by_ids(saved.paper_ids, collection=api._state["collection"])
        web_articles = api._web_articles_from_saved(saved)
        session = ChatSession(papers=papers, web_articles=web_articles, history=[turn.model_dump() for turn in req.history])
        result = api.ask(session, req.question, client=api._state["client"])

        return api.ChatResponse(
            answer=result["answer"],
            answerable=result["answerable"],
            cited_papers=[api.CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in result["cited_papers"]],
            cited_web_articles=[api.CitedWebArticleOut(url=a.url, title=a.title) for a in result.get("cited_web_articles", [])],
            history=[api.ChatTurn(**turn) for turn in session.history],
        )
