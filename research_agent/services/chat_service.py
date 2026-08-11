from __future__ import annotations

import sqlite3

import research_agent.api as api
import research_agent.telemetry as telemetry
from research_agent.api_app.schemas import ChatResponse, ChatTurn, CitedPaperOut, CitedWebArticleOut
from research_agent.api_app.serializers import _web_articles_from_saved
from research_agent.qa import ChatSession
from research_agent.storage import get_search


def answer_search_chat(db: sqlite3.Connection, search_id: int, question: str, history: list) -> ChatResponse | None:
    saved = get_search(db, search_id)
    if saved is None:
        return None

    papers = api.get_papers_by_ids(saved.paper_ids, collection=api._state["collection"])
    web_articles = _web_articles_from_saved(saved)
    session = ChatSession(papers=papers, web_articles=web_articles, history=[turn.model_dump() for turn in history])
    with telemetry.paid_action("search_chat", subject_type="search", subject_id=str(search_id)):
        result = api.ask(session, question, client=api._state["client"])

    return ChatResponse(
        answer=result["answer"],
        answerable=result["answerable"],
        cited_papers=[CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in result["cited_papers"]],
        cited_web_articles=[CitedWebArticleOut(url=a.url, title=a.title) for a in result.get("cited_web_articles", [])],
        history=[ChatTurn(**turn) for turn in session.history],
    )
