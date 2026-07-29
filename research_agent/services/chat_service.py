from __future__ import annotations

import sqlite3

import research_agent.api as api
from research_agent.qa import ChatSession
from research_agent.storage import get_search


def answer_search_chat(db: sqlite3.Connection, search_id: int, question: str, history: list) -> api.ChatResponse | None:
    saved = get_search(db, search_id)
    if saved is None:
        return None

    papers = api.get_papers_by_ids(saved.paper_ids, collection=api._state["collection"])
    web_articles = api._web_articles_from_saved(saved)
    session = ChatSession(papers=papers, web_articles=web_articles, history=[turn.model_dump() for turn in history])
    result = api.ask(session, question, client=api._state["client"])

    return api.ChatResponse(
        answer=result["answer"],
        answerable=result["answerable"],
        cited_papers=[api.CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in result["cited_papers"]],
        cited_web_articles=[api.CitedWebArticleOut(url=a.url, title=a.title) for a in result.get("cited_web_articles", [])],
        history=[api.ChatTurn(**turn) for turn in session.history],
    )
