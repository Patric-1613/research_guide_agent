from __future__ import annotations

import sqlite3

import research_agent.api as api
from research_agent.api_app.schemas import ChatResponse, ChatTurn, CitedPaperOut, CitedWebArticleOut
from research_agent.api_app.serializers import _web_articles_from_saved
from research_agent.qa import ChatSession
from research_agent.storage import get_search
from research_agent.usage_guard import guard_paid_action


def answer_search_chat(db: sqlite3.Connection, search_id: int, question: str, history: list) -> ChatResponse | None:
    saved = get_search(db, search_id)
    if saved is None:
        return None

    papers = api.get_papers_by_ids(saved.paper_ids, collection=api._state["collection"])
    web_articles = _web_articles_from_saved(saved)
    session = ChatSession(papers=papers, web_articles=web_articles, history=[turn.model_dump() for turn in history])
    # Usage Protection M2.2A: search_id is a stable opaque subject
    # (already established by M1's own subject_type="search" convention),
    # so this gets full hourly/daily/global admission. No lease -- the
    # concurrency lease is scoped to "session" subjects (curation work);
    # this is the older, single-shot /search chat, not curation.
    with guard_paid_action("search_chat", subject=("search", str(search_id))):
        result = api.ask(session, question, client=api._state["client"])

    return ChatResponse(
        answer=result["answer"],
        answerable=result["answerable"],
        cited_papers=[CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in result["cited_papers"]],
        cited_web_articles=[CitedWebArticleOut(url=a.url, title=a.title) for a in result.get("cited_web_articles", [])],
        history=[ChatTurn(**turn) for turn in session.history],
    )
