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
    # Usage Protection M2.2A/M2.2B: search_id is a stable opaque subject
    # (already established by M1's own subject_type="search" convention),
    # so this gets full hourly/daily/global admission.
    #
    # Deliberately NO lease, decided by tracing this function end to end
    # (M2.2B), not left over from M2.2A by default: `history` above comes
    # entirely from the caller's own request body (`req.history` in
    # api_app/routers/chat.py), never read from any server-side store for
    # this search_id, and `session` is a fresh, per-call, in-memory
    # ChatSession -- nothing in this function writes to `db` or any other
    # shared store (unlike curation chat, which persists chat_history/
    # session state via save_curation_session). Two concurrent /chat
    # calls for the same search_id each build their own independent
    # ChatSession and return their own response; neither can overwrite,
    # duplicate, or race the other's persisted state, because there IS
    # no persisted per-search_id chat state to race over. Proven by
    # tests/test_api.py::test_concurrent_search_chat_same_search_id_
    # produce_independent_responses_no_shared_state (real threads).
    with guard_paid_action("search_chat", subject=("search", str(search_id))):
        result = api.ask(session, question, client=api._state["client"])

    return ChatResponse(
        answer=result["answer"],
        answerable=result["answerable"],
        cited_papers=[CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in result["cited_papers"]],
        cited_web_articles=[CitedWebArticleOut(url=a.url, title=a.title) for a in result.get("cited_web_articles", [])],
        history=[ChatTurn(**turn) for turn in session.history],
    )
