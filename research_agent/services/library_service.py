from __future__ import annotations

import sqlite3

import research_agent.api as api
from research_agent.api_app.schemas import LibraryItem, SearchResponse
from research_agent.api_app.serializers import _paper_to_out, _web_article_to_out, _web_articles_from_saved
from research_agent.storage import DEFAULT_LIST_LIMIT, get_search, list_searches


def list_library_items(
    db: sqlite3.Connection, limit: int = DEFAULT_LIST_LIMIT
) -> list[LibraryItem]:
    saved_list = list_searches(db, limit)
    return [
        LibraryItem(
            search_id=s.id, topic=s.topic, created_at=s.created_at,
            paper_count=len(s.paper_ids), has_summary=s.summary is not None,
            web_article_count=len(s.web_articles),
        )
        for s in saved_list
    ]


def get_library_item(db: sqlite3.Connection, search_id: int) -> SearchResponse | None:
    saved = get_search(db, search_id)
    if saved is None:
        return None

    papers = api.get_papers_by_ids(saved.paper_ids, collection=api._state["collection"])
    scores_by_id = dict(zip(saved.paper_ids, saved.scores))
    return SearchResponse(
        search_id=saved.id, topic=saved.topic, created_at=saved.created_at,
        papers=[_paper_to_out(p, scores_by_id.get(p.paper_id)) for p in papers],
        web_articles=[_web_article_to_out(a) for a in _web_articles_from_saved(saved)],
    )
