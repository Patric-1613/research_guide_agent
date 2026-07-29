import sqlite3

from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.storage import get_db_connection, get_search, list_searches

router = APIRouter()


@router.get("/library", response_model=list[api.LibraryItem])
def library(db: sqlite3.Connection = Depends(get_db_connection)) -> list[api.LibraryItem]:
    saved_list = list_searches(db)
    return [
        api.LibraryItem(
            search_id=s.id, topic=s.topic, created_at=s.created_at,
            paper_count=len(s.paper_ids), has_summary=s.summary is not None,
            web_article_count=len(s.web_articles),
        )
        for s in saved_list
    ]


@router.get("/library/{search_id}", response_model=api.SearchResponse)
def library_detail(search_id: int, db: sqlite3.Connection = Depends(get_db_connection)) -> api.SearchResponse:
    saved = get_search(db, search_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="search_id not found")

    papers = api.get_papers_by_ids(saved.paper_ids, collection=api._state["collection"])
    scores_by_id = dict(zip(saved.paper_ids, saved.scores))
    return api.SearchResponse(
        search_id=saved.id, topic=saved.topic, created_at=saved.created_at,
        papers=[api._paper_to_out(p, scores_by_id.get(p.paper_id)) for p in papers],
        web_articles=[api._web_article_to_out(a) for a in api._web_articles_from_saved(saved)],
    )
