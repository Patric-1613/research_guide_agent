import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Query

from research_agent.api_app.schemas import LibraryItem, SearchResponse
from research_agent.services.library_service import get_library_item, list_library_items
from research_agent.storage import DEFAULT_LIST_LIMIT, MAX_LIST_LIMIT, get_db_connection

router = APIRouter()


@router.get("/library", response_model=list[LibraryItem])
def library(
    limit: int = Query(DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT),
    db: sqlite3.Connection = Depends(get_db_connection),
) -> list[LibraryItem]:
    return list_library_items(db, limit)


@router.get("/library/{search_id}", response_model=SearchResponse)
def library_detail(search_id: int, db: sqlite3.Connection = Depends(get_db_connection)) -> SearchResponse:
    result = get_library_item(db, search_id)
    if result is None:
        raise HTTPException(status_code=404, detail="search_id not found")
    return result
