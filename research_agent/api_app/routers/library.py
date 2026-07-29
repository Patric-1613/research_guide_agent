import sqlite3

from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.services.library_service import get_library_item, list_library_items
from research_agent.storage import get_db_connection

router = APIRouter()


@router.get("/library", response_model=list[api.LibraryItem])
def library(db: sqlite3.Connection = Depends(get_db_connection)) -> list[api.LibraryItem]:
    return list_library_items(db)


@router.get("/library/{search_id}", response_model=api.SearchResponse)
def library_detail(search_id: int, db: sqlite3.Connection = Depends(get_db_connection)) -> api.SearchResponse:
    result = get_library_item(db, search_id)
    if result is None:
        raise HTTPException(status_code=404, detail="search_id not found")
    return result
