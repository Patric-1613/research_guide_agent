import sqlite3

from fastapi import APIRouter, Depends

import research_agent.api as api
from research_agent.api_app.schemas import SearchRequest, SearchResponse
from research_agent.services.search_service import run_search
from research_agent.storage import get_db_connection

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, db: sqlite3.Connection = Depends(get_db_connection)) -> SearchResponse:
    with api._upstream_error_guard("search"):
        return run_search(db, req)
