import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from research_agent.api_app.errors import _upstream_error_guard
from research_agent.api_app.schemas import SearchRequest, SearchResponse
from research_agent.services.errors import ServiceError
from research_agent.services.search_service import run_search
from research_agent.storage import get_db_connection

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, db: sqlite3.Connection = Depends(get_db_connection)) -> SearchResponse:
    with _upstream_error_guard("search"):
        try:
            return run_search(db, req)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
