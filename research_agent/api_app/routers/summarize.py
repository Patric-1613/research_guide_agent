import sqlite3

from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.api_app.schemas import SummarizeRequest, SummarizeResponse
from research_agent.services.summary_service import summarize_search
from research_agent.storage import get_db_connection

router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest, db: sqlite3.Connection = Depends(get_db_connection)) -> SummarizeResponse:
    with api._upstream_error_guard("summarize"):
        result = summarize_search(db, req.search_id, req.style)
        if result is None:
            raise HTTPException(status_code=404, detail="search_id not found")
        return result
