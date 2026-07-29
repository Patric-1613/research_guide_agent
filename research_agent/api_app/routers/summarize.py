import sqlite3

from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.storage import get_db_connection, get_search

router = APIRouter()


@router.post("/summarize", response_model=api.SummarizeResponse)
def summarize(req: api.SummarizeRequest, db: sqlite3.Connection = Depends(get_db_connection)) -> api.SummarizeResponse:
    with api._upstream_error_guard("summarize"):
        saved = get_search(db, req.search_id)
        if saved is None:
            raise HTTPException(status_code=404, detail="search_id not found")

        summary_json = api._get_or_create_summary(db, req.search_id, saved, style=req.style)
        web_summary_json = api._get_or_create_web_summary(db, req.search_id, saved)
        web_summary_out = api.WebSummaryOut(**web_summary_json) if web_summary_json is not None else None
        return api.SummarizeResponse(
            search_id=req.search_id, topic=saved.topic, style=req.style, web_summary=web_summary_out, **summary_json,
        )
