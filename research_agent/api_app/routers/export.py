import sqlite3

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

import research_agent.api as api
from research_agent.citations import CitationStyle
from research_agent.storage import get_db_connection, get_search

router = APIRouter()


@router.get("/export/{search_id}", response_class=PlainTextResponse)
def export(search_id: int, style: CitationStyle = "apa", db: sqlite3.Connection = Depends(get_db_connection)) -> str:
    with api._upstream_error_guard("export"):
        saved = get_search(db, search_id)
        if saved is None:
            raise HTTPException(status_code=404, detail="search_id not found")

        summary_json = api._get_or_create_summary(db, search_id, saved, style=style)
        web_summary_json = api._get_or_create_web_summary(db, search_id, saved)
        return api._render_markdown(saved.topic, summary_json, style=style, web_summary_json=web_summary_json)
