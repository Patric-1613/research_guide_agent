import sqlite3

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

import research_agent.api as api
from research_agent.citations import CitationStyle
from research_agent.services.summary_service import export_search_markdown
from research_agent.storage import get_db_connection

router = APIRouter()


@router.get("/export/{search_id}", response_class=PlainTextResponse)
def export(search_id: int, style: CitationStyle = "apa", db: sqlite3.Connection = Depends(get_db_connection)) -> str:
    with api._upstream_error_guard("export"):
        result = export_search_markdown(db, search_id, style)
        if result is None:
            raise HTTPException(status_code=404, detail="search_id not found")
        return result
