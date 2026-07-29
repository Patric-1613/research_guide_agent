import sqlite3

from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.services.chat_service import answer_search_chat
from research_agent.storage import get_db_connection

router = APIRouter()


@router.post("/chat", response_model=api.ChatResponse)
def chat(req: api.ChatRequest, db: sqlite3.Connection = Depends(get_db_connection)) -> api.ChatResponse:
    with api._upstream_error_guard("chat"):
        result = answer_search_chat(db, req.search_id, req.question, req.history)
        if result is None:
            raise HTTPException(status_code=404, detail="search_id not found")
        return result
