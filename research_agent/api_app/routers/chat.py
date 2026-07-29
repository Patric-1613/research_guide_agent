import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from research_agent.api_app.errors import _upstream_error_guard
from research_agent.api_app.schemas import ChatRequest, ChatResponse
from research_agent.services.chat_service import answer_search_chat
from research_agent.storage import get_db_connection

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: sqlite3.Connection = Depends(get_db_connection)) -> ChatResponse:
    with _upstream_error_guard("chat"):
        result = answer_search_chat(db, req.search_id, req.question, req.history)
        if result is None:
            raise HTTPException(status_code=404, detail="search_id not found")
        return result
