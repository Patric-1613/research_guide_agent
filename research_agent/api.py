"""Phase 7: FastAPI backend wiring phases 1-6 together, with SQLite-persisted
search history so /summarize, /chat, and /export can operate statelessly
across separate HTTP requests — a client only needs to hold on to a
search_id, not the paper set itself. The server resolves papers back out of
Chroma (already the persistence layer for paper content since phase 3) via
the paper_ids saved in SQLite for that search_id.

/search invokes the full phase-4 agent (source selection, query
reformulation, its own rerank call) rather than calling ingestion/dedup/rank
directly — that orchestration *is* what phase 4 was for. If the agent
finishes without having called its rerank tool (LLM tool use isn't 100%
guaranteed every run), this falls back to reranking server-side rather than
returning nothing.

Chat history is NOT persisted server-side: the client carries it forward
turn-to-turn in the request body. The brief's SQLite requirement covers
saved *searches* (topic/papers/summary), not chat transcripts, and a
request-scoped history keeps this endpoint stateless without adding a
second persistence concept for a single-user v1 app.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from openai import OpenAI

from research_agent.agent import run_research_agent
from research_agent.curation_chat import chat_turn
from research_agent.curation_loop import get_curation_state, resume_curation_turn, start_curation_turn
from research_agent.curation_session import (
    _session_to_dict,
    delete_curation_session,
    list_curation_sessions,
    load_curation_session,
    reopen_curation_session,
    save_curation_session,
    select_paper_from_history,
)
from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, get_papers_by_ids, semantic_search
from research_agent.qa import ChatSession, ask, sqlite_checkpointer
from research_agent.query_expansion import (
    PaperPoolSession,
    build_candidate_pool,
    canonicalize_topic,
    expanded_search,
    rank_full_pool,
)
from research_agent.report import generate_report_for_session, regenerate_report_with_new_sources
from research_agent.storage import get_db_connection, get_search, init_db, list_searches, save_search
from research_agent.summarize import generate_summary, generate_web_summary
from research_agent.web_search import search_web

# Phase 4: request/response models and pure serialization/rendering helpers
# now live in api_app/schemas.py and api_app/serializers.py — re-exported
# here so `research_agent.api.<name>` and `patch.object(api, "<name>", ...)`
# keep working unchanged for every existing caller/test.
from research_agent.api_app.schemas import (
    ChatRequest,
    ChatResponse,
    ChatTurn,
    CitedPaperOut,
    CitedWebArticleOut,
    CurationChatRequest,
    CurationChatResponse,
    CurationDeleteResponse,
    CurationPicksRequest,
    CurationReviewSummary,
    CurationSelectFromHistoryRequest,
    CurationSelectFromHistoryResponse,
    CurationStartRequest,
    CurationStateResponse,
    CurationTurnResponse,
    LibraryItem,
    PaperOut,
    PaperSummaryOut,
    ReportOut,
    ReportSectionOut,
    SearchRequest,
    SearchResponse,
    SummarizeRequest,
    SummarizeResponse,
    ThemeOut,
    TurnHistoryEntryOut,
    WebArticleOut,
    WebSummaryOut,
)
from research_agent.api_app.serializers import (
    _paper_out_from_batch_entry,
    _paper_to_out,
    _render_markdown,
    _report_to_out,
    _summary_to_json,
    _turn_history_out,
    _turn_result_to_response,
    _web_article_to_out,
    _web_articles_from_saved,
    _web_summary_to_json,
)

# Phase 6: non-endpoint behavioral helpers now live in api_app/errors.py and
# services/{summary_cache,search_helpers,curation_helpers}.py — re-exported
# here for the same reason as the Phase 4 block above. Each of those modules
# reaches back into this module (`import research_agent.api as api`) for
# _state and every patch-targeted function; that's safe here despite the
# apparent circularity because Python only needs the (possibly still-
# loading) `research_agent.api` module *object* to exist in sys.modules at
# their import time — the same mechanism api_app/routers/*.py has relied on
# since Phase 2 — and none of them touch `api.<name>` until a function is
# actually called, long after this module has finished loading.
from research_agent.api_app.errors import _UPSTREAM_ERRORS, _upstream_error_guard
from research_agent.services.curation_helpers import _curation_config
from research_agent.services.search_helpers import _filtered_candidate_count, _merge_web_articles, _server_side_rerank
from research_agent.services.summary_cache import _get_or_create_summary, _get_or_create_web_summary, _reselect_style

load_dotenv()

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema creation/migration only needs to happen once, on a throwaway
    # connection — every request after this opens its own via get_db_connection
    # (a FastAPI dependency, safe for the multi-threaded request handling a
    # single shared connection was not).
    init_db().close()
    _state["client"] = OpenAI()
    _state["collection"] = get_chroma_collection()
    yield


app = FastAPI(title="Research Paper Summarizer API", lifespan=lifespan)

# curation-api-and-ui Phase 6c: the React frontend runs as its own Vite
# dev-server process -- a genuinely separate origin from this API, so
# CORS is required for its browser-side fetch calls. FRONTEND_ORIGIN lets
# a non-default dev-server port/deployed origin be configured without a
# code change, same convention as this file's other env-var-driven config
# (e.g. SEMANTIC_SCHOLAR_API_KEY).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"), "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


from research_agent.api_app.routers.health import router as health_router

app.include_router(health_router)


from research_agent.api_app.routers.search import router as search_router

app.include_router(search_router)


from research_agent.api_app.routers.summarize import router as summarize_router

app.include_router(summarize_router)


from research_agent.api_app.routers.chat import router as chat_router

app.include_router(chat_router)


from research_agent.api_app.routers.export import router as export_router

app.include_router(export_router)


from research_agent.api_app.routers.library import router as library_router

app.include_router(library_router)


# =================================================================================
# curation-api-and-ui Phase 6a: HTTP exposure for the curation/report/chat
# backend built in Phases 1-5. Purely additive — nothing above this line is
# touched. A curation session lives in its own checkpointer-backed store
# (qa.py's sqlite_checkpointer/QA_CHECKPOINT_DB_PATH, the same file
# curation_session.py already used), addressed by a server-issued
# session_id (a uuid4 hex string), not storage.py's SQLite search_id — a
# genuinely different persistence mechanism from /search's, reused as-is
# rather than adapted to fit the existing one.
#
# The interrupt/resume HTTP shape (the brief's "least obvious" part): both
# /curation/start and /curation/{id}/picks return the SAME
# CurationTurnResponse shape — either a fresh `batch` to present (still
# curating) or a `stop_reason` (curation finished, batch always []). A
# client never needs to special-case "first turn" vs. "a later turn"; the
# response shape alone tells it what to render next.
# =================================================================================

def get_curation_checkpointer():
    """FastAPI dependency: a fresh SqliteSaver-backed checkpointer per
    request, same per-request-not-shared rationale as get_db_connection()
    above (a single shared connection is not safe across FastAPI's
    threadpool) — just wrapping sqlite_checkpointer()'s own contextmanager
    instead of a raw sqlite3.connect() call."""
    with sqlite_checkpointer() as cp:
        yield cp


from research_agent.api_app.routers.curation_core import router as curation_core_router

app.include_router(curation_core_router)


from research_agent.api_app.routers.curation_sessions import router as curation_sessions_router

app.include_router(curation_sessions_router)


from research_agent.api_app.routers.curation_history import router as curation_history_router
from research_agent.api_app.routers.curation_reports import router as curation_reports_router

app.include_router(curation_history_router)
app.include_router(curation_reports_router)


from research_agent.api_app.routers.curation_chat import router as curation_chat_router

app.include_router(curation_chat_router)
