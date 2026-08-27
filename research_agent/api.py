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

from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.responses import PlainTextResponse
from openai import OpenAI

from research_agent.agent import run_research_agent
from research_agent.curation_chat import (
    approve_web_article_urls,
    chat_turn,
    cited_web_article_urls_for_exchanges,
    delete_chat_exchanges,
    edit_chat_exchange,
    mark_exchanges_added_to_report,
    resolve_approved_web_articles_for_regeneration,
    select_eligible_exchanges_for_report,
)
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
from research_agent.lane_suggestion import suggest_lanes
from research_agent.qa import ChatSession, ask
from research_agent.research_lane_retrieval import retrieve_across_lanes
from research_agent.query_expansion import (
    PaperPoolSession,
    build_candidate_pool,
    canonicalize_topic,
    expanded_search,
    rank_full_pool,
)
from research_agent.report import (
    GENERATION_REASON_CHAT_ADD_TO_REPORT,
    GENERATION_REASON_INITIAL,
    GENERATION_REASON_REGENERATE,
    activate_report_version,
    append_report_version,
    build_report_export_document,
    generate_report_for_session,
    get_active_report_version,
    refine_report_if_requested,
    regenerate_report_with_approved_web_sources,
    regenerate_report_with_new_sources,
    render_report_docx,
    render_report_markdown,
    render_report_pdf,
    report_export_filename,
)
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
    LaneSuggestRequest,
    LaneSuggestResponse,
    LibraryItem,
    PaperOut,
    PaperSummaryOut,
    ReportOut,
    ReportSectionOut,
    ResearchLaneOut,
    SearchRequest,
    SearchResponse,
    SubmittedLane,
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
    _research_lane_to_out,
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

# Phase 9: shared runtime state and the curation checkpointer dependency
# now live in api_app/runtime.py — re-exported here (the same dict/
# function objects, not wrappers) so `research_agent.api._state` and
# `research_agent.api.get_curation_checkpointer` keep resolving exactly
# as before. `_state` is a plain mutable dict; api_app/app.py's
# lifespan() mutates it in place, so every reader of `api._state` sees
# the same updates regardless of which module holds the name.
# get_curation_checkpointer is imported as-is (never wrapped), so
# `app.dependency_overrides[api.get_curation_checkpointer]` (keyed by
# callable identity) keeps matching every router's
# `Depends(api.get_curation_checkpointer)` unchanged.
from research_agent.api_app.runtime import _state, get_curation_checkpointer

load_dotenv()

# Phase 10: FastAPI app creation, lifespan, CORS setup, and all
# app.include_router(...) calls now live in api_app/app.py's
# create_app() — research_agent.api:app remains the exact same public
# ASGI entrypoint (`uvicorn research_agent.api:app` boots this object),
# just constructed by create_app() instead of inline here. Router
# registration order is unchanged (health, search, summarize, chat,
# export, library, curation_core, curation_sessions, curation_history,
# curation_reports, curation_chat).
from research_agent.api_app.app import create_app

app = create_app()
