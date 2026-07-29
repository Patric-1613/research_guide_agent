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

import logging
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager, contextmanager

import arxiv
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from openai import OpenAI, OpenAIError

from research_agent.agent import _merge_web_articles, run_research_agent
from research_agent.citations import CitationStyle, select_citation
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
from research_agent.enrichment import enrich_missing_abstracts
from research_agent.qa import ChatSession, ask, sqlite_checkpointer
from research_agent.query_expansion import (
    PaperPoolSession,
    build_candidate_pool,
    canonicalize_topic,
    expanded_search,
    rank_full_pool,
)
from research_agent.report import generate_report_for_session, regenerate_report_with_new_sources
from research_agent.schema import Paper
from research_agent.storage import get_db_connection, get_search, init_db, list_searches, save_search, update_summary, update_web_summary
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

load_dotenv()

logger = logging.getLogger(__name__)

_state: dict = {}

# Exception types raised by the upstream services this API depends on
# (OpenAI, arXiv, Semantic Scholar/requests) that should surface as a clean
# error response, not a raw 500 with a stack trace leaked to the caller.
_UPSTREAM_ERRORS = (OpenAIError, arxiv.ArxivError, requests.RequestException)


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


# ---- helpers ------------------------------------------------------------------

def _reselect_style(summary_json: dict, style: CitationStyle) -> dict:
    """Re-picks the `citation` field for a cached summary against a
    possibly different style than the one it was first generated with.
    Citation formatting is pure/cheap string logic, not an LLM call — a
    cache hit still needs to honor whatever style THIS request asked for,
    and doing that costs nothing beyond a dict lookup (see
    citations.select_citation)."""
    return {
        "themes": [
            {
                "theme_name": theme["theme_name"],
                "papers": [
                    {
                        **p,
                        "citation": select_citation(
                            p.get("apa_citation", ""),
                            p.get("harvard_citation") or p.get("apa_citation", ""),
                            p.get("bibtex", ""),
                            style,
                        ),
                    }
                    for p in theme["papers"]
                ],
            }
            for theme in summary_json["themes"]
        ],
        "gaps_and_disagreements": summary_json["gaps_and_disagreements"],
        "skipped_paper_ids": summary_json["skipped_paper_ids"],
    }


def _get_or_create_summary(db: sqlite3.Connection, search_id: int, saved, style: CitationStyle = "apa") -> dict:
    """Reuse a previously-generated summary if one exists for this
    search_id, rather than re-billing the LLM every time /summarize or
    /export is called for the same search — mirrors the embedding cache's
    cost-consciousness from phase 3. A different `style` than the one the
    summary was originally generated with is still honored on a cache hit
    (via _reselect_style) since picking a citation format costs nothing."""
    if saved.summary is not None:
        return _reselect_style(saved.summary, style)
    papers = get_papers_by_ids(saved.paper_ids, collection=_state["collection"])
    result = generate_summary(saved.topic, papers, client=_state["client"], style=style)
    summary_json = _summary_to_json(result, style=style)
    update_summary(db, search_id, summary_json)
    return summary_json


def _get_or_create_web_summary(db: sqlite3.Connection, search_id: int, saved) -> dict | None:
    """Mirrors _get_or_create_summary's cost-consciousness for the separate
    web-article corpus — its own cache column, never merged into the paper
    summary's cache. Returns None if this search found no web articles at
    all, so callers render the paper summary alone rather than an empty
    web-context block."""
    if not saved.web_articles:
        return None
    if saved.web_summary is not None:
        return saved.web_summary
    articles = _web_articles_from_saved(saved)
    result = generate_web_summary(saved.topic, articles, client=_state["client"])
    web_summary_json = _web_summary_to_json(result)
    update_web_summary(db, search_id, web_summary_json)
    return web_summary_json


# ---- endpoints ------------------------------------------------------------------

@contextmanager
def _upstream_error_guard(service: str):
    """Wraps an endpoint body that calls out to arXiv, Semantic Scholar, or
    OpenAI. Those calls already retry/degrade internally where they can
    (ingestion.py, embeddings.py) — this is the last line of defense for
    what still gets through: a raw 500 with an internal stack trace leaking
    to the caller instead of a clean, actionable error response.

    HTTPException is re-raised untouched — those are this API's own
    intentional 404s (e.g. "search_id not found"), not upstream failures,
    and must not be swallowed into a 503.
    """
    try:
        yield
    except HTTPException:
        raise
    except _UPSTREAM_ERRORS as exc:
        logger.exception("Upstream service failure during %s", service)
        raise HTTPException(status_code=503, detail={"error": f"{service} service unavailable"}) from exc


def _server_side_rerank(
    session, topic: str, top_k: int, doi_required: bool = False, min_citation_count: int = 0,
):
    collection = _state["collection"]
    client = _state["client"]
    # If the agent's own rerank tool never ran (why we're in this fallback
    # at all), session.papers may not have gone through abstract recovery
    # yet either — try it here too. Cached by DOI, so if it already ran
    # this is just a cheap SQLite lookup, not a repeat network round trip.
    enrich_missing_abstracts(session.papers)
    embed_and_index_papers(session.papers, collection=collection, client=client)
    ids = [p.paper_id for p in session.papers]
    return semantic_search(
        topic, collection=collection, client=client,
        top_k=top_k, where={"paper_id": {"$in": ids}},
        require_doi=doi_required, min_citation_count=min_citation_count or None,
    )


def _filtered_candidate_count(papers: list[Paper], doi_required: bool, min_citation_count: int) -> int:
    """How many of the agent's gathered papers would survive the requested
    filters — used only to decide whether the agent's own ranking already
    honored top_k/filters, or whether a server-side re-rank is needed. Pure
    Python over already-in-memory Paper objects, no extra API/LLM cost."""
    count = 0
    for p in papers:
        if doi_required and not p.doi:
            continue
        if min_citation_count and (p.citation_count or 0) < min_citation_count:
            continue
        count += 1
    return count


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


def _curation_config() -> dict:
    """refill_pool() (called from inside the graph whenever the reserve
    runs low, on ANY turn, not just the first) requires "client" present
    with direct key access, not .get() — confirmed by hitting a real
    KeyError during Phase 6a's own design verification when it was
    omitted. Built fresh per request rather than cached in _state,
    since s2_api_key/openalex_mailto are read from the environment the
    same way /search already does, not assumed constant."""
    return {
        "client": _state["client"],
        "s2_api_key": os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None,
        "openalex_mailto": os.getenv("OPENALEX_MAILTO") or None,
    }


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
