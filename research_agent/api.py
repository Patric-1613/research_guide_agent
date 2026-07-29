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
from typing import Literal

import arxiv
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

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
from research_agent.schema import Paper, WebArticle
from research_agent.storage import get_db_connection, get_search, init_db, list_searches, save_search, update_summary, update_web_summary
from research_agent.summarize import generate_summary, generate_web_summary
from research_agent.web_search import search_web

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


# ---- request/response models -------------------------------------------------

class SearchRequest(BaseModel):
    topic: str
    # 3-30: a deliberate, code-enforced bound on how many results a single
    # request can ask for, independent of any particular frontend.
    top_k: int = Field(default=10, ge=3, le=30)
    # Round-2 enhancement 2: surfaces doi/citation_count metadata that's
    # already stored per-paper in Chroma (embeddings.py) — no re-indexing
    # needed. min_citation_count=0 means "no filter" per the brief.
    doi_required: bool = False
    min_citation_count: int = Field(default=0, ge=0)
    # Round-2 enhancement 5: independent of top_k — web articles are a
    # separate, smaller pool, never counted alongside the paper results.
    web_max_results: int = Field(default=4, ge=1, le=10)
    # LLM-assisted query expansion (query_expansion.py): default False so
    # existing behavior is unchanged unless explicitly opted into. When
    # True, bypasses the agent's own search/rerank entirely in favor of
    # expanded_search() — see that function's docstring for why. Web
    # article search is agent-only right now, so it's skipped (empty)
    # whenever this is True; a known, deliberate gap for this phase.
    use_query_expansion: bool = False


class PaperOut(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str | None
    url: str | None
    doi: str | None
    citation_count: int | None
    source: str
    source_urls: dict[str, str]
    score: float | None = None


class WebArticleOut(BaseModel):
    title: str
    url: str
    snippet: str
    published_date: str | None
    source_domain: str


class SearchResponse(BaseModel):
    search_id: int
    topic: str
    created_at: str
    papers: list[PaperOut]
    # Round-2 enhancement 5: a genuinely separate section from `papers` —
    # never interleaved with it, never counted toward top_k.
    web_articles: list[WebArticleOut] = []


class SummarizeRequest(BaseModel):
    search_id: int
    style: CitationStyle = "apa"


class PaperSummaryOut(BaseModel):
    paper_id: str
    title: str
    summary: str
    apa_citation: str
    harvard_citation: str
    bibtex: str
    citation: str  # whichever of the above matches the requested style


class ThemeOut(BaseModel):
    theme_name: str
    papers: list[PaperSummaryOut]


class WebSummaryOut(BaseModel):
    synthesis: str
    cited_articles: list[WebArticleOut]


class SummarizeResponse(BaseModel):
    search_id: int
    topic: str
    style: CitationStyle
    themes: list[ThemeOut]
    gaps_and_disagreements: str
    skipped_paper_ids: list[str]
    # Round-2 enhancement 5: its own block, never merged into the
    # paper-themes summary above. None when this search found no web
    # articles at all (nothing to summarize).
    web_summary: WebSummaryOut | None = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    search_id: int
    question: str
    history: list[ChatTurn] = []


class CitedPaperOut(BaseModel):
    paper_id: str
    title: str


class CitedWebArticleOut(BaseModel):
    url: str
    title: str


class ChatResponse(BaseModel):
    answer: str
    answerable: bool
    cited_papers: list[CitedPaperOut]
    # Round-2 enhancement 5: kept as its own list (tagged [Web N] in the
    # answer text) so the UI can render it distinguishably from cited_papers
    # ([Paper N]), not merged into one generic citation list.
    cited_web_articles: list[CitedWebArticleOut]
    history: list[ChatTurn]


class LibraryItem(BaseModel):
    search_id: int
    topic: str
    created_at: str
    paper_count: int
    has_summary: bool
    web_article_count: int


# ---- helpers ------------------------------------------------------------------

def _paper_to_out(paper: Paper, score: float | None = None) -> PaperOut:
    return PaperOut(
        paper_id=paper.paper_id, title=paper.title, authors=paper.authors,
        year=paper.year, venue=paper.venue, abstract=paper.abstract,
        url=paper.url, doi=paper.doi, citation_count=paper.citation_count,
        source=paper.source, source_urls=paper.source_urls, score=score,
    )


def _web_article_to_out(article: WebArticle) -> WebArticleOut:
    return WebArticleOut(
        title=article.title, url=article.url, snippet=article.snippet,
        published_date=article.published_date, source_domain=article.source_domain,
    )


def _web_articles_from_saved(saved) -> list[WebArticle]:
    return [WebArticle(**a) for a in saved.web_articles]


def _summary_to_json(result: dict, style: CitationStyle = "apa") -> dict:
    """Adapt summarize.generate_summary()'s return value (which embeds Paper
    objects) into a plain-JSON dict safe to store in SQLite and return over
    HTTP.

    Uses .get() defensively rather than direct key access for the round-2
    citation-style fields: this also runs against hand-built dicts in tests
    that mock generate_summary() and predate harvard_citation/citation, and
    must degrade to an APA-based default rather than KeyError on those.
    """
    themes_out = []
    for theme in result["themes"]:
        papers_out = []
        for entry in theme["papers"]:
            apa_citation = entry.get("apa_citation", "")
            harvard_citation = entry.get("harvard_citation") or apa_citation
            bibtex = entry.get("bibtex", "")
            citation = entry.get("citation") or select_citation(apa_citation, harvard_citation, bibtex, style)
            papers_out.append({
                "paper_id": entry["paper"].paper_id,
                "title": entry["paper"].title,
                "summary": entry["summary"],
                "apa_citation": apa_citation,
                "harvard_citation": harvard_citation,
                "bibtex": bibtex,
                "citation": citation,
            })
        themes_out.append({"theme_name": theme["theme_name"], "papers": papers_out})

    return {
        "themes": themes_out,
        "gaps_and_disagreements": result["gaps_and_disagreements"],
        "skipped_paper_ids": [p.paper_id for p in result["skipped_papers"]],
    }


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


def _web_summary_to_json(result: dict) -> dict:
    """Adapt summarize.generate_web_summary()'s return value (which embeds
    WebArticle objects) into a plain-JSON dict safe to store in SQLite and
    return over HTTP — same purpose as _summary_to_json above, kept
    separate since it has its own cache column (web_summary) and its own
    shape (no themes, just a synthesis + the cited subset)."""
    return {
        "synthesis": result["synthesis"],
        "cited_articles": [a.to_dict() for a in result["cited_articles"]],
    }


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


_STYLE_LABELS: dict[str, str] = {"apa": "APA", "harvard": "Harvard", "bibtex": "BibTeX"}


def _render_markdown(topic: str, summary_json: dict, style: CitationStyle = "apa", web_summary_json: dict | None = None) -> str:
    lines = [f"# Literature Summary: {topic}", ""]
    for theme in summary_json["themes"]:
        lines.append(f"## {theme['theme_name']}")
        lines.append("")
        for p in theme["papers"]:
            lines.append(f"- **{p['title']}**")
            lines.append(f"  {p['summary']}")
            lines.append("")

    lines.append("## Gaps and Disagreements")
    lines.append("")
    lines.append(summary_json["gaps_and_disagreements"])
    lines.append("")

    if web_summary_json is not None:
        # Its own section, positioned after the paper-themes summary but
        # clearly separate from it — never folded into the themes above.
        lines.append("## Web Context")
        lines.append("")
        lines.append(web_summary_json["synthesis"])
        lines.append("")
        for a in web_summary_json["cited_articles"]:
            lines.append(f"- [{a['title']}]({a['url']}) — {a['source_domain']}")
        lines.append("")

    if style == "bibtex":
        # BibTeX is already a structured export format, not prose — a
        # "References (BibTeX)" section duplicating the BibTeX block below
        # would just repeat it, so this is the one style that skips the
        # separate References section entirely.
        lines.append("## References (BibTeX)")
        lines.append("")
        lines.append("```bibtex")
        for theme in summary_json["themes"]:
            for p in theme["papers"]:
                lines.append(p.get("bibtex", ""))
                lines.append("")
        lines.append("```")
    else:
        citation_key = "harvard_citation" if style == "harvard" else "apa_citation"
        lines.append(f"## References ({_STYLE_LABELS.get(style, 'APA')})")
        lines.append("")
        for theme in summary_json["themes"]:
            for p in theme["papers"]:
                lines.append(f"- {p.get(citation_key) or p.get('apa_citation', '')}")
        lines.append("")

        lines.append("## BibTeX")
        lines.append("")
        lines.append("```bibtex")
        for theme in summary_json["themes"]:
            for p in theme["papers"]:
                lines.append(p.get("bibtex", ""))
                lines.append("")
        lines.append("```")

    return "\n".join(lines)


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


class CurationStartRequest(BaseModel):
    topic: str
    # 1-30: matches report.py's own documented 30-paper cap; target_count
    # is "how many picks the user wants total," not a per-batch size.
    target_count: int = Field(default=10, ge=1, le=30)
    use_openalex_fallback: bool = False


class CurationPicksRequest(BaseModel):
    picked_paper_ids: list[str] = []
    stop: bool = False
    # curation-refinement-and-auto-offer Phase 6f: optional free-text
    # steering (e.g. "focus on more recent work"), carried into the
    # SAME resume payload picked_paper_ids/stop already use -- see
    # resume_curation_turn's own docstring for why it can't be a
    # separate out-of-band call instead.
    refinement: str | None = None
    # curation-turn-history Phase 9d: explicit "search for more now"
    # request -- reuses the SAME force_refill mechanism refinement above
    # already triggers, not a second one. False (the default) preserves
    # every existing caller's exact behavior unchanged.
    request_refill: bool = False


class CurationTurnResponse(BaseModel):
    session_id: str
    stage: str
    target_count: int
    selected_paper_ids: list[str]
    batch: list[PaperOut] = []
    stop_reason: str | None = None
    # Phase 6c: whether THIS turn's batch came from a fresh search
    # (refill_pool ran) vs. serving from the already-fetched pool —
    # meaningless once stop_reason is set (batch is always [] then), where
    # it's always False rather than a stale carry-over value.
    refilled: bool = False
    # How many already-fetched, not-yet-served candidates remain in the
    # pool after this turn's batch — PaperPoolSession.remaining(), the
    # exact number that decides whether the NEXT turn needs a fresh
    # search at all. Surfaces the pool panel's "N more candidates
    # already fetched" status line without the frontend having to infer
    # it from anything.
    reserve_remaining: int = 0
    # Phase 6f: every refinement note applied so far this session, so the
    # UI can show what's currently steering the search.
    refinement_notes: list[str] = []


class ReportSectionOut(BaseModel):
    content: str
    cited_papers: list[CitedPaperOut]
    cited_web_articles: list[CitedWebArticleOut] = []


class ReportOut(BaseModel):
    findings: ReportSectionOut
    limitations: ReportSectionOut
    future_scope: ReportSectionOut
    skipped_paper_ids: list[str]


class TurnHistoryEntryOut(BaseModel):
    turn_number: int
    batch: list[PaperOut]
    refilled: bool


class CurationStateResponse(BaseModel):
    session_id: str
    topic: str
    # curation-review-management Phase 8, item 5: canonicalize_topic()'s
    # restatement of `topic`, for display only -- `topic` above is
    # unchanged and still what actually drives search/ranking/refinement.
    display_title: str
    stage: str
    target_count: int
    selected_paper_ids: list[str]
    selected_papers: list[PaperOut]
    # Only non-None mid-curation, with a genuinely pending interrupt — the
    # exact property a page refresh during curation needs to recover from
    # the backend, not from anything held only in browser memory (Phase 6d).
    pending_batch: list[PaperOut] | None = None
    refilled: bool = False
    reserve_remaining: int = 0
    refinement_notes: list[str] = []
    report: ReportOut | None = None
    chat_history: list[ChatTurn] = []
    web_articles_added: list[WebArticleOut] = []
    pending_web_offer: dict | None = None
    pending_report_update: dict | None = None
    # curation-turn-history Phase 9b: every batch ever served, in order --
    # lets a client redraw ANY past turn's cards/abstracts, not just the
    # currently-pending one. Unbounded (see PaperPoolSession.turn_history).
    turn_history: list[TurnHistoryEntryOut] = []
    # Persisted so a reload/reopen can still show WHY curation stopped
    # (target_met / user_stopped / exhausted) -- None while stage=="curate".
    stop_reason: str | None = None


class CurationChatRequest(BaseModel):
    message: str


class CurationChatResponse(BaseModel):
    answer: str
    answerable: bool
    cited_papers: list[CitedPaperOut]
    cited_web_articles: list[CitedWebArticleOut]
    web_offer_made: bool = False
    web_offer_declined: bool = False
    web_search_used: bool = False
    new_web_articles_found: int | None = None
    # curation-refinement-and-auto-offer Phase 6f-3
    report_update_offer_made: bool = False
    report_update_declined: bool = False
    report_updated: bool = False
    chat_history: list[ChatTurn]


def _paper_out_from_batch_entry(entry) -> PaperOut:
    paper_dict, score = entry
    return _paper_to_out(Paper(**paper_dict), score)


def _turn_history_out(turn_history: list[dict]) -> list[TurnHistoryEntryOut]:
    return [
        TurnHistoryEntryOut(
            turn_number=entry["turn_number"],
            batch=[_paper_out_from_batch_entry(e) for e in entry["batch"]],
            refilled=entry["refilled"],
        )
        for entry in turn_history
    ]


def _report_to_out(report: dict) -> ReportOut:
    def _section(name: str) -> ReportSectionOut:
        s = report[name]
        return ReportSectionOut(
            content=s["content"],
            cited_papers=[CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in s["cited_papers"]],
            cited_web_articles=[CitedWebArticleOut(url=a.url, title=a.title) for a in s.get("cited_web_articles", [])],
        )

    return ReportOut(
        findings=_section("findings"), limitations=_section("limitations"), future_scope=_section("future_scope"),
        skipped_paper_ids=[p.paper_id for p in report["skipped_papers"]],
    )


def _turn_result_to_response(session_id: str, target_count: int, result: dict) -> CurationTurnResponse:
    session_dict = result["session"]
    batch = result["__interrupt__"][0].value["batch"] if "__interrupt__" in result else []
    return CurationTurnResponse(
        session_id=session_id, stage=session_dict["stage"], target_count=target_count,
        selected_paper_ids=session_dict["selected_paper_ids"],
        batch=[_paper_out_from_batch_entry(e) for e in batch],
        stop_reason=result.get("stop_reason"),
        refilled=result.get("refilled", False),
        # Computed straight off the raw serialized dict (reserve/cursor),
        # not via _dict_to_session -- no need to reconstruct every
        # reserve Paper's full data just to subtract two lengths.
        reserve_remaining=max(0, len(session_dict["reserve"]) - session_dict["cursor"]),
        refinement_notes=list(session_dict.get("refinement_notes", [])),
    )


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


@app.post("/curation/start", response_model=CurationTurnResponse)
def curation_start(req: CurationStartRequest, cp=Depends(get_curation_checkpointer)) -> CurationTurnResponse:
    with _upstream_error_guard("curation_start"):
        client = _state["client"]
        s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
        deduped = build_candidate_pool(
            req.topic, req.target_count, s2_api_key=s2_key, client=client,
            use_openalex_fallback=req.use_openalex_fallback, openalex_mailto=os.getenv("OPENALEX_MAILTO") or None,
        )
        ranked, _ = rank_full_pool(req.topic, deduped, client=client, collection=_state["collection"])
        if not ranked:
            raise HTTPException(status_code=404, detail="No papers found for this topic.")

        # curation-review-management Phase 8, item 5: after confirming
        # papers actually exist for this topic (no point spending an LLM
        # call otherwise), produce a clean display title. Never touches
        # req.topic itself -- that's still what search/ranking/refinement
        # keep using for the rest of this session's life.
        display_title = canonicalize_topic(req.topic, client=client)

        session_id = uuid.uuid4().hex
        session = PaperPoolSession(topic=req.topic, display_title=display_title, reserve=ranked, target_count=req.target_count)
        result = start_curation_turn(session_id, cp, _session_to_dict(session), config=_curation_config())
        return _turn_result_to_response(session_id, req.target_count, result)


@app.post("/curation/{session_id}/picks", response_model=CurationTurnResponse)
def curation_picks(session_id: str, req: CurationPicksRequest, cp=Depends(get_curation_checkpointer)) -> CurationTurnResponse:
    with _upstream_error_guard("curation_picks"):
        state = get_curation_state(session_id, cp)
        if state is None:
            raise HTTPException(status_code=404, detail="session_id not found")
        if state["pending_batch"] is None:
            raise HTTPException(status_code=400, detail="Session is not awaiting picks (curation already finished).")

        target_count = state["session"].target_count
        result = resume_curation_turn(
            session_id, cp, picked_paper_ids=req.picked_paper_ids, stop=req.stop,
            refinement=req.refinement, request_refill=req.request_refill, config=_curation_config(),
        )
        return _turn_result_to_response(session_id, target_count, result)


class CurationReviewSummary(BaseModel):
    session_id: str
    topic: str
    # curation-review-management Phase 8, item 5: same display-only
    # restatement as CurationStateResponse.display_title -- this is what
    # the left panel's review list should actually show as each review's
    # name.
    display_title: str
    stage: str
    selected_count: int
    target_count: int
    has_report: bool
    has_chat: bool


class CurationDeleteResponse(BaseModel):
    session_id: str
    deleted: bool = True


from research_agent.api_app.routers.curation_sessions import router as curation_sessions_router

app.include_router(curation_sessions_router)


class CurationSelectFromHistoryRequest(BaseModel):
    paper_id: str


class CurationSelectFromHistoryResponse(BaseModel):
    session_id: str
    selected_paper_ids: list[str]


@app.post("/curation/{session_id}/select-from-history", response_model=CurationSelectFromHistoryResponse)
def curation_select_from_history(
    session_id: str, req: CurationSelectFromHistoryRequest, cp=Depends(get_curation_checkpointer),
) -> CurationSelectFromHistoryResponse:
    """curation-turn-history Phase 9c: retroactively add a paper seen in
    an earlier turn, without a new search -- lets a user unstuck from a
    curation that ended short of their target (e.g. the pool genuinely
    ran dry) by picking from what they already saw, even from Report/Chat
    mode. SYNTHESIZE-STAGE ONLY -- see select_paper_from_history()'s own
    docstring for exactly why (out-of-band writes while a real interrupt
    is pending corrupt its bookkeeping, the same failure mode
    curation-refinement-and-auto-offer Phase 6f already found once).
    Picking from history while still curating goes through
    /curation/{session_id}/picks instead (Phase 9d)."""
    session = load_curation_session(session_id, cp)
    if session is None:
        raise HTTPException(status_code=404, detail="session_id not found")
    try:
        select_paper_from_history(session, req.paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    save_curation_session(session, session_id, cp)
    return CurationSelectFromHistoryResponse(session_id=session_id, selected_paper_ids=session.selected_paper_ids)


@app.post("/curation/{session_id}/reopen", response_model=CurationTurnResponse)
def curation_reopen(session_id: str, cp=Depends(get_curation_checkpointer)) -> CurationTurnResponse:
    """curation-editable-until-locked Phase 10c: reopens a review that WAS
    explicitly stopped (stage=="synthesize") back into active curation --
    the counterpart to Phase 10b's routing change, which made stopping
    the ONLY thing that ever locks a review. Gated on reopen_curation_
    session()'s own rules (still curating / report already generated /
    chat already started, in that order) -- see its docstring for why. On
    success, re-invokes start_curation_turn() fresh on the same
    thread_id: not a Command(resume=...) (there's no pending interrupt
    to resume once a real stop has already run through END), just a
    plain graph.invoke() that picks up from wherever cursor/
    selected_paper_ids left off, verified empirically before this was
    implemented."""
    with _upstream_error_guard("curation_reopen"):
        session = load_curation_session(session_id, cp)
        if session is None:
            raise HTTPException(status_code=404, detail="session_id not found")
        try:
            reopen_curation_session(session)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        target_count = session.target_count
        result = start_curation_turn(session_id, cp, _session_to_dict(session), config=_curation_config())
        return _turn_result_to_response(session_id, target_count, result)


@app.post("/curation/{session_id}/report", response_model=ReportOut)
def curation_report(session_id: str, cp=Depends(get_curation_checkpointer)) -> ReportOut:
    """Generate-or-get, same cache-then-generate convention /summarize
    already uses for its own report-like artifact — a second call for the
    same session_id doesn't re-bill the LLM."""
    with _upstream_error_guard("curation_report"):
        session = load_curation_session(session_id, cp)
        if session is None:
            raise HTTPException(status_code=404, detail="session_id not found")
        if session.report is None:
            try:
                session.report = generate_report_for_session(session, client=_state["client"])
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            # curation-refinement-and-auto-offer Phase 6f-3: keeps the
            # auto-offer's staleness check accurate even when a report is
            # generated straight through this endpoint rather than via
            # chat's accept-web-offer path.
            session.report_covered_web_article_count = len(session.web_articles_added)
            save_curation_session(session, session_id, cp)
        return _report_to_out(session.report)


@app.post("/curation/{session_id}/report/regenerate", response_model=ReportOut)
def curation_report_regenerate(session_id: str, cp=Depends(get_curation_checkpointer)) -> ReportOut:
    with _upstream_error_guard("curation_report_regenerate"):
        session = load_curation_session(session_id, cp)
        if session is None:
            raise HTTPException(status_code=404, detail="session_id not found")
        try:
            session.report = regenerate_report_with_new_sources(session, client=_state["client"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session.report_covered_web_article_count = len(session.web_articles_added)
        save_curation_session(session, session_id, cp)
        return _report_to_out(session.report)


@app.post("/curation/{session_id}/chat", response_model=CurationChatResponse)
def curation_chat_turn(session_id: str, req: CurationChatRequest, cp=Depends(get_curation_checkpointer)) -> CurationChatResponse:
    with _upstream_error_guard("curation_chat"):
        session = load_curation_session(session_id, cp)
        if session is None:
            raise HTTPException(status_code=404, detail="session_id not found")
        try:
            result = chat_turn(session, req.message, client=_state["client"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        save_curation_session(session, session_id, cp)

        return CurationChatResponse(
            answer=result["answer"], answerable=result["answerable"],
            cited_papers=[CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in result["cited_papers"]],
            cited_web_articles=[CitedWebArticleOut(url=a.url, title=a.title) for a in result["cited_web_articles"]],
            web_offer_made=result.get("web_offer_made", False),
            web_offer_declined=result.get("web_offer_declined", False),
            web_search_used=result.get("web_search_used", False),
            new_web_articles_found=result.get("new_web_articles_found"),
            report_update_offer_made=result.get("report_update_offer_made", False),
            report_update_declined=result.get("report_update_declined", False),
            report_updated=result.get("report_updated", False),
            chat_history=[ChatTurn(**turn) for turn in session.chat_history],
        )
