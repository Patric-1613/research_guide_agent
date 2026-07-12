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
from contextlib import asynccontextmanager
from dataclasses import asdict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from research_agent.agent import _merge_web_articles, run_research_agent
from research_agent.citations import CitationStyle, select_citation
from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, get_papers_by_ids, semantic_search
from research_agent.enrichment import enrich_missing_abstracts
from research_agent.ingestion import search_arxiv, search_semantic_scholar
from research_agent.qa import ChatSession, ask
from research_agent.schema import Paper, WebArticle
from research_agent.session import TriageSession, add_round
from research_agent.storage import (
    delete_bag,
    get_bag,
    get_search,
    init_db,
    list_bags,
    list_searches,
    paper_ids_referenced_by_other_bags,
    save_bag,
    save_search,
    update_summary,
    update_web_summary,
)
from research_agent.summarize import generate_summary, generate_web_summary
from research_agent.web_search import search_web

load_dotenv()

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["db"] = init_db()
    _state["client"] = OpenAI()
    _state["collection"] = get_chroma_collection()
    yield
    _state["db"].close()


app = FastAPI(title="Research Paper Summarizer API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Cheap connectivity check for the phase-8 Streamlit frontend — no DB
    or LLM calls, just confirms the process is up."""
    return {"status": "ok"}


# ---- request/response models -------------------------------------------------

class SearchRequest(BaseModel):
    topic: str
    # 3-30 mirrors the Streamlit number input's bounds (app.py) — kept here
    # too so a direct API call gets the same guarantee, not just the UI.
    top_k: int = Field(default=10, ge=3, le=30)
    # Round-2 enhancement 2: surfaces doi/citation_count metadata that's
    # already stored per-paper in Chroma (embeddings.py) — no re-indexing
    # needed. min_citation_count=0 means "no filter" per the brief.
    doi_required: bool = False
    min_citation_count: int = Field(default=0, ge=0)
    # Round-2 enhancement 5: independent of top_k — web articles are a
    # separate, smaller pool, never counted alongside the paper results.
    web_max_results: int = Field(default=4, ge=1, le=10)


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
    role: str
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


class RoundSearchRequest(BaseModel):
    """Round 3, phase 2: one round of the interactive multi-round triage
    flow. Deliberately stateless like every other endpoint here — the
    caller (Streamlit's session_state) sends back the whole session it
    got from the previous round-search call (or omits it on the very
    first round of a new session) and receives the updated version.
    Nothing here touches SQLite or Chroma; see session.py's docstring.
    """

    topic: str
    keyword: str = Field(..., min_length=1)
    session_state: dict | None = None
    include_web: bool = True
    max_results_per_source: int = Field(default=10, ge=3, le=30)
    web_max_results: int = Field(default=4, ge=1, le=10)


class RoundOut(BaseModel):
    round_number: int
    keywords_used: list[str]
    timestamp: str
    paper_ids_found: list[str]
    new_paper_ids: list[str]
    web_urls_found: list[str]
    new_web_urls: list[str]


class RoundSearchResponse(BaseModel):
    # Opaque blob the frontend stores as-is and resends verbatim on the
    # next /round_search call — see TriageSession.to_dict()/from_dict().
    session_state: dict
    round: RoundOut


class TriageSummarizeRequest(BaseModel):
    """Round 3, phase 3: the interactive-triage equivalent of /summarize.
    Deliberately takes the whole session_state blob (not just a list of
    ids) so the basket's paper_ids/urls can be resolved back to full
    Paper/WebArticle records without a second round trip — the frontend
    already has them all in session_state["all_papers"]/["all_web_articles"]
    from the last /round_search response.
    """

    session_state: dict
    style: CitationStyle = "apa"


class TriageSummarizeResponse(BaseModel):
    topic: str
    style: CitationStyle
    themes: list[ThemeOut]
    gaps_and_disagreements: str
    skipped_paper_ids: list[str]
    web_summary: WebSummaryOut | None = None
    basket_paper_count: int
    basket_web_article_count: int
    # Exposes embeddings.py's own cache_hits/cache_misses/tokens_billed/
    # estimated_cost_usd straight through — concrete, checkable evidence
    # (both for tests and for the UI itself) that embedding only ever
    # covers the basket, never the full accumulated search pool.
    embed_stats: dict


class TriageDiscardRequest(BaseModel):
    """Round 3, phase 4: the basket's paper_ids that /triage/summarize just
    embedded, sent back so their Chroma vectors can be removed if the user
    chooses not to keep this session as a bag. Never touches SQLite —
    nothing was persisted there in the first place."""

    paper_ids: list[str]


class TriageSaveBagRequest(BaseModel):
    name: str = Field(..., min_length=1)
    session_state: dict
    # The exact dict /triage/summarize returned (themes, gaps_and_disagreements,
    # skipped_paper_ids, web_summary, ...) — saved as-is rather than
    # regenerated, since regenerating would re-bill the LLM for no reason.
    summary: dict


class BagOut(BaseModel):
    bag_id: int
    name: str
    topic: str
    created_at: str
    paper_count: int
    web_article_count: int
    # Union of every round's keywords_used and the creation year — the
    # phase-4 brief's "group/filter by name, keyword, or year" needs
    # something to group/filter on; name is already `name`/`topic`.
    keywords: list[str]
    year: int


class BagDetailResponse(BaseModel):
    bag_id: int
    name: str
    topic: str
    created_at: str
    papers: list[PaperOut]
    web_articles: list[WebArticleOut]
    rounds: list[RoundOut]
    themes: list[ThemeOut]
    gaps_and_disagreements: str
    skipped_paper_ids: list[str]
    web_summary: WebSummaryOut | None = None


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


def _get_or_create_summary(search_id: int, saved, style: CitationStyle = "apa") -> dict:
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
    update_summary(_state["db"], search_id, summary_json)
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


def _get_or_create_web_summary(search_id: int, saved) -> dict | None:
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
    update_web_summary(_state["db"], search_id, web_summary_json)
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


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    session = run_research_agent(
        req.topic, s2_api_key=s2_key, top_k=req.top_k,
        doi_required=req.doi_required, min_citation_count=req.min_citation_count,
        web_max_results=req.web_max_results,
    )

    ranked = session.ranked
    expected_count = min(req.top_k, _filtered_candidate_count(session.papers, req.doi_required, req.min_citation_count))
    if session.papers and len(ranked) != expected_count:
        # The agent is prompted to rerank with exactly top_k results before
        # finishing, and its rerank tool applies the doi/citation filters
        # itself — but that's still a prompted/tool-execution behavior, not
        # a guarantee (the model might skip reranking entirely). Re-rank
        # server-side whenever what came back doesn't match what the user
        # asked for, so the returned count and filters are always
        # code-enforced rather than dependent on the model's behavior.
        ranked = _server_side_rerank(session, req.topic, req.top_k, req.doi_required, req.min_citation_count)

    if not ranked:
        if session.papers and (req.doi_required or req.min_citation_count):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Found {len(session.papers)} paper(s) for this topic, but none matched the "
                    f"active filters (DOI required: {req.doi_required}, "
                    f"min citations: {req.min_citation_count}). Try relaxing the filters."
                ),
            )
        raise HTTPException(status_code=404, detail="No papers found for this topic.")

    paper_ids = [p.paper_id for p, _ in ranked]
    scores = [score for _, score in ranked]
    if len(session.web_articles) < req.web_max_results:
        # Whether to call search_web_tool at all is the agent's judgment
        # call (agent.py's system prompt) — it may skip it entirely for a
        # topic it judges purely historical/theoretical, or just not call it
        # this run. But web_max_results is a user-set request parameter like
        # top_k, so the user gets that many results whenever they're
        # available, not only when the model happened to decide to look.
        # Same code-enforced-count guarantee as top_k's server-side rerank
        # fallback above.
        fallback_articles = search_web(req.topic, max_results=req.web_max_results)
        session.web_articles = _merge_web_articles(session.web_articles, fallback_articles)
    # The agent may have accumulated more than web_max_results across
    # multiple search_web_tool calls (deduped by URL, not by count) —
    # truncate here so the returned count is never silently uncontrolled,
    # same reasoning as top_k in enhancement 1.
    web_articles = session.web_articles[: req.web_max_results]
    search_id, created_at = save_search(
        _state["db"], req.topic, paper_ids, scores,
        web_articles=[a.to_dict() for a in web_articles],
    )

    return SearchResponse(
        search_id=search_id, topic=req.topic, created_at=created_at,
        papers=[_paper_to_out(p, score) for p, score in ranked],
        web_articles=[_web_article_to_out(a) for a in web_articles],
    )


@app.post("/round_search", response_model=RoundSearchResponse)
def round_search(req: RoundSearchRequest) -> RoundSearchResponse:
    """Round 3, phase 2: run one round of the interactive triage flow.

    Deliberately code-driven, not the phase-4 LLM agent: with ranking
    deferred until Summarize (phase 3), a round has nothing left for the
    agent's mandatory rerank-by-relevance tool call to do, so routing
    through run_research_agent would just be an LLM call that changes
    nothing but cost. This searches both arXiv and Semantic Scholar
    directly (matching the old flow's "search both by default" behavior),
    optionally web, and merges into the session's accumulated pool via
    session.add_round (which itself reuses dedup.deduplicate unmodified).

    Never touches SQLite or Chroma — nothing is persisted until the user
    saves a bag (phase 4) or the basket is embedded at Summarize (phase 3).
    """
    session = TriageSession.from_dict(req.session_state) if req.session_state else TriageSession()
    if not session.topic:
        session.topic = req.topic

    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    arxiv_papers = search_arxiv(req.keyword, max_results=req.max_results_per_source)
    s2_papers = search_semantic_scholar(req.keyword, max_results=req.max_results_per_source, api_key=s2_key)
    found_web_articles = search_web(req.keyword, max_results=req.web_max_results) if req.include_web else []

    round_ = add_round(session, [req.keyword], arxiv_papers + s2_papers, found_web_articles)

    return RoundSearchResponse(session_state=session.to_dict(), round=RoundOut(**asdict(round_)))


def _papers_from_session_dict(session_state: dict, paper_ids: list[str]) -> list[Paper]:
    all_papers = session_state.get("all_papers", {})
    return [Paper(**all_papers[pid]) for pid in paper_ids if pid in all_papers]


def _web_articles_from_session_dict(session_state: dict, urls: list[str]) -> list[WebArticle]:
    all_web_articles = session_state.get("all_web_articles", {})
    return [WebArticle(**all_web_articles[url]) for url in urls if url in all_web_articles]


_EMPTY_EMBED_STATS = {"cache_hits": 0, "cache_misses": 0, "tokens_billed": 0, "estimated_cost_usd": 0.0}


@app.post("/triage/summarize", response_model=TriageSummarizeResponse)
def triage_summarize(req: TriageSummarizeRequest) -> TriageSummarizeResponse:
    """Round 3, phase 3: the actual cost-saving change. Every /round_search
    call above only ever gathers raw Paper/WebArticle records in memory —
    it never calls enrich_missing_abstracts or embed_and_index_papers, so
    nothing is embedded or written to Chroma while the user is still
    browsing across however many rounds. Those calls only happen here, and
    only for whatever is in the basket at the moment Summarize is clicked —
    never session_state's full accumulated pool, which after several
    rounds is typically much larger than the basket.

    Not yet persisted to SQLite/Chroma-as-a-bag — phase 4 adds the actual
    named-bag save/discard step on top of the embeddings this call writes.
    """
    session_state = req.session_state
    basket_paper_ids = session_state.get("basket_paper_ids", [])
    basket_web_urls = session_state.get("basket_web_urls", [])
    if not basket_paper_ids and not basket_web_urls:
        raise HTTPException(
            status_code=400,
            detail="Your basket is empty — add at least one paper or web article before summarizing.",
        )

    basket_papers = _papers_from_session_dict(session_state, basket_paper_ids)
    basket_web_articles = _web_articles_from_session_dict(session_state, basket_web_urls)
    topic = session_state.get("topic", "")

    embed_stats = _EMPTY_EMBED_STATS
    if basket_papers:
        enrich_missing_abstracts(basket_papers)
        embed_stats = embed_and_index_papers(basket_papers, collection=_state["collection"], client=_state["client"])

    result = generate_summary(topic, basket_papers, client=_state["client"], style=req.style)
    summary_json = _summary_to_json(result, style=req.style)

    web_summary_json = None
    if basket_web_articles:
        web_result = generate_web_summary(topic, basket_web_articles, client=_state["client"])
        web_summary_json = _web_summary_to_json(web_result)

    return TriageSummarizeResponse(
        topic=topic,
        style=req.style,
        web_summary=WebSummaryOut(**web_summary_json) if web_summary_json is not None else None,
        basket_paper_count=len(basket_papers),
        basket_web_article_count=len(basket_web_articles),
        embed_stats=embed_stats,
        **summary_json,
    )


def _delete_chroma_vectors_unless_shared(paper_ids: list[str], exclude_bag_id: int | None = None) -> list[str]:
    """Deletes each of `paper_ids`' Chroma vectors, except any still
    referenced by some other saved bag (storage.paper_ids_referenced_by_other_bags) —
    used by both /triage/discard (no bag ever existed) and DELETE /bags/{id}
    (a bag existed and is being removed). Returns the ids actually deleted,
    so callers/tests can confirm scope precisely."""
    still_referenced = paper_ids_referenced_by_other_bags(_state["db"], paper_ids, exclude_bag_id=exclude_bag_id)
    ids_to_delete = [pid for pid in paper_ids if pid not in still_referenced]
    if ids_to_delete:
        _state["collection"].delete(ids=ids_to_delete)
    return ids_to_delete


@app.post("/triage/discard")
def triage_discard(req: TriageDiscardRequest) -> dict:
    """Round 3, phase 4: 'discard everything generated this session' — the
    basket was embedded into Chroma at Summarize time (phase 3) but no bag
    was ever saved, so there's no SQLite row to remove, only those Chroma
    vectors (and only the ones no other saved bag still needs)."""
    removed = _delete_chroma_vectors_unless_shared(req.paper_ids)
    return {"discarded": True, "chroma_ids_removed": removed}


def _bag_keywords(rounds: list[dict]) -> list[str]:
    return sorted({kw for r in rounds for kw in r.get("keywords_used", [])})


@app.post("/triage/save_bag", response_model=BagOut)
def triage_save_bag(req: TriageSaveBagRequest) -> BagOut:
    """Round 3, phase 4: persist the basket, its round history, and the
    already-generated summaries as a named bag. The embeddings themselves
    were already written to Chroma by /triage/summarize (phase 3) — this
    only adds the SQLite record; nothing gets re-embedded."""
    session_state = req.session_state
    basket_paper_ids = session_state.get("basket_paper_ids", [])
    basket_web_urls = session_state.get("basket_web_urls", [])
    if not basket_paper_ids and not basket_web_urls:
        raise HTTPException(status_code=400, detail="Nothing to save — the basket is empty.")

    all_web_articles = session_state.get("all_web_articles", {})
    web_articles = [all_web_articles[u] for u in basket_web_urls if u in all_web_articles]
    rounds = session_state.get("rounds", [])
    summary_json = {
        "themes": req.summary.get("themes", []),
        "gaps_and_disagreements": req.summary.get("gaps_and_disagreements", ""),
        "skipped_paper_ids": req.summary.get("skipped_paper_ids", []),
    }
    web_summary_json = req.summary.get("web_summary")
    topic = session_state.get("topic", "")

    bag_id, created_at = save_bag(
        _state["db"], req.name, topic, basket_paper_ids, web_articles, rounds, summary_json, web_summary_json,
    )
    return BagOut(
        bag_id=bag_id, name=req.name, topic=topic, created_at=created_at,
        paper_count=len(basket_paper_ids), web_article_count=len(web_articles),
        keywords=_bag_keywords(rounds), year=int(created_at[:4]),
    )


@app.get("/bags", response_model=list[BagOut])
def list_bags_endpoint() -> list[BagOut]:
    return [
        BagOut(
            bag_id=b.id, name=b.name, topic=b.topic, created_at=b.created_at,
            paper_count=len(b.paper_ids), web_article_count=len(b.web_articles),
            keywords=_bag_keywords(b.rounds), year=int(b.created_at[:4]),
        )
        for b in list_bags(_state["db"])
    ]


@app.get("/bags/{bag_id}", response_model=BagDetailResponse)
def get_bag_detail(bag_id: int) -> BagDetailResponse:
    bag = get_bag(_state["db"], bag_id)
    if bag is None:
        raise HTTPException(status_code=404, detail="bag_id not found")

    papers = get_papers_by_ids(bag.paper_ids, collection=_state["collection"])
    return BagDetailResponse(
        bag_id=bag.id, name=bag.name, topic=bag.topic, created_at=bag.created_at,
        papers=[_paper_to_out(p) for p in papers],
        web_articles=[WebArticleOut(**a) for a in bag.web_articles],
        rounds=[RoundOut(**r) for r in bag.rounds],
        themes=bag.summary.get("themes", []),
        gaps_and_disagreements=bag.summary.get("gaps_and_disagreements", ""),
        skipped_paper_ids=bag.summary.get("skipped_paper_ids", []),
        web_summary=WebSummaryOut(**bag.web_summary) if bag.web_summary is not None else None,
    )


@app.delete("/bags/{bag_id}")
def delete_bag_endpoint(bag_id: int) -> dict:
    """Round 3, phase 4: deleting a bag removes BOTH the SQLite record AND
    its Chroma vectors (unless another saved bag still references the same
    paper — see paper_ids_referenced_by_other_bags) — never just one of the
    two, so no orphaned vectors are left behind in Chroma."""
    bag = get_bag(_state["db"], bag_id)
    if bag is None:
        raise HTTPException(status_code=404, detail="bag_id not found")

    removed = _delete_chroma_vectors_unless_shared(bag.paper_ids, exclude_bag_id=bag_id)
    delete_bag(_state["db"], bag_id)
    return {"deleted": True, "bag_id": bag_id, "chroma_ids_removed": removed}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    saved = get_search(_state["db"], req.search_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="search_id not found")

    summary_json = _get_or_create_summary(req.search_id, saved, style=req.style)
    web_summary_json = _get_or_create_web_summary(req.search_id, saved)
    web_summary_out = WebSummaryOut(**web_summary_json) if web_summary_json is not None else None
    return SummarizeResponse(
        search_id=req.search_id, topic=saved.topic, style=req.style, web_summary=web_summary_out, **summary_json,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    saved = get_search(_state["db"], req.search_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="search_id not found")

    papers = get_papers_by_ids(saved.paper_ids, collection=_state["collection"])
    web_articles = _web_articles_from_saved(saved)
    session = ChatSession(papers=papers, web_articles=web_articles, history=[turn.model_dump() for turn in req.history])
    result = ask(session, req.question, client=_state["client"])

    return ChatResponse(
        answer=result["answer"],
        answerable=result["answerable"],
        cited_papers=[CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in result["cited_papers"]],
        cited_web_articles=[CitedWebArticleOut(url=a.url, title=a.title) for a in result.get("cited_web_articles", [])],
        history=[ChatTurn(**turn) for turn in session.history],
    )


@app.get("/export/{search_id}", response_class=PlainTextResponse)
def export(search_id: int, style: CitationStyle = "apa") -> str:
    saved = get_search(_state["db"], search_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="search_id not found")

    summary_json = _get_or_create_summary(search_id, saved, style=style)
    web_summary_json = _get_or_create_web_summary(search_id, saved)
    return _render_markdown(saved.topic, summary_json, style=style, web_summary_json=web_summary_json)


@app.get("/library", response_model=list[LibraryItem])
def library() -> list[LibraryItem]:
    saved_list = list_searches(_state["db"])
    return [
        LibraryItem(
            search_id=s.id, topic=s.topic, created_at=s.created_at,
            paper_count=len(s.paper_ids), has_summary=s.summary is not None,
            web_article_count=len(s.web_articles),
        )
        for s in saved_list
    ]


@app.get("/library/{search_id}", response_model=SearchResponse)
def library_detail(search_id: int) -> SearchResponse:
    saved = get_search(_state["db"], search_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="search_id not found")

    papers = get_papers_by_ids(saved.paper_ids, collection=_state["collection"])
    scores_by_id = dict(zip(saved.paper_ids, saved.scores))
    return SearchResponse(
        search_id=saved.id, topic=saved.topic, created_at=saved.created_at,
        papers=[_paper_to_out(p, scores_by_id.get(p.paper_id)) for p in papers],
        web_articles=[_web_article_to_out(a) for a in _web_articles_from_saved(saved)],
    )
