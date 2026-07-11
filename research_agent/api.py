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

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from research_agent.agent import run_research_agent
from research_agent.citations import CitationStyle, select_citation
from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, get_papers_by_ids, semantic_search
from research_agent.enrichment import enrich_missing_abstracts
from research_agent.qa import ChatSession, ask
from research_agent.schema import Paper, WebArticle
from research_agent.storage import get_search, init_db, list_searches, save_search, update_summary, update_web_summary
from research_agent.summarize import generate_summary, generate_web_summary

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
    # Same code-enforced-count guarantee as top_k: the agent may have
    # accumulated more than web_max_results across multiple search_web_tool
    # calls (deduped by URL, not by count) — truncate here so the returned
    # count is never silently uncontrolled, same reasoning as top_k in
    # enhancement 1.
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
