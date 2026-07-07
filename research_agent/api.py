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
from pydantic import BaseModel

from research_agent.agent import run_research_agent
from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, get_papers_by_ids, semantic_search
from research_agent.qa import ChatSession, ask
from research_agent.schema import Paper
from research_agent.storage import get_search, init_db, list_searches, save_search, update_summary
from research_agent.summarize import generate_summary

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


class SearchResponse(BaseModel):
    search_id: int
    topic: str
    created_at: str
    papers: list[PaperOut]


class SummarizeRequest(BaseModel):
    search_id: int


class PaperSummaryOut(BaseModel):
    paper_id: str
    title: str
    summary: str
    apa_citation: str
    bibtex: str


class ThemeOut(BaseModel):
    theme_name: str
    papers: list[PaperSummaryOut]


class SummarizeResponse(BaseModel):
    search_id: int
    topic: str
    themes: list[ThemeOut]
    gaps_and_disagreements: str
    skipped_paper_ids: list[str]


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


class ChatResponse(BaseModel):
    answer: str
    answerable: bool
    cited_papers: list[CitedPaperOut]
    history: list[ChatTurn]


class LibraryItem(BaseModel):
    search_id: int
    topic: str
    created_at: str
    paper_count: int
    has_summary: bool


# ---- helpers ------------------------------------------------------------------

def _paper_to_out(paper: Paper, score: float | None = None) -> PaperOut:
    return PaperOut(
        paper_id=paper.paper_id, title=paper.title, authors=paper.authors,
        year=paper.year, venue=paper.venue, abstract=paper.abstract,
        url=paper.url, doi=paper.doi, citation_count=paper.citation_count,
        source=paper.source, source_urls=paper.source_urls, score=score,
    )


def _summary_to_json(result: dict) -> dict:
    """Adapt summarize.generate_summary()'s return value (which embeds Paper
    objects) into a plain-JSON dict safe to store in SQLite and return over
    HTTP."""
    return {
        "themes": [
            {
                "theme_name": theme["theme_name"],
                "papers": [
                    {
                        "paper_id": entry["paper"].paper_id,
                        "title": entry["paper"].title,
                        "summary": entry["summary"],
                        "apa_citation": entry["apa_citation"],
                        "bibtex": entry["bibtex"],
                    }
                    for entry in theme["papers"]
                ],
            }
            for theme in result["themes"]
        ],
        "gaps_and_disagreements": result["gaps_and_disagreements"],
        "skipped_paper_ids": [p.paper_id for p in result["skipped_papers"]],
    }


def _get_or_create_summary(search_id: int, saved) -> dict:
    """Reuse a previously-generated summary if one exists for this
    search_id, rather than re-billing the LLM every time /summarize or
    /export is called for the same search — mirrors the embedding cache's
    cost-consciousness from phase 3."""
    if saved.summary is not None:
        return saved.summary
    papers = get_papers_by_ids(saved.paper_ids, collection=_state["collection"])
    result = generate_summary(saved.topic, papers, client=_state["client"])
    summary_json = _summary_to_json(result)
    update_summary(_state["db"], search_id, summary_json)
    return summary_json


def _render_markdown(topic: str, summary_json: dict) -> str:
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

    lines.append("## References (APA)")
    lines.append("")
    for theme in summary_json["themes"]:
        for p in theme["papers"]:
            lines.append(f"- {p['apa_citation']}")
    lines.append("")

    lines.append("## BibTeX")
    lines.append("")
    lines.append("```bibtex")
    for theme in summary_json["themes"]:
        for p in theme["papers"]:
            lines.append(p["bibtex"])
            lines.append("")
    lines.append("```")

    return "\n".join(lines)


# ---- endpoints ------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    session = run_research_agent(req.topic, s2_api_key=s2_key)

    ranked = session.ranked
    if not ranked and session.papers:
        # Defensive fallback: the agent is instructed to always rerank
        # before finishing, but that's a prompted behavior, not a guarantee.
        collection = _state["collection"]
        client = _state["client"]
        embed_and_index_papers(session.papers, collection=collection, client=client)
        ids = [p.paper_id for p in session.papers]
        ranked = semantic_search(
            req.topic, collection=collection, client=client,
            top_k=len(session.papers), where={"paper_id": {"$in": ids}},
        )

    if not ranked:
        raise HTTPException(status_code=404, detail="No papers found for this topic.")

    paper_ids = [p.paper_id for p, _ in ranked]
    scores = [score for _, score in ranked]
    search_id, created_at = save_search(_state["db"], req.topic, paper_ids, scores)

    return SearchResponse(
        search_id=search_id, topic=req.topic, created_at=created_at,
        papers=[_paper_to_out(p, score) for p, score in ranked],
    )


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    saved = get_search(_state["db"], req.search_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="search_id not found")

    summary_json = _get_or_create_summary(req.search_id, saved)
    return SummarizeResponse(search_id=req.search_id, topic=saved.topic, **summary_json)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    saved = get_search(_state["db"], req.search_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="search_id not found")

    papers = get_papers_by_ids(saved.paper_ids, collection=_state["collection"])
    session = ChatSession(papers=papers, history=[turn.model_dump() for turn in req.history])
    result = ask(session, req.question, client=_state["client"])

    return ChatResponse(
        answer=result["answer"],
        answerable=result["answerable"],
        cited_papers=[CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in result["cited_papers"]],
        history=[ChatTurn(**turn) for turn in session.history],
    )


@app.get("/export/{search_id}", response_class=PlainTextResponse)
def export(search_id: int) -> str:
    saved = get_search(_state["db"], search_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="search_id not found")

    summary_json = _get_or_create_summary(search_id, saved)
    return _render_markdown(saved.topic, summary_json)


@app.get("/library", response_model=list[LibraryItem])
def library() -> list[LibraryItem]:
    saved_list = list_searches(_state["db"])
    return [
        LibraryItem(
            search_id=s.id, topic=s.topic, created_at=s.created_at,
            paper_count=len(s.paper_ids), has_summary=s.summary is not None,
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
    )
