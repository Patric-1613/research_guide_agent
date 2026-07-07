"""Phase 4: LangChain tool-calling agent that orchestrates search, dedup, and
relevance ranking.

The agent decides, per topic: whether to search arXiv, Semantic Scholar, or
both; whether to reformulate an ambiguous/acronym-heavy query before hitting
either API (both are literal keyword search, not semantic); and when to call
rerank_by_relevance to rank whatever it's collected so far.

Model choice: gpt-4.1-mini. This API key also has access to a gpt-5.x
lineup released after my training cutoff (Jan 2026) that I have no reliable
knowledge of the cost/quality tradeoffs for, so gambling on one felt like
the wrong default for someone tracking spend. gpt-4.1-mini is a known
quantity — cheap, and more than capable for tool-call orchestration, which
is a comparatively easy decision task (not the harder summarization/Q&A
work in phases 5-6, where model choice deserves a second look). It's a
single constant below if you want to try a newer one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain.tools import tool
from openai import OpenAI

from research_agent.dedup import deduplicate
from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, semantic_search
from research_agent.ingestion import search_arxiv, search_semantic_scholar
from research_agent.schema import Paper

logger = logging.getLogger(__name__)

AGENT_MODEL = "openai:gpt-4.1-mini"

SYSTEM_PROMPT = """You are a research assistant that finds academic papers on arXiv and Semantic Scholar for a user's research topic.

Both search tools do literal keyword matching, not semantic search — they will miss relevant papers if the query uses different wording than the papers do. Before searching, consider whether the user's topic should be reformulated: expand acronyms (e.g. "PEFT" -> "parameter-efficient fine-tuning"), spell out abbreviations, or add an obvious synonym/related term. You may issue more than one search per source if the first query seems too narrow or too broad.

Decide whether to search arXiv, Semantic Scholar, or both:
- Search both by default — they have different coverage (arXiv: preprints; Semantic Scholar: published/peer-reviewed venues, citation counts).
- Search only one if the user's request specifically scopes to that source (e.g. "arXiv preprints on X").

Once you've gathered enough candidate papers, call rerank_by_relevance with the user's original topic (phrased naturally, not as bare keywords) to rank them by semantic relevance. Always do this before giving your final answer — it's the actual relevance ranking, not optional polish.

When you respond to the user, summarize the top-ranked papers you found (title, why it's relevant) — don't just say you searched, report what you found.
"""


@dataclass
class ResearchSession:
    """Working state for one research run: the accumulated, deduped paper
    pool and the most recent relevance ranking. Tools close over an instance
    of this so the agent's tool calls can stay lightweight (a query string,
    not a payload of papers) while still sharing state across calls.
    """

    s2_api_key: str | None = None
    papers: list[Paper] = field(default_factory=list)
    ranked: list[tuple[Paper, float]] = field(default_factory=list)


def build_tools(session: ResearchSession) -> list:
    @tool
    def search_arxiv_tool(query: str, max_results: int = 10) -> str:
        """Search arXiv for papers matching a query. Returns a short summary;
        the full records are added to the working paper pool for later
        reranking. arXiv's search is a literal keyword match, not semantic —
        use specific, well-formed search terms, and expand any acronyms first."""
        papers = search_arxiv(query, max_results=max_results)
        session.papers = deduplicate(session.papers + papers)
        sample = "; ".join(p.title for p in papers[:5])
        return (
            f"arXiv returned {len(papers)} paper(s) for query {query!r}. "
            f"Working pool now has {len(session.papers)} paper(s) total. Sample: {sample or '(none)'}"
        )

    @tool
    def search_semantic_scholar_tool(query: str, max_results: int = 10) -> str:
        """Search Semantic Scholar for papers matching a query — broader
        coverage than arXiv (published/peer-reviewed venues, citation counts).
        Returns a short summary; full records are added to the working pool.
        Also a literal keyword match, not semantic — expand acronyms first."""
        papers = search_semantic_scholar(query, max_results=max_results, api_key=session.s2_api_key)
        session.papers = deduplicate(session.papers + papers)
        sample = "; ".join(p.title for p in papers[:5])
        return (
            f"Semantic Scholar returned {len(papers)} paper(s) for query {query!r}. "
            f"Working pool now has {len(session.papers)} paper(s) total. Sample: {sample or '(none)'}"
        )

    @tool
    def rerank_by_relevance_tool(query: str, top_k: int = 10) -> str:
        """Rank the papers collected so far by semantic relevance to a query
        (normally the user's original topic, phrased as a natural sentence —
        this step understands meaning, so keyword-only phrasing isn't
        necessary here). This embeds abstracts and retrieves by cosine
        similarity — it IS the relevance ranking, not a preview of one. Call
        this once you've searched, before reporting final results."""
        if not session.papers:
            return "No papers collected yet — search a source first."

        collection = get_chroma_collection()
        client = OpenAI()
        stats = embed_and_index_papers(session.papers, collection=collection, client=client)

        ids = [p.paper_id for p in session.papers]
        ranked = semantic_search(
            query, collection=collection, client=client, top_k=top_k,
            where={"paper_id": {"$in": ids}},
        )
        session.ranked = ranked

        lines = [f"{i + 1}. ({score:.3f}) {p.title}" for i, (p, score) in enumerate(ranked)]
        return (
            f"Ranked {len(ranked)} paper(s) by relevance to {query!r} "
            f"({stats['cache_hits']} cache hit(s), {stats['cache_misses']} newly embedded, "
            f"~${stats['estimated_cost_usd']:.6f}):\n" + "\n".join(lines)
        )

    return [search_arxiv_tool, search_semantic_scholar_tool, rerank_by_relevance_tool]


def run_research_agent(topic: str, s2_api_key: str | None = None, on_step=None) -> ResearchSession:
    """Run the agent on a topic, streaming step-by-step so tool calls and
    reasoning can be logged/observed as they happen (not just the final
    output). `on_step(message)` is called for every message the agent
    produces or receives, in order, if provided.
    """
    session = ResearchSession(s2_api_key=s2_api_key)
    tools = build_tools(session)
    agent = create_agent(AGENT_MODEL, tools=tools, system_prompt=SYSTEM_PROMPT)

    # stream_mode="values" yields the full cumulative message list after each
    # graph step. When the model issues more than one tool call in the same
    # turn (common — e.g. searching both sources at once), several messages
    # can land in a single step, so we diff against what we've already seen
    # rather than assume the last message is the only new one.
    seen = 0
    for step in agent.stream({"messages": [{"role": "user", "content": topic}]}, stream_mode="values"):
        messages = step["messages"]
        for message in messages[seen:]:
            if on_step:
                on_step(message)
        seen = len(messages)

    return session
