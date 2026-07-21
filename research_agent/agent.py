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
from research_agent.enrichment import enrich_missing_abstracts
from research_agent.ingestion import search_arxiv, search_semantic_scholar
from research_agent.schema import Paper, WebArticle
from research_agent.web_search import search_web

logger = logging.getLogger(__name__)

AGENT_MODEL = "openai:gpt-4.1-mini"

# top_k is a user-controlled request parameter (phase-round-2 enhancement 1),
# not something the model should infer. Earlier versions left the number of
# final results unspecified in the prompt, and the model consistently
# defaulted to 5 regardless of how many candidates were actually gathered —
# an LLM-decided default, not a code-enforced one. Baking the exact number
# into the prompt (rather than just relying on the tool's own default
# argument) closes that gap at the source. api.py additionally re-ranks
# server-side if the agent's result count doesn't match what was requested,
# so correctness never depends solely on the model following this
# instruction.
def _build_system_prompt(top_k: int, web_max_results: int) -> str:
    return f"""You are a research assistant that finds academic papers on arXiv and Semantic Scholar for a user's research topic, and can optionally pull in current web context alongside them.

Both academic search tools do literal keyword matching, not semantic search — they will miss relevant papers if the query uses different wording than the papers do. Before searching, consider whether the user's topic should be reformulated: expand acronyms (e.g. "PEFT" -> "parameter-efficient fine-tuning"), spell out abbreviations, or add an obvious synonym/related term. You may issue more than one search per source if the first query seems too narrow or too broad.

Decide whether to search arXiv, Semantic Scholar, or both:
- Search both by default — they have different coverage (arXiv: preprints; Semantic Scholar: published/peer-reviewed venues, citation counts).
- Search only one if the user's request specifically scopes to that source (e.g. "arXiv preprints on X").

Once you've gathered enough candidate papers, call rerank_by_relevance with the user's original topic (phrased naturally, not as bare keywords) and with top_k set to exactly {top_k} — that is the number of results the user asked for, not a default to infer or guess at. Do not pass a different number. Always call it before giving your final answer — it's the actual relevance ranking, not optional polish.

You also have search_web, a separate tool for current web context (news, tooling, blog posts, documentation, benchmarks/leaderboards, industry adoption) — a genuinely different corpus from the academic papers above, never merged with them. Use your judgment on whether the topic calls for it:
- Call it for topics where recent, practical, or fast-moving information matters — e.g. "current state of X", specific tools/frameworks/products, anything time-sensitive, or where the user's phrasing suggests they want more than the academic literature (e.g. "latest", "current", "in practice", "which tools").
- Skip it for purely historical or theoretical topics where academic papers are the complete, appropriate answer (e.g. "foundational results in X", "the original Y algorithm") — searching the web there would just add noise, not signal.
- If you do call it, use max_results set to exactly {web_max_results} — that is the number of web results the user asked for, not a default to infer.
- This tool is independent of the paper searches: skipping it (or it returning nothing, e.g. if no web search provider is configured) never blocks or changes the paper results — always finish and report the papers either way.

When you respond to the user, summarize the top-ranked papers you found (title, why it's relevant) — don't just say you searched, report what you found. If you also searched the web, mention that separately — it is supplementary context, not part of the paper count.
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
    # Round-2 enhancement 2: DOI/citation-count filters. These are set once
    # from the user's request (see run_research_agent) and read directly by
    # rerank_by_relevance_tool below — deliberately NOT exposed as tool-call
    # arguments the model fills in. Enhancement 1 already showed that an
    # LLM-decided value for something the user explicitly configured is the
    # wrong pattern; these filters are a hard user constraint, not a
    # judgment call for the model to make.
    doi_required: bool = False
    min_citation_count: int = 0
    # Round-2 enhancement 5: web articles are a SEPARATE accumulated pool,
    # never merged into `papers` — the paper count/list and the web context
    # section stay independently sized and independently displayed all the
    # way through to the UI.
    web_articles: list[WebArticle] = field(default_factory=list)


def _merge_web_articles(existing: list[WebArticle], new: list[WebArticle]) -> list[WebArticle]:
    """Fold newly-found web articles into the accumulated pool, deduped by
    URL (the natural identity for a web result — unlike papers, there's no
    cross-source fuzzy-matching concern here since every result already
    comes from the one source, Tavily)."""
    seen = {a.url for a in existing}
    merged = list(existing)
    for article in new:
        if article.url not in seen:
            seen.add(article.url)
            merged.append(article)
    return merged


def build_tools(session: ResearchSession) -> list:
    @tool
    def search_arxiv_tool(query: str) -> str:
        """Search arXiv for papers matching a query. Returns a short summary;
        the full records are added to the working paper pool for later
        reranking. arXiv's search is a literal keyword match, not semantic —
        use specific, well-formed search terms, and expand any acronyms first."""
        try:
            # No max_results argument here on purpose (see round-2/measure-
            # langgraph-agent postmortem): this used to hardcode its own
            # max_results=10 default, a second, disconnected copy of
            # ingestion.py's own max_results=20 default that silently drifted
            # out of sync and starved the agent's candidate pool relative to
            # every direct-function retrieval path. There is now exactly one
            # place this number is defined — search_arxiv's own default —
            # and this tool always inherits it, the same way top_k above is a
            # code-enforced value rather than something left for the model
            # to infer.
            papers = search_arxiv(query)
            session.papers = deduplicate(session.papers + papers)
            sample = "; ".join(p.title for p in papers[:5])
            return (
                f"arXiv returned {len(papers)} paper(s) for query {query!r}. "
                f"Working pool now has {len(session.papers)} paper(s) total. Sample: {sample or '(none)'}"
            )
        except Exception as exc:
            # A tool failure (e.g. an unexpected network/API error that
            # ingestion.py's own defensive handling didn't catch) must not
            # kill the whole agent run — surface it as a normal tool
            # observation instead of an unhandled exception, so the agent
            # can retry this tool, fall back to another source, or continue
            # with whatever the working pool already has.
            logger.warning("search_arxiv_tool failed for query %r: %s", query, exc)
            return (
                f"arXiv search failed for query {query!r}: {exc}. The working pool is unchanged "
                f"({len(session.papers)} paper(s) so far) — you can retry this search, try a "
                "different source, or continue with what's already been found."
            )

    @tool
    def search_semantic_scholar_tool(query: str) -> str:
        """Search Semantic Scholar for papers matching a query — broader
        coverage than arXiv (published/peer-reviewed venues, citation counts).
        Returns a short summary; full records are added to the working pool.
        Also a literal keyword match, not semantic — expand acronyms first."""
        try:
            # No max_results argument here either — same single-source-of-
            # truth reasoning as search_arxiv_tool above; inherits
            # search_semantic_scholar's own default.
            papers = search_semantic_scholar(query, api_key=session.s2_api_key)
            session.papers = deduplicate(session.papers + papers)
            sample = "; ".join(p.title for p in papers[:5])
            return (
                f"Semantic Scholar returned {len(papers)} paper(s) for query {query!r}. "
                f"Working pool now has {len(session.papers)} paper(s) total. Sample: {sample or '(none)'}"
            )
        except Exception as exc:
            logger.warning("search_semantic_scholar_tool failed for query %r: %s", query, exc)
            return (
                f"Semantic Scholar search failed for query {query!r}: {exc}. The working pool is "
                f"unchanged ({len(session.papers)} paper(s) so far) — you can retry this search, "
                "try a different source, or continue with what's already been found."
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

        try:
            # Best-effort abstract recovery (round-2 enhancement 4), before the
            # embed step decides whether a paper needs the title-only fallback —
            # a paper that gets a real abstract recovered here never has to take
            # that fallback at all. Failures here are swallowed inside
            # enrich_missing_abstracts itself; this call never raises.
            enrich_missing_abstracts(session.papers)

            collection = get_chroma_collection()
            client = OpenAI()
            stats = embed_and_index_papers(session.papers, collection=collection, client=client)

            ids = [p.paper_id for p in session.papers]
            ranked = semantic_search(
                query, collection=collection, client=client, top_k=top_k,
                where={"paper_id": {"$in": ids}},
                min_citation_count=session.min_citation_count or None,
                require_doi=session.doi_required,
            )
            session.ranked = ranked

            if not ranked:
                return (
                    f"No papers matched the active filters (doi_required={session.doi_required}, "
                    f"min_citation_count={session.min_citation_count}) among the {len(session.papers)} "
                    "collected paper(s). Report this to the user rather than fabricating results — "
                    "they may want to relax the filters."
                )

            lines = [f"{i + 1}. ({score:.3f}) {p.title}" for i, (p, score) in enumerate(ranked)]
            return (
                f"Ranked {len(ranked)} paper(s) by relevance to {query!r} "
                f"({stats['cache_hits']} cache hit(s), {stats['cache_misses']} newly embedded, "
                f"~${stats['estimated_cost_usd']:.6f}):\n" + "\n".join(lines)
            )
        except Exception as exc:
            # Most likely an OpenAI embedding-call failure (network/rate
            # limit/API error) — session.ranked is left exactly as it was
            # before this call (untouched above the try), so a prior
            # successful rerank isn't lost, and the papers gathered so far
            # remain available for a retry.
            logger.warning("rerank_by_relevance_tool failed for query %r: %s", query, exc)
            return (
                f"Reranking failed: {exc}. The {len(session.papers)} collected paper(s) are still "
                "available — you can retry reranking, or report the papers found so far without a "
                "relevance ranking."
            )

    @tool
    def search_web_tool(query: str, max_results: int = 4) -> str:
        """Search the current web (news, tooling, docs, benchmarks, industry
        adoption) for context alongside the academic papers — a genuinely
        separate corpus, never merged into the paper pool or its count.
        Use this for topics where recent/practical information matters, not
        purely historical or theoretical ones. Degrades to an empty result
        (never an error) if no web search provider is configured — that
        never blocks or changes the paper search."""
        try:
            articles = search_web(query, max_results=max_results)
            session.web_articles = _merge_web_articles(session.web_articles, articles)
            sample = "; ".join(a.title for a in articles[:5])
            return (
                f"Web search returned {len(articles)} article(s) for query {query!r}. "
                f"Web context pool now has {len(session.web_articles)} article(s) total. Sample: {sample or '(none)'}"
            )
        except Exception as exc:
            logger.warning("search_web_tool failed for query %r: %s", query, exc)
            return (
                f"Web search failed for query {query!r}: {exc}. This is supplementary context, not "
                "part of the paper results — continue and report the papers found either way."
            )

    return [search_arxiv_tool, search_semantic_scholar_tool, rerank_by_relevance_tool, search_web_tool]


def run_research_agent(
    topic: str,
    s2_api_key: str | None = None,
    top_k: int = 10,
    doi_required: bool = False,
    min_citation_count: int = 0,
    web_max_results: int = 4,
    on_step=None,
) -> ResearchSession:
    """Run the agent on a topic, streaming step-by-step so tool calls and
    reasoning can be logged/observed as they happen (not just the final
    output). `on_step(message)` is called for every message the agent
    produces or receives, in order, if provided.

    top_k is the exact number of final results the user asked for (default
    10, matching the tool's own default). It's baked into the system prompt
    so the model doesn't have to infer a count.

    doi_required and min_citation_count (round-2 enhancement 2) are applied
    by rerank_by_relevance_tool directly from session state, not from
    anything the model decides.

    web_max_results (round-2 enhancement 5) is the exact count to use *if*
    the agent decides web context is relevant for this topic — same
    code-enforced-count pattern as top_k, but WHETHER to search the web at
    all remains the model's judgment call (unlike top_k/filters, that's a
    genuine per-topic decision, not a user-configured hard constraint).
    """
    session = ResearchSession(
        s2_api_key=s2_api_key, doi_required=doi_required, min_citation_count=min_citation_count,
    )
    tools = build_tools(session)
    agent = create_agent(AGENT_MODEL, tools=tools, system_prompt=_build_system_prompt(top_k, web_max_results))

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
