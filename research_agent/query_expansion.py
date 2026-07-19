"""LLM-assisted query expansion for retrieval recall.

Root cause this addresses (confirmed by direct investigation in
scripts/eval_retrieval.py's baseline run): arXiv's and Semantic Scholar's
own keyword search APIs never return foundational papers (e.g. LoRA) for
broad topic-phrase queries, because those papers' titles/abstracts don't
closely match generic topic wording. This is a CANDIDATE-POOL problem —
semantic_search() can only rank what it was given, and a paper that never
enters the pool can never be reranked into it. suggest_related_titles()
widens the pool by asking a cheap LLM to name a few well-known real papers
on the topic, whose TITLES are then searched directly (a literal keyword
search on an exact title reliably surfaces that exact paper, unlike a
literal keyword search on a generic topic phrase).

Anti-hallucination anchor (do not weaken without discussing first): the
suggested titles are used ONLY to widen the search net. Final ranking
(expanded_search() in this same module) is always computed by embedding
similarity against the ORIGINAL topic text, never against a suggested
title. A hallucinated or slightly-wrong title can therefore only ever
waste one extra search call — a real-but-tangential paper it happens to
surface still has to earn its place in the final top-k by actually being
relevant to the original topic, the same bar every other candidate clears.
"""

from __future__ import annotations

import logging

from openai import OpenAI
from pydantic import BaseModel, Field

from research_agent.dedup import deduplicate
from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, semantic_search
from research_agent.ingestion import search_arxiv, search_semantic_scholar
from research_agent.schema import Paper

logger = logging.getLogger(__name__)

# Cheap, cost-tiered model — same reasoning as agent.py's orchestration
# model and qa.py's question-condensing model: this is a narrow, low-stakes
# suggestion task, not the quality-sensitive step.
TITLE_SUGGESTION_MODEL = "gpt-4.1-mini"

# Diagnostic run (see project history) confirmed genuine run-to-run
# non-determinism at the default temperature — e.g. the same topic/prompt
# surfaced a known landmark paper in 2 of 3 identical calls. This is a
# factual-recall task (name real, exact paper titles), not a creative one,
# so there's no upside to sampling variety here — only downside (an eval
# run's "miss" being sampling noise rather than a real gap). Picked 0.1,
# the low end of the requested 0.1-0.2 range: as close to deterministic as
# a real setting gets without asking for literal 0, which OpenAI doesn't
# guarantee is actually deterministic either (batched-inference floating
# point non-associativity), so there's no benefit to going lower than the
# range asked for.
TITLE_SUGGESTION_TEMPERATURE = 0.1

# Point-in-time USD/1M-token pricing for TITLE_SUGGESTION_MODEL, same
# transparency standard as embeddings.py's PRICE_PER_1M_TOKENS — checked
# via web search when this module was written, not fetched live. Verify
# against https://openai.com/api/pricing/ before trusting it for budgeting.
_TITLE_MODEL_PRICE_PER_1M_INPUT = 0.40
_TITLE_MODEL_PRICE_PER_1M_OUTPUT = 1.60

# Locked parameters (do not change without discussing first — see brief):
# original-topic-query pool size is 3x the requested k, floored at 15 and
# capped at 40; each suggested-title search is a fixed 5 per source,
# deliberately NOT scaled by k (a suggested title is searched to confirm/
# locate one specific paper, not to gather a broad candidate pool).
_ORIGINAL_QUERY_POOL_FLOOR = 15
_ORIGINAL_QUERY_POOL_CAP = 40
_ORIGINAL_QUERY_POOL_MULTIPLIER = 3
_SUGGESTED_TITLE_POOL_SIZE = 5

SUGGEST_TITLES_SYSTEM_PROMPT = """You suggest well-known, REAL academic papers relevant to a research topic, to help widen a search net.

Strict rule: only include a title if you could bet money it is the exact, verbatim title of a real, published paper you have encountered many times in training data. If you are reconstructing or guessing a plausible-sounding title for a paper you are not certain exists with that exact wording, DO NOT include it — leave it out entirely rather than approximate it.

Most topics, especially narrow or highly specific ones, do NOT have 5 genuinely well-known landmark papers. Returning 0-2 titles is the common, correct case for a narrow topic. Only return close to the requested count for extremely well-established, widely-taught topics (e.g. attention mechanisms, ResNet, BERT).

Prefer foundational/landmark papers (the kind widely cited as THE reference for a technique or idea) over obscure or tangential ones — but "foundational" means foundational to the topic's SPECIFIC question, not just to its broad research area. A topic phrase usually names a precise focus within a larger field, not the field itself — e.g. "reducing hallucination in retrieval-augmented generation" is specifically about hallucination reduction, not RAG in general. If you know of a paper that directly targets that specific focus, it belongs ahead of a more famous but more general paper from the same broad area: a well-known general RAG paper like REALM is the wrong answer for a hallucination-specific topic if you know of a paper that actually addresses hallucination reduction. Only fall back to the broader area's foundational paper when you genuinely don't know a more specific one — don't reach for the safe, famous default when a more targeted real paper is available in your knowledge."""


class _TitleSuggestions(BaseModel):
    titles: list[str] = Field(
        description="Well-known, real paper titles relevant to the topic. Fewer than requested (even zero) if not genuinely confident about more.",
    )


def suggest_related_titles(topic: str, max_titles: int = 5, client: OpenAI | None = None) -> list[str]:
    """One LLM call. Returns up to max_titles well-known real paper titles
    related to topic, or fewer if the model isn't confident about that many
    — never padded to a fixed count.

    Defensive like the rest of the project's ingestion layer (ingestion.py):
    any failure (API error, malformed/empty response) logs and returns an
    empty list rather than raising, so a failure here degrades to "no
    expansion" instead of breaking the search that's using it.
    """
    if not topic.strip():
        logger.warning("suggest_related_titles called with empty topic")
        return []

    client = client or OpenAI()

    try:
        response = client.chat.completions.parse(
            model=TITLE_SUGGESTION_MODEL,
            temperature=TITLE_SUGGESTION_TEMPERATURE,
            messages=[
                {"role": "system", "content": SUGGEST_TITLES_SYSTEM_PROMPT},
                {"role": "user", "content": f"Topic: {topic}\n\nSuggest up to {max_titles} well-known real papers on this topic."},
            ],
            response_format=_TitleSuggestions,
        )
    except Exception:
        logger.warning("suggest_related_titles: LLM call failed for topic %r", topic, exc_info=True)
        return []

    usage = response.usage
    if usage is not None:
        cost = (usage.prompt_tokens / 1_000_000 * _TITLE_MODEL_PRICE_PER_1M_INPUT
                + usage.completion_tokens / 1_000_000 * _TITLE_MODEL_PRICE_PER_1M_OUTPUT)
        logger.info(
            "suggest_related_titles: %d tokens billed (prompt=%d, completion=%d, ~$%.6f)",
            usage.total_tokens, usage.prompt_tokens, usage.completion_tokens, cost,
        )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        logger.warning("suggest_related_titles: model refused/returned no parsed content for topic %r", topic)
        return []

    titles = [t.strip() for t in parsed.titles if t and t.strip()][:max_titles]
    if len(titles) < max_titles:
        logger.info(
            "suggest_related_titles: model returned %d/%d titles for topic %r (fewer is expected when not confident)",
            len(titles), max_titles, topic,
        )
    return titles


def build_candidate_pool(
    topic: str, k: int, s2_api_key: str | None = None, client: OpenAI | None = None,
) -> list[Paper]:
    """Steps 1-4 of the pipeline documented on expanded_search() below:
    direct topic search widened to 3xk (floor 15, cap 40) + LLM-suggested-
    title search (fixed 5 per source per title) + cross-source dedup.
    Returns the deduped candidate pool, UNRANKED — ranking against the
    topic is a separate, pluggable concern (expanded_search() does it via
    semantic_search() below, the live app's only ranking mode; the
    ranking-stage experiment in scripts/eval_retrieval.py's --ranking-mode
    plugs in research_agent/ranking.py's BM25/hybrid alternatives against
    this SAME pool instead — never against a different or re-built one).

    Extracted out of expanded_search() so that experiment can reuse this
    exact candidate-pool-building logic unchanged (same locked pool-size
    parameters, same suggest_related_titles() call, same dedup) while
    swapping only the final ranking step. Nothing about steps 1-4
    themselves changed by this extraction — expanded_search() calls this
    function and then does exactly what it always did.
    """
    client = client or OpenAI()

    original_pool_size = min(max(_ORIGINAL_QUERY_POOL_MULTIPLIER * k, _ORIGINAL_QUERY_POOL_FLOOR), _ORIGINAL_QUERY_POOL_CAP)

    original_arxiv = search_arxiv(topic, max_results=original_pool_size)
    original_s2 = search_semantic_scholar(topic, max_results=original_pool_size, api_key=s2_api_key)
    original_results = original_arxiv + original_s2

    suggested_titles = suggest_related_titles(topic, client=client)

    suggested_results: list[Paper] = []
    for title in suggested_titles:
        suggested_results += search_arxiv(title, max_results=_SUGGESTED_TITLE_POOL_SIZE)
        suggested_results += search_semantic_scholar(title, max_results=_SUGGESTED_TITLE_POOL_SIZE, api_key=s2_api_key)

    combined_raw = original_results + suggested_results
    deduped = deduplicate(combined_raw)

    logger.info(
        "build_candidate_pool(%r, k=%d): %d suggested title(s), %d raw result(s) "
        "(%d from original query, %d from suggested titles) -> %d after dedup",
        topic, k, len(suggested_titles), len(combined_raw), len(original_results), len(suggested_results),
        len(deduped),
    )

    return deduped


def expanded_search(
    topic: str, k: int, s2_api_key: str | None = None, client: OpenAI | None = None,
    doi_required: bool = False, min_citation_count: int = 0,
) -> list[tuple[Paper, float]]:
    """Widen the candidate pool with LLM-suggested paper titles, then rerank
    against the ORIGINAL topic — never against a suggested title (see the
    anti-hallucination anchor in this module's docstring).

    Pipeline (locked, see module docstring for the parameters):
      1-4. build_candidate_pool() above — direct topic search + LLM-
         suggested-title search + cross-source dedup, unchanged.
      5. semantic_search() against `topic` (never a suggested title),
         cut to top-k. doi_required/min_citation_count pass straight
         through to semantic_search()'s own existing filter params —
         unchanged there, just forwarded.

    A hallucinated or wrong suggested title costs at most one extra pair of
    (likely empty or irrelevant) search calls — step 4's dedup and step 5's
    rerank against the original topic are what actually decide the final
    result, so nothing a suggested-title search turns up can enter the
    top-k without first being genuinely relevant to `topic`.

    Returns (Paper, similarity) pairs, same convention as semantic_search()
    itself — callers that only want the papers can discard the score.
    """
    client = client or OpenAI()

    deduped = build_candidate_pool(topic, k, s2_api_key=s2_api_key, client=client)

    collection = get_chroma_collection()
    embed_stats = embed_and_index_papers(deduped, collection=collection, client=client)
    ids = [p.paper_id for p in deduped]
    ranked = semantic_search(
        topic, collection=collection, client=client, top_k=k, where={"paper_id": {"$in": ids}},
        require_doi=doi_required, min_citation_count=min_citation_count or None,
    )

    logger.info(
        "expanded_search(%r, k=%d): %d candidates -> %d final "
        "(embedding: %d cache hit(s), %d newly embedded, ~$%.6f)",
        topic, k, len(deduped), len(ranked),
        embed_stats["cache_hits"], embed_stats["cache_misses"], embed_stats["estimated_cost_usd"],
    )

    return ranked
