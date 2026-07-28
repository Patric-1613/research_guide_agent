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

import asyncio
import logging
from dataclasses import dataclass, field

from langfuse import get_client, observe
from openai import OpenAI
from pydantic import BaseModel, Field

from research_agent.dedup import deduplicate
from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, semantic_search
from research_agent.ingestion import (
    get_rate_limited_call_count,
    reset_rate_limit_tracking,
    search_arxiv,
    search_openalex,
    search_semantic_scholar,
)
from research_agent.schema import Paper, WebArticle
from research_agent.tracing import ranked_paper_metadata, tag_current_trace

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

# Phase 2 (parallelize-search-calls): how many different suggested titles'
# arXiv+Semantic Scholar pairs may be in flight at once. 2, not the 3
# top of the brief's suggested 2-3 range, and deliberately not unlimited:
# this project's own real trace history shows Semantic Scholar's shared
# rate limit tripping under completely sequential load already (94.1%
# incidence on the agent path, 35.2% on this same expanded_search path) —
# firing every suggested title's pair at once would multiply that
# exposure, not just latency. 2 keeps some real concurrency benefit
# (still lets one pair's search run while another's is in flight) while
# adding the smallest amount of new simultaneous load against a limit
# already shown to be strained.
_MAX_CONCURRENT_TITLE_PAIRS = 2

SUGGEST_TITLES_SYSTEM_PROMPT = """You suggest well-known, REAL academic papers relevant to a research topic, to help widen a search net.

Strict rule: only include a title if you could bet money it is the exact, verbatim title of a real, published paper you have encountered many times in training data. If you are reconstructing or guessing a plausible-sounding title for a paper you are not certain exists with that exact wording, DO NOT include it — leave it out entirely rather than approximate it.

Most topics, especially narrow or highly specific ones, do NOT have 5 genuinely well-known landmark papers. Returning 0-2 titles is the common, correct case for a narrow topic. Only return close to the requested count for extremely well-established, widely-taught topics (e.g. attention mechanisms, ResNet, BERT).

Prefer foundational/landmark papers (the kind widely cited as THE reference for a technique or idea) over obscure or tangential ones — but "foundational" means foundational to the topic's SPECIFIC question, not just to its broad research area. A topic phrase usually names a precise focus within a larger field, not the field itself — e.g. "reducing hallucination in retrieval-augmented generation" is specifically about hallucination reduction, not RAG in general. If you know of a paper that directly targets that specific focus, it belongs ahead of a more famous but more general paper from the same broad area: a well-known general RAG paper like REALM is the wrong answer for a hallucination-specific topic if you know of a paper that actually addresses hallucination reduction. Only fall back to the broader area's foundational paper when you genuinely don't know a more specific one — don't reach for the safe, famous default when a more targeted real paper is available in your knowledge."""


class _TitleSuggestions(BaseModel):
    titles: list[str] = Field(
        description="Well-known, real paper titles relevant to the topic. Fewer than requested (even zero) if not genuinely confident about more.",
    )


@observe(name="suggest_related_titles", as_type="generation", capture_input=False, capture_output=False)
def suggest_related_titles(
    topic: str, max_titles: int = 5, client: OpenAI | None = None,
    exclude_titles: list[str] | None = None, refinement_notes: list[str] | None = None,
) -> list[str]:
    """One LLM call. Returns up to max_titles well-known real paper titles
    related to topic, or fewer if the model isn't confident about that many
    — never padded to a fixed count.

    exclude_titles (curation-pool-foundation Phase 1c): titles already
    found/shown this session — told to the model IN THE PROMPT ITSELF,
    not just filtered out of its response afterward. At
    TITLE_SUGGESTION_TEMPERATURE=0.1 (near-deterministic), a second call
    for the same topic with no such instruction would very likely just
    return the same well-known landmark titles again — post-filtering
    would silently shrink the result (possibly to zero) rather than
    actually prompting the model to think of DIFFERENT ones. A defensive
    post-filter is still applied below on top of the prompt instruction
    (belt and suspenders, same pattern as this project's other
    defensive/enforced-not-just-instructed checks) in case the model
    still repeats one despite being told not to.

    refinement_notes (curation-refinement-and-auto-offer Phase 6f):
    free-text steering the user typed mid-curation (e.g. "focus on more
    recent work") — same prompt-level mechanism as exclude_titles above,
    not post-filtering, since this is genuinely LLM-appropriate guidance
    (unlike the direct arXiv/Semantic Scholar query, a literal keyword
    search that natural-language steering would only dilute, not help).
    Cumulative across the session (never cleared), same growth semantics
    as exclude_titles' own seen_titles source.

    Defensive like the rest of the project's ingestion layer (ingestion.py):
    any failure (API error, malformed/empty response) logs and returns an
    empty list rather than raising, so a failure here degrades to "no
    expansion" instead of breaking the search that's using it.
    """
    if not topic.strip():
        logger.warning("suggest_related_titles called with empty topic")
        return []

    client = client or OpenAI()
    user_content = f"Topic: {topic}\n\nSuggest up to {max_titles} well-known real papers on this topic."
    if exclude_titles:
        excluded_list = "\n".join(f"- {t}" for t in exclude_titles)
        user_content += (
            f"\n\nThe following have already been found and must NOT be suggested again:\n{excluded_list}\n\n"
            "Suggest DIFFERENT real papers instead. If you genuinely don't know any other well-known, "
            "verbatim-titled real papers on this specific topic beyond the ones listed above, return "
            "fewer titles (even zero) rather than repeating one of them."
        )
    if refinement_notes:
        notes_list = "\n".join(f"- {n}" for n in refinement_notes)
        user_content += (
            f"\n\nThe user has additionally asked you to refine your suggestions with this guidance:\n{notes_list}\n\n"
            "Take this into account when choosing which papers to suggest."
        )
    messages = [
        {"role": "system", "content": SUGGEST_TITLES_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    langfuse = get_client()
    langfuse.update_current_generation(
        input=messages,
        model=TITLE_SUGGESTION_MODEL,
        model_parameters={"temperature": TITLE_SUGGESTION_TEMPERATURE},
    )

    try:
        response = client.chat.completions.parse(
            model=TITLE_SUGGESTION_MODEL,
            temperature=TITLE_SUGGESTION_TEMPERATURE,
            messages=messages,
            response_format=_TitleSuggestions,
        )
    except Exception:
        logger.warning("suggest_related_titles: LLM call failed for topic %r", topic, exc_info=True)
        langfuse.update_current_generation(output={"titles": []}, level="WARNING", status_message="LLM call failed")
        return []

    usage = response.usage
    if usage is not None:
        cost = (usage.prompt_tokens / 1_000_000 * _TITLE_MODEL_PRICE_PER_1M_INPUT
                + usage.completion_tokens / 1_000_000 * _TITLE_MODEL_PRICE_PER_1M_OUTPUT)
        logger.info(
            "suggest_related_titles: %d tokens billed (prompt=%d, completion=%d, ~$%.6f)",
            usage.total_tokens, usage.prompt_tokens, usage.completion_tokens, cost,
        )
        langfuse.update_current_generation(
            usage_details={
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        logger.warning("suggest_related_titles: model refused/returned no parsed content for topic %r", topic)
        langfuse.update_current_generation(output={"titles": []})
        return []

    titles = [t.strip() for t in parsed.titles if t and t.strip()][:max_titles]

    if exclude_titles:
        exclude_normalized = {t.strip().lower() for t in exclude_titles}
        before_filter = len(titles)
        titles = [t for t in titles if t.lower() not in exclude_normalized]
        if len(titles) < before_filter:
            logger.warning(
                "suggest_related_titles: model repeated %d excluded title(s) for topic %r despite the prompt "
                "instruction — filtered defensively",
                before_filter - len(titles), topic,
            )

    if len(titles) < max_titles:
        logger.info(
            "suggest_related_titles: model returned %d/%d titles for topic %r (fewer is expected when not confident)",
            len(titles), max_titles, topic,
        )
    langfuse.update_current_generation(output={"titles": titles})
    return titles


# Cheap, cost-tiered model, same reasoning as TITLE_SUGGESTION_MODEL above --
# a narrow, low-stakes rewording task, not the quality-sensitive step.
CANONICALIZE_TOPIC_MODEL = "gpt-4.1-mini"
# Same reasoning as TITLE_SUGGESTION_TEMPERATURE: a display-title restatement
# should be stable for the same input, not creatively varied run to run.
CANONICALIZE_TOPIC_TEMPERATURE = 0.1

CANONICALIZE_TOPIC_SYSTEM_PROMPT = """You rewrite a user's raw, possibly informal research-topic input into a short, well-formed restatement of the SAME topic -- for display as a review's title, not as a search query or a paper title.

Rules:
- Preserve the exact subject matter and scope. Never broaden, narrow, or reinterpret what the user asked about.
- Fix casing, spelling, and grammar; make it read as a clean, professional topic label (e.g. "cars cooling system" -> "Automotive engine cooling systems").
- Do not add information the input didn't contain. Do not turn it into a question, a sentence, or the title of a specific paper -- a short, well-formed topic phrase, like a real topic label.
- If the input is already clean and well-formed, return it unchanged or with only trivial casing/spelling fixes."""


class _CanonicalTopic(BaseModel):
    canonical_topic: str = Field(
        description="A short, well-formed restatement of the same topic -- not a paper title, not a search query, not a question.",
    )


@observe(name="canonicalize_topic", as_type="generation", capture_input=False, capture_output=False)
def canonicalize_topic(topic: str, client: OpenAI | None = None) -> str:
    """curation-review-management Phase 8, item 5: produces a clean DISPLAY
    title for a review from the user's raw, possibly-informal input -- e.g.
    "cars cooling system" -> "Automotive engine cooling systems". Display-
    only: the raw `topic` string is still what actually drives search/
    ranking/refinement everywhere else in the curation flow (PaperPoolSession
    .topic is never touched by this) -- the returned string is meant to be
    stored in a SEPARATE display_title field, per the approved design.

    Confirmed before writing this that no existing step produced anything
    like it: suggest_related_titles() (above) returns candidate PAPER
    titles to search for, not a restatement of the topic itself -- wrong
    shape of data for this job. Neither search_arxiv nor
    search_semantic_scholar (ingestion.py) ever return a corrected/resolved
    query either; the raw string is passed straight through to both.

    Defensive like every other LLM-touching function in this module: any
    failure (API error, empty/malformed response) logs and returns the raw
    topic unchanged, so a failure here degrades to "no cleanup" rather than
    blocking review creation.
    """
    if not topic.strip():
        return topic

    client = client or OpenAI()
    messages = [
        {"role": "system", "content": CANONICALIZE_TOPIC_SYSTEM_PROMPT},
        {"role": "user", "content": f"Raw input: {topic}"},
    ]
    langfuse = get_client()
    langfuse.update_current_generation(
        input=messages,
        model=CANONICALIZE_TOPIC_MODEL,
        model_parameters={"temperature": CANONICALIZE_TOPIC_TEMPERATURE},
    )

    try:
        response = client.chat.completions.parse(
            model=CANONICALIZE_TOPIC_MODEL,
            temperature=CANONICALIZE_TOPIC_TEMPERATURE,
            messages=messages,
            response_format=_CanonicalTopic,
        )
    except Exception:
        logger.warning("canonicalize_topic: LLM call failed for topic %r", topic, exc_info=True)
        langfuse.update_current_generation(output={"canonical_topic": topic}, level="WARNING", status_message="LLM call failed")
        return topic

    usage = response.usage
    if usage is not None:
        logger.info(
            "canonicalize_topic: %d tokens billed (prompt=%d, completion=%d)",
            usage.total_tokens, usage.prompt_tokens, usage.completion_tokens,
        )
        langfuse.update_current_generation(
            usage_details={
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        )

    parsed = response.choices[0].message.parsed
    if parsed is None or not parsed.canonical_topic.strip():
        logger.warning("canonicalize_topic: model refused/returned empty content for topic %r", topic)
        langfuse.update_current_generation(output={"canonical_topic": topic})
        return topic

    canonical = parsed.canonical_topic.strip()
    langfuse.update_current_generation(output={"canonical_topic": canonical})
    return canonical


def _search_semantic_scholar_with_openalex_fallback(
    query: str, max_results: int, s2_api_key: str | None, openalex_mailto: str | None,
) -> list[Paper]:
    """search_semantic_scholar(), falling back to search_openalex() ONLY
    when S2's own retry loop exhausts and gives up (empty result AND a
    real rate-limit event occurred during this call) — never on an
    ordinary zero-result search, which must behave identically whether or
    not the fallback is enabled.

    reset_rate_limit_tracking()/get_rate_limited_call_count() here are
    scoped to THIS call only, not the caller's own rollup: this function
    is always invoked via asyncio.to_thread (see _search_pair below),
    which copies the current contextvars Context into the new thread
    before running — mutating the tracker inside that copy (via .set())
    never propagates back to the parent's own Context, so
    build_candidate_pool's own top-level reset_rate_limit_tracking()/
    get_rate_limited_call_count() rollup (used for its
    search_had_rate_limit span metadata) is completely unaffected by this
    — verified against contextvars' own documented copy-on-spawn behavior,
    not assumed. One accepted, minor side effect: when the fallback is
    enabled and actually fires, that specific call's rate-limit event is
    no longer counted in the OUTER rollup either — arguably the right
    framing anyway (the rate-limiting was mitigated here, not left as an
    unaddressed problem for the caller to see).
    """
    reset_rate_limit_tracking()
    papers = search_semantic_scholar(query, max_results=max_results, api_key=s2_api_key)
    if not papers and get_rate_limited_call_count() > 0:
        logger.warning(
            "Semantic Scholar exhausted retries for query %r — falling back to OpenAlex",
            query,
        )
        return search_openalex(query, max_results=max_results, mailto=openalex_mailto)
    return papers


async def _search_pair(
    query: str, max_results: int, s2_api_key: str | None,
    use_openalex_fallback: bool = False, openalex_mailto: str | None = None,
) -> tuple[list[Paper], list[Paper]]:
    """One arXiv + Semantic Scholar search for the SAME query, run
    concurrently instead of sequentially — real trace data showed these
    two independent network calls running back-to-back inside
    build_candidate_pool (~27s for one pair) despite neither depending on
    the other's result at all.

    asyncio.to_thread(), not a rewrite to async HTTP libraries: search_arxiv/
    search_semantic_scholar stay fully synchronous, unchanged. Everything
    that calls build_candidate_pool()/expanded_search() (api.py's sync
    FastAPI route, scripts/eval_retrieval.py's CLI) is synchronous too, so
    each call site wraps a call to this function in asyncio.run() — a
    short-lived event loop just for one pair's concurrent fetch, not a
    cascading async rewrite of this module's own public signatures.
    asyncio.to_thread() copies the caller's contextvars into the new
    thread, so search_arxiv's/search_semantic_scholar's own @observe spans
    still nest correctly under whichever span called this (verified
    directly against real trace data, not assumed).

    use_openalex_fallback defaults to False — the Semantic Scholar call
    is then exactly what it always was, byte-for-byte. Only when a caller
    explicitly opts in does the S2 half route through
    _search_semantic_scholar_with_openalex_fallback instead.

    Returns (arxiv_results, s2_results) — same order the previous
    sequential code accumulated them in, so callers concatenate identically.
    """
    arxiv_task = asyncio.to_thread(search_arxiv, query, max_results=max_results)
    if use_openalex_fallback:
        s2_task = asyncio.to_thread(
            _search_semantic_scholar_with_openalex_fallback, query, max_results, s2_api_key, openalex_mailto,
        )
    else:
        s2_task = asyncio.to_thread(search_semantic_scholar, query, max_results=max_results, api_key=s2_api_key)
    return await asyncio.gather(arxiv_task, s2_task)


async def _search_title_pairs_bounded(
    titles: list[str], max_results: int, s2_api_key: str | None, max_concurrent_pairs: int,
    use_openalex_fallback: bool = False, openalex_mailto: str | None = None,
) -> list[tuple[list[Paper], list[Paper]]]:
    """Phase 2: runs multiple suggested titles' arXiv+Semantic Scholar
    pairs concurrently, bounded by a semaphore so at most
    max_concurrent_pairs pairs are ever in flight — Phase 1's _search_pair()
    already parallelizes WITHIN one title's own pair; this additionally
    parallelizes ACROSS different titles' pairs, previously strictly
    sequential (title 2 waited for title 1 to fully finish).

    Bounded, not unlimited, given this project's own real rate-limiting
    history — see _MAX_CONCURRENT_TITLE_PAIRS' own comment for the exact
    reasoning behind the chosen cap.

    Returns one (arxiv_results, s2_results) tuple per title, in the SAME
    order as `titles` (asyncio.gather preserves input order), so the
    caller concatenates identically to the old sequential loop.
    """
    semaphore = asyncio.Semaphore(max_concurrent_pairs)

    async def _bounded_pair(title: str) -> tuple[list[Paper], list[Paper]]:
        async with semaphore:
            return await _search_pair(title, max_results, s2_api_key, use_openalex_fallback, openalex_mailto)

    return await asyncio.gather(*[_bounded_pair(title) for title in titles])


@observe(name="build_candidate_pool", capture_input=False, capture_output=False)
def build_candidate_pool(
    topic: str, k: int, s2_api_key: str | None = None, client: OpenAI | None = None,
    use_openalex_fallback: bool = False, openalex_mailto: str | None = None,
    exclude_titles: list[str] | None = None, refinement_notes: list[str] | None = None,
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

    use_openalex_fallback defaults to False — omitting it (or leaving it
    False) produces byte-identical behavior to before this parameter
    existed. When explicitly enabled, a Semantic Scholar call that
    exhausts its own retries falls back to OpenAlex for that one call
    only (see _search_semantic_scholar_with_openalex_fallback) — never a
    third always-on source, and never triggered by anything other than
    S2 genuinely giving up.
    """
    client = client or OpenAI()
    # Starts counting search_semantic_scholar calls (original query + one
    # per suggested title, below) that need a retry, so this function's own
    # span metadata can carry "how many of my child calls hit rate-limiting"
    # instead of that being visible only on each individual child span —
    # see ingestion.py's reset_rate_limit_tracking() docstring for why this
    # is a plain contextvar rather than a Langfuse mechanism.
    reset_rate_limit_tracking()

    original_pool_size = min(max(_ORIGINAL_QUERY_POOL_MULTIPLIER * k, _ORIGINAL_QUERY_POOL_FLOOR), _ORIGINAL_QUERY_POOL_CAP)

    # Phase 1 (parallelize-search-calls): arXiv and Semantic Scholar are
    # independent network calls for the same query — run concurrently, not
    # sequentially. See _search_pair()'s docstring for why asyncio.run()
    # here rather than making this function itself async.
    original_arxiv, original_s2 = asyncio.run(
        _search_pair(topic, original_pool_size, s2_api_key, use_openalex_fallback, openalex_mailto)
    )
    original_results = original_arxiv + original_s2

    suggested_titles = suggest_related_titles(
        topic, client=client, exclude_titles=exclude_titles, refinement_notes=refinement_notes,
    )

    # Phase 2 (parallelize-search-calls): different titles' pairs now run
    # concurrently too, bounded by _MAX_CONCURRENT_TITLE_PAIRS (see its own
    # comment for the exact reasoning) rather than strictly one-after-
    # another. Order preserved — flattened in the same title order as the
    # old sequential loop, arXiv results before Semantic Scholar's for each.
    title_pair_results = asyncio.run(
        _search_title_pairs_bounded(
            suggested_titles, _SUGGESTED_TITLE_POOL_SIZE, s2_api_key, _MAX_CONCURRENT_TITLE_PAIRS,
            use_openalex_fallback, openalex_mailto,
        )
    )
    suggested_results: list[Paper] = []
    for title_arxiv, title_s2 in title_pair_results:
        suggested_results += title_arxiv
        suggested_results += title_s2

    combined_raw = original_results + suggested_results
    deduped = deduplicate(combined_raw)

    logger.info(
        "build_candidate_pool(%r, k=%d): %d suggested title(s), %d raw result(s) "
        "(%d from original query, %d from suggested titles) -> %d after dedup",
        topic, k, len(suggested_titles), len(combined_raw), len(original_results), len(suggested_results),
        len(deduped),
    )

    rate_limited_calls = get_rate_limited_call_count()
    update_kwargs = {
        "input": {"topic": topic, "k": k},
        "output": {
            "suggested_titles": suggested_titles,
            "raw_count": len(combined_raw),
            "original_query_count": len(original_results),
            "suggested_title_count": len(suggested_results),
            "deduped_count": len(deduped),
        },
    }
    if rate_limited_calls:
        # Same "only set when true" convention as search_semantic_scholar's
        # own child-span metadata — a search that never hit rate-limiting
        # gets no such field at all, not an explicit False/0.
        update_kwargs["metadata"] = {"search_had_rate_limit": True, "rate_limit_count": rate_limited_calls}
    get_client().update_current_span(**update_kwargs)
    return deduped


_EMPTY_EMBED_STATS = {"cache_hits": 0, "cache_misses": 0, "tokens_billed": 0, "estimated_cost_usd": 0.0, "papers_skipped": 0}


def rank_full_pool(
    topic: str, deduped: list[Paper], client: OpenAI | None = None,
    doi_required: bool = False, min_citation_count: int = 0,
    collection=None,
) -> tuple[list[tuple[Paper, float]], dict]:
    """Embeds/indexes `deduped` and ranks ALL of it against `topic` — no
    truncation. expanded_search() below is the only truncating caller
    (slices to k for byte-identical existing behavior); the literature-
    review curation feature (curation-pool-foundation) is what actually
    needs the untruncated list, so it can hold the rest back as a reserve
    to serve later turns from without re-searching.

    Deliberately reuses semantic_search() unchanged rather than a new
    ranking path: passing top_k=len(deduped) (every candidate) is enough
    to get the full ranked list back — see
    tests/test_query_expansion.py's equivalence test for direct proof
    this produces the identical top-k a smaller top_k would have, not
    just an assumption that asking for more can't reorder fewer.

    Returns (ranked, embed_stats) — embed_stats mirrors
    embed_and_index_papers()'s own return shape (cache_hits/cache_misses/
    tokens_billed/estimated_cost_usd/papers_skipped) so callers can log
    cost the same way regardless of whether the pool was empty.
    """
    client = client or OpenAI()
    if not deduped:
        return [], dict(_EMPTY_EMBED_STATS)
    collection = collection or get_chroma_collection()
    embed_stats = embed_and_index_papers(deduped, collection=collection, client=client)
    ids = [p.paper_id for p in deduped]
    ranked = semantic_search(
        topic, collection=collection, client=client, top_k=len(ids), where={"paper_id": {"$in": ids}},
        require_doi=doi_required, min_citation_count=min_citation_count or None,
    )
    return ranked, embed_stats


@observe(name="expanded_search", capture_input=False, capture_output=False)
def expanded_search(
    topic: str, k: int, s2_api_key: str | None = None, client: OpenAI | None = None,
    doi_required: bool = False, min_citation_count: int = 0,
    use_openalex_fallback: bool = False, openalex_mailto: str | None = None,
) -> list[tuple[Paper, float]]:
    """Widen the candidate pool with LLM-suggested paper titles, then rerank
    against the ORIGINAL topic — never against a suggested title (see the
    anti-hallucination anchor in this module's docstring).

    Pipeline (locked, see module docstring for the parameters):
      1-4. build_candidate_pool() above — direct topic search + LLM-
         suggested-title search + cross-source dedup, unchanged.
      5. rank_full_pool() against `topic` (never a suggested title), then
         sliced to top-k here — same final result as truncating inside
         the ranking step itself (see rank_full_pool()'s own docstring),
         just with the truncation point moved to the caller so other
         callers (the curation feature) can keep the rest instead of
         discarding it. doi_required/min_citation_count pass straight
         through to semantic_search()'s own existing filter params —
         unchanged there, just forwarded.

    A hallucinated or wrong suggested title costs at most one extra pair of
    (likely empty or irrelevant) search calls — step 4's dedup and step 5's
    rerank against the original topic are what actually decide the final
    result, so nothing a suggested-title search turns up can enter the
    top-k without first being genuinely relevant to `topic`.

    Returns (Paper, similarity) pairs, same convention as semantic_search()
    itself — callers that only want the papers can discard the score.

    use_openalex_fallback/openalex_mailto pass straight through to
    build_candidate_pool() — see its docstring; default False produces
    identical behavior to before these parameters existed.
    """
    client = client or OpenAI()
    tag_current_trace(["expanded_search"])

    deduped = build_candidate_pool(
        topic, k, s2_api_key=s2_api_key, client=client,
        use_openalex_fallback=use_openalex_fallback, openalex_mailto=openalex_mailto,
    )

    full_ranked, embed_stats = rank_full_pool(
        topic, deduped, client=client, doi_required=doi_required, min_citation_count=min_citation_count,
    )
    ranked = full_ranked[:k]

    logger.info(
        "expanded_search(%r, k=%d): %d candidates -> %d final "
        "(embedding: %d cache hit(s), %d newly embedded, ~$%.6f)",
        topic, k, len(deduped), len(ranked),
        embed_stats["cache_hits"], embed_stats["cache_misses"], embed_stats["estimated_cost_usd"],
    )

    update_kwargs = {
        "input": {
            "topic": topic, "k": k,
            "doi_required": doi_required, "min_citation_count": min_citation_count,
        },
        "output": {"count": len(ranked), "papers": ranked_paper_metadata(ranked)},
    }
    # build_candidate_pool() (called above) already reset+read this same
    # counter for its OWN span; reading it again here (not resetting) rolls
    # the same count onto expanded_search's span too, since THIS is the
    # actual root of the trace whenever expanded_search wraps
    # build_candidate_pool (the live app's default path) rather than
    # build_candidate_pool being called directly (scripts/eval_retrieval.py's
    # ranking-mode experiments, where build_candidate_pool's own span above
    # is already the root and already carries this).
    rate_limited_calls = get_rate_limited_call_count()
    if rate_limited_calls:
        update_kwargs["metadata"] = {"search_had_rate_limit": True, "rate_limit_count": rate_limited_calls}
    get_client().update_current_span(**update_kwargs)
    return ranked


# curation-pool-foundation Phase 1b: session-scoped tracking so a
# literature-review curation loop can serve several turns' worth of
# candidates from ONE search, only re-searching once the reserve actually
# runs low. Foundation only — a plain in-memory structure a caller
# creates and threads through calls itself; no persistence yet (that's
# Phase 2's job, activating the checkpointer already built during the
# qa.py LangGraph conversion).
BATCH_SIZE = 10


@dataclass
class PaperPoolSession:
    """topic: what every search/refill in this session is for and ranked
    against (never a suggested title — same anti-hallucination anchor as
    expanded_search() above). reserve: the full ranked candidate list
    from the most recent build_candidate_pool()+rank_full_pool() call —
    NOT just top-k, the whole thing, per Phase 1a. cursor: how much of
    `reserve` has already been served via serve_next_batch(). seen_paper_ids:
    every paper_id ever served (shown OR picked — picked is always a
    subset of shown, so tracking "shown" alone covers both) — used both to
    never re-serve the same paper twice and, on refill, to exclude it from
    the fresh search. seen_titles: the same papers' titles, kept
    separately because paper_id alone isn't enough once refill_pool()
    replaces the reserve — the original Paper objects for already-served
    items aren't stored anywhere else, but suggest_related_titles's
    exclude_titles (Phase 1c) needs the actual title strings, not ids.
    stage (curation-checkpointer Phase 2): which part of the curation flow
    this session is in — "curate" (still picking papers), "synthesize"
    (report generation, Phase 4's future job), or "chat" (Phase 5's future
    job). Added now, alongside real persistence, since a session's stage
    is exactly the kind of thing that must survive a process restart —
    not used by any logic yet in this phase.
    target_count/selected_paper_ids (curation-interrupt-loop Phase 3):
    loop-control bookkeeping, not new pool/ranking/refill logic —
    target_count is how many papers the user wants total (set once at
    session start); selected_paper_ids is which of the seen papers were
    actually picked (ordered, a list not a set, so a UI can show pick
    order), distinct from seen_paper_ids (shown OR picked).
    selected_papers (curation-report-synthesis Phase 4): the SAME picks
    as selected_paper_ids, but as full serialized Paper dicts, not just
    ids — necessary because reserve is NOT a reliable place to resolve a
    picked paper_id back to its full data once a refill has happened
    (refill_pool replaces reserve with unserved_tail + genuinely_new,
    dropping the already-served prefix entirely). Populated at pick-time
    in curation_loop.py's present_and_apply node, which already has the
    full Paper data in scope (current_batch) before any future refill
    could discard it. Always kept in the same order and membership as
    selected_paper_ids — verified explicitly by
    tests/test_curation_loop.py's sync-invariant test across a session
    that includes a refill, not just assumed from populating both in the
    same node.
    """

    topic: str
    reserve: list[tuple[Paper, float]] = field(default_factory=list)
    cursor: int = 0
    seen_paper_ids: set[str] = field(default_factory=set)
    seen_titles: set[str] = field(default_factory=set)
    stage: str = "curate"
    target_count: int = 10
    selected_paper_ids: list[str] = field(default_factory=list)
    selected_papers: list[Paper] = field(default_factory=list)
    # curation-chat-web-escalation Phase 5: report is Phase 4's own gap —
    # generate_report_for_session() produced a report but nothing
    # persisted it anywhere; this is the running/current report
    # (regenerated in place as web sources get approved into it), same
    # dict shape generate_report() returns. chat_history mirrors
    # qa.ChatSession.history exactly, so each chat turn constructs a real
    # qa.ChatSession and calls qa.ask() unmodified. web_articles_added is
    # the running set of web sources ever approved — feeds both ongoing
    # chat citations (qa.ask()'s existing dual paper+web mechanism) and
    # report regeneration's expanded source set. pending_web_offer/
    # pending_report_update are the two "offer, then decide" moments —
    # set when an offer is made, cleared once the next turn resolves it.
    report: dict | None = None
    chat_history: list[dict] = field(default_factory=list)
    web_articles_added: list[WebArticle] = field(default_factory=list)
    pending_web_offer: dict | None = None
    pending_report_update: dict | None = None
    # curation-refinement-and-auto-offer Phase 6f: free-text guidance the
    # user typed mid-curation (e.g. "focus on more recent work"),
    # cumulative like seen_titles above -- flows into
    # suggest_related_titles' prompt via build_candidate_pool's own
    # refinement_notes param (see refill_pool), never the literal
    # arXiv/Semantic Scholar query string. report_covered_web_article_count
    # is the persisted version of what Phase 6c's frontend banner
    # originally tracked client-side only (and got wrong): how many of
    # web_articles_added the CURRENT report actually reflects, so a
    # report-update offer can be triggered by comparing this against
    # len(web_articles_added) rather than a "have any ever been added"
    # check that stays true forever after the first approval.
    refinement_notes: list[str] = field(default_factory=list)
    report_covered_web_article_count: int = 0
    # curation-review-management Phase 8, item 5: a canonicalize_topic()
    # restatement of `topic` for DISPLAY only (review titles/headers) --
    # `topic` itself is untouched and still what actually drives search/
    # ranking/refinement everywhere. Empty default here is never the real
    # value in practice: api.py's curation_start() always sets this
    # explicitly at session creation, the only place a fresh session is
    # built. Deserializing an OLDER session saved before this field
    # existed falls back to its own `topic` (see _dict_to_session) rather
    # than an empty string.
    display_title: str = ""
    # curation-turn-history Phase 9b: every batch ever served, in order --
    # NOT just seen_paper_ids/seen_titles (those only ever held bare
    # ids/strings, enough for search exclusion but not enough to redraw a
    # past turn's cards). Each entry: {"turn_number": int, "batch":
    # [[paper_dict, score], ...], "refilled": bool} -- turn_number is
    # derived from list position at append-time (len(turn_history) + 1),
    # not stored redundantly as its own counter. Intentionally unbounded
    # for now: no truncation/retention policy. Real cost, stated not
    # hidden -- this makes every future checkpoint snapshot strictly
    # larger over a session's life, compounding the same full-state-per-
    # checkpoint growth already observed in data/qa_checkpoints.sqlite. A
    # retention policy is a reasonable separate follow-up if this becomes
    # a real problem, not built preemptively here.
    turn_history: list[dict] = field(default_factory=list)
    # Persisted so a reload/reopen can still show WHY curation stopped
    # (target_met / user_stopped / exhausted) -- previously only ever
    # existed transiently in the one HTTP response of the turn that
    # caused it (CurationTurnResponse.stop_reason), never survived a
    # reload. None while stage == "curate"; set exactly once, in
    # _make_stop_node, same one-way semantics as `stage` itself.
    stop_reason: str | None = None

    def remaining(self) -> int:
        """How many un-served candidates are left in the current reserve."""
        return len(self.reserve) - self.cursor

    def needs_refill(self, batch_size: int = BATCH_SIZE) -> bool:
        return self.remaining() < batch_size


def serve_next_batch(session: PaperPoolSession, batch_size: int = BATCH_SIZE) -> list[tuple[Paper, float]]:
    """Returns the next up-to-batch_size unseen candidates from the
    reserve, advances the cursor, and marks them seen. Pure bookkeeping —
    never triggers a refill itself (that's a real search, a different,
    explicitly-opted-into concern; check session.needs_refill() and call
    refill_pool() separately). Returns fewer than batch_size (or zero)
    without error if the reserve doesn't have that many left — the caller
    decides whether that's a refill trigger or genuine exhaustion.
    """
    batch = session.reserve[session.cursor : session.cursor + batch_size]
    session.cursor += len(batch)
    for paper, _ in batch:
        session.seen_paper_ids.add(paper.paper_id)
        session.seen_titles.add(paper.title)
    return batch


def refill_pool(
    session: PaperPoolSession, k_for_widening: int = BATCH_SIZE,
    s2_api_key: str | None = None, client: OpenAI | None = None,
    use_openalex_fallback: bool = False, openalex_mailto: str | None = None,
) -> int:
    """Reruns build_candidate_pool()+rank_full_pool() for session.topic,
    then merges the result with whatever's still un-served in the current
    reserve (never re-searched, just carried forward) — the combined set
    is re-ranked and replaces the reserve; the cursor resets to 0, which
    is safe because both halves of the new reserve are guaranteed
    seen-free (the unserved tail was never shown by definition; the fresh
    results are filtered against session.seen_paper_ids below).

    Exclusion happens at TWO levels, not just one: session.seen_titles
    goes into build_candidate_pool()'s exclude_titles (Phase 1c — the
    suggested-title LLM call is actually told what's already been found,
    not just filtered afterward), and the fresh search's actual paper
    results are additionally filtered against session.seen_paper_ids
    below (catches anything the direct topic search turns up again, which
    exclude_titles can't reach since that only constrains the LLM's own
    title suggestions).

    session.refinement_notes (curation-refinement-and-auto-offer Phase
    6f) flows into build_candidate_pool()'s refinement_notes the same
    way seen_titles flows into exclude_titles above — read straight off
    the session already passed in, no new parameter needed here. Also
    folded into the RANKING query below (topic + refinement_notes, not
    bare topic) — confirmed necessary by a real, live e2e run, not
    assumed: refinement-influenced candidates were genuinely being
    added to the pool, but rank_full_pool still ranked purely against
    the original topic, so they could rank below the existing top
    candidates and never actually surface in the served batch, making
    the feature look like a no-op to a real user even though the
    underlying pool had measurably changed.

    Returns how many genuinely new (not already in the reserve or seen)
    papers this refill found — 0 is a real, valid signal that the topic
    is exhausted, not an error; callers should surface that to the user
    rather than looping forever (Phase 1d's adversarial case).
    """
    client = client or OpenAI()
    unserved_tail = session.reserve[session.cursor:]
    unserved_ids = {p.paper_id for p, _ in unserved_tail}

    fresh_pool = build_candidate_pool(
        session.topic, k_for_widening, s2_api_key=s2_api_key, client=client,
        use_openalex_fallback=use_openalex_fallback, openalex_mailto=openalex_mailto,
        exclude_titles=list(session.seen_titles) or None,
        refinement_notes=list(session.refinement_notes) or None,
    )
    exclude_ids = session.seen_paper_ids | unserved_ids
    genuinely_new = [p for p in fresh_pool if p.paper_id not in exclude_ids]

    combined_papers = [p for p, _ in unserved_tail] + genuinely_new
    ranking_query = session.topic
    if session.refinement_notes:
        ranking_query = f"{session.topic}. {' '.join(session.refinement_notes)}"
    ranked, _ = rank_full_pool(ranking_query, combined_papers, client=client)

    session.reserve = ranked
    session.cursor = 0
    return len(genuinely_new)
