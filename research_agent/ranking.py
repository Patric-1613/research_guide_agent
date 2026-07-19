"""Ranking-stage experiment: pluggable alternatives to embeddings.py's
semantic_search() as the FINAL ranking step applied to an already-built
candidate pool (query_expansion.py's expanded_search() widens the pool;
this module only concerns itself with scoring/ordering it).

Why a new module rather than adding to embeddings.py: embeddings.py owns
vector embedding + Chroma persistence — a genuinely different concern from
the algorithms here (BM25 lexical scoring, RRF rank fusion), neither of
which touches the vector index at all. Keeping them separate means
embeddings.py doesn't grow a dependency on rank_bm25 for a scoring method
semantic_search() itself never uses, and this module reads as "the set of
interchangeable final-ranking strategies" in one place — which is exactly
what scripts/eval_retrieval.py's --ranking-mode flag needs to pick between.

Anti-hallucination anchor (same rule as query_expansion.py's own, extended
to every ranking mode here, not just semantic): every function in this
module ranks against the query text the CALLER passes in — which must
always be the original topic, never one of query_expansion.py's
LLM-suggested titles. This module has no opinion on what "query" means;
it's the caller's responsibility (scripts/eval_retrieval.py) to always
pass the original topic, exactly as expanded_search() already does for its
own semantic-only ranking today.
"""

from __future__ import annotations

import re

from openai import OpenAI
from rank_bm25 import BM25Okapi

from research_agent.embeddings import semantic_search
from research_agent.schema import Paper

# Standard RRF constant, confirmed via current best-practice research
# rather than assumed: k=60 is the value from the original Cormack et al.
# 2009 SIGIR paper ("Reciprocal Rank Fusion outperforms Condorcet and
# Individual Rank Learning Methods") and remains the default across
# Elasticsearch, OpenSearch, and Azure AI Search's hybrid-search RRF
# implementations as of 2026 — benchmarks consistently land in a k=40-80
# range with similar quality, and retuning it is only advisable with a
# labeled eval set of 200+ queries, which this project's 17-topic
# reference set does not approach. Not touched without that data.
RRF_K = 60

_TOKEN_RE = re.compile(r"\w+")


def _document_text(paper: Paper) -> str:
    """Text BM25 scores a paper against — abstract preferred, title as
    fallback. Deliberately mirrors embeddings.py's _embedding_text() same
    preference order, so semantic and BM25 rank the same textual
    representation of each paper; a mismatch there would confound any
    comparison between the two methods with a difference in what text
    each one even saw, rather than a difference in ranking algorithm."""
    return paper.abstract if paper.abstract else paper.title


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenization — the standard, unremarkable choice for
    BM25 (rank_bm25 itself ships no tokenizer; str.split() is its own
    README's example). \\w+ over bare .split() so punctuation-adjacent
    tokens ("transformers," vs "transformers") aren't spuriously treated
    as distinct terms, which would understate real term-frequency overlap."""
    return _TOKEN_RE.findall(text.lower())


def bm25_search(query: str, papers: list[Paper], top_k: int = 10) -> list[tuple[Paper, float]]:
    """BM25-rank `papers` against `query`. Structurally parallel to
    semantic_search()'s return convention — (Paper, score) pairs, ranked
    descending, cut to top_k — but NOT parallel in its parameters: BM25
    needs no vector index, no embedding client, and no persisted
    collection. It's a stateless, in-memory scoring function over whatever
    pool the caller hands it, computed fresh every call. That's a
    deliberate signature difference from semantic_search(), not an
    oversight — forcing BM25 through a Chroma-shaped signature would mean
    inventing an embedding-free "index" for a method that has no use for
    one.

    Returns a raw BM25 score (Okapi BM25, unbounded, NOT in [0, 1] the way
    semantic_search()'s cosine similarity is) — the two scores are on
    genuinely incompatible scales, which is exactly why hybrid_search()
    below fuses by RANK (RRF), never by combining these raw scores
    directly.
    """
    if not papers:
        return []

    corpus = [_tokenize(_document_text(p)) for p in papers]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(zip(papers, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[Paper, float]]], k: int = RRF_K, top_k: int = 10,
) -> list[tuple[Paper, float]]:
    """Combines multiple FULL rank-orderings of the same candidate pool
    into one fused ranking via Reciprocal Rank Fusion: each paper's fused
    score is the sum, across every input ranking it appears in, of
    1/(k + rank_position) (rank_position is 1-indexed). Rank-based by
    design (not raw-score blending) — this is what lets a cosine-
    similarity ranking and a BM25 ranking combine at all without
    normalizing two incompatible score scales against each other.

    Each element of `rankings` must be a FULL ranking of the same
    candidate pool (not independently pre-truncated to top_k) — a paper
    missing from one ranking simply contributes 0 from that ranking
    rather than being penalized further, but a ranking that was already
    cut to a small top_k before reaching this function would make every
    paper below that cutoff look identical (silently absent) instead of
    genuinely ranked-but-low, corrupting the fusion. Truncation to top_k
    happens ONCE, here, on the final fused result.

    Matches papers ACROSS rankings by paper_id (the project's existing
    convention for paper identity everywhere else — dedup.py, Chroma
    metadata filters), not by object identity or dataclass equality: the
    same paper can arrive as distinct Paper instances from different code
    paths (e.g. reconstructed from Chroma metadata vs. the original
    in-memory object), and those are not guaranteed to be `==`.
    """
    scores: dict[str, float] = {}
    paper_by_id: dict[str, Paper] = {}

    for ranking in rankings:
        for position, (paper, _score) in enumerate(ranking, start=1):
            scores[paper.paper_id] = scores.get(paper.paper_id, 0.0) + 1.0 / (k + position)
            paper_by_id.setdefault(paper.paper_id, paper)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(paper_by_id[pid], score) for pid, score in fused[:top_k]]


def partition_by_citation(papers: list[Paper], n: int) -> tuple[list[Paper], list[Paper]]:
    """Splits `papers` into (partition_A, partition_B) by citation_count,
    RELATIVE not threshold-based: papers WITH a known citation_count are
    sorted descending, Partition A = the top n of those, Partition B =
    everyone else. There is no fixed citation-count cutoff anywhere in
    this function — "top n" is defined purely by rank within this pool.

    Papers with citation_count=None (arXiv-only results Semantic Scholar
    never matched) are NOT eligible for Partition A at all — they land in
    Partition B unconditionally, never treated as citation_count=0 (which
    would make them the WORST-ranked eligible papers rather than simply
    ineligible; those are different claims, and only the latter is true —
    "we don't know this paper's citation count" is not "this paper has
    zero citations").

    If fewer than n papers have a citation_count at all, Partition A comes
    back smaller than n — it is NEVER force-filled with ineligible or
    weak candidates to reach n. Backfilling unfilled slots from Partition
    B is a decision for the caller (merge_with_guaranteed_slots() below),
    not something this function silently does on its own.

    Tie-breaking for equal citation_count is by paper_id (ascending) —
    deterministic and intrinsic to each paper, not dependent on `papers`'
    input order. This matters beyond single-call determinism: real
    arXiv/Semantic Scholar search result order is not itself guaranteed
    stable run-to-run, so relying on Python's stable-sort-preserves-input-
    order behavior alone would make ties resolve differently across
    separate real pipeline executions even though each individual sort
    call is internally stable. paper_id is this project's existing
    canonical per-paper identity key (dedup.py, Chroma filters, this
    module's own reciprocal_rank_fusion) — reused here for the same
    reason: it's already the established way to break a tie between two
    Paper records deterministically.
    """
    eligible = [p for p in papers if p.citation_count is not None]
    ineligible = [p for p in papers if p.citation_count is None]

    eligible_sorted = sorted(eligible, key=lambda p: (-p.citation_count, p.paper_id))

    partition_a = eligible_sorted[:n]
    partition_b = eligible_sorted[n:] + ineligible

    return partition_a, partition_b


def hybrid_search(
    query: str, papers: list[Paper], collection=None, client: OpenAI | None = None,
    top_k: int = 10, rrf_k: int = RRF_K,
) -> list[tuple[Paper, float]]:
    """Semantic + BM25, fused via RRF (see reciprocal_rank_fusion above).

    Both underlying rankings are computed over the FULL candidate pool
    (top_k = len(papers) for each), not independently truncated, per
    reciprocal_rank_fusion's own requirement — the fused ranking is what
    gets cut to the caller's requested top_k, once, at the end.

    `collection` must already have `papers` embedded and indexed (same
    precondition semantic_search() itself has — this function doesn't
    embed anything new); `query` must be the original topic text, never an
    LLM-suggested title, same anti-hallucination anchor as every other
    ranking mode in this module.
    """
    if not papers:
        return []

    ids = [p.paper_id for p in papers]
    semantic_full = semantic_search(
        query, collection=collection, client=client, top_k=len(papers),
        where={"paper_id": {"$in": ids}},
    )
    bm25_full = bm25_search(query, papers, top_k=len(papers))

    return reciprocal_rank_fusion([semantic_full, bm25_full], k=rrf_k, top_k=top_k)


def merge_with_guaranteed_slots(
    query: str, partition_a: list[Paper], partition_b: list[Paper], n: int,
    collection=None, client: OpenAI | None = None, top_k: int = 10,
) -> list[tuple[Paper, float]]:
    """Semantically scores every paper in BOTH partitions against `query`
    (embeddings.py's semantic_search(), never reimplemented), then builds
    a final top-k that (1) is ordered by semantic score — NOT partition_a-
    then-partition_b stacking — while (2) guaranteeing at least
    min(n, len(partition_a)) of those top-k slots are partition_a members.
    The min() is the relaxation Phase 1's partition_by_citation() fallback
    requires: if partition_a has fewer than n eligible papers at all, the
    guarantee quietly becomes "as many as exist," never an error and never
    a demand partition_b can't satisfy.

    Algorithm: rank the WHOLE pool once by semantic score. The best
    `required` partition_a papers BY THAT SAME SCORE are the guaranteed
    set — not arbitrarily chosen, not re-scored some other way. Fill the
    remaining top_k slots by walking the full ranking in score order,
    skipping anything already guaranteed. Finally, re-sort the assembled
    set by semantic score for the actual output order. When partition_a
    already has >= `required` members ranking within the natural top-k on
    merit alone, this provably produces the exact same result as plain
    top-k semantic ranking would have (the guaranteed set is necessarily
    already a subset of the natural top-k in that case) — the guarantee
    only changes anything when it needs to.

    `collection` must already have every paper in partition_a + partition_b
    embedded and indexed (same precondition as hybrid_search() above);
    `query` must be the original topic text, never an LLM-suggested title.
    """
    all_papers = partition_a + partition_b
    if not all_papers:
        return []

    a_ids = {p.paper_id for p in partition_a}
    ids = [p.paper_id for p in all_papers]

    full_ranked = semantic_search(
        query, collection=collection, client=client, top_k=len(all_papers),
        where={"paper_id": {"$in": ids}},
    )

    required = min(n, len(partition_a))

    a_ranked_full = [item for item in full_ranked if item[0].paper_id in a_ids]
    guaranteed = a_ranked_full[:required]
    guaranteed_ids = {item[0].paper_id for item in guaranteed}

    final_items = list(guaranteed)
    for item in full_ranked:
        if len(final_items) >= top_k:
            break
        if item[0].paper_id in guaranteed_ids:
            continue
        final_items.append(item)

    final_items.sort(key=lambda item: item[1], reverse=True)
    return final_items[:top_k]
