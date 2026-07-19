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
