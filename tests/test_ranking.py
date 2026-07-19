"""BM25 and RRF fusion tests — deterministic, no network/API calls at all.

Unlike embeddings.py's semantic_search() (which needs a mocked OpenAI
client for its embedding call), bm25_search() and reciprocal_rank_fusion()
are pure in-memory text/rank computation with no external dependency to
mock — these tests exercise the REAL algorithm, not a stand-in for it.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.ranking import bm25_search, reciprocal_rank_fusion
from research_agent.schema import Paper


def _paper(paper_id: str, title: str, abstract: str) -> Paper:
    return Paper(
        title=title, authors=[], year=None, venue=None, abstract=abstract,
        url=None, doi=None, citation_count=None, source="test", paper_id=paper_id,
    )


# --- bm25_search ----------------------------------------------------------

def test_bm25_search_ranks_exact_term_overlap_above_unrelated_paper():
    # 3+ documents deliberately, not 2: with a 2-document corpus, a term
    # appearing in exactly one of the two documents sits at BM25's classic
    # idf zero-crossing (idf = log((N-n+0.5)/(n+0.5)) = log(1) = 0 when
    # n = N/2), which would silently zero out the very signal this test
    # means to check. A third (also off-topic) filler paper avoids that
    # degenerate small-N edge case without changing what's being tested.
    query = "parameter-efficient fine-tuning for large language models"
    on_topic = _paper(
        "p1", "RoCoFT",
        "RoCoFT is a parameter-efficient fine-tuning method for large language models "
        "based on updating rows and columns of weight matrices.",
    )
    off_topic = _paper(
        "p2", "Coral reef biodiversity",
        "This paper surveys coral reef biodiversity decline across the Pacific Ocean.",
    )
    off_topic_filler = _paper(
        "p3", "Bee pollinator monitoring",
        "This paper studies bee pollinator population monitoring using camera traps.",
    )
    ranked = bm25_search(query, [off_topic, on_topic, off_topic_filler], top_k=10)

    assert ranked[0][0].paper_id == "p1"
    # BM25 score is unbounded raw score, not a [0,1] similarity — just
    # confirm the winner actually scores higher, not any particular scale.
    assert ranked[0][1] > ranked[1][1]


def test_bm25_search_respects_top_k():
    query = "retrieval augmented generation"
    papers = [_paper(f"p{i}", f"Paper {i}", f"retrieval augmented generation paper number {i}") for i in range(5)]
    ranked = bm25_search(query, papers, top_k=2)
    assert len(ranked) == 2


def test_bm25_search_empty_pool_returns_empty():
    assert bm25_search("anything", [], top_k=10) == []


def test_bm25_search_falls_back_to_title_when_abstract_missing():
    # _document_text prefers abstract, falls back to title — a paper with
    # no abstract should still be scorable (and matchable) via its title.
    # 3 documents for the same reason as the test above (avoid the N=2
    # idf zero-crossing masking the very signal being tested).
    query = "RoCoFT row column updates"
    title_only = _paper("p1", "RoCoFT: Row-Column Updates for Fine-Tuning", None)
    unrelated = _paper("p2", "Coral reefs", "coral reef biodiversity decline")
    unrelated_filler = _paper("p3", "Bee monitoring", "bee pollinator population monitoring")
    ranked = bm25_search(query, [unrelated, title_only, unrelated_filler], top_k=10)
    assert ranked[0][0].paper_id == "p1"


# --- reciprocal_rank_fusion -------------------------------------------------

def test_rrf_ranks_paper_agreed_on_by_both_methods_first():
    # p1 is #1 in both rankings — should come out on top of the fusion.
    p1, p2, p3 = _paper("p1", "A", "a"), _paper("p2", "B", "b"), _paper("p3", "C", "c")
    semantic_ranking = [(p1, 0.9), (p2, 0.8), (p3, 0.7)]
    bm25_ranking = [(p1, 5.0), (p2, 3.0), (p3, 1.0)]

    fused = reciprocal_rank_fusion([semantic_ranking, bm25_ranking], k=60, top_k=3)

    assert fused[0][0].paper_id == "p1"


def test_rrf_disagreement_lands_in_between_not_at_either_extreme():
    # p_agree: rank 1 in both methods (strongest possible signal).
    # p_disagree: rank 1 in semantic, rank 4 (last) in BM25 — real
    # disagreement between the two methods.
    # p_last_both: rank 4 (last) in both methods (weakest possible signal).
    # p_disagree should land STRICTLY between p_agree and p_last_both in
    # the fused ranking — not tied with the top, and not tied with the
    # bottom, which is the concrete, checkable form of "landing in between
    # rather than at either extreme."
    p_agree = _paper("agree", "Agree", "x")
    p_disagree = _paper("disagree", "Disagree", "x")
    p_last_both = _paper("last_both", "LastBoth", "x")
    p_filler = _paper("filler", "Filler", "x")

    semantic_ranking = [
        (p_agree, 0.99), (p_disagree, 0.98), (p_filler, 0.5), (p_last_both, 0.1),
    ]
    bm25_ranking = [
        (p_agree, 9.0), (p_filler, 5.0), (p_disagree, 2.0), (p_last_both, 1.0),
    ]

    fused = reciprocal_rank_fusion([semantic_ranking, bm25_ranking], k=60, top_k=4)
    fused_order = [p.paper_id for p, _ in fused]

    agree_score = [s for p, s in fused if p.paper_id == "agree"][0]
    disagree_score = [s for p, s in fused if p.paper_id == "disagree"][0]
    last_both_score = [s for p, s in fused if p.paper_id == "last_both"][0]

    assert agree_score > disagree_score > last_both_score
    assert fused_order[0] == "agree"
    assert fused_order[-1] == "last_both"


def test_rrf_paper_missing_from_one_ranking_still_scores_from_the_other():
    # p2 doesn't appear in the BM25 ranking at all (e.g. filtered out
    # upstream) — RRF should still credit it for its semantic rank rather
    # than treating absence-from-one-list as a hard zero overall.
    p1, p2 = _paper("p1", "A", "a"), _paper("p2", "B", "b")
    semantic_ranking = [(p1, 0.9), (p2, 0.8)]
    bm25_ranking = [(p1, 5.0)]  # p2 absent

    fused = reciprocal_rank_fusion([semantic_ranking, bm25_ranking], k=60, top_k=2)
    fused_ids = [p.paper_id for p, _ in fused]

    assert "p2" in fused_ids
    # p1 (present, agreed-upon rank 1) should still outrank p2 (present in
    # only one list).
    assert fused_ids[0] == "p1"


def test_rrf_respects_top_k():
    papers = [_paper(f"p{i}", f"P{i}", "x") for i in range(5)]
    ranking = [(p, 1.0 / (i + 1)) for i, p in enumerate(papers)]
    fused = reciprocal_rank_fusion([ranking], k=60, top_k=2)
    assert len(fused) == 2
