"""BM25 and RRF fusion tests — deterministic, no network/API calls at all.

Unlike embeddings.py's semantic_search() (which needs a mocked OpenAI
client for its embedding call), bm25_search() and reciprocal_rank_fusion()
are pure in-memory text/rank computation with no external dependency to
mock — these tests exercise the REAL algorithm, not a stand-in for it.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.ranking import (
    bm25_search,
    get_partition_n,
    merge_with_guaranteed_slots,
    partition_by_citation,
    reciprocal_rank_fusion,
)
from research_agent.schema import Paper


def _paper(paper_id: str, title: str, abstract: str, citation_count: int | None = None) -> Paper:
    return Paper(
        title=title, authors=[], year=None, venue=None, abstract=abstract,
        url=None, doi=None, citation_count=citation_count, source="test", paper_id=paper_id,
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


# --- partition_by_citation ---------------------------------------------------

def test_partition_by_citation_basic_ranking():
    high = _paper("high", "High", "x", citation_count=500)
    mid = _paper("mid", "Mid", "x", citation_count=100)
    low = _paper("low", "Low", "x", citation_count=10)
    a, b = partition_by_citation([low, high, mid], n=2)

    assert [p.paper_id for p in a] == ["high", "mid"]
    assert [p.paper_id for p in b] == ["low"]


def test_partition_by_citation_none_values_never_enter_partition_a():
    # A None-citation paper must never be treated as citation_count=0 —
    # it should land in B regardless of how it compares to A's members,
    # even when A isn't full.
    no_count = _paper("no_count", "NoCount", "x", citation_count=None)
    low_but_real = _paper("low_real", "LowReal", "x", citation_count=1)
    a, b = partition_by_citation([no_count, low_but_real], n=5)

    assert [p.paper_id for p in a] == ["low_real"]
    assert "no_count" not in [p.paper_id for p in a]
    assert "no_count" in [p.paper_id for p in b]


def test_partition_by_citation_fewer_eligible_than_n_does_not_force_fill_or_crash():
    # Only 2 papers have a real citation_count at all; n=5 asks for more
    # than exist. Partition A must come back with exactly those 2 — never
    # padded with the None-citation papers to reach 5, and never raising.
    eligible_1 = _paper("e1", "E1", "x", citation_count=50)
    eligible_2 = _paper("e2", "E2", "x", citation_count=20)
    no_count_1 = _paper("nc1", "NC1", "x", citation_count=None)
    no_count_2 = _paper("nc2", "NC2", "x", citation_count=None)

    a, b = partition_by_citation([no_count_1, eligible_1, no_count_2, eligible_2], n=5)

    assert len(a) == 2
    assert {p.paper_id for p in a} == {"e1", "e2"}
    assert {p.paper_id for p in b} == {"nc1", "nc2"}


def test_partition_by_citation_empty_pool_does_not_crash():
    a, b = partition_by_citation([], n=5)
    assert a == []
    assert b == []


def test_partition_by_citation_ties_break_deterministically_by_paper_id():
    # Three papers tied at citation_count=100 — tie-break must be by
    # paper_id (ascending), not by input list order, so the SAME logical
    # pool produces the SAME partition regardless of what order the
    # papers happened to arrive in (e.g. from a real API response whose
    # ordering isn't itself guaranteed stable run-to-run).
    tied_z = _paper("z_paper", "Z", "x", citation_count=100)
    tied_a = _paper("a_paper", "A", "x", citation_count=100)
    tied_m = _paper("m_paper", "M", "x", citation_count=100)

    order1_a, order1_b = partition_by_citation([tied_z, tied_a, tied_m], n=2)
    order2_a, order2_b = partition_by_citation([tied_m, tied_z, tied_a], n=2)
    order3_a, order3_b = partition_by_citation([tied_a, tied_m, tied_z], n=2)

    expected_a_ids = ["a_paper", "m_paper"]  # ascending paper_id, top 2
    assert [p.paper_id for p in order1_a] == expected_a_ids
    assert [p.paper_id for p in order2_a] == expected_a_ids
    assert [p.paper_id for p in order3_a] == expected_a_ids
    assert [p.paper_id for p in order1_b] == [p.paper_id for p in order2_b] == [p.paper_id for p in order3_b]


# --- merge_with_guaranteed_slots ---------------------------------------------
#
# semantic_search() needs a real embedding client + Chroma collection, so
# these tests patch research_agent.ranking.semantic_search directly (the
# name as looked up in ranking.py's own namespace, not embeddings.py's) to
# return a fully controlled, deterministic ranking — the point of these
# tests is the MERGE logic, not semantic_search() itself (already covered
# by test_embeddings.py).

def _fake_semantic_search_returning(scored: list[tuple[Paper, float]]):
    """Builds a stand-in for semantic_search() that ignores the real
    query/collection/client and just returns `scored`, sorted descending
    and cut to whatever top_k the caller asks for — exactly semantic_
    search()'s own return contract, just backed by fixed test data."""
    def fake(query, collection=None, client=None, top_k=10, where=None):
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        return ranked[:top_k]
    return fake


def test_merge_pulls_in_a_paper_that_ranks_outside_topk_on_merit_alone():
    # a_foundational is the ONLY Partition A member, and it scores dead
    # last semantically (0.10) — well outside top_k=5 among 10 total
    # papers. This is exactly the diagnosed real-world failure mode: a
    # foundational paper losing rerank against generic ones.
    a_foundational = _paper("a_found", "Foundational", "x", citation_count=1000)
    b_papers = [_paper(f"b{i}", f"B{i}", "x") for i in range(9)]
    scored = [(p, 0.90 - 0.05 * i) for i, p in enumerate(b_papers)]  # 0.90 down to 0.50
    scored.append((a_foundational, 0.10))

    with patch("research_agent.ranking.semantic_search", side_effect=_fake_semantic_search_returning(scored)):
        result = merge_with_guaranteed_slots(
            "topic", partition_a=[a_foundational], partition_b=b_papers, n=1, top_k=5,
        )

    result_ids = [p.paper_id for p, _ in result]
    assert len(result) == 5
    assert "a_found" in result_ids
    # Ordered by semantic score, not promoted to the front just because
    # the guarantee is why it's present at all — it lands at its own real
    # (low) score position within the chosen set, last among the 5.
    assert result_ids[-1] == "a_found"


def test_merge_does_not_disturb_result_when_a_already_ranks_well_on_merit():
    # a_strong is Partition A AND already scores highest overall — the
    # guarantee (n=1) is already naturally satisfied. Result must be
    # identical to what plain top-k semantic ranking would have produced;
    # the merge logic shouldn't perturb an already-satisfied guarantee.
    a_strong = _paper("a_strong", "Strong", "x", citation_count=1000)
    b_papers = [_paper(f"b{i}", f"B{i}", "x") for i in range(9)]
    scored = [(a_strong, 0.95)] + [(p, 0.90 - 0.05 * i) for i, p in enumerate(b_papers)]

    with patch("research_agent.ranking.semantic_search", side_effect=_fake_semantic_search_returning(scored)):
        result = merge_with_guaranteed_slots(
            "topic", partition_a=[a_strong], partition_b=b_papers, n=1, top_k=5,
        )

    expected_ids = [p.paper_id for p, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:5]]
    assert [p.paper_id for p, _ in result] == expected_ids
    assert result[0][0].paper_id == "a_strong"


def test_merge_relaxes_guarantee_when_partition_a_smaller_than_n():
    # Only 1 real Partition A member exists, but n=3 asks for more than
    # that. The guarantee must relax to "1" (not error, not force-fill
    # with partition_b papers relabeled as A) — min(n, len(partition_a)).
    a_only = _paper("a_only", "OnlyA", "x", citation_count=1000)
    b_papers = [_paper(f"b{i}", f"B{i}", "x") for i in range(9)]
    scored = [(p, 0.90 - 0.05 * i) for i, p in enumerate(b_papers)]
    scored.append((a_only, 0.10))

    with patch("research_agent.ranking.semantic_search", side_effect=_fake_semantic_search_returning(scored)):
        result = merge_with_guaranteed_slots(
            "topic", partition_a=[a_only], partition_b=b_papers, n=3, top_k=5,
        )

    assert len(result) == 5
    assert "a_only" in [p.paper_id for p, _ in result]


def test_merge_empty_pool_does_not_crash():
    with patch("research_agent.ranking.semantic_search", side_effect=_fake_semantic_search_returning([])):
        result = merge_with_guaranteed_slots("topic", partition_a=[], partition_b=[], n=3, top_k=5)
    assert result == []


# --- get_partition_n ---------------------------------------------------------

def test_get_partition_n_is_flat_2_across_the_real_production_k_range():
    # api.py's SearchRequest.top_k is ge=3, le=30 — the actual range a real
    # user can select. The derived rule is a flat constant (not a function
    # of k) across that whole range, confirmed via real k-generalization
    # testing at k=3,5,10,20,25,30 — this test locks that in.
    for k in [3, 5, 10, 20, 25, 30]:
        assert get_partition_n(k) == 2


def test_get_partition_n_clamps_defensively_below_2():
    # k<2 never occurs in this project's real bounds, but the clamp is a
    # defensive floor, not a finding — a caller passing an out-of-range k
    # should get a valid n back, never one exceeding the pool size it's
    # drawn from.
    assert get_partition_n(1) == 1
    assert get_partition_n(0) == 0
