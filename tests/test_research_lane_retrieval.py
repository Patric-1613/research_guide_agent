"""Research Lanes (RL3/RL3a/RL4): tests for
research_agent.research_lane_retrieval -- retrieve_across_lanes (first
search) and refill_lane_session (multi-lane refill).

Pure domain layer -- every network/embedding/Chroma boundary is mocked at
the module seam (build_candidate_pool / embed_and_index_papers /
semantic_search / get_chroma_collection). deduplicate_with_clusters and
extract_keywords run FOR REAL (the cross-lane dedup + provenance +
keyword-repair logic is exactly what this module owns). No network /
provider calls. The real checkpoint / usage / Chroma DBs are
fingerprinted (module level) and re-checked at the end.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import research_agent.research_lane_retrieval as rlr
import research_agent.telemetry as telemetry
from research_agent.qa import QA_CHECKPOINT_DB_PATH
from research_agent.query_expansion import BATCH_SIZE, PaperPoolSession, _EMPTY_EMBED_STATS
from research_agent.research_lane_retrieval import MultiLaneRetrievalResult, retrieve_across_lanes
from research_agent.research_lanes import ResearchLane, new_lane_id
from research_agent.schema import Paper
from tests._usage_db_fingerprint import fingerprint_usage_db

_REAL_CHECKPOINT_DB = QA_CHECKPOINT_DB_PATH
_REAL_USAGE_DB = telemetry.USAGE_DB_PATH
_REAL_CHROMA_DB = Path("data/chroma_db/chroma.sqlite3")
_CHECKPOINT_FP_BEFORE = fingerprint_usage_db(_REAL_CHECKPOINT_DB)
_USAGE_FP_BEFORE = fingerprint_usage_db(_REAL_USAGE_DB)

_ABSTRACT = (
    "Retrieval augmented generation grounds a language model in external "
    "documents to reduce hallucination and improve factual accuracy on "
    "knowledge intensive question answering benchmarks."
)


def _lane(query: str, *, lane_id: str | None = None, enabled: bool = True, label: str | None = None) -> ResearchLane:
    return ResearchLane(
        lane_id=lane_id or new_lane_id(),
        label=label or f"The {query} facet",  # kept distinct from any short lane_id
        question=f"what about {query}?",
        query=query,
        enabled=enabled,
        origin="suggested",
        generation_version=1,
    )


_DISTINCT_TITLES = [
    "Dense Passage Retrieval for Open Domain Question Answering",
    "Fusion in Decoder for Knowledge Grounded Generation",
    "Self Consistency Decoding Improves Chain of Thought",
    "Contrastive Pretraining for Sentence Embeddings",
    "Adaptive Retrieval Confidence Calibration in Language Models",
    "Hallucination Detection via Token Level Uncertainty",
    "Long Context Windows and Positional Interpolation",
    "Reranking Candidate Passages with Cross Encoders",
    "Instruction Tuning for Factual Question Answering",
    "Graph Structured Evidence Aggregation for Claims",
    "Sparse Mixture of Experts for Efficient Inference",
    "Multi Hop Reasoning over Knowledge Bases",
]


def _paper(pid: str, title: str, *, doi: str | None = None, abstract: str | None = _ABSTRACT,
           keywords: list[str] | None = None) -> Paper:
    p = Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="V", abstract=abstract,
        url=f"http://arxiv.org/abs/{pid}", doi=doi, citation_count=None, source="arxiv", paper_id=pid,
    )
    # mimic build_candidate_pool's own once-per-deduped-paper keyword step
    p.keywords = keywords if keywords is not None else ["retrieval augmented generation", "hallucination"]
    return p


class _Fakes:
    """Records every mocked boundary call and serves deterministic
    results. `pools` maps a lane query -> the list of Papers that lane's
    build_candidate_pool returns. `rank_key` optionally maps paper_id ->
    sort key (lower ranks first) for semantic_search; default keeps the
    where-list order."""

    def __init__(self, pools: dict[str, list[Paper]], *, rank_key: dict[str, int] | None = None,
                 score_by_id: dict[str, float] | None = None):
        self.pools = pools
        self.rank_key = rank_key
        self.score_by_id = score_by_id
        self.build_calls: list[dict] = []
        self.embed_calls: list[list[Paper]] = []
        self.search_calls: list[dict] = []
        self._indexed: dict[str, Paper] = {}

    # --- build_candidate_pool(topic_or_query, k, *, ...) ---
    def build(self, query, k, **kwargs):
        self.build_calls.append({"query": query, "k": k, "kwargs": kwargs})
        if query not in self.pools:
            raise AssertionError(f"unexpected build_candidate_pool query: {query!r}")
        # fresh copies so each call yields distinct Paper objects (like the real fn)
        return [copy.deepcopy(p) for p in self.pools[query]]

    # --- embed_and_index_papers(papers, *, collection, client) ---
    def embed(self, papers, *, collection=None, client=None):
        self.embed_calls.append(list(papers))
        for p in papers:
            self._indexed[p.paper_id] = p
        return {"cache_hits": len(papers), "cache_misses": 0, "tokens_billed": 0,
                "estimated_cost_usd": 0.0, "papers_skipped": 0}

    # --- semantic_search(query, *, collection, client, top_k, where) ---
    def search(self, query, *, collection=None, client=None, top_k=10, where=None):
        ids = list(where["paper_id"]["$in"])
        self.search_calls.append({"query": query, "top_k": top_k, "ids": ids})
        picked = [self._indexed[i] for i in ids if i in self._indexed]
        if self.rank_key is not None:
            picked = sorted(picked, key=lambda p: self.rank_key.get(p.paper_id, 10_000))
        if self.score_by_id is not None:
            return [(p, self.score_by_id[p.paper_id]) for p in picked]
        return [(p, 1.0 - 0.001 * n) for n, p in enumerate(picked)]


def _run(topic: str, lanes: list[ResearchLane], fakes: _Fakes, **kwargs) -> MultiLaneRetrievalResult:
    with patch.object(rlr, "build_candidate_pool", side_effect=fakes.build), \
         patch.object(rlr, "embed_and_index_papers", side_effect=fakes.embed), \
         patch.object(rlr, "semantic_search", side_effect=fakes.search), \
         patch.object(rlr, "get_chroma_collection", side_effect=AssertionError("collection must be injected in tests")):
        return retrieve_across_lanes(topic, lanes, collection=object(), **kwargs)


# --- 1-3: validation happens before any work ---------------------------

def test_disabled_lanes_are_ignored_entirely_zero_calls_for_them():
    a, b_off, c = _lane("methods", lane_id="A"), _lane("evaluation", lane_id="B", enabled=False), _lane("risks", lane_id="C")
    fakes = _Fakes({"methods": [_paper("m1", "M1")], "risks": [_paper("r1", "R1")]})
    result = _run("topic", [a, b_off, c], fakes)
    assert [c["query"] for c in fakes.build_calls] == ["methods", "risks"]  # never "evaluation"
    assert "B" not in result.paper_lane_ids.get("m1", []) + result.paper_lane_ids.get("r1", [])
    assert set(result.lane_result_counts) == {"A", "C"}


def test_no_enabled_lanes_rejected_before_any_work():
    fakes = _Fakes({})
    with patch.object(rlr, "build_candidate_pool", side_effect=AssertionError("no work")):
        with pytest.raises(ValueError, match="at least one enabled"):
            retrieve_across_lanes("t", [_lane("x", enabled=False)], collection=object())
    assert fakes.build_calls == []


def test_more_than_max_enabled_lanes_rejected_before_any_work():
    from research_agent.research_lanes import MAX_LANES_PER_REVIEW

    lanes = [_lane(f"q{i}", lane_id=f"L{i}") for i in range(MAX_LANES_PER_REVIEW + 1)]
    with patch.object(rlr, "build_candidate_pool", side_effect=AssertionError("no work")), \
         patch.object(rlr, "embed_and_index_papers", side_effect=AssertionError("no work")):
        with pytest.raises(ValueError, match="at most"):
            retrieve_across_lanes("t", lanes, collection=object())


def test_exactly_max_enabled_lanes_is_accepted():
    from research_agent.research_lanes import MAX_LANES_PER_REVIEW

    lanes = [_lane(f"q{i}", lane_id=f"L{i}") for i in range(MAX_LANES_PER_REVIEW)]
    fakes = _Fakes({f"q{i}": [_paper(f"p{i}", f"P{i}")] for i in range(MAX_LANES_PER_REVIEW)})
    result = _run("t", lanes, fakes)
    assert len(fakes.build_calls) == MAX_LANES_PER_REVIEW


def test_lane_objects_validated_through_rl1_contract():
    # a lane whose lane_id equals its label fails validate_lane_for_construction
    bad = ResearchLane(lane_id="Methods", label="Methods", question="q", query="methods")
    with patch.object(rlr, "build_candidate_pool", side_effect=AssertionError("no work")):
        with pytest.raises(ValueError):
            retrieve_across_lanes("t", [bad], collection=object())


# --- 4-8: per-lane retrieval call bounds / query / order --------------

def test_exactly_one_build_candidate_pool_call_per_enabled_lane_with_its_own_query():
    lanes = [_lane("methods", lane_id="A"), _lane("evaluation", lane_id="B"), _lane("risks", lane_id="C")]
    fakes = _Fakes({q: [_paper(f"{q}1", q)] for q in ("methods", "evaluation", "risks")})
    _run("the topic", lanes, fakes)
    assert len(fakes.build_calls) == 3
    assert [c["query"] for c in fakes.build_calls] == ["methods", "evaluation", "risks"]  # exact query, frozen order
    assert all(c["k"] == BATCH_SIZE for c in fakes.build_calls)


def test_k_for_widening_is_the_same_for_every_lane_and_overridable():
    lanes = [_lane("a", lane_id="A"), _lane("b", lane_id="B")]
    fakes = _Fakes({"a": [_paper("a1", "A1")], "b": [_paper("b1", "B1")]})
    _run("t", lanes, fakes, k_for_widening=7)
    assert [c["k"] for c in fakes.build_calls] == [7, 7]


def test_optional_params_forwarded_to_build_candidate_pool():
    lanes = [_lane("a", lane_id="A")]
    fakes = _Fakes({"a": [_paper("a1", "A1")]})
    _run("t", lanes, fakes, s2_api_key="k", use_openalex_fallback=True, openalex_mailto="m@x",
         exclude_titles=["Old Title"], refinement_notes=["recent work"])
    kw = fakes.build_calls[0]["kwargs"]
    assert kw["s2_api_key"] == "k" and kw["use_openalex_fallback"] is True and kw["openalex_mailto"] == "m@x"
    assert kw["exclude_titles"] == ["Old Title"] and kw["refinement_notes"] == ["recent work"]


def test_original_topic_and_facet_anchor_every_lane_ranking_query():
    lanes = [_lane("evaluation methods", lane_id="A"), _lane("failure modes", lane_id="B")]
    fakes = _Fakes({"evaluation methods": [_paper("e1", "E1")], "failure modes": [_paper("f1", "F1")]})
    _run("reducing hallucination in RAG", lanes, fakes)
    ranking_queries = [c["query"] for c in fakes.search_calls]
    assert ranking_queries == [
        "reducing hallucination in RAG\nResearch facet: evaluation methods",
        "reducing hallucination in RAG\nResearch facet: failure modes",
    ]
    for q in ranking_queries:
        assert "reducing hallucination in RAG" in q


def test_refinement_notes_appended_to_lane_ranking_query_like_single_query_refill():
    lanes = [_lane("methods", lane_id="A")]
    fakes = _Fakes({"methods": [_paper("m1", "M1")]})
    _run("topic here", lanes, fakes, refinement_notes=["focus on recent work", "prefer surveys"])
    assert fakes.search_calls[0]["query"] == (
        "topic here\nResearch facet: methods. focus on recent work prefer surveys"
    )


# --- 9-13: cross-lane dedup + provenance + keywords -------------------

def test_global_doi_duplicate_merges_once_with_both_lane_ids():
    lanes = [_lane("methods", lane_id="A"), _lane("evaluation", lane_id="B")]
    pa = _paper("arxiv:1", "Retrieval Augmented Generation", doi="10.1/shared")
    pb = _paper("s2:1", "A Completely Different Sounding Title", doi="10.1/shared")
    fakes = _Fakes({"methods": [pa], "evaluation": [pb]})
    result = _run("t", lanes, fakes)

    merged_ids = list(result.paper_lane_ids)
    assert len(merged_ids) == 1
    assert result.paper_lane_ids[merged_ids[0]] == ["A", "B"]  # enabled-lane order
    assert len(result.ranked) == 1
    assert result.lane_result_counts == {"A": 1, "B": 1}       # shared paper counts for both


def test_global_fuzzy_title_duplicate_merges_once():
    lanes = [_lane("methods", lane_id="A"), _lane("evaluation", lane_id="B")]
    pa = _paper("arxiv:2", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks")
    pb = _paper("s2:2", "Retrieval Augmented Generation for Knowledge Intensive NLP Tasks")
    fakes = _Fakes({"methods": [pa], "evaluation": [pb]})
    result = _run("t", lanes, fakes)
    assert len(result.paper_lane_ids) == 1
    assert list(result.paper_lane_ids.values())[0] == ["A", "B"]


def test_multi_lane_duplicate_gets_all_lane_ids_in_enabled_lane_order():
    # discovered by C first, then A -- provenance must still be [A, C]
    lanes = [_lane("aa", lane_id="A"), _lane("bb", lane_id="B"), _lane("cc", lane_id="C")]
    shared = lambda: _paper("shared", "Retrieval Augmented Generation", doi="10.9/x")  # noqa: E731
    fakes = _Fakes({"aa": [shared()], "bb": [_paper("b-only", "B Only Title")], "cc": [shared()]})
    result = _run("t", lanes, fakes)
    assert result.paper_lane_ids["shared"] == ["A", "C"]
    assert result.paper_lane_ids["b-only"] == ["B"]
    assert result.lane_result_counts == {"A": 1, "B": 1, "C": 1}


def test_existing_deduplicate_behavior_is_unchanged_by_the_new_helper():
    from research_agent.dedup import deduplicate, deduplicate_with_clusters

    papers = [
        _paper("a", "Retrieval-Augmented Generation for X"),
        _paper("b", "Retrieval Augmented Generation for X"),
        _paper("c", "Something Else Entirely"),
    ]
    plain = deduplicate([copy.deepcopy(p) for p in papers])
    withc = [m for m, _ in deduplicate_with_clusters([copy.deepcopy(p) for p in papers])]
    assert [p.to_dict() for p in withc] == [p.to_dict() for p in plain]


def test_cross_lane_merged_paper_does_not_lose_its_keywords():
    """Regression: _merge_cluster() builds a fresh Paper with keywords=[]
    for a >1-member cluster. RL3 must recompute deterministic YAKE-v2
    keywords so the merged paper is not silently keyword-less."""
    lanes = [_lane("methods", lane_id="A"), _lane("evaluation", lane_id="B")]
    pa = _paper("arxiv:k", "Retrieval Augmented Generation", doi="10.5/k", abstract=_ABSTRACT)
    pb = _paper("s2:k", "Grounding Language Models In Documents", doi="10.5/k", abstract=_ABSTRACT)
    fakes = _Fakes({"methods": [pa], "evaluation": [pb]})
    result = _run("t", lanes, fakes)

    (merged_paper, _score), = result.ranked
    assert merged_paper.keywords, "cross-lane merged paper lost its keywords"
    # deterministic + matches direct YAKE-v2 on the merged text
    from research_agent.keywords import extract_keywords
    assert merged_paper.keywords == extract_keywords(merged_paper.title, merged_paper.abstract)


def test_singleton_paper_keeps_its_original_keywords_untouched():
    lanes = [_lane("methods", lane_id="A")]
    p = _paper("solo", "A Solo Paper", keywords=["custom keyword one", "custom keyword two"])
    fakes = _Fakes({"methods": [p]})
    result = _run("t", lanes, fakes)
    (paper, _), = result.ranked
    assert paper.keywords == ["custom keyword one", "custom keyword two"]


# --- 14-16: single global embedding + scoped per-lane ranking --------

def test_global_pool_is_embedded_exactly_once():
    lanes = [_lane("a", lane_id="A"), _lane("b", lane_id="B"), _lane("c", lane_id="C")]
    fakes = _Fakes({q: [_paper(f"{q}1", q), _paper(f"{q}2", q + " two")] for q in ("a", "b", "c")})
    _run("t", lanes, fakes)
    assert len(fakes.embed_calls) == 1
    embedded_ids = sorted(p.paper_id for p in fakes.embed_calls[0])
    assert embedded_ids == ["a1", "a2", "b1", "b2", "c1", "c2"]


def test_each_semantic_search_is_restricted_to_its_own_lane_ids_and_requests_full_ranking():
    lanes = [_lane("a", lane_id="A"), _lane("b", lane_id="B")]
    fakes = _Fakes({
        "a": [_paper("a1", "A1"), _paper("a2", "A2"), _paper("a3", "A3")],
        "b": [_paper("b1", "B1")],
    })
    _run("t", lanes, fakes)
    by_query = {c["query"].splitlines()[-1]: c for c in fakes.search_calls}
    assert sorted(by_query["Research facet: a"]["ids"]) == ["a1", "a2", "a3"]
    assert by_query["Research facet: a"]["top_k"] == 3          # full subset, no early truncation
    assert by_query["Research facet: b"]["ids"] == ["b1"]
    assert by_query["Research facet: b"]["top_k"] == 1


def test_the_global_pool_is_never_re_embedded_per_lane():
    lanes = [_lane("a", lane_id="A"), _lane("b", lane_id="B")]
    fakes = _Fakes({"a": [_paper("a1", "A1")], "b": [_paper("b1", "B1")]})
    _run("t", lanes, fakes)
    assert len(fakes.embed_calls) == 1  # not once-per-lane


# --- 17-21: interleaving ----------------------------------------------

def test_round_robin_for_balanced_lanes():
    lanes = [_lane("a", lane_id="A"), _lane("b", lane_id="B")]
    fakes = _Fakes(
        {"a": [_paper("a1", "A1"), _paper("a2", "A2"), _paper("a3", "A3")],
         "b": [_paper("b1", "B1"), _paper("b2", "B2"), _paper("b3", "B3")]},
        rank_key={"a1": 0, "a2": 1, "a3": 2, "b1": 0, "b2": 1, "b3": 2},
    )
    result = _run("t", lanes, fakes)
    assert [p.paper_id for p, _ in result.ranked] == ["a1", "b1", "a2", "b2", "a3", "b3"]


def test_early_exhausted_lane_does_not_stop_the_others():
    lanes = [_lane("a", lane_id="A"), _lane("b", lane_id="B")]
    fakes = _Fakes(
        {"a": [_paper("a1", "A1")],
         "b": [_paper("b1", "B1"), _paper("b2", "B2"), _paper("b3", "B3")]},
        rank_key={"a1": 0, "b1": 0, "b2": 1, "b3": 2},
    )
    result = _run("t", lanes, fakes)
    assert [p.paper_id for p, _ in result.ranked] == ["a1", "b1", "b2", "b3"]


def test_shared_top_paper_appears_once_and_interleaving_continues():
    lanes = [_lane("a", lane_id="A"), _lane("b", lane_id="B")]
    shared = lambda: _paper("shared", "Retrieval Augmented Generation", doi="10.7/s")  # noqa: E731
    fakes = _Fakes(
        {"a": [shared(), _paper("a2", "A2")],
         "b": [shared(), _paper("b2", "B2")]},
        rank_key={"shared": 0, "a2": 1, "b2": 1},
    )
    result = _run("t", lanes, fakes)
    ids = [p.paper_id for p, _ in result.ranked]
    assert ids.count("shared") == 1
    assert ids == ["shared", "b2", "a2"] or ids == ["shared", "a2", "b2"]
    # deterministic: lane A emits `shared` first (round 1), then lane B
    # (its `shared` already emitted) emits b2, then lane A emits a2
    assert ids == ["shared", "b2", "a2"]


def test_score_is_from_the_lane_that_first_emits_the_shared_paper():
    lanes = [_lane("a", lane_id="A"), _lane("b", lane_id="B")]
    shared = lambda: _paper("shared", "Retrieval Augmented Generation", doi="10.7/s2")  # noqa: E731
    fakes = _Fakes({"a": [shared()], "b": [shared()]}, rank_key={"shared": 0})
    result = _run("t", lanes, fakes)
    (paper, score), = result.ranked
    # lane A is first in enabled order -> its search result (score 1.0 for rank 0) wins
    assert score == pytest.approx(1.0)


def test_final_reserve_is_not_re_sorted_by_cross_lane_scores():
    # lane B's scores are strictly higher than lane A's -- a global
    # score-sort would put every B paper first. Round-robin + no-resort
    # must keep lane A's first paper first.
    lanes = [_lane("a", lane_id="A"), _lane("b", lane_id="B")]
    fakes = _Fakes(
        {"a": [_paper("a1", "A1"), _paper("a2", "A2")],
         "b": [_paper("b1", "B1"), _paper("b2", "B2")]},
        rank_key={"a1": 0, "a2": 1, "b1": 0, "b2": 1},
        score_by_id={"a1": 0.20, "a2": 0.10, "b1": 0.90, "b2": 0.80},
    )
    result = _run("t", lanes, fakes)
    ids = [p.paper_id for p, _ in result.ranked]
    assert ids == ["a1", "b1", "a2", "b2"]           # round-robin order preserved
    scores = [s for _, s in result.ranked]
    assert scores == [0.20, 0.90, 0.10, 0.80]        # each paper's own lane score
    assert scores != sorted(scores, reverse=True)    # NOT globally score-sorted


def test_lane_result_counts_includes_shared_papers_for_every_applicable_lane():
    lanes = [_lane("a", lane_id="A"), _lane("b", lane_id="B"), _lane("c", lane_id="C")]
    shared = lambda: _paper("shared", "Retrieval Augmented Generation", doi="10.3/x")  # noqa: E731
    fakes = _Fakes({
        "a": [shared(), _paper("a-only", "A Only")],
        "b": [shared()],
        "c": [_paper("c-only", "C Only Title")],
    })
    result = _run("t", lanes, fakes)
    assert result.lane_result_counts == {"A": 2, "B": 1, "C": 1}
    assert sum(1 for lids in result.paper_lane_ids.values() if "A" in lids) == 2


# --- 22-23: empty + failure -----------------------------------------

def test_all_lanes_empty_performs_no_embedding_or_ranking():
    lanes = [_lane("a", lane_id="A"), _lane("b", lane_id="B")]
    fakes = _Fakes({"a": [], "b": []})
    with patch.object(rlr, "build_candidate_pool", side_effect=fakes.build), \
         patch.object(rlr, "embed_and_index_papers", side_effect=AssertionError("no embedding on empty")), \
         patch.object(rlr, "semantic_search", side_effect=AssertionError("no ranking on empty")), \
         patch.object(rlr, "get_chroma_collection", side_effect=AssertionError("no chroma on empty")):
        result = retrieve_across_lanes("t", lanes, collection=object())
    assert result.ranked == []
    assert result.paper_lane_ids == {}
    assert result.lane_result_counts == {"A": 0, "B": 0}
    assert result.embed_stats == dict(_EMPTY_EMBED_STATS)


def test_one_failed_lane_propagates_and_is_not_silently_omitted():
    lanes = [_lane("ok", lane_id="A"), _lane("boom", lane_id="B"), _lane("never", lane_id="C")]

    def _build(query, k, **kwargs):
        if query == "boom":
            raise RuntimeError("semantic scholar exploded")
        return [_paper(f"{query}1", query)]

    with patch.object(rlr, "build_candidate_pool", side_effect=_build), \
         patch.object(rlr, "embed_and_index_papers", side_effect=AssertionError("should not reach embed")), \
         patch.object(rlr, "semantic_search", side_effect=AssertionError("should not reach ranking")):
        with pytest.raises(RuntimeError, match="exploded"):
            retrieve_across_lanes("t", lanes, collection=object())


def test_a_lane_with_zero_candidates_does_not_block_other_lanes():
    lanes = [_lane("empty", lane_id="A"), _lane("full", lane_id="B")]
    fakes = _Fakes({"empty": [], "full": [_paper("f1", "F1"), _paper("f2", "F2")]})
    result = _run("t", lanes, fakes)
    assert result.lane_result_counts == {"A": 0, "B": 2}
    assert [p.paper_id for p, _ in result.ranked] == ["f1", "f2"] or \
           sorted(p.paper_id for p, _ in result.ranked) == ["f1", "f2"]
    # lane A got no semantic_search call (zero candidates)
    assert all("Research facet: empty" not in c["query"] for c in fakes.search_calls)


# --- 24: no mutation of inputs -------------------------------------

def test_input_research_lane_list_and_session_are_not_mutated():
    lane_a = _lane("methods", lane_id="A", label="Methods")
    lane_b = _lane("evaluation", lane_id="B", enabled=False, label="Evaluation")
    session = PaperPoolSession(topic="my review", lanes=[lane_a, lane_b])

    lanes_snapshot = [copy.deepcopy(l).to_dict() for l in session.lanes]
    session_snapshot = {
        "lanes": [l.to_dict() for l in session.lanes],
        "paper_lane_ids": copy.deepcopy(session.paper_lane_ids),
        "lane_result_counts": copy.deepcopy(session.lane_result_counts),
        "topic": session.topic,
    }

    fakes = _Fakes({"methods": [_paper("m1", "M1")]})
    _run(session.topic, session.lanes, fakes)

    assert [l.to_dict() for l in session.lanes] == lanes_snapshot
    assert session.paper_lane_ids == session_snapshot["paper_lane_ids"] == {}
    assert session.lane_result_counts == session_snapshot["lane_result_counts"] == {}
    assert session.topic == "my review"


# --- 25-26: existing behavior + real DBs untouched ----------------

def test_existing_single_query_helpers_are_not_imported_shadowed_or_changed():
    import research_agent.query_expansion as qe

    # RL3 reuses these unchanged -- it must not rebind or wrap them
    assert qe.build_candidate_pool.__module__ == "research_agent.query_expansion"
    assert qe.rank_full_pool.__module__ == "research_agent.query_expansion"
    # RL3's own module references the SAME build_candidate_pool object
    assert rlr.build_candidate_pool is qe.build_candidate_pool


def test_rl3_never_constructs_a_real_chromadb_client_or_touches_state_dbs():
    """Test-isolation proof. _run() already mocks get_chroma_collection
    and passes an injected collection, so a full retrieve_across_lanes
    call here constructs no real chromadb.PersistentClient (hard tripwire)
    and leaves the real checkpoint / usage DBs byte-identical. The
    real-Chroma sqlite file is fingerprinted with a LOCAL before/after
    around this one call (not module level) so an unrelated earlier test
    file's own Chroma drift can never make this assertion flake."""
    import chromadb

    chroma_before = fingerprint_usage_db(_REAL_CHROMA_DB)
    with patch.object(
        chromadb, "PersistentClient",
        side_effect=AssertionError("RL3 tests must never construct a real chromadb.PersistentClient"),
    ) as persistent_client_spy:
        lanes = [_lane("a", lane_id="A"), _lane("b", lane_id="B")]
        fakes = _Fakes({"a": [_paper("a1", "A1")], "b": [_paper("b1", "B1")]})
        _run("t", lanes, fakes)
    persistent_client_spy.assert_not_called()

    assert fingerprint_usage_db(_REAL_CHROMA_DB) == chroma_before
    assert fingerprint_usage_db(_REAL_CHECKPOINT_DB) == _CHECKPOINT_FP_BEFORE
    assert fingerprint_usage_db(_REAL_USAGE_DB) == _USAGE_FP_BEFORE


# --- interleaving property-style (deterministic, parametrized) ------

@pytest.mark.parametrize("lane_sizes", [(1, 1), (3, 1), (1, 3), (2, 2, 2), (4, 0, 2), (5, 5)])
def test_interleave_emits_every_candidate_exactly_once_and_is_deterministic(lane_sizes):
    lanes = [_lane(f"q{i}", lane_id=f"L{i}") for i in range(len(lane_sizes))]
    title_iter = iter(_DISTINCT_TITLES)
    pools = {
        f"q{i}": [_paper(f"L{i}p{j}", next(title_iter)) for j in range(n)]
        for i, n in enumerate(lane_sizes)
    }
    fakes = _Fakes(pools)
    r1 = _run("t", lanes, fakes)
    fakes2 = _Fakes(pools)
    r2 = _run("t", lanes, fakes2)
    ids1 = [p.paper_id for p, _ in r1.ranked]
    assert ids1 == [p.paper_id for p, _ in r2.ranked]           # deterministic
    assert len(ids1) == len(set(ids1)) == sum(lane_sizes)       # each once, all emitted


# ======================================================================
# RL3a: shared-object provenance, item-type validation, cluster-only
# keyword recompute.
# ======================================================================


def _run_with_build(topic, lanes, build_side_effect, *, embed_side_effect=None, search_side_effect=None):
    """Like _run() but with a fully custom build_candidate_pool side
    effect (so a test can hand back the SAME Paper object to two lanes)."""
    fakes = _Fakes({})
    with patch.object(rlr, "build_candidate_pool", side_effect=build_side_effect), \
         patch.object(rlr, "embed_and_index_papers", side_effect=embed_side_effect or fakes.embed), \
         patch.object(rlr, "semantic_search", side_effect=search_side_effect or fakes.search), \
         patch.object(rlr, "get_chroma_collection", side_effect=AssertionError("collection must be injected")):
        return retrieve_across_lanes(topic, lanes, collection=object()), fakes


# --- Fix 1: the SAME Paper instance returned by multiple lanes ---------

def test_same_paper_object_returned_by_two_lanes_keeps_both_lane_ids():
    """Regression: the old dict[int, str] kept only the LAST lane_id for a
    shared object. build_candidate_pool here returns the IDENTICAL Paper
    instance for lane A and lane B."""
    shared = _paper("shared", "Dense Passage Retrieval for QA")

    def _build(query, k, **kwargs):
        assert query in ("qa", "qb")
        return [shared]  # the exact same object both times

    lanes = [_lane("qa", lane_id="A"), _lane("qb", lane_id="B")]
    result, _ = _run_with_build("t", lanes, _build)

    assert result.paper_lane_ids == {"shared": ["A", "B"]}
    assert result.lane_result_counts == {"A": 1, "B": 1}
    assert len(result.ranked) == 1


def test_same_paper_object_across_three_lanes_in_enabled_order_even_when_discovered_out_of_order():
    shared = _paper("s", "Fusion in Decoder for Knowledge Grounded Generation")

    def _build(query, k, **kwargs):
        # lane C ("qc") and lane A ("qa") return the shared object; B does not
        return [shared] if query in ("qc", "qa") else [_paper("b1", "Reranking Candidate Passages")]

    lanes = [_lane("qa", lane_id="A"), _lane("qb", lane_id="B"), _lane("qc", lane_id="C")]
    result, _ = _run_with_build("t", lanes, _build)

    assert result.paper_lane_ids["s"] == ["A", "C"]           # enabled-lane order, not discovery order
    assert result.paper_lane_ids["b1"] == ["B"]


def test_shared_object_provenance_has_no_duplicates_even_if_a_lane_returns_it_twice():
    shared = _paper("s2", "Instruction Tuning for Factual Question Answering")

    def _build(query, k, **kwargs):
        return [shared, shared] if query == "qa" else [shared]  # lane A hands it back twice

    lanes = [_lane("qa", lane_id="A"), _lane("qb", lane_id="B")]
    result, _ = _run_with_build("t", lanes, _build)
    assert result.paper_lane_ids["s2"] == ["A", "B"]          # A once, then B


def test_distinct_duplicate_objects_merged_by_doi_still_get_every_lane():
    # unchanged-behaviour guard: the DOI-merge path (distinct objects)
    # must keep working alongside the shared-object fix.
    def _build(query, k, **kwargs):
        pid = "arxiv:x" if query == "qa" else "s2:x"
        return [_paper(pid, f"Title {query}", doi="10.1/same")]

    lanes = [_lane("qa", lane_id="A"), _lane("qb", lane_id="B")]
    result, _ = _run_with_build("t", lanes, _build)
    (lane_ids,) = list(result.paper_lane_ids.values())
    assert lane_ids == ["A", "B"]


# --- Fix 2: every item validated as ResearchLane before .enabled ------

@pytest.mark.parametrize("bad", [
    {"lane_id": "A", "enabled": True},     # dict
    None,
    "a lane",                             # str
    SimpleNamespace(),                     # object without .enabled
    SimpleNamespace(enabled=True),         # has .enabled but not a ResearchLane
    SimpleNamespace(enabled=False),        # would-be-disabled non-ResearchLane
    123,
])
def test_malformed_list_item_raises_controlled_typeerror_before_any_work(bad):
    good = _lane("methods", lane_id="A")
    with patch.object(rlr, "build_candidate_pool", side_effect=AssertionError("no build")), \
         patch.object(rlr, "embed_and_index_papers", side_effect=AssertionError("no embed")), \
         patch.object(rlr, "semantic_search", side_effect=AssertionError("no rank")), \
         patch.object(rlr, "get_chroma_collection", side_effect=AssertionError("no chroma")):
        with pytest.raises(TypeError, match="ResearchLane"):
            retrieve_across_lanes("t", [good, bad], collection=object())


def test_malformed_item_rejected_even_when_it_would_have_been_disabled():
    good = _lane("methods", lane_id="A")
    would_be_disabled = SimpleNamespace(enabled=False, lane_id="X", query="x")
    with patch.object(rlr, "build_candidate_pool", side_effect=AssertionError("no build")):
        with pytest.raises(TypeError, match="ResearchLane"):
            retrieve_across_lanes("t", [good, would_be_disabled], collection=object())


def test_malformed_item_as_sole_entry_raises_typeerror_not_valueerror():
    with patch.object(rlr, "build_candidate_pool", side_effect=AssertionError("no build")):
        with pytest.raises(TypeError, match="ResearchLane"):
            retrieve_across_lanes("t", [None], collection=object())


def test_type_check_precedes_the_no_enabled_lanes_valueerror():
    # a valid disabled lane + a malformed item: TypeError (item type)
    # wins over ValueError (no enabled lane)
    with patch.object(rlr, "build_candidate_pool", side_effect=AssertionError("no build")):
        with pytest.raises(TypeError):
            retrieve_across_lanes("t", [_lane("x", enabled=False), {}], collection=object())


# --- Fix 3: recompute keywords ONLY for >1-member clusters -----------

def test_singleton_with_nonempty_keywords_never_calls_extract_keywords():
    lanes = [_lane("methods", lane_id="A")]
    p = _paper("solo", "A Solo Standalone Paper", keywords=["kept keyword a", "kept keyword b"])
    fakes = _Fakes({"methods": [p]})
    with patch.object(rlr, "extract_keywords", wraps=rlr.extract_keywords) as ek:
        result = _run("t", lanes, fakes)
    ek.assert_not_called()
    (paper, _), = result.ranked
    assert paper.keywords == ["kept keyword a", "kept keyword b"]


def test_singleton_with_empty_keywords_is_left_empty_and_not_recomputed():
    lanes = [_lane("methods", lane_id="A")]
    p = _paper("solo0", "A Legitimately Keywordless Paper", keywords=[])
    fakes = _Fakes({"methods": [p]})
    with patch.object(rlr, "extract_keywords", wraps=rlr.extract_keywords) as ek:
        result = _run("t", lanes, fakes)
    ek.assert_not_called()
    (paper, _), = result.ranked
    assert paper.keywords == []


def test_multi_member_merged_cluster_recomputes_yake_v2_exactly_once():
    lanes = [_lane("methods", lane_id="A"), _lane("evaluation", lane_id="B")]
    pa = _paper("arxiv:m", "Retrieval Augmented Generation", doi="10.2/m", abstract=_ABSTRACT)
    pb = _paper("s2:m", "Grounding Language Models In External Documents", doi="10.2/m", abstract=_ABSTRACT)
    fakes = _Fakes({"methods": [pa], "evaluation": [pb]})
    with patch.object(rlr, "extract_keywords", wraps=rlr.extract_keywords) as ek:
        result = _run("t", lanes, fakes)
    ek.assert_called_once()
    (merged, _), = result.ranked
    assert ek.call_args == call(merged.title, merged.abstract)
    assert merged.keywords and merged.keywords == rlr.extract_keywords(merged.title, merged.abstract)


def test_two_merged_clusters_and_a_singleton_recompute_once_each_and_never_for_the_singleton():
    lanes = [_lane("a", lane_id="A"), _lane("b", lane_id="B")]
    # cluster 1: same DOI; cluster 2: same DOI; plus one singleton in lane A
    a1 = _paper("a:1", "Sparse Mixture of Experts", doi="10.4/one", abstract=_ABSTRACT)
    b1 = _paper("b:1", "Efficient Inference via Expert Routing", doi="10.4/one", abstract=_ABSTRACT)
    a2 = _paper("a:2", "Multi Hop Reasoning over Knowledge Bases", doi="10.4/two", abstract=_ABSTRACT)
    b2 = _paper("b:2", "Chained Retrieval for Complex Questions", doi="10.4/two", abstract=_ABSTRACT)
    solo = _paper("a:solo", "A Distinct Unshared Contribution", keywords=["untouched kw"])
    fakes = _Fakes({"a": [a1, a2, solo], "b": [b1, b2]})
    with patch.object(rlr, "extract_keywords", wraps=rlr.extract_keywords) as ek:
        result = _run("t", lanes, fakes)
    assert ek.call_count == 2  # once per merged cluster, never for the singleton
    by_id = {p.paper_id: p for p, _ in result.ranked}
    assert by_id["a:solo"].keywords == ["untouched kw"]
    for merged_id in ("a:1+b:1", "a:2+b:2"):
        assert by_id[merged_id].keywords  # merged output retains valid keywords


# ======================================================================
# RL4: refill_lane_session -- the multi-lane counterpart of refill_pool.
# ======================================================================

from research_agent.research_lane_retrieval import refill_lane_session  # noqa: E402


def _lane_session(lanes, *, reserve, cursor, paper_lane_ids, seen_paper_ids, seen_titles=(), refinement_notes=()):
    s = PaperPoolSession(
        topic="the review topic",
        lanes=list(lanes),
        reserve=list(reserve),
        cursor=cursor,
        paper_lane_ids={k: list(v) for k, v in paper_lane_ids.items()},
        seen_paper_ids=set(seen_paper_ids),
        seen_titles=set(seen_titles),
        refinement_notes=list(refinement_notes),
    )
    return s


def _run_refill(session, fakes, **kwargs):
    with patch.object(rlr, "build_candidate_pool", side_effect=fakes.build), \
         patch.object(rlr, "embed_and_index_papers", side_effect=fakes.embed), \
         patch.object(rlr, "semantic_search", side_effect=fakes.search), \
         patch.object(rlr, "get_chroma_collection", side_effect=AssertionError("collection must be injected in tests")):
        return refill_lane_session(session, collection=object(), **kwargs)


def _refill_scenario():
    A, B, C = _lane("qa", lane_id="A"), _lane("qb", lane_id="B"), _lane("qc", lane_id="C", enabled=False)
    p0 = _paper("p0", "Served Paper Zero")
    t1 = _paper("t1", "Unserved Tail One")
    t2 = _paper("t2", "Unserved Tail Two")
    t3 = _paper("t3", "Unserved Tail Three")
    session = _lane_session(
        [A, B, C],
        reserve=[(p0, 0.99), (t1, 0.9), (t2, 0.8), (t3, 0.7)],
        cursor=1,  # p0 already served; tail = t1, t2, t3
        paper_lane_ids={"p0": ["A"], "t1": ["A"], "t2": ["B"], "t3": ["A", "B"]},
        seen_paper_ids={"p0"},
        seen_titles={"Served Paper Zero"},
    )
    # fresh search: lane A re-finds t1 + brings n1,n2 ; lane B re-finds t3 + brings n3
    fakes = _Fakes({
        "qa": [_paper("t1", "Unserved Tail One"), _paper("n1", "New One"), _paper("n2", "New Two")],
        "qb": [_paper("t3", "Unserved Tail Three"), _paper("n3", "New Three")],
    })
    return session, fakes


def test_refill_searches_every_enabled_lane_only_and_forwards_exclusions():
    session, fakes = _refill_scenario()
    _run_refill(session, fakes)
    assert [c["query"] for c in fakes.build_calls] == ["qa", "qb"]  # C is disabled -> never searched
    for c in fakes.build_calls:
        assert c["kwargs"]["exclude_titles"] == ["Served Paper Zero"]
        assert c["kwargs"]["refinement_notes"] is None


def test_refill_embeds_once_and_runs_one_semantic_pass_per_enabled_lane():
    session, fakes = _refill_scenario()
    _run_refill(session, fakes)
    assert len(fakes.embed_calls) == 1
    assert len(fakes.search_calls) == 2  # one per enabled lane, over the COMBINED pool
    embedded_ids = sorted(p.paper_id for p in fakes.embed_calls[0])
    assert embedded_ids == ["n1", "n2", "n3", "t1", "t2", "t3"]  # tail + genuinely-new, deduped, once


def test_refill_each_lane_pass_is_restricted_to_that_lanes_cumulative_subset():
    session, fakes = _refill_scenario()
    _run_refill(session, fakes)
    by_lane = {}
    for c in fakes.search_calls:
        facet = c["query"].splitlines()[-1]
        by_lane[facet] = set(c["ids"])
    assert by_lane["Research facet: qa"] == {"t1", "t3", "n1", "n2"}
    assert by_lane["Research facet: qb"] == {"t2", "t3", "n3"}


def test_refill_preserves_the_unserved_reserve_tail_and_merges_new_results():
    session, fakes = _refill_scenario()
    count = _run_refill(session, fakes)
    assert count == 3  # n1, n2, n3
    ids = [p.paper_id for p, _ in session.reserve]
    assert set(ids) == {"t1", "t2", "t3", "n1", "n2", "n3"}
    assert len(ids) == len(set(ids))  # each once
    assert session.cursor == 0


def test_refill_unions_cumulative_provenance_and_never_removes_old():
    session, fakes = _refill_scenario()
    _run_refill(session, fakes)
    assert session.paper_lane_ids["t3"] == ["A", "B"]        # was ["A","B"], B re-found -> unchanged, ordered
    assert session.paper_lane_ids["t1"] == ["A"]             # A re-found -> still ["A"]
    assert session.paper_lane_ids["t2"] == ["B"]             # NOT re-found -> old provenance kept
    assert session.paper_lane_ids["p0"] == ["A"]             # served paper's provenance kept
    assert session.paper_lane_ids["n1"] == ["A"]
    assert session.paper_lane_ids["n3"] == ["B"]


def test_refill_recomputes_lane_result_counts_from_cumulative_provenance():
    session, fakes = _refill_scenario()
    _run_refill(session, fakes)
    # A: p0,t1,t3,n1,n2 = 5 ; B: t2,t3,n3 = 3 ; C (disabled) = 0
    assert session.lane_result_counts == {"A": 5, "B": 3, "C": 0}


def test_refill_duplicate_discovery_does_not_inflate_provenance_or_counts():
    session, fakes = _refill_scenario()
    _run_refill(session, fakes)
    prov_after_1 = {k: list(v) for k, v in session.paper_lane_ids.items()}
    counts_after_1 = dict(session.lane_result_counts)

    # run the SAME fresh search again -- every result is now already seen /
    # in the tail, so nothing is genuinely new and nothing inflates.
    session.seen_paper_ids |= {p.paper_id for p, _ in session.reserve[: 0]}  # no-op, keep seen as-is
    fakes2 = _Fakes({
        "qa": [_paper("t1", "Unserved Tail One"), _paper("n1", "New One"), _paper("n2", "New Two")],
        "qb": [_paper("t3", "Unserved Tail Three"), _paper("n3", "New Three")],
    })
    # mark the new papers as seen so the 2nd refill treats them as duplicates
    session.seen_paper_ids |= {"n1", "n2", "n3"}
    count2 = _run_refill(session, fakes2)
    assert count2 == 0
    for k, v in prov_after_1.items():
        assert session.paper_lane_ids[k] == v  # unchanged
    assert session.lane_result_counts == counts_after_1


def test_refill_with_empty_fresh_results_still_re_ranks_the_tail():
    A, B = _lane("qa", lane_id="A"), _lane("qb", lane_id="B")
    t1, t2 = _paper("t1", "Tail One Title"), _paper("t2", "Tail Two Title")
    session = _lane_session(
        [A, B], reserve=[(t1, 0.9), (t2, 0.8)], cursor=0,
        paper_lane_ids={"t1": ["A"], "t2": ["B"]}, seen_paper_ids=set(),
    )
    fakes = _Fakes({"qa": [], "qb": []})
    count = _run_refill(session, fakes)
    assert count == 0
    assert len(fakes.embed_calls) == 1  # tail still embedded/ranked
    assert {p.paper_id for p, _ in session.reserve} == {"t1", "t2"}
    assert session.lane_result_counts == {"A": 1, "B": 1}


def test_refill_with_no_tail_and_no_fresh_results_empties_the_reserve():
    A = _lane("qa", lane_id="A")
    session = _lane_session([A], reserve=[], cursor=0, paper_lane_ids={}, seen_paper_ids=set())
    fakes = _Fakes({"qa": []})
    with patch.object(rlr, "build_candidate_pool", side_effect=fakes.build), \
         patch.object(rlr, "embed_and_index_papers", side_effect=AssertionError("no embed on empty")), \
         patch.object(rlr, "semantic_search", side_effect=AssertionError("no rank on empty")), \
         patch.object(rlr, "get_chroma_collection", side_effect=AssertionError("no chroma on empty")):
        count = refill_lane_session(session, collection=object())
    assert count == 0
    assert session.reserve == []
    assert session.lane_result_counts == {"A": 0}


def test_refill_does_not_mutate_the_frozen_lane_objects():
    session, fakes = _refill_scenario()
    lanes_before = [l.to_dict() for l in session.lanes]
    _run_refill(session, fakes)
    assert [l.to_dict() for l in session.lanes] == lanes_before


def test_refill_folds_refinement_notes_into_search_and_ranking():
    A = _lane("qa", lane_id="A")
    t1 = _paper("t1", "Tail One")
    session = _lane_session(
        [A], reserve=[(t1, 0.9)], cursor=0, paper_lane_ids={"t1": ["A"]},
        seen_paper_ids=set(), refinement_notes=["focus on recent work"],
    )
    fakes = _Fakes({"qa": [_paper("t1", "Tail One"), _paper("n1", "New One Title")]})
    _run_refill(session, fakes)
    assert fakes.build_calls[0]["kwargs"]["refinement_notes"] == ["focus on recent work"]
    assert fakes.search_calls[0]["query"] == "the review topic\nResearch facet: qa. focus on recent work"


# --- RL4a: fresh-vs-tail content dedup, provenance union, tail safety ---

def test_refill_fresh_result_that_content_duplicates_a_tail_paper_merges_not_duplicates():
    """Regression (RL4a): a fresh lane record that is the SAME real paper
    as an unserved-tail paper -- matched by DOI even though its paper_id
    differs -- must NOT appear as a second reserve entry, and its lane
    provenance must be added to the surviving tail paper."""
    A, B = _lane("qa", lane_id="A"), _lane("qb", lane_id="B")
    tail = _paper("arxiv-t1", "Grounded Retrieval For Factual QA", doi="10.1/shared")
    t2 = _paper("t2", "An Unrelated Tail Paper")
    session = _lane_session(
        [A, B], reserve=[(tail, 0.9), (t2, 0.8)], cursor=0,
        paper_lane_ids={"arxiv-t1": ["A"], "t2": ["B"]}, seen_paper_ids=set(),
    )
    # lane B re-discovers the same real paper, DIFFERENT paper_id, same DOI
    fakes = _Fakes({
        "qa": [_paper("n1", "Dense Passage Retrieval for Open Domain QA")],
        "qb": [_paper("s2-t1", "Grounded Retrieval For Factual QA", doi="10.1/shared"),
               _paper("n2", "Sparse Mixture of Experts for Efficient Inference")],
    })
    count = _run_refill(session, fakes)

    reserve_ids = [p.paper_id for p, _ in session.reserve]
    assert reserve_ids.count("arxiv-t1") == 1        # the tail paper, once
    assert "s2-t1" not in reserve_ids               # the fresh duplicate never enters
    assert set(reserve_ids) == {"arxiv-t1", "t2", "n1", "n2"}
    assert session.paper_lane_ids["arxiv-t1"] == ["A", "B"]   # fresh lane B provenance unioned in
    assert "s2-t1" not in session.paper_lane_ids
    assert count == 2                               # n1, n2 only
    # counts recomputed from cumulative provenance -- the shared paper
    # counts for both lanes, but is not double-counted.
    assert session.lane_result_counts == {"A": 2, "B": 3}  # A: arxiv-t1,n1 ; B: arxiv-t1,t2,n2


def test_refill_tail_paper_with_no_surviving_provenance_is_still_carried_forward():
    """A tail paper whose lane provenance is (somehow) empty must not be
    silently dropped by the per-lane ranking -- it is appended with its
    prior score."""
    A = _lane("qa", lane_id="A")
    orphan = _paper("orphan", "Orphaned Tail Paper")
    kept = _paper("kept", "Well Provenanced Tail Paper")
    session = _lane_session(
        [A], reserve=[(kept, 0.9), (orphan, 0.4)], cursor=0,
        paper_lane_ids={"kept": ["A"]},  # 'orphan' deliberately absent
        seen_paper_ids=set(),
    )
    fakes = _Fakes({"qa": []})
    _run_refill(session, fakes)
    reserve_ids = {p.paper_id for p, _ in session.reserve}
    assert reserve_ids == {"kept", "orphan"}       # tail fully preserved
    orphan_score = next(s for p, s in session.reserve if p.paper_id == "orphan")
    assert orphan_score == 0.4                     # prior reserve score kept


def test_refill_multi_lane_paper_is_included_in_every_applicable_lane_ranking_pass():
    A, B, C = _lane("qa", lane_id="A"), _lane("qb", lane_id="B"), _lane("qc", lane_id="C")
    shared = _paper("shared", "A Paper Two Lanes Both Want", doi="10.2/s")
    session = _lane_session(
        [A, B, C], reserve=[(shared, 0.9)], cursor=0,
        paper_lane_ids={"shared": ["A", "B"]}, seen_paper_ids=set(),
    )
    fakes = _Fakes({"qa": [], "qb": [], "qc": [_paper("c1", "Lane C Only Paper")]})
    _run_refill(session, fakes)
    subsets = {c["query"].splitlines()[-1]: set(c["ids"]) for c in fakes.search_calls}
    assert "shared" in subsets["Research facet: qa"]
    assert "shared" in subsets["Research facet: qb"]
    assert "shared" not in subsets.get("Research facet: qc", set())
    ids = [p.paper_id for p, _ in session.reserve]
    assert ids.count("shared") == 1               # emitted once despite two lanes


def test_refill_zero_new_results_keeps_the_full_unserved_tail():
    A, B = _lane("qa", lane_id="A"), _lane("qb", lane_id="B")
    t1, t2, t3 = _paper("t1", "Tail A"), _paper("t2", "Tail B"), _paper("t3", "Tail AB", doi="10.3/x")
    session = _lane_session(
        [A, B], reserve=[(t1, 0.9), (t2, 0.8), (t3, 0.7)], cursor=0,
        paper_lane_ids={"t1": ["A"], "t2": ["B"], "t3": ["A", "B"]}, seen_paper_ids=set(),
    )
    fakes = _Fakes({"qa": [], "qb": []})
    count = _run_refill(session, fakes)
    assert count == 0
    assert {p.paper_id for p, _ in session.reserve} == {"t1", "t2", "t3"}
    assert session.paper_lane_ids == {"t1": ["A"], "t2": ["B"], "t3": ["A", "B"]}  # untouched
    assert session.lane_result_counts == {"A": 2, "B": 2}
