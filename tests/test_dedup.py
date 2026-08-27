"""Deterministic dedup tests — don't depend on live, rate-limited APIs.

Mirrors the exact shape of a real cross-source duplicate: same paper,
different paper_id/venue/citation_count/DOI presence per source.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.dedup import deduplicate, deduplicate_with_clusters
from research_agent.schema import Paper


def _arxiv_paper() -> Paper:
    return Paper(
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=["Patrick Lewis", "Ethan Perez", "Douwe Kiela"],
        year=2020,
        venue="arXiv preprint",
        abstract="Large pre-trained language models have been shown to store factual knowledge.",
        url="http://arxiv.org/abs/2005.11401v4",
        doi=None,
        citation_count=None,
        source="arxiv",
        paper_id="2005.11401v4",
    )


def _s2_paper() -> Paper:
    return Paper(
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=["Patrick Lewis", "Ethan Perez", "Aleksandara Piktus", "Douwe Kiela"],
        year=2020,
        venue="Neural Information Processing Systems",
        abstract=(
            "Large pre-trained language models have been shown to store factual "
            "knowledge in their parameters, and achieve state-of-the-art results "
            "when fine-tuned on downstream NLP tasks."
        ),
        url="https://www.semanticscholar.org/paper/659bf9ce7175e1ec266ff54359e2bd76e0b7ff31",
        doi="10.48550/arXiv.2005.11401",
        citation_count=15509,
        source="semantic_scholar",
        paper_id="659bf9ce7175e1ec266ff54359e2bd76e0b7ff31",
    )


def test_cross_source_duplicate_collapses_to_one_record():
    merged = deduplicate([_arxiv_paper(), _s2_paper()])
    assert len(merged) == 1

    p = merged[0]
    assert p.source == "arxiv+semantic_scholar"
    assert set(p.source_urls) == {"arxiv", "semantic_scholar"}
    assert p.citation_count == 15509  # max of [None, 15509]
    assert p.doi == "10.48550/arXiv.2005.11401"  # only S2 had one
    assert p.venue == "Neural Information Processing Systems"  # preferred over "arXiv preprint"
    assert len(p.abstract) == len(_s2_paper().abstract)  # the longer of the two
    assert "Aleksandara Piktus" in p.authors  # union of both author lists
    assert len(p.authors) == 4  # Lewis + Perez are shared, not double-counted


def test_distinct_papers_are_not_merged():
    a = _arxiv_paper()
    b = _arxiv_paper()
    b.title = "A Completely Unrelated Paper About Reinforcement Learning"
    b.doi = None
    b.paper_id = "9999.99999"
    merged = deduplicate([a, b])
    assert len(merged) == 2


def test_doi_match_overrides_dissimilar_titles():
    a = _arxiv_paper()
    b = _arxiv_paper()
    b.title = "Retrieval Augmented Generation for Knowledge Intensive NLP Tasks (v2, camera-ready)"
    b.doi = "10.1234/same-paper"
    b.paper_id = "different-id"
    a.doi = "10.1234/same-paper"
    merged = deduplicate([a, b])
    assert len(merged) == 1


def test_empty_input_returns_empty():
    assert deduplicate([]) == []


# --- RL3: deduplicate_with_clusters is deduplicate() + cluster membership ---

def _p(pid: str, title: str, *, doi=None, abstract="an abstract here for the paper") -> Paper:
    return Paper(
        title=title, authors=["A"], year=2024, venue="V", abstract=abstract,
        url=None, doi=doi, citation_count=None, source="arxiv", paper_id=pid,
    )


def _fixtures():
    return [
        [],
        [_p("a", "Alpha")],
        [_p("a", "Alpha"), _p("b", "Beta"), _p("c", "Gamma")],
        [_arxiv_paper(), _s2_paper(), _p("x", "Totally Different Title")],
        # DOI match over dissimilar titles + a fuzzy-title pair
        [_p("d1", "Neural Scaling Laws", doi="10.1/x"), _p("d2", "A Study Of Something Else", doi="10.1/x"),
         _p("t1", "Retrieval Augmented Generation"), _p("t2", "Retrieval-Augmented Generation")],
    ]


def test_deduplicate_with_clusters_merged_list_is_byte_for_byte_deduplicate():
    for papers in _fixtures():
        plain = deduplicate(list(papers))
        withc = deduplicate_with_clusters(list(papers))
        merged_only = [m for m, _ in withc]
        assert len(merged_only) == len(plain)
        for a, b in zip(merged_only, plain):
            assert a.to_dict() == b.to_dict()


def test_deduplicate_with_clusters_exposes_the_dedup_cluster_members():
    a, x, s = _arxiv_paper(), _p("x", "Unrelated"), _s2_paper()
    withc = deduplicate_with_clusters([a, x, s])
    # cluster order follows first-seen input order: the RAG pair, then x
    assert [len(members) for _, members in withc] == [2, 1]
    rag_cluster = withc[0][1]
    assert {m.source for m in rag_cluster} == {"arxiv", "semantic_scholar"}
    assert rag_cluster[0] is a and rag_cluster[1] is s  # exact input objects, in order
    assert withc[1][1] == [x]


def test_deduplicate_with_clusters_singleton_merged_is_the_same_object():
    p = _p("solo", "Solo Paper")
    withc = deduplicate_with_clusters([p])
    (merged, members), = withc
    assert merged is p and members == [p]


if __name__ == "__main__":
    test_cross_source_duplicate_collapses_to_one_record()
    test_distinct_papers_are_not_merged()
    test_doi_match_overrides_dissimilar_titles()
    test_empty_input_returns_empty()
    print("All dedup tests passed.")
