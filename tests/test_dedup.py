"""Deterministic dedup tests — don't depend on live, rate-limited APIs.

Mirrors the exact shape of a real cross-source duplicate: same paper,
different paper_id/venue/citation_count/DOI presence per source.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.dedup import deduplicate
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


if __name__ == "__main__":
    test_cross_source_duplicate_collapses_to_one_record()
    test_distinct_papers_are_not_merged()
    test_doi_match_overrides_dissimilar_titles()
    test_empty_input_returns_empty()
    print("All dedup tests passed.")
