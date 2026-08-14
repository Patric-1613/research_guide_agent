"""Paper Keywords and Filtering, K1: backward-compatibility proof for
research_agent.schema.Paper's new `keywords` field -- the direct, minimal
version of the claim tests/test_curation_session.py's own real-session
round-trip test also exercises at a higher level.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.schema import Paper


def _old_paper_dict() -> dict:
    """Exactly the field set Paper.to_dict() produced before K1 -- no
    "keywords" key at all, matching every Paper dict already persisted in
    a real session/checkpoint before this phase."""
    return {
        "title": "An Old Paper", "authors": ["A. Uthor"], "year": 2023, "venue": "X",
        "abstract": "An old abstract.", "url": "http://example.com/old", "doi": None,
        "citation_count": None, "source": "arxiv", "paper_id": "arxiv:an old paper",
        "source_urls": {"arxiv": "http://example.com/old"},
    }


def test_old_paper_dict_without_keywords_key_reconstructs_with_empty_list():
    paper = Paper(**_old_paper_dict())
    assert paper.keywords == []


def test_new_paper_to_dict_round_trips_keywords_unchanged():
    paper = Paper(
        title="A New Paper", authors=["A"], year=2024, venue="X", abstract="abstract",
        url=None, doi=None, citation_count=None, source="arxiv", paper_id="p1",
        keywords=["graph neural networks", "molecular property prediction"],
    )
    rebuilt = Paper(**paper.to_dict())
    assert rebuilt.keywords == ["graph neural networks", "molecular property prediction"]


def test_keywords_defaults_to_empty_list_when_omitted_entirely():
    paper = Paper(
        title="A Paper", authors=["A"], year=2024, venue="X", abstract="abstract",
        url=None, doi=None, citation_count=None, source="arxiv",
    )
    assert paper.keywords == []
