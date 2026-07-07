"""Deterministic tests for the embedding cache and Chroma metadata round-trip.
No live OpenAI/Chroma calls — those are covered by scripts/test_ranking.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.embeddings import (
    _get_cached,
    _hash_text,
    _init_cache_db,
    _paper_from_metadata,
    _serialize_metadata,
    _set_cached,
)
from research_agent.schema import Paper


def test_hash_is_stable_and_whitespace_insensitive():
    assert _hash_text("hello world") == _hash_text("hello world")
    assert _hash_text("hello world") == _hash_text("  hello world  ")
    assert _hash_text("hello world") != _hash_text("hello World")


def test_cache_roundtrip_avoids_recompute():
    with tempfile.TemporaryDirectory() as tmp:
        conn = _init_cache_db(Path(tmp) / "cache.sqlite")
        h = _hash_text("some abstract text")
        assert _get_cached(conn, h) is None

        vector = [0.1, 0.2, 0.3]
        _set_cached(conn, h, vector, char_len=len("some abstract text"))
        assert _get_cached(conn, h) == vector
        conn.close()


def test_metadata_roundtrip_preserves_paper_fields():
    paper = Paper(
        title="Some Paper",
        authors=["Alice", "Bob"],
        year=2023,
        venue="NeurIPS",
        abstract="An abstract.",
        url="http://arxiv.org/abs/1234",
        doi="10.1/x",
        citation_count=42,
        source="arxiv",
        paper_id="1234",
    )
    meta = _serialize_metadata(paper, used_title_fallback=False)
    restored = _paper_from_metadata(meta)

    assert restored.title == paper.title
    assert restored.authors == paper.authors
    assert restored.year == paper.year
    assert restored.venue == paper.venue
    assert restored.abstract == paper.abstract
    assert restored.doi == paper.doi
    assert restored.citation_count == paper.citation_count
    assert restored.source_urls == paper.source_urls


def test_metadata_omits_none_fields_chroma_would_reject():
    paper = Paper(
        title="No DOI Paper",
        authors=[],
        year=None,
        venue=None,
        abstract=None,
        url=None,
        doi=None,
        citation_count=None,
        source="arxiv",
        paper_id="x",
    )
    meta = _serialize_metadata(paper, used_title_fallback=True)
    assert None not in meta.values()
    assert "doi" not in meta
    assert "abstract" not in meta


if __name__ == "__main__":
    test_hash_is_stable_and_whitespace_insensitive()
    test_cache_roundtrip_avoids_recompute()
    test_metadata_roundtrip_preserves_paper_fields()
    test_metadata_omits_none_fields_chroma_would_reject()
    print("All embedding tests passed.")
