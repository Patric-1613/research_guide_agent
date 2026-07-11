"""Deterministic tests for the embedding cache and Chroma metadata round-trip.
No live OpenAI/Chroma calls — those are covered by scripts/test_ranking.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb

from research_agent.embeddings import (
    _get_cached,
    _hash_text,
    _init_cache_db,
    _paper_from_metadata,
    _serialize_metadata,
    _set_cached,
    semantic_search,
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


def _fake_openai_client(query_vector: list[float]) -> MagicMock:
    """Mocks the one client.embeddings.create() call semantic_search makes
    for the query text, so these tests exercise real Chroma filtering
    without a live OpenAI call."""
    client = MagicMock()
    response = MagicMock()
    response.usage.total_tokens = 3
    response.data = [MagicMock(embedding=query_vector)]
    client.embeddings.create.return_value = response
    return client


def _filter_test_paper(paper_id: str, doi: str | None, citation_count: int | None) -> Paper:
    return Paper(
        title=f"Paper {paper_id}", authors=["A"], year=2024, venue="X",
        abstract=f"abstract for {paper_id}", url=None, doi=doi, citation_count=citation_count,
        source="arxiv", paper_id=paper_id,
    )


def _seed_collection(collection, entries: list[tuple[str, list[float], str | None, int | None]]) -> None:
    ids, vectors, metas, docs = [], [], [], []
    for paper_id, vector, doi, citation_count in entries:
        paper = _filter_test_paper(paper_id, doi, citation_count)
        ids.append(paper_id)
        vectors.append(vector)
        metas.append(_serialize_metadata(paper, used_title_fallback=False))
        docs.append(paper.abstract)
    collection.upsert(ids=ids, embeddings=vectors, metadatas=metas, documents=docs)


def test_semantic_search_min_citation_count_filters_via_chroma_where():
    collection = chromadb.EphemeralClient().get_or_create_collection("filter-test-1", metadata={"hnsw:space": "cosine"})
    _seed_collection(collection, [
        ("a", [1.0, 0.0], None, 50),
        ("b", [1.0, 0.0], None, 5),
        ("c", [1.0, 0.0], None, None),  # unknown citation count must not pass a nonzero minimum
    ])
    results = semantic_search(
        "q", collection=collection, client=_fake_openai_client([1.0, 0.0]),
        top_k=10, min_citation_count=10,
    )
    assert {p.paper_id for p, _ in results} == {"a"}


def test_semantic_search_require_doi_excludes_papers_without_one():
    collection = chromadb.EphemeralClient().get_or_create_collection("filter-test-2", metadata={"hnsw:space": "cosine"})
    _seed_collection(collection, [
        ("a", [1.0, 0.0], "10.1/a", None),
        ("b", [0.9, 0.1], None, None),
        ("c", [0.8, 0.2], "10.1/c", None),
    ])
    results = semantic_search(
        "q", collection=collection, client=_fake_openai_client([1.0, 0.0]),
        top_k=10, require_doi=True,
    )
    assert {p.paper_id for p, _ in results} == {"a", "c"}


def test_semantic_search_require_doi_truncates_top_k_after_filtering_not_before():
    collection = chromadb.EphemeralClient().get_or_create_collection("filter-test-3", metadata={"hnsw:space": "cosine"})
    _seed_collection(collection, [
        ("a", [1.0, 0.0], "10.1/a", None),
        ("b", [0.95, 0.05], None, None),  # closer to query than c, but no doi
        ("c", [0.9, 0.1], "10.1/c", None),
        ("d", [0.8, 0.2], "10.1/d", None),
    ])
    results = semantic_search(
        "q", collection=collection, client=_fake_openai_client([1.0, 0.0]),
        top_k=2, require_doi=True,
    )
    # b is naively closer than c but has no DOI — top_k must be applied
    # after filtering, so a DOI-bearing paper ranked lower still makes it in
    # rather than b silently consuming one of the two slots.
    assert [p.paper_id for p, _ in results] == ["a", "c"]


def test_semantic_search_combines_paper_id_scoping_with_citation_filter():
    collection = chromadb.EphemeralClient().get_or_create_collection("filter-test-4", metadata={"hnsw:space": "cosine"})
    _seed_collection(collection, [
        ("a", [1.0, 0.0], None, 100),
        ("b", [1.0, 0.0], None, 100),  # would pass the citation filter alone, excluded by paper_id scoping
    ])
    results = semantic_search(
        "q", collection=collection, client=_fake_openai_client([1.0, 0.0]), top_k=10,
        where={"paper_id": {"$in": ["a"]}}, min_citation_count=10,
    )
    assert {p.paper_id for p, _ in results} == {"a"}


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
    test_semantic_search_min_citation_count_filters_via_chroma_where()
    test_semantic_search_require_doi_excludes_papers_without_one()
    test_semantic_search_require_doi_truncates_top_k_after_filtering_not_before()
    test_semantic_search_combines_paper_id_scoping_with_citation_filter()
    test_metadata_omits_none_fields_chroma_would_reject()
    print("All embedding tests passed.")
