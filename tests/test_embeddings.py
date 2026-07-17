"""Deterministic tests for the embedding cache and Chroma metadata round-trip.
No live OpenAI/Chroma calls — those are covered by scripts/test_ranking.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb

from research_agent.embeddings import (
    _embed_texts,
    _get_cached,
    _hash_text,
    _init_cache_db,
    _paper_from_metadata,
    _serialize_metadata,
    _set_cached,
    embed_and_index_papers,
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
    response.data = [MagicMock(embedding=query_vector, index=0)]
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


def test_embed_texts_sorts_out_of_order_response_by_index_field():
    # OpenAI's embeddings API includes an `index` on each result specifically
    # because return order isn't guaranteed. Simulate a response that comes
    # back out of order (indices [2, 0, 1] instead of [0, 1, 2]) and confirm
    # each vector still lands at the position matching its original input.
    client = MagicMock()
    response = MagicMock()
    response.usage.total_tokens = 9
    response.data = [
        MagicMock(embedding=[0.0, 0.0, 1.0], index=2),
        MagicMock(embedding=[1.0, 0.0, 0.0], index=0),
        MagicMock(embedding=[0.0, 1.0, 0.0], index=1),
    ]
    client.embeddings.create.return_value = response

    vectors, tokens = _embed_texts(client, ["text-for-input-0", "text-for-input-1", "text-for-input-2"])

    assert vectors[0] == [1.0, 0.0, 0.0]
    assert vectors[1] == [0.0, 1.0, 0.0]
    assert vectors[2] == [0.0, 0.0, 1.0]
    assert tokens == 9


def _fake_batch_embed_client() -> MagicMock:
    """Mocks client.embeddings.create() for embed_and_index_papers, returning
    one fixed-length vector per input text regardless of batch size."""
    client = MagicMock()

    def _create(model, input):
        response = MagicMock()
        response.usage.total_tokens = 3 * len(input)
        response.data = [MagicMock(embedding=[0.1, 0.2], index=i) for i in range(len(input))]
        return response

    client.embeddings.create.side_effect = _create
    return client


def _isolated_cache(tmp_path_str: str):
    """embed_and_index_papers always calls _init_cache_db() with no args
    (hardcoded to the real on-disk cache), so tests must patch the function
    itself to point at a fresh temp DB — otherwise a second test run (or a
    second test using the same abstract text) gets a spurious cache hit."""
    return patch(
        "research_agent.embeddings._init_cache_db",
        return_value=_init_cache_db(Path(tmp_path_str) / "cache.sqlite"),
    )


def test_embed_and_index_papers_happy_path_embeds_normal_papers():
    # Proves the Phase 1 skip-fix below doesn't change behavior for
    # well-formed papers (title + abstract both present).
    collection = chromadb.EphemeralClient().get_or_create_collection("skip-test-happy", metadata={"hnsw:space": "cosine"})
    papers = [
        Paper(
            title="Real Paper One", authors=["A"], year=2024, venue="X",
            abstract="a genuinely unique abstract for the happy path test one",
            url=None, doi=None, citation_count=None, source="arxiv", paper_id="happy-1",
        ),
        Paper(
            title="Real Paper Two", authors=["B"], year=2024, venue="Y",
            abstract="a genuinely unique abstract for the happy path test two",
            url=None, doi=None, citation_count=None, source="arxiv", paper_id="happy-2",
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp, _isolated_cache(tmp):
        stats = embed_and_index_papers(papers, collection=collection, client=_fake_batch_embed_client())
    assert stats["papers_skipped"] == 0
    assert stats["cache_misses"] == 2
    assert collection.count() == 2


def test_embed_and_index_papers_skips_paper_with_empty_title_and_no_abstract():
    collection = chromadb.EphemeralClient().get_or_create_collection("skip-test-bad", metadata={"hnsw:space": "cosine"})
    good = Paper(
        title="A Real Paper With Content", authors=["A"], year=2024, venue="X",
        abstract="a genuinely unique abstract for the skip test",
        url=None, doi=None, citation_count=None, source="arxiv", paper_id="skip-good-1",
    )
    bad = Paper(
        title="", authors=[], year=None, venue=None, abstract=None,
        url=None, doi=None, citation_count=None, source="arxiv", paper_id="skip-bad-1",
    )
    with tempfile.TemporaryDirectory() as tmp, _isolated_cache(tmp):
        stats = embed_and_index_papers([good, bad], collection=collection, client=_fake_batch_embed_client())
    # The bad paper is skipped, not crashed on — the good paper still gets
    # embedded and indexed in the same batch call.
    assert stats["papers_skipped"] == 1
    assert stats["cache_misses"] == 1
    assert collection.count() == 1
    assert collection.get(ids=["skip-good-1"])["ids"] == ["skip-good-1"]
    assert collection.get(ids=["skip-bad-1"])["ids"] == []


def test_embed_and_index_papers_all_papers_bad_returns_without_crashing():
    collection = chromadb.EphemeralClient().get_or_create_collection("skip-test-allbad", metadata={"hnsw:space": "cosine"})
    bad = Paper(
        title="   ", authors=[], year=None, venue=None, abstract="",
        url=None, doi=None, citation_count=None, source="arxiv", paper_id="skip-allbad-1",
    )
    with tempfile.TemporaryDirectory() as tmp, _isolated_cache(tmp):
        stats = embed_and_index_papers([bad], collection=collection, client=_fake_batch_embed_client())
    assert stats["papers_skipped"] == 1
    assert stats["tokens_billed"] == 0
    assert collection.count() == 0


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
    test_embed_texts_sorts_out_of_order_response_by_index_field()
    test_embed_and_index_papers_happy_path_embeds_normal_papers()
    test_embed_and_index_papers_skips_paper_with_empty_title_and_no_abstract()
    test_embed_and_index_papers_all_papers_bad_returns_without_crashing()
    test_metadata_omits_none_fields_chroma_would_reject()
    print("All embedding tests passed.")
