"""Phase 3: embed abstracts, persist in ChromaDB, retrieve by cosine similarity.

Retrieval here *is* the relevance ranking (per the project brief) — there is
deliberately no separate scoring layer stacked on top of the vector search.

Caching design: the cache key is a SHA-256 hash of the embedding input text
(abstract, or title as fallback), not the paper_id. paper_id is stable for a
single-source paper but changes when dedup (phase 2) merges two records into
"arxiv_id+s2_id" — keying the cache on content instead of identity means a
paper we've already embedded is never re-billed just because it later got
merged with a duplicate found under a different query.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path

import chromadb
from openai import OpenAI

from research_agent.schema import Paper

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# USD per 1M tokens for text-embedding-3-small, per OpenAI's published pricing
# as of training data cutoff (Jan 2026). Prices change — verify at
# https://openai.com/api/pricing before relying on this figure for real budgeting.
PRICE_PER_1M_TOKENS = 0.02

# Abstracts are short (usually well under 500 tokens), so batching this many
# per request stays far under OpenAI's per-request token/size limits while
# still cutting the number of round trips by ~100x vs. one-call-per-paper.
EMBED_BATCH_SIZE = 100

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
CACHE_DB_PATH = DATA_DIR / "cache" / "embeddings.sqlite"
COLLECTION_NAME = "papers"


def _init_cache_db(path: Path = CACHE_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_cache (
            text_hash TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            char_len INTEGER NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _get_cached(conn: sqlite3.Connection, text_hash: str) -> list[float] | None:
    row = conn.execute(
        "SELECT vector_json FROM embedding_cache WHERE text_hash = ? AND model = ?",
        (text_hash, EMBEDDING_MODEL),
    ).fetchone()
    return json.loads(row[0]) if row else None


def _set_cached(conn: sqlite3.Connection, text_hash: str, vector: list[float], char_len: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO embedding_cache (text_hash, model, vector_json, char_len, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (text_hash, EMBEDDING_MODEL, json.dumps(vector), char_len, time.time()),
    )
    conn.commit()


def get_chroma_collection(persist_dir: Path = CHROMA_PERSIST_DIR):
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_texts(client: OpenAI, texts: list[str]) -> tuple[list[list[float]], int]:
    """Batch-embed texts via the OpenAI API. Returns (vectors, total_tokens_billed)."""
    vectors: list[list[float]] = []
    total_tokens = 0
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_tokens = response.usage.total_tokens
        total_tokens += batch_tokens
        vectors.extend(item.embedding for item in response.data)
        cost = batch_tokens / 1_000_000 * PRICE_PER_1M_TOKENS
        logger.info(
            "Embedded batch of %d texts: %d tokens billed (~$%.6f)",
            len(batch), batch_tokens, cost,
        )
    return vectors, total_tokens


def _embedding_text(paper: Paper) -> tuple[str, bool]:
    """Text used as the embedding input. Falls back to title if the abstract
    is missing, which happens occasionally on both APIs. Returns
    (text, used_fallback) so callers can flag lower-confidence entries.
    """
    if paper.abstract:
        return paper.abstract, False
    logger.warning("Paper %r has no abstract; embedding title only", paper.title)
    return paper.title, True


def _serialize_metadata(paper: Paper, used_title_fallback: bool) -> dict:
    meta: dict = {
        "title": paper.title,
        "authors_json": json.dumps(paper.authors),
        "source": paper.source,
        "source_urls_json": json.dumps(paper.source_urls),
        "paper_id": paper.paper_id,
        "used_title_fallback": used_title_fallback,
    }
    if paper.year is not None:
        meta["year"] = paper.year
    if paper.venue:
        meta["venue"] = paper.venue
    if paper.url:
        meta["url"] = paper.url
    if paper.doi:
        meta["doi"] = paper.doi
    if paper.citation_count is not None:
        meta["citation_count"] = paper.citation_count
    if paper.abstract:
        meta["abstract"] = paper.abstract
    return meta


def _paper_from_metadata(metadata: dict) -> Paper:
    paper = Paper(
        title=metadata.get("title", ""),
        authors=json.loads(metadata.get("authors_json", "[]")),
        year=metadata.get("year"),
        venue=metadata.get("venue"),
        abstract=metadata.get("abstract"),
        url=metadata.get("url"),
        doi=metadata.get("doi"),
        citation_count=metadata.get("citation_count"),
        source=metadata.get("source", ""),
        paper_id=metadata.get("paper_id", ""),
    )
    paper.source_urls = json.loads(metadata.get("source_urls_json", "{}"))
    return paper


def embed_and_index_papers(
    papers: list[Paper],
    collection=None,
    client: OpenAI | None = None,
) -> dict:
    """Embed each paper's abstract (cached by content hash) and upsert into Chroma.

    Returns a stats dict: cache_hits, cache_misses, tokens_billed, estimated_cost_usd.
    """
    if not papers:
        return {"cache_hits": 0, "cache_misses": 0, "tokens_billed": 0, "estimated_cost_usd": 0.0}

    collection = collection or get_chroma_collection()
    client = client or OpenAI()
    cache_conn = _init_cache_db()

    texts_and_fallback = [_embedding_text(p) for p in papers]
    hashes = [_hash_text(text) for text, _ in texts_and_fallback]

    vectors: list[list[float] | None] = [None] * len(papers)
    to_embed_indices: list[int] = []

    for i, h in enumerate(hashes):
        cached = _get_cached(cache_conn, h)
        if cached is not None:
            vectors[i] = cached
        else:
            to_embed_indices.append(i)

    cache_hits = len(papers) - len(to_embed_indices)
    tokens_billed = 0

    if to_embed_indices:
        new_texts = [texts_and_fallback[i][0] for i in to_embed_indices]
        new_vectors, tokens_billed = _embed_texts(client, new_texts)
        for idx, vec in zip(to_embed_indices, new_vectors):
            vectors[idx] = vec
            _set_cached(cache_conn, hashes[idx], vec, len(texts_and_fallback[idx][0]))

    cache_conn.close()

    estimated_cost = tokens_billed / 1_000_000 * PRICE_PER_1M_TOKENS
    logger.info(
        "embed_and_index_papers: %d cache hit(s), %d newly embedded, %d tokens billed (~$%.6f)",
        cache_hits, len(to_embed_indices), tokens_billed, estimated_cost,
    )

    collection.upsert(
        ids=[p.paper_id for p in papers],
        embeddings=vectors,
        documents=[text for text, _ in texts_and_fallback],
        metadatas=[_serialize_metadata(p, fallback) for p, (_, fallback) in zip(papers, texts_and_fallback)],
    )

    return {
        "cache_hits": cache_hits,
        "cache_misses": len(to_embed_indices),
        "tokens_billed": tokens_billed,
        "estimated_cost_usd": estimated_cost,
    }


def semantic_search(
    query: str,
    collection=None,
    client: OpenAI | None = None,
    top_k: int = 10,
    where: dict | None = None,
) -> list[tuple[Paper, float]]:
    """Embed the query and retrieve the top_k most similar indexed papers.

    Returns (Paper, similarity) pairs, similarity in [0, 1] (cosine
    similarity — Chroma reports cosine *distance*, so similarity = 1 - distance).

    `where` is a Chroma metadata filter (e.g. `{"paper_id": {"$in": [...]}}`)
    to scope retrieval to a subset of the persistent collection — the agent
    (phase 4) uses this so reranking a search only considers papers gathered
    in the current session, not every paper ever indexed across past runs.
    """
    collection = collection or get_chroma_collection()
    client = client or OpenAI()

    if collection.count() == 0:
        logger.warning("semantic_search: collection is empty, nothing to retrieve")
        return []

    query_vector, tokens_billed = _embed_texts(client, [query])
    cost = tokens_billed / 1_000_000 * PRICE_PER_1M_TOKENS
    logger.info("Embedded query %r: %d tokens billed (~$%.6f)", query, tokens_billed, cost)

    n_results = min(top_k, collection.count())
    results = collection.query(
        query_embeddings=query_vector,
        n_results=n_results,
        where=where,
        include=["metadatas", "documents", "distances"],
    )

    papers_with_scores = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        similarity = 1.0 - distance
        papers_with_scores.append((_paper_from_metadata(metadata), similarity))
    return papers_with_scores


def get_papers_by_ids(paper_ids: list[str], collection=None) -> list[Paper]:
    """Fetch previously-indexed papers back out of Chroma by id, in the
    given order. Used by the phase-7 API to resolve a saved search's
    paper_ids (from SQLite) back into full Paper objects across separate
    HTTP requests — Chroma is the persistence layer for paper content, so
    nothing needs to be re-fetched from arXiv/Semantic Scholar or re-embedded.

    Chroma's own .get() does not guarantee its result order matches the
    input id list, so results are re-ordered here to match paper_ids.
    """
    if not paper_ids:
        return []
    collection = collection or get_chroma_collection()
    result = collection.get(ids=paper_ids, include=["metadatas"])
    by_id = {pid: _paper_from_metadata(meta) for pid, meta in zip(result["ids"], result["metadatas"])}
    return [by_id[pid] for pid in paper_ids if pid in by_id]
