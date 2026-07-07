#!/usr/bin/env python3
"""Phase 3 sanity check: index a candidate pool, then compare naive keyword
ranking against embedding-based semantic search for an ambiguous query.

The candidate pool is fetched with a single ambiguous word ("star" — which
returns both astrophysics papers about stellar formation and unrelated
graph-theory/combinatorics papers about "star graphs"/"star colorings" on
arXiv). We then rank that same pool two ways for the user's actual, more
specific intent, to show semantic search resolving the ambiguity that plain
keyword overlap can't: a graph-theory abstract that repeats the word "star"
many times can out-score a genuine astrophysics paper on raw term frequency
alone, since naive keyword overlap has no notion of topical relevance.

The keyword-overlap ranker here is a throwaway comparison baseline for this
demo only — it is NOT part of the shipped pipeline. Per the brief, retrieval
via embeddings *is* the relevance ranking; no separate scoring system is
layered on top of it in research_agent/.

Usage:
    python scripts/test_ranking.py
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from research_agent.dedup import deduplicate
from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, semantic_search
from research_agent.ingestion import search_arxiv, search_semantic_scholar
from research_agent.schema import Paper

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

POOL_QUERY = "star"
REFINED_QUERY = "star formation in molecular clouds"
STOPWORDS = {"a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "is", "with", "'s"}


def keyword_rank(query: str, papers: list[Paper]) -> list[tuple[Paper, int]]:
    """Naive bag-of-words overlap ranking — the 'plain keyword search' baseline."""
    terms = [w for w in re.findall(r"[a-z0-9']+", query.lower()) if w not in STOPWORDS]
    scored = []
    for p in papers:
        text = f"{p.title} {p.abstract or ''}".lower()
        score = sum(text.count(t) for t in terms)
        scored.append((p, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _print_ranked(label: str, ranked: list[tuple[Paper, float]], top_n: int = 5) -> None:
    print(f"\n{'=' * 80}\n{label}\n{'=' * 80}")
    for i, (p, score) in enumerate(ranked[:top_n], 1):
        print(f"[{i}] (score={score:.4f}) {p.title}")
        print(f"    {(p.abstract or '(no abstract)')[:140]}")


def main() -> None:
    load_dotenv()
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    client = OpenAI()

    print(f"Fetching candidate pool for ambiguous query: {POOL_QUERY!r}")
    arxiv_papers = search_arxiv(POOL_QUERY, max_results=40)
    s2_papers = search_semantic_scholar(POOL_QUERY, max_results=20, api_key=s2_key)
    pool = deduplicate(arxiv_papers + s2_papers)
    pool = [p for p in pool if p.abstract]  # keyword ranker needs real abstract text to be a fair comparison
    print(f"Candidate pool: {len(pool)} papers (after dedup, abstract required)")

    # Isolated, throwaway persist dir so leftover data from previous demo runs
    # (different ambiguous query) doesn't pollute this comparison's top-k.
    demo_persist_dir = Path(__file__).resolve().parent.parent / "data" / "demo_chroma_db"
    shutil.rmtree(demo_persist_dir, ignore_errors=True)
    collection = get_chroma_collection(persist_dir=demo_persist_dir)
    stats = embed_and_index_papers(pool, collection=collection, client=client)
    print(
        f"Indexed pool: {stats['cache_hits']} cache hit(s), {stats['cache_misses']} newly embedded, "
        f"{stats['tokens_billed']} tokens billed (~${stats['estimated_cost_usd']:.6f})"
    )

    print(f"\nRefined query (user's actual intent): {REFINED_QUERY!r}")

    kw_ranked = keyword_rank(REFINED_QUERY, pool)
    _print_ranked("BEFORE — naive keyword overlap ranking", kw_ranked)

    semantic_ranked = semantic_search(REFINED_QUERY, collection=collection, client=client, top_k=len(pool))
    _print_ranked("AFTER — embedding-based semantic search (cosine similarity)", semantic_ranked)


if __name__ == "__main__":
    main()
