#!/usr/bin/env python3
"""Phase 1 sanity check: run both searches for a topic and print results
side by side so the normalized schema can be eyeballed before anything is
built on top of it.

Also asserts the search actually found the right thing, not just that it
didn't crash: with no query given, defaults to a query that should surface
one specific, extremely well-known paper ("Attention Is All You Need") by
title, and checks it's actually there.

Usage:
    python scripts/test_ingestion.py ["retrieval augmented generation"] [max_results]
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from research_agent.ingestion import search_arxiv, search_semantic_scholar
from research_agent.schema import Paper

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# A famous, exact-title-match query — if this doesn't come back, something in
# the search/normalization path is broken, not just "the query was too niche."
DEFAULT_QUERY = "attention is all you need"


def _print_table(source_label: str, papers: list[Paper]) -> None:
    print(f"\n{'=' * 80}\n{source_label} ({len(papers)} results)\n{'=' * 80}")
    if not papers:
        print("  (no results)")
        return
    for i, p in enumerate(papers, 1):
        abstract_preview = (p.abstract or "(no abstract)")[:160]
        print(f"\n[{i}] {p.title}")
        print(f"    authors:   {', '.join(p.authors) or '(none listed)'}")
        print(f"    year:      {p.year}   venue: {p.venue}")
        print(f"    citations: {p.citation_count}   doi: {p.doi}")
        print(f"    url:       {p.url}")
        print(f"    abstract:  {textwrap.shorten(abstract_preview, width=160)}")


def main() -> None:
    load_dotenv()

    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None

    print(f"Query: {query!r}  max_results={max_results}")

    arxiv_papers = search_arxiv(query, max_results=max_results)
    _print_table("arXiv", arxiv_papers)

    s2_papers = search_semantic_scholar(query, max_results=max_results, api_key=s2_key)
    _print_table("Semantic Scholar", s2_papers)

    print(f"\n{'=' * 80}\nRaw normalized JSON\n{'=' * 80}")
    combined = {
        "query": query,
        "arxiv": [p.to_dict() for p in arxiv_papers],
        "semantic_scholar": [p.to_dict() for p in s2_papers],
    }
    print(json.dumps(combined, indent=2))

    if query.strip().lower() == DEFAULT_QUERY:
        all_papers = arxiv_papers + s2_papers
        assert all_papers, f"expected at least one result for {DEFAULT_QUERY!r}, got none from either source"
        assert any("attention is all you need" in p.title.lower() for p in all_papers), (
            f"expected to find the paper 'Attention Is All You Need' for query {DEFAULT_QUERY!r}, "
            f"but got titles: {[p.title for p in all_papers]}"
        )
        # Every normalized record must carry the fields citations.py and
        # embeddings.py assume are always present, not just non-crashing ones.
        for p in all_papers:
            assert p.title.strip(), "normalized Paper has an empty title"
            assert p.paper_id, "normalized Paper has an empty paper_id"
            assert p.source in ("arxiv", "semantic_scholar"), f"unexpected source: {p.source!r}"
        print("\nPASS: found 'Attention Is All You Need' and every result has well-formed required fields.")


if __name__ == "__main__":
    main()
