#!/usr/bin/env python3
"""Phase 1 sanity check: run both searches for a topic and print results
side by side so the normalized schema can be eyeballed before anything is
built on top of it.

Usage:
    python scripts/test_ingestion.py "retrieval augmented generation" [max_results]
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

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    query = sys.argv[1]
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


if __name__ == "__main__":
    main()
