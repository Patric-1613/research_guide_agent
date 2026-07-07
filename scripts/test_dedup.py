#!/usr/bin/env python3
"""Phase 2 sanity check: search both sources, dedupe, and show the collapse.

Usage:
    python scripts/test_dedup.py "retrieval augmented generation for knowledge intensive nlp tasks" [max_results]
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from research_agent.dedup import deduplicate
from research_agent.ingestion import search_arxiv, search_semantic_scholar
from research_agent.schema import Paper

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _print_paper(p: Paper, indent: str = "") -> None:
    print(f"{indent}title:      {p.title}")
    print(f"{indent}authors:    {', '.join(p.authors)}")
    print(f"{indent}year/venue: {p.year} / {p.venue}")
    print(f"{indent}citations:  {p.citation_count}   doi: {p.doi}")
    print(f"{indent}source:     {p.source}")
    print(f"{indent}source_urls:")
    for src, url in p.source_urls.items():
        print(f"{indent}    {src}: {url}")


def main() -> None:
    load_dotenv()
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    query = sys.argv[1]
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None

    arxiv_papers = search_arxiv(query, max_results=max_results)
    s2_papers = search_semantic_scholar(query, max_results=max_results, api_key=s2_key)
    combined = arxiv_papers + s2_papers

    print(f"Query: {query!r}")
    print(f"Before dedup: {len(arxiv_papers)} arXiv + {len(s2_papers)} Semantic Scholar = {len(combined)} total\n")

    merged = deduplicate(combined)

    print(f"After dedup: {len(merged)} record(s)\n{'=' * 80}")
    for i, p in enumerate(merged, 1):
        is_merged = "+" in p.source
        tag = " [MERGED]" if is_merged else ""
        print(f"\n[{i}]{tag}")
        _print_paper(p, indent="    ")

    n_merged = sum(1 for p in merged if "+" in p.source)
    print(f"\n{'=' * 80}")
    print(f"{n_merged} record(s) were merged from multiple sources.")


if __name__ == "__main__":
    main()
