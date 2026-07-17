#!/usr/bin/env python3
"""Phase 2 sanity check: search both sources, dedupe, and show the collapse.

Also asserts the dedup actually did its job, not just that it ran: no
duplicate titles survive into the merged pool, the merged pool never grows
past the combined input, and (for the default query, known to return the
same well-known RAG paper from both arXiv and Semantic Scholar) at least one
record actually got merged across sources.

Usage:
    python scripts/test_dedup.py ["retrieval augmented generation for knowledge intensive nlp tasks"] [max_results]
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

# The exact title of a well-known paper — likely to be independently found
# by both arXiv and Semantic Scholar, so dedup has a genuine cross-source
# merge to do, not just a no-op pass over already-unique records.
DEFAULT_QUERY = "retrieval augmented generation for knowledge intensive nlp tasks"


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
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
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

    # Structural correctness, not just "didn't crash": dedup can only ever
    # shrink or preserve the input count, never grow it, and no two
    # surviving records should share the same normalized title.
    assert len(merged) <= len(combined), (
        f"dedup grew the pool ({len(combined)} -> {len(merged)}) — it must only merge/shrink, never add records"
    )
    titles = [p.title.strip().lower() for p in merged]
    assert len(titles) == len(set(titles)), f"duplicate titles survived dedup: {titles}"

    if query.strip().lower() == DEFAULT_QUERY:
        assert n_merged >= 1, (
            f"expected at least one cross-source merge for the well-known query {DEFAULT_QUERY!r} "
            f"(same paper should turn up on both arXiv and Semantic Scholar), but none merged"
        )
        print(f"\nPASS: {n_merged} record(s) merged, no duplicate titles survived, pool did not grow.")


if __name__ == "__main__":
    main()
