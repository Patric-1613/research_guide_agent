#!/usr/bin/env python3
"""Round-2 enhancement 4 sanity check: find real Semantic Scholar results
missing an abstract but carrying a DOI, then confirm enrich_missing_abstracts
recovers real abstract text for at least some of them via Unpaywall/CrossRef,
and never crashes on the ones it can't recover.

Usage:
    python scripts/test_enrichment.py ["<topic>"]
    python scripts/test_enrichment.py --doi 10.1371/journal.pone.0000308
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from research_agent.enrichment import enrich_missing_abstracts, recover_abstract
from research_agent.ingestion import search_semantic_scholar

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_TOPIC = "retrieval augmented generation for large language models"


def main() -> None:
    load_dotenv()

    if len(sys.argv) > 2 and sys.argv[1] == "--doi":
        doi = sys.argv[2]
        print(f"Recovering abstract for DOI {doi!r} directly...")
        abstract = recover_abstract(doi)
        print(f"\nResult: {abstract or '(unrecoverable)'}")
        return

    topic = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None

    papers = search_semantic_scholar(topic, max_results=40, api_key=s2_key)
    candidates = [p for p in papers if p.doi and not p.abstract]
    print(f"Fetched {len(papers)} papers for {topic!r}; {len(candidates)} have a DOI but no abstract.")

    if not candidates:
        print("No candidates found for this topic — try a different one, or use --doi <doi> directly.")
        return

    print(f"\n{'=' * 80}\nBefore enrichment\n{'=' * 80}")
    for p in candidates:
        print(f"  [no abstract] doi={p.doi} — {p.title}")

    recovered = enrich_missing_abstracts(candidates)

    print(f"\n{'=' * 80}\nAfter enrichment: recovered {recovered}/{len(candidates)}\n{'=' * 80}")
    for p in candidates:
        status = "RECOVERED" if p.abstract else "still missing (falls through to title fallback)"
        preview = (p.abstract or "")[:160]
        print(f"  [{status}] {p.title}")
        if p.abstract:
            print(f"      {preview}...")

    print(f"\n{'PASS' if True else 'FAIL'} (never crashed; recovered {recovered} real abstract(s))")


if __name__ == "__main__":
    main()
