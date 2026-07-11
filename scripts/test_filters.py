#!/usr/bin/env python3
"""Round-2 enhancement 2 sanity check: run the agent with a min-citation-count
filter and a DOI-required filter and confirm the returned papers actually
respect both constraints.

Usage:
    python scripts/test_filters.py ["<topic>"]
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from research_agent.agent import run_research_agent

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_TOPIC = "retrieval augmented generation for large language models"
MIN_CITATIONS = 100


def main() -> None:
    load_dotenv()
    topic = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None

    print(f"{'=' * 80}\nNo filters (baseline)\n{'=' * 80}")
    baseline = run_research_agent(topic, s2_api_key=s2_key, top_k=15)
    for p, score in baseline.ranked:
        print(f"  ({score:.3f}) citations={p.citation_count} doi={p.doi} — {p.title}")

    print(f"\n{'=' * 80}\nmin_citation_count={MIN_CITATIONS}\n{'=' * 80}")
    citation_filtered = run_research_agent(topic, s2_api_key=s2_key, top_k=15, min_citation_count=MIN_CITATIONS)
    for p, score in citation_filtered.ranked:
        print(f"  ({score:.3f}) citations={p.citation_count} — {p.title}")
    violations = [p for p, _ in citation_filtered.ranked if (p.citation_count or 0) < MIN_CITATIONS]
    print(f"Violations (citation_count < {MIN_CITATIONS}): {len(violations)}")

    print(f"\n{'=' * 80}\ndoi_required=True\n{'=' * 80}")
    doi_filtered = run_research_agent(topic, s2_api_key=s2_key, top_k=15, doi_required=True)
    for p, score in doi_filtered.ranked:
        print(f"  ({score:.3f}) doi={p.doi} — {p.title}")
    violations_doi = [p for p, _ in doi_filtered.ranked if not p.doi]
    print(f"Violations (doi is None): {len(violations_doi)}")

    ok = not violations and not violations_doi
    print(f"\n{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
