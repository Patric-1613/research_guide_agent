#!/usr/bin/env python3
"""Round-2 enhancement 4 sanity check: find real Semantic Scholar results
missing an abstract but carrying a DOI, then confirm enrich_missing_abstracts
recovers real abstract text for at least some of them via Unpaywall/CrossRef,
and never crashes on the ones it can't recover.

With no arguments, runs a deterministic check against a specific DOI already
confirmed to have a recoverable CrossRef abstract, and asserts recovery
actually succeeds — a topic-driven candidate pool depends on whatever real
papers happen to turn up for that query having a recoverable abstract in
Unpaywall/CrossRef, which isn't guaranteed even when enrichment itself is
working correctly, so that path only gets the weaker "did the return value
match reality" assertion, not "recovery must succeed."

Usage:
    python scripts/test_enrichment.py                    # deterministic --doi check (default)
    python scripts/test_enrichment.py --topic "<topic>"
    python scripts/test_enrichment.py --doi <doi>
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
# Confirmed (manually, against the live CrossRef API) to have a real,
# recoverable abstract — a deterministic positive case for this demo,
# independent of which candidates a topic search happens to surface.
DEFAULT_DOI = "10.58496/bjml/2023/006"


def _run_doi_check(doi: str) -> None:
    print(f"Recovering abstract for DOI {doi!r} directly...")
    abstract = recover_abstract(doi)
    print(f"\nResult: {abstract or '(unrecoverable)'}")

    if doi == DEFAULT_DOI:
        assert abstract, f"expected a recoverable abstract for the known-good DOI {doi!r}, got none"
        assert len(abstract) > 40, f"recovered abstract looks suspiciously short: {abstract!r}"
        print(f"\nPASS: recovered a {len(abstract)}-character abstract for a known-good DOI.")


def main() -> None:
    load_dotenv()

    if len(sys.argv) > 1 and sys.argv[1] == "--doi":
        doi = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DOI
        _run_doi_check(doi)
        return

    if len(sys.argv) == 1:
        _run_doi_check(DEFAULT_DOI)
        return

    topic = sys.argv[2] if sys.argv[1] == "--topic" and len(sys.argv) > 2 else DEFAULT_TOPIC
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

    # A real assertion, not a hardcoded PASS — but only the part that's
    # actually deterministic: the return value must match how many
    # candidates ended up with an abstract. Whether recovery finds >0 real
    # abstracts for THIS topic's specific candidates depends on live
    # Unpaywall/CrossRef coverage, not just enrichment.py's correctness — see
    # the deterministic --doi check above for that guarantee instead.
    actually_recovered = sum(1 for p in candidates if p.abstract)
    assert recovered == actually_recovered, (
        f"enrich_missing_abstracts() reported recovering {recovered}, but {actually_recovered} "
        "candidate(s) actually ended up with an abstract"
    )
    print(f"\nPASS: recovered {recovered}/{len(candidates)} real abstract(s), return value matches reality, never crashed.")


if __name__ == "__main__":
    main()
