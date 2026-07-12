#!/usr/bin/env python3
"""Round 3, phase 6: a small, hand-curated retrieval-quality sanity check.

Runs each topic in eval_set.json through the actual retrieval pipeline —
search_arxiv + search_semantic_scholar, dedup.deduplicate, abstract
enrichment, embed_and_index_papers, and embeddings.semantic_search's
cosine-similarity ranking — the same functions the app's search flows use,
called directly rather than through the phase-4 LLM agent (which only adds
source-selection/query-reformulation judgment on top; the actual relevance
ranking being evaluated here is entirely the embedding step, unchanged
since round 2 and untouched by this round's UI/orchestration changes). For
each topic it reports what fraction of the curated "expected" papers (matched
by title, not the source-specific paper_id, since those aren't stable across
live API calls) turn up in the top-k ranked results.

This is a manual sanity-check tool, not a pytest test: it costs real
OpenAI embedding-API money per run and depends on live, rate-limited arXiv/
Semantic Scholar responses, so it deliberately isn't part of the automated
suite (tests/) or CI (.github/workflows/tests.yml).

Uses its own isolated Chroma persist dir (data/eval_chroma_db), separate
from both the app's real dev collection (data/chroma_db) and the ad-hoc
demo script's throwaway one (data/demo_chroma_db) — but unlike the demo
script, this one is NOT wiped between runs: the whole point of running this
repeatedly over time is to catch a regression, and embeddings.py's
content-hash cache means a second run against the same curated topics is
nearly free.

Usage:
    python scripts/eval_retrieval.py [--top-k 10] [--max-results-per-source 15]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz

from research_agent.dedup import deduplicate
from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, semantic_search
from research_agent.enrichment import enrich_missing_abstracts
from research_agent.ingestion import search_arxiv, search_semantic_scholar
from research_agent.schema import Paper

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"
EVAL_PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_chroma_db"

# Same normalize-then-fuzzy-ratio technique as dedup.py's own title matching,
# just a touch more lenient (85 vs. dedup's 90): here we're matching a
# hand-typed canonical title against whatever arXiv/Semantic Scholar actually
# returns, which can have minor punctuation/subtitle differences that would
# never occur between two records of the SAME paper from two APIs.
TITLE_MATCH_THRESHOLD = 85

# A pass threshold this crude (1-2 expected titles per topic) means "found
# at least 80%" is effectively "found all of them" — deliberately strict,
# since every expected title here is a specific, extremely well-known,
# canonical paper for its topic that a working retrieval pipeline should
# reliably surface, not a fuzzy "reasonable coverage" bar.
PASS_FRACTION = 0.8


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def _title_matches(expected: str, candidate: str) -> bool:
    e, c = _normalize_title(expected), _normalize_title(candidate)
    if e in c or c in e:
        return True
    return fuzz.ratio(e, c) >= TITLE_MATCH_THRESHOLD


def _retrieve(topic: str, top_k: int, max_results_per_source: int, client: OpenAI, collection) -> list[Paper]:
    arxiv_papers = search_arxiv(topic, max_results=max_results_per_source)
    s2_papers = search_semantic_scholar(topic, max_results=max_results_per_source, api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None)
    pool = deduplicate(arxiv_papers + s2_papers)
    enrich_missing_abstracts(pool)
    if not pool:
        return []

    embed_and_index_papers(pool, collection=collection, client=client)
    ids = [p.paper_id for p in pool]
    ranked = semantic_search(topic, collection=collection, client=client, top_k=top_k, where={"paper_id": {"$in": ids}})
    return [p for p, _ in ranked]


def run_eval(top_k: int, max_results_per_source: int) -> list[dict]:
    load_dotenv()
    eval_set = json.loads(EVAL_SET_PATH.read_text())
    client = OpenAI()
    collection = get_chroma_collection(persist_dir=EVAL_PERSIST_DIR)

    results = []
    for entry in eval_set:
        topic = entry["topic"]
        expected_titles = entry["expected_titles"]
        ranked_papers = _retrieve(topic, top_k, max_results_per_source, client, collection)

        matched = [e for e in expected_titles if any(_title_matches(e, p.title) for p in ranked_papers)]
        missing = [e for e in expected_titles if e not in matched]
        fraction = len(matched) / len(expected_titles) if expected_titles else 1.0

        results.append({
            "topic": topic,
            "expected_titles": expected_titles,
            "matched": matched,
            "missing": missing,
            "fraction": fraction,
            "passed": fraction >= PASS_FRACTION,
            "num_candidates": len(ranked_papers),
        })
    return results


def _print_report(results: list[dict], top_k: int) -> None:
    print(f"\n{'=' * 88}\nRETRIEVAL EVALUATION — top_k={top_k}\n{'=' * 88}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{status}] {r['topic']!r}  ({len(r['matched'])}/{len(r['expected_titles'])} expected found, "
              f"{r['fraction']:.0%}, {r['num_candidates']} candidates ranked)")
        for title in r["matched"]:
            print(f"    ✓ {title}")
        for title in r["missing"]:
            print(f"    ✗ {title}  (NOT found in top {top_k})")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    avg_fraction = sum(r["fraction"] for r in results) / total if total else 0.0
    print(f"\n{'=' * 88}")
    print(f"SUMMARY: {passed}/{total} topics found at least {PASS_FRACTION:.0%} of expected papers in the top {top_k}.")
    print(f"Average fraction of expected papers found across all topics: {avg_fraction:.0%}.")
    print(f"{'=' * 88}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=10, help="Ranked results considered per topic (default: 10).")
    parser.add_argument("--max-results-per-source", type=int, default=15, help="Raw results fetched per source before ranking (default: 15).")
    args = parser.parse_args()

    results = run_eval(args.top_k, args.max_results_per_source)
    _print_report(results, args.top_k)


if __name__ == "__main__":
    main()
