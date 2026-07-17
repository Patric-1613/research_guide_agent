#!/usr/bin/env python3
"""Retrieval evaluation: does the app's real search pipeline actually
surface the papers a human confirmed are relevant, for a fixed set of test
topics (eval_data/reference_topics.json, produced by
scripts/convert_reference_sheet.py from the human-curated reference
spreadsheet)?

Calls the underlying ingestion/dedup/embeddings functions directly — the
same search_arxiv() + search_semantic_scholar() + deduplicate() +
semantic_search() flow scripts/test_qa.py and scripts/ragas_eval.py already
use — NOT the LLM agent (agent.py). This measures retrieval quality
specifically, not agent orchestration; the agent adds its own query
reformulation and tool-use decisions that would confound what's being
measured here. No mocked data: every precision/recall number comes from a
real pipeline call, matched against expected papers using dedup.py's own
_same_paper() (DOI exact match first, fuzzy title fallback) — reused
directly, not reimplemented.

Not part of pytest or CI. Makes real, billable OpenAI embedding calls (for
indexing candidates + the query embedding) plus real arXiv/Semantic
Scholar API calls, once per topic.

Usage:
    python scripts/eval_retrieval.py --note "phase 1 baseline"
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from research_agent.dedup import _same_paper, deduplicate
from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, semantic_search
from research_agent.ingestion import search_arxiv, search_semantic_scholar
from research_agent.schema import Paper

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_JSON = os.path.join(REPO_ROOT, "eval_data", "reference_topics.json")
HISTORY_CSV = os.path.join(REPO_ROOT, "eval_results", "retrieval_history.csv")

# Matches semantic_search()'s own default and api.py's SearchRequest
# default top_k — the value the app itself uses for a plain search, not an
# inflated number chosen to flatter recall.
TOP_K = 10

DIFFICULTY_TIERS = ["easy", "ambiguous", "thin", "domain"]

HISTORY_FIELDS = [
    "run_id", "date", "git_commit", "note", "num_topics", "top_k",
    "mean_precision", "mean_recall",
    "recall_easy", "recall_ambiguous", "recall_thin", "recall_domain",
]


def _expected_to_paper(expected: dict) -> Paper:
    """Synthetic Paper built from a reference row — just enough for
    dedup.py's _same_paper() to compare against (title + doi are all it
    looks at; it doesn't consider arxiv_id)."""
    return Paper(
        title=expected["title"], authors=[], year=None, venue=None, abstract=None,
        url=None, doi=expected.get("doi"), citation_count=None, source="reference",
    )


def load_reference_topics() -> dict:
    with open(REFERENCE_JSON) as f:
        return json.load(f)


def run_topic_retrieval(topic: str, client: OpenAI) -> list[Paper]:
    """Real search/dedup/rerank pass, no agent — search_arxiv() +
    search_semantic_scholar() + deduplicate() + semantic_search(), each
    called with its own function default (no override), matching the app's
    plain search path when nothing inflates the candidate pool."""
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    arxiv_papers = search_arxiv(topic)
    s2_papers = search_semantic_scholar(topic, api_key=s2_key)
    pool = deduplicate(arxiv_papers + s2_papers)
    pool = [p for p in pool if p.abstract]

    collection = get_chroma_collection()
    embed_and_index_papers(pool, collection=collection, client=client)
    ids = [p.paper_id for p in pool]
    ranked = semantic_search(
        topic, collection=collection, client=client, top_k=TOP_K,
        where={"paper_id": {"$in": ids}},
    )
    return [p for p, _ in ranked]


def evaluate_topic(topic_id: str, topic_data: dict, client: OpenAI) -> dict:
    expected = topic_data["expected_papers"]
    expected_papers = [_expected_to_paper(e) for e in expected]

    returned = run_topic_retrieval(topic_data["topic"], client)

    matched_expected_indices: set[int] = set()
    matched_returned_count = 0
    for r in returned:
        matched_any = False
        for i, exp_paper in enumerate(expected_papers):
            if _same_paper(r, exp_paper):
                matched_any = True
                matched_expected_indices.add(i)
        if matched_any:
            matched_returned_count += 1

    precision = matched_returned_count / len(returned) if returned else 0.0
    recall = len(matched_expected_indices) / len(expected) if expected else 0.0
    missed = [expected[i] for i in range(len(expected)) if i not in matched_expected_indices]

    return {
        "topic_id": topic_id,
        "topic": topic_data["topic"],
        "difficulty": topic_data["difficulty"],
        "num_expected": len(expected),
        "num_returned": len(returned),
        "precision": precision,
        "recall": recall,
        "missed": missed,
    }


def print_results_table(results: list[dict]) -> None:
    print("=" * 100)
    print("Retrieval evaluation results (per topic)")
    print("=" * 100)
    for r in results:
        print(f"\n[{r['topic_id']}] ({r['difficulty']}) {r['topic']}")
        print(f"    precision@{TOP_K}: {r['precision']:.3f}  ({r['num_returned']} returned)")
        print(f"    recall@{TOP_K}:    {r['recall']:.3f}  ({r['num_expected']} expected)")
        if r["missed"]:
            print(f"    missed {len(r['missed'])} expected paper(s):")
            for m in r["missed"]:
                weak = " [weaker signal: has_abstract=no in reference data]" if m.get("has_abstract") == "no" else ""
                print(f"      - {m['title']}{weak}")

    mean_precision = sum(r["precision"] for r in results) / len(results)
    mean_recall = sum(r["recall"] for r in results) / len(results)
    print("\n" + "=" * 100)
    print("Overall means")
    print("=" * 100)
    print(f"  mean precision@{TOP_K}: {mean_precision:.3f}")
    print(f"  mean recall@{TOP_K}:    {mean_recall:.3f}")

    print("\nRecall by difficulty tier (an overall average would hide whether misses cluster in harder topics):")
    for tier in DIFFICULTY_TIERS:
        tier_results = [r for r in results if r["difficulty"] == tier]
        if not tier_results:
            print(f"  {tier}: (no topics)")
            continue
        tier_recall = sum(r["recall"] for r in tier_results) / len(tier_results)
        print(f"  {tier}: {tier_recall:.3f}  ({len(tier_results)} topics)")


def append_history_row(results: list[dict], note: str) -> dict:
    os.makedirs(os.path.dirname(HISTORY_CSV), exist_ok=True)
    file_exists = os.path.isfile(HISTORY_CSV)

    next_run_id = 1
    if file_exists:
        with open(HISTORY_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            next_run_id = max(int(r["run_id"]) for r in rows) + 1

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()

    mean_precision = sum(r["precision"] for r in results) / len(results)
    mean_recall = sum(r["recall"] for r in results) / len(results)

    tier_recalls: dict[str, float | None] = {}
    for tier in DIFFICULTY_TIERS:
        tier_results = [r for r in results if r["difficulty"] == tier]
        tier_recalls[tier] = (sum(r["recall"] for r in tier_results) / len(tier_results)) if tier_results else None

    row = {
        "run_id": next_run_id,
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "note": note,
        "num_topics": len(results),
        "top_k": TOP_K,
        "mean_precision": f"{mean_precision:.4f}",
        "mean_recall": f"{mean_recall:.4f}",
        "recall_easy": f"{tier_recalls['easy']:.4f}" if tier_recalls["easy"] is not None else "",
        "recall_ambiguous": f"{tier_recalls['ambiguous']:.4f}" if tier_recalls["ambiguous"] is not None else "",
        "recall_thin": f"{tier_recalls['thin']:.4f}" if tier_recalls["thin"] is not None else "",
        "recall_domain": f"{tier_recalls['domain']:.4f}" if tier_recalls["domain"] is not None else "",
    }

    with open(HISTORY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--note", required=True, help="Short description of this run, e.g. 'phase 1 baseline'")
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI()

    topics = load_reference_topics()

    skipped = sorted(tid for tid, t in topics.items() if not t["expected_papers"])
    usable = {tid: t for tid, t in topics.items() if t["expected_papers"]}
    if skipped:
        print(f"Skipping {len(skipped)} topic(s) with zero expected papers: {skipped}")

    print(f"Evaluating {len(usable)} topics at top_k={TOP_K}\n")

    results = []
    for i, (topic_id, topic_data) in enumerate(usable.items(), 1):
        print(f"[{i}/{len(usable)}] {topic_id}: {topic_data['topic']!r}")
        r = evaluate_topic(topic_id, topic_data, client)
        print(f"    precision={r['precision']:.3f} recall={r['recall']:.3f}\n")
        results.append(r)

    print_results_table(results)
    row = append_history_row(results, args.note)

    print("\n" + "=" * 100)
    print(f"Appended run {row['run_id']} to {os.path.relpath(HISTORY_CSV, REPO_ROOT)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
