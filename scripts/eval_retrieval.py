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
from langfuse import get_client, observe
from openai import OpenAI

from research_agent.dedup import _same_paper, deduplicate
from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, semantic_search
from research_agent.ingestion import search_arxiv, search_semantic_scholar
from research_agent.query_expansion import build_candidate_pool, expanded_search
from research_agent.ranking import (
    bm25_search,
    get_partition_n,
    hybrid_search,
    merge_with_guaranteed_slots,
    partition_by_citation,
)
from research_agent.schema import Paper

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_JSON = os.path.join(REPO_ROOT, "eval_data", "reference_topics.json")
HISTORY_CSV = os.path.join(REPO_ROOT, "eval_results", "retrieval_history.csv")

# Matches semantic_search()'s own default and api.py's SearchRequest
# default top_k — the value the app itself uses for a plain search, not an
# inflated number chosen to flatter recall. Overridable per-run via
# --top-k (k-generalization experiment) — every existing call site below
# now takes top_k as a runtime parameter instead of reading this constant
# directly, so omitting --top-k reproduces the exact k=10 behavior/results
# already established, while passing it lets the SAME harness test other k
# values without touching query_expansion.py or ranking.py's partition/
# merge logic at all — this is a harness-level parameter, not a change to
# either.
DEFAULT_TOP_K = 10

DIFFICULTY_TIERS = ["easy", "ambiguous", "thin", "domain"]

HISTORY_FIELDS = [
    "run_id", "date", "git_commit", "note", "num_topics", "top_k", "ranking_mode", "partition_n",
    "mean_precision", "mean_recall",
    "recall_easy", "recall_ambiguous", "recall_thin", "recall_domain",
]

RANKING_MODES = ["semantic", "hybrid", "bm25", "citation_partition"]


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


def run_topic_retrieval(topic: str, client: OpenAI, top_k: int = DEFAULT_TOP_K) -> list[Paper]:
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
        topic, collection=collection, client=client, top_k=top_k,
        where={"paper_id": {"$in": ids}},
    )
    return [p for p, _ in ranked]


def run_topic_retrieval_expanded(topic: str, client: OpenAI, top_k: int = DEFAULT_TOP_K) -> list[Paper]:
    """Same evaluation target, but via query_expansion.py's expanded_search()
    instead of the direct search_arxiv/search_semantic_scholar/deduplicate/
    semantic_search flow above — LLM-suggested paper titles widen the
    candidate pool before the identical dedup + rerank-against-original-topic
    steps. See research_agent/query_expansion.py for the full mechanism."""
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    ranked = expanded_search(topic, k=top_k, s2_api_key=s2_key, client=client)
    return [p for p, _ in ranked]


def run_topic_retrieval_ranked(
    topic: str, client: OpenAI, ranking_mode: str, partition_n: int | None = None, top_k: int = DEFAULT_TOP_K,
) -> list[Paper]:
    """Ranking-stage experiment: reuses query_expansion.py's
    build_candidate_pool() UNCHANGED (same locked pool-size parameters,
    same suggest_related_titles() call, same dedup) — only the final
    ranking step differs between 'semantic' (embeddings.py's cosine
    similarity — identical algorithm to run_topic_retrieval_expanded()'s
    own default), 'bm25' (lexical scoring, research_agent/ranking.py),
    'hybrid' (RRF fusion of both, also ranking.py), or 'citation_partition'
    (citation-count-partitioned reranking with a guaranteed slot count,
    also ranking.py — see partition_by_citation()/merge_with_guaranteed_
    slots() there for the actual mechanism). Every mode ranks against
    `topic` itself, never an LLM-suggested title — the anti-hallucination
    anchor query_expansion.py's docstring establishes, enforced
    identically across every mode here, not just semantic.

    'bm25' skips embedding/indexing entirely — BM25 is pure in-memory
    lexical scoring over the pool's own text, so there's no vector index
    to build for it, unlike the other three modes.
    """
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    pool = build_candidate_pool(topic, k=top_k, s2_api_key=s2_key, client=client)
    pool = [p for p in pool if p.abstract]

    if ranking_mode == "bm25":
        ranked = bm25_search(topic, pool, top_k=top_k)
        return [p for p, _ in ranked]

    collection = get_chroma_collection()
    embed_and_index_papers(pool, collection=collection, client=client)
    ids = [p.paper_id for p in pool]

    if ranking_mode == "semantic":
        ranked = semantic_search(
            topic, collection=collection, client=client, top_k=top_k, where={"paper_id": {"$in": ids}},
        )
    elif ranking_mode == "hybrid":
        ranked = hybrid_search(topic, pool, collection=collection, client=client, top_k=top_k)
    elif ranking_mode == "citation_partition":
        if partition_n is None:
            raise ValueError("ranking_mode='citation_partition' requires partition_n")
        partition_a, partition_b = partition_by_citation(pool, n=partition_n)
        ranked = merge_with_guaranteed_slots(
            topic, partition_a, partition_b, n=partition_n,
            collection=collection, client=client, top_k=top_k,
        )
    else:
        raise ValueError(f"Unknown ranking_mode: {ranking_mode!r} (expected one of {RANKING_MODES})")

    return [p for p, _ in ranked]


@observe(name="eval_retrieval_topic", capture_input=False, capture_output=False)
def evaluate_topic(
    topic_id: str, topic_data: dict, client: OpenAI, expand: bool,
    ranking_mode: str | None = None, partition_n: int | None = None, top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Wrapping this whole function in one @observe span (rather than
    relying on whatever @observe-decorated research_agent function(s) it
    happens to call) gives every ranking mode ONE coherent trace per topic
    to attach Precision@k/Recall@k to — 'plain' mode alone calls three
    separately-traced functions (search_arxiv, search_semantic_scholar,
    semantic_rerank) with no shared parent otherwise, and 'ranked' modes
    mix traced (semantic_rerank) and untraced (bm25_search, hybrid_search)
    calls. This span is the SAME kind of thing regardless of mode: the
    real retrieval pass for one topic, precision/recall attached to it as
    a second, complementary view — eval_results/retrieval_history.csv
    below remains the unchanged source of truth; this doesn't replace it.
    """
    expected = topic_data["expected_papers"]
    expected_papers = [_expected_to_paper(e) for e in expected]

    if ranking_mode:
        # Ranking-stage experiment: always uses the expanded candidate pool
        # (query_expansion.py's build_candidate_pool()) regardless of
        # --expand — Phase 4 of the experiment runs all three modes "with
        # query expansion enabled", so that's not a separate toggle here.
        returned = run_topic_retrieval_ranked(topic_data["topic"], client, ranking_mode, partition_n, top_k)
    else:
        retrieval_fn = run_topic_retrieval_expanded if expand else run_topic_retrieval
        returned = retrieval_fn(topic_data["topic"], client, top_k)

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

    ranking_mode_label = ranking_mode or ("expanded" if expand else "plain")
    langfuse = get_client()
    langfuse.update_current_span(
        input={"topic_id": topic_id, "topic": topic_data["topic"], "ranking_mode": ranking_mode_label, "top_k": top_k},
        output={"num_returned": len(returned), "num_expected": len(expected), "precision": precision, "recall": recall},
    )
    # Real eval scores attached directly to this topic's own trace — a
    # second, complementary view in the Langfuse dashboard alongside the
    # CSV history below, not a replacement for it.
    langfuse.score_current_trace(name="precision_at_k", value=precision, data_type="NUMERIC")
    langfuse.score_current_trace(name="recall_at_k", value=recall, data_type="NUMERIC")

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


def print_results_table(results: list[dict], top_k: int = DEFAULT_TOP_K) -> None:
    print("=" * 100)
    print("Retrieval evaluation results (per topic)")
    print("=" * 100)
    for r in results:
        print(f"\n[{r['topic_id']}] ({r['difficulty']}) {r['topic']}")
        print(f"    precision@{top_k}: {r['precision']:.3f}  ({r['num_returned']} returned)")
        print(f"    recall@{top_k}:    {r['recall']:.3f}  ({r['num_expected']} expected)")
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
    print(f"  mean precision@{top_k}: {mean_precision:.3f}")
    print(f"  mean recall@{top_k}:    {mean_recall:.3f}")

    print("\nRecall by difficulty tier (an overall average would hide whether misses cluster in harder topics):")
    for tier in DIFFICULTY_TIERS:
        tier_results = [r for r in results if r["difficulty"] == tier]
        if not tier_results:
            print(f"  {tier}: (no topics)")
            continue
        tier_recall = sum(r["recall"] for r in tier_results) / len(tier_results)
        print(f"  {tier}: {tier_recall:.3f}  ({len(tier_results)} topics)")


def append_history_row(
    results: list[dict], note: str, ranking_mode_label: str,
    partition_n: int | None = None, top_k: int = DEFAULT_TOP_K,
) -> dict:
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
        "top_k": top_k,
        "ranking_mode": ranking_mode_label,
        "partition_n": partition_n if partition_n is not None else "",
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
    parser.add_argument(
        "--expand", action="store_true",
        help="Use query_expansion.py's expanded_search() (LLM-suggested title search) instead of the "
             "direct search_arxiv/search_semantic_scholar/deduplicate/semantic_search flow. Ignored "
             "if --ranking-mode is also given (see below).",
    )
    parser.add_argument(
        "--ranking-mode", choices=RANKING_MODES, default=None,
        help="Ranking-stage experiment: rank query_expansion.py's candidate pool via 'semantic' "
             "(cosine similarity — same algorithm --expand alone already uses), 'bm25' (lexical "
             "scoring, research_agent/ranking.py), 'hybrid' (RRF fusion of both, same module), or "
             "'citation_partition' (guaranteed slots for high-citation papers, also ranking.py — "
             "automatically uses ranking.py's derived get_partition_n(k) rule unless overridden via "
             "--partition-n or --partition-proportion, see below). Always uses the expanded candidate "
             "pool for building it regardless of --expand's value — this is an opt-in evaluation "
             "mode; omit entirely to leave existing --expand/plain behavior (and the live app's own "
             "default) completely untouched.",
    )
    parser.add_argument(
        "--partition-proportion", type=float, default=None,
        help="Only used when --ranking-mode citation_partition: fraction of top_k reserved as "
             "guaranteed Partition-A slots, e.g. 0.3 at top_k=10 means n=3. n = round(proportion * "
             "top_k). Overrides the derived get_partition_n(k) rule (see --partition-n and "
             "ranking.py's get_partition_n) for this run only — the proportion-based experiment that "
             "originally motivated this flag is superseded now that get_partition_n(k) exists, but "
             "it's kept for anyone who wants to re-run that comparison. Ignored if --partition-n is "
             "also given.",
    )
    parser.add_argument(
        "--partition-n", type=int, default=None,
        help="Only used when --ranking-mode citation_partition: an explicit, raw guaranteed-slot "
             "count, overriding both --partition-proportion and the derived get_partition_n(k) rule "
             "for this run only. Takes precedence over --partition-proportion if both are given.",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"k-generalization experiment: override the evaluation's top_k (default {DEFAULT_TOP_K}, "
             f"matching semantic_search()'s own default and api.py's SearchRequest default). Omitting "
             f"this reproduces the exact k=10 results already established — this flag lets the same "
             f"harness test other k values without touching query_expansion.py or ranking.py's "
             f"partition/merge logic, which stay exactly as already built and tested.",
    )
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI()

    topics = load_reference_topics()

    skipped = sorted(tid for tid, t in topics.items() if not t["expected_papers"])
    usable = {tid: t for tid, t in topics.items() if t["expected_papers"]}
    if skipped:
        print(f"Skipping {len(skipped)} topic(s) with zero expected papers: {skipped}")

    partition_n = None
    partition_source = None
    if args.ranking_mode == "citation_partition":
        if args.partition_n is not None:
            partition_n = args.partition_n
            partition_source = "explicit --partition-n"
        elif args.partition_proportion is not None:
            partition_n = round(args.partition_proportion * args.top_k)
            partition_source = f"explicit --partition-proportion={args.partition_proportion}"
        else:
            # Automatic default: the derived production rule from the
            # k-generalization study (research_agent/ranking.py's
            # get_partition_n), used whenever the caller doesn't override
            # it — no manually-specified n required for the normal case.
            partition_n = get_partition_n(args.top_k)
            partition_source = "get_partition_n(k) [derived rule]"

    ranking_mode_label = args.ranking_mode or ("expanded" if args.expand else "plain")
    partition_note = f" (n={partition_n}, source={partition_source})" if partition_n is not None else ""
    print(f"Evaluating {len(usable)} topics at top_k={args.top_k} (ranking mode: {ranking_mode_label}{partition_note})\n")

    results = []
    for i, (topic_id, topic_data) in enumerate(usable.items(), 1):
        print(f"[{i}/{len(usable)}] {topic_id}: {topic_data['topic']!r}")
        r = evaluate_topic(topic_id, topic_data, client, args.expand, args.ranking_mode, partition_n, args.top_k)
        print(f"    precision={r['precision']:.3f} recall={r['recall']:.3f}\n")
        results.append(r)

    print_results_table(results, args.top_k)
    row = append_history_row(results, args.note, ranking_mode_label, partition_n, args.top_k)

    print("\n" + "=" * 100)
    print(f"Appended run {row['run_id']} to {os.path.relpath(HISTORY_CSV, REPO_ROOT)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
