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
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langfuse import get_client, observe
from openai import OpenAI

from research_agent.agent import run_research_agent
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

# The one mode in this file that does NOT call build_candidate_pool()/any
# direct-function retrieval flow — it runs the real LangChain/LangGraph
# tool-calling agent (agent.py's run_research_agent(), the actual default
# path a live user hits today) and measures whatever it decides to search,
# reformulate, and rank. Kept out of RANKING_MODES (which all funnel through
# run_topic_retrieval_ranked()'s shared candidate-pool step) since this mode
# needs its own dispatch: there is no shared candidate pool, and "k" doesn't
# mean the same thing here (see run_topic_retrieval_agent's docstring).
AGENT_RANKING_MODE = "langgraph_agent"

# USD per 1M tokens for gpt-4.1-mini (agent.py's AGENT_MODEL), per OpenAI's
# published pricing as of training data cutoff (Jan 2026). Prices change —
# verify at https://openai.com/api/pricing before relying on this figure for
# real budgeting. Embedding cost (text-embedding-3-small, spent inside the
# agent's own rerank_by_relevance_tool call) is tracked separately using
# embeddings.py's own PRICE_PER_1M_TOKENS via the cost the tool itself
# reports, not re-derived here.
AGENT_MODEL_PRICE_PER_1M_INPUT_TOKENS = 0.40
AGENT_MODEL_PRICE_PER_1M_OUTPUT_TOKENS = 1.60

# rerank_by_relevance_tool's own ToolMessage content ends with
# "... N newly embedded, ~$0.000123):\n1. (0.912) Title..." — this is the
# authoritative per-call embedding cost embeddings.py itself computed;
# parsing it back out here avoids recomputing (and risking drift from) that
# same number.
_EMBED_COST_RE = re.compile(r"~\$(\d+\.\d+)\)")


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


def run_topic_retrieval_agent(topic: str, top_k: int = DEFAULT_TOP_K) -> tuple[list[Paper], dict]:
    """Runs the real LangGraph tool-calling agent (agent.py's
    run_research_agent(), unmodified — this is genuinely the same code path
    a live user hits by default today), instead of calling any
    ingestion/dedup/embeddings function directly.

    What "k" means here, honestly: `top_k` is passed to run_research_agent(),
    which bakes it into the system prompt as an instruction ("call
    rerank_by_relevance with top_k set to exactly N") — it is NOT a
    code-enforced constraint on this evaluation path the way it is for every
    other mode in this file. The agent decides for itself whether to call
    rerank_by_relevance_tool at all, how many times, and what top_k argument
    to actually pass. We observe (not control) that behavior via on_step:
    every rerank_by_relevance_tool call the agent actually issues is
    recorded in observed_top_k_args below. If that list is empty, [10],  or
    [10, 10], the agent obeyed the instruction as designed; if it's
    something else (or the agent never reranks at all, in which case
    session.ranked — and thus the returned paper list — is simply empty),
    that's the agent choosing its own k, and the honest thing to report is
    "whatever the agent chose," not a fixed k=N result comparable
    apples-to-apples with the other modes' code-enforced top_k.

    Returns (ranked_papers, cost_info) where cost_info carries token counts,
    an estimated USD cost, and the observed-k diagnostics above.
    """
    llm_input_tokens = 0
    llm_output_tokens = 0
    embedding_cost_usd = 0.0
    rerank_calls = 0
    observed_top_k_args: list[int | None] = []

    def on_step(message) -> None:
        nonlocal llm_input_tokens, llm_output_tokens, embedding_cost_usd, rerank_calls
        if isinstance(message, AIMessage):
            usage = message.usage_metadata
            if usage:
                llm_input_tokens += usage.get("input_tokens", 0) or 0
                llm_output_tokens += usage.get("output_tokens", 0) or 0
            for tc in message.tool_calls or []:
                if tc.get("name") == "rerank_by_relevance_tool":
                    observed_top_k_args.append(tc.get("args", {}).get("top_k"))
        elif isinstance(message, ToolMessage) and message.name == "rerank_by_relevance_tool":
            rerank_calls += 1
            match = _EMBED_COST_RE.search(message.content)
            if match:
                embedding_cost_usd += float(match.group(1))

    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    session = run_research_agent(topic, s2_api_key=s2_key, top_k=top_k, on_step=on_step)

    llm_cost_usd = (
        llm_input_tokens / 1_000_000 * AGENT_MODEL_PRICE_PER_1M_INPUT_TOKENS
        + llm_output_tokens / 1_000_000 * AGENT_MODEL_PRICE_PER_1M_OUTPUT_TOKENS
    )

    cost_info = {
        "instructed_top_k": top_k,
        "observed_top_k_args": observed_top_k_args,
        "rerank_calls": rerank_calls,
        "num_papers_in_pool": len(session.papers),
        "num_papers_returned": len(session.ranked),
        "llm_input_tokens": llm_input_tokens,
        "llm_output_tokens": llm_output_tokens,
        "llm_cost_usd": llm_cost_usd,
        "embedding_cost_usd": embedding_cost_usd,
        "total_cost_usd": llm_cost_usd + embedding_cost_usd,
    }
    return [p for p, _ in session.ranked], cost_info


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

    agent_cost_info = None
    if ranking_mode == AGENT_RANKING_MODE:
        # Its own dispatch, not run_topic_retrieval_ranked(): the agent does
        # its own retrieval end to end (no shared build_candidate_pool()
        # step), and reports its own cost/k diagnostics alongside the papers.
        returned, agent_cost_info = run_topic_retrieval_agent(topic_data["topic"], top_k)
    elif ranking_mode:
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
        "agent_cost_info": agent_cost_info,
    }


def print_results_table(results: list[dict], top_k: int = DEFAULT_TOP_K) -> None:
    print("=" * 100)
    print("Retrieval evaluation results (per topic)")
    print("=" * 100)
    is_agent_mode = any(r.get("agent_cost_info") for r in results)
    for r in results:
        print(f"\n[{r['topic_id']}] ({r['difficulty']}) {r['topic']}")
        if is_agent_mode and r["agent_cost_info"]:
            ci = r["agent_cost_info"]
            print(
                f"    precision: {r['precision']:.3f}  recall: {r['recall']:.3f}  "
                f"({r['num_returned']} returned / {r['num_expected']} expected)"
            )
            print(
                f"    k: instructed={ci['instructed_top_k']}  observed rerank top_k arg(s)="
                f"{ci['observed_top_k_args']}  rerank_calls={ci['rerank_calls']}  "
                f"pool_size={ci['num_papers_in_pool']}"
            )
            print(
                f"    cost: ${ci['total_cost_usd']:.6f}  (llm=${ci['llm_cost_usd']:.6f} "
                f"[{ci['llm_input_tokens']} in / {ci['llm_output_tokens']} out tokens], "
                f"embedding=${ci['embedding_cost_usd']:.6f})"
            )
        else:
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
    if is_agent_mode:
        print(
            f"  mean precision: {mean_precision:.3f}  mean recall: {mean_recall:.3f}  "
            f"(k is 'whatever the agent chose' per topic — see per-topic k lines above, "
            f"NOT a fixed, controlled top_k={top_k} the way other modes' numbers are)"
        )
    else:
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

    if is_agent_mode:
        costed = [r["agent_cost_info"] for r in results if r.get("agent_cost_info")]
        total_cost = sum(ci["total_cost_usd"] for ci in costed)
        print("\n" + "=" * 100)
        print(f"Agent path cost: ${total_cost:.6f} total across {len(costed)} topic(s), "
              f"${total_cost / len(costed):.6f} mean/topic (LLM orchestration + embedding only — "
              f"arXiv/Semantic Scholar/web-search API calls are unbilled)")
        print("=" * 100)


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
        "--ranking-mode", choices=[*RANKING_MODES, AGENT_RANKING_MODE], default=None,
        help="Ranking-stage experiment: rank query_expansion.py's candidate pool via 'semantic' "
             "(cosine similarity — same algorithm --expand alone already uses), 'bm25' (lexical "
             "scoring, research_agent/ranking.py), 'hybrid' (RRF fusion of both, same module), or "
             "'citation_partition' (guaranteed slots for high-citation papers, also ranking.py — "
             "automatically uses ranking.py's derived get_partition_n(k) rule unless overridden via "
             "--partition-n or --partition-proportion, see below). Always uses the expanded candidate "
             "pool for building it regardless of --expand's value — this is an opt-in evaluation "
             "mode; omit entirely to leave existing --expand/plain behavior (and the live app's own "
             "default) completely untouched. 'langgraph_agent' is different in kind, not degree: it "
             "runs the real LangChain/LangGraph tool-calling agent (agent.py's run_research_agent(), "
             "unmodified) end to end — its own query reformulation, its own tool/source choices, its "
             "own decision of whether and how to rerank — instead of any direct-function retrieval "
             "flow. --top-k is only an instruction baked into its system prompt here, not a "
             "code-enforced constraint; see run_topic_retrieval_agent()'s docstring. Makes real LLM "
             "tool-calling calls in addition to the embedding calls every other mode already makes — "
             "materially more expensive per topic. --expand and --partition-*/--partition-proportion "
             "are ignored when this mode is selected.",
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
    parser.add_argument(
        "--topic-ids", default=None,
        help="Comma-separated subset of topic IDs to evaluate (e.g. 'peft-01,rag-02'), for a small-scale "
             "sanity/cost check before committing to a full run. Omit to evaluate every usable topic "
             "(existing behavior, unchanged). Not appended to retrieval_history.csv's num_topics "
             "comparison history as a full run — use --note to make a partial run's scope obvious.",
    )
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI()

    topics = load_reference_topics()

    skipped = sorted(tid for tid, t in topics.items() if not t["expected_papers"])
    usable = {tid: t for tid, t in topics.items() if t["expected_papers"]}
    if skipped:
        print(f"Skipping {len(skipped)} topic(s) with zero expected papers: {skipped}")

    if args.topic_ids:
        requested = [t.strip() for t in args.topic_ids.split(",") if t.strip()]
        unknown = [t for t in requested if t not in usable]
        if unknown:
            raise SystemExit(f"Unknown/unusable topic id(s): {unknown}. Usable topics: {sorted(usable)}")
        usable = {tid: usable[tid] for tid in requested}

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
    top_k_label = f"instructed top_k={args.top_k} (not code-enforced)" if args.ranking_mode == AGENT_RANKING_MODE else f"top_k={args.top_k}"
    print(f"Evaluating {len(usable)} topics at {top_k_label} (ranking mode: {ranking_mode_label}{partition_note})\n")

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
