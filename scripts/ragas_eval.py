#!/usr/bin/env python3
"""RAGAS integration: runs the curated Stage 1 test set
(eval_data/stage1_ragas_questions.json) against the REAL pipeline — a real
search/rerank pass (same flow as scripts/test_qa.py) feeding real papers
into qa.py's ask(), whose real `answer` and real retrieved abstracts become
RAGAS's `response` and `retrieved_contexts`. Nothing here is mocked or
fabricated.

Every scenario in the curated set was independently verified (see that
file's own `_meta` block) with the real production defaults this script
also uses: Stage 1 candidate-pool search at STAGE1_TOP_K=10 (matching
research_agent/api.py's SearchRequest.top_k default), then the real qa.ask()
for Stage 2 (condense + re-rank at qa.py's own TOP_K_DEFAULT=5 — deliberately
NOT overridden here, so this exercises the exact same re-rank the live app
uses). Multi-turn scenarios reuse one ChatSession across their turns so
condensing is genuinely exercised, not simulated.

Only two metrics run: Faithfulness and Answer Relevancy. Context Precision
and Context Recall are NOT attempted — both need a reference/ground-truth
answer set that doesn't exist yet (Stage 1b, a separate deferred phase), and
asking RAGAS for them without one would either error or silently score
against a fabricated reference.

Not part of pytest or CI. This is a manually-run tool that makes real,
billable OpenAI calls: the pipeline's own answer-generation and
question-condensing calls (qa.py) PLUS the RAGAS judge model's scoring calls
PLUS embedding calls for Answer Relevancy and for indexing candidates. See
the printed cost summary at the end of a run for actual token counts and an
estimated dollar cost. The curated set is 24 scenarios / 27 total turns (3
are 2-turn multi-turn pairs) — meaningfully more expensive than a smoke test.

Usage:
    python scripts/ragas_eval.py --note "stage 1 curated set, first real run"
    python scripts/ragas_eval.py --note "trying gpt-4.1 as judge" --judge-model gpt-4.1
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
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from research_agent.dedup import deduplicate
from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, semantic_search
from research_agent.ingestion import search_arxiv, search_semantic_scholar
from research_agent.qa import ChatSession, ask

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Cheap judge model by default, same cost-tiering approach used throughout
# this project (agent.py's orchestration model, qa.py's condense model) —
# override with --judge-model to re-run on a stronger model for an
# "official" report later.
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"

# Matches embeddings.py's EMBEDDING_MODEL, so Answer Relevancy's question-
# similarity scoring uses the same embedding space the rest of the project
# already relies on and already has pricing tracked for.
JUDGE_EMBEDDING_MODEL = "text-embedding-3-small"

# Matches research_agent/api.py's SearchRequest.top_k default exactly — this
# is Stage 1 (candidate-pool search), distinct from qa.py's own
# TOP_K_DEFAULT=5 (Stage 2, the answer-time re-rank). Verified via grep
# against both files before use; see eval_data/stage1_ragas_questions.json's
# own _meta block for the investigation that established this distinction.
STAGE1_TOP_K = 10

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DATA_PATH = os.path.join(REPO_ROOT, "eval_data", "stage1_ragas_questions.json")
HISTORY_CSV = os.path.join(REPO_ROOT, "eval_results", "history.csv")

CATEGORIES = ["single_paper", "comparison", "multi_turn", "unanswerable"]

HISTORY_FIELDS = [
    "run_id", "date", "git_commit", "judge_model", "num_scenarios", "num_turns",
    "faithfulness_mean", "answer_relevancy_mean",
    "faithfulness_single_paper", "faithfulness_comparison", "faithfulness_multi_turn", "faithfulness_unanswerable",
    "answer_relevancy_single_paper", "answer_relevancy_comparison", "answer_relevancy_multi_turn", "answer_relevancy_unanswerable",
    "note",
]

# Point-in-time OpenAI pricing (USD per 1M tokens), checked via web search
# at the time this script was written — not guaranteed current. Verify
# against https://openai.com/api/pricing/ before trusting a cost figure
# for budgeting.
PRICING_PER_1M_USD = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def load_eval_scenarios() -> dict:
    """Loads the curated Stage 1 set, stripping the '_meta' documentation
    block so callers only see actual scenario entries."""
    with open(EVAL_DATA_PATH) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k != "_meta"}


def _search_and_rank(topic: str, client: OpenAI, top_k: int = STAGE1_TOP_K):
    """Real Stage-1 search/rerank pass — identical flow to
    scripts/test_qa.py and research_agent/api.py's plain (non-expanded)
    search path: arXiv + Semantic Scholar search, cross-source dedup, embed,
    then semantic rerank at top_k=10. Returns a ranked list of Paper
    objects that becomes the ChatSession's candidate pool."""
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    arxiv_papers = search_arxiv(topic, max_results=15)
    s2_papers = search_semantic_scholar(topic, max_results=15, api_key=s2_key)
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


def collect_cases(client: OpenAI) -> tuple[list[dict], list[dict]]:
    """Runs every scenario in the curated set through the real pipeline —
    Stage 1 search/rerank, then a real ChatSession where each of the
    scenario's turns is answered in order via qa.ask() (Stage 2, using
    qa.py's own TOP_K_DEFAULT=5, not overridden here). Multi-turn scenarios
    reuse one session across turns so _condense_question is genuinely
    exercised on the real conversation history, not simulated.

    Returns (samples, metadata): `samples` is exactly the RAGAS-schema shape
    ({"user_input", "retrieved_contexts", "response"}) with no extra keys,
    since EvaluationDataset.from_list validates against SingleTurnSample's
    strict schema. `metadata` is a same-length, same-order parallel list
    carrying scenario_id/turn_index/category/answerable/etc. for this
    script's own reporting — kept separate so nothing here risks breaking
    RAGAS's own validation.
    """
    scenarios = load_eval_scenarios()
    samples: list[dict] = []
    metadata: list[dict] = []

    total_turns = sum(len(s["turns"]) for s in scenarios.values())
    turn_counter = 0

    for scenario_id, scenario in scenarios.items():
        topic = scenario["topic"]
        category = scenario["category"]
        turns = scenario["turns"]

        print(f"[{scenario_id}] topic={topic!r} category={category}")
        papers = _search_and_rank(topic, client)
        print(f"    Stage-1 pool: {len(papers)} papers")

        session = ChatSession(papers=papers)

        for turn_index, question in enumerate(turns):
            turn_counter += 1
            print(f"    [{turn_counter}/{total_turns}] turn {turn_index + 1}: {question!r}")
            result = ask(session, question, client=client)
            contexts = [p.abstract for p in result["retrieved_papers"] if p.abstract]

            print(f"        answerable={result['answerable']}, {len(contexts)} contexts, "
                  f"answer length={len(result['answer'])} chars")

            if not contexts:
                print(f"        SKIPPED from RAGAS dataset: no retrieved contexts (see qa.py's "
                      f"_no_sources_result path) — nothing for Faithfulness to check claims against.")
                continue

            samples.append({
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": result["answer"],
            })
            metadata.append({
                "scenario_id": scenario_id,
                "turn_index": turn_index,
                "category": category,
                "unanswerable_type": scenario.get("unanswerable_type"),
                "answerable": result["answerable"],
            })
        print()

    return samples, metadata


def run_ragas_eval(samples: list[dict], judge_model: str):
    from ragas import evaluate
    from ragas.cost import get_token_usage_for_openai
    from ragas.dataset_schema import EvaluationDataset
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics import AnswerRelevancy, Faithfulness

    dataset = EvaluationDataset.from_list(samples)
    llm = llm_factory(model=judge_model)
    embeddings = embedding_factory(model=JUDGE_EMBEDDING_MODEL)

    return evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=llm,
        embeddings=embeddings,
        token_usage_parser=get_token_usage_for_openai,
    )


def print_results_table(result, metadata: list[dict]) -> dict[str, float]:
    df = result.to_pandas()

    print("=" * 100)
    print("RAGAS results (per question)")
    print("=" * 100)
    for i, (row, meta) in enumerate(zip(df.itertuples(), metadata), 1):
        tag = meta["category"]
        if meta.get("unanswerable_type"):
            tag += f"/{meta['unanswerable_type']}"
        print(f"\n[{i}] ({meta['scenario_id']} turn {meta['turn_index'] + 1}, {tag}, "
              f"answerable={meta['answerable']}) {row.user_input}")
        print(f"    faithfulness:      {row.faithfulness:.3f}")
        print(f"    answer_relevancy:  {row.answer_relevancy:.3f}")

    means = {
        "faithfulness_mean": float(df["faithfulness"].mean()),
        "answer_relevancy_mean": float(df["answer_relevancy"].mean()),
    }

    print("\n" + "=" * 100)
    print("Mean scores by category")
    print("=" * 100)
    category_means: dict[str, dict[str, float]] = {}
    for category in CATEGORIES:
        idxs = [i for i, m in enumerate(metadata) if m["category"] == category]
        if not idxs:
            print(f"  {category}: (no scenarios)")
            continue
        faith = df["faithfulness"].iloc[idxs].mean()
        rel = df["answer_relevancy"].iloc[idxs].mean()
        category_means[category] = {"faithfulness": float(faith), "answer_relevancy": float(rel)}
        print(f"  {category}: faithfulness={faith:.3f}  answer_relevancy={rel:.3f}  ({len(idxs)} turns)")

    print("\n" + "=" * 100)
    print("Overall mean scores")
    print("=" * 100)
    for k, v in means.items():
        print(f"  {k}: {v:.3f}")

    means["_category_means"] = category_means
    return means


def report_cost(result, judge_model: str) -> None:
    print("\n" + "=" * 100)
    print("Judge/embedding call cost (RAGAS scoring only — does NOT include the")
    print("pipeline's own answer-generation AND question-condensing calls in qa.py,")
    print("which are separately billable and already logged via that module's own")
    print("usage logging)")
    print("=" * 100)
    try:
        usage_list = result.total_tokens()
    except ValueError as e:
        print(f"  (token usage unavailable: {e})")
        return

    if not isinstance(usage_list, list):
        usage_list = [usage_list]

    totals: dict[str, dict[str, int]] = {}
    for u in usage_list:
        model = u.model or judge_model
        bucket = totals.setdefault(model, {"input_tokens": 0, "output_tokens": 0})
        bucket["input_tokens"] += u.input_tokens
        bucket["output_tokens"] += u.output_tokens

    grand_total_usd = 0.0
    for model, tok in totals.items():
        pricing = PRICING_PER_1M_USD.get(model)
        print(f"  {model}: {tok['input_tokens']} input tokens, {tok['output_tokens']} output tokens")
        if pricing is None:
            print(f"    (no pricing on file for {model!r} — add it to PRICING_PER_1M_USD to estimate cost)")
            continue
        cost = (tok["input_tokens"] / 1_000_000 * pricing["input"]
                + tok["output_tokens"] / 1_000_000 * pricing["output"])
        grand_total_usd += cost
        print(f"    estimated cost: ${cost:.4f} (at ${pricing['input']}/1M in, ${pricing['output']}/1M out)")

    print(f"\n  Total estimated judge LLM cost: ${grand_total_usd:.4f}")
    print("  (Pricing is point-in-time, not fetched live — verify against")
    print("   https://openai.com/api/pricing/ before budgeting off this number.)")
    print(f"\n  NOTE: this total is LLM (judge) calls only. RAGAS's cost callback")
    print(f"  (get_token_usage_for_openai) does not capture embedding API calls —")
    print(f"  Answer Relevancy also makes {JUDGE_EMBEDDING_MODEL} calls that are")
    print(f"  real, billable, and NOT included above. At $"
          f"{PRICING_PER_1M_USD[JUDGE_EMBEDDING_MODEL]['input']}/1M tokens that's")
    print(f"  negligible in absolute terms, but the number above is not the full total.")


def append_history_row(means: dict, judge_model: str, num_scenarios: int, num_turns: int, note: str) -> dict:
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

    category_means = means.get("_category_means", {})

    def cat_score(category: str, metric: str) -> str:
        if category not in category_means:
            return ""
        return f"{category_means[category][metric]:.4f}"

    row = {
        "run_id": next_run_id,
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "judge_model": judge_model,
        "num_scenarios": num_scenarios,
        "num_turns": num_turns,
        "faithfulness_mean": f"{means['faithfulness_mean']:.4f}",
        "answer_relevancy_mean": f"{means['answer_relevancy_mean']:.4f}",
        "faithfulness_single_paper": cat_score("single_paper", "faithfulness"),
        "faithfulness_comparison": cat_score("comparison", "faithfulness"),
        "faithfulness_multi_turn": cat_score("multi_turn", "faithfulness"),
        "faithfulness_unanswerable": cat_score("unanswerable", "faithfulness"),
        "answer_relevancy_single_paper": cat_score("single_paper", "answer_relevancy"),
        "answer_relevancy_comparison": cat_score("comparison", "answer_relevancy"),
        "answer_relevancy_multi_turn": cat_score("multi_turn", "answer_relevancy"),
        "answer_relevancy_unanswerable": cat_score("unanswerable", "answer_relevancy"),
        "note": note,
    }

    with open(HISTORY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--note", required=True, help="Short description of this run, e.g. 'stage 1 curated set, first real run'")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                         help=f"OpenAI model used by RAGAS to judge answers (default: {DEFAULT_JUDGE_MODEL})")
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI()

    scenarios = load_eval_scenarios()
    total_turns = sum(len(s["turns"]) for s in scenarios.values())

    print(f"Judge model: {args.judge_model}")
    print(f"Dataset:     {EVAL_DATA_PATH}")
    print(f"Scenarios:   {len(scenarios)} ({total_turns} total turns)\n")

    samples, metadata = collect_cases(client)
    if len(samples) < total_turns:
        print(f"NOTE: {total_turns - len(samples)} turn(s) were skipped from RAGAS scoring (no retrieved contexts) — "
              f"scoring {len(samples)} of {total_turns} turns.\n")

    result = run_ragas_eval(samples, args.judge_model)
    means = print_results_table(result, metadata)
    report_cost(result, args.judge_model)
    row = append_history_row(means, args.judge_model, len(scenarios), len(samples), args.note)

    print("\n" + "=" * 100)
    print(f"Appended run {row['run_id']} to {os.path.relpath(HISTORY_CSV, REPO_ROOT)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
