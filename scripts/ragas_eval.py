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

All four RAGAS metrics run: Faithfulness and Answer Relevancy (reference-
free, computed for every turn) plus Context Precision and Context Recall
(need the `reference` field added to eval_data in Stage 1b — computed only
for turns whose scenario carries one). 7 of 24 scenarios (the 6 unanswerable
ones plus 'ann-comparison-01-orig', flagged during Stage 1b review as too
thin to ground a confident reference) have no `reference` and are excluded
from Context Precision/Recall specifically — see each scenario's
`context_pr_exclusion_reason` in the eval_data file. Their Faithfulness/
Answer Relevancy scores are still computed normally alongside everyone
else's; nothing is silently dropped from those two metrics.

This runs as two separate RAGAS evaluate() passes over the SAME real
generated answers (not two separate generations): one over all turns for
Faithfulness+AnswerRelevancy (neither needs `reference`), one over just the
turns that have a `reference` for ContextPrecision+ContextRecall (both
require it). Splitting into two passes is necessary because RAGAS errors
if a metric's required column is missing for a sample; it does NOT mean two
different sets of answers were generated.

Not part of pytest or CI. This is a manually-run tool that makes real,
billable OpenAI calls: the pipeline's own answer-generation and
question-condensing calls (qa.py) PLUS RAGAS's judge model calls for all
four metrics PLUS embedding calls for Answer Relevancy and for indexing
candidates. See the printed cost summary at the end of a run for actual
token counts and an estimated dollar cost. The curated set is 24 scenarios /
27 total turns (3 are 2-turn multi-turn pairs), of which 20 turns carry a
reference — meaningfully more expensive than the two-metric run, since
Context Precision/Recall each make their own LLM judge calls per turn.

Every run also writes eval_results/runs/run_<run_id>.json: one record per
scored turn (question, real retrieved paper titles, real cited paper
titles, real retrieved abstracts, real generated answer, reference if any,
every per-turn score). history.csv only ever carries aggregate means, which
made it impossible to root-cause a surprising per-turn score after the
fact — this fills that gap without any extra API calls (pure local
serialization of data already produced during the run).

Usage:
    python scripts/ragas_eval.py --note "stage 1b, all four metrics, first real run"
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
# Per-run artifact directory: history.csv only ever carries aggregate means,
# which made two real runs' surprising per-turn outliers (comparison-category
# Context Precision cratering, one unanswerable turn's Answer Relevancy
# swinging from 0.000 to 0.652 run-to-run) impossible to root-cause after the
# fact — the actual retrieved paper titles and generated answer text for
# those specific turns were already gone. This directory exists so every run
# leaves that evidence behind instead of only its aggregate scores.
RUNS_DIR = os.path.join(REPO_ROOT, "eval_results", "runs")

CATEGORIES = ["single_paper", "comparison", "multi_turn", "unanswerable"]

HISTORY_FIELDS = [
    "run_id", "date", "git_commit", "judge_model", "num_scenarios", "num_turns", "num_turns_with_reference",
    "faithfulness_mean", "answer_relevancy_mean", "context_precision_mean", "context_recall_mean",
    "faithfulness_single_paper", "faithfulness_comparison", "faithfulness_multi_turn", "faithfulness_unanswerable",
    "answer_relevancy_single_paper", "answer_relevancy_comparison", "answer_relevancy_multi_turn", "answer_relevancy_unanswerable",
    "context_precision_single_paper", "context_precision_comparison", "context_precision_multi_turn",
    "context_recall_single_paper", "context_recall_comparison", "context_recall_multi_turn",
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


def _raw_turn_record(meta: dict, sample: dict) -> dict:
    """The persistable shape of one turn's real, already-paid-for pipeline
    output — no scores, since this is written before scoring even starts
    (see collect_cases). save_run_artifact() reuses this same shape and
    just adds a "scores" key once RAGAS has run, so the raw and final
    records never drift out of sync with each other."""
    return {
        "scenario_id": meta["scenario_id"],
        "turn_index": meta["turn_index"],
        "category": meta["category"],
        "unanswerable_type": meta.get("unanswerable_type"),
        "question": sample["user_input"],
        "answerable": meta["answerable"],
        "retrieved_paper_titles": meta["retrieved_paper_titles"],
        "cited_paper_titles": meta["cited_paper_titles"],
        "retrieved_contexts": sample["retrieved_contexts"],
        "response": sample["response"],
        "reference": sample["reference"],
    }


def collect_cases(client: OpenAI, raw_jsonl_path: str) -> tuple[list[dict], list[dict]]:
    """Runs every scenario in the curated set through the real pipeline —
    Stage 1 search/rerank, then a real ChatSession where each of the
    scenario's turns is answered in order via qa.ask() (Stage 2, using
    qa.py's own TOP_K_DEFAULT=5, not overridden here). Multi-turn scenarios
    reuse one session across turns so _condense_question is genuinely
    exercised on the real conversation history, not simulated.

    Each turn's real, already-billable output (question, retrieved paper
    titles, retrieved contexts, generated answer, reference) is written to
    raw_jsonl_path IMMEDIATELY after that turn's ask() call — one JSON
    object per line, flushed and fsync'd before moving to the next turn.
    This is deliberately decoupled from RAGAS scoring, which only happens
    after every turn in the whole run has been generated: if scoring later
    crashes or rate-limits (both have happened to this project's real runs
    already), the real generation work already paid for is safely on disk
    the moment it's produced, not only if the run finishes end-to-end.

    Returns (samples, metadata): `samples` is exactly the RAGAS-schema shape
    ({"user_input", "retrieved_contexts", "response", "reference"}), with
    "reference" set to None for turns whose scenario has no reference
    (SingleTurnSample's own schema treats reference as Optional[str], so a
    None value is valid and simply ignored by metrics that don't need it —
    see run_faithfulness_relevancy/run_context_precision_recall below for
    how each pass filters accordingly). `metadata` is a same-length,
    same-order parallel list carrying scenario_id/turn_index/category/
    has_reference/etc. for this script's own reporting, PLUS
    retrieved_paper_titles/cited_paper_titles — not needed by RAGAS at all,
    kept here purely so save_run_artifact() can persist enough to diagnose a
    surprising per-turn score later without re-running anything.
    """
    scenarios = load_eval_scenarios()
    samples: list[dict] = []
    metadata: list[dict] = []

    total_turns = sum(len(s["turns"]) for s in scenarios.values())
    turn_counter = 0

    with open(raw_jsonl_path, "w") as raw_f:
        for scenario_id, scenario in scenarios.items():
            topic = scenario["topic"]
            category = scenario["category"]
            turns = scenario["turns"]
            references = scenario.get("references")

            print(f"[{scenario_id}] topic={topic!r} category={category}")
            papers = _search_and_rank(topic, client)
            print(f"    Stage-1 pool: {len(papers)} papers")

            session = ChatSession(papers=papers)

            for turn_index, question in enumerate(turns):
                turn_counter += 1
                print(f"    [{turn_counter}/{total_turns}] turn {turn_index + 1}: {question!r}")
                result = ask(session, question, client=client)
                contexts = [p.abstract for p in result["retrieved_papers"] if p.abstract]

                reference = references[turn_index] if references else None

                print(f"        answerable={result['answerable']}, {len(contexts)} contexts, "
                      f"answer length={len(result['answer'])} chars, has_reference={reference is not None}")

                if not contexts:
                    print(f"        SKIPPED from RAGAS dataset: no retrieved contexts (see qa.py's "
                          f"_no_sources_result path) — nothing for Faithfulness to check claims against.")
                    continue

                sample = {
                    "user_input": question,
                    "retrieved_contexts": contexts,
                    "response": result["answer"],
                    "reference": reference,
                }
                meta = {
                    "scenario_id": scenario_id,
                    "turn_index": turn_index,
                    "category": category,
                    "unanswerable_type": scenario.get("unanswerable_type"),
                    "answerable": result["answerable"],
                    "has_reference": reference is not None,
                    "retrieved_paper_titles": [p.title for p in result["retrieved_papers"]],
                    "cited_paper_titles": [p.title for p in result["cited_papers"]],
                }

                # Decoupled from RAGAS scoring on purpose — see docstring.
                # Written and fsync'd before this turn's loop iteration ends,
                # so it survives even if a later turn or the scoring pass
                # crashes.
                raw_f.write(json.dumps(_raw_turn_record(meta, sample), ensure_ascii=False) + "\n")
                raw_f.flush()
                os.fsync(raw_f.fileno())

                samples.append(sample)
                metadata.append(meta)
            print()

    return samples, metadata


def run_faithfulness_relevancy(samples: list[dict], judge_model: str):
    """Faithfulness + AnswerRelevancy over every turn — neither metric
    needs `reference`, so the full sample set (including reference=None
    entries) is used unfiltered."""
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


def run_context_precision_recall(samples_with_reference: list[dict], judge_model: str):
    """ContextPrecision + ContextRecall — both require `reference` (verified
    via ContextPrecision().required_columns / ContextRecall().required_columns
    before writing this), so this ONLY ever receives the pre-filtered subset
    of turns that actually carry one — never the reference=None turns."""
    from ragas import evaluate
    from ragas.cost import get_token_usage_for_openai
    from ragas.dataset_schema import EvaluationDataset
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics import ContextPrecision, ContextRecall

    dataset = EvaluationDataset.from_list(samples_with_reference)
    llm = llm_factory(model=judge_model)
    embeddings = embedding_factory(model=JUDGE_EMBEDDING_MODEL)

    return evaluate(
        dataset=dataset,
        metrics=[ContextPrecision(), ContextRecall()],
        llm=llm,
        embeddings=embeddings,
        token_usage_parser=get_token_usage_for_openai,
    )


def print_results_table(fa_result, pr_result, metadata: list[dict], pr_metadata: list[dict]) -> dict:
    fa_df = fa_result.to_pandas()
    pr_df = pr_result.to_pandas()

    # pr_df rows are a subset of metadata, in the same relative order —
    # build a lookup from (scenario_id, turn_index) to its P/R row so the
    # main per-turn loop below can print "excluded" for the 7 turns with no
    # reference instead of assuming positional alignment with fa_df.
    pr_lookup = {
        (m["scenario_id"], m["turn_index"]): row
        for m, row in zip(pr_metadata, pr_df.itertuples())
    }

    print("=" * 100)
    print("RAGAS results (per question)")
    print("=" * 100)
    for i, (row, meta) in enumerate(zip(fa_df.itertuples(), metadata), 1):
        tag = meta["category"]
        if meta.get("unanswerable_type"):
            tag += f"/{meta['unanswerable_type']}"
        print(f"\n[{i}] ({meta['scenario_id']} turn {meta['turn_index'] + 1}, {tag}, "
              f"answerable={meta['answerable']}) {row.user_input}")
        print(f"    faithfulness:       {row.faithfulness:.3f}")
        print(f"    answer_relevancy:   {row.answer_relevancy:.3f}")
        pr_row = pr_lookup.get((meta["scenario_id"], meta["turn_index"]))
        if pr_row is not None:
            print(f"    context_precision:  {pr_row.context_precision:.3f}")
            print(f"    context_recall:     {pr_row.context_recall:.3f}")
        else:
            print(f"    context_precision:  excluded (no reference — see context_pr_exclusion_reason)")
            print(f"    context_recall:     excluded (no reference — see context_pr_exclusion_reason)")

    means = {
        "faithfulness_mean": float(fa_df["faithfulness"].mean()),
        "answer_relevancy_mean": float(fa_df["answer_relevancy"].mean()),
        "context_precision_mean": float(pr_df["context_precision"].mean()),
        "context_recall_mean": float(pr_df["context_recall"].mean()),
    }

    print("\n" + "=" * 100)
    print("Mean scores by category — Faithfulness / Answer Relevancy (all turns)")
    print("=" * 100)
    category_means: dict[str, dict[str, float]] = {}
    for category in CATEGORIES:
        idxs = [i for i, m in enumerate(metadata) if m["category"] == category]
        if not idxs:
            print(f"  {category}: (no scenarios)")
            continue
        faith = fa_df["faithfulness"].iloc[idxs].mean()
        rel = fa_df["answer_relevancy"].iloc[idxs].mean()
        category_means.setdefault(category, {})["faithfulness"] = float(faith)
        category_means.setdefault(category, {})["answer_relevancy"] = float(rel)
        print(f"  {category}: faithfulness={faith:.3f}  answer_relevancy={rel:.3f}  ({len(idxs)} turns)")

    print("\n" + "=" * 100)
    print(f"Mean scores by category — Context Precision / Context Recall "
          f"({len(pr_df)}/{len(fa_df)} turns with a reference; unanswerable category excluded entirely)")
    print("=" * 100)
    for category in CATEGORIES:
        idxs = [i for i, m in enumerate(pr_metadata) if m["category"] == category]
        if not idxs:
            print(f"  {category}: (no scenarios with a reference)")
            continue
        prec = pr_df["context_precision"].iloc[idxs].mean()
        rec = pr_df["context_recall"].iloc[idxs].mean()
        category_means.setdefault(category, {})["context_precision"] = float(prec)
        category_means.setdefault(category, {})["context_recall"] = float(rec)
        print(f"  {category}: context_precision={prec:.3f}  context_recall={rec:.3f}  ({len(idxs)} turns)")

    print("\n" + "=" * 100)
    print("Overall mean scores")
    print("=" * 100)
    for k, v in means.items():
        print(f"  {k}: {v:.3f}")

    means["_category_means"] = category_means
    return means


def report_cost(fa_result, pr_result, judge_model: str) -> None:
    print("\n" + "=" * 100)
    print("Judge/embedding call cost (RAGAS scoring only — does NOT include the")
    print("pipeline's own answer-generation AND question-condensing calls in qa.py,")
    print("which are separately billable and already logged via that module's own")
    print("usage logging)")
    print("=" * 100)

    totals: dict[str, dict[str, int]] = {}
    for label, result in [("Faithfulness + AnswerRelevancy", fa_result), ("ContextPrecision + ContextRecall", pr_result)]:
        try:
            usage_list = result.total_tokens()
        except ValueError as e:
            print(f"  {label}: token usage unavailable ({e})")
            continue
        if not isinstance(usage_list, list):
            usage_list = [usage_list]
        pass_totals: dict[str, dict[str, int]] = {}
        for u in usage_list:
            model = u.model or judge_model
            bucket = pass_totals.setdefault(model, {"input_tokens": 0, "output_tokens": 0})
            bucket["input_tokens"] += u.input_tokens
            bucket["output_tokens"] += u.output_tokens
            all_bucket = totals.setdefault(model, {"input_tokens": 0, "output_tokens": 0})
            all_bucket["input_tokens"] += u.input_tokens
            all_bucket["output_tokens"] += u.output_tokens
        print(f"\n  [{label}]")
        for model, tok in pass_totals.items():
            print(f"    {model}: {tok['input_tokens']} input tokens, {tok['output_tokens']} output tokens")

    grand_total_usd = 0.0
    print(f"\n  Combined judge token totals:")
    for model, tok in totals.items():
        pricing = PRICING_PER_1M_USD.get(model)
        print(f"    {model}: {tok['input_tokens']} input tokens, {tok['output_tokens']} output tokens")
        if pricing is None:
            print(f"      (no pricing on file for {model!r} — add it to PRICING_PER_1M_USD to estimate cost)")
            continue
        cost = (tok["input_tokens"] / 1_000_000 * pricing["input"]
                + tok["output_tokens"] / 1_000_000 * pricing["output"])
        grand_total_usd += cost
        print(f"      estimated cost: ${cost:.4f} (at ${pricing['input']}/1M in, ${pricing['output']}/1M out)")

    print(f"\n  Total estimated judge LLM cost (all 4 metrics combined): ${grand_total_usd:.4f}")
    print("  (Pricing is point-in-time, not fetched live — verify against")
    print("   https://openai.com/api/pricing/ before budgeting off this number.)")
    print(f"\n  NOTE: this total is LLM (judge) calls only. RAGAS's cost callback")
    print(f"  (get_token_usage_for_openai) does not capture embedding API calls —")
    print(f"  Answer Relevancy also makes {JUDGE_EMBEDDING_MODEL} calls that are")
    print(f"  real, billable, and NOT included above. At $"
          f"{PRICING_PER_1M_USD[JUDGE_EMBEDDING_MODEL]['input']}/1M tokens that's")
    print(f"  negligible in absolute terms, but the number above is not the full total.")
    print(f"  Also NOT included: qa.py's own real answer-generation (gpt-4.1) and")
    print(f"  question-condensing (gpt-4.1-mini) calls for all 27 turns — separately")
    print(f"  billable, not captured by RAGAS's cost callback at all.")


def append_history_row(means: dict, judge_model: str, num_scenarios: int, num_turns: int,
                        num_turns_with_reference: int, note: str) -> dict:
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
        if category not in category_means or metric not in category_means[category]:
            return ""
        return f"{category_means[category][metric]:.4f}"

    row = {
        "run_id": next_run_id,
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "judge_model": judge_model,
        "num_scenarios": num_scenarios,
        "num_turns": num_turns,
        "num_turns_with_reference": num_turns_with_reference,
        "faithfulness_mean": f"{means['faithfulness_mean']:.4f}",
        "answer_relevancy_mean": f"{means['answer_relevancy_mean']:.4f}",
        "context_precision_mean": f"{means['context_precision_mean']:.4f}",
        "context_recall_mean": f"{means['context_recall_mean']:.4f}",
        "faithfulness_single_paper": cat_score("single_paper", "faithfulness"),
        "faithfulness_comparison": cat_score("comparison", "faithfulness"),
        "faithfulness_multi_turn": cat_score("multi_turn", "faithfulness"),
        "faithfulness_unanswerable": cat_score("unanswerable", "faithfulness"),
        "answer_relevancy_single_paper": cat_score("single_paper", "answer_relevancy"),
        "answer_relevancy_comparison": cat_score("comparison", "answer_relevancy"),
        "answer_relevancy_multi_turn": cat_score("multi_turn", "answer_relevancy"),
        "answer_relevancy_unanswerable": cat_score("unanswerable", "answer_relevancy"),
        "context_precision_single_paper": cat_score("single_paper", "context_precision"),
        "context_precision_comparison": cat_score("comparison", "context_precision"),
        "context_precision_multi_turn": cat_score("multi_turn", "context_precision"),
        "context_recall_single_paper": cat_score("single_paper", "context_recall"),
        "context_recall_comparison": cat_score("comparison", "context_recall"),
        "context_recall_multi_turn": cat_score("multi_turn", "context_recall"),
        "note": note,
    }

    with open(HISTORY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return row


def save_run_artifact(
    run_id: int, judge_model: str, note: str,
    samples: list[dict], metadata: list[dict],
    fa_result, pr_result, pr_metadata: list[dict],
) -> str:
    """Writes eval_results/runs/run_<run_id>.json — the same per-turn shape
    _raw_turn_record() already wrote incrementally during collect_cases(),
    with a "scores" key added now that RAGAS has run. This is the "raw data
    plus scores, once scoring succeeded" artifact; the raw JSONL file
    written turn-by-turn during generation is the "already-paid-for data,
    regardless of whether scoring succeeds" artifact — see collect_cases's
    docstring for why those two are deliberately separate writes rather
    than one write at the end.

    Reuses run_id from append_history_row's return value rather than
    recomputing it, so the artifact filename and the history.csv row it
    corresponds to always agree. Uses the SAME run (fa_result, pr_result)
    already scored earlier in main() — no new API calls here, this is pure
    local serialization of data that already exists in memory.
    """
    os.makedirs(RUNS_DIR, exist_ok=True)

    fa_df = fa_result.to_pandas()
    pr_df = pr_result.to_pandas()
    pr_lookup = {
        (m["scenario_id"], m["turn_index"]): row
        for m, row in zip(pr_metadata, pr_df.itertuples())
    }

    turns = []
    for sample, meta, fa_row in zip(samples, metadata, fa_df.itertuples()):
        pr_row = pr_lookup.get((meta["scenario_id"], meta["turn_index"]))
        record = _raw_turn_record(meta, sample)
        record["scores"] = {
            "faithfulness": float(fa_row.faithfulness),
            "answer_relevancy": float(fa_row.answer_relevancy),
            "context_precision": float(pr_row.context_precision) if pr_row is not None else None,
            "context_recall": float(pr_row.context_recall) if pr_row is not None else None,
        }
        turns.append(record)

    artifact = {
        "run_id": run_id,
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "judge_model": judge_model,
        "note": note,
        "num_turns": len(turns),
        "num_turns_with_reference": sum(1 for t in turns if t["reference"] is not None),
        "turns": turns,
    }

    path = os.path.join(RUNS_DIR, f"run_{run_id}.json")
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--note", required=True, help="Short description of this run, e.g. 'stage 1b, all four metrics, first real run'")
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

    os.makedirs(RUNS_DIR, exist_ok=True)
    run_start_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_jsonl_path = os.path.join(RUNS_DIR, f"raw_{run_start_ts}.jsonl")
    print(f"Raw per-turn data (contexts/response/metadata, no scores yet) will be written "
          f"incrementally to {os.path.relpath(raw_jsonl_path, REPO_ROOT)} as each turn completes — "
          f"independent of whether scoring below succeeds.\n")

    samples, metadata = collect_cases(client, raw_jsonl_path)
    if len(samples) < total_turns:
        print(f"NOTE: {total_turns - len(samples)} turn(s) were skipped from RAGAS scoring (no retrieved contexts) — "
              f"scoring {len(samples)} of {total_turns} turns.\n")

    pr_samples = [s for s in samples if s["reference"] is not None]
    pr_metadata = [m for m in metadata if m["has_reference"]]
    print(f"Context Precision/Recall will run on {len(pr_samples)}/{len(samples)} turns "
          f"(the rest have no reference — see eval_data's context_pr_exclusion_reason).\n")

    fa_result = run_faithfulness_relevancy(samples, args.judge_model)
    pr_result = run_context_precision_recall(pr_samples, args.judge_model)

    means = print_results_table(fa_result, pr_result, metadata, pr_metadata)
    report_cost(fa_result, pr_result, args.judge_model)
    row = append_history_row(means, args.judge_model, len(scenarios), len(samples), len(pr_samples), args.note)
    artifact_path = save_run_artifact(
        row["run_id"], args.judge_model, args.note, samples, metadata, fa_result, pr_result, pr_metadata,
    )

    print("\n" + "=" * 100)
    print(f"Appended run {row['run_id']} to {os.path.relpath(HISTORY_CSV, REPO_ROOT)}")
    print(f"Saved per-turn artifact to {os.path.relpath(artifact_path, REPO_ROOT)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
