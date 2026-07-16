#!/usr/bin/env python3
"""RAGAS integration, phase 1: plumbing only.

Proves the RAGAS wiring works end-to-end against REAL output from the
existing pipeline — a real search/rerank pass (same flow as
scripts/test_qa.py) feeding real papers into qa.py's ask(), whose real
`answer` and real retrieved abstracts become RAGAS's `response` and
`retrieved_contexts`. Nothing here is mocked or fabricated.

Only two metrics run: Faithfulness and Answer Relevancy. Context Precision
and Context Recall are NOT attempted — both need a reference/ground-truth
answer set that doesn't exist yet (a separate, later phase), and asking
RAGAS for them without one would either error or silently score against a
fabricated reference.

Not part of pytest or CI. This is a manually-run tool that makes real,
billable OpenAI calls: the pipeline's own answer-generation calls (qa.py,
already billable on its own) PLUS the RAGAS judge model's scoring calls
PLUS embedding calls for Answer Relevancy. See the printed cost summary at
the end of a run for actual token counts and an estimated dollar cost.

Usage:
    python scripts/ragas_eval.py --note "phase 1 smoke test"
    python scripts/ragas_eval.py --note "trying gpt-4.1 as judge" --judge-model gpt-4.1
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_CSV = os.path.join(REPO_ROOT, "eval_results", "history.csv")
HISTORY_FIELDS = [
    "run_id", "date", "git_commit", "judge_model", "num_questions",
    "faithfulness_mean", "answer_relevancy_mean", "note",
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

# --- THROWAWAY test questions -------------------------------------------
# Placeholder questions for proving the RAGAS wiring works end-to-end.
# This is NOT the curated reference eval set (a separate, later phase) —
# just enough real, answerable questions to exercise the real pipeline and
# get real, non-degenerate Faithfulness/Answer Relevancy scores. Disposable;
# replace or extend freely without needing to update anything else.
EVAL_CASES = [
    {
        "topic": "parameter-efficient fine-tuning methods for large language models",
        "question": "What is RoCoFT and how does it work?",
    },
    {
        "topic": "retrieval augmented generation for large language models",
        "question": "What problem does retrieval-augmented generation solve for large language models?",
    },
    {
        "topic": "vector databases for embedding search",
        "question": "What tradeoffs do vector databases make to support fast approximate nearest neighbor search?",
    },
    {
        "topic": "instruction tuning for large language models",
        "question": "How does instruction tuning differ from standard supervised fine-tuning?",
    },
]


def _search_and_rank(topic: str, client: OpenAI, top_k: int = 8):
    """Real search/rerank pass — identical flow to scripts/test_qa.py, not
    mocked: arXiv + Semantic Scholar search, cross-source dedup, embed,
    then semantic rerank. Returns a ranked list of Paper objects."""
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


def collect_cases(client: OpenAI) -> list[dict]:
    """Runs each throwaway question through the real pipeline (search/rerank
    then qa.ask()) and returns RAGAS single-turn samples built from genuine
    output: user_input (the question), retrieved_contexts (the real
    retrieved abstracts), response (the real generated answer)."""
    samples = []
    for i, case in enumerate(EVAL_CASES, 1):
        topic, question = case["topic"], case["question"]
        print(f"[{i}/{len(EVAL_CASES)}] topic={topic!r}")
        papers = _search_and_rank(topic, client)
        print(f"    retrieved {len(papers)} papers, asking: {question!r}")

        session = ChatSession(papers=papers)
        result = ask(session, question, client=client)
        contexts = [p.abstract for p in result["retrieved_papers"] if p.abstract]

        print(f"    answerable={result['answerable']}, {len(contexts)} contexts, "
              f"answer length={len(result['answer'])} chars\n")
        samples.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": result["answer"],
        })
    return samples


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


def print_results_table(result) -> dict[str, float]:
    df = result.to_pandas()

    print("=" * 100)
    print("RAGAS results (per question)")
    print("=" * 100)
    for i, row in df.iterrows():
        print(f"\n[{i + 1}] {row['user_input']}")
        print(f"    faithfulness:      {row['faithfulness']:.3f}")
        print(f"    answer_relevancy:  {row['answer_relevancy']:.3f}")

    means = {
        "faithfulness_mean": float(df["faithfulness"].mean()),
        "answer_relevancy_mean": float(df["answer_relevancy"].mean()),
    }
    print("\n" + "=" * 100)
    print("Mean scores")
    print("=" * 100)
    for k, v in means.items():
        print(f"  {k}: {v:.3f}")
    return means


def report_cost(result, judge_model: str) -> None:
    print("\n" + "=" * 100)
    print("Judge/embedding call cost (RAGAS scoring only — does NOT include the")
    print("pipeline's own answer-generation calls in qa.py, which are separately")
    print("billable and already logged via that module's own usage logging)")
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


def append_history_row(means: dict[str, float], judge_model: str, num_questions: int, note: str) -> dict:
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

    row = {
        "run_id": next_run_id,
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "judge_model": judge_model,
        "num_questions": num_questions,
        "faithfulness_mean": f"{means['faithfulness_mean']:.4f}",
        "answer_relevancy_mean": f"{means['answer_relevancy_mean']:.4f}",
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
    parser.add_argument("--note", required=True, help="Short description of this run, e.g. 'phase 1 smoke test'")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                         help=f"OpenAI model used by RAGAS to judge answers (default: {DEFAULT_JUDGE_MODEL})")
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI()

    print(f"Judge model: {args.judge_model}")
    print(f"Questions:   {len(EVAL_CASES)} (throwaway placeholder set, not the curated eval set)\n")

    samples = collect_cases(client)
    result = run_ragas_eval(samples, args.judge_model)
    means = print_results_table(result)
    report_cost(result, args.judge_model)
    row = append_history_row(means, args.judge_model, len(samples), args.note)

    print("\n" + "=" * 100)
    print(f"Appended run {row['run_id']} to {os.path.relpath(HISTORY_CSV, REPO_ROOT)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
