# Evaluation workflow

This document is the canonical reference for how this project's evaluation
harnesses work, what they produce, and how their output is organized.
Nothing here changes retrieval/ranking behavior or algorithm logic — this
is a workflow/artifact-organization document, written as part of Phase 15
of `specs/migration-plan.md`'s standardization effort (see
`specs/remaining-standardization-plan.md` for that effort's current
status).

Both harnesses below make real, billable OpenAI calls (plus real arXiv/
Semantic Scholar/Tavily calls where relevant) — neither is part of
`pytest`/CI, and neither is mocked. `tests/` (`uv run pytest -q`) is the
fully deterministic, mocked, zero-API-key suite; these two scripts are
the opposite: real-pipeline, manually-run, cost-aware tools.

## Canonical commands

### Retrieval evaluation (`scripts/eval_retrieval.py`)

Measures whether the real search pipeline surfaces the papers a human
confirmed are relevant, for a fixed 17-topic reference set
(`eval_data/reference_topics.json`).

```bash
# Baseline: the live app's actual default retrieval path
uv run python scripts/eval_retrieval.py --note "baseline run"

# Ranking-stage experiments (opt-in only — never the live app's own path
# except citation_partition, which IS the live default; see docs/
# architecture.md's ranking.py note)
uv run python scripts/eval_retrieval.py --note "..." --ranking-mode bm25
uv run python scripts/eval_retrieval.py --note "..." --ranking-mode hybrid
uv run python scripts/eval_retrieval.py --note "..." --ranking-mode citation_partition
uv run python scripts/eval_retrieval.py --note "..." --ranking-mode langgraph_agent

# Small-scale sanity check before a full run (cost control)
uv run python scripts/eval_retrieval.py --note "sanity check" --topic-ids peft-01,rag-02

# Full flag reference
uv run python scripts/eval_retrieval.py --help
```

Every run appends a row to `eval_results/retrieval_history.csv` (unless
`--topic-ids` narrows the run — see `--help`).

### RAGAS quality evaluation (`scripts/ragas_eval.py`)

Runs the real pipeline (real search + `qa.py`'s real `ask()`) through all
four RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision,
Context Recall) against a curated, hand-verified 24-scenario test set
(`eval_data/stage1_ragas_questions.json`).

```bash
uv run python scripts/ragas_eval.py --note "..."
uv run python scripts/ragas_eval.py --help
```

Every run appends a row to `eval_results/history.csv`, writes
`eval_results/runs/run_<id>.json` (the full scored per-turn record), and
writes an incremental `eval_results/runs/raw_<timestamp>.jsonl` turn-by-
turn during generation (so already-paid-for generation data survives
even if scoring itself later crashes or rate-limits).

### Latency measurement — currently a documented gap, not a command

Root `README.md`'s "Search-call parallelization" section links to
`eval_results/latency_history.csv` as the full per-topic before/after
data for that specific measurement. **As of this writing, no script in
`scripts/` reproduces it** — this is stated honestly here rather than
inventing a command that doesn't exist. If this needs re-measuring in
the future, either commit a small `scripts/eval_latency.py` that
reproduces the methodology described in that README section, or
relabel the CSV as a one-time historical measurement rather than a
repeatable eval. See `eval_results/archive/README.md` for the same note
in the archive context.

## Artifact policy

| Location | What it is | Tracked in git? |
|---|---|---|
| `eval_data/` | **Input fixtures** — the reference sets both harnesses read from. `reference_topics.json` (17-topic retrieval set) and `stage1_ragas_questions.json` (24-scenario RAGAS set) are hand-curated/converted and tracked. `reference_topics.xlsx` (the raw human-maintained spreadsheet `reference_topics.json` is converted from, via `scripts/convert_reference_sheet.py`) is binary and not diff-friendly, so it's `.gitignore`d — the JSON is the reviewable, tracked source of truth. |
| `eval_results/retrieval_history.csv` | **Current, actively-appended-to running log** for `eval_retrieval.py`. Tracked. Expected to show as locally modified after any real local run — that's the log working as designed, not debris (this has been the case throughout this project's whole standardization effort). |
| `eval_results/history.csv` | Same, for `ragas_eval.py`. Tracked, same expected-modified behavior. |
| `eval_results/runs/` | **Generated, per-run artifacts** from `ragas_eval.py` (`run_<id>.json` + `raw_<timestamp>.jsonl`) — one pair of files per run, growing without bound. Not currently tracked; `.gitignore`d as of Phase 15, since this is a repeatable-command output rather than a small aggregate log, and committing every run's full per-turn detail indefinitely doesn't scale the way one history-log row per run does. Review these locally; commit a specific one manually (`git add -f`) only if a specific run's detail genuinely needs to be shared. |
| `eval_results/archive/` | **Historical/manual snapshots** — before/after comparison points for specific past experiments, not ongoing logs. See `eval_results/archive/README.md` for what each one captures. Tracked (they're small, finite, and already-created; no new ones are expected to be added routinely). |
| `eval_results/latency_history.csv` | A specific historical measurement, tracked, referenced directly by README — see "Latency measurement" above for its reproducibility gap. |

**Rule of thumb for anything new**: a small, ever-growing aggregate log
that one eval run appends one row to → tracked, in `eval_results/`
directly. A larger, per-run detail dump → `eval_results/runs/`,
`.gitignore`d, reviewed locally. A deliberate before/after comparison
snapshot of an aggregate log → `eval_results/archive/`, tracked, with an
entry in that directory's `README.md` explaining what it captures.

## Related docs

- `docs/architecture.md` — backend architecture, including why
  `ranking.py`'s citation-partitioned reranking is live in `agent.py`'s
  default path (relevant context for the `--ranking-mode
  citation_partition` command above).
- `eval_results/archive/README.md` — what each archived snapshot
  captures.
- `specs/remaining-standardization-plan.md` — current status of the
  broader current-project standardization effort this document is part
  of.
