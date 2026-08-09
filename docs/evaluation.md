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

## Planned evaluation architecture (E0 decision, 2026-08-08)

**Update (R7D.1, 2026-08-08)**: the `chat_relevance` suite below is now
real — `research_agent/evals/` (`cli.py`, `runners/`, `evaluators/`),
`eval_data/chat_web_relevance_redteam.jsonl` (9 hand-curated cases
covering the R7A-R7C red-team scenarios), and `eval_results/
chat_relevance_history.csv`.

**Update (R7D.2, 2026-08-08)**: `--mode live` is now implemented for
`chat_relevance` — it runs the real `_filter_relevant_web_articles`
through a real `OpenAI()` client (same construction `qa.ask()` uses),
never falls back to mock, and requires explicit opt-in (`--mode` still
defaults to `mock`). It fails cleanly (no traceback) if OpenAI
credentials aren't set, and prints a cost warning before running.
Fixture cases that simulate a condition only reachable in mock mode
(originally `chat_relevance_008`/`009`'s embedding-API failures; as of
R7E.5b also `chat_relevance_014`'s forced judge-uncertain verdict) are
marked `mock_only: true` and are skipped in live mode with a clear,
**fixture-specific** reason (an optional `mock_only_reason` field on
the fixture record itself, falling back to a generic truthful message
if absent) — not a single hardcoded message, since different mock_only
cases simulate genuinely different things. Everything else in this
section (the mentor-repo comparison, phase order, the `report_quality`
suite) is still design-only, not yet implemented.

This section records the architecture decided during
E0, an audit-and-design-only checkpoint that studied a mentor repo's
`backend/evals/` folder (github.com/cwijayasundara/document_intelligence_
adv_v2) as a reference pattern for this project's own next eval phases
(R7D, R6) — adapting what fits, deliberately not copying what doesn't.
No code, dependency, or eval run happened as part of E0 itself.

**Package layout (later)**: a small `research_agent/evals/` code
package — `cli.py`, `runners/`, `evaluators/`, and shared base-runner
utilities (dataset loading, the predict → evaluate → aggregate loop).
Code only, safe inside the package: nothing in it will be imported by
`api_app/app.py` or any router, so it stays inert at runtime, the same
precedent `ranking.py`'s own eval-only BM25/hybrid modes already set
for eval-only code living inside `research_agent/`. This isn't a new
idea — `specs/migration-plan.md`'s own original Phase 6 already
proposed almost exactly this shape (`research_agent/evals/{datasets,
runners,evaluators}/` + `cli.py`); it was deferred and never executed
(see `specs/remaining-standardization-plan.md`). The mentor repo is
independent validation that the original plan's shape was right, not a
new direction.

**Fixtures stay in `eval_data/`.** New golden/red-team sets (e.g. a
future `chat_web_relevance_redteam.json`, `report_quality_golden.json`)
land alongside the existing `reference_topics.json`/
`stage1_ragas_questions.json` — one canonical place for "where are eval
fixtures," not split across `eval_data/` and a second location inside
`research_agent/evals/`.

**Results stay in `eval_results/`**, one new CSV per new suite —
`eval_results/chat_relevance_history.csv`, `eval_results/
report_quality_history.csv` — never appended into the existing
`retrieval_history.csv`/`history.csv`, which stay scoped to their own
harnesses. Per-run detail follows the existing `eval_results/runs/`
convention (gitignored, growing without bound, reviewed locally).

**`scripts/eval_retrieval.py` and `scripts/ragas_eval.py` are
unchanged by this decision** — both stay exactly as documented above,
not wrapped or migrated into the new package in this phase. Revisiting
that is an explicit future option, not a requirement of building the
new suites.

**CLI shape** (the first three lines are real as of R7D.1/R7D.2; the
`report_quality` line is still planned, not yet implemented):

```bash
uv run python -m research_agent.evals.cli list-suites
uv run python -m research_agent.evals.cli run --suite chat_relevance --mode mock
uv run python -m research_agent.evals.cli run --suite chat_relevance --mode live --subset 5
uv run python -m research_agent.evals.cli run --suite report_quality --mode live --tags redteam
```

`--mode` defaults to `mock` (offline, no live model/web calls) —
`live` is always an explicit, opt-in flag, never the default and never
invoked implicitly by another command. Same cost-conscious posture the
two existing harnesses already have, made explicit as a flag instead of
being the only mode available.

**pytest and eval runners stay two separate things**, continuing this
doc's own existing line (see the top of this file): pytest
(`tests/`, `uv run pytest -q`) proves deterministic *code* behavior —
fully mocked, zero API key, part of the routine dev loop. Eval runners
measure *product/agent* behavior over realistic scenarios — separate,
manually invoked, cost-aware, never part of pytest/CI.

**Borrowed from the mentor repo**: the JSONL example format, the
evaluator function shape (`(prediction, expected) -> {"key", "score",
"comment"}`), `--subset`/`--tags` flags, a deterministic-checks +
LLM-as-judge split, the red-team-suite concept (a tagged subset of a
golden set specifically probing known failure classes), and a
consolidated result-summary table at the end of a run.

**Deliberately not copied yet**: Postgres-backed eval persistence
(`eval_runs`/`eval_examples`/`eval_results` tables) — this project has
no database engine beyond SQLite and hasn't decided to add one;
LangSmith as a dataset/experiment system of record — present in
`uv.lock` only as a transitive pin, not a direct dependency, and not
something any of this project's own code imports; a FastAPI `/evals`
dashboard route — downstream of the Postgres decision above; a
synthetic (LLM-generated) golden-dataset generator — this project has
consistently preferred small, hand-verified fixture sets over
LLM-synthesized ones at its current scale; automated regression
harvesting from production "user correction" memory — no persistent
memory store exists here to harvest from (this project already applies
the same underlying principle by hand instead — R7A's entire red-team
fixture set was built directly from one real reported incident).

**Phase order after E0**:
1. **R7D** — chat/web retrieval evaluation foundation.
2. **R6** — report quality evaluation foundation.
3. **Later** — threshold calibration, an LLM gray-zone judge for
   borderline relevance scores, Langfuse metrics for the new relevance
   signals, and trend reports/a dashboard.

**Why this order**: R7D comes first because it directly targets a real
failure already observed in the app (the housing-vs-AI-governance
citation — see `docs/architecture.md`'s R7 section). R6 follows because
report quality should be measured once report generation, export, and
refinement are already stable — which they now are (R2C through R5D
are complete).

See `specs/backend-backlog.md`'s E0 entry for the same decisions in
backlog form.

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
