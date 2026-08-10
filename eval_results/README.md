# Eval outputs

Generated output from the evaluation harnesses in `scripts/`. See
`docs/evaluation.md` for the full canonical commands and artifact
policy — this file is a short local index.

- `retrieval_history.csv` — current running log, appended to by every
  `scripts/eval_retrieval.py` run. **Tracked**, and expected to show as
  locally modified after any real local run — that's the log working as
  designed, not debris.
- `history.csv` — current running log, appended to by every
  `scripts/ragas_eval.py` run. **Tracked**, same expected-modified
  behavior.
- `chat_relevance_history.csv` — current running log, appended to by
  every `uv run python -m research_agent.evals.cli run --suite
  chat_relevance` run, mock (default) or live (opt-in via `--mode
  live`, R7D.2/R7E.5 — live now covers both the embedding calls and the
  direct-relevance judge model). **Tracked**, same expected-modified
  behavior as the two logs above. Separate file by design — new eval
  suites get their own CSV, never appended into
  `retrieval_history.csv`/`history.csv`. Header stays the same
  11-column shape R7D.1 established either way — a live run's
  skipped-case count and mean latency are folded into the free-text
  `note` column instead of adding new columns, to keep every row (mock
  or live, old or new) reading against one stable header. `run_id`
  correlates each row to its per-example detail file in `runs/` below
  (R7E.1). See `docs/evaluation.md`'s "R7E — chat relevance evaluation
  arc" section for the live-run evidence history.
- `latency_history.csv` — a specific historical measurement, referenced
  directly by root `README.md`'s "Search-call parallelization" section.
  **Tracked.** No script currently reproduces it — see
  `docs/evaluation.md`'s "Latency measurement" note.
- `report_quality_history.csv` — current running log, appended to by
  every `uv run python -m research_agent.evals.cli run --suite
  report_quality` run. `--mode mock` (R6B, the default) runs
  deterministic structural/citation checks only, no OpenAI calls, no
  qualitative judge score. **`--mode live` is implemented (R6C.2)** —
  two opt-in live judges (claim/source + holistic; see `docs/
  evaluation.md`'s "R6C.1"-"R6C.3" sections) add real, billable model
  calls and populate `judge_dimensions`/`judge_metadata` in the
  per-run detail file. Run_id 1 is the R6B mock baseline; run_ids 2-9
  are live runs spanning the R6C.2 smoke checks through the R6C.3
  full-benchmark and targeted-validation calibration pass (`tags`
  column: empty for early single-fixture smokes, `baseline`/
  `foundational`/`security` for R6C.3a's targeted reruns). **Tracked**,
  same expected-modified behavior and same 11-column header shape as
  the other suite logs above. `run_id` correlates each row to its
  per-example detail file in `runs/` below. See `specs/
  report-quality-evaluation-plan.md` for the frozen result schema/
  hard-failure identifiers and §12-14 for the aggregation policy this
  suite's live `prediction` follows.
- `runs/` — per-run detail artifacts. From `ragas_eval.py`
  (`run_<id>.json` + `raw_<timestamp>.jsonl`), from the `chat_relevance`
  suite (`chat_relevance_run_<run_id>.json`, R7E.1), and from the
  `report_quality` suite (`report_quality_run_<run_id>.json`, one per
  run — mock runs hold each fixture's structural-check prediction
  only (`structural_integrity`, `informational_signals`, `warnings`,
  the fixture-agreement evaluator's result); live runs additionally
  hold each judge's full per-claim verdicts, `judge_dimensions`, and
  `judge_metadata` — model, both prompt versions, the aggregation
  policy version, latency, token usage, sampling coverage, and
  sanitization counts). **Not tracked** (`.gitignore`d as of Phase 15)
  — reviewed locally; growing without bound is expected, and isn't
  meant to accumulate in git history the way the history CSVs above
  do.
- `archive/` — historical/manual before-after snapshots of the two
  history CSVs above, kept as comparison points for specific past
  experiments. **Tracked.** See `archive/README.md` for what each one
  captures.
