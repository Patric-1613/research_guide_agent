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
- `report_refinement_history.csv` — current running log, appended to
  by every `uv run python -m research_agent.evals.cli run --suite
  report_refinement` run (R6D.2 mock + R6D.3 live). `--mode mock` runs
  both the draft and refined report in each R6D.1 pair fixture through
  R6B's own deterministic checks (`run_report_quality.predict()`,
  called directly, never reimplemented) and scores only whether the
  derived `hard_failure_direction` (`improved`/`unchanged`/`regressed`/
  `mixed`) matches the fixture's expectation — **this average_score is
  structural-direction agreement only, not a report-quality or
  refinement-effectiveness score**; the 7 semantic R6C dimensions are
  never fabricated (`dimension_directions` is always `null` in a mock
  prediction, zero OpenAI calls). `--mode live` (R6D.3) runs both
  sides through R6C's existing live prediction path (`run_report_
  quality.predict_live()`, reused directly) independently — **up to 4
  real judge calls per pair**, fewer when the identical-input
  optimization applies for a `revision_applied=false` pair with
  byte-identical reports — then scores per-dimension expectation
  agreement (`matched / 7`, exact 7/7 required to fully pass); missing
  credentials exit 2 immediately with no API call and no artifact
  written, never a silent fallback to mock. **No conclusion is drawn
  yet that refinement improves report quality** — the 0.10 holistic
  direction threshold (`run_report_refinement.HOLISTIC_DIRECTION_MIN_
  DELTA`) is explicitly provisional and uncalibrated. **Tracked**,
  same conventions as the other suite logs above. See `docs/
  evaluation.md`'s "R6D.2"/"R6D.3" sections and `eval_data/report_
  refinement/README.md` for the pair schema this suite's `prediction`
  follows.
- `report_refinement_real_history.csv` — current running log, appended
  to by every `uv run python -m research_agent.evals.cli run --suite
  report_refinement_real` run (R6D.4d). Evaluates exactly the 3 frozen
  real R4 capture pairs in `captures/` below against their own frozen
  adjudications in `eval_data/report_refinement/real_reviews/` —
  reuses `report_refinement`'s own `predict`/`predict_live` and
  evaluators unchanged, and is kept in a separate history/detail-file
  series so 3 real, individually-adjudicated pairs are never pooled
  with the 7 synthetic fixtures above. `average_score` here is exact
  agreement with the frozen adjudication labels, never a report-quality
  or refinement-benefit measurement. Two rows: `run_id=1` (mock
  plumbing check), `run_id=2` (the one bounded live run, commit
  `003ad81`, `average_score=0.6667`) — **both permanent**; a later,
  independently-verified structural correction to `real-analytical-01`'s
  adjudication (its `hard_failure_direction` label, not this CSV) never
  rewrote or reran either row. **Tracked**, same conventions as the
  other suite logs above. **R6D is closed as of this suite's one live
  run** — no further live reruns are planned. See `eval_data/report_
  refinement/README.md`'s own "`report_refinement_real` suite" section
  for the loader's hash-binding/validation behavior, the live-mode call
  bound, and the full result, and `docs/evaluation.md`'s "R6D.4d"
  section for the per-pair/per-dimension breakdown.
- `runs/` — per-run detail artifacts. From `ragas_eval.py`
  (`run_<id>.json` + `raw_<timestamp>.jsonl`), from the `chat_relevance`
  suite (`chat_relevance_run_<run_id>.json`, R7E.1), from the
  `report_quality` suite (`report_quality_run_<run_id>.json`, one per
  run — mock runs hold each fixture's structural-check prediction
  only (`structural_integrity`, `informational_signals`, `warnings`,
  the fixture-agreement evaluator's result); live runs additionally
  hold each judge's full per-claim verdicts, `judge_dimensions`, and
  `judge_metadata` — model, both prompt versions, the aggregation
  policy version, latency, token usage, sampling coverage, and
  sanitization counts), and from the `report_refinement` suite
  (`report_refinement_run_<run_id>.json`, R6D.2 mock + R6D.3 live —
  each pair's own `prediction` holds `draft`/`refined` sub-dicts, each
  with `hard_failures`/`structural_status`/`informational_signals`
  (mock and live) plus `judge_dimensions`/`judge_metadata`/`error`
  (live only), and the pair-level `hard_failure_direction`,
  `dimension_directions` (always `null` in mock mode, one of 7
  directions per dimension in live mode), `semantic_evaluation_status`,
  and `identical_input_reused`). **Not
  tracked** (`.gitignore`d as of Phase 15) — reviewed locally; growing
  without bound is expected, and isn't meant to accumulate in git
  history the way the history CSVs above do.
- `archive/` — historical/manual before-after snapshots of the two
  history CSVs above, kept as comparison points for specific past
  experiments. **Tracked.** See `archive/README.md` for what each one
  captures.
- `captures/` — R6D.4b's own output: real R4-generated draft/refined
  report pairs (`research_agent.evals.cli capture-refinement`), one
  unlabelled `r6d4-capture-v1` JSON file per `--pair-id`. **Not
  tracked** (`.gitignore`d) — a temporary, local working area, separate
  from `eval_data/report_refinement/`'s tracked, labelled synthetic
  fixtures and from `runs/` above. Every file here is unlabelled by
  construction (no `expected` block, no `overall_score`/`winner`/
  `accept_refinement` field) until a separate, later human-adjudication
  step assigns expectations -- which must always happen BEFORE that
  pair's own R6D live evaluation runs, never after, to avoid post-hoc
  answer-key bias. See `docs/evaluation.md`'s "R6D.4a"/"R6D.4b"
  sections.
