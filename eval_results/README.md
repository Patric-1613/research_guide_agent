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
- `runs/` — per-run detail artifacts. From `ragas_eval.py`
  (`run_<id>.json` + `raw_<timestamp>.jsonl`), and, as of R7E.1, from
  the `chat_relevance` suite (`chat_relevance_run_<run_id>.json`, one
  per mock or live run, holding the full per-candidate debug record).
  **Not tracked** (`.gitignore`d as of Phase 15) — reviewed locally;
  growing without bound is expected, and isn't meant to accumulate in
  git history the way the history CSVs above do.
- `archive/` — historical/manual before-after snapshots of the two
  history CSVs above, kept as comparison points for specific past
  experiments. **Tracked.** See `archive/README.md` for what each one
  captures.
