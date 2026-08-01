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
- `latency_history.csv` — a specific historical measurement, referenced
  directly by root `README.md`'s "Search-call parallelization" section.
  **Tracked.** No script currently reproduces it — see
  `docs/evaluation.md`'s "Latency measurement" note.
- `runs/` — per-run detail artifacts from `ragas_eval.py`
  (`run_<id>.json` + `raw_<timestamp>.jsonl`). **Not tracked**
  (`.gitignore`d as of Phase 15) — reviewed locally; growing without
  bound is expected, and isn't meant to accumulate in git history the
  way the two history CSVs above do.
- `archive/` — historical/manual before-after snapshots of the two
  history CSVs above, kept as comparison points for specific past
  experiments. **Tracked.** See `archive/README.md` for what each one
  captures.
