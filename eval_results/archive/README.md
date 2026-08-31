# Archived eval snapshots

The files in this directory are **manual, historical snapshots** — not
the canonical, currently-generated eval output. They were kept as
before/after comparison points for specific pieces of work (now
documented in `docs/evaluation.md`), not as ongoing run logs.

The canonical, currently-generated eval outputs live one level up, in
`eval_results/` itself:
- `eval_results/retrieval_history.csv` — `scripts/eval_retrieval.py`'s
  running history log (appended to on every real run).
- `eval_results/history.csv` — `scripts/ragas_eval.py`'s running history
  log, plus `eval_results/runs/` for its per-run JSON/JSONL artifacts.

Nothing in this `archive/` directory is read by either harness or by any
test — moving these files here does not change what either script writes
to or reads from.

## What's here

| File | What it captures |
|---|---|
| `retrieval_history_pre_ranking_experiment.csv` | `eval_retrieval.py`'s history log as it stood before the BM25/hybrid/citation-partitioned ranking-mode experiments (see `docs/evaluation.md`'s ranking-experiment findings) — a before-snapshot, not a duplicate of the current log. |
| `retrieval_history_pre_citation_partition.csv` | Same log, snapshotted again immediately before the citation-partitioned reranking experiment specifically — narrower before-point than the file above. |
| `history_throwaway_smoketest.csv` | `ragas_eval.py`'s history log from an early smoke-test run, kept as a small reference example rather than a comparison point. |
| `history_two_metric_run1.csv` | `ragas_eval.py`'s history log from a run before all four RAGAS metrics were wired in (two metrics only) — a before-snapshot of that expansion. |

## `latency_history.csv`

`eval_results/latency_history.csv` (one level up, **not** moved into this
archive) holds the per-topic before/after data for the search-call
parallelization measurement (`docs/evaluation.md`'s "Search-call
parallelization" note). **No script in `scripts/` reproduces it** — it
was committed alongside the latency win, but the measurement script was
never committed. It is a one-time historical measurement; regenerating it
would mean committing a reproducing script first. Tracked in
`specs/backend-backlog.md`'s Technical Debt section.

## Convention going forward

Any new historical/comparison snapshot of a running eval log should be
named descriptively (as the files above are) and placed directly in this
`archive/` directory, not left flat in `eval_results/` alongside the
current running logs. Any new *generated* eval output that isn't meant to
be a permanent running log (e.g. a one-off experiment's raw dump) should
either have a documented, repeatable command producing it, or be
`.gitignore`d rather than committed as an undocumented file.
