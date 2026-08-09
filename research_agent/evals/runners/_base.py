"""Shared eval-runner plumbing -- JSONL loading, the predict -> evaluate
-> aggregate loop, and a compact printed/returned summary.

E0's design decision (docs/evaluation.md's "Planned evaluation
architecture" section) is what this implements: fixtures live in
eval_data/, not inside this package; results log to eval_results/ as a
small appended CSV, never a database; every suite defaults to a mocked/
offline mode, live is a separate, explicitly opt-in mode.

R7D.2: this module itself stays mode-agnostic -- it has no idea what
"mock" or "live" means, and never will. `run_suite`'s `predict`
callable and optional `skip_if` predicate are where a runner's own
mode-specific behavior (a mocked embedding client vs. a real one, which
cases don't make sense to run for real) lives -- see runners/
run_chat_relevance.py. This keeps the predict -> evaluate -> aggregate
loop itself identical across every mode of every suite, so mock and
live never duplicate scoring logic.

Example/evaluator shape borrowed from the mentor repo studied during E0
(github.com/cwijayasundara/document_intelligence_adv_v2's backend/evals),
adapted to this project: no LangSmith `Run`/`Example` SDK objects, no
Postgres persistence -- just plain dataclasses and a CSV append.

R7E.1: `SuiteResult.per_example` now carries full per-example detail
(inputs, expected outputs, prediction, evaluator results, latency,
skip/error info), and `write_run_detail_json` persists it to
`eval_results/runs/<suite>_run_<run_id>.json` -- instrumentation only,
no filtering/scoring semantics changed.
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

EVAL_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "eval_data"
EVAL_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "eval_results"

# A single evaluator: (prediction, expected) -> {"key": str, "score": float | bool | None, "comment": str}
Evaluator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]

# Keys that are metadata, never inputs or outputs -- same split the mentor
# repo's own load_examples()/dataset_sync.py use, adapted (no `pe_checklist`-
# style domain-specific exception here; this project's suites don't need one
# yet, and adding one speculatively would be exactly the kind of unused
# generality this project avoids elsewhere). `mock_only` (R7D.2) marks a
# fixture case that simulates something (e.g. an embedding-API exception,
# or -- as of R7E.5b -- a forced direct-relevance judge verdict) that
# can't be forced against a real API -- a runner's own `skip_if` can use
# it to skip the case in live mode instead of forcing an artificial
# failure; see run_chat_relevance.py. `mock_only_reason` (R7E.5b) is an
# optional, case-specific explanation of WHY -- a single hardcoded skip
# message (originally written when embedding-failure cases were the only
# mock_only ones) went stale the moment a second, differently-motivated
# mock_only case (judge-uncertain) was added; see run_chat_relevance.py's
# own `_mock_only_skip_reason` for the fallback when this is absent.
_METADATA_KEYS = {"id", "tags", "source", "notes", "mock_only", "mock_only_reason"}


class LiveModeSetupError(RuntimeError):
    """Raised when a suite's live mode can't even start -- missing API
    credentials, an embedding client that fails to construct, etc. A
    runner raises this (not a bare exception) specifically so the CLI
    can catch it and print one clean, non-traceback line: this is a
    setup failure, not a per-example prediction failure (those are
    caught individually inside run_suite's own loop and simply recorded
    as a failed/errored example, not a fatal error)."""


@dataclass
class Example:
    id: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    metadata: dict[str, Any]


def _split_inputs_outputs(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """`expected_`/`reference_`-prefixed keys are expected outputs; every
    other non-metadata key is an input. Same heuristic the mentor repo's
    own `runners/_base.py::load_examples` and `dataset_sync.py::
    _split_inputs_outputs` both use (duplicated there; kept in this one
    place here on purpose)."""
    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    for key, value in record.items():
        if key in _METADATA_KEYS:
            continue
        if key.startswith("expected_") or key.startswith("reference_"):
            outputs[key] = value
        else:
            inputs[key] = value
    return inputs, outputs


def load_examples(
    dataset_file: str, subset: int | None = None, tags: list[str] | None = None,
) -> list[Example]:
    """Loads one JSONL fixture file from eval_data/ (never from inside
    this package -- see EVAL_DATA_DIR). `subset` takes the first N
    examples AFTER tag filtering, matching the mentor repo's own runner
    convention (a `--tags redteam --subset 3` call means "the first 3
    red-team cases", not "3 cases, then filter to red-team among them").
    Blank lines and `#`-prefixed lines are skipped, same convention this
    project's other JSON/CSV-adjacent eval fixtures already tolerate.
    """
    path = EVAL_DATA_DIR / dataset_file
    if not path.exists():
        raise FileNotFoundError(f"eval fixture not found: {path}")

    examples: list[Example] = []
    with path.open() as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num}: invalid JSON: {exc}") from exc

            if tags:
                record_tags = set(record.get("tags") or [])
                if not any(t in record_tags for t in tags):
                    continue

            example_id = record.get("id") or f"anon_{len(examples)}"
            metadata = {
                "tags": record.get("tags") or [],
                "source": record.get("source") or "",
                "notes": record.get("notes") or "",
                "mock_only": bool(record.get("mock_only", False)),
                "mock_only_reason": record.get("mock_only_reason") or None,
            }
            inputs, outputs = _split_inputs_outputs(record)
            examples.append(Example(id=example_id, inputs=inputs, outputs=outputs, metadata=metadata))

    if subset is not None:
        examples = examples[:subset]
    return examples


@dataclass
class SuiteResult:
    suite: str
    mode: str
    total: int
    passed: int
    failed: int
    average_score: float | None
    per_example: list[dict[str, Any]] = field(default_factory=list)
    skipped: int = 0
    mean_latency_ms: float | None = None

    def summary_line(self) -> str:
        avg = f"{self.average_score:.3f}" if self.average_score is not None else "n/a"
        skipped_bit = f" skipped={self.skipped}" if self.skipped else ""
        latency_bit = f" mean_latency_ms={self.mean_latency_ms:.1f}" if self.mean_latency_ms is not None else ""
        return (
            f"[eval] suite={self.suite} mode={self.mode} "
            f"total={self.total} passed={self.passed} failed={self.failed} "
            f"average_score={avg}{skipped_bit}{latency_bit}"
        )


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:  # noqa: BLE001 -- git not available (e.g. a source tarball) degrades to "unknown", not a crash.
        return None


def run_suite(
    suite: str,
    dataset_file: str,
    predict: Callable[[Example], dict[str, Any]],
    evaluators: list[tuple[str, Evaluator]],
    mode: str,
    subset: int | None = None,
    tags: list[str] | None = None,
    skip_if: Callable[[Example], str | None] | None = None,
) -> SuiteResult:
    """Runs one suite end-to-end: load -> predict -> evaluate -> aggregate.

    An example "passes" when every evaluator's score for it is exactly
    1.0 (missing/None scores don't count against it -- an evaluator that
    declines to score a case, e.g. "no expected_relevant_urls given", is
    not a failure). `average_score` is the mean of every finite numeric
    score across every example/evaluator pair, or None if nothing
    produced a numeric score at all.

    `skip_if(example)`, if given, returning a non-None reason string
    skips that example entirely -- no predict() call, no evaluator call,
    excluded from total/passed/failed/average_score (R7D.2: this is how
    a live-mode runner opts out of fixture cases that only make sense
    against a mocked client, e.g. a simulated embedding-API exception --
    see run_chat_relevance.py's own skip_if). Skipped examples still
    appear in per_example, tagged `"skipped": True`, so a run's output
    always accounts for every loaded example.
    """
    examples = load_examples(dataset_file, subset=subset, tags=tags)
    logger.info("suite=%s mode=%s loaded %d examples from %s", suite, mode, len(examples), dataset_file)

    passed = 0
    failed = 0
    skipped = 0
    all_scores: list[float] = []
    latencies_ms: list[float] = []
    per_example: list[dict[str, Any]] = []

    for example in examples:
        example_tags = example.metadata.get("tags", [])
        skip_reason = skip_if(example) if skip_if else None
        if skip_reason:
            skipped += 1
            per_example.append({
                "example_id": example.id, "tags": example_tags,
                "inputs": example.inputs, "expected": example.outputs,
                "skipped": True, "skipped_reason": skip_reason,
                "prediction": None, "evaluator_results": None, "latency_ms": None, "error": None,
            })
            continue

        start = time.perf_counter()
        error: str | None = None
        try:
            prediction = predict(example)
        except Exception as exc:  # noqa: BLE001 -- record the failure, keep going, same posture the mentor repo's runner uses.
            logger.exception("predict() failed for example %s", example.id)
            error = str(exc)
            prediction = {"error": error}
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        evaluator_results: dict[str, Any] = {}
        example_passed = True
        for name, evaluator in evaluators:
            try:
                result = evaluator(prediction, example.outputs)
            except Exception as exc:  # noqa: BLE001 -- one evaluator must not kill the whole run.
                logger.exception("evaluator=%s example=%s failed", name, example.id)
                result = {"key": name, "score": None, "comment": f"error: {exc}"}
            key = result.get("key", name)
            evaluator_results[key] = result
            raw_score = result.get("score")
            if isinstance(raw_score, (int, float)):
                all_scores.append(float(raw_score))
                if float(raw_score) != 1.0:
                    example_passed = False
            # A None score (evaluator declined to judge) never fails the example on its own.

        if example_passed:
            passed += 1
        else:
            failed += 1
        per_example.append({
            "example_id": example.id, "tags": example_tags,
            "inputs": example.inputs, "expected": example.outputs,
            "skipped": False, "skipped_reason": None,
            "prediction": prediction, "evaluator_results": evaluator_results,
            "latency_ms": round(elapsed_ms, 2), "error": error,
        })

    average_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else None
    mean_latency_ms = round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else None
    return SuiteResult(
        suite=suite, mode=mode, total=passed + failed, passed=passed, failed=failed,
        average_score=average_score, per_example=per_example, skipped=skipped,
        mean_latency_ms=mean_latency_ms,
    )


_CSV_FIELDS = ["run_id", "date", "git_commit", "suite", "mode", "total", "passed", "failed", "average_score", "tags", "note"]


def _enrich_note(result: SuiteResult, note: str) -> str:
    """Folds skipped-count/mean-latency into the free-text `note` column
    instead of adding new CSV columns (R7D.2). `chat_relevance_history.csv`
    already has tracked rows written under R7D.1's original 11-column
    header -- appending rows with a wider header would misalign every
    existing row's columns against a mismatched header line, contrary to
    docs/evaluation.md's append-only artifact policy. Free-text is a
    smaller, fully backward-compatible way to still capture this."""
    extras = []
    if result.skipped:
        extras.append(f"skipped={result.skipped}")
    if result.mean_latency_ms is not None:
        extras.append(f"mean_latency_ms={result.mean_latency_ms:.1f}")
    return " | ".join(part for part in (note, ", ".join(extras)) if part)


def append_result_csv(
    result: SuiteResult, csv_path: Path, tags: list[str] | None = None, note: str = "",
) -> int:
    """Appends one row to `csv_path` (creating it with a header if it
    doesn't exist yet) -- the exact append-only convention `docs/
    evaluation.md`'s artifact policy already establishes for
    `retrieval_history.csv`/`history.csv`. `csv_path` is a parameter
    (not hardcoded to eval_results/), specifically so tests can point it
    at a tmp_path instead of ever touching a real tracked file.

    Returns the `run_id` this row was assigned, so a caller (R7E.1:
    cli.py's cmd_run) can use the same id to name a companion per-example
    detail file under eval_results/runs/ -- see write_run_detail_json.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    next_run_id = 1
    if file_exists:
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            next_run_id = max(int(r["run_id"]) for r in rows) + 1

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "run_id": next_run_id,
            "date": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_sha() or "unknown",
            "suite": result.suite,
            "mode": result.mode,
            "total": result.total,
            "passed": result.passed,
            "failed": result.failed,
            "average_score": result.average_score if result.average_score is not None else "",
            "tags": ";".join(tags) if tags else "",
            "note": _enrich_note(result, note),
        })
    return next_run_id


def write_run_detail_json(
    result: SuiteResult,
    run_id: int,
    runs_dir: Path,
    subset: int | None = None,
    tags: list[str] | None = None,
    note: str = "",
) -> Path:
    """R7E.1: writes the full per-example detail `append_result_csv`'s
    aggregate-only CSV row can't carry -- predictions, evaluator
    results/comments, latency, skip reasons, and (for chat_relevance's
    live mode) the raw query/topic similarity scores riding along inside
    each example's `prediction` dict (see run_chat_relevance.py's
    `predict_live`). Filed at `runs_dir / f"{suite}_run_{run_id}.json"`,
    the same `run_id` `append_result_csv` assigned that run's CSV row, so
    the two artifacts correlate 1:1. Mirrors the `eval_results/runs/`
    convention `scripts/ragas_eval.py` already established (gitignored,
    reviewed locally, not meant to accumulate in git history the way the
    history CSVs do) -- extended here to every suite this package runs,
    not just ragas_eval.py's own.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    detail = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha() or "unknown",
        "suite": result.suite,
        "mode": result.mode,
        "subset": subset,
        "tags": tags or [],
        "note": note,
        "total": result.total,
        "passed": result.passed,
        "failed": result.failed,
        "skipped": result.skipped,
        "average_score": result.average_score,
        "mean_latency_ms": result.mean_latency_ms,
        "per_example": result.per_example,
    }
    path = runs_dir / f"{result.suite}_run_{run_id}.json"
    with path.open("w") as f:
        json.dump(detail, f, indent=2, default=str)
    return path
