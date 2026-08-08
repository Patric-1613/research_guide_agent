"""Shared eval-runner plumbing -- JSONL loading, the predict -> evaluate
-> aggregate loop, and a compact printed/returned summary.

E0's design decision (docs/evaluation.md's "Planned evaluation
architecture" section) is what this implements: fixtures live in
eval_data/, not inside this package; results log to eval_results/ as a
small appended CSV, never a database; every suite defaults to a mocked/
offline mode, live is a separate, later, explicitly opt-in mode this
chunk does not implement at all.

Example/evaluator shape borrowed from the mentor repo studied during E0
(github.com/cwijayasundara/document_intelligence_adv_v2's backend/evals),
adapted to this project: no LangSmith `Run`/`Example` SDK objects, no
Postgres persistence -- just plain dataclasses and a CSV append.
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
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
# generality this project avoids elsewhere).
_METADATA_KEYS = {"id", "tags", "source", "notes"}


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

    def summary_line(self) -> str:
        avg = f"{self.average_score:.3f}" if self.average_score is not None else "n/a"
        return (
            f"[eval] suite={self.suite} mode={self.mode} "
            f"total={self.total} passed={self.passed} failed={self.failed} average_score={avg}"
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
) -> SuiteResult:
    """Runs one suite end-to-end: load -> predict -> evaluate -> aggregate.

    An example "passes" when every evaluator's score for it is exactly
    1.0 (missing/None scores don't count against it -- an evaluator that
    declines to score a case, e.g. "no expected_relevant_urls given", is
    not a failure). `average_score` is the mean of every finite numeric
    score across every example/evaluator pair, or None if nothing
    produced a numeric score at all.
    """
    examples = load_examples(dataset_file, subset=subset, tags=tags)
    logger.info("suite=%s mode=%s loaded %d examples from %s", suite, mode, len(examples), dataset_file)

    passed = 0
    failed = 0
    all_scores: list[float] = []
    per_example: list[dict[str, Any]] = []

    for example in examples:
        try:
            prediction = predict(example)
        except Exception as exc:  # noqa: BLE001 -- record the failure, keep going, same posture the mentor repo's runner uses.
            logger.exception("predict() failed for example %s", example.id)
            prediction = {"error": str(exc)}

        scores: dict[str, Any] = {}
        example_passed = True
        for name, evaluator in evaluators:
            try:
                result = evaluator(prediction, example.outputs)
            except Exception as exc:  # noqa: BLE001 -- one evaluator must not kill the whole run.
                logger.exception("evaluator=%s example=%s failed", name, example.id)
                result = {"key": name, "score": None, "comment": f"error: {exc}"}
            key = result.get("key", name)
            scores[key] = result
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
        per_example.append({"example_id": example.id, "prediction": prediction, "scores": scores})

    average_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else None
    return SuiteResult(
        suite=suite, mode=mode, total=len(examples), passed=passed, failed=failed,
        average_score=average_score, per_example=per_example,
    )


_CSV_FIELDS = ["run_id", "date", "git_commit", "suite", "mode", "total", "passed", "failed", "average_score", "tags", "note"]


def append_result_csv(
    result: SuiteResult, csv_path: Path, tags: list[str] | None = None, note: str = "",
) -> None:
    """Appends one row to `csv_path` (creating it with a header if it
    doesn't exist yet) -- the exact append-only convention `docs/
    evaluation.md`'s artifact policy already establishes for
    `retrieval_history.csv`/`history.csv`. `csv_path` is a parameter
    (not hardcoded to eval_results/), specifically so tests can point it
    at a tmp_path instead of ever touching a real tracked file.
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
            "note": note,
        })
