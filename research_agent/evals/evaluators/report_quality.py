"""Fixture-agreement evaluators for the report_quality suite.

`report_quality_hard_failure_agreement` (R6B) compares which stable
hard-failure identifiers a prediction detected
(`prediction["hard_failures"]`, produced by
runners/run_report_quality.py's own independent deterministic checks)
against a fixture's `expected_hard_failures`. Purely a set comparison,
no qualitative judgment of its own and no network/API call -- the same
"no network call, only compares data" posture evaluators/relevance.py
already established for chat_relevance.

Distinguishes two different kinds of "pass" that are easy to conflate:
- `structural_integrity.status` (inside the prediction) says whether
  the REPORT ITSELF is structurally clean.
- This evaluator's own `score` says whether the harness correctly
  DETECTED that state against the fixture's own expectation -- a
  deliberately broken fixture (structural_integrity.status="fail")
  still scores 1.0 here when every expected hard-failure identifier
  was found and nothing extra was reported.

`report_quality_dimension_agreement` (R6C.2) compares
`prediction["judge_dimensions"]` (populated only in live mode; still
`None` in every R6B mock prediction, unchanged) against a fixture's
`expected_dimension_labels` across all 7 frozen R6 dimensions. Purely a
CATEGORICAL label comparison -- continuous `score`/`confidence` values
anywhere in a judge's own output are never read here, per the R6A/R6B
decision that continuous scores stay informational only and never
control pass/fail. In mock mode, `judge_dimensions` is always `None`,
so this evaluator always returns `score=None` there -- a `None` score
never counts toward `run_suite`'s pass/fail or average, which is
exactly what keeps R6B's own mock behavior byte-for-byte unchanged now
that this second evaluator is registered alongside the first one.
"""

from __future__ import annotations

from typing import Any


def report_quality_hard_failure_agreement(prediction: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Score 1.0 when `prediction["hard_failures"]` (a list of stable
    identifiers, see run_report_quality.py's CANONICAL_HARD_FAILURE_
    ORDER) is EXACTLY the same set as `expected["expected_hard_
    failures"]` -- score 0.0 otherwise, never a partial credit score,
    since a hard-failure identifier is either correctly detected or
    it isn't. `comment` names exactly which identifiers were missing
    (expected but not detected) and which were unexpected (detected
    but not expected), so a failing case is debuggable without needing
    the full prediction dict.
    """
    expected_hard_failures = expected.get("expected_hard_failures")
    if expected_hard_failures is None:
        return {
            "key": "report_quality_hard_failure_agreement", "score": None,
            "comment": "no expected_hard_failures given",
        }

    if "error" in prediction:
        return {
            "key": "report_quality_hard_failure_agreement", "score": 0.0,
            "comment": f"predict() raised: {prediction['error']}",
        }

    actual = set(prediction.get("hard_failures") or [])
    expected_set = set(expected_hard_failures)
    missing = sorted(expected_set - actual)
    unexpected = sorted(actual - expected_set)

    score = 1.0 if actual == expected_set else 0.0
    return {
        "key": "report_quality_hard_failure_agreement",
        "score": score,
        "comment": f"missing={missing} unexpected={unexpected}",
    }


REQUIRED_DIMENSION_NAMES = (
    "citation_correctness", "groundedness", "synthesis_quality", "analytical_quality",
    "template_fit", "coherence", "source_balance",
)


def report_quality_dimension_agreement(prediction: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Score 1.0 only when EVERY one of the 7 frozen dimensions'
    predicted `label` exactly matches the fixture's own expected label
    -- score 0.0 if even one disagrees, the same all-or-nothing
    philosophy `report_quality_hard_failure_agreement` already uses
    (not a fractional/partial-credit score). `comment` names exactly
    which dimensions mismatched; the full per-dimension expected/
    actual/match breakdown also rides along under `detail` for the
    run's detail JSON.

    Returns `score=None` (not 0.0) whenever there is nothing real to
    compare yet: no `expected_dimension_labels` given, or
    `judge_dimensions` is `None` (mock mode, or a live run that hasn't
    reached this example) -- a `None` score never counts against
    pass/fail or the run's average, matching every other "no
    expectation -> no score" evaluator in this project.

    Deliberately does NOT alter or second-guess a fixture's own
    expected labels to make a live run pass -- a live judge
    disagreeing with a synthetic expectation is evaluation evidence
    about the judge (or, possibly, about a stale expectation), not
    something this function is allowed to paper over.
    """
    expected_labels = expected.get("expected_dimension_labels")
    if not expected_labels:
        return {
            "key": "report_quality_dimension_agreement", "score": None,
            "comment": "no expected_dimension_labels given",
        }

    if "error" in prediction:
        return {
            "key": "report_quality_dimension_agreement", "score": 0.0,
            "comment": f"predict() raised: {prediction['error']}",
        }

    judge_dimensions = prediction.get("judge_dimensions")
    if judge_dimensions is None:
        return {
            "key": "report_quality_dimension_agreement", "score": None,
            "comment": "no live judge_dimensions to compare (mock mode)",
        }

    detail: dict[str, dict[str, Any]] = {}
    mismatches: list[str] = []
    for dim in REQUIRED_DIMENSION_NAMES:
        actual_label = (judge_dimensions.get(dim) or {}).get("label")
        expected_label = (expected_labels.get(dim) or {}).get("label")
        match = actual_label == expected_label
        detail[dim] = {"expected": expected_label, "actual": actual_label, "match": match}
        if not match:
            mismatches.append(dim)

    score = 1.0 if not mismatches else 0.0
    return {
        "key": "report_quality_dimension_agreement",
        "score": score,
        "comment": f"mismatches={mismatches}",
        "detail": detail,
    }


ALL_EVALUATORS: dict[str, Any] = {
    "report_quality_hard_failure_agreement": report_quality_hard_failure_agreement,
    "report_quality_dimension_agreement": report_quality_dimension_agreement,
}
