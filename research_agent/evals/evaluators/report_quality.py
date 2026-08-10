"""Fixture-agreement evaluator for the report_quality suite (R6B) --
compares which stable hard-failure identifiers a prediction detected
(`prediction["hard_failures"]`, produced by
runners/run_report_quality.py's own independent deterministic checks)
against a fixture's `expected_hard_failures`. Purely a set comparison,
no qualitative judgment of its own and no network/API call -- the same
"no network call, only compares data" posture evaluators/relevance.py
already established for chat_relevance.

R6A decision 2/R6B scope: `expected_dimension_labels` is deliberately
NEVER read here -- those are synthetic fixture expectations reserved
for R6C's independent live judges to be scored against later; scoring
them in this deterministic suite would blur exactly the R4-independent
distinction R6 exists to enforce (see specs/
report-quality-evaluation-plan.md section 0, decision 5).

Distinguishes two different kinds of "pass" that are easy to conflate:
- `structural_integrity.status` (inside the prediction) says whether
  the REPORT ITSELF is structurally clean.
- This evaluator's own `score` says whether the harness correctly
  DETECTED that state against the fixture's own expectation -- a
  deliberately broken fixture (structural_integrity.status="fail")
  still scores 1.0 here when every expected hard-failure identifier
  was found and nothing extra was reported.
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


ALL_EVALUATORS: dict[str, Any] = {
    "report_quality_hard_failure_agreement": report_quality_hard_failure_agreement,
}
