"""Fixture-agreement evaluators for the report_refinement suite (R6D.2).

`report_refinement_hard_failure_direction_agreement` compares which
direction (`improved`/`unchanged`/`regressed`/`mixed`) `run_report_
refinement.py`'s own deterministic pair prediction derived
(`prediction["hard_failure_direction"]`, built purely from R6B's own
`run_report_quality.predict()` called once per side -- never a second
interpretation of the 6 hard-failure identifiers) against a fixture's
`expected_hard_failure_direction`. Purely a categorical-label
comparison, no qualitative judgment of its own and no network/API
call -- same "no network call, only compares data" posture every
other evaluator in this project already establishes.

`report_refinement_semantic_dimensions_not_evaluated` NEVER produces a
real score -- it exists purely to make it structurally visible, in
every MOCK-mode run's own `evaluator_results`, that the 7 R6C semantic
dimensions (citation_correctness, groundedness, synthesis_quality,
analytical_quality, template_fit, coherence, source_balance) were not
measured by this mock-mode prediction. Registered only in mock mode
(`run_report_refinement.py::run_experiment`) -- R6D.2's own mock
behavior stays exactly byte-compatible.

`report_refinement_semantic_direction_agreement` (R6D.3) is the live-
mode counterpart: compares `prediction["dimension_directions"]`
(populated only in live mode by `run_report_refinement.py::predict_
live`) against a fixture's own `expected_dimension_directions`, one of
the 7 R6C dimensions at a time. `score = matched / 7` -- a genuine
fractional score, not the binary 1.0/0.0 the hard-failure-direction
evaluator uses, per R6D.3's own explicit requirement. Because
`run_suite`'s own pass/fail rule requires every evaluator's score to
be exactly `1.0` for an example to count as passed, this automatically
means **exact 7/7 direction agreement is required for a fully-passed
example** -- no extra code needed to enforce that, it falls out of the
existing shared aggregation rule the same way `report_quality_
dimension_agreement`'s binary score already does for `report_quality`.
`"unknown"` is compared with plain equality, exactly like every other
direction value -- it is never treated as a wildcard that matches
anything. **This score describes expectation agreement with a
synthetic, hand-constructed fixture -- it is not a measurement of
report quality, and the CLI's own aggregate `average_score` must never
be read as one.** Registered only in live mode.
"""

from __future__ import annotations

from typing import Any

REQUIRED_DIMENSION_NAMES = (
    "citation_correctness", "groundedness", "synthesis_quality", "analytical_quality",
    "template_fit", "coherence", "source_balance",
)


def report_refinement_hard_failure_direction_agreement(
    prediction: dict[str, Any], expected: dict[str, Any],
) -> dict[str, Any]:
    """Score 1.0 when `prediction["hard_failure_direction"]` exactly
    matches `expected["expected_hard_failure_direction"]`, 0.0
    otherwise -- never a partial-credit score, the same all-or-nothing
    philosophy `report_quality_hard_failure_agreement` already uses.
    This is the ONLY thing R6D.2's mock mode actually measures; see
    `report_refinement_semantic_dimensions_not_evaluated` below for why
    nothing about `expected_dimension_directions` is scored here.
    """
    expected_direction = expected.get("expected_hard_failure_direction")
    if expected_direction is None:
        return {
            "key": "report_refinement_hard_failure_direction_agreement", "score": None,
            "comment": "no expected_hard_failure_direction given",
        }

    if "error" in prediction:
        return {
            "key": "report_refinement_hard_failure_direction_agreement", "score": 0.0,
            "comment": f"predict() raised: {prediction['error']}",
        }

    actual_direction = prediction.get("hard_failure_direction")
    score = 1.0 if actual_direction == expected_direction else 0.0
    return {
        "key": "report_refinement_hard_failure_direction_agreement",
        "score": score,
        "comment": f"expected={expected_direction!r} actual={actual_direction!r}",
    }


def report_refinement_semantic_dimensions_not_evaluated(
    prediction: dict[str, Any], expected: dict[str, Any],
) -> dict[str, Any]:
    """Always returns `score=None` in mock mode -- deliberately never
    compares `prediction["dimension_directions"]` (always `None` in a
    mock prediction) against `expected["expected_dimension_directions"]`.
    Doing so would be tautological in the wrong direction if a future
    change ever copied expected data into the prediction, and
    meaningless as written today since mock mode has no semantic
    judgment to compare in the first place. `detail` lists which
    dimensions remain unmeasured, purely for transparency in the run's
    own detail JSON -- never scored.
    """
    return {
        "key": "report_refinement_semantic_dimensions_not_evaluated",
        "score": None,
        "comment": "semantic dimensions require live paired judging (R6D.3) -- not evaluated in mock mode",
        "detail": {"unevaluated_dimensions": list(REQUIRED_DIMENSION_NAMES)},
    }


def report_refinement_semantic_direction_agreement(
    prediction: dict[str, Any], expected: dict[str, Any],
) -> dict[str, Any]:
    """Score `matched / 7` comparing each of the 7 R6C dimensions'
    predicted direction against `expected["expected_dimension_
    directions"]`'s own direction for that dimension -- exact equality
    only, `"unknown"` included (never a wildcard). Returns `score=None`
    (not 0.0) when there is nothing real to compare: no expected
    directions given, or `prediction["dimension_directions"]` is `None`
    (mock mode) -- a `None` score never counts against pass/fail or the
    run's average, matching every other "no expectation -> no score"
    evaluator in this project. `detail` carries the full per-dimension
    expected/actual/match breakdown for the run's own detail JSON.
    """
    expected_directions = expected.get("expected_dimension_directions")
    if not expected_directions:
        return {
            "key": "report_refinement_semantic_direction_agreement", "score": None,
            "comment": "no expected_dimension_directions given",
        }

    if "error" in prediction:
        return {
            "key": "report_refinement_semantic_direction_agreement", "score": 0.0,
            "comment": f"predict() raised: {prediction['error']}",
        }

    predicted_directions = prediction.get("dimension_directions")
    if predicted_directions is None:
        return {
            "key": "report_refinement_semantic_direction_agreement", "score": None,
            "comment": "no live dimension_directions to compare (mock mode)",
        }

    detail: dict[str, dict[str, Any]] = {}
    matches = 0
    for dim in REQUIRED_DIMENSION_NAMES:
        expected_entry = expected_directions.get(dim) or {}
        expected_direction = expected_entry.get("direction")
        actual_direction = predicted_directions.get(dim)
        match = actual_direction == expected_direction
        detail[dim] = {"expected": expected_direction, "actual": actual_direction, "match": match}
        if match:
            matches += 1

    score = round(matches / len(REQUIRED_DIMENSION_NAMES), 4)
    return {
        "key": "report_refinement_semantic_direction_agreement",
        "score": score,
        "comment": f"{matches}/{len(REQUIRED_DIMENSION_NAMES)} dimensions matched expected direction "
                   "(expectation agreement, not a report-quality measurement)",
        "detail": detail,
    }


ALL_EVALUATORS: dict[str, Any] = {
    "report_refinement_hard_failure_direction_agreement": report_refinement_hard_failure_direction_agreement,
    "report_refinement_semantic_dimensions_not_evaluated": report_refinement_semantic_dimensions_not_evaluated,
    "report_refinement_semantic_direction_agreement": report_refinement_semantic_direction_agreement,
}
