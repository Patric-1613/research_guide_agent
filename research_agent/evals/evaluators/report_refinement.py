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
every run's own `evaluator_results`, that the 7 R6C semantic
dimensions (citation_correctness, groundedness, synthesis_quality,
analytical_quality, template_fit, coherence, source_balance) were not
measured by this mock-mode prediction. Mirrors `report_quality.py`'s
own `report_quality_dimension_agreement` precedent exactly: that
evaluator also always returns `score=None` in mock mode, for the same
reason (nothing was actually judged yet) -- a `None` score never
counts toward `run_suite`'s pass/fail or average, so registering this
evaluator alongside the direction-agreement one changes nothing about
which fixtures pass, it only documents, in the run's own data, that
semantic quality was never claimed to be measured. Live paired
semantic judging is R6D.3's job, not this phase's.
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


ALL_EVALUATORS: dict[str, Any] = {
    "report_refinement_hard_failure_direction_agreement": report_refinement_hard_failure_direction_agreement,
    "report_refinement_semantic_dimensions_not_evaluated": report_refinement_semantic_dimensions_not_evaluated,
}
