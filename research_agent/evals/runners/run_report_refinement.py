"""Runner for the report_refinement suite -- R6D.2's deterministic/mock
pair evaluation (unchanged), plus R6D.3's opt-in live semantic pair
evaluation, over R6D.1's own frozen pair fixtures.

**Reuses R6B's deterministic checks AND R6C's live judge path directly,
never a second interpretation of either.** Each side of a pair
(`draft_report`, `refined_report`) is run through `research_agent.
evals.runners.run_report_quality.predict()` (mock) or `...predict_live()`
(live) -- the exact same functions R6B/R6C's own single-report suite
calls -- by wrapping it in a throwaway `Example`. This module contains
no hard-failure-detection logic, no claim-extraction/evidence-registry/
injection-sanitization logic, no judge-calling logic, and no dimension-
aggregation logic of its own; `_side_prediction`/`_side_prediction_live`
below are thin adapters, never reimplementations. R6C's own claim/
source and holistic judges, `claim_source.py`/`holistic.py`, are never
imported or called directly here -- only through `predict_live`.

**No third, pairwise LLM judge exists anywhere in this module.** Each
side is judged completely independently by R6C's existing single-
report judges; R6D.3 only *compares* the two already-independent
results after the fact. Maximum judge-call cost per non-identical
eligible pair: 4 (one claim/source + one holistic, per side) -- never
5, never a call that sees both reports at once.

**R6D.2 (mock) is byte-compatible and unchanged**: `predict()`,
`_side_prediction`, `_hard_failure_direction`, and the mock-mode
evaluator set are exactly as R6D.2 left them. A mock prediction's
`dimension_directions` is always `None` and `semantic_evaluation_
status` is always `"not_evaluated_in_mock_mode"` -- unchanged.

**R6D.3 (live) derives per-dimension direction from R6C's own
categorical labels and continuous scores, never from informational
signals** (word counts, citation density, source coverage) and never
by copying a fixture's own `expected.dimension_directions` into a
prediction. Holistic-dimension direction uses a single, explicitly
PROVISIONAL, uncalibrated delta threshold
(`HOLISTIC_DIRECTION_MIN_DELTA = 0.10`) -- see that constant's own
docstring. `citation_correctness`/`groundedness` direction never uses
a score at all: those two dimensions currently have a categorical
aggregated outcome (R6C.2c), not a calibrated continuous one, and this
module does not invent one from verdict counts.

**Identical-pair optimization**: when a fixture declares
`revision_applied=false` AND `draft_report == refined_report` exactly
(`report_refinement_inputs.reports_are_equal`, deep equality -- never
"same length" or "same references"), the draft side is evaluated once
and the completed result is deep-copied for the refined side --
`identical_input_reused=True` is recorded, and every comparable
dimension direction naturally comes out `unchanged` (identical inputs
always produce identical categorical labels/scores). This is both a
cost control (zero extra paid calls for a pair that never should have
needed any) and a stability guarantee (no risk of same-input judge
variability making an intentional non-revision look like a spurious
direction).

R6A decision 1 applies here too, one level removed: this module never
imports `research_agent.report` and never calls `generate_report`/
`evaluate_report`/`revise_report` -- R6D measures whether R4's
existing "refine once" step already changed a report's structural AND
(as of R6D.3) semantic state, using fixtures that already contain both
a draft and a refined body; it does not itself perform, request, or
simulate a refinement, and it does not alter the production R4
refinement loop in any way.
"""

from __future__ import annotations

import copy
from typing import Any

from openai import OpenAI

from research_agent.evals import report_refinement_inputs as rri
from research_agent.evals.evaluators.report_refinement import ALL_EVALUATORS
from research_agent.evals.runners import run_report_quality as rq
from research_agent.evals.runners._base import Example, LiveModeSetupError, SuiteResult, run_suite

SUITE = "report_refinement"

_SIDE_EXAMPLE_ID = "__report_refinement_side__"

# R6D.3: PROVISIONAL, uncalibrated. A holistic dimension (synthesis_
# quality/analytical_quality/template_fit/coherence/source_balance)
# only moves direction when both sides expose a valid numeric score AND
# their labels are equal (a label transition already resolves direction
# via rule D, before this threshold is ever consulted) AND the score
# delta clears this margin. Chosen as a round, conservative starting
# point -- NOT derived from any calibration study, NOT statistically
# validated, and not claimed to be either. Revisit only after R6D.4's
# real-report evidence (or a dedicated calibration phase) exists, the
# same "no invented weights without evidence" posture R6A/R6C.2c both
# already established for this project's other thresholds.
HOLISTIC_DIRECTION_MIN_DELTA = 0.10

_CLAIM_SOURCE_DIMENSION_NAMES = ("citation_correctness", "groundedness")
_HOLISTIC_DIMENSION_NAMES = ("synthesis_quality", "analytical_quality", "template_fit", "coherence", "source_balance")


def _side_prediction(
    report: dict[str, Any], selected_papers: list[dict[str, Any]], approved_web_articles: list[dict[str, Any]],
) -> dict[str, Any]:
    """R6D.2, unchanged. Runs R6B's own deterministic hard-failure/
    informational-signal checks against ONE report body by calling
    `run_report_quality.predict()` directly, wrapped in a throwaway
    `Example` -- the exact same function, the exact same 6 hard-failure
    identifiers, never reimplemented or reinterpreted here."""
    side_example = Example(
        id=_SIDE_EXAMPLE_ID,
        inputs={
            "generated_report": report,
            "selected_papers": selected_papers,
            "approved_web_articles": approved_web_articles,
        },
        outputs={}, metadata={},
    )
    side_result = rq.predict(side_example)
    return {
        "hard_failures": side_result["hard_failures"],
        "structural_status": side_result["structural_integrity"]["status"],
        "informational_signals": side_result["informational_signals"],
    }


def _hard_failure_direction(draft_hard_failures: list[str], refined_hard_failures: list[str]) -> str:
    """R6D.2, unchanged. Set-based, not count-based -- a report that
    fixes 3 old defects while introducing 1 brand-new one has strictly
    FEWER hard failures by raw count, but is not defensibly "improved":
    it traded one class of problem for another.
      - `improved`  : refined_set is a STRICT SUBSET of draft_set
      - `regressed` : draft_set is a STRICT SUBSET of refined_set
      - `unchanged` : the two sets are identical
      - `mixed`     : neither is a subset of the other
    """
    draft_set = set(draft_hard_failures)
    refined_set = set(refined_hard_failures)
    draft_only = draft_set - refined_set
    refined_only = refined_set - draft_set

    if draft_only and refined_only:
        return "mixed"
    if refined_only:
        return "regressed"
    if draft_only:
        return "improved"
    return "unchanged"


def predict(example: Example) -> dict[str, Any]:
    """R6D.2, unchanged (byte-compatible). Deterministic-only -- never
    calls OpenAI, never generates or refines anything, never scores a
    semantic dimension. `dimension_directions` is always `None`."""
    selected_papers = example.inputs["selected_papers"]
    approved_web_articles = example.inputs["approved_web_articles"]

    draft = _side_prediction(example.inputs["draft_report"], selected_papers, approved_web_articles)
    refined = _side_prediction(example.inputs["refined_report"], selected_papers, approved_web_articles)

    return {
        "pair_id": example.id,
        "draft": draft,
        "refined": refined,
        "hard_failure_direction": _hard_failure_direction(draft["hard_failures"], refined["hard_failures"]),
        "dimension_directions": None,
        "semantic_evaluation_status": "not_evaluated_in_mock_mode",
    }


# --- R6D.3: live pair evaluation ------------------------------------------

def _side_prediction_live(
    report: dict[str, Any], selected_papers: list[dict[str, Any]], approved_web_articles: list[dict[str, Any]],
    topic: str, template: str, client: OpenAI,
) -> dict[str, Any]:
    """Runs R6C's full live prediction path against ONE report body by
    calling `run_report_quality.predict_live()` directly, wrapped in a
    throwaway `Example` -- the exact same function that runs claim
    extraction, evidence-registry construction, injection sanitization,
    both judges, hard-failure skip gating, and dimension aggregation
    for the `report_quality` suite itself. Returns `predict_live`'s
    full, unmodified return shape (`structural_integrity`,
    `informational_signals`, `judge_dimensions`, `judge_metadata`,
    `hard_failures`, ...) -- nothing here re-derives or reinterprets
    any of it."""
    side_example = Example(
        id=_SIDE_EXAMPLE_ID,
        inputs={
            "generated_report": report,
            "selected_papers": selected_papers,
            "approved_web_articles": approved_web_articles,
            "topic": topic,
            "template": template,
        },
        outputs={}, metadata={},
    )
    return rq.predict_live(side_example, client)


def _evaluate_side_live(
    report: dict[str, Any], selected_papers: list[dict[str, Any]], approved_web_articles: list[dict[str, Any]],
    topic: str, template: str, client: OpenAI,
) -> tuple[dict[str, Any] | None, str | None]:
    """Isolates one side's evaluation from the other's: `predict_live`
    itself never raises (both judges have their own "never raises"
    contract, degrading to a recorded `error` internally), but this
    wrapper still catches any unexpected exception so that one side
    failing can never prevent the OTHER side from being evaluated, or
    crash the whole example -- returns `(full_prediction, None)` on
    success, `(None, error_message)` on an unexpected failure."""
    try:
        return _side_prediction_live(report, selected_papers, approved_web_articles, topic, template, client), None
    except Exception as exc:  # noqa: BLE001 -- one side's crash must never take down the other side or the whole example.
        return None, str(exc)


def _extract_side_summary(full_prediction: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    """Builds this suite's own compact per-side shape from `predict_
    live`'s full return value. On the (expected-to-be-rare) case where
    `_evaluate_side_live` itself caught an unexpected exception,
    degrades every dimension to `unknown` with the error recorded,
    rather than crashing or fabricating a result -- the same "never
    invent a judgment, only unknown or a real one" posture R6C's own
    aggregation already uses."""
    if full_prediction is None:
        return {
            "hard_failures": [],
            "structural_status": "unknown",
            "informational_signals": {},
            "judge_dimensions": {
                dim: {"label": "unknown", "score": None, "reasons": [f"side evaluation failed: {error}"]}
                for dim in rri.REQUIRED_DIMENSION_NAMES
            },
            "judge_metadata": None,
            "error": error,
        }
    return {
        "hard_failures": full_prediction["hard_failures"],
        "structural_status": full_prediction["structural_integrity"]["status"],
        "informational_signals": full_prediction["informational_signals"],
        "judge_dimensions": full_prediction["judge_dimensions"],
        "judge_metadata": full_prediction["judge_metadata"],
        "error": error,
    }


def _is_valid_score(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _dimension_direction(dim_name: str, draft_entry: dict[str, Any], refined_entry: dict[str, Any]) -> str:
    """Rules A-G, applied in this exact order (see module docstring for
    A-C/D-G's own reasoning); never reads informational signals, never
    invents a continuous score for citation_correctness/groundedness."""
    draft_label = draft_entry.get("label")
    refined_label = refined_entry.get("label")

    # A: either side unknown -> unknown.
    if draft_label == "unknown" or refined_label == "unknown":
        return "unknown"
    # B: both not_applicable -> unchanged.
    if draft_label == "not_applicable" and refined_label == "not_applicable":
        return "unchanged"
    # C: exactly one not_applicable -> unknown (applicability changed; direction can't be inferred safely).
    if draft_label == "not_applicable" or refined_label == "not_applicable":
        return "unknown"
    # D: label transition (only pass/fail remain at this point).
    if draft_label == "fail" and refined_label == "pass":
        return "improved"
    if draft_label == "pass" and refined_label == "fail":
        return "regressed"
    # E: same label, citation_correctness/groundedness -- categorical only, never score-derived.
    if dim_name in _CLAIM_SOURCE_DIMENSION_NAMES:
        return "unchanged"
    # F/G: same label, holistic dimension -- compare scores against the provisional delta.
    draft_score = draft_entry.get("score")
    refined_score = refined_entry.get("score")
    if not _is_valid_score(draft_score) or not _is_valid_score(refined_score):
        return "unknown"
    delta = refined_score - draft_score
    if delta >= HOLISTIC_DIRECTION_MIN_DELTA:
        return "improved"
    if delta <= -HOLISTIC_DIRECTION_MIN_DELTA:
        return "regressed"
    return "unchanged"


def _semantic_evaluation_status(dimension_directions: dict[str, str]) -> str:
    """Derived directly from the 7 directions just computed -- never a
    separate ad-hoc judgment. `"evaluated"` only when all 7 dimensions
    produced a real (non-`unknown`) direction; `"not_evaluated"` only
    when every single one came back `unknown` (e.g. both sides
    structurally skipped); `"partially_evaluated"` otherwise."""
    directions = list(dimension_directions.values())
    if all(d == "unknown" for d in directions):
        return "not_evaluated"
    if any(d == "unknown" for d in directions):
        return "partially_evaluated"
    return "evaluated"


def predict_live(example: Example, client: OpenAI) -> dict[str, Any]:
    """R6D.3: evaluates both sides of a pair through R6C's full live
    prediction path independently, then compares. Never a third,
    pairwise LLM call -- both sides are judged completely on their own,
    exactly as `report_quality`'s own live mode already judges any
    single report; this function only diffs the two already-independent
    results afterward, entirely in Python.
    """
    selected_papers = example.inputs["selected_papers"]
    approved_web_articles = example.inputs["approved_web_articles"]
    topic = example.inputs["topic"]
    template = example.inputs["template"]
    draft_report = example.inputs["draft_report"]
    refined_report = example.inputs["refined_report"]
    refinement_context = example.inputs.get("refinement_context") or {}

    draft_full, draft_error = _evaluate_side_live(
        draft_report, selected_papers, approved_web_articles, topic, template, client,
    )

    identical_input_reused = False
    revision_applied = refinement_context.get("revision_applied")
    if draft_error is None and revision_applied is False and rri.reports_are_equal(draft_report, refined_report):
        # Exact report equality only -- never inferred from equal length/references. A fresh
        # deep copy so the two sides never share mutable state, even though the input was identical.
        refined_full = copy.deepcopy(draft_full)
        refined_error = None
        identical_input_reused = True
    else:
        refined_full, refined_error = _evaluate_side_live(
            refined_report, selected_papers, approved_web_articles, topic, template, client,
        )

    draft_side = _extract_side_summary(draft_full, draft_error)
    refined_side = _extract_side_summary(refined_full, refined_error)

    dimension_directions = {
        dim: _dimension_direction(
            dim,
            draft_side["judge_dimensions"].get(dim) or {"label": "unknown"},
            refined_side["judge_dimensions"].get(dim) or {"label": "unknown"},
        )
        for dim in rri.REQUIRED_DIMENSION_NAMES
    }

    return {
        "pair_id": example.id,
        "draft": draft_side,
        "refined": refined_side,
        "hard_failure_direction": _hard_failure_direction(draft_side["hard_failures"], refined_side["hard_failures"]),
        "dimension_directions": dimension_directions,
        "semantic_evaluation_status": _semantic_evaluation_status(dimension_directions),
        "identical_input_reused": identical_input_reused,
    }


# --- Example bridging (RefinementPairExample -> runners._base.Example) ---

def _to_example(pair: rri.RefinementPairExample) -> Example:
    """R6D.1's own loader returns `RefinementPairExample` objects (a
    richer, pair-specific shape); this bridges to the plain `Example`
    shape `run_suite`'s existing predict -> evaluate -> aggregate loop
    already knows how to run, the same "reuse run_suite via a suite's
    own thin adapter" pattern `run_report_quality.py`'s own
    `load_report_quality_examples` already establishes for its
    manifest+fixture-file loader."""
    return Example(
        id=pair.id,
        inputs={
            "topic": pair.topic,
            "template": pair.template,
            "selected_papers": pair.selected_papers,
            "approved_web_articles": pair.approved_web_articles,
            "draft_report": pair.draft_report,
            "refined_report": pair.refined_report,
            "refinement_context": pair.refinement_context,
        },
        outputs={
            "expected_hard_failure_direction": pair.expected.get("hard_failure_direction"),
            "expected_dimension_directions": pair.expected.get("dimension_directions"),
        },
        metadata={"tags": pair.tags, "source_origin": pair.source_origin, "notes": pair.notes},
    )


def _load_examples(tags: list[str] | None, subset: int | None) -> list[Example]:
    pairs = rri.load_report_refinement_examples(tags=tags, subset=subset)
    return [_to_example(pair) for pair in pairs]


# --- run_experiment() -----------------------------------------------------

def run_experiment(mode: str = "mock", subset: int | None = None, tags: list[str] | None = None) -> SuiteResult:
    """`mode="mock"` (the default, R6D.2, unchanged): deterministic,
    offline, free, calls no network path at all.

    `mode="live"` (R6D.3): constructs a real OpenAI client up front via
    `run_report_quality._build_live_client()` -- the exact same
    convention `report_quality`'s own live mode uses (the same 2
    checks: `REPORT_QUALITY_JUDGE_MODEL` non-empty, `OpenAI()`
    construction succeeds), reused directly rather than reimplemented
    -- raising `LiveModeSetupError` (exit 2, no traceback, no CSV/
    detail side effects) before any example is loaded if setup fails.
    Never silently falls back to mock mode on any failure.
    """
    if mode == "mock":
        run_predict = predict
        evaluators = [
            ("report_refinement_hard_failure_direction_agreement", ALL_EVALUATORS["report_refinement_hard_failure_direction_agreement"]),
            ("report_refinement_semantic_dimensions_not_evaluated", ALL_EVALUATORS["report_refinement_semantic_dimensions_not_evaluated"]),
        ]
    elif mode == "live":
        client = rq._build_live_client()
        run_predict = lambda example: predict_live(example, client)  # noqa: E731
        evaluators = [
            ("report_refinement_hard_failure_direction_agreement", ALL_EVALUATORS["report_refinement_hard_failure_direction_agreement"]),
            ("report_refinement_semantic_direction_agreement", ALL_EVALUATORS["report_refinement_semantic_direction_agreement"]),
        ]
    else:
        raise ValueError(f"unknown report_refinement mode {mode!r} -- expected 'mock' or 'live'")

    examples = _load_examples(tags=tags, subset=subset)
    return run_suite(
        suite=SUITE, dataset_file=str(rri.MANIFEST_PATH), predict=run_predict, evaluators=evaluators,
        mode=mode, examples=examples,
    )
