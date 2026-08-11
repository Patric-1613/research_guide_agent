"""Runner for the report_refinement suite -- R6D.2's deterministic/mock
pair evaluation, over R6D.1's own frozen pair fixtures.

**Reuses R6B's own deterministic checks directly, never a second
interpretation of them.** Each side of a pair (`draft_report`,
`refined_report`) is run through `research_agent.evals.runners.
run_report_quality.predict()` -- the exact same function R6B's own
single-report suite calls -- by wrapping it in a throwaway `Example`.
This module contains no hard-failure-detection logic of its own at
all; `_side_prediction` below is a thin adapter, not a reimplementation.

**No R6C live judges are called anywhere in this module.** A mock
prediction's `dimension_directions` is always `None` and
`semantic_evaluation_status` is always `"not_evaluated_in_mock_mode"`
-- R6D.2 measures only the deterministic `hard_failure_direction`
between the two sides of a pair; the 7 R6C semantic dimensions
(citation_correctness, groundedness, synthesis_quality, analytical_
quality, template_fit, coherence, source_balance) are never fabricated
from informational signals (word counts, citation density, source
coverage) and a fixture's own `expected.dimension_directions` is never
copied into a prediction -- that would make `report_refinement_
semantic_dimensions_not_evaluated` (evaluators/report_refinement.py)
tautological instead of honest about what mock mode does not measure.

R6A decision 1 applies here too, one level removed: this module never
imports `research_agent.report` and never calls `generate_report`/
`evaluate_report`/`revise_report` -- R6D measures whether R4's
existing "refine once" step already changed a report's structural
state, using fixtures that already contain both a draft and a refined
body; it does not itself perform, request, or simulate a refinement.

`--mode live` is intentionally not implemented in this phase (R6D.3's
job: live paired semantic judging) -- selecting it raises
`LiveModeSetupError` before any example is loaded, so `cli.py`'s own
`cmd_run` catches it and exits 2 with no traceback, no CSV row, and no
detail JSON, the same "setup failure, not a per-example failure"
contract every other suite's live mode already establishes.
"""

from __future__ import annotations

from typing import Any

from research_agent.evals import report_refinement_inputs as rri
from research_agent.evals.evaluators.report_refinement import ALL_EVALUATORS
from research_agent.evals.runners import run_report_quality as rq
from research_agent.evals.runners._base import Example, LiveModeSetupError, SuiteResult, run_suite

SUITE = "report_refinement"

_SIDE_EXAMPLE_ID = "__report_refinement_side__"


def _side_prediction(
    report: dict[str, Any], selected_papers: list[dict[str, Any]], approved_web_articles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Runs R6B's own deterministic hard-failure/informational-signal
    checks against ONE report body by calling `run_report_quality.
    predict()` directly, wrapped in a throwaway `Example` -- the exact
    same function, the exact same 6 hard-failure identifiers, never
    reimplemented or reinterpreted here. Only the 3 fields R6D.2 itself
    needs are carried forward; the rest of `predict()`'s own return
    shape (schema_version, judge_dimensions=None, judge_metadata=None,
    warnings, not_applicable=[]) is R6C's own concern, not R6D's."""
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
    """Set-based, not count-based -- a report that fixes 3 old defects
    while introducing 1 brand-new one has strictly FEWER hard failures
    by raw count, but is not defensibly "improved": it traded one class
    of problem for another. `mixed` is checked first and wins whenever
    BOTH sides have at least one identifier the other lacks (R6D.2's
    own explicit requirement: never silently collapsed into
    `unchanged`, and -- by this same set-subset reasoning -- never
    silently collapsed into `improved`/`regressed` by raw count either).
    Only when one side's failure set is a clean superset/subset of the
    other's does a directional verdict apply:
      - `improved`  : refined_set is a STRICT SUBSET of draft_set
                       (draft_only non-empty, refined_only empty)
      - `regressed` : draft_set is a STRICT SUBSET of refined_set
                       (refined_only non-empty, draft_only empty)
      - `unchanged` : the two sets are identical (both empty)
      - `mixed`     : neither is a subset of the other (both non-empty)
    None of R6D.1's 7 frozen fixtures currently exercise `mixed` for
    hard_failure_direction (R6D.1's own fixtures were not modified to
    add one) -- this function still supports it defensively, per R6D.2's
    own explicit requirement.
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
    """Deterministic-only (R6D.2) -- never calls OpenAI, never
    generates or refines anything, never scores a semantic dimension.
    Evaluates both sides of the pair independently through R6B's own
    checks, derives `hard_failure_direction` from the two resulting
    hard-failure sets, and marks the 7 R6C semantic dimensions
    unmeasured -- `dimension_directions` is always `None` here, never
    a fixture's own `expected.dimension_directions` copied across."""
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
    """`mode="mock"` (the default): deterministic, offline, free, calls
    no network path at all. `mode="live"` is not implemented in this
    phase -- raises `LiveModeSetupError` before any example is loaded
    or any CSV/detail artifact is written, exactly like R6B's own
    report_quality suite did before R6C.2 existed."""
    if mode == "mock":
        run_predict = predict
    elif mode == "live":
        raise LiveModeSetupError(
            "report_refinement --mode live is not implemented until R6D.3 (live paired semantic judging) -- "
            "use --mode mock for deterministic hard-failure-direction comparison only"
        )
    else:
        raise ValueError(f"unknown report_refinement mode {mode!r} -- expected 'mock' or 'live'")

    evaluators = [
        ("report_refinement_hard_failure_direction_agreement", ALL_EVALUATORS["report_refinement_hard_failure_direction_agreement"]),
        ("report_refinement_semantic_dimensions_not_evaluated", ALL_EVALUATORS["report_refinement_semantic_dimensions_not_evaluated"]),
    ]

    examples = _load_examples(tags=tags, subset=subset)
    return run_suite(
        suite=SUITE, dataset_file=str(rri.MANIFEST_PATH), predict=run_predict, evaluators=evaluators,
        mode=mode, examples=examples,
    )
