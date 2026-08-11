"""R6D.4d Part 3: the `report_refinement_real` suite -- evaluates
exactly the 3 frozen real R4 capture pairs (R6D.4c) against their
already-frozen human/deterministic adjudications (R6D.4d Parts 1-2).

Reuses `run_report_refinement.py`'s own `predict`/`predict_live` and
`evaluators/report_refinement.py`'s own evaluators DIRECTLY -- this
module contains no claim-comparison, pairwise-holistic-judging,
direction-aggregation, or hard-failure logic of its own. The only new
code in this suite is `report_refinement_real_inputs.py`'s own
capture+adjudication loader; see that module for the load-time
validation this suite's own `run_experiment` depends on.

Separate history/detail-JSON files from the synthetic `report_
refinement` suite by design (`report_refinement_real_history.csv`,
`report_refinement_real_run_<id>.json`, both wired up generically via
`cli.py`'s own `SUITES["report_refinement_real"]` entry) -- three real,
individually adjudicated pairs are a fundamentally different kind of
evidence than seven hand-authored synthetic fixtures with authored
answer keys, and must never be silently pooled into the same running
log.

**The aggregate score this suite reports is EXACT AGREEMENT with the
frozen adjudication labels above -- never a report-quality score,
never a refinement-benefit score, and never evidence by itself that
production R4 refinement helps.** Three pairs is far too small a
sample for that claim regardless of what this suite's own score comes
out to; see `evaluators/report_refinement.py`'s own
`report_refinement_semantic_direction_agreement` docstring, which
already carries the identical caveat for the synthetic suite and
applies here unchanged, unmodified, and unextended.
"""

from __future__ import annotations

from research_agent.evals import report_refinement_real_inputs as rrri
from research_agent.evals.evaluators.report_refinement import ALL_EVALUATORS
from research_agent.evals.runners import run_report_quality as rq
from research_agent.evals.runners import run_report_refinement as rrr
from research_agent.evals.runners._base import LiveModeSetupError, SuiteResult, run_suite

SUITE = "report_refinement_real"

_DATASET_LABEL = "eval_results/captures/real-*.json + eval_data/report_refinement/real_reviews/"


def run_experiment(mode: str = "mock", subset: int | None = None, tags: list[str] | None = None) -> SuiteResult:
    """`subset`/`tags` are accepted only for CLI-shape compatibility
    with every other suite's own `run_experiment` -- both are no-ops
    here. This suite is always exactly the 3 frozen pairs `report_
    refinement_real_inputs.PAIR_IDS` names, in that fixed order; there
    is no larger pool to subset or tag-filter.

    Examples are loaded and validated FIRST, unconditionally, in both
    modes -- `report_refinement_real_inputs.load_real_refinement_
    examples` rejects a missing file, a hash mismatch, an unknown
    dimension/direction, or embedded forbidden content immediately.
    Only if that succeeds does `mode="live"` go on to construct a real
    OpenAI client (`run_report_quality._build_live_client()`, the same
    convention every other live suite already uses) -- so a corrupted
    local capture/adjudication file is caught before credentials are
    even touched, and a missing-credentials failure still exits
    cleanly via `LiveModeSetupError` exactly as it already does for
    `report_refinement`'s own live mode.

    `mode="mock"`: deterministic, offline, zero OpenAI calls -- reuses
    `run_report_refinement.predict` unchanged; `dimension_directions`
    stays `None` on every prediction, exactly as it already does for
    the synthetic suite's own mock mode.

    `mode="live"`: reuses `run_report_refinement.predict_live`
    unchanged. Under current pair content, at most 5 real judge calls
    total across the 3 pairs: real-foundational-01 and real-expert-01
    are both byte-identical (revision_applied=false) pairs -- 1 claim/
    source call each, reused for the refined side, no pairwise holistic
    call. real-analytical-01 is the one changed pair -- 2 claim/source
    calls (draft + refined) + 1 pairwise holistic call. Never a 6th
    call, and never another overall/pairwise winner call.
    """
    try:
        examples = rrri.load_real_refinement_examples()
    except rrri.RealRefinementLoadError as exc:
        raise LiveModeSetupError(str(exc)) from exc

    if mode == "mock":
        run_predict = rrr.predict
        evaluators = [
            ("report_refinement_hard_failure_direction_agreement", ALL_EVALUATORS["report_refinement_hard_failure_direction_agreement"]),
            ("report_refinement_semantic_dimensions_not_evaluated", ALL_EVALUATORS["report_refinement_semantic_dimensions_not_evaluated"]),
        ]
    elif mode == "live":
        client = rq._build_live_client()
        run_predict = lambda example: rrr.predict_live(example, client)  # noqa: E731
        evaluators = [
            ("report_refinement_hard_failure_direction_agreement", ALL_EVALUATORS["report_refinement_hard_failure_direction_agreement"]),
            ("report_refinement_semantic_direction_agreement", ALL_EVALUATORS["report_refinement_semantic_direction_agreement"]),
        ]
    else:
        raise ValueError(f"unknown report_refinement_real mode {mode!r} -- expected 'mock' or 'live'")

    return run_suite(
        suite=SUITE, dataset_file=_DATASET_LABEL, predict=run_predict, evaluators=evaluators,
        mode=mode, examples=examples,
    )
