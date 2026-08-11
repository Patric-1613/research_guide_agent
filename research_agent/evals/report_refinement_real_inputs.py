"""R6D.4d Part 3: loader + validation adapting the three frozen real R4
capture artifacts (R6D.4c, `eval_results/captures/real-*.json`) and
their already-frozen adjudications (R6D.4d Parts 1-2, `eval_data/
report_refinement/real_reviews/`) into the exact same `runners._base.
Example` shape `run_report_refinement.py`'s own `predict`/`predict_
live` already consume -- so this module reuses those functions
directly and contains no claim-comparison, pairwise-holistic-judging,
direction-aggregation, or hard-failure logic of its own.

Deliberately separate from `report_refinement_inputs.py`'s own frozen
`r6d1-v1` synthetic-fixture loader -- neither module imports the
other, same "independent copies, never cross-import validation
internals" posture `r6d4_capture.py`'s own module docstring already
established. A real capture's own "expected" directions live entirely
in a separate, already-committed adjudication file -- never inside the
capture artifact itself (which stays deliberately unlabelled per
R6D.4a's own schema) -- this loader is the one place that reunites
capture + adjudication into a single `Example`, and it does so
read-only: neither file is ever written to by this module.

**Never copies expected labels into a prediction.** The adjudication's
own `hard_failure_direction`/`dimension_directions` only ever end up in
`Example.outputs` (the "expected" side `run_suite`'s evaluators read
AFTER prediction) -- never in `Example.inputs`, which is the only
thing `predict`/`predict_live` ever read. Reviewer provenance is kept
in `Example.metadata` for the same reason: metadata is never read by
`predict`/`predict_live` either, so it can never reach a judge prompt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from research_agent.evals import r6d4_capture
from research_agent.evals.runners._base import EVAL_RESULTS_DIR, Example

# Deterministic, frozen order -- foundational, analytical, expert --
# never re-derived from directory-listing order (filesystem-dependent,
# not guaranteed stable).
PAIR_IDS = ("real-foundational-01", "real-analytical-01", "real-expert-01")

# Each pair_id's own frozen template assignment, cross-checked against
# the capture file's own `template` field at load time -- catches a
# capture/adjudication pairing mistake (wrong file for the wrong id)
# before it ever reaches prediction.
PAIR_ID_TEMPLATES = {
    "real-foundational-01": "foundational",
    "real-analytical-01": "analytical",
    "real-expert-01": "expert",
}

CAPTURES_DIR = EVAL_RESULTS_DIR / "captures"
REAL_REVIEWS_DIR = Path(__file__).resolve().parent.parent.parent / "eval_data" / "report_refinement" / "real_reviews"

# Independent copy of evaluators/report_refinement.py's own
# REQUIRED_DIMENSION_NAMES/VALID_DIRECTIONS (which is itself an
# independent copy of report_refinement_inputs.py's own constants) --
# same reasoning every other R6D module already gives: a future
# refactor of any one of those must never silently change what this
# loader accepts.
REQUIRED_DIMENSION_NAMES = (
    "citation_correctness", "groundedness", "synthesis_quality", "analytical_quality",
    "template_fit", "coherence", "source_balance",
)
VALID_DIRECTIONS = ("improved", "unchanged", "regressed", "unknown")

ADJUDICATION_SCHEMA_VERSION = "r6d4-adjudication-v1"
BLIND_ASSESSMENT_SCHEMA_VERSION = "r6d4-review-v1"


class RealRefinementLoadError(ValueError):
    """Raised for any load-time defect found while assembling the 3
    real-capture examples -- a missing capture/adjudication/blind-
    assessment file, a duplicate pair_id, a capture hash that no longer
    matches what its adjudication (directly, or via a blind-assessment
    chain) recorded, a malformed adjudication schema, an unknown
    dimension/direction, a template mismatch, or forbidden embedded
    content. Deliberately loud and immediate -- raised on the FIRST
    violation found, before any OpenAI client is ever constructed by
    this suite's own `run_experiment`."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, pair_id: str, kind: str) -> dict[str, Any]:
    if not path.exists():
        raise RealRefinementLoadError(f"{pair_id}: missing {kind} file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RealRefinementLoadError(f"{pair_id}: {kind} file is not valid JSON: {path}") from exc


def _resolve_expected_capture_sha256(adjudication: dict[str, Any], pair_id: str) -> str:
    """Two adjudication shapes exist by design (see `eval_data/report_
    refinement/real_reviews/`'s own files): a deterministic-equality
    adjudication (real-foundational-01, real-expert-01) records
    `capture_sha256` directly; a human-reviewed adjudication (real-
    analytical-01) instead records `blind_assessment_sha256`, because
    the capture's own hash was already bound once, at blind-review
    time, inside the blind-assessment file -- re-deriving it here means
    hopping through that file (verifying ITS OWN hash first) rather
    than duplicating a second `capture_sha256` field that would then
    need to independently stay in sync with the first."""
    if "capture_sha256" in adjudication:
        return adjudication["capture_sha256"]

    if "blind_assessment_sha256" in adjudication:
        blind_path = REAL_REVIEWS_DIR / f"{pair_id}-blind-assessment.json"
        blind_assessment = _load_json(blind_path, pair_id, "blind-assessment")
        actual_blind_hash = _sha256(blind_path)
        expected_blind_hash = adjudication["blind_assessment_sha256"]
        if actual_blind_hash != expected_blind_hash:
            raise RealRefinementLoadError(
                f"{pair_id}: blind-assessment file hash {actual_blind_hash!r} does not match the "
                f"adjudication's own recorded blind_assessment_sha256 {expected_blind_hash!r} -- the "
                "blind-assessment file may have changed since it was adjudicated"
            )
        if blind_assessment.get("schema_version") != BLIND_ASSESSMENT_SCHEMA_VERSION:
            raise RealRefinementLoadError(
                f"{pair_id}: blind-assessment schema_version must be {BLIND_ASSESSMENT_SCHEMA_VERSION!r}, "
                f"got {blind_assessment.get('schema_version')!r}"
            )
        if blind_assessment.get("pair_id") != pair_id:
            raise RealRefinementLoadError(
                f"{pair_id}: blind-assessment pair_id mismatch ({blind_assessment.get('pair_id')!r})"
            )
        if "capture_sha256" not in blind_assessment:
            raise RealRefinementLoadError(f"{pair_id}: blind-assessment file is missing capture_sha256")
        return blind_assessment["capture_sha256"]

    raise RealRefinementLoadError(
        f"{pair_id}: adjudication has neither capture_sha256 nor blind_assessment_sha256 -- cannot "
        "verify capture integrity"
    )


def _validate_adjudication(adjudication: dict[str, Any], pair_id: str) -> None:
    if adjudication.get("schema_version") != ADJUDICATION_SCHEMA_VERSION:
        raise RealRefinementLoadError(
            f"{pair_id}: adjudication schema_version must be {ADJUDICATION_SCHEMA_VERSION!r}, "
            f"got {adjudication.get('schema_version')!r}"
        )
    if adjudication.get("pair_id") != pair_id:
        raise RealRefinementLoadError(f"{pair_id}: adjudication pair_id mismatch ({adjudication.get('pair_id')!r})")

    directions = adjudication.get("dimension_directions") or {}
    present = set(directions)
    required = set(REQUIRED_DIMENSION_NAMES)
    if present != required:
        raise RealRefinementLoadError(
            f"{pair_id}: adjudication dimension_directions must have exactly {sorted(required)}, "
            f"got {sorted(present)}"
        )
    for dim, direction in directions.items():
        if direction not in VALID_DIRECTIONS:
            raise RealRefinementLoadError(
                f"{pair_id}: adjudication dimension {dim!r} direction {direction!r} not one of {VALID_DIRECTIONS}"
            )

    hard_failure_direction = adjudication.get("hard_failure_direction")
    if hard_failure_direction not in VALID_DIRECTIONS:
        raise RealRefinementLoadError(
            f"{pair_id}: adjudication hard_failure_direction {hard_failure_direction!r} not one of "
            f"{VALID_DIRECTIONS}"
        )

    # Reject any embedded report/session content BEFORE it could ever reach
    # a prediction -- reuses r6d4_capture's own recursive forbidden-key
    # scan directly, never a second reimplementation of that check.
    forbidden = r6d4_capture.find_forbidden_keys(adjudication)
    if forbidden:
        raise RealRefinementLoadError(f"{pair_id}: adjudication contains forbidden key(s): {forbidden}")


def _load_one(pair_id: str) -> Example:
    capture_path = CAPTURES_DIR / f"{pair_id}.json"
    adjudication_path = REAL_REVIEWS_DIR / f"{pair_id}-adjudication.json"

    capture = _load_json(capture_path, pair_id, "capture")
    try:
        r6d4_capture.validate_r6d4_capture(capture)
    except r6d4_capture.R6D4CaptureError as exc:
        raise RealRefinementLoadError(f"{pair_id}: capture failed r6d4-capture-v1 validation: {exc}") from exc

    adjudication = _load_json(adjudication_path, pair_id, "adjudication")
    _validate_adjudication(adjudication, pair_id)

    expected_hash = _resolve_expected_capture_sha256(adjudication, pair_id)
    actual_hash = _sha256(capture_path)
    if actual_hash != expected_hash:
        raise RealRefinementLoadError(
            f"{pair_id}: capture file hash {actual_hash!r} does not match the frozen review/"
            f"adjudication's own recorded hash {expected_hash!r} -- the capture file may have changed "
            "since it was reviewed"
        )

    expected_template = PAIR_ID_TEMPLATES[pair_id]
    if capture.get("template") != expected_template:
        raise RealRefinementLoadError(
            f"{pair_id}: capture template={capture.get('template')!r} != expected {expected_template!r}"
        )

    return Example(
        id=pair_id,
        inputs={
            "topic": capture["topic"],
            "template": capture["template"],
            "selected_papers": capture["selected_papers"],
            "approved_web_articles": capture["approved_web_articles"],
            "draft_report": capture["draft_report"],
            "refined_report": capture["refined_report"],
            # Only revision_applied is passed through -- the only field of
            # refinement_context predict/predict_live ever read. R4's own
            # r4_refinement_metadata/generation_model/etc. are kept out of
            # .inputs (the only part of an Example a judge prompt can ever
            # be built from) and preserved in .metadata instead, below.
            "refinement_context": {"revision_applied": capture["refinement_context"]["revision_applied"]},
        },
        outputs={
            "expected_hard_failure_direction": adjudication["hard_failure_direction"],
            "expected_dimension_directions": {
                dim: {
                    "direction": direction,
                    "rationale": f"see eval_data/report_refinement/real_reviews/{pair_id}-adjudication.json",
                }
                for dim, direction in adjudication["dimension_directions"].items()
            },
        },
        metadata={
            "source_origin": "real_r4_generated",
            "reviewer_provenance": adjudication.get("reviewer_provenance", {}),
            "capture_refinement_context": capture["refinement_context"],
        },
    )


def load_real_refinement_examples() -> list[Example]:
    """Loads exactly the 3 frozen real pairs named by `PAIR_IDS`, in
    that fixed order (foundational, analytical, expert) -- never
    directory-listing order, and never a larger/smaller set. Raises
    `RealRefinementLoadError` immediately on the first violation found
    for any pair: a missing capture/adjudication/blind-assessment file,
    a duplicate pair_id, a hash mismatch against the frozen review, an
    unknown dimension/direction, a template mismatch, or forbidden
    embedded content -- always before any OpenAI client is constructed
    (this function itself never constructs one; see `run_report_
    refinement_real.run_experiment` for where that ordering is
    enforced for live mode)."""
    seen_ids: set[str] = set()
    examples: list[Example] = []
    for pair_id in PAIR_IDS:
        if pair_id in seen_ids:
            raise RealRefinementLoadError(f"duplicate pair_id: {pair_id}")
        seen_ids.add(pair_id)
        examples.append(_load_one(pair_id))
    return examples
