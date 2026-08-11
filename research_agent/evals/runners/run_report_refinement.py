"""Runner for the report_refinement suite -- R6D.2's deterministic/mock
pair evaluation (unchanged), plus R6D.3a's opt-in live, changed-claim-
calibrated semantic pair evaluation, over R6D.1's own frozen pair
fixtures.

**R6D.3a supersedes R6D.3's live design** (R6D.3's own commit
`4aae124` first produced a real paid live pair, run_id 3 -- see
`docs/evaluation.md`'s "R6D.3a" section for the full evidence). Two
independent, whole-report judgments per side turned out to be the
wrong granularity for MEASURING A REFINEMENT specifically:

1. Judging draft and refined independently means ordinary claim/
   source and holistic sampling VARIANCE on content that never
   changed at all gets mistaken for a real semantic direction. run_id
   3's `gap_analysis:0:0` claim -- BYTE-IDENTICAL between draft and
   refined -- flipped from "supported" to "partially_supported"
   between the two independent claim/source calls; two independent
   holistic calls disagreed on source_balance/synthesis_quality
   (`not_applicable` -> `pass`) and moved analytical_quality/coherence
   scores by 0.15+ points, again over UNCHANGED report content.
2. R6C's whole-report groundedness aggregation is intentionally
   strict (any `partially_supported`/`unsupported` claim fails the
   whole dimension) -- exactly right for judging ONE report, but it
   means a genuine, isolated fix (run_id 3's Conclusion:
   "eliminates all" [unsupported] -> "reduces the rate of" [supported])
   can be invisible at the whole-report label level whenever any
   OTHER, unrelated claim is imperfect on both sides.

R6D.3a's fix, both parts required together (see the module-level
functions below for each):

- **Changed-claim comparison** for citation_correctness/groundedness:
  derive direction from EXACTLY the claim units that changed between
  draft and refined (by claim_id, exact-field equality -- never fuzzy
  text similarity, never an LLM), ignoring verdict variation on
  claim units that did not change at all. This directly answers "did
  the EDIT make this better or worse", not "does the whole report
  happen to have zero remaining imperfections anywhere".
- **One pairwise holistic call** (`judges/refinement_holistic.py`)
  replacing two independent standalone holistic calls, for the same
  reason: a call that sees both reports side-by-side and is asked to
  judge only the EFFECT of the actual edit cannot mistake unchanged
  content for a changed direction the way two independent calls can.

**Cost bound dropped from 4 to 3** for a normal, structurally valid,
non-identical pair: 1 claim/source call (draft) + 1 claim/source call
(refined) + 1 pairwise holistic call. No standalone holistic call is
ever made in this path -- `judges/holistic.py` is never imported here.

**No third, whole-pair LLM judge replaces citation_correctness/
groundedness** -- those two dimensions are still derived from two
INDEPENDENT claim/source judge calls (never a pairwise claim judge);
only the AGGREGATION changed (changed-claim-only, not whole-report).
Only the 5 holistic dimensions get a genuinely pairwise call.

**R6D.2 (mock) is byte-compatible and completely unchanged**:
`predict()`, `_side_prediction`, `_hard_failure_direction`, and the
mock-mode evaluator set are exactly as R6D.2 left them. A mock
prediction's `dimension_directions` is always `None` and `semantic_
evaluation_status` is always `"not_evaluated_in_mock_mode"`.

**Identical-pair optimization, tightened further under R6D.3a**: when
a fixture declares `revision_applied=false` AND `draft_report ==
refined_report` exactly (`report_refinement_inputs.reports_are_equal`,
deep equality -- never "same length" or "same references"), the draft
side's claim/source call is made once (for real diagnostics) and its
bundle is deep-copied for the refined side -- `identical_input_
reused=True` -- and the pairwise holistic judge is never called at
all (byte-identical content trivially implies `unchanged` on all 5
holistic dimensions; asking a judge to confirm that would be a wasted
call, not a real question). Maximum cost for an identical pair: 1 call
(down from R6D.3's 2).

R6A decision 1 applies here too, one level removed: this module never
imports `research_agent.report` and never calls `generate_report`/
`evaluate_report`/`revise_report` -- R6D measures whether R4's
existing "refine once" step already changed a report's structural AND
semantic state, using fixtures that already contain both a draft and
a refined body; it does not itself perform, request, or simulate a
refinement, and it does not alter the production R4 refinement loop
in any way.
"""

from __future__ import annotations

import copy
from typing import Any

from openai import OpenAI

from research_agent.evals import report_quality_inputs, report_refinement_inputs as rri
from research_agent.evals.evaluators.report_refinement import ALL_EVALUATORS
from research_agent.evals.judges import refinement_holistic
from research_agent.evals.runners import run_report_quality as rq
from research_agent.evals.runners._base import Example, LiveModeSetupError, SuiteResult, run_suite

SUITE = "report_refinement"

_SIDE_EXAMPLE_ID = "__report_refinement_side__"

_CLAIM_SOURCE_DIMENSION_NAMES = ("citation_correctness", "groundedness")
_HOLISTIC_DIMENSION_NAMES = ("synthesis_quality", "analytical_quality", "template_fit", "coherence", "source_balance")

# The exact judge-input fields a claim unit's identity depends on --
# matches report_quality_inputs.py's own claim-unit shape
# (`_claim_unit`). Two same-claim_id units are "the same claim" (never
# treated as changed) only when ALL FOUR are equal; a change to any one
# of them (a citation added/removed from a claim, a claim's kind
# reclassified) is a real, judgeable change, exactly as much as prose
# text changing. Deliberately NOT fuzzy: exact equality only, never a
# text-similarity threshold or an LLM call, per this phase's own
# explicit requirement.
_CLAIM_UNIT_IDENTITY_FIELDS = ("claim_text", "claim_kind", "reference_numbers", "evidence_ids")


# --- R6D.2: mock/deterministic pair evaluation (unchanged) -----------------

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


# --- R6D.3a: live claim/source side evaluation ------------------------------

def _evaluate_side_claim_only(
    report: dict[str, Any], selected_papers: list[dict[str, Any]], approved_web_articles: list[dict[str, Any]],
    topic: str, template: str, client: OpenAI,
) -> tuple[dict[str, Any] | None, str | None]:
    """Runs R6C's claim/source-only live path against ONE report body,
    via `run_report_quality.prepare_and_judge_claims_only` (the R6D.3a
    extraction -- never `predict_live`, which would also make a
    standalone holistic call this suite no longer wants per side),
    wrapped in a throwaway `Example`. Mirrors R6D.3's own `_evaluate_
    side_live` isolation: `prepare_and_judge_claims_only`/`judge_
    claims` themselves never raise, but this wrapper still catches any
    unexpected exception so one side's crash can never take down the
    other side or the whole example. Returns `(bundle, None)` on
    success, `(None, error_message)` on an unexpected failure.
    """
    side_example = Example(
        id=_SIDE_EXAMPLE_ID,
        inputs={
            "generated_report": report, "selected_papers": selected_papers,
            "approved_web_articles": approved_web_articles, "topic": topic, "template": template,
        },
        outputs={}, metadata={},
    )
    try:
        return rq.prepare_and_judge_claims_only(side_example, client), None
    except Exception as exc:  # noqa: BLE001 -- one side's crash must never take down the other side or the whole example.
        return None, str(exc)


def _side_summary_from_bundle(bundle: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    """Builds this suite's own compact per-side shape from `prepare_
    and_judge_claims_only`'s bundle. Preserves the raw claim/source
    judge output (verdicts, latency, token usage) for diagnostics, even
    though direction is derived from the changed-claim comparison
    below, never from `claim_source_dimensions`'s whole-report
    aggregation directly."""
    if bundle is None:
        unknown_reason = [f"side evaluation failed: {error}"]
        return {
            "hard_failures": [], "structural_status": "unknown", "informational_signals": {},
            "claim_source_dimensions": {
                dim: {"label": "unknown", "score": None, "reasons": unknown_reason}
                for dim in _CLAIM_SOURCE_DIMENSION_NAMES
            },
            "claim_source_judge_metadata": None, "error": error,
        }

    base_prediction = bundle["base_prediction"]
    claim_result = bundle["claim_result"]
    claim_source_judge_metadata = None
    if claim_result is not None:
        claim_source_judge_metadata = {
            "latency_ms": claim_result["latency_ms"], "error": claim_result["error"],
            "claims_judged": claim_result["claims_judged"], "token_usage": claim_result["token_usage"],
            "verdicts": claim_result["verdicts"],
            "not_a_verifiable_claim_ids": claim_result["not_a_verifiable_claim_ids"],
        }
    return {
        "hard_failures": base_prediction["hard_failures"],
        "structural_status": base_prediction["structural_integrity"]["status"],
        "informational_signals": base_prediction["informational_signals"],
        "claim_source_dimensions": bundle["claim_source_dimensions"],
        "claim_source_judge_metadata": claim_source_judge_metadata,
        "error": error,
    }


# --- R6D.3a: changed-claim comparison ---------------------------------------

def _claims_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """R6C.1's own REAL prepared (and bounded-sampled) claim units --
    `selected_cited_claims`/`selected_uncited_candidates` are exactly
    what the claim/source judge actually saw and judged, so matching
    against these (never the full unsampled extraction) means the
    change inventory below always reflects what was actually judged."""
    claims = payload["selected_cited_claims"] + payload["selected_uncited_candidates"]
    return {c["claim_id"]: c for c in claims}


def _claim_units_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(a.get(field) == b.get(field) for field in _CLAIM_UNIT_IDENTITY_FIELDS)


def compute_claim_change_inventory(draft_payload: dict[str, Any], refined_payload: dict[str, Any]) -> dict[str, Any]:
    """Classifies every claim_id present in either side's prepared
    claim units as exactly one of unchanged/changed/added/removed --
    matched by claim_id, "unchanged" requires exact equality on all of
    `_CLAIM_UNIT_IDENTITY_FIELDS` (never fuzzy text similarity, never
    an LLM call). Returns both the four id lists (what gets persisted
    in the prediction) and the two claims-by-id maps (kept for this
    module's own internal use deriving direction/building the pairwise
    judge's changed-claim summary -- never persisted directly, since
    they duplicate the full claim-unit payload)."""
    draft_claims = _claims_by_id(draft_payload)
    refined_claims = _claims_by_id(refined_payload)
    draft_ids = set(draft_claims)
    refined_ids = set(refined_claims)
    shared_ids = draft_ids & refined_ids

    unchanged_ids = sorted(cid for cid in shared_ids if _claim_units_equal(draft_claims[cid], refined_claims[cid]))
    changed_ids = sorted(shared_ids - set(unchanged_ids))
    added_ids = sorted(refined_ids - draft_ids)
    removed_ids = sorted(draft_ids - refined_ids)

    return {
        "unchanged_claim_ids": unchanged_ids,
        "changed_claim_ids": changed_ids,
        "added_claim_ids": added_ids,
        "removed_claim_ids": removed_ids,
        "draft_claims_by_id": draft_claims,
        "refined_claims_by_id": refined_claims,
    }


def _verdict_lookup(claim_result: dict[str, Any] | None) -> dict[str, dict[str, Any]] | None:
    """Returns `claim_result["verdicts"]` when the judge call actually
    succeeded, or `None` when it failed/was never made -- callers must
    treat a `None` lookup as producing "unknown" for every claim_id
    (never silently default to "not_applicable", which would hide a
    real judge failure behind a falsely-neutral direction)."""
    if claim_result is None or claim_result.get("error") is not None:
        return None
    return claim_result["verdicts"]


def _per_claim_citation_status(verdict_entry: dict[str, Any] | None) -> str:
    """The frozen R6C citation policy (CITATION_AGGREGATION_POLICY_
    VERSION's own per-source rule, `run_report_quality._aggregate_
    claim_source_dimensions`), applied to ONE claim's own source
    verdicts instead of aggregated across a whole report:
      - "fail": any attached source verdict is "does_not_support".
      - "unknown": no fail, but any attached source verdict is
        "insufficient_evidence".
      - "pass": every attached source is "supports" or "partially_
        supports".
      - "not_applicable": no attached source verdict exists at all
        (an uncited claim, or a claim never judged)."""
    if verdict_entry is None:
        return "not_applicable"
    source_verdicts = verdict_entry.get("source_verdicts") or []
    if not source_verdicts:
        return "not_applicable"
    values = [sv["verdict"] for sv in source_verdicts]
    if "does_not_support" in values:
        return "fail"
    if "insufficient_evidence" in values:
        return "unknown"
    return "pass"


def _per_claim_groundedness_status(verdict_entry: dict[str, Any] | None) -> str:
    """R6C's own strict groundedness policy (`partially_supported`
    remains a failure state -- this phase deliberately does NOT
    introduce severity/materiality scoring), applied to ONE claim's
    own collective verdict instead of aggregated across a whole report:
      - "pass": collective_verdict == "supported".
      - "fail": collective_verdict in ("partially_supported",
        "unsupported").
      - "unknown": collective_verdict == "insufficient_evidence".
      - "not_applicable": collective_verdict == "not_a_verifiable_
        claim", the claim was never judged, or the entry is malformed
        (defensive; never invents "fail"/"pass" from an unrecognized
        value)."""
    if verdict_entry is None:
        return "not_applicable"
    verdict = verdict_entry.get("collective_verdict")
    if verdict == "supported":
        return "pass"
    if verdict in ("partially_supported", "unsupported"):
        return "fail"
    if verdict == "insufficient_evidence":
        return "unknown"
    return "not_applicable"


def _claim_status_direction(draft_status: str, refined_status: str) -> str:
    """Compares one claim's own draft/refined status (never a whole-
    report label):
      - fail -> pass = improved
      - pass -> fail = regressed
      - same comparable status (fail==fail or pass==pass) = unchanged
      - either unknown = unknown
      - both not_applicable = unchanged
      - exactly one not_applicable = unknown (applicability itself
        changed; direction can't be inferred safely)"""
    if draft_status == "unknown" or refined_status == "unknown":
        return "unknown"
    if draft_status == "not_applicable" and refined_status == "not_applicable":
        return "unchanged"
    if draft_status == "not_applicable" or refined_status == "not_applicable":
        return "unknown"
    if draft_status == "fail" and refined_status == "pass":
        return "improved"
    if draft_status == "pass" and refined_status == "fail":
        return "regressed"
    return "unchanged"


def _aggregate_claim_directions(directions: list[str]) -> str:
    """Aggregates a list of per-claim directions (already computed by
    `_claim_status_direction`, plus any "unknown" contributed by an
    added/removed relevant claim) into one dimension-level direction:
      - no directions at all (no changed/relevant claims) -> unchanged
      - only "unchanged" entries -> unchanged
      - any "unknown" present, OR both "improved" and "regressed"
        present -> unknown (a real mixed tradeoff is NOT silently
        resolved into one direction -- see run_report_refinement.py's
        own dimension_directions docstring for the equivalent
        whole-dimension rule this mirrors)
      - only "improved" (and no "regressed"/"unknown") -> improved
      - only "regressed" (and no "improved"/"unknown") -> regressed"""
    if not directions:
        return "unchanged"
    if any(d == "unknown" for d in directions):
        return "unknown"
    has_improved = "improved" in directions
    has_regressed = "regressed" in directions
    if has_improved and has_regressed:
        return "unknown"
    if has_improved:
        return "improved"
    if has_regressed:
        return "regressed"
    return "unchanged"


def _citation_correctness_from_claims(
    inventory: dict[str, Any], draft_verdicts: dict[str, dict[str, Any]] | None,
    refined_verdicts: dict[str, dict[str, Any]] | None,
) -> tuple[str, dict[str, Any]]:
    """citation_correctness direction, changed-claim-only. An added or
    removed CITED claim (present on only one side, with no baseline to
    compare against on the other) contributes "unknown" -- per this
    phase's own explicit requirement, this is never inferred as an
    improvement or regression just because a citation appeared or
    disappeared. An added/removed UNCITED claim never affects citation_
    correctness at all (it never had a source verdict to begin with on
    either side)."""
    directions: list[str] = []
    detail: dict[str, Any] = {}

    for claim_id in inventory["changed_claim_ids"]:
        draft_status = _per_claim_citation_status((draft_verdicts or {}).get(claim_id)) if draft_verdicts is not None else "unknown"
        refined_status = _per_claim_citation_status((refined_verdicts or {}).get(claim_id)) if refined_verdicts is not None else "unknown"
        direction = _claim_status_direction(draft_status, refined_status)
        directions.append(direction)
        detail[claim_id] = {"draft_status": draft_status, "refined_status": refined_status, "direction": direction}

    for claim_id in inventory["added_claim_ids"]:
        if inventory["refined_claims_by_id"][claim_id]["claim_kind"] != "cited":
            continue
        refined_status = _per_claim_citation_status((refined_verdicts or {}).get(claim_id)) if refined_verdicts is not None else "unknown"
        directions.append("unknown")
        detail[claim_id] = {
            "draft_status": None, "refined_status": refined_status, "direction": "unknown",
            "reason": "added cited claim -- no draft baseline to compare",
        }

    for claim_id in inventory["removed_claim_ids"]:
        if inventory["draft_claims_by_id"][claim_id]["claim_kind"] != "cited":
            continue
        draft_status = _per_claim_citation_status((draft_verdicts or {}).get(claim_id)) if draft_verdicts is not None else "unknown"
        directions.append("unknown")
        detail[claim_id] = {
            "draft_status": draft_status, "refined_status": None, "direction": "unknown",
            "reason": "removed cited claim -- no refined baseline to compare",
        }

    return _aggregate_claim_directions(directions), detail


def _groundedness_from_claims(
    inventory: dict[str, Any], draft_verdicts: dict[str, dict[str, Any]] | None,
    refined_verdicts: dict[str, dict[str, Any]] | None,
) -> tuple[str, dict[str, Any]]:
    """groundedness direction, changed-claim-only. An added/removed
    claim whose only available (single-side) verdict is "not_a_
    verifiable_claim" is excluded entirely -- consistent with R6C's own
    "framing/organizational prose is neither pass, fail, nor unknown"
    convention -- never invented as "unknown" for content that was
    never a factual assertion to begin with. Any other added/removed
    claim contributes "unknown" (no baseline to compare against)."""
    directions: list[str] = []
    detail: dict[str, Any] = {}

    for claim_id in inventory["changed_claim_ids"]:
        draft_status = _per_claim_groundedness_status((draft_verdicts or {}).get(claim_id)) if draft_verdicts is not None else "unknown"
        refined_status = _per_claim_groundedness_status((refined_verdicts or {}).get(claim_id)) if refined_verdicts is not None else "unknown"
        direction = _claim_status_direction(draft_status, refined_status)
        directions.append(direction)
        detail[claim_id] = {"draft_status": draft_status, "refined_status": refined_status, "direction": direction}

    for claim_id in inventory["added_claim_ids"]:
        refined_status = _per_claim_groundedness_status((refined_verdicts or {}).get(claim_id)) if refined_verdicts is not None else "unknown"
        if refined_status == "not_applicable":
            continue
        directions.append("unknown")
        detail[claim_id] = {
            "draft_status": None, "refined_status": refined_status, "direction": "unknown",
            "reason": "added claim -- no draft baseline to compare",
        }

    for claim_id in inventory["removed_claim_ids"]:
        draft_status = _per_claim_groundedness_status((draft_verdicts or {}).get(claim_id)) if draft_verdicts is not None else "unknown"
        if draft_status == "not_applicable":
            continue
        directions.append("unknown")
        detail[claim_id] = {
            "draft_status": draft_status, "refined_status": None, "direction": "unknown",
            "reason": "removed claim -- no refined baseline to compare",
        }

    return _aggregate_claim_directions(directions), detail


# --- R6D.3a: pairwise holistic judge -----------------------------------------

def _build_changed_claim_summary(inventory: dict[str, Any]) -> str:
    """A deterministic, ID-ONLY summary -- NEVER embeds raw claim/
    report TEXT outside the already-sanitized draft/refined report
    blocks the pairwise judge also receives, so this summary can never
    become a second, unsanitized channel for injected content. Lists
    only claim_ids and their section_key, grouped by change type."""
    changed_sections = sorted({
        *(inventory["draft_claims_by_id"][cid]["section_key"] for cid in inventory["changed_claim_ids"]),
        *(inventory["refined_claims_by_id"][cid]["section_key"] for cid in inventory["added_claim_ids"]),
        *(inventory["draft_claims_by_id"][cid]["section_key"] for cid in inventory["removed_claim_ids"]),
    })
    if not changed_sections:
        return "No claim-unit-level differences were detected between the draft and refined report."

    lines = [f"Sections containing a claim-level change: {changed_sections}"]
    if inventory["changed_claim_ids"]:
        lines.append(f"Changed claim ids: {inventory['changed_claim_ids']}")
    if inventory["added_claim_ids"]:
        lines.append(f"Added claim ids (present only in refined): {inventory['added_claim_ids']}")
    if inventory["removed_claim_ids"]:
        lines.append(f"Removed claim ids (present only in draft): {inventory['removed_claim_ids']}")
    return "\n".join(lines)


def _run_pairwise_holistic(
    topic: str, template: str, draft_payload: dict[str, Any], refined_payload: dict[str, Any],
    inventory: dict[str, Any], client: OpenAI,
) -> dict[str, Any]:
    """The suite's only holistic call for a normal pair -- passes the
    SAME `sanitized_report_sections` R6C.1's own preparation already
    produced per side (never a second, independent sanitization pass),
    plus the ID-only changed-claim summary above."""
    changed_claim_summary = _build_changed_claim_summary(inventory)
    return refinement_holistic.judge_refinement_holistic(
        topic, template, draft_payload["sanitized_report_sections"], refined_payload["sanitized_report_sections"],
        changed_claim_summary, client, rq.REPORT_QUALITY_JUDGE_MODEL,
    )


def _holistic_directions_from_pairwise(pairwise_result: dict[str, Any]) -> dict[str, str]:
    if pairwise_result["error"] is not None or not pairwise_result["dimensions"]:
        return {dim: "unknown" for dim in _HOLISTIC_DIMENSION_NAMES}
    return {
        dim: (pairwise_result["dimensions"].get(dim) or {}).get("direction", "unknown")
        for dim in _HOLISTIC_DIMENSION_NAMES
    }


# --- R6D.3a: predict_live orchestration --------------------------------------

def predict_live(example: Example, client: OpenAI) -> dict[str, Any]:
    """R6D.3a: evaluates both sides' claim/source judgments
    independently (never a pairwise claim judge), derives citation_
    correctness/groundedness direction from ONLY the claim units that
    actually changed, and derives the 5 holistic directions from ONE
    pairwise call that sees both reports together. Maximum 3 judge
    calls for a normal, structurally valid, non-identical pair.
    """
    selected_papers = example.inputs["selected_papers"]
    approved_web_articles = example.inputs["approved_web_articles"]
    topic = example.inputs["topic"]
    template = example.inputs["template"]
    draft_report = example.inputs["draft_report"]
    refined_report = example.inputs["refined_report"]
    refinement_context = example.inputs.get("refinement_context") or {}

    draft_bundle, draft_bundle_error = _evaluate_side_claim_only(
        draft_report, selected_papers, approved_web_articles, topic, template, client,
    )

    identical_input_reused = False
    revision_applied = refinement_context.get("revision_applied")
    if (
        draft_bundle_error is None and draft_bundle is not None
        and revision_applied is False and rri.reports_are_equal(draft_report, refined_report)
    ):
        # Exact report equality only -- never inferred from equal length/references. A fresh
        # deep copy so the two sides never share mutable state, even though the input was identical.
        refined_bundle = copy.deepcopy(draft_bundle)
        refined_bundle_error = None
        identical_input_reused = True
    else:
        refined_bundle, refined_bundle_error = _evaluate_side_claim_only(
            refined_report, selected_papers, approved_web_articles, topic, template, client,
        )

    draft_side = _side_summary_from_bundle(draft_bundle, draft_bundle_error)
    refined_side = _side_summary_from_bundle(refined_bundle, refined_bundle_error)
    hard_failure_direction = _hard_failure_direction(draft_side["hard_failures"], refined_side["hard_failures"])

    draft_call_made = bool(draft_bundle and draft_bundle["call_made"])
    refined_call_made = bool((not identical_input_reused) and refined_bundle and refined_bundle["call_made"])

    draft_eligible = draft_bundle is not None and draft_bundle["payload"]["evaluation_status"] != report_quality_inputs.EVALUATION_STATUS_SKIPPED
    refined_eligible = refined_bundle is not None and refined_bundle["payload"]["evaluation_status"] != report_quality_inputs.EVALUATION_STATUS_SKIPPED

    if not draft_eligible or not refined_eligible:
        # A structurally invalid side is never judged (preserves R6C's own hard-failure skip
        # gating) -- nothing here can be fairly compared, so every one of the 7 dimensions is
        # "unknown", exactly as `structural_regression` has always frozen this behavior.
        dimension_directions = {dim: "unknown" for dim in rri.REQUIRED_DIMENSION_NAMES}
        claim_change_inventory = None
        claim_direction_detail = {"citation_correctness": {}, "groundedness": {}}
        pairwise_holistic_metadata = {
            "attempted": False,
            "skip_reason": "one or both sides failed R6B's structural hard-failure gate -- no semantic comparison is possible",
        }
    else:
        inventory = compute_claim_change_inventory(draft_bundle["payload"], refined_bundle["payload"])
        draft_verdicts = _verdict_lookup(draft_bundle["claim_result"])
        refined_verdicts = _verdict_lookup(refined_bundle["claim_result"])

        citation_direction, citation_detail = _citation_correctness_from_claims(inventory, draft_verdicts, refined_verdicts)
        groundedness_direction, groundedness_detail = _groundedness_from_claims(inventory, draft_verdicts, refined_verdicts)

        if identical_input_reused:
            holistic_directions = {dim: "unchanged" for dim in _HOLISTIC_DIMENSION_NAMES}
            pairwise_holistic_metadata = {
                "attempted": False,
                "skip_reason": "identical_input_reused -- draft and refined reports are byte-identical",
            }
        else:
            pairwise_result = _run_pairwise_holistic(
                topic, template, draft_bundle["payload"], refined_bundle["payload"], inventory, client,
            )
            holistic_directions = _holistic_directions_from_pairwise(pairwise_result)
            pairwise_holistic_metadata = {
                "attempted": True, "model": pairwise_result["model"], "prompt_version": pairwise_result["prompt_version"],
                "latency_ms": pairwise_result["latency_ms"], "token_usage": pairwise_result["token_usage"],
                "error": pairwise_result["error"], "dimensions": pairwise_result["dimensions"],
            }

        dimension_directions = {
            "citation_correctness": citation_direction, "groundedness": groundedness_direction, **holistic_directions,
        }
        claim_change_inventory = {
            "unchanged_claim_ids": inventory["unchanged_claim_ids"],
            "changed_claim_ids": inventory["changed_claim_ids"],
            "added_claim_ids": inventory["added_claim_ids"],
            "removed_claim_ids": inventory["removed_claim_ids"],
        }
        claim_direction_detail = {"citation_correctness": citation_detail, "groundedness": groundedness_detail}

    judge_call_count = int(draft_call_made) + int(refined_call_made) + int(pairwise_holistic_metadata["attempted"])

    return {
        "pair_id": example.id,
        "draft": draft_side,
        "refined": refined_side,
        "hard_failure_direction": hard_failure_direction,
        "dimension_directions": dimension_directions,
        "semantic_evaluation_status": _semantic_evaluation_status(dimension_directions),
        "identical_input_reused": identical_input_reused,
        "claim_change_inventory": claim_change_inventory,
        "claim_direction_detail": claim_direction_detail,
        "pairwise_holistic": pairwise_holistic_metadata,
        "judge_call_count": judge_call_count,
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

    `mode="live"` (R6D.3a): constructs a real OpenAI client up front via
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
