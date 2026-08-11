"""R6D.4a: developer-only, in-memory capture of a genuine R4 draft/
refined report pair, for later R6D live evaluation. Makes real R4
calls when actually invoked (report generation, R4's own evaluator,
and -- when R4 decides one is warranted -- exactly one revision); this
module itself never decides to run, that is entirely up to whatever
calls `capture_real_refinement_pair` below (in R6D.4a, only tests,
always against a mocked `client`).

**Eval-only, in-memory, no persistence.** This module writes nothing
to disk (R6D.4b's job) and never touches production session state:
it never calls `research_agent.report.append_report_version` or
`research_agent.curation_session.save_curation_session` (neither is
even imported here), and it never assigns to `session.report`,
`session.report_versions`, or `session.active_report_version_id`.
`session` is read from, never written to -- see `capture_real_
refinement_pair`'s own docstring for the exact fields read.

**Reuses the exact production generation/refinement functions the
real "Generate" path (`services/curation_report_service.py::get_or_
create_report`) already calls, in the exact same order, with the
exact same arguments** -- `research_agent.report.generate_report_for_
session` (session-stage validation, `session.selected_papers`
resolution -- never `session.reserve`, which curation_report_service.
py's own get_or_create_report deliberately avoids for the same reason)
then `research_agent.report.refine_report_if_requested` with
`web_articles=[]` (a literal empty list, exactly what get_or_create_
report itself hardcodes for its own initial-generation call -- **not**
derived from `session.web_articles_added`, even when that pool is
non-empty; the initial Generate path has never offered web sources to
the model, only `/report/regenerate` does, and this module captures
the initial-Generate path specifically). No other production entry
point (`generate_report` directly, `regenerate_report_with_new_
sources`, chat-triggered regeneration) is used here -- doing so would
capture a different production path with different evidence-
resolution semantics, not merely a "simpler" version of the same one.

**Draft-preservation order** (the one sequence that survives BOTH of
`refine_report_if_requested`'s branches correctly -- see that
function's own docstring: the no-revision branch mutates its `draft`
argument in place, `final = draft; final["refinement"] = {...}`; the
revision branch returns an entirely distinct dict and never touches
the original `draft` object again):

1. Generate the draft once (`generate_report_for_session`).
2. Immediately `copy.deepcopy(draft)` into `draft_snapshot`, BEFORE
   `draft` is ever passed to refinement -- this is the only point at
   which an untouched draft copy can be taken; after step 3, `draft`
   itself may already have been mutated.
3. Pass the ORIGINAL (non-copied) `draft` object into `refine_report_
   if_requested`.
4. Capture whatever it returns as `final`.
5. `copy.deepcopy(final)` into `final_snapshot`.
6. Pop `final_snapshot["refinement"]` off as this pair's own
   refinement metadata (R4's own evaluation of the draft -- issues,
   scores, rounds -- kept in `refinement_context.r4_refinement_
   metadata`, clearly namespaced as R4's OWN judgment, never conflated
   with R6D's own, independent, later live re-judgment of the
   resulting pair).
7. `draft_snapshot` never legitimately carries a `refinement` key at
   all (nothing is ever stamped onto it) -- popped defensively anyway,
   so a stored `draft_report`/`refined_report` pair is always
   comparable on equal footing, matching every existing R6D.1
   synthetic fixture's own report-body shape (none of which carry a
   `refinement` key either).
8. `draft_snapshot` is never touched again after step 2.

**No answer key.** A captured artifact deliberately has no `expected`
block, no `overall_direction`/`overall_score`/`winner`/`accept_
refinement` field, and no evaluator-model/prompt-version metadata --
see this module's own docstring section on `SCHEMA_VERSION` below for
why: assigning expectations only after seeing a live judge's own
output would be answer-key bias. If human labels are ever collected
for a captured pair, they must be written before that pair's own R6D
live evaluation runs, exactly as a synthetic R6D.1 fixture's `expected`
block is authored before any live judge ever sees it -- never derived
from, or corrected merely to match, the live result.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from research_agent import report as report_module
from research_agent.query_expansion import PaperPoolSession

# The one schema version this module ever produces -- deliberately
# distinct from `report_refinement_inputs.SCHEMA_VERSION` ("r6d1-v1",
# the frozen, LABELLED synthetic-fixture schema, which requires a
# complete, human-authored `expected.dimension_directions` block for
# every fixture it loads). An r6d4-capture-v1 artifact is the opposite
# on purpose: UNLABELLED by construction, with no `expected` block at
# all -- conflating the two schemas, or routing a capture artifact
# through `report_refinement_inputs.validate_pair`, would either reject
# it outright (missing `expected`) or -- far worse -- silently invite
# fabricating one after the fact. This module's own `validate_r6d4_
# capture` below is a separate, independent validator; it never
# imports from `report_refinement_inputs.py`, and `report_refinement_
# inputs.py` never imports from here -- the same "independent copies,
# never cross-import validation internals" posture every other R6
# module already established for its own frozen constants.
SCHEMA_VERSION = "r6d4-capture-v1"

REQUIRED_SOURCE_ORIGIN = "real_r4_generated"
REQUIRED_REFINEMENT_MODE = "single"
VALID_TEMPLATES = ("foundational", "analytical", "expert")

# Independent copy of report_quality_inputs.py/report_refinement_
# inputs.py's own REQUIRED_SECTION_KEYS -- same reasoning as always in
# this project: a future refactor of either of those modules must
# never silently change what THIS module's validator checks a real
# captured report body against.
REQUIRED_SECTION_KEYS = (
    "executive_summary", "introduction_scope", "thematic_findings",
    "methodology_landscape", "contradictions_open_debates", "gap_analysis",
    "future_research_directions", "conclusion",
)

# A captured artifact must never carry any of these -- either because
# they're an accept/reject-style composite this project has never
# introduced anywhere in R6D (see report_refinement_inputs.py's own
# "no overall score, weighted composite, or winner field" requirement,
# which applies here just as strictly), or because they're evaluator-
# side metadata that belongs to a LATER live-run's own detail JSON,
# never to the capture artifact itself (an artifact must stay usable
# before any evaluator has ever looked at it).
_FORBIDDEN_TOP_LEVEL_KEYS = (
    "expected", "expected_dimension_directions", "overall_direction", "overall_score",
    "winner", "accept_refinement", "evaluator_model", "claim_source_prompt_version",
    "pairwise_holistic_prompt_version",
)

# Recursive forbidden-key check (Part 6/9): every key name below is
# compared case-insensitively against every dict key anywhere in the
# artifact, at any nesting depth -- defense in depth against a future
# refactor accidentally serializing more of `session` than this module
# intends to read from it today (session_id/chat_history/turn_history/
# pending_web_offer/pending_report_update are PaperPoolSession/session-
# envelope fields this module never reads and must never leak;
# api_key/openai_api_key guard against a credential ever ending up
# inside a captured `client`-adjacent value by mistake).
_FORBIDDEN_RECURSIVE_KEYS = frozenset({
    "session_id", "chat_history", "turn_history", "pending_web_offer", "pending_report_update",
    "openai_api_key", "api_key",
})

# A `source_session_ref` shaped like the checkpointer's own internal
# thread-id namespace, or like a bare uuid4 hex (the shape every real
# session_id/version_id in this codebase actually has -- see report.py's
# own `append_report_version`/`curation_session.py`'s `curation_thread_
# id`), is exactly the "raw identifier, not an opaque eval label" this
# field must never be. Best-effort only -- this cannot prove a string
# is NOT some other system's real identifier, only reject the shapes
# this codebase itself is known to produce.
_CURATION_THREAD_ID_PREFIX = "curation-session:"


class R6D4CaptureError(ValueError):
    """Raised for any capture-time or validation-time defect in an
    r6d4-capture-v1 artifact -- a contradictory revision/equality
    provenance, a malformed capture timestamp, a forbidden key found
    anywhere in the artifact, or (in `validate_r6d4_capture`) any
    structural defect in an already-built artifact dict. Deliberately
    loud and immediate, same posture as `report_refinement_inputs.
    ReportRefinementFixtureError` -- a bad capture must never silently
    produce a partial or misleading artifact."""


def find_forbidden_keys(obj: Any, _path: str = "$") -> list[str]:
    """Recursively walks `obj` (an already-built, JSON-native dict/
    list/scalar structure -- never a live `PaperPoolSession`/dataclass
    instance) and returns every JSONPath-ish location whose own dict
    key matches (case-insensitively) `_FORBIDDEN_RECURSIVE_KEYS`.
    Returns an empty list when the structure is clean. Exposed
    directly (not just used internally) so a test can assert both "an
    innocent artifact reports nothing" and "a deliberately poisoned
    dict is actually caught by this exact function," not merely by
    `capture_real_refinement_pair`'s own internal use of it."""
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{_path}.{key}"
            if isinstance(key, str) and key.lower() in _FORBIDDEN_RECURSIVE_KEYS:
                found.append(key_path)
            found.extend(find_forbidden_keys(value, key_path))
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            found.extend(find_forbidden_keys(item, f"{_path}[{index}]"))
    return found


def _validate_source_session_ref(source_session_ref: str) -> None:
    if not source_session_ref or not source_session_ref.strip():
        raise R6D4CaptureError("source_session_ref must be a non-empty opaque evaluation reference")
    if source_session_ref.startswith(_CURATION_THREAD_ID_PREFIX):
        raise R6D4CaptureError(
            f"source_session_ref {source_session_ref!r} looks like a raw checkpointer thread id, "
            "not an opaque evaluation reference -- e.g. use 'real-pair-foundational-01' instead"
        )
    bare = source_session_ref.replace("-", "")
    if len(bare) == 32 and all(c in "0123456789abcdefABCDEF" for c in bare):
        raise R6D4CaptureError(
            f"source_session_ref {source_session_ref!r} looks like a raw uuid4-hex session/version id, "
            "not an opaque evaluation reference -- e.g. use 'real-pair-foundational-01' instead"
        )


def _validate_capture_timestamp(capture_timestamp: str) -> None:
    try:
        parsed = datetime.fromisoformat(capture_timestamp)
    except (TypeError, ValueError) as exc:
        raise R6D4CaptureError(f"capture_timestamp {capture_timestamp!r} is not a parseable ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise R6D4CaptureError(f"capture_timestamp {capture_timestamp!r} must be timezone-aware")


def _check_revision_consistency(revision_applied: bool, bodies_equal: bool) -> None:
    if revision_applied and bodies_equal:
        raise R6D4CaptureError(
            "contradictory provenance: r4_refinement_metadata reports rounds > 0 (a revision happened), "
            "but draft_report and refined_report are equal after refinement-key stripping"
        )
    if not revision_applied and not bodies_equal:
        raise R6D4CaptureError(
            "contradictory provenance: r4_refinement_metadata reports rounds == 0 (no revision happened), "
            "but draft_report and refined_report differ after refinement-key stripping"
        )


def _check_section_shape(report: dict[str, Any], side: str) -> None:
    missing = [k for k in REQUIRED_SECTION_KEYS if k not in report]
    if missing:
        raise R6D4CaptureError(f"{side}_report missing required section(s): {missing}")
    sections_list = report.get("sections") or []
    section_list_keys = [s.get("key") for s in sections_list]
    if section_list_keys != list(REQUIRED_SECTION_KEYS):
        raise R6D4CaptureError(
            f"{side}_report.sections key order {section_list_keys} != canonical order {list(REQUIRED_SECTION_KEYS)}"
        )


# The 3 legacy projection keys (`_project_legacy_fields`'s own
# `findings`/`limitations`/`future_scope`) are pure aliases of 3 of the
# 8 real Analytical section keys -- never independently generated text,
# never read by any R6C/R6D evaluation code. Matches R6D.1's own
# established fixture-body convention exactly (see `eval_data/report_
# refinement/README.md`'s own "the 3 legacy projection keys are
# omitted -- pure aliases, never read by any deterministic check or
# claim extraction" note) -- an independent copy of that same
# omission, not an import of it.
_LEGACY_SECTION_KEYS = ("findings", "limitations", "future_scope")


def _strip_report_body_for_capture(report: dict[str, Any]) -> dict[str, Any]:
    """Returns a deep-copied, JSON-native version of `report` --
    `generate_report`/`refine_report_if_requested`'s own real return
    shape nests raw `Paper`/`WebArticle` dataclass instances in every
    section's `cited_papers`/`cited_web_articles` and in the top-level
    `skipped_papers`, exactly the same non-JSON-native shape `research_
    agent/curation_session.py::_serialize_report` exists to handle for
    session persistence -- this module needs its own independent
    handling (never importing that private, persistence-specific
    helper) for its own, narrower eval-artifact purpose:

    - Each of the 8 real section keys' own `cited_papers`/`cited_web_
      articles` is dropped entirely -- matching R6D.1's own established
      fixture-body convention (`_LEGACY_SECTION_KEYS`'s own docstring
      above), since nothing in the R6C/R6D evaluation pipeline reads
      per-section citation objects at all; evidence resolution goes
      entirely through the report's own `references` list plus the
      pair-level `selected_papers`/`approved_web_articles`.
    - The 3 legacy section keys are dropped entirely (see
      `_LEGACY_SECTION_KEYS` above).
    - `skipped_papers` is converted to plain `.to_dict()` form, never
      dropped -- unlike per-section citations, its own LENGTH is a real
      informational signal `run_report_quality.py`'s own skipped-
      paper-rate computation reads (never a field of an individual
      skipped paper, only `len(...)`), so the list survives, just
      JSON-native.
    - The top-level `refinement` key (R4's own evaluation metadata,
      stamped by `refine_report_if_requested`) is dropped -- callers of
      this function extract it from the ORIGINAL `final` dict first
      (see `capture_real_refinement_pair`), never from this stripped
      copy.

    Never mutates `report` itself -- everything above operates on a
    fresh `copy.deepcopy`.
    """
    stripped = copy.deepcopy(report)
    for key in REQUIRED_SECTION_KEYS:
        if key in stripped:
            stripped[key].pop("cited_papers", None)
            stripped[key].pop("cited_web_articles", None)
    for legacy_key in _LEGACY_SECTION_KEYS:
        stripped.pop(legacy_key, None)
    if "skipped_papers" in stripped:
        stripped["skipped_papers"] = [
            p.to_dict() if hasattr(p, "to_dict") else p for p in stripped["skipped_papers"]
        ]
    stripped.pop("refinement", None)
    return stripped


def _check_references_resolve(
    report: dict[str, Any], selected_papers: list[dict[str, Any]], approved_web_articles: list[dict[str, Any]], side: str,
) -> None:
    known_paper_ids = {p["paper_id"] for p in selected_papers if p.get("paper_id")}
    known_urls = {a["url"] for a in approved_web_articles if a.get("url")}
    unresolved: list[int] = []
    for r in report.get("references") or []:
        kind = r.get("kind")
        if kind == "paper":
            if not r.get("paper_id") or r["paper_id"] not in known_paper_ids:
                unresolved.append(r["number"])
        elif kind == "web":
            if not r.get("url") or r["url"] not in known_urls:
                unresolved.append(r["number"])
        else:
            unresolved.append(r["number"])
    if unresolved:
        raise R6D4CaptureError(f"{side}_report has reference(s) not resolvable against the shared evidence snapshot: {unresolved}")


def capture_real_refinement_pair(
    session: PaperPoolSession,
    client: OpenAI,
    *,
    report_template: str,
    pair_id: str,
    source_session_ref: str,
    tags: list[str] | None = None,
    notes: str = "",
    now: str | None = None,
    source_commit_sha: str | None = None,
) -> dict[str, Any]:
    """Runs the real "Generate" production path once (`generate_report_
    for_session` then `refine_report_if_requested`, `refinement_mode=
    "single"` -- see this module's own docstring for the exact call
    shape reused) and returns a complete, unlabelled `r6d4-capture-v1`
    artifact dict, entirely in memory. Writes no file. Makes no
    production-session mutation of any kind -- `session.report`,
    `session.report_versions`, and `session.active_report_version_id`
    are never assigned to, and neither `research_agent.report.append_
    report_version` nor `research_agent.curation_session.save_
    curation_session` is called or even imported here.

    Reads exactly two fields off `session`: `session.topic` and
    `session.selected_papers` (both already read internally by
    `generate_report_for_session` itself, which also independently
    re-reads `session.stage` for its own readiness check -- this
    function never bypasses that validation). Never reads `session.
    web_articles_added`, `session.chat_history`, `session.turn_
    history`, or any other session field -- `approved_web_articles` in
    the returned artifact is always `[]`, matching exactly what `get_
    or_create_report`'s own initial-generation call hardcodes for
    `refine_report_if_requested`'s own `web_articles` argument,
    regardless of what a session's own discovered-web-source pool
    might separately contain.

    `source_session_ref` must be a caller-chosen, opaque evaluation
    label (e.g. `"real-pair-foundational-01"`) -- never the raw
    checkpointer session id, which this function's own signature has
    no parameter for at all. Raises `R6D4CaptureError` immediately if
    it looks like a raw session/version identifier instead (see
    `_validate_source_session_ref`).

    `now`, if given, must already be a parseable, timezone-aware ISO
    8601 string -- used directly as `capture_timestamp`. Defaults to
    `datetime.now(timezone.utc).isoformat()` (matching `report.py`'s
    own `_utcnow_iso()` convention, reimplemented independently here
    rather than importing that private helper).

    `source_commit_sha` is never resolved automatically (no `git`
    subprocess call from an eval-only module) -- pass it explicitly, or
    leave it `None` and let a later stage (R6D.4b's own CLI) fill it
    in.

    Never catches an exception raised by the real R4 calls
    (generation, R4's own evaluation, or R4's own optional revision) --
    a failure at any of those three points propagates straight to the
    caller, and no artifact is ever returned in that case (see this
    module's own "generation/evaluation/revision failure produces no
    artifact" test coverage).

    Raises `R6D4CaptureError` (never silently produces a bad artifact)
    if `now` is malformed, if `source_session_ref` looks like a raw
    identifier, or if R4's own reported `rounds` contradicts whether
    the draft/refined report bodies actually differ -- see `_check_
    revision_consistency`.
    """
    _validate_source_session_ref(source_session_ref)
    capture_timestamp = now if now is not None else datetime.now(timezone.utc).isoformat()
    _validate_capture_timestamp(capture_timestamp)

    draft = report_module.generate_report_for_session(session, client=client, report_template=report_template)
    # Step 2: MUST happen before `draft` is ever handed to refinement -- the
    # no-revision branch mutates its own `draft` argument in place. `_strip_
    # report_body_for_capture` deep-copies internally, so `draft` itself is
    # still completely untouched by this call.
    draft_snapshot = _strip_report_body_for_capture(draft)

    web_articles: list[Any] = []  # matches get_or_create_report's own hardcoded initial-Generate call exactly
    final = report_module.refine_report_if_requested(
        draft, session.topic, session.selected_papers, web_articles,
        draft.get("report_template", report_template),
        refinement_mode=REQUIRED_REFINEMENT_MODE, client=client,
    )

    # Read (never pop) R4's own refinement metadata from `final` itself, BEFORE
    # stripping -- `_strip_report_body_for_capture` drops "refinement" from its
    # own returned copy, so this is the only point this value is ever available.
    r4_refinement_metadata = final.get("refinement")
    if r4_refinement_metadata is None:
        raise R6D4CaptureError(
            "refine_report_if_requested did not stamp refinement metadata -- "
            f"expected refinement_mode={REQUIRED_REFINEMENT_MODE!r} to always produce one"
        )
    r4_refinement_metadata = copy.deepcopy(r4_refinement_metadata)  # never share mutable state with final_snapshot

    final_snapshot = _strip_report_body_for_capture(final)

    rounds = r4_refinement_metadata.get("rounds")
    revision_applied = bool(rounds and rounds > 0)
    bodies_equal = draft_snapshot == final_snapshot
    _check_revision_consistency(revision_applied, bodies_equal)

    selected_papers = [p.to_dict() for p in session.selected_papers]
    approved_web_articles: list[dict[str, Any]] = []  # see web_articles above -- always [] on this path

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": pair_id,
        "topic": session.topic,
        "template": report_template,
        "tags": list(tags) if tags is not None else [],
        "notes": notes,
        "selected_papers": selected_papers,
        "approved_web_articles": approved_web_articles,
        "draft_report": draft_snapshot,
        "refined_report": final_snapshot,
        "refinement_context": {
            "source_origin": REQUIRED_SOURCE_ORIGIN,
            "revision_applied": revision_applied,
            "capture_timestamp": capture_timestamp,
            "source_session_ref": source_session_ref,
            "generation_model": report_module.REPORT_MODEL,
            "refinement_mode": REQUIRED_REFINEMENT_MODE,
            "r4_refinement_metadata": r4_refinement_metadata,
            "capture_commit_sha": source_commit_sha,
        },
    }

    forbidden = find_forbidden_keys(artifact)
    if forbidden:
        raise R6D4CaptureError(f"captured artifact contains forbidden key(s): {forbidden}")

    return artifact


def validate_r6d4_capture(artifact: dict[str, Any]) -> None:
    """Independent, pure validator for an already-built `r6d4-
    capture-v1` artifact dict (freshly returned by `capture_real_
    refinement_pair`, or reloaded from a file R6D.4b will later write).
    Raises `R6D4CaptureError` on the first violation found; never
    silently repairs or drops a bad artifact. Deliberately separate
    from `report_refinement_inputs.validate_pair` -- see this module's
    own `SCHEMA_VERSION` docstring for why the two must never be
    conflated."""
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise R6D4CaptureError(f"schema_version must be {SCHEMA_VERSION!r}, got {artifact.get('schema_version')!r}")
    if not artifact.get("id") or not str(artifact["id"]).strip():
        raise R6D4CaptureError("id must be a non-empty string")
    if not artifact.get("topic") or not str(artifact["topic"]).strip():
        raise R6D4CaptureError("topic must be a non-empty string")
    template = artifact.get("template")
    if template not in VALID_TEMPLATES:
        raise R6D4CaptureError(f"template={template!r} not one of {VALID_TEMPLATES}")

    for forbidden_key in _FORBIDDEN_TOP_LEVEL_KEYS:
        if forbidden_key in artifact:
            raise R6D4CaptureError(f"unlabelled capture artifact must not contain top-level key {forbidden_key!r}")

    forbidden = find_forbidden_keys(artifact)
    if forbidden:
        raise R6D4CaptureError(f"artifact contains forbidden key(s): {forbidden}")

    refinement_context = artifact.get("refinement_context") or {}
    if refinement_context.get("source_origin") != REQUIRED_SOURCE_ORIGIN:
        raise R6D4CaptureError(f"refinement_context.source_origin must be {REQUIRED_SOURCE_ORIGIN!r}")
    if refinement_context.get("refinement_mode") != REQUIRED_REFINEMENT_MODE:
        raise R6D4CaptureError(f"refinement_context.refinement_mode must be {REQUIRED_REFINEMENT_MODE!r}")
    if "r4_refinement_metadata" not in refinement_context or refinement_context["r4_refinement_metadata"] is None:
        raise R6D4CaptureError("refinement_context.r4_refinement_metadata is required")

    _validate_source_session_ref(refinement_context.get("source_session_ref") or "")
    _validate_capture_timestamp(refinement_context.get("capture_timestamp") or "")

    revision_applied = refinement_context.get("revision_applied")
    if not isinstance(revision_applied, bool):
        raise R6D4CaptureError("refinement_context.revision_applied must be a bool")

    draft_report = artifact.get("draft_report") or {}
    refined_report = artifact.get("refined_report") or {}
    if "refinement" in draft_report:
        raise R6D4CaptureError("draft_report must not contain a top-level 'refinement' key")
    if "refinement" in refined_report:
        raise R6D4CaptureError("refined_report must not contain a top-level 'refinement' key")

    for side, report in (("draft", draft_report), ("refined", refined_report)):
        if report.get("report_template") != template:
            raise R6D4CaptureError(f"{side}_report.report_template={report.get('report_template')!r} != template={template!r}")
        _check_section_shape(report, side)

    selected_papers = artifact.get("selected_papers") or []
    approved_web_articles = artifact.get("approved_web_articles") or []
    if not isinstance(selected_papers, list) or not selected_papers:
        raise R6D4CaptureError("selected_papers must be a non-empty list")
    if not isinstance(approved_web_articles, list):
        raise R6D4CaptureError("approved_web_articles must be a list")
    _check_references_resolve(draft_report, selected_papers, approved_web_articles, "draft")
    _check_references_resolve(refined_report, selected_papers, approved_web_articles, "refined")

    bodies_equal = draft_report == refined_report
    _check_revision_consistency(revision_applied, bodies_equal)
