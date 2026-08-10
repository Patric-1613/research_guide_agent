"""R6D.1: manifest + fixture loading and pair-invariant validation for
the report-refinement-effectiveness benchmark. This module provides
**schema/loading only** -- no live judges, no OpenAI/API calls of any
kind, no CLI suite registration, and no call into `research_agent.
report`'s generation/evaluation/revision functions. R6D.2 (not built
here) is responsible for actually running deterministic and/or live
judge dimensions against the pairs this module loads.

R6C's own independence decision applies here too: this module never
imports `research_agent.report` and never imports `research_agent.
evals.runners.run_report_quality` or `research_agent.evals.report_
quality_inputs` -- the frozen section keys, hard-failure identifiers,
and dimension names below are independent copies (same reasoning
`report_quality_inputs.py`'s own module docstring gives: a future
refactor of one suite must never silently change what a different
suite validates against). R6D.1 reuses the *shape* R6C already
validated (the real stored report-dict shape, confirmed against
`research_agent/curation_session.py::_serialize_report`), not R6C's
code.

A "pair" fixture holds one topic/template/evidence set with TWO
report bodies -- `draft_report` (before refinement) and
`refined_report` (after) -- plus a synthetic, hand-constructed
`expected.dimension_directions` block recording, per R6C's 7 frozen
judge dimensions, whether that dimension moved `improved`/
`unchanged`/`regressed`/`unknown` between the two. **No overall
score, weighted composite, or winner field exists anywhere in this
schema** -- R6D explicitly measures direction per dimension, never a
single number, per `specs/report-quality-evaluation-plan.md` section
8's own frozen requirement.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_agent.evals.runners._base import EVAL_DATA_DIR

SCHEMA_VERSION = "r6d1-v1"

REPORT_REFINEMENT_DIR = EVAL_DATA_DIR / "report_refinement"
MANIFEST_PATH = REPORT_REFINEMENT_DIR / "manifest.jsonl"
FIXTURES_DIR = REPORT_REFINEMENT_DIR / "fixtures"

# The only fixture shape this loader currently accepts -- a fixture
# declaring any other value is rejected at load time, never silently
# coerced (same posture as run_report_quality.py's own
# SUPPORTED_FIXTURE_SCHEMA_VERSIONS).
SUPPORTED_FIXTURE_SCHEMA_VERSIONS = {SCHEMA_VERSION}

VALID_TEMPLATES = ("foundational", "analytical", "expert")

# The 7 R6C judge dimensions, independent copy of run_report_quality.py's
# REQUIRED_DIMENSION_NAMES -- R6D measures direction on exactly these,
# never an 8th "overall" dimension.
REQUIRED_DIMENSION_NAMES = (
    "citation_correctness", "groundedness", "synthesis_quality", "analytical_quality",
    "template_fit", "coherence", "source_balance",
)

# The only 4 values a direction may take, everywhere in this schema
# (hard_failure_direction and every per-dimension direction). "unknown"
# is reserved for a genuinely unjudgeable comparison (e.g. one side of
# the pair is structurally broken, gating off live judgment entirely,
# per R6C's own "hard failure -> unknown, never not_applicable"
# convention) -- never a stand-in for sloppy fixture authoring. This
# module cannot mechanically tell WHY a fixture author chose "unknown"
# for a given dimension; that judgment is a fixture-authoring
# discipline documented here and in eval_data/report_refinement/
# README.md, not something `validate_pair` can enforce on its own.
VALID_DIRECTIONS = ("improved", "unchanged", "regressed", "unknown")

# Independent copy of report_quality_inputs.py/run_report_quality.py's
# REQUIRED_SECTION_KEYS -- all three templates share this exact key
# set; only per-template prose instructions differ.
REQUIRED_SECTION_KEYS = (
    "executive_summary", "introduction_scope", "thematic_findings",
    "methodology_landscape", "contradictions_open_debates", "gap_analysis",
    "future_research_directions", "conclusion",
)

# Frozen order from specs/report-quality-evaluation-plan.md section 2,
# independent copy of run_report_quality.py's CANONICAL_HARD_FAILURE_
# ORDER -- check_structural_validity below walks this tuple in order
# so a returned hard-failure list is deterministic regardless of
# dict/set iteration order.
CANONICAL_HARD_FAILURE_ORDER = (
    "missing_required_section",
    "empty_required_section",
    "unresolved_citation_marker",
    "non_sequential_reference_numbering",
    "orphan_reference",
    "reference_source_unavailable",
)

_PAPER_WEB_MARKER_RE = re.compile(r"\[\s*(?:Paper|Web)\s+\d+(?:\s*,\s*(?:Paper|Web)\s+\d+)*\s*\]")
_SINGLE_TOKEN_BRACKET_RE = re.compile(r"\[([^\s\[\]]+)\]")
_FINAL_NUMERIC_MARKER_RE = re.compile(r"\[(\d+)\]")


class ReportRefinementFixtureError(ValueError):
    """Raised for any manifest/fixture data-integrity problem at load
    time -- a missing fixture path, a duplicate manifest id, a path
    escaping eval_data/report_refinement/, an unsupported schema_
    version, a template mismatch between the pair and either report, a
    malformed/incomplete expected.dimension_directions block, a
    revision_applied/report-equality inconsistency, or a fixture
    expectation string leaking into judge-ready report content.
    Deliberately loud and immediate -- a malformed pair fixture must
    never silently degrade into a shorter/wrong example set."""


@dataclass(frozen=True)
class RefinementPairExample:
    """One fully-loaded, validated pair fixture, ready for R6D.2 to
    consume. All mutable fields are deep copies made at load time --
    mutating a returned example never affects this module's own
    parsed fixture data, and reloading always returns fresh objects
    (see `test_loader_never_mutates_source_fixture_dicts`)."""

    id: str
    topic: str
    template: str
    selected_papers: list[dict[str, Any]]
    approved_web_articles: list[dict[str, Any]]
    draft_report: dict[str, Any]
    refined_report: dict[str, Any]
    refinement_context: dict[str, Any]
    expected: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    source_origin: str = ""
    notes: str = ""


# --- Manifest + fixture loading -----------------------------------------

def _load_manifest_rows() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"report_refinement manifest not found: {MANIFEST_PATH}")

    rows: list[dict[str, Any]] = []
    with MANIFEST_PATH.open() as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReportRefinementFixtureError(f"{MANIFEST_PATH}:{line_num}: invalid JSON: {exc}") from exc
            for required_key in ("id", "path", "tags", "source_origin"):
                if required_key not in row:
                    raise ReportRefinementFixtureError(
                        f"{MANIFEST_PATH}:{line_num}: manifest row missing required key {required_key!r}"
                    )
            rows.append(row)

    seen_ids: set[str] = set()
    for row in rows:
        if row["id"] in seen_ids:
            raise ReportRefinementFixtureError(f"duplicate fixture id in manifest: {row['id']!r}")
        seen_ids.add(row["id"])

    seen_paths: set[str] = set()
    for row in rows:
        if row["path"] in seen_paths:
            raise ReportRefinementFixtureError(f"duplicate fixture path in manifest: {row['path']!r}")
        seen_paths.add(row["path"])

    return rows


def _resolve_fixture_path(raw_path: str) -> Path:
    """Resolves `raw_path` strictly beneath eval_data/report_refinement/
    -- rejects anything (e.g. a `../` traversal) that would resolve
    outside that root, independent of whether the underlying OS would
    actually permit the read."""
    root = REPORT_REFINEMENT_DIR.resolve()
    resolved = (REPORT_REFINEMENT_DIR / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ReportRefinementFixtureError(
            f"fixture path escapes eval_data/report_refinement/: {raw_path!r}"
        ) from None
    return resolved


def _load_fixture_json(row: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_fixture_path(row["path"])
    if not path.exists():
        raise ReportRefinementFixtureError(f"fixture id={row['id']!r}: path does not exist: {path}")

    try:
        fixture = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ReportRefinementFixtureError(f"{path}: invalid JSON: {exc}") from exc

    if fixture.get("id") != row["id"]:
        raise ReportRefinementFixtureError(
            f"{path}: fixture id {fixture.get('id')!r} does not match manifest id {row['id']!r}"
        )

    schema_version = fixture.get("schema_version")
    if schema_version not in SUPPORTED_FIXTURE_SCHEMA_VERSIONS:
        raise ReportRefinementFixtureError(
            f"fixture id={row['id']!r} ({path}): unsupported schema_version {schema_version!r} -- "
            f"supported: {sorted(SUPPORTED_FIXTURE_SCHEMA_VERSIONS)}"
        )

    return fixture


# --- Structural validity (independent of R6C's own checks) --------------

def check_structural_validity(
    report: dict[str, Any], selected_papers: list[dict[str, Any]], approved_web_articles: list[dict[str, Any]],
) -> list[str]:
    """Returns the (possibly empty) list of R6A/R6B's 6 frozen hard-
    failure identifiers this report exhibits, in CANONICAL_HARD_
    FAILURE_ORDER. A fresh, independent implementation (same posture
    as report_quality_inputs.py's own injection detector) -- never
    imports run_report_quality.py's checks. Exported for both
    `validate_pair` (below) and any test/tool that wants to check a
    single report body directly."""
    missing = [k for k in REQUIRED_SECTION_KEYS if k not in report]
    empty = [k for k in REQUIRED_SECTION_KEYS if k in report and not (report[k].get("content") or "").strip()]

    known_paper_ids = {p["paper_id"] for p in selected_papers if p.get("paper_id")}
    known_urls = {a["url"] for a in approved_web_articles if a.get("url")}

    unresolved_by_section: dict[str, list[str]] = {}
    cited_in_prose: set[int] = set()
    for key in REQUIRED_SECTION_KEYS:
        if key not in report:
            continue
        content = report[key].get("content") or ""
        matches = [m.group() for m in _PAPER_WEB_MARKER_RE.finditer(content)]
        for m in _SINGLE_TOKEN_BRACKET_RE.finditer(content):
            token = m.group(1)
            if token.isdigit():
                continue
            if token in known_paper_ids or token in known_urls:
                matches.append(m.group())
        if matches:
            unresolved_by_section[key] = matches
        cited_in_prose.update(int(m.group(1)) for m in _FINAL_NUMERIC_MARKER_RE.finditer(content))

    references = report.get("references") or []
    numbers = sorted(r["number"] for r in references)
    non_sequential = bool(numbers) and numbers != list(range(1, len(numbers) + 1))

    orphans = [r["number"] for r in references if r["number"] not in cited_in_prose]

    unavailable: list[int] = []
    for r in references:
        kind = r.get("kind")
        if kind == "paper":
            if not r.get("paper_id") or r["paper_id"] not in known_paper_ids:
                unavailable.append(r["number"])
        elif kind == "web":
            if not r.get("url") or r["url"] not in known_urls:
                unavailable.append(r["number"])
        else:
            unavailable.append(r["number"])

    detected = {
        "missing_required_section": bool(missing),
        "empty_required_section": bool(empty),
        "unresolved_citation_marker": bool(unresolved_by_section),
        "non_sequential_reference_numbering": non_sequential,
        "orphan_reference": bool(orphans),
        "reference_source_unavailable": bool(unavailable),
    }
    return [identifier for identifier in CANONICAL_HARD_FAILURE_ORDER if detected[identifier]]


# --- Deterministic report-content equality/difference ------------------

def reports_are_equal(report_a: dict[str, Any], report_b: dict[str, Any]) -> bool:
    """Deep-equality over the two report dicts. Deterministic (plain
    `==` over already-parsed JSON data -- dict comparison in Python is
    order-independent and value-based, so key order never affects the
    result)."""
    return report_a == report_b


def diff_report_sections(report_a: dict[str, Any], report_b: dict[str, Any]) -> list[str]:
    """Returns the section keys (from REQUIRED_SECTION_KEYS, in
    canonical order) whose `content` differs between the two reports.
    A pure content diff -- does not compare `reference_numbers` or any
    other per-section field, since `content` is what a judge actually
    reads (same "prose is what's real" posture report_quality_inputs.
    py's own claim extraction uses)."""
    differing: list[str] = []
    for key in REQUIRED_SECTION_KEYS:
        content_a = (report_a.get(key) or {}).get("content")
        content_b = (report_b.get(key) or {}).get("content")
        if content_a != content_b:
            differing.append(key)
    return differing


# --- Pair-invariant validation ------------------------------------------

def _require(condition: bool, fixture_id: str, message: str) -> None:
    if not condition:
        raise ReportRefinementFixtureError(f"fixture id={fixture_id!r}: {message}")


def _validate_report_shape(report: dict[str, Any], fixture_id: str, side: str, template: str) -> None:
    _require(
        report.get("report_template") == template, fixture_id,
        f"{side}_report.report_template={report.get('report_template')!r} != pair template={template!r}",
    )
    for key in ("selected_papers", "approved_web_articles"):
        _require(
            key not in report, fixture_id,
            f"{side}_report must not embed its own {key!r} -- shared once at pair level only (invariant 6)",
        )
    missing_keys = [k for k in REQUIRED_SECTION_KEYS if k not in report]
    _require(not missing_keys, fixture_id, f"{side}_report missing section key(s): {missing_keys}")

    sections_list = report.get("sections") or []
    section_list_keys = [s.get("key") for s in sections_list]
    _require(
        section_list_keys == list(REQUIRED_SECTION_KEYS), fixture_id,
        f"{side}_report.sections key order {section_list_keys} != canonical order {list(REQUIRED_SECTION_KEYS)}",
    )
    for key in REQUIRED_SECTION_KEYS:
        section_content = (report.get(key) or {}).get("content")
        list_entry = next((s for s in sections_list if s.get("key") == key), None)
        _require(
            list_entry is not None and list_entry.get("content") == section_content, fixture_id,
            f"{side}_report.sections[{key!r}].content does not mirror {side}_report[{key!r}].content",
        )


def _validate_expected_directions(expected: dict[str, Any], fixture_id: str) -> None:
    hard_failure_direction = expected.get("hard_failure_direction")
    _require(
        hard_failure_direction in VALID_DIRECTIONS, fixture_id,
        f"expected.hard_failure_direction={hard_failure_direction!r} not one of {VALID_DIRECTIONS}",
    )

    dimension_directions = expected.get("dimension_directions") or {}
    present = set(dimension_directions)
    required = set(REQUIRED_DIMENSION_NAMES)
    missing_dims = required - present
    extra_dims = present - required
    _require(not missing_dims, fixture_id, f"expected.dimension_directions missing dimension(s): {sorted(missing_dims)}")
    _require(not extra_dims, fixture_id, f"expected.dimension_directions has unknown dimension(s): {sorted(extra_dims)}")

    for dim_name in REQUIRED_DIMENSION_NAMES:
        entry = dimension_directions.get(dim_name) or {}
        direction = entry.get("direction")
        rationale = (entry.get("rationale") or "").strip()
        _require(
            direction in VALID_DIRECTIONS, fixture_id,
            f"dimension {dim_name!r} direction={direction!r} not one of {VALID_DIRECTIONS}",
        )
        _require(bool(rationale), fixture_id, f"dimension {dim_name!r} has an empty rationale")

    # No overall_direction/overall_score/accept_refinement/winner field
    # is ever permitted -- those decisions require later calibration
    # (specs/report-quality-evaluation-plan.md section 8).
    forbidden_keys = {"overall_direction", "overall_score", "accept_refinement", "winner"}
    present_forbidden = forbidden_keys & set(expected)
    _require(not present_forbidden, fixture_id, f"expected block must not contain: {sorted(present_forbidden)}")


def _validate_no_expectation_leakage(fixture: dict[str, Any], fixture_id: str) -> None:
    """Invariant 14: a fixture's own expected-direction rationale text
    must never appear verbatim inside either report's judge-ready
    content -- otherwise a future judge reading the report would be
    reading the answer key, not the report."""
    rationale_strings = [
        entry.get("rationale", "")
        for entry in (fixture.get("expected", {}).get("dimension_directions") or {}).values()
        if (entry.get("rationale") or "").strip()
    ]
    for side in ("draft_report", "refined_report"):
        report = fixture.get(side) or {}
        for key in REQUIRED_SECTION_KEYS:
            content = (report.get(key) or {}).get("content") or ""
            for rationale in rationale_strings:
                _require(
                    rationale not in content, fixture_id,
                    f"{side}[{key!r}] leaks an expected-direction rationale string verbatim: {rationale[:60]!r}...",
                )


def validate_pair(fixture: dict[str, Any]) -> None:
    """Enforces all 14 R6D.1 pair invariants against an already-parsed
    fixture dict. Raises `ReportRefinementFixtureError` on the first
    violation found; never silently repairs or drops a bad fixture."""
    fixture_id = fixture.get("id", "<unknown>")
    template = fixture.get("template")
    _require(template in VALID_TEMPLATES, fixture_id, f"template={template!r} not one of {VALID_TEMPLATES}")

    draft = fixture.get("draft_report") or {}
    refined = fixture.get("refined_report") or {}
    _validate_report_shape(draft, fixture_id, "draft", template)
    _validate_report_shape(refined, fixture_id, "refined", template)

    selected_papers = fixture.get("selected_papers") or []
    approved_web_articles = fixture.get("approved_web_articles") or []
    _require(isinstance(selected_papers, list) and selected_papers, fixture_id, "selected_papers must be a non-empty list")
    _require(isinstance(approved_web_articles, list), fixture_id, "approved_web_articles must be a list")

    refinement_context = fixture.get("refinement_context") or {}
    revision_applied = refinement_context.get("revision_applied")
    _require(isinstance(revision_applied, bool), fixture_id, "refinement_context.revision_applied must be a bool")

    are_equal = reports_are_equal(draft, refined)
    if revision_applied is False:
        _require(are_equal, fixture_id, "revision_applied=false requires draft_report == refined_report exactly (invariant 12)")
    else:
        _require(not are_equal, fixture_id, "revision_applied=true requires some report-body difference (invariant 13)")

    expected = fixture.get("expected") or {}
    _validate_expected_directions(expected, fixture_id)
    _validate_no_expectation_leakage(fixture, fixture_id)

    draft_hard_failures = check_structural_validity(draft, selected_papers, approved_web_articles)
    refined_hard_failures = check_structural_validity(refined, selected_papers, approved_web_articles)
    hard_failure_direction = expected.get("hard_failure_direction")
    if hard_failure_direction == "unchanged":
        _require(
            draft_hard_failures == refined_hard_failures, fixture_id,
            f"hard_failure_direction=unchanged but draft={draft_hard_failures} != refined={refined_hard_failures}",
        )
    elif hard_failure_direction == "regressed":
        _require(not draft_hard_failures, fixture_id, f"hard_failure_direction=regressed but draft already has hard failures: {draft_hard_failures}")
        _require(bool(refined_hard_failures), fixture_id, "hard_failure_direction=regressed but refined has no hard failures")
    elif hard_failure_direction == "improved":
        _require(bool(draft_hard_failures), fixture_id, "hard_failure_direction=improved but draft has no hard failures")
        _require(not refined_hard_failures, fixture_id, f"hard_failure_direction=improved but refined still has hard failures: {refined_hard_failures}")
    # hard_failure_direction == "unknown": no fixed structural relationship enforced --
    # reserved for genuinely unjudgeable pairs, none of which R6D.1's own 7 fixtures use.


# --- Top-level loader -----------------------------------------------------

def load_report_refinement_examples(
    tags: list[str] | None = None, subset: int | None = None,
) -> list[RefinementPairExample]:
    """Reads manifest.jsonl, applies tags/subset filtering using the
    manifest's own metadata (same OR-semantics/first-N convention
    `run_report_quality.py::load_report_quality_examples` already
    uses), loads and validates every selected fixture, and returns
    fresh `RefinementPairExample` objects. Never mutates the parsed
    fixture dict it validates -- every field on the returned example
    is a deep copy."""
    rows = _load_manifest_rows()
    if tags:
        rows = [r for r in rows if set(r.get("tags") or []) & set(tags)]
    if subset is not None:
        rows = rows[:subset]

    examples: list[RefinementPairExample] = []
    for row in rows:
        fixture = _load_fixture_json(row)
        validate_pair(fixture)
        examples.append(RefinementPairExample(
            id=fixture["id"],
            topic=fixture.get("topic", ""),
            template=fixture["template"],
            selected_papers=copy.deepcopy(fixture["selected_papers"]),
            approved_web_articles=copy.deepcopy(fixture["approved_web_articles"]),
            draft_report=copy.deepcopy(fixture["draft_report"]),
            refined_report=copy.deepcopy(fixture["refined_report"]),
            refinement_context=copy.deepcopy(fixture.get("refinement_context") or {}),
            expected=copy.deepcopy(fixture.get("expected") or {}),
            tags=list(row.get("tags") or []),
            source_origin=row.get("source_origin", ""),
            notes=row.get("notes", ""),
        ))
    return examples
