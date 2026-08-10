"""Runner for the report_quality suite -- R6B's deterministic/mock
checks plus R6C.2's opt-in live judges.

R6A decision 1 (specs/report-quality-evaluation-plan.md): R6 is
independent from R4 -- this module never calls research_agent.report's
generate_report/evaluate_report/revise_report, and never reuses R4's
own `_deterministic_report_checks` (research_agent/report.py), even
though the two check similar things -- R6's checks are intentionally
finer-grained (6 separately-identified failure modes vs. R4's 4
combined ones) and cover one condition R4's live generation path can
never itself produce (reference_source_unavailable, only reachable by
a stored/regressed report dict). REQUIRED_SECTION_KEYS below is a
deliberately independent, hardcoded copy of the same 8 keys
research_agent/report.py's REPORT_SECTION_DEFINITIONS defines -- not
imported from it -- so a future report.py refactor can never silently
change what R6B checks without R6B's own tests catching the drift.

Fixtures live under eval_data/report_quality/ as a manifest
(manifest.jsonl) + individual JSON files (fixtures/*.json), not one
flat JSONL the way chat_relevance's fixtures do -- a report-quality
fixture embeds a full 8-section report plus full paper abstracts/web
snippets, too large to stay reviewable on one JSONL line (see
eval_data/report_quality/README.md). load_report_quality_examples
below is this suite's own thin loader, producing the same runners/
_base.py::Example shape chat_relevance's load_examples produces, so
run_suite's predict -> evaluate -> aggregate loop (runners/_base.py)
can run it completely unmodified via that function's `examples=`
parameter -- this module never forks that loop.

R6C.2: `predict_live` layers two independent, opt-in live judges
(research_agent/evals/judges/claim_source.py,
research_agent/evals/judges/holistic.py) on top of R6C.1's own
deterministic preparation (research_agent/evals/report_quality_
inputs.py) -- this module orchestrates (which judges run, how their
structured output maps onto the 7 frozen R6 dimensions) but never
builds a prompt or calls the OpenAI SDK directly itself; that stays in
the judges/ package. `predict()` (R6B, mock) is completely unchanged
by any of this -- `predict_live` calls it internally for the
structural baseline every prediction (mock or live) shares.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError

from research_agent.evals import report_quality_inputs
from research_agent.evals.evaluators.report_quality import ALL_EVALUATORS
from research_agent.evals.judges import claim_source, holistic
from research_agent.evals.runners._base import EVAL_DATA_DIR, Example, LiveModeSetupError, SuiteResult, run_suite

SUITE = "report_quality"

# R6C.2: env-configurable so a deployment/operator can point at a
# different model without a code change -- never silently substituted
# at call time regardless of what's configured (see _build_live_client
# below for the one pre-flight check this module performs: the
# configured value must be non-empty). Deliberately a DIFFERENT model
# family from research_agent.report.REPORT_MODEL ("gpt-4.1", the
# production report generator) -- same reasoning R6A/R6C's own design
# doc gives for judge independence: a judge sharing a model with the
# thing it grades is not independent evidence against self-evaluation
# bias. This module never imports research_agent.report at all (R6A
# decision 1's own "independent from R4" boundary), so this constant is
# an independently-chosen literal, not derived from or compared against
# REPORT_MODEL in code.
REPORT_QUALITY_JUDGE_MODEL = os.environ.get("REPORT_QUALITY_JUDGE_MODEL", "gpt-5.6-terra")

# R6C.2c: identifies the RULE VERSION _aggregate_claim_source_dimensions
# implements, independent of the claim/source judge's own PROMPT version
# (CLAIM_SOURCE_JUDGE_PROMPT_VERSION) -- the same judge output can be
# aggregated by different rule versions over time, so a reader of a run's
# judge_metadata must be able to tell which aggregation semantics produced
# a given citation_correctness/groundedness label without cross-referencing
# code history. Bumped from an unversioned v1 (the R6C.2/R6C.2a "any
# partially_supports source verdict fails citation_correctness" rule) to
# v2 (this module's current "citation correctness is per-source relevance;
# groundedness is complete-claim support" split -- see
# _aggregate_claim_source_dimensions's own docstring).
CITATION_AGGREGATION_POLICY_VERSION = "r6c2-citation-aggregation-v2"

REPORT_QUALITY_DIR = EVAL_DATA_DIR / "report_quality"
MANIFEST_PATH = REPORT_QUALITY_DIR / "manifest.jsonl"
FIXTURES_DIR = REPORT_QUALITY_DIR / "fixtures"

# Frozen in specs/report-quality-evaluation-plan.md section 1 -- the
# only fixture shape this loader currently accepts. A fixture declaring
# any other value is rejected at load time (see _load_fixture) rather
# than silently coerced.
SUPPORTED_FIXTURE_SCHEMA_VERSIONS = {"r6a-v1"}

# Independent copy of research_agent/report.py's 8 Analytical section
# keys (REPORT_SECTION_DEFINITIONS) -- see module docstring for why
# this is hardcoded here rather than imported. All three templates
# (foundational/analytical/expert) share this exact key set; only
# per-template prose instructions differ, so no template-specific
# variant of this tuple is needed.
REQUIRED_SECTION_KEYS = (
    "executive_summary", "introduction_scope", "thematic_findings",
    "methodology_landscape", "contradictions_open_debates", "gap_analysis",
    "future_research_directions", "conclusion",
)

# The 7 judge dimensions frozen in specs/report-quality-evaluation-plan.md
# section 1 -- used only to validate a fixture's expected.dimension_labels
# is complete at load time; R6B never scores against these (see
# evaluators/report_quality.py's own docstring).
REQUIRED_DIMENSION_NAMES = (
    "citation_correctness", "groundedness", "synthesis_quality", "analytical_quality",
    "template_fit", "coherence", "source_balance",
)

# Frozen order from specs/report-quality-evaluation-plan.md section 2 --
# every hard_failures list this module produces is built by walking
# this tuple in order and keeping only the identifiers actually
# detected, so ordering is deterministic regardless of dict/set
# iteration order.
CANONICAL_HARD_FAILURE_ORDER = (
    "missing_required_section",
    "empty_required_section",
    "unresolved_citation_marker",
    "non_sequential_reference_numbering",
    "orphan_reference",
    "reference_source_unavailable",
)

# A raw, never-densified "[Paper N]"/"[Web N]" marker (see research_
# agent/report.py's own _SECTION_CITATION_MARKER_RE) -- also matches a
# grouped form like "[Paper 6, Paper 8]". Always a defect if present in
# FINAL rendered content; a report that never reached the marker ->
# reference-number resolution pass leaves these behind verbatim.
_PAPER_WEB_MARKER_RE = re.compile(r"\[\s*(?:Paper|Web)\s+\d+(?:\s*,\s*(?:Paper|Web)\s+\d+)*\s*\]")

# A single bracket whose content has no internal whitespace -- the same
# shape a real paper_id/url always has, and the shape research_agent/
# report.py's own _RAW_SOURCE_ID_MARKER_RE targets ("a raw identifier
# leaked into the report body instead of resolving to [N]"). A bare
# digit string is handled separately below (always the valid FINAL
# marker form, never flagged).
_SINGLE_TOKEN_BRACKET_RE = re.compile(r"\[([^\s\[\]]+)\]")

# The only valid FINAL citation marker shape -- a bracket containing
# nothing but digits. Used both to detect "already resolved, never
# flag" and to find which reference numbers are actually visible in
# rendered prose (orphan_reference's own definition below).
_FINAL_NUMERIC_MARKER_RE = re.compile(r"\[(\d+)\]")

_WORD_RE = re.compile(r"\S+")


class FixtureLoadError(ValueError):
    """Raised for any manifest/fixture data-integrity problem at load
    time -- a missing fixture path, a duplicate manifest id, a path
    escaping eval_data/report_quality/, an unsupported schema_version,
    or a manifest/fixture identity inconsistency (e.g. a fixture's own
    top-level `template` disagreeing with its `generated_report.
    report_template`, or an incomplete expected.dimension_labels
    block). Deliberately loud and immediate -- a malformed fixture
    should never silently degrade into a shorter/wrong example set."""


# --- Manifest + fixture loading ---------------------------------------

def _load_manifest_rows() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"report_quality manifest not found: {MANIFEST_PATH}")

    rows: list[dict[str, Any]] = []
    with MANIFEST_PATH.open() as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FixtureLoadError(f"{MANIFEST_PATH}:{line_num}: invalid JSON: {exc}") from exc
            for required_key in ("id", "path", "tags", "source_origin"):
                if required_key not in row:
                    raise FixtureLoadError(
                        f"{MANIFEST_PATH}:{line_num}: manifest row missing required key {required_key!r}"
                    )
            rows.append(row)

    seen_ids: set[str] = set()
    for row in rows:
        if row["id"] in seen_ids:
            raise FixtureLoadError(f"duplicate fixture id in manifest: {row['id']!r}")
        seen_ids.add(row["id"])

    return rows


def _resolve_fixture_path(raw_path: str) -> Path:
    """Resolves `raw_path` strictly beneath eval_data/report_quality/ --
    rejects anything (e.g. a `../` traversal) that would resolve
    outside that root, independent of whether the underlying OS would
    actually permit the read."""
    root = REPORT_QUALITY_DIR.resolve()
    resolved = (REPORT_QUALITY_DIR / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise FixtureLoadError(f"fixture path escapes eval_data/report_quality/: {raw_path!r}") from None
    return resolved


def _load_fixture(row: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_fixture_path(row["path"])
    if not path.exists():
        raise FixtureLoadError(f"fixture id={row['id']!r}: path does not exist: {path}")

    try:
        fixture = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise FixtureLoadError(f"{path}: invalid JSON: {exc}") from exc

    schema_version = fixture.get("schema_version")
    if schema_version not in SUPPORTED_FIXTURE_SCHEMA_VERSIONS:
        raise FixtureLoadError(
            f"fixture id={row['id']!r} ({path}): unsupported schema_version {schema_version!r} -- "
            f"supported: {sorted(SUPPORTED_FIXTURE_SCHEMA_VERSIONS)}"
        )

    report_template = (fixture.get("generated_report") or {}).get("report_template")
    if fixture.get("template") != report_template:
        raise FixtureLoadError(
            f"fixture id={row['id']!r} ({path}): manifest/fixture identity inconsistency -- "
            f"top-level template={fixture.get('template')!r} != "
            f"generated_report.report_template={report_template!r}"
        )

    expected = fixture.get("expected") or {}
    if "hard_failures" not in expected or not isinstance(expected["hard_failures"], list):
        raise FixtureLoadError(f"fixture id={row['id']!r} ({path}): expected.hard_failures missing or not a list")

    dimension_labels = expected.get("dimension_labels") or {}
    missing_dims = set(REQUIRED_DIMENSION_NAMES) - set(dimension_labels)
    if missing_dims:
        raise FixtureLoadError(
            f"fixture id={row['id']!r} ({path}): expected.dimension_labels missing dimension(s): "
            f"{sorted(missing_dims)}"
        )
    for dim_name, dim in dimension_labels.items():
        if "label" not in dim or not (dim.get("rationale") or "").strip():
            raise FixtureLoadError(
                f"fixture id={row['id']!r} ({path}): dimension {dim_name!r} missing label or rationale"
            )

    return fixture


def load_report_quality_examples(subset: int | None = None, tags: list[str] | None = None) -> list[Example]:
    """Reads manifest.jsonl, applies tags/subset using the manifest's
    OWN metadata (tag filtering is OR-semantics against a case's tag
    set, subset takes the first N after tag filtering -- the exact same
    two conventions runners/_base.py::load_examples already uses for
    chat_relevance, kept identical here on purpose so both suites'
    `--tags`/`--subset` flags behave the same way), then resolves and
    parses each selected fixture file, returning ordinary Example
    objects runners/_base.py::run_suite already knows how to consume.

    Fixture-level validation (_load_fixture) runs only against the
    SELECTED rows, after tag/subset filtering -- a malformed fixture
    outside the current selection is not loaded and so cannot block an
    otherwise-valid `--tags`/`--subset` run. Duplicate-id detection is
    the one check that always runs against the FULL manifest regardless
    of filtering, since it's a manifest-wide invariant, not a
    per-selection concern.
    """
    rows = _load_manifest_rows()
    if tags:
        rows = [r for r in rows if set(r.get("tags") or []) & set(tags)]
    if subset is not None:
        rows = rows[:subset]

    examples: list[Example] = []
    for row in rows:
        fixture = _load_fixture(row)
        inputs = {
            "topic": fixture["topic"],
            "template": fixture["template"],
            "selected_papers": fixture["selected_papers"],
            "approved_web_articles": fixture["approved_web_articles"],
            "generated_report": fixture["generated_report"],
        }
        outputs = {
            "expected_hard_failures": fixture["expected"]["hard_failures"],
            "expected_dimension_labels": fixture["expected"]["dimension_labels"],
        }
        metadata = {
            "tags": row.get("tags") or [],
            "source_origin": row.get("source_origin") or "",
            "notes": row.get("notes") or "",
            "fixture_notes": fixture.get("notes") or "",
        }
        examples.append(Example(id=row["id"], inputs=inputs, outputs=outputs, metadata=metadata))
    return examples


# --- Deterministic structural checks -----------------------------------

def _check_required_sections(report: dict[str, Any]) -> tuple[list[str], list[str]]:
    missing = [k for k in REQUIRED_SECTION_KEYS if k not in report]
    empty = [k for k in REQUIRED_SECTION_KEYS if k in report and not (report[k].get("content") or "").strip()]
    return missing, empty


def _check_unresolved_markers(
    report: dict[str, Any], selected_papers: list[dict[str, Any]], approved_web_articles: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """A marker is unresolved when it's either (a) a raw "[Paper N]"/
    "[Web N]" form (unambiguous source-marker syntax, always a defect),
    or (b) a single-token bracket whose content exactly matches a KNOWN
    paper_id or approved web url -- a "known raw source id" that leaked
    through unresolved. A bracket that matches neither -- ordinary
    prose like "[note]" or "[sic]", or an already-resolved "[7]" -- is
    never flagged, per this check's own "don't classify ordinary
    non-citation bracketed prose as a citation" requirement."""
    known_ids = {p["paper_id"] for p in selected_papers if p.get("paper_id")}
    known_ids |= {a["url"] for a in approved_web_articles if a.get("url")}

    found: dict[str, list[str]] = {}
    for key in REQUIRED_SECTION_KEYS:
        if key not in report:
            continue
        content = report[key].get("content") or ""
        matches = [m.group() for m in _PAPER_WEB_MARKER_RE.finditer(content)]
        for m in _SINGLE_TOKEN_BRACKET_RE.finditer(content):
            token = m.group(1)
            if token.isdigit():
                continue  # the valid final [N] form -- never a defect
            if token in known_ids:
                matches.append(m.group())
        if matches:
            found[key] = matches
    return found


def _check_reference_numbering(report: dict[str, Any]) -> tuple[list[int], bool]:
    numbers = sorted(r["number"] for r in report.get("references") or [])
    non_sequential = bool(numbers) and numbers != list(range(1, len(numbers) + 1))
    return numbers, non_sequential


def _check_orphan_references(report: dict[str, Any]) -> tuple[set[int], list[int]]:
    """Scans every section's own rendered CONTENT for a "[N]" marker --
    never trusts a section's `reference_numbers` metadata field alone,
    per this check's own explicit requirement -- so a reference is only
    considered cited when a real inline marker for it is actually
    visible in the prose a reader would see."""
    cited_in_prose: set[int] = set()
    for key in REQUIRED_SECTION_KEYS:
        if key not in report:
            continue
        content = report[key].get("content") or ""
        cited_in_prose.update(int(m.group(1)) for m in _FINAL_NUMERIC_MARKER_RE.finditer(content))

    orphans = sorted(
        r["number"] for r in report.get("references") or [] if r["number"] not in cited_in_prose
    )
    return cited_in_prose, orphans


def _check_reference_availability(
    report: dict[str, Any], selected_papers: list[dict[str, Any]], approved_web_articles: list[dict[str, Any]],
) -> list[int]:
    """Availability only -- this deliberately never checks whether a
    reference's SOURCE CONTENT actually supports what it's cited for
    (that's citation_correctness/groundedness, R6C's live judge
    territory); it only checks the reference points at something that
    exists in this report's own evidence pool at all."""
    known_paper_ids = {p["paper_id"] for p in selected_papers if p.get("paper_id")}
    known_urls = {a["url"] for a in approved_web_articles if a.get("url")}

    unavailable: list[int] = []
    for r in report.get("references") or []:
        kind = r.get("kind")
        if kind == "paper":
            if not r.get("paper_id") or r["paper_id"] not in known_paper_ids:
                unavailable.append(r["number"])
        elif kind == "web":
            if not r.get("url") or r["url"] not in known_urls:
                unavailable.append(r["number"])
        else:
            unavailable.append(r["number"])  # malformed: missing/unrecognized kind
    return sorted(unavailable)


def _build_hard_failures(
    missing_sections: list[str], empty_sections: list[str], unresolved_by_section: dict[str, list[str]],
    non_sequential: bool, orphans: list[int], unavailable_refs: list[int],
) -> list[str]:
    detected = {
        "missing_required_section": bool(missing_sections),
        "empty_required_section": bool(empty_sections),
        "unresolved_citation_marker": bool(unresolved_by_section),
        "non_sequential_reference_numbering": non_sequential,
        "orphan_reference": bool(orphans),
        "reference_source_unavailable": bool(unavailable_refs),
    }
    return [identifier for identifier in CANONICAL_HARD_FAILURE_ORDER if detected[identifier]]


# --- Warnings and informational signals (never gate pass/fail) --------

def _collect_warnings(
    selected_papers: list[dict[str, Any]], approved_web_articles: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    for p in selected_papers:
        if not (p.get("abstract") or "").strip():
            warnings.append(f"selected paper {p.get('paper_id') or p.get('title') or '?'!r} has no abstract")
    for a in approved_web_articles:
        if not (a.get("snippet") or "").strip():
            warnings.append(f"approved web article {a.get('url') or a.get('title') or '?'!r} has an empty snippet")
    return warnings


def _compute_informational_signals(
    report: dict[str, Any], selected_papers: list[dict[str, Any]], approved_web_articles: list[dict[str, Any]],
    cited_in_prose: set[int],
) -> dict[str, Any]:
    """Deterministic measurements only -- no threshold is ever applied
    here (per this suite's own explicit design constraint); these ride
    along in the prediction/detail JSON for a human to look at, never
    as an evaluator score. See specs/report-quality-evaluation-plan.md
    section 3 for what each signal deliberately does NOT imply on its
    own (high density isn't automatically good, one dominant source
    isn't automatically bad, etc.)."""
    section_word_counts: dict[str, int] = {}
    citation_density_by_section: dict[str, float] = {}
    source_citation_counts: dict[str, int] = {}

    for key in REQUIRED_SECTION_KEYS:
        if key not in report:
            continue
        content = report[key].get("content") or ""
        word_count = len(_WORD_RE.findall(content))
        markers = list(_FINAL_NUMERIC_MARKER_RE.finditer(content))
        section_word_counts[key] = word_count
        citation_density_by_section[key] = round(len(markers) / word_count, 4) if word_count else 0.0
        for m in markers:
            source_citation_counts[m.group(1)] = source_citation_counts.get(m.group(1), 0) + 1

    total_selected_papers = len(selected_papers)
    skipped_papers = report.get("skipped_papers") or []
    skipped_paper_rate = round(len(skipped_papers) / total_selected_papers, 4) if total_selected_papers else None

    known_paper_ids = {p["paper_id"] for p in selected_papers if p.get("paper_id")}
    known_urls = {a["url"] for a in approved_web_articles if a.get("url")}
    references = report.get("references") or []
    cited_paper_ids = {
        r["paper_id"] for r in references
        if r.get("kind") == "paper" and r.get("paper_id") in known_paper_ids and r["number"] in cited_in_prose
    }
    cited_urls = {
        r["url"] for r in references
        if r.get("kind") == "web" and r.get("url") in known_urls and r["number"] in cited_in_prose
    }
    selected_source_coverage = {
        "papers": {"cited": len(cited_paper_ids), "total": total_selected_papers},
        "web_articles": {"cited": len(cited_urls), "total": len(approved_web_articles)},
    }

    dominant_source_share = None
    if source_citation_counts:
        total_markers = sum(source_citation_counts.values())
        if total_markers:
            dominant_source_share = round(max(source_citation_counts.values()) / total_markers, 4)

    return {
        "section_word_counts": section_word_counts,
        "citation_density_by_section": citation_density_by_section,
        "source_citation_counts": source_citation_counts,
        "skipped_paper_rate": skipped_paper_rate,
        "selected_source_coverage": selected_source_coverage,
        "dominant_source_share": dominant_source_share,
    }


# --- predict() ----------------------------------------------------------

def predict(example: Example) -> dict[str, Any]:
    """Deterministic-only (R6B) -- never calls OpenAI, never generates
    or revises anything, never scores a judge_dimension. Returns the
    frozen R6A result shape (specs/report-quality-evaluation-plan.md
    section 1) with `judge_dimensions`/`judge_metadata` both `None` and
    `not_applicable` always `[]` -- R6B makes no attempt at any judge
    dimension, so there is nothing to mark not_applicable yet; that's
    R6C's job."""
    report = example.inputs["generated_report"]
    selected_papers = example.inputs["selected_papers"]
    approved_web_articles = example.inputs["approved_web_articles"]

    missing_sections, empty_sections = _check_required_sections(report)
    unresolved_by_section = _check_unresolved_markers(report, selected_papers, approved_web_articles)
    reference_numbers, non_sequential = _check_reference_numbering(report)
    cited_in_prose, orphans = _check_orphan_references(report)
    unavailable_refs = _check_reference_availability(report, selected_papers, approved_web_articles)

    hard_failures = _build_hard_failures(
        missing_sections, empty_sections, unresolved_by_section, non_sequential, orphans, unavailable_refs,
    )

    checks = {
        "missing_sections": missing_sections,
        "empty_sections": empty_sections,
        "unresolved_markers": unresolved_by_section,
        "reference_numbers": reference_numbers,
        "non_sequential_reference_numbering": non_sequential,
        "cited_in_prose": sorted(cited_in_prose),
        "orphan_references": orphans,
        "unavailable_references": unavailable_refs,
    }

    return {
        "schema_version": "r6a-v1",
        "structural_integrity": {
            "status": "fail" if hard_failures else "pass",
            "checks": checks,
            "hard_failures": list(hard_failures),
        },
        "informational_signals": _compute_informational_signals(
            report, selected_papers, approved_web_articles, cited_in_prose,
        ),
        "judge_dimensions": None,
        "hard_failures": list(hard_failures),
        "warnings": _collect_warnings(selected_papers, approved_web_articles),
        "not_applicable": [],
        "judge_metadata": None,
    }


# --- R6C.2: live judge orchestration --------------------------------------

def _build_live_client() -> OpenAI:
    """Same pre-flight-before-the-loop pattern run_chat_relevance.py's
    own `_build_live_client` already proves live: fails BEFORE
    run_suite's per-example loop even starts, so a bad setup never
    burns a single paid call. Two checks, both raising the same
    LiveModeSetupError cli.py's cmd_run already catches and turns into
    a clean exit-2 line (no traceback, no CSV/detail side effects):

    1. REPORT_QUALITY_JUDGE_MODEL must be non-empty -- an operator
       accidentally exporting an empty env var is caught here, cheaply,
       without needing a real API call to discover it.
    2. OpenAI() client construction must succeed -- the same bare call
       qa.py's own ask()/condense_question use, reading OPENAI_API_KEY
       (or another SDK-recognized credential) from the environment.

    A configured model that IS a non-empty string but genuinely
    doesn't exist/isn't accessible can only be discovered by an actual
    API call -- that failure surfaces later, per-example, as a
    judge_metadata error (see predict_live below), not here. This
    module never retries and never silently substitutes a different
    model either way.
    """
    if not REPORT_QUALITY_JUDGE_MODEL or not REPORT_QUALITY_JUDGE_MODEL.strip():
        raise LiveModeSetupError(
            "REPORT_QUALITY_JUDGE_MODEL is empty -- set a real model id, or unset the environment "
            "variable entirely to use the built-in default"
        )
    try:
        return OpenAI()
    except OpenAIError as exc:
        raise LiveModeSetupError(
            "live mode requires OpenAI credentials (OPENAI_API_KEY or another OpenAI-SDK-recognized "
            f"credential) -- client construction failed: {exc}"
        ) from exc


def _skipped_judge_dimensions(reason: str) -> dict[str, dict[str, Any]]:
    """R6C.1's own documented convention (report_quality_inputs.py's
    SKIPPED_JUDGE_DIMENSION_LABEL docstring): when a report already
    failed R6B's deterministic gate, every judge dimension is
    "unknown" -- never "not_applicable". "not_applicable" means a
    dimension genuinely doesn't apply to an otherwise-normal report;
    "unknown" means no judgment was ever attempted, which is exactly
    what a skip is."""
    label = report_quality_inputs.SKIPPED_JUDGE_DIMENSION_LABEL
    return {dim: {"label": label, "score": None, "reasons": [reason]} for dim in REQUIRED_DIMENSION_NAMES}


def _aggregate_claim_source_dimensions(
    claim_result: dict[str, Any], cited_claims: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Maps Judge 1's per-claim structured verdicts onto the 2 frozen
    dimensions it owns -- a deterministic, PROVISIONAL, documented-as-
    such aggregation rule (still no invented WEIGHTS, per specs/
    report-quality-evaluation-plan.md section 0 decision 2 -- but this IS
    an invented rule, calibrated once against direct live-run evidence at
    R6C.2c, see CITATION_AGGREGATION_POLICY_VERSION).

    R6C.2c recalibration: the R6C.2/R6C.2a rule ("citation_correctness
    fails if ANY attached source verdict is anything less than a clean
    'supports'") conflated two genuinely different questions -- (1) does
    each attached source genuinely and relevantly contribute to the claim
    it's cited for, and (2) do the attached sources, TOGETHER, fully
    support the complete claim. A legitimate multi-source comparative
    sentence ("ChunkRank does X [1], while LongMem does Y [2]") will
    ALWAYS produce a "partially_supports" per-source verdict for each
    half -- neither source alone covers the other's clause -- even when
    the claim is accurately written and properly cited. The old rule
    mechanically failed citation_correctness on exactly the well-formed,
    well-cited sentences R6C.2b's own fixture corrections produced (see
    eval_results/runs/report_quality_run_4.json's methodology_landscape:
    0:2 and future_research_directions:0:0: both moved from a
    partially_supported COLLECTIVE verdict to a fully "supported"
    collective verdict after correction, yet citation_correctness's
    per-source failure count didn't move, because each half-covering
    source still verdicted "partially_supports"). Question (2) -- is the
    complete claim fully supported -- is already groundedness's own job.

    citation_correctness now answers ONLY question (1), over CITED
    claims' per-source verdicts:
    - "fail" if any attached source verdict is "does_not_support" (a
      source that does not even relevantly support what it's cited
      for is always a real citation defect, regardless of how the
      claim's other sources or its collective verdict come out).
    - "unknown" if no source verdict is "does_not_support" but at least
      one is "insufficient_evidence" (a real, inconclusive result --
      never silently dropped into "not_applicable", which is reserved
      for when nothing was judgeable in the first place).
    - "pass" when every attached source is "supports" or
      "partially_supports" (a source that partially, relevantly
      supports its own clause of a grouped comparative claim is doing
      its job -- it is not required to single-handedly cover clauses
      that belong to a DIFFERENT attached source).
    - "not_applicable" when there are no cited claims with any source
      verdict to judge at all.
    `counts` (supports/partially_supports/does_not_support/
    insufficient_evidence, over all judged attached sources) is always
    persisted alongside the label so a reader can audit the ratio behind
    a "pass" without re-deriving it from raw verdicts.

    groundedness is UNCHANGED in spirit from R6C.2/R6C.2a, only gaining
    an "unknown" state that used to be silently folded into
    "not_applicable": aggregated from COLLECTIVE verdicts across BOTH
    cited and uncited claims (does the claim's content, overall, fully
    hold up against its available evidence -- this is what catches an
    UNCITED fabricated claim, not just a miscited one).
    - "fail" if any collective verdict is "unsupported" or "partially_
      supported" (a definite failure always wins, even alongside
      unrelated insufficient-evidence claims).
    - "unknown" if no collective verdict is unsupported/partially_
      supported but at least one judged claim is "insufficient_
      evidence".
    - "pass" only when every judged claim's collective verdict is
      "supported".
    - "not_a_verifiable_claim" collective verdicts (R6C.2a) are excluded
      from the judged set entirely, exactly as before -- framing/
      organizational prose is neither pass, fail, nor unknown.
    - "not_applicable" when nothing factual (i.e. nothing beyond
      not_a_verifiable_claim exclusions) was ever judgeable.
    `counts` (supported/partially_supported/unsupported/insufficient_
    evidence, over judged collective verdicts) is persisted the same way.

    Never invents support: a claim/source pair with no usable evidence
    contributes to NEITHER pass nor fail -- only to "unknown" (when
    something else was judged) or "not_applicable" (when nothing was).
    """
    if claim_result["error"] is not None:
        reason = [f"claim/source judge failed: {claim_result['error']}"]
        return {
            "citation_correctness": {"label": "unknown", "score": None, "reasons": reason},
            "groundedness": {"label": "unknown", "score": None, "reasons": reason},
        }

    verdicts = claim_result["verdicts"]
    if not verdicts:
        reason = ["no cited or uncited claims were sampled for this report"]
        return {
            "citation_correctness": {"label": "not_applicable", "score": None, "reasons": reason},
            "groundedness": {"label": "not_applicable", "score": None, "reasons": reason},
        }

    cited_claim_ids = {c["claim_id"] for c in cited_claims}
    source_verdict_values = [
        sv["verdict"]
        for claim_id in cited_claim_ids
        for sv in verdicts.get(claim_id, {}).get("source_verdicts", [])
    ]
    source_counts = {
        "supports": source_verdict_values.count("supports"),
        "partially_supports": source_verdict_values.count("partially_supports"),
        "does_not_support": source_verdict_values.count("does_not_support"),
        "insufficient_evidence": source_verdict_values.count("insufficient_evidence"),
    }
    if not source_verdict_values:
        citation_correctness = {
            "label": "not_applicable", "score": None,
            "reasons": ["no cited claim had any source verdict to judge"], "counts": source_counts,
        }
    elif source_counts["does_not_support"]:
        judged = len(source_verdict_values) - source_counts["insufficient_evidence"]
        citation_correctness = {
            "label": "fail",
            "score": round(1 - source_counts["does_not_support"] / judged, 4) if judged else None,
            "reasons": [f"{source_counts['does_not_support']}/{len(source_verdict_values)} attached source(s) did not relevantly support their claim"],
            "counts": source_counts,
        }
    elif source_counts["insufficient_evidence"]:
        citation_correctness = {
            "label": "unknown", "score": None,
            "reasons": [f"{source_counts['insufficient_evidence']}/{len(source_verdict_values)} attached source(s) had insufficient evidence to judge"],
            "counts": source_counts,
        }
    else:
        citation_correctness = {
            "label": "pass",
            "score": round(source_counts["supports"] / len(source_verdict_values), 4),
            "reasons": ["every attached source relevantly contributed to the claim it was cited for"],
            "counts": source_counts,
        }

    collective_values = [v["collective_verdict"] for v in verdicts.values()]
    judgeable_collective = [v for v in collective_values if v != "not_a_verifiable_claim"]
    excluded_as_non_verifiable = sum(1 for v in collective_values if v == "not_a_verifiable_claim")
    collective_counts = {
        "supported": judgeable_collective.count("supported"),
        "partially_supported": judgeable_collective.count("partially_supported"),
        "unsupported": judgeable_collective.count("unsupported"),
        "insufficient_evidence": judgeable_collective.count("insufficient_evidence"),
    }
    if not judgeable_collective:
        reason = "no claim had judgeable (non-framing-prose) evidence"
        if excluded_as_non_verifiable:
            reason += f"; {excluded_as_non_verifiable} sampled item(s) excluded as non-verifiable framing prose"
        groundedness = {"label": "not_applicable", "score": None, "reasons": [reason], "counts": collective_counts}
    else:
        bad = collective_counts["unsupported"] + collective_counts["partially_supported"]
        if bad:
            groundedness = {
                "label": "fail", "score": round(1 - bad / len(judgeable_collective), 4),
                "reasons": [f"{bad}/{len(judgeable_collective)} judged claim(s) were not fully supported by their evidence"],
                "counts": collective_counts,
            }
        elif collective_counts["insufficient_evidence"]:
            groundedness = {
                "label": "unknown", "score": None,
                "reasons": [f"{collective_counts['insufficient_evidence']}/{len(judgeable_collective)} judged claim(s) had insufficient evidence to judge"],
                "counts": collective_counts,
            }
        else:
            groundedness = {
                "label": "pass", "score": 1.0,
                "reasons": ["every judged claim was fully supported by its evidence"],
                "counts": collective_counts,
            }

    return {"citation_correctness": citation_correctness, "groundedness": groundedness}


def _holistic_dimensions_from_result(holistic_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if holistic_result["error"] is not None:
        reason = [f"holistic judge failed: {holistic_result['error']}"]
        return {
            dim: {"label": "unknown", "score": None, "reasons": reason}
            for dim in ("synthesis_quality", "analytical_quality", "template_fit", "coherence", "source_balance")
        }
    return holistic_result["dimensions"]


def predict_live(example: Example, client: OpenAI) -> dict[str, Any]:
    """Live-mode prediction (R6C.2). Always starts from `predict(example)`
    -- R6B's own deterministic structural/informational result, byte-
    identical to what mock mode computes for the exact same report,
    since structural correctness does not depend on mode. Only
    `judge_dimensions`/`judge_metadata`/`not_applicable` differ from a
    mock prediction.

    When the report already failed R6B's hard-failure gate: zero judge
    calls, all 7 dimensions "unknown" (see _skipped_judge_dimensions),
    `judge_metadata` records the skip reason and still carries model/
    prompt-version identity even though nothing was called -- a reader
    of the detail JSON should never have to guess which model/prompt
    version WOULD have been used.

    Otherwise: calls research_agent.evals.report_quality_inputs.
    prepare_report_quality_judge_inputs for R6C.1's own bounded,
    sanitized payload, then makes exactly one claim/source call
    (judges/claim_source.py) and exactly one holistic call
    (judges/holistic.py) -- both independent, both wrapped in their own
    "never raises" contract, so one judge failing never suppresses the
    other's result (see _aggregate_claim_source_dimensions/
    _holistic_dimensions_from_result, each only degrading its OWN
    dimensions to "unknown" on its own judge's failure).
    """
    base_prediction = predict(example)
    payload = report_quality_inputs.prepare_report_quality_judge_inputs(example, base_prediction)

    if payload["evaluation_status"] == report_quality_inputs.EVALUATION_STATUS_SKIPPED:
        judge_dimensions = _skipped_judge_dimensions(
            "structural hard failure present (see hard_failures) -- no live judge call was made",
        )
        judge_metadata = {
            "model": REPORT_QUALITY_JUDGE_MODEL,
            "claim_source_prompt_version": claim_source.CLAIM_SOURCE_JUDGE_PROMPT_VERSION,
            "holistic_prompt_version": holistic.HOLISTIC_JUDGE_PROMPT_VERSION,
            "citation_aggregation_policy_version": CITATION_AGGREGATION_POLICY_VERSION,
            "claim_source_judge": None,
            "holistic_judge": None,
            "sampling_coverage": payload["sampling_coverage"],
            "sanitization_counts": {"source_injection_findings": 0, "report_prose_injection_findings": 0},
            "scores_are_informational": True,
            "skip_reason": payload["skip_reason"],
        }
        return {**base_prediction, "judge_dimensions": judge_dimensions, "judge_metadata": judge_metadata, "not_applicable": []}

    topic = example.inputs["topic"]
    template = example.inputs["template"]

    claim_result = claim_source.judge_claims(
        topic, template, payload["selected_cited_claims"], payload["selected_uncited_candidates"],
        payload["evidence_registry"], client, REPORT_QUALITY_JUDGE_MODEL,
    )
    holistic_result = holistic.judge_report(
        topic, template, payload["sanitized_report_sections"], base_prediction["informational_signals"],
        payload["sampling_coverage"], client, REPORT_QUALITY_JUDGE_MODEL,
    )

    judge_dimensions = {
        **_aggregate_claim_source_dimensions(claim_result, payload["selected_cited_claims"]),
        **_holistic_dimensions_from_result(holistic_result),
    }
    not_applicable = [dim for dim, v in judge_dimensions.items() if v["label"] == "not_applicable"]

    judge_metadata = {
        "model": REPORT_QUALITY_JUDGE_MODEL,
        "claim_source_prompt_version": claim_source.CLAIM_SOURCE_JUDGE_PROMPT_VERSION,
        "holistic_prompt_version": holistic.HOLISTIC_JUDGE_PROMPT_VERSION,
        "citation_aggregation_policy_version": CITATION_AGGREGATION_POLICY_VERSION,
        "claim_source_judge": {
            "latency_ms": claim_result["latency_ms"], "error": claim_result["error"],
            "claims_judged": claim_result["claims_judged"], "token_usage": claim_result["token_usage"],
            "verdicts": claim_result["verdicts"],
            "not_a_verifiable_claim_count": len(claim_result["not_a_verifiable_claim_ids"]),
            "not_a_verifiable_claim_ids": claim_result["not_a_verifiable_claim_ids"],
        },
        "holistic_judge": {
            "latency_ms": holistic_result["latency_ms"], "error": holistic_result["error"],
            "token_usage": holistic_result["token_usage"],
        },
        "sampling_coverage": payload["sampling_coverage"],
        "sanitization_counts": {
            "source_injection_findings": len(payload["source_injection_findings"]),
            "report_prose_injection_findings": len(payload["report_prose_injection_findings"]),
        },
        "scores_are_informational": True,
    }

    return {**base_prediction, "judge_dimensions": judge_dimensions, "judge_metadata": judge_metadata, "not_applicable": not_applicable}


# --- run_experiment() -----------------------------------------------------

def run_experiment(mode: str = "mock", subset: int | None = None, tags: list[str] | None = None) -> SuiteResult:
    """`mode="mock"` (the default, unchanged from R6B): deterministic,
    offline, free, calls no evaluator/prediction logic that could touch
    a network. `mode="live"` (R6C.2): constructs a real OpenAI client
    up front (`_build_live_client`, raising LiveModeSetupError -- exit
    2, no traceback, no CSV/detail side effects -- before any example
    is loaded if setup fails) and runs `predict_live` per example,
    making up to two real judge calls per eligible report."""
    evaluators = [
        ("report_quality_hard_failure_agreement", ALL_EVALUATORS["report_quality_hard_failure_agreement"]),
        ("report_quality_dimension_agreement", ALL_EVALUATORS["report_quality_dimension_agreement"]),
    ]

    if mode == "mock":
        run_predict = predict
    elif mode == "live":
        client = _build_live_client()
        run_predict = lambda example: predict_live(example, client)  # noqa: E731
    else:
        raise ValueError(f"unknown report_quality mode {mode!r} -- expected 'mock' or 'live'")

    examples = load_report_quality_examples(subset=subset, tags=tags)
    return run_suite(
        suite=SUITE, dataset_file=str(MANIFEST_PATH), predict=run_predict, evaluators=evaluators,
        mode=mode, examples=examples,
    )
