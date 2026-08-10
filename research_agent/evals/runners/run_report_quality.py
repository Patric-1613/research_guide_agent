"""Runner for the report_quality suite -- deterministic/mock only (R6B).
No live mode exists yet (R6C's job: two bounded live judge tasks, per
specs/report-quality-evaluation-plan.md section 0 decision 4).

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
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from research_agent.evals.evaluators.report_quality import ALL_EVALUATORS
from research_agent.evals.runners._base import EVAL_DATA_DIR, Example, LiveModeSetupError, SuiteResult, run_suite

SUITE = "report_quality"

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


# --- run_experiment() -----------------------------------------------------

def run_experiment(mode: str = "mock", subset: int | None = None, tags: list[str] | None = None) -> SuiteResult:
    """R6B: `mode="mock"` (the default) is the only mode implemented --
    deterministic, offline, free, calls no evaluator/prediction logic
    that could touch a network. `mode="live"` raises LiveModeSetupError
    immediately, before any example is loaded or any file is written --
    the same clean, non-traceback, no-side-effect failure shape
    chat_relevance's own missing-credentials path already established
    (cli.py's cmd_run catches LiveModeSetupError and exits 2), reused
    here for a different reason (not implemented yet, rather than
    missing credentials). R6C is what will replace this raise with a
    real live judge implementation."""
    if mode == "live":
        raise LiveModeSetupError(
            "report_quality has no live mode yet -- R6B is deterministic/mock-only. "
            "Live judge dimensions (bounded claim/source judge + holistic judge) are R6C's job."
        )
    if mode != "mock":
        raise ValueError(f"unknown report_quality mode {mode!r} -- expected 'mock' or 'live'")

    examples = load_report_quality_examples(subset=subset, tags=tags)
    return run_suite(
        suite=SUITE,
        dataset_file=str(MANIFEST_PATH),
        predict=predict,
        evaluators=[("report_quality_hard_failure_agreement", ALL_EVALUATORS["report_quality_hard_failure_agreement"])],
        mode=mode,
        examples=examples,
    )
