"""R6C.1: deterministic preparation of trustworthy, bounded inputs for
the future report_quality live judges (R6C.2 -- not built here). Makes
**zero OpenAI/API calls** and implements no judge of any kind; this
module only builds the evidence registry, extracts and samples claim
units, and detects/redacts prompt injection so that whatever judge
code comes later has a clean, bounded, already-sanitized payload to
work from instead of raw report text.

Deliberately kept OUT of runners/run_report_quality.py (R6B's own
deterministic hard-failure/informational-signal checks stay exactly as
they are, untouched) and out of any future prompt/judge module -- this
is purely the "prepare trustworthy input" layer in between, consumed
by (not consuming) run_report_quality.py's own `predict()` output for
the skip decision (see `prepare_report_quality_judge_inputs` below).

R6A decision 1 (specs/report-quality-evaluation-plan.md) applies here
too: independent from R4 and from R6B's own internals beyond reading
its already-computed `structural_integrity.status`. The prompt-
injection detector below is a fresh, R6-owned implementation -- it
does NOT import research_agent.qa's `_detect_retrieved_prompt_
injection`, for the same reason R6B doesn't import
research_agent.report's `_deterministic_report_checks`: a future
qa.py refactor must never silently change what this module protects
against.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "r6c1-v1"

# --- evaluation_status / skip convention (for future R6C.2) ------------

EVALUATION_STATUS_PREPARED = "prepared"
EVALUATION_STATUS_SKIPPED = "skipped_due_to_hard_failure"

# R6C.2 convention, not enforced here: when evaluation_status ==
# EVALUATION_STATUS_SKIPPED, every future judge_dimensions[...]["label"]
# must be stamped this value -- never "not_applicable". "not_applicable"
# means a dimension genuinely doesn't apply to an otherwise-normal
# report (e.g. source_balance with only one source); "unknown" means no
# valid judgment was ever attempted, which is exactly what a skip is --
# a report R6B already proved structurally broken never reaches a live
# judge call at all, so there is nothing "not applicable" about it,
# there is simply no judgment. This phase does not touch R6B's
# `judge_dimensions` (still `None`, per R6B, unchanged) -- this constant
# only documents the value R6C.2 must use later.
SKIPPED_JUDGE_DIMENSION_LABEL = "unknown"


# --- Frozen section keys (independent copy, same reasoning as run_report_quality.py) ---

REQUIRED_SECTION_KEYS = (
    "executive_summary", "introduction_scope", "thematic_findings",
    "methodology_landscape", "contradictions_open_debates", "gap_analysis",
    "future_research_directions", "conclusion",
)

# --- Bounded sampling caps (provisional -- same "documented starting
# point, not derived from real data" posture as every other magic
# number in this eval suite, e.g. qa.py's own
# _DIRECT_RELEVANCE_JUDGE_MAX_BATCH_SIZE) ---------------------------

MAX_CITED_CLAIM_UNITS = 16
MAX_UNCITED_CLAIM_CANDIDATES = 8
MAX_TOTAL_CLAIM_UNITS = MAX_CITED_CLAIM_UNITS + MAX_UNCITED_CLAIM_CANDIDATES

# Minimum word count for a marker-free sentence to be worth surfacing
# as an "uncited claim candidate" -- a transparent, provisional
# heuristic (documented, not derived from data) meant to exclude
# fragments/asides ("In short:", "See above.") without excluding a
# genuine short factual claim.
MIN_UNCITED_CANDIDATE_WORDS = 6

EVIDENCE_SCOPE_DISCLAIMER = (
    "abstract/snippet-level only -- full-paper or full-article verification is not "
    "possible with the evidence available to report generation itself"
)
SAMPLING_TRUNCATED_DISCLAIMER = (
    "claim/source sampling was truncated -- results reflect only the sampled subset, "
    "never full-report groundedness"
)

BLOCKED_INSTRUCTION_PLACEHOLDER = "[BLOCKED_UNTRUSTED_INSTRUCTION]"

STATUS_AVAILABLE = "available"
STATUS_MISSING_TEXT = "missing_text"
STATUS_BLOCKED = "blocked_untrusted_source"


# --- Independent prompt-injection detector ------------------------------
#
# Same two-part discipline R7E.5b's qa.py::_detect_retrieved_prompt_
# injection already proved live: (1) normalize (NFKC + lowercase +
# collapsed whitespace) so trivial formatting/case/unicode variance
# can't dodge detection, (2) match multi-word PHRASES only, never a
# single ordinary word ("system"/"prompt"/"instructions"/"model"/
# "rating" alone appear constantly in genuine academic/technical
# writing and must never trigger this on their own -- proven by
# dedicated tests, not just asserted).
#
# EXPLICITLY a high-precision FIRST guard, not a complete detector:
# encoded (base64/leetspeak), multilingual, indirect, and quoted-
# attack variants remain future work, exactly the same accepted
# limitation qa.py's own guard documents for its own patterns.

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("system_override", re.compile(r"\bsystem\s+override\b")),
    (
        "ignore_prior_instructions",
        re.compile(r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:prior|previous|above)\s+instructions\b"),
    ),
    (
        "disregard_prior_instructions",
        re.compile(r"\bdisregard\s+(?:all\s+)?(?:the\s+)?(?:prior|previous|above)\s+instructions\b"),
    ),
    (
        "forced_evaluator_rating",
        re.compile(
            r"\b(?:rate|rated|score|scored)\s+(?:this\s+\w+\s+)?as\s+(?:the\s+)?(?:single\s+)?most\s+important\b"
            r"|\b(?:should\s+be\s+)?(?:scored|rated)\s+(?:100(?:/100)?|10(?:/10)?)\b"
        ),
    ),
    (
        "forced_verdict_output",
        re.compile(r"\b(?:return|output)\s+(?:a\s+|the\s+)?(?:required\s+)?(?:verdict|result|answer|rating|score)\b"),
    ),
    (
        "directive_addressed_to_model",
        re.compile(
            r"\byou\s+(?:must|should)\s+(?:mark|classify|treat|report|answer|respond|say|rate|score)\b"
            r"|\bnote\s+to\s+(?:any\s+)?(?:ai\s+)?(?:system|assistant|model|evaluator|reviewer)\b"
        ),
    ),
]


def _normalize_for_injection_check(text: str) -> str:
    """Independent reimplementation of the same technique qa.py's own
    _normalize_for_injection_check uses -- NFKC normalization (folds
    visually-similar/full-width character variants to a canonical
    form), lowercased, then repeated whitespace collapsed to a single
    space."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", normalized)


def detect_prompt_injection(text: str) -> list[str]:
    """Returns every matched pattern id (usually zero or one, but text
    can trip more than one), never the matched text itself -- findings
    surface stable pattern IDs, not a copy of the manipulative string,
    same privacy-conscious convention qa.py's own guard already
    established."""
    normalized = _normalize_for_injection_check(text)
    return [pattern_id for pattern_id, pattern in _INJECTION_PATTERNS if pattern.search(normalized)]


# --- Paragraph / sentence splitting (documented heuristics, not NLP) ---

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")

# Splits after sentence-ending punctuation followed by whitespace, only
# when followed by an uppercase letter, a digit, a quote, or an opening
# bracket/paren -- deliberately conservative (won't split "e.g. next"
# or "47.3%" or "Dr. Smith", none of which are followed by whitespace-
# then-one-of-these-trigger-characters in normal prose). Both split
# operands are zero-width (lookbehind/lookahead), so re.split's
# returned pieces are exact, untouched substrings of the original text
# -- verified directly (not assumed) so BLOCKED_INSTRUCTION_PLACEHOLDER
# substitution (below) can safely `.replace()` a piece back into the
# original content.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[])')

# A fragment consisting of nothing but one or more "[N]" markers
# (optionally trailed by one sentence-ending punctuation mark) -- can
# arise when a marker run sits right after a period-space (the split
# regex above deliberately treats "[" as a valid sentence-start
# trigger, so this is an expected, not accidental, output shape this
# module explicitly handles below, not something a smarter regex tries
# to preempt).
_MARKER_ONLY_FRAGMENT_RE = re.compile(r"^(?:\[\d+\]\s*)+[.!?]?$")

_FINAL_NUMERIC_MARKER_RE = re.compile(r"\[(\d+)\]")
_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class _RawSentence:
    """One sentence-level fragment, pre-merge -- `text` is always an
    exact, untouched substring of the section's original `content`
    string (never re-normalized), so it can always be located via
    `content.replace(text, ...)` for redaction."""

    section_key: str
    paragraph_index: int
    sentence_index: int
    text: str


def _split_paragraphs(content: str) -> list[str]:
    paragraphs = [p for p in _PARAGRAPH_SPLIT_RE.split(content) if p.strip()]
    return paragraphs or ([content] if content.strip() else [])


def _split_sentences(paragraph: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def _flatten_section_sentences(report: dict[str, Any]) -> dict[str, list[_RawSentence]]:
    """One flat, canonically-ordered (section, paragraph, sentence)
    list per section -- the single shared primitive both claim
    extraction and report-prose injection scanning are built on, so
    there is exactly one place sentence-splitting logic lives."""
    by_section: dict[str, list[_RawSentence]] = {}
    for section_key in REQUIRED_SECTION_KEYS:
        if section_key not in report:
            continue
        content = report[section_key].get("content") or ""
        sentences: list[_RawSentence] = []
        for p_idx, paragraph in enumerate(_split_paragraphs(content)):
            for s_idx, sentence in enumerate(_split_sentences(paragraph)):
                sentences.append(_RawSentence(section_key, p_idx, s_idx, sentence))
        by_section[section_key] = sentences
    return by_section


def _merge_marker_only_fragments(sentences: list[_RawSentence]) -> list[_RawSentence]:
    """Returns a NEW list (never mutates the input) where a marker-only
    fragment is merged into the immediately preceding fragment --
    `[1][2]` trailing a sentence never becomes its own bogus claim.
    The merged sentence keeps the PRECEDING fragment's own section/
    paragraph/sentence position (a marker-only fragment that gets
    absorbed contributes only its reference numbers, never its own
    claim_id). A marker-only fragment with nothing preceding it (rare
    edge case -- the very first sentence of a section is itself
    marker-only) is left standalone, since there is nothing to merge
    it into."""
    merged: list[_RawSentence] = []
    for sentence in sentences:
        if merged and _MARKER_ONLY_FRAGMENT_RE.match(sentence.text.strip()):
            previous = merged[-1]
            merged[-1] = _RawSentence(
                previous.section_key, previous.paragraph_index, previous.sentence_index,
                f"{previous.text} {sentence.text}",
            )
        else:
            merged.append(sentence)
    return merged


# --- Evidence registry --------------------------------------------------

def build_evidence_registry(
    report: dict[str, Any], selected_papers: list[dict[str, Any]], approved_web_articles: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[int, str], list[dict[str, Any]]]:
    """Builds a deduplicated evidence registry from the report's OWN
    `references` list (the source of `reference_number`) enriched with
    the fixture's evidence (`selected_papers`/`approved_web_articles`,
    the source of actual abstract/snippet text) -- never mutates any of
    the three inputs. Returns `(registry, number_to_evidence_id,
    source_injection_findings)`.

    Dedup is automatic: the registry is keyed by evidence id
    (`paper:<paper_id>` / `web:<url>`), derived from each reference's
    own identity, so the same source cited under more than one
    reference number (not possible from a genuine report.py
    generation, but not assumed impossible for a stored/corrupted
    report either) collapses to one entry keyed on the first (lowest)
    reference number encountered.

    A reference whose source text fails the independent injection
    detector (`detect_prompt_injection`) is marked
    `status="blocked_untrusted_source"` and its `text` is cleared to
    `""` -- injected source text is never carried into a judge-ready
    evidence entry. This is deliberately distinct from
    `"missing_text"` (a genuinely absent abstract/snippet) -- the two
    must never be conflated, since one is an ordinary data gap and the
    other is an active adversarial signal.
    """
    paper_by_id = {p["paper_id"]: p for p in selected_papers if p.get("paper_id")}
    web_by_url = {a["url"]: a for a in approved_web_articles if a.get("url")}

    registry: dict[str, dict[str, Any]] = {}
    number_to_evidence_id: dict[int, str] = {}
    source_injection_findings: list[dict[str, Any]] = []

    references = sorted(report.get("references") or [], key=lambda r: r["number"])
    for ref in references:
        kind = ref.get("kind")
        if kind == "paper":
            identity = ref.get("paper_id")
        elif kind == "web":
            identity = ref.get("url")
        else:
            identity = None

        if kind in ("paper", "web") and identity:
            evidence_id = f"{kind}:{identity}"
        else:
            evidence_id = f"unknown:{ref['number']}"

        number_to_evidence_id[ref["number"]] = evidence_id
        if evidence_id in registry:
            continue  # already registered under an earlier (lower) reference number

        if kind == "paper":
            source = paper_by_id.get(identity)
            text = (source.get("abstract") or "") if source else ""
            title = (source.get("title") if source else None) or ref.get("title") or ""
        elif kind == "web":
            source = web_by_url.get(identity)
            text = (source.get("snippet") or "") if source else ""
            title = (source.get("title") if source else None) or ref.get("title") or ""
        else:
            text = ""
            title = ref.get("title") or ""

        status = STATUS_AVAILABLE if text.strip() else STATUS_MISSING_TEXT

        if status == STATUS_AVAILABLE:
            pattern_ids = detect_prompt_injection(f"{title}\n{text}")
            if pattern_ids:
                status = STATUS_BLOCKED
                source_injection_findings.append({
                    "evidence_id": evidence_id, "kind": kind, "reference_number": ref["number"],
                    "pattern_ids": pattern_ids,
                })
                text = ""  # injected text is never carried into judge-ready evidence

        registry[evidence_id] = {
            "kind": kind if kind in ("paper", "web") else "unknown",
            "reference_number": ref["number"],
            "title": title,
            "text": text,
            "status": status,
        }

    return registry, number_to_evidence_id, source_injection_findings


# --- Claim-unit extraction ------------------------------------------------

def _claim_unit(
    section_key: str, sentence: _RawSentence, claim_kind: str,
    reference_numbers: list[int], evidence_ids: list[str], selection_reason: str | None = None,
) -> dict[str, Any]:
    unit = {
        "claim_id": f"{section_key}:{sentence.paragraph_index}:{sentence.sentence_index}",
        "section_key": section_key,
        "claim_kind": claim_kind,
        "claim_text": sentence.text,
        "reference_numbers": reference_numbers,
        "evidence_ids": evidence_ids,
    }
    if selection_reason is not None:
        unit["selection_reason"] = selection_reason
    return unit


def extract_claim_units(
    report: dict[str, Any], number_to_evidence_id: dict[int, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Extracts every claim unit from every section, in canonical
    (section, paragraph, sentence) order. Returns two per-section maps:
    `(cited_by_section, uncited_by_section)`.

    A cited claim unit's `reference_numbers` are exactly the marker
    numbers present in its (possibly marker-merged) text, in ascending
    numeric order -- `[1][2]` on one sentence produces ONE claim with
    `reference_numbers=[1, 2]`, never two independent claims, per this
    module's own explicit requirement. `evidence_ids` resolves each
    number through `number_to_evidence_id`; a marker number with no
    matching registry entry (an unresolved/malformed marker -- R6B's
    own responsibility to flag as a hard failure, not re-validated
    here) is simply omitted from `evidence_ids` while still appearing
    in `reference_numbers`, so the mismatch stays visible rather than
    silently hidden.

    An uncited claim candidate is a marker-free sentence clearing
    MIN_UNCITED_CANDIDATE_WORDS -- never asserted as factual (that is
    the future judge's call), only flagged as worth checking, with its
    own `selection_reason` recorded for transparency.
    """
    cited_by_section: dict[str, list[dict[str, Any]]] = {}
    uncited_by_section: dict[str, list[dict[str, Any]]] = {}

    raw_by_section = _flatten_section_sentences(report)
    for section_key, raw_sentences in raw_by_section.items():
        merged = _merge_marker_only_fragments(raw_sentences)
        cited: list[dict[str, Any]] = []
        uncited: list[dict[str, Any]] = []

        for sentence in merged:
            marker_numbers = sorted({int(m.group(1)) for m in _FINAL_NUMERIC_MARKER_RE.finditer(sentence.text)})
            if marker_numbers:
                evidence_ids = [
                    number_to_evidence_id[n] for n in marker_numbers if n in number_to_evidence_id
                ]
                cited.append(_claim_unit(section_key, sentence, "cited", marker_numbers, evidence_ids))
            else:
                word_count = len(_WORD_RE.findall(sentence.text))
                if word_count >= MIN_UNCITED_CANDIDATE_WORDS:
                    reason = (
                        f"non-empty prose, {word_count} words, no inline citation marker present "
                        f"(minimum {MIN_UNCITED_CANDIDATE_WORDS} words to qualify)"
                    )
                    uncited.append(_claim_unit(section_key, sentence, "uncited_candidate", [], [], reason))

        cited_by_section[section_key] = cited
        uncited_by_section[section_key] = uncited

    return cited_by_section, uncited_by_section


# --- Bounded stratified (round-robin) sampling ---------------------------

def _round_robin_select(by_section: dict[str, list[dict[str, Any]]], cap: int) -> list[dict[str, Any]]:
    """Deterministic round-robin: in each round, take at most one item
    from each section (in REQUIRED_SECTION_KEYS order) that still has
    unselected items remaining, so as many sections as possible are
    represented before any section contributes a second item. Never
    randomizes. Returns the selected claim_ids as a set-preserving
    list in ROUND-ROBIN order -- the caller re-sorts back into
    canonical order (see sample_claim_units)."""
    cursors = {key: 0 for key in REQUIRED_SECTION_KEYS}
    selected: list[dict[str, Any]] = []
    if cap <= 0:
        return selected
    while len(selected) < cap:
        made_progress = False
        for section_key in REQUIRED_SECTION_KEYS:
            if len(selected) >= cap:
                break
            items = by_section.get(section_key) or []
            cursor = cursors[section_key]
            if cursor < len(items):
                selected.append(items[cursor])
                cursors[section_key] = cursor + 1
                made_progress = True
        if not made_progress:
            break  # every section's items are exhausted
    return selected


def sample_claim_units(
    cited_by_section: dict[str, list[dict[str, Any]]], uncited_by_section: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Applies MAX_CITED_CLAIM_UNITS / MAX_UNCITED_CLAIM_CANDIDATES via
    section_round_robin selection (never first-N), then re-sorts each
    selected set back into canonical (section, paragraph, sentence)
    order for the final returned lists -- round-robin decides WHICH
    claims are selected, canonical order decides how they are
    PRESENTED. Selecting a cited claim always includes every reference
    already attached to it (grouped markers were merged into one claim
    unit at extraction time, so there is nothing left to split here).
    Returns `(selected_cited, selected_uncited, coverage_metadata)`.
    """
    selected_cited_set = _round_robin_select(cited_by_section, MAX_CITED_CLAIM_UNITS)
    selected_uncited_set = _round_robin_select(uncited_by_section, MAX_UNCITED_CLAIM_CANDIDATES)

    selected_cited_ids = {c["claim_id"] for c in selected_cited_set}
    selected_uncited_ids = {c["claim_id"] for c in selected_uncited_set}

    selected_cited = [
        claim for section_key in REQUIRED_SECTION_KEYS
        for claim in cited_by_section.get(section_key, [])
        if claim["claim_id"] in selected_cited_ids
    ]
    selected_uncited = [
        claim for section_key in REQUIRED_SECTION_KEYS
        for claim in uncited_by_section.get(section_key, [])
        if claim["claim_id"] in selected_uncited_ids
    ]

    per_section: dict[str, dict[str, int]] = {}
    cited_total = 0
    uncited_total = 0
    for section_key in REQUIRED_SECTION_KEYS:
        section_cited_total = len(cited_by_section.get(section_key, []))
        section_uncited_total = len(uncited_by_section.get(section_key, []))
        if section_key not in cited_by_section and section_key not in uncited_by_section:
            continue  # section absent from the report entirely
        cited_total += section_cited_total
        uncited_total += section_uncited_total
        per_section[section_key] = {
            "cited_total": section_cited_total,
            "cited_selected": sum(1 for c in cited_by_section.get(section_key, []) if c["claim_id"] in selected_cited_ids),
            "uncited_total": section_uncited_total,
            "uncited_selected": sum(1 for c in uncited_by_section.get(section_key, []) if c["claim_id"] in selected_uncited_ids),
        }

    truncated = (len(selected_cited) < cited_total) or (len(selected_uncited) < uncited_total)

    coverage = {
        "strategy": "section_round_robin",
        "cited_total": cited_total,
        "cited_selected": len(selected_cited),
        "uncited_total": uncited_total,
        "uncited_selected": len(selected_uncited),
        "truncated": truncated,
        "per_section": per_section,
        "evidence_scope": EVIDENCE_SCOPE_DISCLAIMER,
    }
    return selected_cited, selected_uncited, coverage


# --- Report-prose injection detection + sanitized copy --------------------

def build_sanitized_report_and_findings(
    report: dict[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Scans every section's own rendered prose (the pre-merge sentence
    list -- deliberately NOT the marker-merged claim-unit text, so
    every redaction is guaranteed to be an exact substring of the
    original content and `.replace()` can never fail to find it) for
    the same independent injection detector `build_evidence_registry`
    uses on source evidence. The ORIGINAL `report` is never mutated --
    this returns a brand-new `{section_key: sanitized_content}` map. A
    flagged sentence is replaced with BLOCKED_INSTRUCTION_PLACEHOLDER
    in the sanitized copy only; every other sentence/section is
    untouched. Returns `(sanitized_sections, report_prose_injection_
    findings)`.
    """
    raw_by_section = _flatten_section_sentences(report)
    sanitized_sections: dict[str, str] = {}
    findings: list[dict[str, Any]] = []

    for section_key, raw_sentences in raw_by_section.items():
        original_content = report[section_key].get("content") or ""
        sanitized_content = original_content
        for sentence in raw_sentences:
            pattern_ids = detect_prompt_injection(sentence.text)
            if not pattern_ids:
                continue
            findings.append({
                "section_key": section_key,
                "paragraph_index": sentence.paragraph_index,
                "sentence_index": sentence.sentence_index,
                "pattern_ids": pattern_ids,
            })
            sanitized_content = sanitized_content.replace(sentence.text, BLOCKED_INSTRUCTION_PLACEHOLDER, 1)
        sanitized_sections[section_key] = sanitized_content

    return sanitized_sections, findings


# --- Top-level: prepared payload -----------------------------------------

def _empty_sampling_coverage() -> dict[str, Any]:
    return {
        "strategy": "section_round_robin",
        "cited_total": 0, "cited_selected": 0,
        "uncited_total": 0, "uncited_selected": 0,
        "truncated": False, "per_section": {},
        "evidence_scope": EVIDENCE_SCOPE_DISCLAIMER,
    }


def prepare_report_quality_judge_inputs(
    example: Any, structural_prediction: dict[str, Any],
) -> dict[str, Any]:
    """The one entry point future R6C.2 judge code will call.
    `structural_prediction` is exactly whatever `runners/
    run_report_quality.py::predict(example)` already returned (R6B's
    own, untouched deterministic result) -- this function never
    recomputes hard-failure checks itself, it only reads `structural_
    integrity.status` to decide whether preparation should even
    attempt claim extraction. Makes zero OpenAI/API calls under any
    circumstance and never mutates `example.inputs` or anything inside
    it -- every returned structure is freshly built.

    When R6B already found a hard failure, preparation short-circuits
    entirely (no claim extraction, no injection scan, no wasted work
    on a report already known to be broken) and returns
    `evaluation_status="skipped_due_to_hard_failure"` -- see
    SKIPPED_JUDGE_DIMENSION_LABEL's own docstring for the convention
    R6C.2 must follow when it sees this status. Contains no judge
    scores, labels, model metadata, or prompts -- that is entirely
    R6C.2's job, not this phase's.
    """
    if structural_prediction["structural_integrity"]["status"] == "fail":
        hard_failures = structural_prediction["structural_integrity"]["hard_failures"]
        return {
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": EVALUATION_STATUS_SKIPPED,
            "skip_reason": (
                f"structural_integrity.status=fail (hard_failures={hard_failures}); "
                "no live judge call will occur"
            ),
            "sanitized_report_sections": None,
            "evidence_registry": {},
            "claim_counts": {"cited_total": 0, "uncited_total": 0},
            "selected_cited_claims": [],
            "selected_uncited_candidates": [],
            "sampling_coverage": _empty_sampling_coverage(),
            "source_injection_findings": [],
            "report_prose_injection_findings": [],
            "disclaimers": {
                "evidence_scope": EVIDENCE_SCOPE_DISCLAIMER,
                "sampling_truncated": None,
            },
        }

    report = example.inputs["generated_report"]
    selected_papers = example.inputs["selected_papers"]
    approved_web_articles = example.inputs["approved_web_articles"]

    registry, number_to_evidence_id, source_injection_findings = build_evidence_registry(
        report, selected_papers, approved_web_articles,
    )
    cited_by_section, uncited_by_section = extract_claim_units(report, number_to_evidence_id)
    selected_cited, selected_uncited, coverage = sample_claim_units(cited_by_section, uncited_by_section)
    sanitized_sections, report_prose_injection_findings = build_sanitized_report_and_findings(report)

    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": EVALUATION_STATUS_PREPARED,
        "sanitized_report_sections": sanitized_sections,
        "evidence_registry": registry,
        "claim_counts": {"cited_total": coverage["cited_total"], "uncited_total": coverage["uncited_total"]},
        "selected_cited_claims": selected_cited,
        "selected_uncited_candidates": selected_uncited,
        "sampling_coverage": coverage,
        "source_injection_findings": source_injection_findings,
        "report_prose_injection_findings": report_prose_injection_findings,
        "disclaimers": {
            "evidence_scope": EVIDENCE_SCOPE_DISCLAIMER,
            "sampling_truncated": SAMPLING_TRUNCATED_DISCLAIMER if coverage["truncated"] else None,
        },
    }
