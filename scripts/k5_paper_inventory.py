#!/usr/bin/env python3
"""K5B.1: local, read-only paper inventory and sample-freeze tooling for
the K5 keyword-quality evaluation (see the K5A design report — this file
implements ONLY the inventory + sample-freeze boundary it describes, not
the annotation workbook or any candidate generation).

**Scope, strictly bounded**:
- Reads already-saved curation sessions via the exact same production
  checkpointer/session-loader path the app itself uses
  (`research_agent.curation_session.list_curation_sessions`/
  `load_curation_session`, via `research_agent.qa.sqlite_checkpointer`) --
  never a hand-rolled SQL query, never a checkpoint write of any kind.
- Never calls `research_agent.keywords.extract_keywords` -- no YAKE
  candidate is generated or exposed anywhere in this file. The two
  `_MIN_ABSTRACT_WORDS`/`_MIN_ABSTRACT_CHARS` constants ARE imported
  (to reuse the exact same "usable abstract" floor the extractor itself
  uses, rather than silently duplicating/drifting from it), but importing
  two integers is not running the extractor.
- Never prints or writes an abstract's own text or any keyword phrase
  VALUE to terminal output or to any file this script produces --  only
  counts, presence booleans, hashes, and metadata. "Stored keyword
  version" is recorded as `null` with an explicit note, never guessed,
  since guessing it would require re-running the extractor to compare
  against stored output -- exactly what this checkpoint is not allowed
  to do.
- Never calls a provider, performs a search, mutates a session, or
  touches `data/usage_telemetry.sqlite`.

**Known-failure sourcing (sampling safeguard).** `CONFIRMED_KNOWN_FAILURE_PAPER_IDS`
below is sourced directly from already-committed project documentation
(`docs/architecture.md`'s K4.1b section: the "Hai Phong University" and
"SLAC National Accelerator Laboratory" affiliation-leakage cases). A
second, clearly-labelled discovery mechanism (`_looks_like_fragment`)
additionally flags a paper as a *candidate* known failure only by
pattern-matching its ALREADY-STORED `Paper.keywords` values (real,
previously-computed production output already sitting in the checkpoint
-- reading it is not running the extractor) against a small, fixed list
of sentence-fragment boundary words. This is a discovery aid only, not a
new keyword-quality rule -- flagged papers are reported with
`known_failure_source="heuristic_stored_keyword_scan"`, distinct from
`"documented"`, and the sample-freeze step never silently promotes a
heuristic hit into the frozen 8 without it being visible as such in the
output.

**Two-step CLI**:
    python scripts/k5_paper_inventory.py inventory
        -> eval_working/paper_keywords/inventory.jsonl (safe to rerun any time)

    python scripts/k5_paper_inventory.py freeze-sample [--replace]
        -> eval_working/paper_keywords/proposed_sample.jsonl
           eval_working/paper_keywords/selection_rules.json
           (refuses to overwrite an existing frozen sample unless --replace)

    python scripts/k5_paper_inventory.py validate
        -> re-checks an already-frozen sample against inventory.jsonl,
           prints violations (if any) and exits non-zero if unsafe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.curation_session import list_curation_sessions, load_curation_session
from research_agent.keywords import _MIN_ABSTRACT_CHARS, _MIN_ABSTRACT_WORDS
from research_agent.qa import QA_CHECKPOINT_DB_PATH, sqlite_checkpointer
from research_agent.schema import Paper

INVENTORY_SCHEMA_VERSION = "k5b1-inventory-v1"
SAMPLE_SCHEMA_VERSION = "k5b1-sample-v1"
SELECTION_RULE_VERSION = "k5b1-v1"

# Frozen for K5B.1. Changing this seed changes the deterministic
# cross-domain/stress selection below -- treat it exactly like a rubric
# version: changing it requires --replace on an already-frozen sample,
# and any pre-existing frozen sample stays valid evidence of what THAT
# seed produced.
K5_RANDOM_SEED = 5202608

EVAL_WORKING_DIR = Path(__file__).resolve().parent.parent / "eval_working" / "paper_keywords"
INVENTORY_PATH = EVAL_WORKING_DIR / "inventory.jsonl"
SAMPLE_PATH = EVAL_WORKING_DIR / "proposed_sample.jsonl"
RULES_PATH = EVAL_WORKING_DIR / "selection_rules.json"

STRATA_TARGETS = {"known_failure": 8, "cross_domain": 8, "stress_case": 4}
PILOT_COUNT = 5
DOUBLE_REVIEW_COUNT = 5

# Sourced directly from docs/architecture.md's K4.1b section -- real,
# already-documented affiliation-leakage failures, not rediscovered here.
CONFIRMED_KNOWN_FAILURE_PAPER_IDS = frozenset({
    "8a30576ea1aebfbe9dd6e227a5c9427cf3040dff",  # "Hai Phong University"
    "7bbe04578073b4afebeffaab4bbd42f5132afe6a",  # "SLAC National Accelerator Laboratory"
})

# Discovery-only: a stored keyword phrase whose first or last token is one
# of these is flagged as "looks like a sentence fragment" -- matches the
# structural shape of the K5A report's own examples ("causing all
# tokens", "boom of linear", "fields from Natural", "introduced for
# natural", "networks purely based", "performance remains unclear").
# Deliberately small and boundary-position-only (never a mid-phrase
# check) -- this is not a proposed production rule, only an inventory
# signal for a human to confirm.
_FRAGMENT_BOUNDARY_WORDS = frozenset({
    "causing", "boom", "fields", "introduced", "purely", "remains",
    "based", "from", "for", "with", "of", "the", "and", "unclear",
})

# Deliberately small, topic-substring-only domain proxy -- no NER/LLM/
# embedding classifier. A paper's provenance topics are matched against
# these; a paper matching none is left unclassified (excluded from the
# cross-domain eligible pool, never guessed).
_DOMAIN_BUCKET_KEYWORDS: dict[str, frozenset[str]] = {
    "ml_nlp": frozenset({"language model", "nlp", "llm", "retrieval", "rag", "natural language", "transformer"}),
    "computer_vision": frozenset({"vision", "image", "video", "detection", "segmentation", "visual"}),
    "security_software": frozenset({"security", "vulnerability", "software engineering", "privacy", "attack"}),
    "applied_domain": frozenset({"accelerator", "biomolecular", "clinical", "education", "student", "physics", "chemistry", "biology"}),
}

_URL_RE = re.compile(r"https?://\S+")


@dataclass
class ProvenanceEntry:
    session_id: str
    topic: str
    stage: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_hash(title: str, abstract: str | None) -> str:
    blob = (title or "") + "\x00" + (abstract or "")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _abstract_word_char_counts(abstract: str | None) -> tuple[int, int]:
    if not abstract:
        return 0, 0
    stripped = _URL_RE.sub(" ", abstract).strip()
    return len(stripped.split()), len(stripped)


def _usable_abstract(abstract: str | None) -> bool:
    """Same floor `research_agent.keywords.extract_keywords` itself uses
    (imported, not duplicated) -- but this is a plain length check on
    already-local text, never a call into the extractor."""
    if not abstract:
        return False
    words, chars = _abstract_word_char_counts(abstract)
    return chars >= _MIN_ABSTRACT_CHARS and words >= _MIN_ABSTRACT_WORDS


def _stress_signals(title: str, abstract: str | None, provenance: list[ProvenanceEntry]) -> dict:
    words, chars = _abstract_word_char_counts(abstract)
    text = abstract or ""
    alnum_space = sum(1 for c in text if c.isalnum() or c.isspace())
    noisy_punctuation_ratio = round(1 - (alnum_space / len(text)), 4) if text else 0.0
    acronym_tokens = re.findall(r"\b[A-Z]{2,6}\b", f"{title} {text}")
    acronym_density = round(len(acronym_tokens) / words, 4) if words else 0.0
    distinct_topics = {p.topic.strip().lower() for p in provenance if p.topic}
    return {
        "abstract_word_count": words,
        "abstract_char_count": chars,
        "short_abstract": words > 0 and words < 60,
        "noisy_punctuation_ratio": noisy_punctuation_ratio,
        "acronym_density": acronym_density,
        "cross_topic_provenance_count": len(distinct_topics),
    }


def _looks_like_fragment(phrase: str) -> bool:
    tokens = phrase.strip().lower().split()
    if not tokens:
        return False
    return tokens[0] in _FRAGMENT_BOUNDARY_WORDS or tokens[-1] in _FRAGMENT_BOUNDARY_WORDS


def _domain_bucket_guess(provenance: list[ProvenanceEntry]) -> str | None:
    topics_lower = " ".join(p.topic.lower() for p in provenance if p.topic)
    for bucket, needles in _DOMAIN_BUCKET_KEYWORDS.items():
        if any(needle in topics_lower for needle in needles):
            return bucket
    return None


def _external_id(paper: Paper) -> str:
    return paper.doi or paper.url or paper.paper_id


def build_inventory() -> list[dict]:
    """Read-only: enumerates every locally-saved curation session
    (`list_curation_sessions`, a pure `checkpointer.list(None)` scan --
    the exact same primitive the production `/curation/reviews` GET
    endpoint already uses) and every Paper-bearing location inside each
    (`reserve`/`selected_papers`/`turn_history`, the same three locations
    `scripts/re_extract_keywords.py` already inspects), deduplicated by
    `paper_id`, with every session/topic occurrence recorded (many-to-
    many, never collapsed to "first seen"). No checkpoint is written --
    `load_curation_session` is a pure `graph.get_state()` read, confirmed
    by its own docstring and by this module's own tests below."""
    papers_by_id: dict[str, Paper] = {}
    provenance_by_id: dict[str, list[ProvenanceEntry]] = {}
    stored_keyword_counts: dict[str, int] = {}

    with sqlite_checkpointer(QA_CHECKPOINT_DB_PATH) as checkpointer:
        summaries = list_curation_sessions(checkpointer)
        for summary in summaries:
            session_id = summary["session_id"]
            session = load_curation_session(session_id, checkpointer)
            if session is None:
                continue
            topic = session.topic
            stage = session.stage

            def _record(paper: Paper, kw_count: int) -> None:
                papers_by_id.setdefault(paper.paper_id, paper)
                provenance_by_id.setdefault(paper.paper_id, []).append(
                    ProvenanceEntry(session_id=session_id, topic=topic, stage=stage)
                )
                # Keep the max observed count across occurrences -- a
                # cheap, honest "has this paper ever had keywords stored"
                # signal without claiming to know which extractor version
                # produced them.
                stored_keyword_counts[paper.paper_id] = max(stored_keyword_counts.get(paper.paper_id, 0), kw_count)

            for paper, _score in session.reserve:
                _record(paper, len(paper.keywords))
            for paper in session.selected_papers:
                _record(paper, len(paper.keywords))
            for entry in session.turn_history:
                for paper_dict, _score in entry.get("batch", []):
                    pid = paper_dict.get("paper_id")
                    if not pid:
                        continue
                    kw = paper_dict.get("keywords", [])
                    if pid not in papers_by_id:
                        papers_by_id[pid] = Paper(**paper_dict)
                    provenance_by_id.setdefault(pid, []).append(
                        ProvenanceEntry(session_id=session_id, topic=topic, stage=stage)
                    )
                    stored_keyword_counts[pid] = max(stored_keyword_counts.get(pid, 0), len(kw))

    records: list[dict] = []
    for paper_id, paper in papers_by_id.items():
        provenance = provenance_by_id.get(paper_id, [])
        # Dedup provenance entries (a paper served in the same session
        # across multiple turns would otherwise repeat identical rows).
        seen_prov = set()
        dedup_provenance = []
        for p in provenance:
            key = (p.session_id, p.topic, p.stage)
            if key in seen_prov:
                continue
            seen_prov.add(key)
            dedup_provenance.append(p)

        known_failure_source = None
        if paper_id in CONFIRMED_KNOWN_FAILURE_PAPER_IDS:
            known_failure_source = "documented"
        elif any(_looks_like_fragment(kw) for kw in paper.keywords):
            known_failure_source = "heuristic_stored_keyword_scan"

        records.append({
            "schema_version": INVENTORY_SCHEMA_VERSION,
            "paper_id": paper_id,
            "source": paper.source,
            "external_id": _external_id(paper),
            "title": paper.title,
            "usable_abstract": _usable_abstract(paper.abstract),
            "stored_keywords_present": stored_keyword_counts.get(paper_id, 0) > 0,
            "stored_keywords_count": stored_keyword_counts.get(paper_id, 0),
            "stored_keyword_version": None,
            "stored_keyword_version_note": "not determinable without re-extraction (out of scope for K5B.1)",
            "source_hash": _source_hash(paper.title, paper.abstract),
            "provenance": [
                {"session_id": p.session_id, "topic": p.topic, "stage": p.stage} for p in dedup_provenance
            ],
            "stress_signals": _stress_signals(paper.title, paper.abstract, dedup_provenance),
            "known_failure_candidate": known_failure_source is not None,
            "known_failure_source": known_failure_source,
            "domain_bucket_guess": _domain_bucket_guess(dedup_provenance),
            "inventoried_at": _now_iso(),
        })

    records.sort(key=lambda r: r["paper_id"])
    return records


def write_inventory(records: list[dict], path: Path | None = None) -> None:
    # `path` resolves against the CURRENT module-level INVENTORY_PATH at
    # call time, not a default baked in at function-definition time --
    # deliberately, so tests can monkeypatch `INVENTORY_PATH` (matching
    # every other function in this module, which already reads the
    # globals directly) without silently writing to the real project
    # path. A `path: Path = INVENTORY_PATH` default parameter would NOT
    # do this -- it evaluates once at import time and never sees a later
    # reassignment, confirmed directly while building this module's own
    # tests.
    resolved = path if path is not None else INVENTORY_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def load_inventory(path: Path | None = None) -> list[dict]:
    resolved = path if path is not None else INVENTORY_PATH
    records = []
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_sample(
    inventory: list[dict], seed: int = K5_RANDOM_SEED,
) -> tuple[list[dict], dict[str, list[str]], list[str], list[str]]:
    """Deterministic given `inventory` (already sorted by paper_id) and
    `seed`. Returns (selected_records_with_stratum_metadata, shortfalls).
    Never substitutes one stratum's shortfall with another stratum's
    papers -- a stratum that can't reach its target is reported short,
    not padded."""
    shortfalls: dict[str, list[str]] = {}

    # --- known_failure: confirmed docs first, heuristic hits next, both
    # sorted by paper_id for determinism. ---
    confirmed = sorted(r["paper_id"] for r in inventory if r["known_failure_source"] == "documented")
    heuristic = sorted(r["paper_id"] for r in inventory if r["known_failure_source"] == "heuristic_stored_keyword_scan")
    known_failure_ids = (confirmed + heuristic)[:STRATA_TARGETS["known_failure"]]
    if len(known_failure_ids) < STRATA_TARGETS["known_failure"]:
        shortfalls["known_failure"] = [
            f"requested {STRATA_TARGETS['known_failure']}, found {len(known_failure_ids)} "
            f"({len(confirmed)} documented + {len(heuristic)} heuristic)"
        ]

    used_ids = set(known_failure_ids)

    # --- stress_case: one per subtype, deterministic tie-break by
    # paper_id, from usable-abstract papers not already used. ---
    eligible_for_stress = [r for r in inventory if r["usable_abstract"] and r["paper_id"] not in used_ids]
    stress_case_ids: list[str] = []
    stress_subtypes_assigned: dict[str, str] = {}
    stress_shortfall_notes = []

    def _pick_extreme(pool: list[dict], key, reverse: bool) -> dict | None:
        candidates = sorted(pool, key=lambda r: (key(r), r["paper_id"]), reverse=reverse)
        return candidates[0] if candidates else None

    subtype_pickers = {
        "acronym_heavy": lambda pool: _pick_extreme(pool, lambda r: r["stress_signals"]["acronym_density"], True),
        "noisy_prose": lambda pool: _pick_extreme(pool, lambda r: r["stress_signals"]["noisy_punctuation_ratio"], True),
        "short_abstract": lambda pool: next(
            (r for r in sorted(pool, key=lambda x: x["paper_id"]) if r["stress_signals"]["short_abstract"]), None,
        ),
        "cross_disciplinary": lambda pool: _pick_extreme(pool, lambda r: r["stress_signals"]["cross_topic_provenance_count"], True),
    }
    for subtype, picker in subtype_pickers.items():
        pool = [r for r in eligible_for_stress if r["paper_id"] not in stress_case_ids]
        pick = picker(pool)
        if pick is None:
            stress_shortfall_notes.append(f"stress subtype {subtype!r}: no eligible local paper found")
            continue
        stress_case_ids.append(pick["paper_id"])
        stress_subtypes_assigned[pick["paper_id"]] = subtype

    if len(stress_case_ids) < STRATA_TARGETS["stress_case"]:
        shortfalls["stress_case"] = stress_shortfall_notes or [
            f"requested {STRATA_TARGETS['stress_case']}, found {len(stress_case_ids)}"
        ]

    used_ids |= set(stress_case_ids)

    # --- cross_domain: predeclared eligible pool (usable abstract, not
    # already used, has a domain_bucket_guess), fixed seed + stable
    # (paper_id-sorted) ordering before sampling. ---
    eligible_for_domain = sorted(
        (r for r in inventory if r["usable_abstract"] and r["paper_id"] not in used_ids and r["domain_bucket_guess"]),
        key=lambda r: r["paper_id"],
    )
    rng = Random(seed)
    shuffled = eligible_for_domain[:]
    rng.shuffle(shuffled)
    target = STRATA_TARGETS["cross_domain"]
    cross_domain_ids = [r["paper_id"] for r in shuffled[:target]]
    if len(cross_domain_ids) < target:
        shortfalls["cross_domain"] = [
            f"requested {target}, found {len(cross_domain_ids)} eligible (usable abstract + classified domain, unused)"
        ]

    by_id = {r["paper_id"]: r for r in inventory}
    selected: list[dict] = []
    all_ids = known_failure_ids + cross_domain_ids + stress_case_ids
    for paper_id in known_failure_ids:
        selected.append({"paper_id": paper_id, "stratum": "known_failure", "stress_subtype": None,
                          "domain_bucket": by_id[paper_id]["domain_bucket_guess"]})
    for paper_id in cross_domain_ids:
        selected.append({"paper_id": paper_id, "stratum": "cross_domain", "stress_subtype": None,
                          "domain_bucket": by_id[paper_id]["domain_bucket_guess"]})
    for paper_id in stress_case_ids:
        selected.append({"paper_id": paper_id, "stratum": "stress_case",
                          "stress_subtype": stress_subtypes_assigned.get(paper_id),
                          "domain_bucket": by_id[paper_id]["domain_bucket_guess"]})

    # Pilot (5, excluded from headline metrics) / double-review (5,
    # included, extra-labelled) -- disjoint, each spanning all 3 strata
    # present, deterministic (paper_id order within each stratum).
    def _spread_pick(ids_by_stratum: dict[str, list[str]], n: int, already_used: set[str]) -> list[str]:
        picked: list[str] = []
        strata_cycle = [s for s in ("known_failure", "cross_domain", "stress_case") if ids_by_stratum.get(s)]
        idx = {s: 0 for s in strata_cycle}
        while len(picked) < n and strata_cycle:
            progressed = False
            for s in list(strata_cycle):
                pool = [pid for pid in sorted(ids_by_stratum[s]) if pid not in already_used and pid not in picked]
                if idx[s] < len(pool):
                    picked.append(pool[idx[s]])
                    idx[s] += 1
                    progressed = True
                    if len(picked) >= n:
                        break
                else:
                    strata_cycle.remove(s)
            if not progressed:
                break
        return picked

    ids_by_stratum = {"known_failure": known_failure_ids, "cross_domain": cross_domain_ids, "stress_case": stress_case_ids}
    pilot_ids = _spread_pick(ids_by_stratum, PILOT_COUNT, set())
    double_review_ids = _spread_pick(ids_by_stratum, DOUBLE_REVIEW_COUNT, set(pilot_ids))

    for record in selected:
        record["metrics_role"] = "pilot_only" if record["paper_id"] in pilot_ids else "headline"
        record["is_double_reviewed"] = record["paper_id"] in double_review_ids
        record["source_hash"] = by_id[record["paper_id"]]["source_hash"]

    return selected, shortfalls, pilot_ids, double_review_ids


def _canonical_bytes_for_hash(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def freeze_sample(replace: bool = False) -> int:
    if not INVENTORY_PATH.exists():
        print(f"error: no inventory found at {INVENTORY_PATH} -- run the 'inventory' subcommand first.", file=sys.stderr)
        return 1

    if (SAMPLE_PATH.exists() or RULES_PATH.exists()) and not replace:
        print(
            f"refused: a frozen sample already exists ({SAMPLE_PATH.name}/{RULES_PATH.name}). "
            "Pass --replace to overwrite deliberately. Nothing was changed.",
            file=sys.stderr,
        )
        return 3

    inventory = load_inventory()
    selected, shortfalls, pilot_ids, double_review_ids = select_sample(inventory, seed=K5_RANDOM_SEED)

    strata_actual = {
        "known_failure": sum(1 for r in selected if r["stratum"] == "known_failure"),
        "cross_domain": sum(1 for r in selected if r["stratum"] == "cross_domain"),
        "stress_case": sum(1 for r in selected if r["stratum"] == "stress_case"),
    }

    frozen_at = _now_iso()
    paper_ids_sorted = sorted(r["paper_id"] for r in selected)
    source_hashes = {r["paper_id"]: r["source_hash"] for r in selected}

    rules_payload = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "selection_rule_version": SELECTION_RULE_VERSION,
        "random_seed": K5_RANDOM_SEED,
        "generated_at": frozen_at,
        "selected_at": frozen_at,
        "strata_targets": STRATA_TARGETS,
        "strata_actual": strata_actual,
        "total_selected": len(selected),
        "shortfalls": shortfalls,
        "pilot_ids": sorted(pilot_ids),
        "double_review_ids": sorted(double_review_ids),
        "paper_ids": paper_ids_sorted,
        "source_hashes": source_hashes,
    }
    manifest_sha256 = hashlib.sha256(_canonical_bytes_for_hash(rules_payload)).hexdigest()
    rules_payload["manifest_sha256"] = manifest_sha256

    for record in selected:
        record["schema_version"] = SAMPLE_SCHEMA_VERSION
        record["selection_rule_version"] = SELECTION_RULE_VERSION
        record["frozen_at"] = frozen_at

    EVAL_WORKING_DIR.mkdir(parents=True, exist_ok=True)
    with SAMPLE_PATH.open("w", encoding="utf-8") as f:
        for record in sorted(selected, key=lambda r: r["paper_id"]):
            f.write(json.dumps(record, sort_keys=True) + "\n")
    with RULES_PATH.open("w", encoding="utf-8") as f:
        json.dump(rules_payload, f, sort_keys=True, indent=2)
        f.write("\n")

    print(f"selected: {len(selected)} of 20 target")
    print(f"strata_actual: {strata_actual}")
    if shortfalls:
        print(f"SHORTFALLS (not silently substituted): {shortfalls}")
    print(f"manifest_sha256: {manifest_sha256}")
    return 0


def validate_frozen_sample() -> list[str]:
    """Re-checks an already-frozen sample against the inventory.
    Returns a list of violation strings (empty = valid)."""
    violations: list[str] = []
    if not RULES_PATH.exists() or not SAMPLE_PATH.exists():
        return ["no frozen sample found"]

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    sample = [json.loads(line) for line in SAMPLE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    inventory_by_id = {r["paper_id"]: r for r in load_inventory()} if INVENTORY_PATH.exists() else {}

    # manifest hash re-derivation
    rules_copy = dict(rules)
    claimed_hash = rules_copy.pop("manifest_sha256", None)
    recomputed_hash = hashlib.sha256(_canonical_bytes_for_hash(rules_copy)).hexdigest()
    if claimed_hash != recomputed_hash:
        violations.append(f"manifest_sha256 mismatch: recorded={claimed_hash} recomputed={recomputed_hash}")

    ids = [r["paper_id"] for r in sample]
    if len(ids) != len(set(ids)):
        violations.append("duplicate paper_id in proposed_sample.jsonl")

    valid_strata = {"known_failure", "cross_domain", "stress_case"}
    for record in sample:
        pid = record["paper_id"]
        if record.get("stratum") not in valid_strata:
            violations.append(f"{pid}: invalid or missing stratum {record.get('stratum')!r}")
        if not record.get("source_hash"):
            violations.append(f"{pid}: missing source_hash")
        if not record.get("frozen_at"):
            violations.append(f"{pid}: missing frozen_at")
        if pid in inventory_by_id:
            if not inventory_by_id[pid]["usable_abstract"]:
                violations.append(f"{pid}: selected but usable_abstract is False in inventory")
            if inventory_by_id[pid]["source_hash"] != record.get("source_hash"):
                violations.append(f"{pid}: source_hash does not match inventory")
            if not inventory_by_id[pid]["provenance"]:
                violations.append(f"{pid}: no provenance recorded in inventory")
        # No candidate-output-shaped field allowed on a sample row.
        forbidden_keys = {"keywords", "candidates", "abstract", "abstract_text"}
        present_forbidden = forbidden_keys & set(record.keys())
        if present_forbidden:
            violations.append(f"{pid}: forbidden embedded field(s) {sorted(present_forbidden)}")

    pilot_ids = set(rules.get("pilot_ids", []))
    double_review_ids = set(rules.get("double_review_ids", []))
    if pilot_ids & double_review_ids:
        violations.append("pilot_ids and double_review_ids overlap -- must be disjoint")
    if len(pilot_ids) != PILOT_COUNT and not rules.get("shortfalls"):
        violations.append(f"pilot_ids count {len(pilot_ids)} != {PILOT_COUNT}")
    if len(double_review_ids) != DOUBLE_REVIEW_COUNT and not rules.get("shortfalls"):
        violations.append(f"double_review_ids count {len(double_review_ids)} != {DOUBLE_REVIEW_COUNT}")

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="K5B.1 local paper inventory and sample-freeze tooling.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    freeze_parser = sub.add_parser("freeze-sample")
    freeze_parser.add_argument("--replace", action="store_true")
    sub.add_parser("validate")
    args = parser.parse_args(argv)

    if args.command == "inventory":
        records = build_inventory()
        write_inventory(records)
        print(f"inventoried {len(records)} unique local paper(s) -> {INVENTORY_PATH}")
        return 0

    if args.command == "freeze-sample":
        return freeze_sample(replace=args.replace)

    if args.command == "validate":
        violations = validate_frozen_sample()
        if violations:
            for v in violations:
                print(f"VIOLATION: {v}", file=sys.stderr)
            return 1
        print("valid: frozen sample passes all checks.")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
