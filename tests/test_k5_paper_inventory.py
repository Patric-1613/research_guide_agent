"""K5B.1: tests for scripts/k5_paper_inventory.py -- the read-only local
paper inventory and sample-freeze tooling for the K5 keyword-quality
evaluation. Real SQLite reads throughout (via sqlite_checkpointer/
save_curation_session, the exact production path), against temporary,
synthetic checkpointers only -- never the real project database, except
in the one test explicitly proving the real database is untouched.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import scripts.k5_paper_inventory as k5
from research_agent.curation_session import save_curation_session
from research_agent.qa import QA_CHECKPOINT_DB_PATH, sqlite_checkpointer
from research_agent.query_expansion import PaperPoolSession
from research_agent.schema import Paper
from research_agent.telemetry import USAGE_DB_PATH
from tests._usage_db_fingerprint import fingerprint_usage_db

_ABSTRACT = (
    "We present a comprehensive study of graph neural networks for molecular "
    "property prediction. Our method combines message passing with attention "
    "mechanisms to capture long-range dependencies between atoms across five "
    "benchmark datasets, improving mean absolute error by a substantial margin "
    "while remaining computationally efficient at inference time."
)

_SHORT_ABSTRACT = "A brief study of anomaly detection methods for industrial sensor deployments today."


def _paper(pid: str, title: str, abstract: str | None = _ABSTRACT, keywords: list[str] | None = None) -> Paper:
    return Paper(
        title=title, authors=["A"], year=2024, venue="X", abstract=abstract,
        url=None, doi=None, citation_count=None, source="arxiv", paper_id=pid,
        keywords=keywords or [],
    )


@pytest.fixture
def working_dirs(tmp_path, monkeypatch):
    """Redirects every path this module writes/reads to a tmp_path,
    matching the "path resolves at call time" fix -- monkeypatch.setattr
    on the module attribute is exactly the scenario that fix exists for."""
    db_path = tmp_path / "checkpoints.sqlite"
    working_dir = tmp_path / "working"
    monkeypatch.setattr(k5, "QA_CHECKPOINT_DB_PATH", db_path)
    monkeypatch.setattr(k5, "EVAL_WORKING_DIR", working_dir)
    monkeypatch.setattr(k5, "INVENTORY_PATH", working_dir / "inventory.jsonl")
    monkeypatch.setattr(k5, "SAMPLE_PATH", working_dir / "proposed_sample.jsonl")
    monkeypatch.setattr(k5, "RULES_PATH", working_dir / "selection_rules.json")
    return db_path, working_dir


def _save_session(db_path: Path, session_id: str, session: PaperPoolSession) -> None:
    with sqlite_checkpointer(db_path) as cp:
        save_curation_session(session, session_id, cp)


# ---------------------------------------------------------------------------
# build_inventory: read-only, deduplication, provenance, stress signals
# ---------------------------------------------------------------------------


def test_inventory_deduplicates_a_paper_seen_across_two_sessions_and_records_both_provenance(working_dirs):
    db_path, _ = working_dirs
    p1a = _paper("p1", "Federated Anomaly Detection", _ABSTRACT)
    p1b = _paper("p1", "Federated Anomaly Detection", _ABSTRACT)
    _save_session(db_path, "sidA", PaperPoolSession(topic="federated learning security", reserve=[(p1a, 0.9)], cursor=1))
    _save_session(db_path, "sidB", PaperPoolSession(topic="anomaly detection systems", reserve=[(p1b, 0.9)], cursor=1))

    records = k5.build_inventory()

    assert len(records) == 1
    record = records[0]
    assert record["paper_id"] == "p1"
    session_ids = {p["session_id"] for p in record["provenance"]}
    topics = {p["topic"] for p in record["provenance"]}
    assert session_ids == {"sidA", "sidB"}
    assert topics == {"federated learning security", "anomaly detection systems"}


def test_inventory_covers_reserve_selected_papers_and_turn_history(working_dirs):
    db_path, _ = working_dirs
    p_reserve = _paper("p_reserve", "Reserve Paper", _ABSTRACT)
    p_selected = _paper("p_selected", "Selected Paper", _ABSTRACT)
    p_history = _paper("p_history", "History Paper", _ABSTRACT)
    session = PaperPoolSession(
        topic="t", reserve=[(p_reserve, 0.9)], cursor=1,
        selected_papers=[p_selected], selected_paper_ids=["p_selected"],
        turn_history=[{"turn_number": 1, "batch": [[p_history.to_dict(), 0.9]], "refilled": False}],
    )
    _save_session(db_path, "sid1", session)

    records = k5.build_inventory()
    ids = {r["paper_id"] for r in records}
    assert ids == {"p_reserve", "p_selected", "p_history"}


def test_inventory_never_includes_abstract_text_or_keyword_phrase_values(working_dirs):
    db_path, _ = working_dirs
    paper = _paper("p1", "A Paper", _ABSTRACT, keywords=["a genuinely useful keyword phrase"])
    _save_session(db_path, "sid1", PaperPoolSession(topic="t", reserve=[(paper, 0.9)], cursor=1))

    records = k5.build_inventory()
    blob = json.dumps(records)
    assert _ABSTRACT not in blob
    assert "a genuinely useful keyword phrase" not in blob
    # Only counts/booleans about stored keywords, never the values.
    assert records[0]["stored_keywords_count"] == 1
    assert records[0]["stored_keywords_present"] is True
    assert records[0]["stored_keyword_version"] is None


def test_inventory_usable_abstract_matches_extractor_floor_without_calling_the_extractor(working_dirs):
    db_path, _ = working_dirs
    usable = _paper("p_usable", "Usable", _ABSTRACT)
    unusable = _paper("p_short", "Too Short", "A short note.")
    missing = _paper("p_none", "No Abstract", None)
    session = PaperPoolSession(
        topic="t", reserve=[(usable, 0.9), (unusable, 0.8), (missing, 0.7)], cursor=3,
    )
    _save_session(db_path, "sid1", session)

    with patch("research_agent.keywords.extract_keywords") as mock_extract:
        records = k5.build_inventory()
        mock_extract.assert_not_called()  # the public extraction API is never called

    by_id = {r["paper_id"]: r for r in records}
    assert by_id["p_usable"]["usable_abstract"] is True
    assert by_id["p_short"]["usable_abstract"] is False
    assert by_id["p_none"]["usable_abstract"] is False


def test_inventory_flags_documented_known_failure_and_heuristic_fragment_separately(working_dirs):
    db_path, _ = working_dirs
    documented = _paper("8a30576ea1aebfbe9dd6e227a5c9427cf3040dff", "Documented Failure", _ABSTRACT)
    heuristic = _paper("p_frag", "Heuristic Fragment Paper", _ABSTRACT, keywords=["causing all tokens"])
    clean = _paper("p_clean", "Clean Paper", _ABSTRACT, keywords=["gradient compression"])
    session = PaperPoolSession(topic="t", reserve=[(documented, 0.9), (heuristic, 0.8), (clean, 0.7)], cursor=3)
    _save_session(db_path, "sid1", session)

    records = k5.build_inventory()
    by_id = {r["paper_id"]: r for r in records}
    assert by_id["8a30576ea1aebfbe9dd6e227a5c9427cf3040dff"]["known_failure_source"] == "documented"
    assert by_id["p_frag"]["known_failure_source"] == "heuristic_stored_keyword_scan"
    assert by_id["p_clean"]["known_failure_source"] is None
    assert by_id["p_clean"]["known_failure_candidate"] is False


def test_inventory_stress_signals_are_computed_without_the_extractor(working_dirs):
    db_path, _ = working_dirs
    acronym_heavy = _paper("p_acr", "RAG NLP LLM GPU CPU Study", _ABSTRACT)
    short = _paper("p_short_ok", "Short Usable", _SHORT_ABSTRACT)
    session = PaperPoolSession(topic="t", reserve=[(acronym_heavy, 0.9), (short, 0.8)], cursor=2)
    _save_session(db_path, "sid1", session)

    records = k5.build_inventory()
    by_id = {r["paper_id"]: r for r in records}
    assert by_id["p_acr"]["stress_signals"]["acronym_density"] > 0
    assert by_id["p_short_ok"]["stress_signals"]["short_abstract"] is True


def test_inventory_domain_bucket_guess_from_topic_only_no_model(working_dirs):
    db_path, _ = working_dirs
    cv_paper = _paper("p_cv", "A Vision Paper", _ABSTRACT)
    unclassified = _paper("p_unknown", "Ambiguous Paper", _ABSTRACT)
    _save_session(db_path, "sidA", PaperPoolSession(topic="image segmentation and detection", reserve=[(cv_paper, 0.9)], cursor=1))
    _save_session(db_path, "sidB", PaperPoolSession(topic="a topic with no domain keywords", reserve=[(unclassified, 0.9)], cursor=1))

    records = k5.build_inventory()
    by_id = {r["paper_id"]: r for r in records}
    assert by_id["p_cv"]["domain_bucket_guess"] == "computer_vision"
    assert by_id["p_unknown"]["domain_bucket_guess"] is None


# ---------------------------------------------------------------------------
# select_sample / freeze_sample: shortfalls, disjoint pilot/double-review,
# strata, determinism, refuse-without-replace
# ---------------------------------------------------------------------------


def _large_synthetic_pool(db_path: Path) -> None:
    """Enough distinct, usable-abstract, domain-classified papers to
    reach all three strata targets, deterministically."""
    papers = []
    for i in range(10):
        papers.append((_paper(f"nlp{i}", f"NLP Paper {i}", _ABSTRACT), 0.9))
    session_nlp = PaperPoolSession(topic="natural language processing retrieval", reserve=papers, cursor=len(papers))
    _save_session(db_path, "sid_nlp", session_nlp)

    cv_papers = [(_paper(f"cv{i}", f"Vision Paper {i}", _ABSTRACT), 0.9) for i in range(10)]
    _save_session(db_path, "sid_cv", PaperPoolSession(topic="image detection and segmentation", reserve=cv_papers, cursor=len(cv_papers)))

    sec_papers = [(_paper(f"sec{i}", f"Security Paper {i}", _ABSTRACT), 0.9) for i in range(10)]
    _save_session(db_path, "sid_sec", PaperPoolSession(topic="software security vulnerability", reserve=sec_papers, cursor=len(sec_papers)))

    applied_papers = [(_paper(f"app{i}", f"Applied Paper {i}", _ABSTRACT), 0.9) for i in range(4)]
    _save_session(db_path, "sid_app", PaperPoolSession(topic="clinical education biomolecular", reserve=applied_papers, cursor=len(applied_papers)))

    fail1 = _paper("8a30576ea1aebfbe9dd6e227a5c9427cf3040dff", "Documented Failure One", _ABSTRACT)
    fail2 = _paper("7bbe04578073b4afebeffaab4bbd42f5132afe6a", "Documented Failure Two", _ABSTRACT)
    heuristic_fails = [
        _paper(f"heur{i}", f"Heuristic Failure {i}", _ABSTRACT, keywords=["causing all tokens"]) for i in range(6)
    ]
    _save_session(
        db_path, "sid_fail",
        PaperPoolSession(topic="failure review", reserve=[(fail1, 0.9), (fail2, 0.9)] + [(p, 0.9) for p in heuristic_fails], cursor=8),
    )

    stress_acronym = _paper("stress_acr", "RAG NLP LLM GPU CPU TPU FPGA Study", _ABSTRACT)
    stress_short = _paper("stress_short", "Short Stress", _SHORT_ABSTRACT)
    stress_noisy = _paper("stress_noisy", "Noisy!!! Study@@@###", _ABSTRACT + " %%%^^^&&&***(((")
    _save_session(db_path, "sid_stress1", PaperPoolSession(topic="natural language processing edge case", reserve=[(stress_acronym, 0.9)], cursor=1))
    _save_session(db_path, "sid_stress2", PaperPoolSession(topic="image detection edge case", reserve=[(stress_short, 0.9), (stress_noisy, 0.9)], cursor=2))
    # A paper seen under many distinct topics -- cross_topic_provenance signal.
    cross = _paper("stress_cross", "Cross Disciplinary", _ABSTRACT)
    for i, t in enumerate(["natural language processing", "image detection", "software security", "clinical education"]):
        _save_session(db_path, f"sid_cross{i}", PaperPoolSession(topic=t, reserve=[(cross, 0.9)], cursor=1))


def test_freeze_sample_is_deterministic_across_two_runs(working_dirs):
    db_path, _ = working_dirs
    _large_synthetic_pool(db_path)

    records = k5.build_inventory()
    selected_a, shortfalls_a, pilot_a, dr_a = k5.select_sample(records, seed=k5.K5_RANDOM_SEED)
    selected_b, shortfalls_b, pilot_b, dr_b = k5.select_sample(records, seed=k5.K5_RANDOM_SEED)

    assert [r["paper_id"] for r in selected_a] == [r["paper_id"] for r in selected_b]
    assert shortfalls_a == shortfalls_b
    assert pilot_a == pilot_b
    assert dr_a == dr_b


def test_freeze_sample_reports_shortfall_truthfully_never_substitutes(working_dirs):
    db_path, working_dir = working_dirs
    # Deliberately sparse pool: only 1 known-failure paper available.
    only_fail = _paper("8a30576ea1aebfbe9dd6e227a5c9427cf3040dff", "Documented Failure", _ABSTRACT)
    other = _paper("p_other", "Other Paper", _ABSTRACT)
    _save_session(db_path, "sid1", PaperPoolSession(topic="natural language processing", reserve=[(only_fail, 0.9), (other, 0.9)], cursor=2))

    records = k5.build_inventory()
    k5.write_inventory(records)
    rc = k5.freeze_sample()

    assert rc == 0
    rules = json.loads((working_dir / "selection_rules.json").read_text())
    assert "known_failure" in rules["shortfalls"]
    assert rules["strata_actual"]["known_failure"] == 1
    assert rules["total_selected"] < 20
    # Never padded with a cross_domain paper mislabeled as known_failure.
    sample = [json.loads(line) for line in (working_dir / "proposed_sample.jsonl").read_text().splitlines()]
    known_failure_ids = {r["paper_id"] for r in sample if r["stratum"] == "known_failure"}
    assert known_failure_ids == {"8a30576ea1aebfbe9dd6e227a5c9427cf3040dff"}


def test_freeze_sample_pilot_and_double_review_are_disjoint_and_span_strata(working_dirs):
    db_path, working_dir = working_dirs
    _large_synthetic_pool(db_path)
    records = k5.build_inventory()
    k5.write_inventory(records)
    rc = k5.freeze_sample()
    assert rc == 0

    rules = json.loads((working_dir / "selection_rules.json").read_text())
    pilot_ids = set(rules["pilot_ids"])
    double_review_ids = set(rules["double_review_ids"])
    assert not (pilot_ids & double_review_ids)
    assert len(pilot_ids) == k5.PILOT_COUNT
    assert len(double_review_ids) == k5.DOUBLE_REVIEW_COUNT

    sample = [json.loads(line) for line in (working_dir / "proposed_sample.jsonl").read_text().splitlines()]
    by_id = {r["paper_id"]: r for r in sample}
    pilot_strata = {by_id[pid]["stratum"] for pid in pilot_ids}
    double_review_strata = {by_id[pid]["stratum"] for pid in double_review_ids}
    assert len(pilot_strata) > 1, "pilot IDs should span more than one stratum"
    assert len(double_review_strata) > 1, "double-review IDs should span more than one stratum"

    for pid in pilot_ids:
        assert by_id[pid]["metrics_role"] == "pilot_only"
    for pid in (set(by_id) - pilot_ids):
        assert by_id[pid]["metrics_role"] == "headline"


def test_freeze_sample_refuses_to_overwrite_without_replace_flag(working_dirs):
    db_path, working_dir = working_dirs
    _large_synthetic_pool(db_path)
    records = k5.build_inventory()
    k5.write_inventory(records)

    rc1 = k5.freeze_sample()
    assert rc1 == 0
    original_rules = (working_dir / "selection_rules.json").read_text()

    rc2 = k5.freeze_sample(replace=False)
    assert rc2 == 3
    assert (working_dir / "selection_rules.json").read_text() == original_rules  # untouched

    rc3 = k5.freeze_sample(replace=True)
    assert rc3 == 0  # explicit replace succeeds


def test_manifest_sha256_changes_if_frozen_file_is_tampered_with(working_dirs):
    db_path, working_dir = working_dirs
    _large_synthetic_pool(db_path)
    records = k5.build_inventory()
    k5.write_inventory(records)
    k5.freeze_sample()

    violations = k5.validate_frozen_sample()
    assert violations == []

    rules_path = working_dir / "selection_rules.json"
    rules = json.loads(rules_path.read_text())
    rules["paper_ids"].append("tampered-extra-id")  # mutate content without recomputing the hash
    rules_path.write_text(json.dumps(rules))

    violations = k5.validate_frozen_sample()
    assert any("manifest_sha256 mismatch" in v for v in violations)


def test_validate_flags_selected_paper_with_unusable_abstract(working_dirs):
    db_path, working_dir = working_dirs
    _large_synthetic_pool(db_path)
    records = k5.build_inventory()
    k5.write_inventory(records)
    k5.freeze_sample()

    sample_path = working_dir / "proposed_sample.jsonl"
    lines = sample_path.read_text().splitlines()
    tampered = json.loads(lines[0])
    # Poison the inventory entry this sample row points to.
    inventory = k5.load_inventory()
    for rec in inventory:
        if rec["paper_id"] == tampered["paper_id"]:
            rec["usable_abstract"] = False
    k5.write_inventory(inventory)

    violations = k5.validate_frozen_sample()
    assert any(f"{tampered['paper_id']}: selected but usable_abstract is False" in v for v in violations)


def test_no_yake_extraction_anywhere_in_the_inventory_or_freeze_path(working_dirs):
    db_path, working_dir = working_dirs
    _large_synthetic_pool(db_path)
    with patch("research_agent.keywords.extract_keywords") as mock_extract:
        records = k5.build_inventory()
        k5.write_inventory(records)
        rc = k5.freeze_sample()
        mock_extract.assert_not_called()
    assert rc == 0


def test_no_checkpoint_write_occurs_during_inventory_build(working_dirs):
    """A fresh checkpointer over the SAME db_path, opened only for a
    read, must see byte-identical checkpoint rows before and after
    build_inventory() runs -- proves list_curation_sessions/
    load_curation_session never write."""
    import sqlite3

    db_path, _ = working_dirs
    _large_synthetic_pool(db_path)

    def _row_count() -> int:
        conn = sqlite3.connect(db_path)
        try:
            return conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        finally:
            conn.close()

    before = _row_count()
    k5.build_inventory()
    after = _row_count()
    assert before == after


# ---------------------------------------------------------------------------
# Proof against the REAL project database (read-only) -- the one test in
# this file that touches the real QA_CHECKPOINT_DB_PATH/USAGE_DB_PATH,
# specifically to prove neither is mutated by a real invocation.
# ---------------------------------------------------------------------------


def test_real_checkpoint_and_telemetry_databases_are_unchanged_by_a_real_inventory_run():
    checkpoint_fp_before = fingerprint_usage_db(QA_CHECKPOINT_DB_PATH)
    telemetry_fp_before = fingerprint_usage_db(USAGE_DB_PATH)

    k5.build_inventory()  # real QA_CHECKPOINT_DB_PATH, real read path, no monkeypatching

    checkpoint_fp_after = fingerprint_usage_db(QA_CHECKPOINT_DB_PATH)
    telemetry_fp_after = fingerprint_usage_db(USAGE_DB_PATH)

    assert checkpoint_fp_before == checkpoint_fp_after
    assert telemetry_fp_before == telemetry_fp_after
