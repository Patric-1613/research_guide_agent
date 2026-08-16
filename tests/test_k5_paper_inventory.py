"""Focused K5B.1b frozen-screening and bounded-sample tests."""

from __future__ import annotations

import json
import os
import socket
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest
from openpyxl import load_workbook

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    "mechanisms to capture long range dependencies between atoms across five "
    "benchmark datasets while remaining computationally efficient at inference."
)
_SHORT_ABSTRACT = "A brief study of anomaly detection methods for industrial sensor deployments today."


def _paper(pid: str, title: str, abstract: str | None = _ABSTRACT, keywords: list[str] | None = None) -> Paper:
    return Paper(
        title=title,
        authors=["A"],
        year=2024,
        venue="X",
        abstract=abstract,
        url=None,
        doi=None,
        citation_count=None,
        source="arxiv",
        paper_id=pid,
        keywords=keywords or [],
    )


@pytest.fixture
def working_dirs(tmp_path, monkeypatch):
    db_path = tmp_path / "checkpoints.sqlite"
    working_dir = tmp_path / "working"
    monkeypatch.setattr(k5, "QA_CHECKPOINT_DB_PATH", db_path)
    monkeypatch.setattr(k5, "EVAL_WORKING_DIR", working_dir)
    monkeypatch.setattr(k5, "INVENTORY_PATH", working_dir / "inventory.jsonl")
    monkeypatch.setattr(k5, "SAMPLE_PATH", working_dir / "proposed_sample.jsonl")
    monkeypatch.setattr(k5, "RULES_PATH", working_dir / "selection_rules.json")
    monkeypatch.setattr(k5, "SCREENING_WORKBOOK_PATH", working_dir / "failure_screening.xlsx")
    monkeypatch.setattr(k5, "SCREENING_MANIFEST_PATH", working_dir / "failure_screening_manifest.json")
    monkeypatch.setattr(k5, "SOURCE_SNAPSHOT_PATH", working_dir / "bounded_source_snapshot.jsonl")
    return db_path, working_dir


def _save_session(db_path: Path, session_id: str, session: PaperPoolSession) -> None:
    with sqlite_checkpointer(db_path) as checkpointer:
        save_curation_session(session, session_id, checkpointer)


def _build_full_pool(db_path: Path) -> None:
    failures = [
        (_paper(f"failure-{index}", f"Failure paper {index}", keywords=[f"causing tokens {index}"]), 0.9)
        for index in range(8)
    ]
    _save_session(db_path, "screened-failures", PaperPoolSession(topic="quality review", reserve=failures, cursor=8))

    domain_topics = (
        ("nlp", "natural language retrieval"),
        ("vision", "image segmentation"),
        ("security", "software vulnerability detection"),
        ("applied", "clinical chemistry"),
        ("ambiguous", "natural language privacy"),
        ("unknown", "operations research"),
    )
    for prefix, topic in domain_topics:
        papers = [(_paper(f"{prefix}-{index}", f"{prefix} paper {index}"), 0.8) for index in range(4)]
        _save_session(db_path, f"domain-{prefix}", PaperPoolSession(topic=topic, reserve=papers, cursor=4))

    acronym = _paper("edge-acronym", "RAG NLP LLM GPU CPU TPU FPGA Study")
    noisy = _paper("edge-noisy", "Noisy prose", _ABSTRACT + " %%%^^^&&&***(((())))")
    short = _paper("edge-short", "Short abstract", _SHORT_ABSTRACT)
    cross = _paper("edge-cross", "Cross topic paper")
    _save_session(
        db_path,
        "edge-1",
        PaperPoolSession(
            topic="language model",
            reserve=[(acronym, 0.9), (noisy, 0.8), (short, 0.7), (cross, 0.6)],
            cursor=4,
        ),
    )
    _save_session(db_path, "edge-2", PaperPoolSession(topic="clinical education", reserve=[(cross, 0.9)], cursor=1))


def _set_screening_expectations(monkeypatch, row_count: int = 8, paper_count: int = 8) -> None:
    monkeypatch.setattr(k5, "EXPECTED_SCREENING_ROW_COUNT", row_count)
    monkeypatch.setattr(k5, "EXPECTED_SCREENING_PAPER_COUNT", paper_count)
    monkeypatch.setattr(k5, "EXPECTED_SCREENING_DECISIONS", {"yes": row_count})
    monkeypatch.setattr(k5, "EXPECTED_SCREENING_REASONS", {"fragment": row_count})


def _complete_screening(working_dir: Path, decisions: dict[str, str] | None = None) -> None:
    path = working_dir / "failure_screening.xlsx"
    workbook = load_workbook(path)
    sheet = workbook["Failure screening"]
    for row_number in range(2, sheet.max_row + 1):
        paper_id = sheet[f"A{row_number}"].value
        decision = decisions.get(paper_id, "no") if decisions is not None else "yes"
        sheet[f"K{row_number}"] = decision
        sheet[f"L{row_number}"] = "fragment" if decision == "yes" else None
    workbook.save(path)


def _prepare_frozen_screening(db_path: Path, working_dir: Path, monkeypatch) -> list[dict]:
    _build_full_pool(db_path)
    inventory = k5.build_inventory()
    k5.write_inventory(inventory)
    assert k5.export_failure_screening() == 0
    _complete_screening(working_dir)
    _set_screening_expectations(monkeypatch)
    return inventory


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_additive_live_papers_do_not_invalidate_frozen_screening(working_dirs, monkeypatch):
    db_path, working_dir = working_dirs
    inventory = _prepare_frozen_screening(db_path, working_dir, monkeypatch)
    added = _paper("new-live-paper", "New live paper", keywords=["causing new drift"])
    _save_session(db_path, "later-session", PaperPoolSession(topic="review", reserve=[(added, 0.9)], cursor=1))

    violations, confirmed = k5.validate_failure_screening(require_complete=True, inventory=inventory)
    assert violations == []
    assert len(confirmed) == 8
    drift = k5.audit_live_screening_drift()
    assert drift["status"] == "drift_observed"
    assert drift["added_paper_ids"] == ["new-live-paper"]
    assert drift["informational_only"] is True


def test_protected_workbook_tampering_still_fails(working_dirs, monkeypatch):
    db_path, working_dir = working_dirs
    inventory = _prepare_frozen_screening(db_path, working_dir, monkeypatch)
    workbook_path = working_dir / "failure_screening.xlsx"
    workbook = load_workbook(workbook_path)
    workbook["Failure screening"]["B2"] = "tampered title"
    workbook.save(workbook_path)

    violations, _ = k5.validate_failure_screening(require_complete=True, inventory=inventory)
    assert any("prefilled evidence hash mismatch" in violation for violation in violations)


def test_screening_manifest_tampering_still_fails(working_dirs, monkeypatch):
    db_path, working_dir = working_dirs
    inventory = _prepare_frozen_screening(db_path, working_dir, monkeypatch)
    path = working_dir / "failure_screening_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["row_count"] += 1
    path.write_text(json.dumps(manifest))

    violations, _ = k5.validate_failure_screening(require_complete=True, inventory=inventory)
    assert any("manifest_sha256 mismatch" in violation for violation in violations)


def test_incomplete_human_labels_block_freezing_without_replacing_artifacts(working_dirs, monkeypatch):
    db_path, working_dir = working_dirs
    _prepare_frozen_screening(db_path, working_dir, monkeypatch)
    workbook_path = working_dir / "failure_screening.xlsx"
    workbook = load_workbook(workbook_path)
    workbook["Failure screening"]["K2"] = None
    workbook["Failure screening"]["L2"] = None
    workbook.save(workbook_path)
    sample_path = working_dir / "proposed_sample.jsonl"
    sample_path.write_text("old sample\n")
    old_hash = k5._file_sha256(sample_path)

    assert k5.freeze_sample(replace=True) == 4
    assert k5._file_sha256(sample_path) == old_hash
    assert not (working_dir / "bounded_source_snapshot.jsonl").exists()


def test_only_human_confirmed_papers_enter_confirmed_failure(working_dirs):
    db_path, _ = working_dirs
    _build_full_pool(db_path)
    inventory = k5.build_inventory()
    confirmed = {"failure-0", "failure-1", "failure-2", "failure-3"}
    selected, shortfalls, _, _ = k5.select_sample(inventory, confirmed_screening_ids=confirmed)

    assert shortfalls == {}
    selected_failures = {row["paper_id"] for row in selected if row["stratum"] == "confirmed_failure"}
    assert selected_failures == confirmed
    assert all(row["paper_id"] in confirmed for row in selected if row["stratum"] == "confirmed_failure")


def test_bounded_selection_has_exact_4_4_2_and_ten_unique_papers(working_dirs):
    db_path, _ = working_dirs
    _build_full_pool(db_path)
    inventory = k5.build_inventory()
    confirmed = {f"failure-{index}" for index in range(8)}
    selected, shortfalls, _, _ = k5.select_sample(inventory, confirmed_screening_ids=confirmed)

    assert shortfalls == {}
    assert Counter(row["stratum"] for row in selected) == Counter(k5.STRATA_TARGETS)
    assert len(selected) == len({row["paper_id"] for row in selected}) == 10


def test_bounded_selection_is_deterministic_under_frozen_seed(working_dirs):
    db_path, _ = working_dirs
    _build_full_pool(db_path)
    inventory = k5.build_inventory()
    confirmed = {f"failure-{index}" for index in range(8)}

    first = k5.select_sample(inventory, seed=k5.K5_RANDOM_SEED, confirmed_screening_ids=confirmed)
    second = k5.select_sample(inventory, seed=k5.K5_RANDOM_SEED, confirmed_screening_ids=confirmed)
    assert first == second


def test_pilot_and_double_review_roles_are_exact_disjoint_and_span_strata(working_dirs):
    db_path, _ = working_dirs
    _build_full_pool(db_path)
    selected, _, pilot_ids, double_ids = k5.select_sample(
        k5.build_inventory(), confirmed_screening_ids={f"failure-{index}" for index in range(8)},
    )
    by_id = {row["paper_id"]: row for row in selected}

    assert len(pilot_ids) == 2 and len(double_ids) == 3
    assert not (set(pilot_ids) & set(double_ids))
    assert len({by_id[paper_id]["stratum"] for paper_id in pilot_ids}) >= 2
    assert len({by_id[paper_id]["stratum"] for paper_id in double_ids}) >= 2
    assert all(by_id[paper_id]["metrics_role"] == "pilot_only" for paper_id in pilot_ids)
    assert all(by_id[paper_id]["metrics_role"] == "headline" for paper_id in double_ids)


def test_live_drift_is_never_automatically_incorporated(working_dirs, monkeypatch):
    db_path, working_dir = working_dirs
    _prepare_frozen_screening(db_path, working_dir, monkeypatch)
    added_id = "new-live-paper"
    added = _paper(added_id, "New live paper", keywords=["causing new drift"])
    _save_session(db_path, "later-session", PaperPoolSession(topic="review", reserve=[(added, 0.9)], cursor=1))

    assert k5.freeze_sample(replace=True) == 0
    selected = _load_jsonl(working_dir / "proposed_sample.jsonl")
    snapshot = _load_jsonl(working_dir / "bounded_source_snapshot.jsonl")
    rules = json.loads((working_dir / "selection_rules.json").read_text())
    assert added_id not in {row["paper_id"] for row in selected}
    assert added_id not in {row["paper_id"] for row in snapshot}
    assert rules["informational_live_drift"]["added_paper_ids"] == [added_id]


def test_freeze_writes_hash_bound_sample_screening_and_source_snapshot(working_dirs, monkeypatch):
    db_path, working_dir = working_dirs
    _prepare_frozen_screening(db_path, working_dir, monkeypatch)
    assert k5.freeze_sample(replace=True) == 0

    rules = json.loads((working_dir / "selection_rules.json").read_text())
    assert rules["status"] == "frozen_bounded_sample"
    assert rules["sample_sha256"] == k5._file_sha256(working_dir / "proposed_sample.jsonl")
    assert rules["screening_workbook_sha256"] == k5._file_sha256(working_dir / "failure_screening.xlsx")
    assert rules["screening_manifest_sha256"] == k5._file_sha256(working_dir / "failure_screening_manifest.json")
    assert rules["inventory_sha256"] == k5._file_sha256(working_dir / "inventory.jsonl")
    assert rules["bounded_source_snapshot_sha256"] == k5._file_sha256(working_dir / "bounded_source_snapshot.jsonl")
    assert len(_load_jsonl(working_dir / "bounded_source_snapshot.jsonl")) == 10
    assert k5.validate_frozen_sample() == []


def test_tampered_source_snapshot_fails_frozen_validation(working_dirs, monkeypatch):
    db_path, working_dir = working_dirs
    _prepare_frozen_screening(db_path, working_dir, monkeypatch)
    assert k5.freeze_sample(replace=True) == 0
    snapshot_path = working_dir / "bounded_source_snapshot.jsonl"
    rows = _load_jsonl(snapshot_path)
    rows[0]["abstract"] = "tampered abstract"
    snapshot_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    violations = k5.validate_frozen_sample()
    assert any("snapshot hash" in violation or "source_hash mismatch" in violation for violation in violations)


def test_unresolvable_selected_source_stops_without_substitution(working_dirs, monkeypatch):
    db_path, working_dir = working_dirs
    _prepare_frozen_screening(db_path, working_dir, monkeypatch)
    with patch.object(k5, "_resolve_source_snapshot", return_value=([], ["missing source"])):
        assert k5.freeze_sample(replace=True) == 4
    assert not (working_dir / "bounded_source_snapshot.jsonl").exists()
    assert not (working_dir / "proposed_sample.jsonl").exists()


def test_observed_domain_composition_is_descriptive_not_cross_domain_claim(working_dirs, monkeypatch):
    db_path, working_dir = working_dirs
    _prepare_frozen_screening(db_path, working_dir, monkeypatch)
    assert k5.freeze_sample(replace=True) == 0
    sample = _load_jsonl(working_dir / "proposed_sample.jsonl")
    rules = json.loads((working_dir / "selection_rules.json").read_text())
    diversity = [row for row in sample if row["stratum"] == "product_local_diversity"]
    observed = Counter(
        row["domain_bucket_guess"] if row["domain_bucket_status"] == "unambiguous" else row["domain_bucket_status"]
        for row in diversity
    )
    assert rules["observed_product_local_domain_composition"] == dict(observed)
    assert "not an external benchmark" in rules["limitations"]
    assert "cross_domain" not in json.dumps(rules)


def test_old_sample_schema_fails_current_validation(working_dirs):
    _, working_dir = working_dirs
    working_dir.mkdir(parents=True)
    (working_dir / "inventory.jsonl").write_text(json.dumps({"schema_version": "k5b1a-inventory-v2", "paper_id": "old"}) + "\n")
    (working_dir / "proposed_sample.jsonl").write_text(json.dumps({
        "schema_version": "k5b1a-sample-v2", "selection_rule_version": "k5b1a-v2",
        "paper_id": "old", "stratum": "product_local_diversity", "source_hash": "x",
    }) + "\n")
    (working_dir / "selection_rules.json").write_text(json.dumps({
        "schema_version": "k5b1a-sample-v2", "selection_rule_version": "k5b1a-v2",
        "status": "approved", "paper_ids": ["old"], "manifest_sha256": "old",
    }))
    (working_dir / "bounded_source_snapshot.jsonl").write_text("{}\n")

    violations = k5.validate_frozen_sample()
    assert any("not current" in violation or "obsolete" in violation for violation in violations)


def test_no_extractor_or_network_call_in_freeze_path(working_dirs, monkeypatch):
    db_path, working_dir = working_dirs
    _prepare_frozen_screening(db_path, working_dir, monkeypatch)
    with (
        patch("research_agent.keywords.extract_keywords") as extract,
        patch.object(socket, "create_connection", side_effect=AssertionError("network call")),
    ):
        assert k5.freeze_sample(replace=True) == 0
        extract.assert_not_called()


def test_real_inventory_and_drift_reads_leave_checkpoint_and_telemetry_unchanged():
    checkpoint_before = fingerprint_usage_db(QA_CHECKPOINT_DB_PATH)
    telemetry_before = fingerprint_usage_db(USAGE_DB_PATH)

    k5.build_inventory()
    k5.audit_live_screening_drift()

    assert fingerprint_usage_db(QA_CHECKPOINT_DB_PATH) == checkpoint_before
    assert fingerprint_usage_db(USAGE_DB_PATH) == telemetry_before
