"""Focused K5D.1b annotation-freeze tests.

Freezing consumes only the already-corrected, on-disk K5D.1a workbook
(eval_working is gitignored but present in this checked-out repo after
the human review pass) -- consistent with how test_k5_keyword_metrics.py
and test_k5_llm_filter_policy_analysis.py test their own frozen
artifacts against real on-disk evidence rather than rebuilding it.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from openpyxl import load_workbook

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import k5_heldout_annotation_freeze as freeze
from scripts import k5_heldout_selection as k5d1
from research_agent.qa import QA_CHECKPOINT_DB_PATH
from research_agent.telemetry import USAGE_DB_PATH
from tests._usage_db_fingerprint import fingerprint_usage_db


def _real_frozen() -> dict:
    assert freeze.FROZEN_PATH.exists(), "run `python -m scripts.k5_heldout_annotation_freeze freeze` first"
    return json.loads(freeze.FROZEN_PATH.read_text())


def test_frozen_annotation_reproduces_the_canonical_workbook_exactly():
    checkpoint_before = fingerprint_usage_db(QA_CHECKPOINT_DB_PATH)
    telemetry_before = fingerprint_usage_db(USAGE_DB_PATH)

    assert freeze.validate_reproducibility() == []

    assert fingerprint_usage_db(QA_CHECKPOINT_DB_PATH) == checkpoint_before
    assert fingerprint_usage_db(USAGE_DB_PATH) == telemetry_before


def test_frozen_payload_schema_reviewer_type_and_self_hash():
    data = _real_frozen()
    assert freeze.validate_frozen_payload(data) == []
    assert data["schema_version"] == freeze.FROZEN_SCHEMA
    assert data["reviewer_type"] == "ai_assisted_human_approved" == freeze.REVIEWER_TYPE
    assert data["held_out_count"] == 6
    assert len(data["papers"]) == 6 and len(data["candidates"]) == 36
    assert data["paper_codes"] == [f"H{i:02d}" for i in range(1, 7)]


def test_frozen_bindings_match_current_workbook_mapping_manifest_selection():
    data = _real_frozen()
    bindings = data["bindings"]
    assert bindings["workbook_sha256"] == freeze.file_hash(k5d1.WORKBOOK_PATH)
    assert bindings["candidate_mapping_sha256"] == freeze.file_hash(k5d1.MAPPING_PATH)
    assert bindings["annotation_manifest_sha256"] == freeze.file_hash(k5d1.MANIFEST_PATH)
    assert bindings["selection_sha256"] == freeze.file_hash(k5d1.SELECTION_PATH)
    assert bindings["source_snapshot_sha256"] == freeze.file_hash(k5d1.SOURCE_SNAPSHOT_PATH)
    assert data["extractor_version"] == k5d1.production_keywords.KEYWORD_EXTRACTOR_VERSION == "yake-v2"


def test_frozen_annotation_binds_prior_k5_evidence_identities():
    data = _real_frozen()
    prior = data["prior_k5_evidence"]
    assert len(prior["paper_ids"]) == 10
    assert prior["pilot_paper_codes"] == ["P01", "P02"]
    assert prior["headline_paper_codes"] == [f"P{n:02d}" for n in range(3, 11)]
    assert set(prior["bindings"]) >= {
        "frozen_annotation_sha256", "k5b_method_results_sha256",
        "k5c_prompt_sha256", "k5c_raw_results_sha256", "k5c_metrics_sha256", "k5c1_analysis_sha256",
    }


def test_self_hash_tampering_is_detected():
    data = _real_frozen()
    tampered = copy.deepcopy(data)
    tampered["candidates"][0]["decision"] = "uncertain" if tampered["candidates"][0]["decision"] != "uncertain" else "reject"
    errors = freeze.validate_frozen_payload(tampered)
    assert any("self-hash" in e for e in errors)


def test_binding_tampering_is_detected_by_reproducibility_check():
    data = _real_frozen()
    tampered = copy.deepcopy(data)
    tampered["bindings"]["workbook_sha256"] = "0" * 64
    tampered["frozen_annotation_sha256"] = freeze.payload_hash({k: v for k, v in tampered.items() if k != "frozen_annotation_sha256"})
    errors = freeze.validate_reproducibility(tampered)
    assert any("no longer matches" in e for e in errors)


def test_accept_reject_matched_concept_rules_are_enforced_on_synthetic_payload():
    base_paper = {"paper_code": "H01", "concepts": ["a", "b", "c", None, None], "reviewer_notes": None}
    good_candidate = {
        "candidate_id": "HC-AAA", "paper_code": "H01", "candidate_phrase": "x",
        "decision": "accept", "rejection_reason": None, "matched_concept_ids": "C1",
        "confidence": "high", "reviewer_notes": None,
    }
    payload = {
        "schema_version": freeze.FROZEN_SCHEMA, "status": "frozen_complete_heldout_annotation",
        "frozen_at": "t", "reviewer_type": freeze.REVIEWER_TYPE, "held_out_count": 1,
        "paper_codes": ["H01"], "extractor_version": "yake-v2",
        "bindings": {}, "prior_k5_evidence": {}, "papers": [base_paper], "candidates": [good_candidate],
    }

    def errors_for(candidate_overrides: dict) -> list[str]:
        candidate = {**good_candidate, **candidate_overrides}
        data = {**payload, "candidates": [candidate]}
        data["frozen_annotation_sha256"] = freeze.payload_hash(data)
        # Filter out the (expected, irrelevant here) count-mismatch noise from
        # using a single-paper/single-candidate payload instead of the real 6/36.
        return [e for e in freeze.validate_frozen_payload(data) if "count/identity mismatch" not in e]

    assert errors_for({}) == []
    assert any("requires at least one matched" in e for e in errors_for({"matched_concept_ids": None}))
    assert any("does not identify a populated concept" in e for e in errors_for({"matched_concept_ids": "C4"}))
    assert any(
        "must not carry matched concept ids" in e
        for e in errors_for({"decision": "reject", "rejection_reason": "fragment", "matched_concept_ids": "C1"})
    )
    assert any("semantic mismatch" in e for e in errors_for({"decision": "reject", "matched_concept_ids": None, "rejection_reason": None}))


def test_freeze_refuses_overwrite_without_replace():
    assert freeze.FROZEN_PATH.exists()
    before = freeze.file_hash(freeze.FROZEN_PATH)
    with pytest.raises(FileExistsError):
        freeze.freeze_annotations(replace=False)
    assert freeze.file_hash(freeze.FROZEN_PATH) == before


def test_reproducibility_detects_a_tampered_workbook_copy(tmp_path, monkeypatch):
    tampered_workbook = tmp_path / "tampered.xlsx"
    shutil.copyfile(k5d1.WORKBOOK_PATH, tampered_workbook)
    workbook = load_workbook(tampered_workbook)
    workbook["Candidate review"]["D2"] = "uncertain"
    workbook.save(tampered_workbook)
    monkeypatch.setattr(k5d1, "WORKBOOK_PATH", tampered_workbook)
    data = _real_frozen()
    errors = freeze.validate_reproducibility(data)
    assert any("exact readback" in e or "no longer matches" in e for e in errors)
