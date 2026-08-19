"""Focused K5D.1c tests: independent verification that the metrics
script's frozen-YAKE-v2-vs-Policy-C comparison recomputes correctly
from the real, already-completed K5D.1 evidence -- and that its pure
per-paper logic (denominators, uncertain retention, fail-open
accounting, the H04 zero-accepted-baseline edge case) holds on
synthetic data too. No test here makes a provider/network call or a
second live run; the six approved calls already happened.
"""
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import k5_heldout_annotation_freeze as freeze
from scripts import k5_heldout_llm_metrics as metrics
from scripts import k5_heldout_llm_prep as prep
from scripts import k5_heldout_selection as k5d1
from scripts import k5_llm_filter_eval as k5c
from research_agent.qa import QA_CHECKPOINT_DB_PATH
from research_agent.telemetry import USAGE_DB_PATH
from tests._usage_db_fingerprint import fingerprint_usage_db

EXPECTED = {
    "baseline_candidates": 36,
    "retained_candidates": 19,
    "human_accepted_baseline": 11,
    "human_accepted_retained": 10,
    "baseline_precision": 0.305556,
    "filtered_precision": 0.526316,
    "precision_improvement_pp": 22.08,
    "accepted_keyword_retention": 0.909091,
    "rejected_keyword_removal": 0.64,
    "false_removal_rate": 0.090909,
    "baseline_macro_coverage": 0.377778,
    "filtered_macro_coverage": 0.344444,
    "coverage_retention": 0.911765,
    "uncertain_decisions": 2,
    "provider_failures": 0,
}


def _real_metrics() -> dict:
    assert metrics.METRICS_PATH.exists(), "run `python -m scripts.k5_heldout_llm_metrics metrics` first"
    return json.loads(metrics.METRICS_PATH.read_text())


def _round(value: float, places: int = 6) -> float:
    return round(value, places)


# ---------------------------------------------------------------------------
# Exact recomputation from the real frozen inputs (independent of the script)
# ---------------------------------------------------------------------------

def test_exact_recomputation_from_frozen_inputs_matches_expected_numbers():
    checkpoint_before = fingerprint_usage_db(QA_CHECKPOINT_DB_PATH)
    telemetry_before = fingerprint_usage_db(USAGE_DB_PATH)

    data = _real_metrics()

    assert data["frozen_yake_v2"]["candidate_count"] == EXPECTED["baseline_candidates"]
    assert data["frozen_policy_c"]["retained_candidate_count"] == EXPECTED["retained_candidates"]
    assert data["frozen_yake_v2"]["accept"] == EXPECTED["human_accepted_baseline"]
    assert data["frozen_policy_c"]["accept"] == EXPECTED["human_accepted_retained"]
    assert _round(data["frozen_yake_v2"]["resolved_precision"]) == _round(EXPECTED["baseline_precision"])
    assert _round(data["frozen_policy_c"]["resolved_precision"]) == _round(EXPECTED["filtered_precision"])
    assert round(data["frozen_policy_c"]["resolved_precision_delta"] * 100, 2) == EXPECTED["precision_improvement_pp"]
    assert _round(data["frozen_policy_c"]["accepted_keyword_retention"]) == _round(EXPECTED["accepted_keyword_retention"])
    assert _round(data["frozen_policy_c"]["rejected_keyword_removal_rate"]) == _round(EXPECTED["rejected_keyword_removal"])
    assert _round(data["frozen_policy_c"]["false_removal_rate"]) == _round(EXPECTED["false_removal_rate"])
    assert data["frozen_yake_v2"]["macro_concept_coverage"] == pytest.approx(EXPECTED["baseline_macro_coverage"], abs=1e-4)
    assert data["frozen_policy_c"]["macro_concept_coverage"] == pytest.approx(EXPECTED["filtered_macro_coverage"], abs=1e-4)
    assert data["frozen_policy_c"]["macro_concept_coverage_retention"] == pytest.approx(EXPECTED["coverage_retention"], abs=1e-4)
    assert data["frozen_policy_c"]["llm_uncertain_decisions"] == EXPECTED["uncertain_decisions"]
    assert data["frozen_policy_c"]["malformed_or_failed_calls"] == EXPECTED["provider_failures"]
    assert data["per_paper_failures"] == []

    assert fingerprint_usage_db(QA_CHECKPOINT_DB_PATH) == checkpoint_before
    assert fingerprint_usage_db(USAGE_DB_PATH) == telemetry_before


def test_all_frozen_gate_checks_pass_and_conclusion_matches():
    data = _real_metrics()
    assert data["provisional_gate"]["passed"] is True
    assert all(data["provisional_gate"]["checks"].values())
    assert data["conclusion"] == (
        "held-out validation passed: Policy C may proceed to a guarded, off-by-default production pilot"
    )


def test_uncertain_decisions_are_both_retained_in_the_real_run():
    raw = json.loads(prep.RAW_PATH.read_text())
    uncertain_ids = [
        row["candidate_id"] for call in raw["calls"] for row in call.get("results", [])
        if row["decision"] == "uncertain"
    ]
    assert len(uncertain_ids) == EXPECTED["uncertain_decisions"]
    for call in raw["calls"]:
        removed = prep.apply_policy_c(call, [r["candidate_id"] for r in call.get("results", [])])
        assert removed.isdisjoint(uncertain_ids)


# ---------------------------------------------------------------------------
# Gate thresholds are reused unchanged from the already-frozen K5C gate
# ---------------------------------------------------------------------------

def test_gate_thresholds_are_the_unmodified_frozen_k5c_gate():
    data = _real_metrics()
    # Read back from JSON, so identity doesn't survive the round-trip -- compare by value.
    assert data["provisional_gate"]["definition"] == k5c.PROVISIONAL_GATE
    assert data["provisional_gate"]["definition"] == {
        "resolved_precision_improvement_minimum": 0.08,
        "accepted_candidate_retention_minimum": 0.90,
        "macro_concept_coverage_retention_minimum": 0.90,
        "every_paper_retains_an_accepted_keyword": True,
        "malformed_or_failed_calls_maximum": 0,
        "interpretation": "provisional product gate; not a statistical claim",
    }


# ---------------------------------------------------------------------------
# H04 zero-accepted-baseline semantics: wording and vacuous-truth correctness
# ---------------------------------------------------------------------------

def test_h04_is_recorded_as_zero_accepted_baseline_not_as_retention_evidence():
    data = _real_metrics()
    h04 = next(row for row in data["per_paper"] if row["paper_code"] == "H04")
    assert h04["baseline"]["accept"] == 0
    assert h04["filtered"]["accept"] == 0
    assert data["provisional_gate"]["papers_with_zero_accepted_baseline_keywords"] == ["H04"]
    assert data["provisional_gate"]["safety_condition_wording"] == (
        "Every paper that had at least one accepted YAKE-v2 keyword retained at least one."
    )
    check_key = "every_paper_with_an_accepted_baseline_keyword_retained_one"
    assert check_key in data["provisional_gate"]["checks"]
    assert "retains_an_accepted_keyword" not in "".join(data["provisional_gate"]["checks"])  # old, misleading key gone


def test_zero_accepted_baseline_paper_does_not_force_a_gate_failure():
    """A paper with zero accepted YAKE-v2 candidates must be vacuously
    satisfied by the per-paper safety check -- never penalized, and never
    claimed as evidence Policy C preserved something for it."""
    candidates = [{"candidate_id": f"K{i}"} for i in range(3)]
    judgments = {f"K{i}": {"decision": "reject", "matched_concept_ids": None} for i in range(3)}
    concepts = ["c1", "c2", "c3", None, None]
    call_all_removed = {"status": "success", "results": [
        {"candidate_id": f"K{i}", "decision": "sentence_fragment"} for i in range(3)
    ]}
    row = metrics.evaluate_paper("HXX", candidates, judgments, concepts, call_all_removed)
    assert row["baseline"]["accept"] == 0
    assert row["filtered"]["accept"] == 0
    # This is what the per-paper safety check treats as vacuously fine:
    assert row["baseline"]["accept"] == 0 or row["filtered"]["accept"] > 0


# ---------------------------------------------------------------------------
# Denominator correctness (pure evaluate_paper, synthetic data)
# ---------------------------------------------------------------------------

def test_precision_denominator_excludes_uncertain_candidates():
    candidates = [{"candidate_id": f"K{i}"} for i in range(4)]
    judgments = {
        "K0": {"decision": "accept", "matched_concept_ids": "C1"},
        "K1": {"decision": "reject", "matched_concept_ids": None},
        "K2": {"decision": "uncertain", "matched_concept_ids": None},
        "K3": {"decision": "uncertain", "matched_concept_ids": None},
    }
    concepts = ["c1", None, None, None, None]
    call = {"status": "success", "results": [{"candidate_id": f"K{i}", "decision": "keep"} for i in range(4)]}
    row = metrics.evaluate_paper("H99", candidates, judgments, concepts, call)
    # 1 accept + 1 reject resolved; 2 uncertain excluded from the denominator
    assert row["baseline"]["resolved_precision"] == 0.5
    assert row["baseline"]["candidate_count"] == 4


def test_coverage_denominator_is_populated_concepts_not_all_five():
    candidates = [{"candidate_id": "K0"}]
    judgments = {"K0": {"decision": "accept", "matched_concept_ids": "C1"}}
    concepts = ["c1", "c2", "c3", None, None]  # only 3 populated
    call = {"status": "success", "results": [{"candidate_id": "K0", "decision": "keep"}]}
    row = metrics.evaluate_paper("H98", candidates, judgments, concepts, call)
    assert row["baseline"]["concept_coverage"] == pytest.approx(1 / 3)


def test_coverage_is_none_when_no_concepts_are_populated():
    candidates = [{"candidate_id": "K0"}]
    judgments = {"K0": {"decision": "reject", "matched_concept_ids": None}}
    concepts = [None, None, None, None, None]
    call = {"status": "success", "results": [{"candidate_id": "K0", "decision": "sentence_fragment"}]}
    row = metrics.evaluate_paper("H97", candidates, judgments, concepts, call)
    assert row["baseline"]["concept_coverage"] is None


# ---------------------------------------------------------------------------
# Complete fail-open accounting (pure evaluate_paper, synthetic data)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("call", [
    {"status": "failed", "failure_type": "provider_error", "results": []},
    {"status": "failed", "failure_type": "timeout", "results": []},
    {"status": "failed", "failure_type": "malformed_response", "results": [{"candidate_id": "K0", "decision": "sentence_fragment"}]},
])
def test_fail_open_retains_every_candidate_regardless_of_partial_results(call):
    candidates = [{"candidate_id": "K0"}, {"candidate_id": "K1"}]
    judgments = {
        "K0": {"decision": "accept", "matched_concept_ids": "C1"},
        "K1": {"decision": "reject", "matched_concept_ids": None},
    }
    concepts = ["c1", None, None, None, None]
    row = metrics.evaluate_paper("H96", candidates, judgments, concepts, call)
    assert row["filtered"]["retained_candidate_count"] == 2
    assert row["filtered"]["removed_candidate_count"] == 0
    assert row["filtered"]["accept"] == 1 and row["filtered"]["reject"] == 1


# ---------------------------------------------------------------------------
# Artifact hashes and bindings
# ---------------------------------------------------------------------------

def test_metrics_self_hash_and_bindings_match_current_files():
    data = _real_metrics()
    assert metrics.self_hash_valid(data, "metrics_sha256")
    assert data["bindings"] == metrics.bindings()
    assert data["bindings"]["frozen_heldout_annotation_sha256"] == metrics.file_hash(freeze.FROZEN_PATH)
    assert data["bindings"]["raw_results_sha256"] == metrics.file_hash(prep.RAW_PATH)
    assert data["bindings"]["prompt_contract_sha256"] == metrics.file_hash(prep.PROMPT_PATH)


def test_tampered_metrics_self_hash_is_detected():
    data = _real_metrics()
    tampered = dict(data)
    tampered["frozen_policy_c"] = {**tampered["frozen_policy_c"], "accept": 999}
    assert not metrics.self_hash_valid(tampered, "metrics_sha256")


def test_raw_result_binding_tampering_is_rejected():
    frozen = json.loads(freeze.FROZEN_PATH.read_text())
    mapping = json.loads(k5d1.MAPPING_PATH.read_text())
    raw = json.loads(prep.RAW_PATH.read_text())
    tampered_raw = dict(raw)
    tampered_raw["bindings"] = {**tampered_raw["bindings"], "prompt_contract_sha256": "0" * 64}
    tampered_raw["raw_results_sha256"] = metrics.payload_hash({k: v for k, v in tampered_raw.items() if k != "raw_results_sha256"})
    with pytest.raises(ValueError, match="binding mismatch"):
        metrics.validate_raw_results(tampered_raw, frozen, mapping)


def test_calculate_metrics_refuses_overwrite_without_replace(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    csv_path = tmp_path / "c.csv"
    summary_path = tmp_path / "s.md"
    metrics.calculate_metrics(metrics_path=metrics_path, csv_path=csv_path, summary_path=summary_path)
    before = metrics.file_hash(metrics_path)
    with pytest.raises(FileExistsError):
        metrics.calculate_metrics(metrics_path=metrics_path, csv_path=csv_path, summary_path=summary_path)
    assert metrics.file_hash(metrics_path) == before


# ---------------------------------------------------------------------------
# No provider/network call; no session/checkpoint/telemetry mutation
# ---------------------------------------------------------------------------

def test_recomputation_makes_no_network_call_and_mutates_nothing(tmp_path, monkeypatch):
    checkpoint_before = fingerprint_usage_db(QA_CHECKPOINT_DB_PATH)
    telemetry_before = fingerprint_usage_db(USAGE_DB_PATH)
    real_metrics_before = metrics.file_hash(metrics.METRICS_PATH)
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("network call")))

    metrics_path = tmp_path / "metrics.json"
    csv_path = tmp_path / "c.csv"
    summary_path = tmp_path / "s.md"
    result = metrics.calculate_metrics(metrics_path=metrics_path, csv_path=csv_path, summary_path=summary_path)

    assert result["frozen_yake_v2"]["candidate_count"] == EXPECTED["baseline_candidates"]
    assert metrics_path.exists() and csv_path.exists() and summary_path.exists()
    # the real, canonical artifact this session already produced was not touched by this tmp-path run
    assert metrics.file_hash(metrics.METRICS_PATH) == real_metrics_before
    assert fingerprint_usage_db(QA_CHECKPOINT_DB_PATH) == checkpoint_before
    assert fingerprint_usage_db(USAGE_DB_PATH) == telemetry_before
