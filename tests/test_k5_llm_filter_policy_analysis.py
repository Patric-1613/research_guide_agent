"""Focused zero-provider tests for K5C.1 targeted policy analysis."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import k5_llm_filter_eval as k5c
from scripts import k5_llm_filter_policy_analysis as policy


def _row(candidate_id: str, human: str, llm: str, reason: str, paper: str = "P03", concept: str | None = None):
    return {
        "paper_code": paper, "candidate_id": candidate_id, "human_decision": human,
        "matched_concept_ids": concept, "llm_decision": llm, "reason_code": reason, "call_status": "success",
    }


def _db_hashes() -> dict[Path, str]:
    result = {}
    for pattern in ("*.db", "*.sqlite", "*.sqlite3"):
        for path in Path("data").rglob(pattern):
            if path.is_file():
                result[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def test_reason_analysis_distinguishes_human_labels_for_every_reason():
    rows = [
        _row("K1", "accept", "remove", "malformed_fragment"),
        _row("K2", "reject", "remove", "malformed_fragment"),
        _row("K3", "accept", "remove", "sentence_fragment", paper="P04"),
        _row("K4", "uncertain", "uncertain", "uncertain"),
    ]
    result = policy.analyze_reasons(rows, ["P03", "P04"])
    malformed = result["all_candidates_by_reason_code"]["malformed_fragment"]
    assert malformed["human_accept"] == 1
    assert malformed["human_reject"] == 1
    assert malformed["human_uncertain"] == 0
    assert result["all_candidates_by_reason_code"]["too_broad"]["total"] == 0
    false = result["full_filter_false_removals"]
    assert false["count"] == 2
    assert false["reason_code_distribution"] == {"malformed_fragment": 1, "sentence_fragment": 1}
    assert false["by_paper"]["P03"]["count"] == 1
    assert false["by_paper"]["P04"]["count"] == 1


def test_policy_definitions_are_exact_and_fixed():
    assert policy.POLICIES == {
        "A": {"remove_reason_codes": ["malformed_fragment"]},
        "B": {"remove_reason_codes": ["sentence_fragment"]},
        "C": {"remove_reason_codes": ["malformed_fragment", "sentence_fragment"]},
        "D": {"remove_reason_codes": ["malformed_fragment", "sentence_fragment", "redundant_variant"]},
    }


def test_policy_transform_retains_uncertain_and_non_targeted_remove(monkeypatch):
    captured = {}
    original = k5c.evaluate_paper

    def spy(code, candidates, judgments, concepts, call):
        captured[code] = {row["candidate_id"]: row["decision"] for row in call["results"]}
        return original(code, candidates, judgments, concepts, call)

    monkeypatch.setattr(policy.k5c, "evaluate_paper", spy)
    frozen, mapping, raw, _metrics = policy.validate_inputs()
    policy.evaluate_policy("A", {"malformed_fragment"}, frozen, mapping, raw)
    raw_by_id = {
        row["candidate_id"]: row
        for call in raw["calls"] for row in call["results"]
    }
    observed = {candidate_id: decision for paper in captured.values() for candidate_id, decision in paper.items()}
    for candidate_id, result in raw_by_id.items():
        expected = "remove" if result["decision"] == "remove" and result["reason_code"] == "malformed_fragment" else result["decision"] if result["decision"] == "uncertain" else "keep"
        assert observed[candidate_id] == expected


def test_each_policy_reports_all_required_metrics_and_unchanged_gate(tmp_path):
    data = policy.generate(
        analysis_path=tmp_path / "analysis.json", csv_path=tmp_path / "comparison.csv",
        summary_path=tmp_path / "summary.md",
    )
    assert set(data["policies"]) == {"A", "B", "C", "D"}
    for result in data["policies"].values():
        assert len(result["per_paper"]) == 8
        assert result["provisional_gate"]["definition"] == k5c.PROVISIONAL_GATE
        for field in (
            "retained_candidate_count", "resolved_precision", "resolved_precision_delta",
            "accepted_keyword_retention", "rejected_keyword_removal_rate", "false_removal_rate",
            "macro_concept_coverage", "macro_concept_coverage_retention",
            "any_paper_loses_all_accepted_keywords",
        ):
            assert field in result
    assert data["analysis_type"] == "post_hoc_exploratory"
    assert data["provider_calls_made"] == 0
    assert len(list(__import__("csv").DictReader((tmp_path / "comparison.csv").open()))) == 32


def test_analysis_self_hash_and_input_tampering_fail(tmp_path):
    path = tmp_path / "analysis.json"
    data = policy.generate(analysis_path=path, csv_path=tmp_path / "c.csv", summary_path=tmp_path / "s.md")
    assert policy.validate_analysis(data) == []
    tampered = copy.deepcopy(data)
    tampered["policies"]["A"]["retained_candidate_count"] += 1
    assert "analysis self-hash mismatch" in policy.validate_analysis(tampered)
    rebound = copy.deepcopy(data)
    rebound["bindings"]["k5c_raw_results_sha256"] = "0" * 64
    rebound["analysis_sha256"] = k5c.payload_hash({k: v for k, v in rebound.items() if k != "analysis_sha256"})
    assert "analysis input binding mismatch" in policy.validate_analysis(rebound)


def test_actual_offline_analysis_does_not_mutate_databases_or_call_provider(tmp_path, monkeypatch):
    before = _db_hashes()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("provider path must not be reached")

    monkeypatch.setattr(k5c, "run_live", forbidden)
    result = policy.generate(
        analysis_path=tmp_path / "analysis.json", csv_path=tmp_path / "comparison.csv",
        summary_path=tmp_path / "summary.md",
    )
    assert result["provider_calls_made"] == 0
    assert _db_hashes() == before


def test_overwrite_protection(tmp_path):
    analysis = tmp_path / "analysis.json"
    csv_path = tmp_path / "comparison.csv"
    summary = tmp_path / "summary.md"
    policy.generate(analysis_path=analysis, csv_path=csv_path, summary_path=summary)
    with pytest.raises(FileExistsError, match="pass --replace"):
        policy.generate(analysis_path=analysis, csv_path=csv_path, summary_path=summary)
