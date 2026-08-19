#!/usr/bin/env python3
"""K5D.1b: bounded, hash-bound metrics comparing frozen YAKE-v2 against
frozen Policy C on the independent six-paper held-out sample.

Offline only -- consumes the already-executed, human-approved live run
(scripts/k5_heldout_llm_prep.py's ``live`` command) and the frozen
human annotation. Never calls a provider itself, and reuses the
already-frozen provisional gate (scripts/k5_llm_filter_eval.PROVISIONAL_GATE)
unchanged -- this checkpoint does not redefine or tune thresholds.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import k5_heldout_annotation_freeze as freeze
from scripts import k5_heldout_llm_prep as prep
from scripts import k5_heldout_selection as k5d1
from scripts import k5_llm_filter_eval as k5c  # frozen PROVISIONAL_GATE, reused unchanged

BASE = k5d1.K5D1_DIR
METRICS_PATH = BASE / "heldout_llm_metrics.json"
CSV_PATH = BASE / "heldout_llm_comparison.csv"
SUMMARY_PATH = BASE / "heldout_llm_summary.md"
METRICS_SCHEMA = "k5d1b-heldout-llm-metrics-v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def payload_hash(value: Any) -> str:
    return freeze.payload_hash(value)


def file_hash(path: Path) -> str:
    return freeze.file_hash(path)


def json_bytes(value: Any) -> bytes:
    return freeze.json_bytes(value)


def atomic_write(path: Path, data: bytes) -> None:
    freeze.atomic_write(path, data)


def self_hash_valid(value: dict[str, Any], field: str) -> bool:
    return freeze.self_hash_valid(value, field)


def rate(numerator: int | float, denominator: int | float) -> float | None:
    return None if not denominator else round(numerator / denominator, 6)


def concept_ids(value: Any) -> set[str]:
    return set(k5d1._parse_concept_ids(value)[0])


def bindings() -> dict[str, str]:
    return {
        "frozen_heldout_annotation_sha256": file_hash(freeze.FROZEN_PATH),
        "candidate_mapping_sha256": file_hash(k5d1.MAPPING_PATH),
        "annotation_manifest_sha256": file_hash(k5d1.MANIFEST_PATH),
        "selection_sha256": file_hash(k5d1.SELECTION_PATH),
        "source_snapshot_sha256": file_hash(k5d1.SOURCE_SNAPSHOT_PATH),
        "workbook_sha256": file_hash(k5d1.WORKBOOK_PATH),
        "prompt_contract_sha256": file_hash(prep.PROMPT_PATH),
        "raw_results_sha256": file_hash(prep.RAW_PATH),
    }


def validate_raw_results(raw: dict[str, Any], frozen: dict[str, Any], mapping: dict[str, Any]) -> list[dict[str, Any]]:
    if raw.get("schema_version") != prep.RAW_SCHEMA or not self_hash_valid(raw, "raw_results_sha256"):
        raise ValueError("raw result schema or self-hash mismatch")
    fixed = (raw.get("method_id"), raw.get("model"), raw.get("prompt_version"), raw.get("approved_maximum_call_count"), raw.get("no_retries"))
    if fixed != (prep.METHOD_ID, prep.MODEL, prep.PROMPT_VERSION, prep.MAX_CALLS, True):
        raise ValueError("raw result method/model/prompt/call bound changed")
    if raw.get("status") not in ("complete", "complete_with_failures"):
        raise ValueError("raw result run is incomplete")
    if raw.get("actual_call_count") != prep.MAX_CALLS or len(raw.get("calls", [])) != prep.MAX_CALLS:
        raise ValueError(f"raw result does not contain exactly {prep.MAX_CALLS} call attempts")

    contract = json.loads(prep.PROMPT_PATH.read_text(encoding="utf-8"))
    sessions = prep.build_sessions(mapping, frozen["paper_codes"])
    prep.validate_prompt_contract(contract, sessions)
    if raw.get("bindings") != {
        "prompt_contract_sha256": file_hash(prep.PROMPT_PATH),
        "frozen_heldout_annotation_sha256": file_hash(freeze.FROZEN_PATH),
        "candidate_mapping_sha256": file_hash(k5d1.MAPPING_PATH),
    }:
        raise ValueError("raw result binding mismatch")

    calls = {row.get("paper_code"): row for row in raw["calls"]}
    if set(calls) != set(frozen["paper_codes"]):
        raise ValueError("raw result papers are missing, duplicated, or invented")
    session_by_code = {row["paper_code"]: row for row in sessions}
    for code, call in calls.items():
        if call.get("input_sha256") != session_by_code[code]["input_sha256"]:
            raise ValueError(f"{code}: raw call input hash mismatch")
        if call.get("status") == "success":
            prep.validate_result_set(call.get("results", []), session_by_code[code]["payload"]["candidates"])
    return sessions


def evaluate_paper(
    code: str, candidates: list[dict[str, str]], judgments: dict[str, dict[str, Any]],
    concepts: list[Any], call: dict[str, Any],
) -> dict[str, Any]:
    baseline = [judgments[row["candidate_id"]] for row in candidates]
    candidate_ids = [row["candidate_id"] for row in candidates]
    removed_ids = prep.apply_policy_c(call, candidate_ids)  # fails open: empty on any non-success call
    retained_candidates = [row for row in candidates if row["candidate_id"] not in removed_ids]
    retained = [judgments[row["candidate_id"]] for row in retained_candidates]

    base_counts = Counter(row["decision"] for row in baseline)
    kept_counts = Counter(row["decision"] for row in retained)
    base_matched = set().union(*(concept_ids(row["matched_concept_ids"]) for row in baseline if row["decision"] == "accept")) if base_counts["accept"] else set()
    kept_matched = set().union(*(concept_ids(row["matched_concept_ids"]) for row in retained if row["decision"] == "accept")) if kept_counts["accept"] else set()
    populated = sum(value not in (None, "") for value in concepts)
    base_precision = rate(base_counts["accept"], base_counts["accept"] + base_counts["reject"])
    kept_precision = rate(kept_counts["accept"], kept_counts["accept"] + kept_counts["reject"])
    base_coverage = rate(len(base_matched), populated)
    kept_coverage = rate(len(kept_matched), populated)
    removed_actual = {row["candidate_id"] for row in candidates} - {row["candidate_id"] for row in retained_candidates}
    accepted_removed = sum(judgments[cid]["decision"] == "accept" for cid in removed_actual)
    rejected_removed = sum(judgments[cid]["decision"] == "reject" for cid in removed_actual)
    llm_uncertain = sum(row.get("decision") == "uncertain" for row in call.get("results", []))

    return {
        "paper_code": code,
        "call_status": call["status"],
        "failure_type": call.get("failure_type"),
        "baseline": {
            "candidate_count": len(baseline), "accept": base_counts["accept"], "reject": base_counts["reject"],
            "uncertain": base_counts["uncertain"], "resolved_precision": base_precision, "concept_coverage": base_coverage,
        },
        "filtered": {
            "retained_candidate_count": len(retained), "removed_candidate_count": len(baseline) - len(retained),
            "accept": kept_counts["accept"], "reject": kept_counts["reject"], "uncertain": kept_counts["uncertain"],
            "resolved_precision": kept_precision,
            "accepted_keyword_retention": rate(kept_counts["accept"], base_counts["accept"]),
            "rejected_keyword_removal_rate": rate(rejected_removed, base_counts["reject"]),
            "false_removal_rate": rate(accepted_removed, base_counts["accept"]),
            "concept_coverage": kept_coverage,
            "percentage_candidates_removed": rate(len(baseline) - len(retained), len(baseline)),
            "llm_uncertain_decisions": llm_uncertain,
        },
        "delta_filtered_minus_production": {
            "candidate_count": len(retained) - len(baseline),
            "accept": kept_counts["accept"] - base_counts["accept"],
            "reject": kept_counts["reject"] - base_counts["reject"],
            "uncertain": kept_counts["uncertain"] - base_counts["uncertain"],
            "resolved_precision": round((kept_precision or 0) - (base_precision or 0), 6),
            "concept_coverage": round((kept_coverage or 0) - (base_coverage or 0), 6),
        },
    }


def calculate_metrics(
    replace: bool = False, metrics_path: Path = METRICS_PATH, csv_path: Path = CSV_PATH, summary_path: Path = SUMMARY_PATH,
) -> dict[str, Any]:
    for path in (metrics_path, csv_path, summary_path):
        if path.exists() and not replace:
            raise FileExistsError("K5D.1b held-out metrics output exists; pass --replace")

    frozen = json.loads(freeze.FROZEN_PATH.read_text(encoding="utf-8"))
    errors = freeze.validate_frozen_payload(frozen) + freeze.validate_reproducibility(frozen)
    if errors:
        raise ValueError("frozen held-out annotation invalid: " + "; ".join(errors))
    mapping = json.loads(k5d1.MAPPING_PATH.read_text(encoding="utf-8"))
    if not self_hash_valid(mapping, "mapping_sha256"):
        raise ValueError("candidate mapping self-hash mismatch")
    raw = json.loads(prep.RAW_PATH.read_text(encoding="utf-8"))
    sessions = validate_raw_results(raw, frozen, mapping)

    judgments = {row["candidate_id"]: row for row in frozen["candidates"]}
    concepts_by_code = {row["paper_code"]: row["concepts"] for row in frozen["papers"]}
    session_by_code = {row["paper_code"]: row for row in sessions}
    calls = {row["paper_code"]: row for row in raw["calls"]}
    per_paper = [
        evaluate_paper(code, session_by_code[code]["payload"]["candidates"], judgments, concepts_by_code[code], calls[code])
        for code in frozen["paper_codes"]
    ]

    baseline_counts: Counter = Counter()
    filtered_counts: Counter = Counter()
    for row in per_paper:
        baseline_counts.update({key: row["baseline"][key] for key in ("accept", "reject", "uncertain")})
        filtered_counts.update({key: row["filtered"][key] for key in ("accept", "reject", "uncertain")})
    total_input = sum(row["baseline"]["candidate_count"] for row in per_paper)
    total_retained = sum(row["filtered"]["retained_candidate_count"] for row in per_paper)
    total_removed = total_input - total_retained
    rejected_removed_total = baseline_counts["reject"] - filtered_counts["reject"]
    accepted_removed_total = baseline_counts["accept"] - filtered_counts["accept"]
    base_precision = rate(baseline_counts["accept"], baseline_counts["accept"] + baseline_counts["reject"])
    filtered_precision = rate(filtered_counts["accept"], filtered_counts["accept"] + filtered_counts["reject"])
    base_macro_coverage = round(sum(row["baseline"]["concept_coverage"] for row in per_paper) / len(per_paper), 6)
    filtered_macro_coverage = round(sum(row["filtered"]["concept_coverage"] for row in per_paper) / len(per_paper), 6)
    malformed = sum(row["call_status"] != "success" for row in per_paper)
    coverage_retention = rate(filtered_macro_coverage, base_macro_coverage)
    accepted_retention = rate(filtered_counts["accept"], baseline_counts["accept"])
    precision_delta = round((filtered_precision or 0) - (base_precision or 0), 6)
    # Vacuously true for a paper with zero accepted YAKE-v2 candidates (e.g. H04):
    # such a paper has nothing to lose, so it must never be read as evidence that
    # Policy C "preserved an accepted keyword" for it. See
    # papers_with_zero_accepted_baseline_keywords below for exactly which papers
    # that vacuous case applies to.
    papers_with_zero_accepted_baseline = sorted(
        row["paper_code"] for row in per_paper if row["baseline"]["accept"] == 0
    )
    every_paper_with_accepted_baseline_retained_one = all(
        row["baseline"]["accept"] == 0 or row["filtered"]["accept"] > 0 for row in per_paper
    )

    gate = k5c.PROVISIONAL_GATE
    checks = {
        "resolved_precision_improvement": precision_delta >= gate["resolved_precision_improvement_minimum"],
        "accepted_candidate_retention": (accepted_retention or 0) >= gate["accepted_candidate_retention_minimum"],
        "macro_concept_coverage_retention": (coverage_retention or 0) >= gate["macro_concept_coverage_retention_minimum"],
        "every_paper_with_an_accepted_baseline_keyword_retained_one": every_paper_with_accepted_baseline_retained_one,
        "zero_malformed_or_failed_calls": malformed == 0,
    }
    gate_passed = all(checks.values())
    conclusion = (
        "held-out validation passed: Policy C may proceed to a guarded, off-by-default production pilot"
        if gate_passed else
        "held-out validation failed: retain YAKE-v2 alone"
    )

    result = {
        "schema_version": METRICS_SCHEMA,
        "status": "held_out_bounded_descriptive_results",
        "created_at": now(),
        "method_id": prep.METHOD_ID,
        "model": prep.MODEL,
        "prompt_version": prep.PROMPT_VERSION,
        "reviewer_type": freeze.REVIEWER_TYPE,
        "scope": {
            "paper_codes": frozen["paper_codes"], "paper_count": len(frozen["paper_codes"]),
            "held_out_independent_sample": True,
            "human_uncertain_treatment": "neither accepted nor rejected",
            "llm_uncertain_treatment": "retained",
        },
        "bindings": bindings(),
        "frozen_yake_v2": {
            "candidate_count": total_input, "accept": baseline_counts["accept"], "reject": baseline_counts["reject"],
            "uncertain": baseline_counts["uncertain"], "resolved_precision": base_precision,
            "macro_concept_coverage": base_macro_coverage,
        },
        "frozen_policy_c": {
            "retained_candidate_count": total_retained, "removed_candidate_count": total_removed,
            "accept": filtered_counts["accept"], "reject": filtered_counts["reject"], "uncertain": filtered_counts["uncertain"],
            "resolved_precision": filtered_precision,
            "resolved_precision_delta": precision_delta,
            "accepted_keyword_retention": accepted_retention,
            "rejected_keyword_removal_rate": rate(rejected_removed_total, baseline_counts["reject"]),
            "false_removal_rate": rate(accepted_removed_total, baseline_counts["accept"]),
            "macro_concept_coverage": filtered_macro_coverage,
            "macro_concept_coverage_retention": coverage_retention,
            "percentage_candidates_removed": rate(total_removed, total_input),
            "llm_uncertain_decisions": sum(row["filtered"]["llm_uncertain_decisions"] for row in per_paper),
            "malformed_or_failed_calls": malformed,
        },
        "per_paper": per_paper,
        "per_paper_failures": [row["paper_code"] for row in per_paper if row["call_status"] != "success"],
        "provisional_gate": {
            "definition": gate,
            "checks": checks,
            "passed": gate_passed,
            "safety_condition_wording": (
                "Every paper that had at least one accepted YAKE-v2 keyword retained at least one."
            ),
            "papers_with_zero_accepted_baseline_keywords": papers_with_zero_accepted_baseline,
        },
        "conclusion": conclusion,
        "limitation": (
            "Six product-local, independent held-out papers with a single bounded live run; "
            "descriptive only, not an external benchmark or statistical significance claim."
        ),
        "api_observability": {
            "approved_call_count": raw["approved_maximum_call_count"], "actual_call_count": raw["actual_call_count"],
            "total_latency_ms": round(sum(row.get("latency_ms", 0) for row in raw["calls"]), 2),
            "token_usage_by_call": {row["paper_code"]: row.get("usage") for row in raw["calls"]},
            "cost_usd": raw.get("cost_usd"), "cost_note": raw.get("cost_note"),
        },
    }
    result["metrics_sha256"] = payload_hash(result)
    atomic_write(metrics_path, json_bytes(result))

    fields = [
        "paper_code", "call_status", "yake_count", "retained_count", "removed_count",
        "yake_accept", "retained_accept", "yake_reject", "retained_reject", "yake_uncertain", "retained_uncertain",
        "yake_resolved_precision", "policy_c_resolved_precision", "resolved_precision_delta",
        "accepted_keyword_retention", "rejected_keyword_removal_rate", "false_removal_rate",
        "yake_concept_coverage", "policy_c_concept_coverage", "concept_coverage_delta",
        "percentage_candidates_removed", "llm_uncertain_decisions",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in per_paper:
        writer.writerow({
            "paper_code": row["paper_code"], "call_status": row["call_status"],
            "yake_count": row["baseline"]["candidate_count"], "retained_count": row["filtered"]["retained_candidate_count"],
            "removed_count": row["filtered"]["removed_candidate_count"], "yake_accept": row["baseline"]["accept"],
            "retained_accept": row["filtered"]["accept"], "yake_reject": row["baseline"]["reject"],
            "retained_reject": row["filtered"]["reject"], "yake_uncertain": row["baseline"]["uncertain"],
            "retained_uncertain": row["filtered"]["uncertain"], "yake_resolved_precision": row["baseline"]["resolved_precision"],
            "policy_c_resolved_precision": row["filtered"]["resolved_precision"],
            "resolved_precision_delta": row["delta_filtered_minus_production"]["resolved_precision"],
            "accepted_keyword_retention": row["filtered"]["accepted_keyword_retention"],
            "rejected_keyword_removal_rate": row["filtered"]["rejected_keyword_removal_rate"],
            "false_removal_rate": row["filtered"]["false_removal_rate"],
            "yake_concept_coverage": row["baseline"]["concept_coverage"],
            "policy_c_concept_coverage": row["filtered"]["concept_coverage"],
            "concept_coverage_delta": row["delta_filtered_minus_production"]["concept_coverage"],
            "percentage_candidates_removed": row["filtered"]["percentage_candidates_removed"],
            "llm_uncertain_decisions": row["filtered"]["llm_uncertain_decisions"],
        })
    atomic_write(csv_path, output.getvalue().encode("utf-8"))

    filtered = result["frozen_policy_c"]
    lines = [
        "# K5D.1b held-out validation: frozen YAKE-v2 vs. frozen Policy C", "",
        f"Model: `{prep.MODEL}`; prompt: `{prep.PROMPT_VERSION}`; calls: {raw['actual_call_count']}/{prep.MAX_CALLS} approved.", "",
        "| Method | Candidates retained | Accept | Reject | Uncertain | Resolved precision | Macro concept coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| frozen_yake_v2 | {total_input} | {baseline_counts['accept']} | {baseline_counts['reject']} | {baseline_counts['uncertain']} | {base_precision:.1%} | {base_macro_coverage:.1%} |",
        f"| frozen_policy_c | {total_retained} | {filtered_counts['accept']} | {filtered_counts['reject']} | {filtered_counts['uncertain']} | {filtered_precision:.1%} | {filtered_macro_coverage:.1%} |",
        "", f"Provisional gate: **{'PASS' if gate_passed else 'FAIL'}**.", "",
        f"Conclusion: **{conclusion}**.", "", result["limitation"],
    ]
    atomic_write(summary_path, ("\n".join(lines) + "\n").encode("utf-8"))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("metrics",))
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = calculate_metrics(replace=args.replace)
        print(f"metrics complete: gate_passed={result['provisional_gate']['passed']} conclusion={result['conclusion']!r}")
        return 0
    except Exception as exc:  # noqa: BLE001 -- concise offline CLI failure.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
