#!/usr/bin/env python3
"""K5C.1 zero-cost, post-hoc analysis of fixed targeted filter policies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import k5_keyword_annotation as annotation
from scripts import k5_keyword_metrics as k5b_metrics
from scripts import k5_llm_filter_eval as k5c


BASE = annotation.EVAL_WORKING_DIR
ANALYSIS_PATH = BASE / "llm_filter_reason_analysis.json"
CSV_PATH = BASE / "llm_filter_policy_comparison.csv"
SUMMARY_PATH = BASE / "llm_filter_policy_summary.md"
ANALYSIS_SCHEMA = "k5c1-targeted-policy-analysis-v1"

POLICIES: dict[str, dict[str, Any]] = {
    "A": {"remove_reason_codes": ["malformed_fragment"]},
    "B": {"remove_reason_codes": ["sentence_fragment"]},
    "C": {"remove_reason_codes": ["malformed_fragment", "sentence_fragment"]},
    "D": {"remove_reason_codes": ["malformed_fragment", "sentence_fragment", "redundant_variant"]},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str:
    return annotation._file_sha256(path)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def input_bindings() -> dict[str, str]:
    return {
        "annotation_workbook_sha256": file_hash(annotation.WORKBOOK_PATH),
        "frozen_annotation_sha256": file_hash(k5b_metrics.FROZEN_PATH),
        "candidate_mapping_sha256": file_hash(annotation.MAPPING_PATH),
        "k5b_method_results_sha256": file_hash(k5b_metrics.RESULTS_PATH),
        "k5c_prompt_sha256": file_hash(k5c.PROMPT_PATH),
        "k5c_raw_results_sha256": file_hash(k5c.RAW_PATH),
        "k5c_metrics_sha256": file_hash(k5c.METRICS_PATH),
        "sample_sha256": file_hash(annotation.SAMPLE_PATH),
        "selection_rules_sha256": file_hash(annotation.RULES_PATH),
        "source_snapshot_sha256": file_hash(annotation.SOURCE_SNAPSHOT_PATH),
        "annotation_manifest_sha256": file_hash(annotation.MANIFEST_PATH),
    }


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen, k5b_results, mapping = k5c.validate_k5b_evidence()
    prompt = json.loads(k5c.PROMPT_PATH.read_text(encoding="utf-8"))
    raw = json.loads(k5c.RAW_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(k5c.METRICS_PATH.read_text(encoding="utf-8"))
    sessions = k5c.build_sessions(mapping, frozen["headline_codes"])
    k5c.validate_prompt_contract(prompt, sessions)

    if raw.get("schema_version") != k5c.RAW_SCHEMA or not k5c.self_hash_valid(raw, "raw_results_sha256"):
        raise ValueError("K5C raw result schema or self-hash mismatch")
    if metrics.get("schema_version") != k5c.METRICS_SCHEMA or not k5c.self_hash_valid(metrics, "metrics_sha256"):
        raise ValueError("K5C metrics schema or self-hash mismatch")
    if raw.get("bindings") != {
        "prompt_contract_sha256": file_hash(k5c.PROMPT_PATH),
        "frozen_annotation_sha256": file_hash(k5b_metrics.FROZEN_PATH),
        "candidate_mapping_sha256": file_hash(annotation.MAPPING_PATH),
    }:
        raise ValueError("K5C raw result binding mismatch")
    expected_metric_bindings = {
        "frozen_annotation_sha256": file_hash(k5b_metrics.FROZEN_PATH),
        "k5b_results_sha256": file_hash(k5b_metrics.RESULTS_PATH),
        "candidate_mapping_sha256": file_hash(annotation.MAPPING_PATH),
        "prompt_contract_sha256": file_hash(k5c.PROMPT_PATH),
        "raw_results_sha256": file_hash(k5c.RAW_PATH),
    }
    if metrics.get("bindings") != expected_metric_bindings:
        raise ValueError("K5C metrics input binding mismatch")
    if raw.get("model") != k5c.MODEL or raw.get("prompt_version") != k5c.PROMPT_VERSION:
        raise ValueError("K5C model or prompt changed")
    if raw.get("actual_call_count") != k5c.MAX_CALLS or len(raw.get("calls", [])) != k5c.MAX_CALLS:
        raise ValueError("K5C raw result does not contain exactly eight call attempts")
    calls = {call.get("paper_code"): call for call in raw["calls"]}
    if set(calls) != set(frozen["headline_codes"]):
        raise ValueError("K5C raw paper set is missing, duplicated, or invented")
    session_by_code = {session["paper_code"]: session for session in sessions}
    for code, call in calls.items():
        if call.get("input_sha256") != session_by_code[code]["input_sha256"]:
            raise ValueError(f"{code}: K5C call input hash mismatch")
        if call.get("status") == "success":
            k5c.validate_result_set(call.get("results", []), session_by_code[code]["payload"]["candidates"])
    if metrics.get("production_yake_v2") != {
        "candidate_count": k5b_results["methods"][annotation.PRODUCTION_METHOD_ID]["total_emitted_candidates"],
        "accept": k5b_results["methods"][annotation.PRODUCTION_METHOD_ID]["accept"],
        "reject": k5b_results["methods"][annotation.PRODUCTION_METHOD_ID]["reject"],
        "uncertain": k5b_results["methods"][annotation.PRODUCTION_METHOD_ID]["uncertain"],
        "resolved_precision": k5b_results["methods"][annotation.PRODUCTION_METHOD_ID]["resolved_precision"],
        "macro_concept_coverage": k5b_results["methods"][annotation.PRODUCTION_METHOD_ID]["macro_average_concept_coverage"],
    }:
        raise ValueError("K5C production baseline differs from frozen K5B.3 metrics")
    return frozen, mapping, raw, metrics


def candidate_rows(frozen: dict[str, Any], mapping: dict[str, Any], raw: dict[str, Any]) -> list[dict[str, Any]]:
    sessions = k5c.build_sessions(mapping, frozen["headline_codes"])
    judgments = {row["candidate_id"]: row for row in frozen["candidates"]}
    calls = {call["paper_code"]: call for call in raw["calls"]}
    rows = []
    for session in sessions:
        code = session["paper_code"]
        llm = {row["candidate_id"]: row for row in calls[code].get("results", [])}
        for candidate in session["payload"]["candidates"]:
            candidate_id = candidate["candidate_id"]
            judgment = judgments[candidate_id]
            result = llm.get(candidate_id)
            rows.append({
                "paper_code": code,
                "candidate_id": candidate_id,
                "human_decision": judgment["decision"],
                "matched_concept_ids": judgment["matched_concept_ids"],
                "llm_decision": result["decision"] if result else None,
                "reason_code": result["reason_code"] if result else None,
                "call_status": calls[code]["status"],
            })
    return rows


def analyze_reasons(rows: list[dict[str, Any]], headline_codes: list[str]) -> dict[str, Any]:
    by_reason = {}
    for reason in k5c.REASON_CODES:
        subset = [row for row in rows if row["reason_code"] == reason]
        labels = Counter(row["human_decision"] for row in subset)
        by_reason[reason] = {
            "total": len(subset),
            "human_accept": labels["accept"],
            "human_reject": labels["reject"],
            "human_uncertain": labels["uncertain"],
            "llm_keep": sum(row["llm_decision"] == "keep" for row in subset),
            "llm_remove": sum(row["llm_decision"] == "remove" for row in subset),
            "llm_uncertain": sum(row["llm_decision"] == "uncertain" for row in subset),
            "candidate_ids_by_human_label": {
                label: [row["candidate_id"] for row in subset if row["human_decision"] == label]
                for label in ("accept", "reject", "uncertain")
            },
        }
    false_removals = [row for row in rows if row["human_decision"] == "accept" and row["llm_decision"] == "remove"]
    return {
        "all_candidates_by_reason_code": by_reason,
        "full_filter_false_removals": {
            "count": len(false_removals),
            "reason_code_distribution": dict(sorted(Counter(row["reason_code"] for row in false_removals).items())),
            "by_paper": {
                code: {
                    "count": sum(row["paper_code"] == code for row in false_removals),
                    "reason_code_distribution": dict(sorted(Counter(
                        row["reason_code"] for row in false_removals if row["paper_code"] == code
                    ).items())),
                    "candidate_ids": [row["candidate_id"] for row in false_removals if row["paper_code"] == code],
                }
                for code in headline_codes
            },
        },
    }


def rate(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def evaluate_policy(
    policy_id: str, remove_reasons: set[str], frozen: dict[str, Any], mapping: dict[str, Any], raw: dict[str, Any],
) -> dict[str, Any]:
    sessions = k5c.build_sessions(mapping, frozen["headline_codes"])
    judgments = {row["candidate_id"]: row for row in frozen["candidates"]}
    concepts = {row["paper_code"]: row["concepts"] for row in frozen["papers"]}
    calls = {call["paper_code"]: call for call in raw["calls"]}
    per_paper = []
    malformed = sum(call["status"] != "success" for call in raw["calls"])
    for session in sessions:
        code = session["paper_code"]
        source_call = calls[code]
        transformed = []
        for result in source_call.get("results", []):
            if result["decision"] == "uncertain":
                decision = "uncertain"
            elif result["decision"] == "remove" and result["reason_code"] in remove_reasons:
                decision = "remove"
            else:
                decision = "keep"
            transformed.append({"candidate_id": result["candidate_id"], "decision": decision})
        policy_call = {
            "status": source_call["status"], "failure_type": source_call.get("failure_type"), "results": transformed,
        }
        row = k5c.evaluate_paper(
            code, session["payload"]["candidates"], judgments, concepts[code], policy_call,
        )
        row["loses_all_accepted_keywords"] = row["baseline"]["accept"] > 0 and row["filtered"]["accept"] == 0
        per_paper.append(row)

    baseline = Counter()
    retained = Counter()
    for row in per_paper:
        baseline.update({key: row["baseline"][key] for key in ("accept", "reject", "uncertain")})
        retained.update({key: row["filtered"][key] for key in ("accept", "reject", "uncertain")})
    input_count = sum(row["baseline"]["candidate_count"] for row in per_paper)
    retained_count = sum(row["filtered"]["retained_candidate_count"] for row in per_paper)
    base_precision = rate(baseline["accept"], baseline["accept"] + baseline["reject"])
    precision = rate(retained["accept"], retained["accept"] + retained["reject"])
    base_coverage = round(sum(row["baseline"]["concept_coverage"] for row in per_paper) / len(per_paper), 6)
    coverage = round(sum(row["filtered"]["concept_coverage"] for row in per_paper) / len(per_paper), 6)
    accepted_retention = rate(retained["accept"], baseline["accept"])
    coverage_retention = rate(coverage, base_coverage)
    precision_delta = round((precision or 0) - (base_precision or 0), 6)
    no_paper_loses_all = not any(row["loses_all_accepted_keywords"] for row in per_paper)
    checks = {
        "resolved_precision_improvement": precision_delta >= k5c.PROVISIONAL_GATE["resolved_precision_improvement_minimum"],
        "accepted_candidate_retention": (accepted_retention or 0) >= k5c.PROVISIONAL_GATE["accepted_candidate_retention_minimum"],
        "macro_concept_coverage_retention": (coverage_retention or 0) >= k5c.PROVISIONAL_GATE["macro_concept_coverage_retention_minimum"],
        "every_paper_retains_an_accepted_keyword": no_paper_loses_all,
        "zero_malformed_or_failed_calls": malformed == 0,
    }
    return {
        "policy_id": policy_id,
        "remove_rule": {"llm_decision": "remove", "reason_codes": sorted(remove_reasons)},
        "retained_candidate_count": retained_count,
        "removed_candidate_count": input_count - retained_count,
        "human_labels_retained": {key: retained[key] for key in ("accept", "reject", "uncertain")},
        "resolved_precision": precision,
        "resolved_precision_delta": precision_delta,
        "accepted_keyword_retention": accepted_retention,
        "rejected_keyword_removal_rate": rate(baseline["reject"] - retained["reject"], baseline["reject"]),
        "false_removal_rate": rate(baseline["accept"] - retained["accept"], baseline["accept"]),
        "macro_concept_coverage": coverage,
        "macro_concept_coverage_retention": coverage_retention,
        "percentage_candidates_removed": rate(input_count - retained_count, input_count),
        "llm_uncertain_decisions_retained": sum(row["filtered"]["llm_uncertain_decisions"] for row in per_paper),
        "any_paper_loses_all_accepted_keywords": not no_paper_loses_all,
        "malformed_or_failed_calls": malformed,
        "per_paper": per_paper,
        "provisional_gate": {"definition": k5c.PROVISIONAL_GATE, "checks": checks, "passed": all(checks.values())},
    }


def validate_analysis(data: dict[str, Any]) -> list[str]:
    errors = []
    if data.get("schema_version") != ANALYSIS_SCHEMA:
        errors.append("analysis schema mismatch")
    if not k5c.self_hash_valid(data, "analysis_sha256"):
        errors.append("analysis self-hash mismatch")
    if data.get("bindings") != input_bindings():
        errors.append("analysis input binding mismatch")
    if set(data.get("policies", {})) != set(POLICIES):
        errors.append("analysis does not contain exactly policies A-D")
    if data.get("analysis_type") != "post_hoc_exploratory":
        errors.append("analysis is not labelled post-hoc exploratory")
    return errors


def generate(
    replace: bool = False, analysis_path: Path = ANALYSIS_PATH, csv_path: Path = CSV_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> dict[str, Any]:
    for path in (analysis_path, csv_path, summary_path):
        if path.exists() and not replace:
            raise FileExistsError("K5C.1 output exists; pass --replace")
    frozen, mapping, raw, k5c_metrics = validate_inputs()
    rows = candidate_rows(frozen, mapping, raw)
    reason_analysis = analyze_reasons(rows, frozen["headline_codes"])
    if reason_analysis["full_filter_false_removals"]["count"] != 5:
        raise ValueError("full K5C filter no longer has exactly five human-accepted removals")
    policies = {
        policy_id: evaluate_policy(policy_id, set(spec["remove_reason_codes"]), frozen, mapping, raw)
        for policy_id, spec in POLICIES.items()
    }
    passed = [policy_id for policy_id, policy in policies.items() if policy["provisional_gate"]["passed"]]
    malformed = k5c_metrics["llm_filtered_yake_v2"]["malformed_or_failed_calls"]
    if malformed:
        choice = "evidence is inconclusive"
        basis = "The source K5C run contained malformed or failed LLM calls."
    elif passed:
        choice = "a targeted policy is promising but requires guarded product piloting"
        basis = f"Post-hoc policies {', '.join(passed)} pass the original frozen provisional gate; this exploratory reuse of one small sample requires prospective validation."
    else:
        choice = "no targeted policy is promising — retain plain YAKE-v2"
        basis = "None of the four fixed post-hoc policies passes the original frozen provisional gate."
    data = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "complete_post_hoc_exploratory",
        "created_at": now(),
        "analysis_type": "post_hoc_exploratory",
        "policy_timing": "policies were defined after observing K5C results",
        "provider_calls_made": 0,
        "headline_codes": frozen["headline_codes"],
        "excluded_pilot_codes": frozen["pilot_codes"],
        "bindings": input_bindings(),
        "baseline": k5c_metrics["production_yake_v2"],
        "reason_analysis": reason_analysis,
        "policies": policies,
        "passing_policy_ids": passed,
        "recommendation": {
            "choice": choice,
            "basis": basis,
            "limitation": "Post-hoc exploratory analysis of eight product-local papers using the same frozen AI-assisted, human-approved annotations; not independent confirmation, an external benchmark, or a statistical claim.",
        },
        "provisional_gate": k5c.PROVISIONAL_GATE,
    }
    data["analysis_sha256"] = k5c.payload_hash(data)
    errors = validate_analysis(data)
    if errors:
        raise ValueError("; ".join(errors))
    atomic_write(analysis_path, json_bytes(data))

    fields = [
        "policy_id", "paper_code", "retained_candidates", "accept", "reject", "human_uncertain",
        "resolved_precision", "precision_delta", "accepted_keyword_retention", "rejected_keyword_removal_rate",
        "false_removal_rate", "concept_coverage", "coverage_retention", "loses_all_accepted_keywords", "gate_passed",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for policy_id, policy in policies.items():
        for row in policy["per_paper"]:
            writer.writerow({
                "policy_id": policy_id, "paper_code": row["paper_code"],
                "retained_candidates": row["filtered"]["retained_candidate_count"],
                "accept": row["filtered"]["accept"], "reject": row["filtered"]["reject"],
                "human_uncertain": row["filtered"]["uncertain"],
                "resolved_precision": row["filtered"]["resolved_precision"],
                "precision_delta": row["delta_filtered_minus_production"]["resolved_precision"],
                "accepted_keyword_retention": row["filtered"]["accepted_keyword_retention"],
                "rejected_keyword_removal_rate": row["filtered"]["rejected_keyword_removal_rate"],
                "false_removal_rate": row["filtered"]["false_removal_rate"],
                "concept_coverage": row["filtered"]["concept_coverage"],
                "coverage_retention": k5c.rate(row["filtered"]["concept_coverage"], row["baseline"]["concept_coverage"]),
                "loses_all_accepted_keywords": row["loses_all_accepted_keywords"],
                "gate_passed": policy["provisional_gate"]["passed"],
            })
    atomic_write(csv_path, output.getvalue().encode("utf-8"))

    lines = [
        "# K5C.1 targeted LLM-filter policy analysis", "",
        "**Post-hoc exploratory analysis:** policies A–D were defined after observing K5C. No new provider call was made.", "",
        "| Policy | Retained | Precision | Δ precision | Accepted retention | Rejected removal | False removal | Macro coverage | Coverage retention | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for policy_id, policy in policies.items():
        lines.append(
            f"| {policy_id} | {policy['retained_candidate_count']} | {policy['resolved_precision']:.1%} | "
            f"{policy['resolved_precision_delta']:+.1%} | {policy['accepted_keyword_retention']:.1%} | "
            f"{policy['rejected_keyword_removal_rate']:.1%} | {policy['false_removal_rate']:.1%} | "
            f"{policy['macro_concept_coverage']:.1%} | {policy['macro_concept_coverage_retention']:.1%} | "
            f"{'PASS' if policy['provisional_gate']['passed'] else 'FAIL'} |"
        )
    lines += ["", "## Recommendation", "", f"**{choice}.** {basis}", "", data["recommendation"]["limitation"]]
    atomic_write(summary_path, ("\n".join(lines) + "\n").encode("utf-8"))
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "validate"))
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            data = generate(replace=args.replace)
            print(f"complete: policies=4 passing={','.join(data['passing_policy_ids']) or 'none'} provider_calls=0")
        else:
            validate_inputs()
            data = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
            errors = validate_analysis(data)
            if errors:
                raise ValueError("; ".join(errors))
            print("valid: K5C.1 hashes, bindings, and post-hoc policy artifact")
        return 0
    except Exception as exc:  # noqa: BLE001 -- concise offline CLI failure.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
