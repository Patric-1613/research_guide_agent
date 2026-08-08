"""Deterministic evaluator for the chat_relevance suite -- compares which
web-article URLs a prediction judged relevant against a fixture's
expected_relevant_urls/expected_rejected_urls. No network/API call of
its own; it only compares two sets of URLs. (The mock/live distinction
lives in how the PREDICTION was produced -- see runners/
run_chat_relevance.py -- not in this evaluator, which is identical
either way.)
"""

from __future__ import annotations

from typing import Any


def chat_relevance_correctness(prediction: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Score 1.0 when the prediction's relevant_urls exactly matches
    expected_relevant_urls (and, symmetrically, none of
    expected_rejected_urls leaked through) -- score None (not scored)
    when the fixture gives no expectation at all, matching this
    project's existing "no constraints -> no score, not a failure"
    convention (e.g. report.py's own optional-field handling).
    """
    expected_relevant = set(expected.get("expected_relevant_urls") or [])
    expected_rejected = set(expected.get("expected_rejected_urls") or [])
    if not expected_relevant and not expected_rejected:
        return {"key": "chat_relevance_correctness", "score": None, "comment": "no expectations given"}

    if "error" in prediction:
        return {
            "key": "chat_relevance_correctness", "score": 0.0,
            "comment": f"predict() raised: {prediction['error']}",
        }

    actual_relevant = set(prediction.get("relevant_urls") or [])
    missing = expected_relevant - actual_relevant  # expected relevant, not returned
    leaked = actual_relevant & expected_rejected     # expected rejected, but returned anyway
    unexpected = actual_relevant - expected_relevant - expected_rejected  # not accounted for either way

    ok = not missing and not leaked
    total_checked = len(expected_relevant) + len(expected_rejected)
    errors = len(missing) + len(leaked)
    score = 1.0 if ok else round(max(0.0, 1.0 - errors / max(1, total_checked)), 3)

    return {
        "key": "chat_relevance_correctness",
        "score": score,
        "comment": (
            f"missing={sorted(missing)} leaked={sorted(leaked)} "
            f"unexpected={sorted(unexpected)} actual={sorted(actual_relevant)}"
        ),
    }


ALL_EVALUATORS: dict[str, Any] = {
    "chat_relevance_correctness": chat_relevance_correctness,
}
