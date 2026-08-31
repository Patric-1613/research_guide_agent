"""Deterministic tests for ingestion helpers that don't need a live API call."""

from __future__ import annotations

import math
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.ingestion import (
    _S2_MAX_RETRY_DELAY_SECONDS,
    _clean_abstract,
    _clean_venue,
    _parse_retry_after,
    search_semantic_scholar,
)


def _fake_response(status_code: int, json_data=None, headers=None, raise_on_json: bool = False) -> MagicMock:
    resp = MagicMock(status_code=status_code, headers=headers or {})
    if raise_on_json:
        resp.json.side_effect = ValueError("malformed")
    else:
        resp.json.return_value = json_data
    return resp


def test_clean_venue_accepts_plausible_venue_names():
    assert _clean_venue("Neural Information Processing Systems") == "Neural Information Processing Systems"
    assert _clean_venue(None) is None


def test_clean_venue_rejects_citation_dumps():
    dump = (
        "Afzal, Z.R., Esmaeilbeig, T., Soltanalian, M. and Ohannessian, M.I., 2025. "
        "Linearization Explains Fine-Tuning in Large Language Models. In The "
        "Thirty-ninth Annual Conference on Neural Information Processing Systems"
    )
    assert _clean_venue(dump) is None


def test_clean_abstract_collapses_whitespace_and_normalizes_empty():
    assert _clean_abstract("line one\nline   two  ") == "line one line two"
    assert _clean_abstract("   ") is None
    assert _clean_abstract(None) is None


def test_search_semantic_scholar_happy_path_parses_normal_response():
    # Proves the Phase 1 defensive fixes below don't touch normal-response
    # parsing behavior at all.
    payload = {
        "data": [
            {
                "title": "A Great Paper",
                "authors": [{"name": "Jane Smith"}],
                "year": 2023,
                "venue": "NeurIPS",
                "abstract": "An abstract.",
                "externalIds": {"DOI": "10.1/x"},
                "citationCount": 12,
                "url": "https://example.org/paper",
                "paperId": "abc123",
            }
        ]
    }
    with patch("research_agent.ingestion.requests.get", return_value=_fake_response(200, payload)):
        papers = search_semantic_scholar("test query")
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "A Great Paper"
    assert p.authors == ["Jane Smith"]
    assert p.year == 2023
    assert p.doi == "10.1/x"
    assert p.paper_id == "abc123"


def test_search_semantic_scholar_handles_malformed_json_body_gracefully():
    with patch("research_agent.ingestion.requests.get", return_value=_fake_response(200, raise_on_json=True)):
        assert search_semantic_scholar("test query") == []


def test_parse_retry_after_accepts_plain_seconds():
    assert _parse_retry_after("5", default=1.0) == 5.0
    assert _parse_retry_after("2.5", default=1.0) == 2.5


def test_parse_retry_after_accepts_http_date():
    from datetime import datetime, timedelta, timezone

    future = datetime.now(timezone.utc) + timedelta(seconds=30)
    http_date = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    wait = _parse_retry_after(http_date, default=1.0)
    # Allow slack for test execution time between formatting and parsing.
    assert 25 <= wait <= 30


def test_parse_retry_after_falls_back_to_default_on_garbage():
    assert _parse_retry_after("not-a-date-or-number", default=3.5) == 3.5


def test_parse_retry_after_falls_back_to_default_when_missing():
    assert _parse_retry_after(None, default=3.5) == 3.5


def test_search_semantic_scholar_retry_after_date_header_does_not_crash():
    from datetime import datetime, timedelta, timezone

    future = datetime.now(timezone.utc) + timedelta(seconds=1)
    http_date = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    rate_limited = _fake_response(429, headers={"Retry-After": http_date})
    ok = _fake_response(200, {"data": []})
    with patch("research_agent.ingestion.requests.get", side_effect=[rate_limited, ok]), \
         patch("research_agent.ingestion.time.sleep"):
        assert search_semantic_scholar("test query", max_retries=2) == []


# --- Retry waits are bounded so one search cannot stall on a bad value ---

def test_parse_retry_after_small_numeric_delay_is_unchanged():
    assert _parse_retry_after("5", default=1.0) == 5.0
    assert _parse_retry_after("2.5", default=1.0) == 2.5
    assert _parse_retry_after("0", default=1.0) == 0.0


def test_parse_retry_after_caps_a_huge_numeric_delay():
    assert _parse_retry_after("3600", default=1.0) == _S2_MAX_RETRY_DELAY_SECONDS
    assert _parse_retry_after("999999", default=1.0) == _S2_MAX_RETRY_DELAY_SECONDS


def test_parse_retry_after_caps_a_far_future_http_date():
    from datetime import datetime, timedelta, timezone

    future = datetime.now(timezone.utc) + timedelta(days=1)
    http_date = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert _parse_retry_after(http_date, default=1.0) == _S2_MAX_RETRY_DELAY_SECONDS


def test_parse_retry_after_near_future_http_date_stays_approximately_correct():
    from datetime import datetime, timedelta, timezone

    future = datetime.now(timezone.utc) + timedelta(seconds=10)
    http_date = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    wait = _parse_retry_after(http_date, default=1.0)
    # Under the 30s cap, so it passes through; slack for execution time.
    assert 8.0 <= wait <= 10.0


def test_parse_retry_after_past_http_date_becomes_zero():
    from datetime import datetime, timedelta, timezone

    past = datetime.now(timezone.utc) - timedelta(hours=1)
    http_date = past.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert _parse_retry_after(http_date, default=5.0) == 0.0


def test_parse_retry_after_missing_and_malformed_use_a_bounded_fallback():
    assert _parse_retry_after(None, default=3.5) == 3.5
    assert _parse_retry_after("not-a-date-or-number", default=3.5) == 3.5
    # An over-large fallback is itself clamped.
    assert _parse_retry_after(None, default=10_000.0) == _S2_MAX_RETRY_DELAY_SECONDS
    assert _parse_retry_after("garbage", default=10_000.0) == _S2_MAX_RETRY_DELAY_SECONDS


def test_parse_retry_after_non_finite_and_negative_never_yield_invalid_delays():
    for raw in ("-30", "nan", "inf", "-inf", "infinity", "-infinity"):
        wait = _parse_retry_after(raw, default=1.0)
        assert math.isfinite(wait)
        assert 0.0 <= wait <= _S2_MAX_RETRY_DELAY_SECONDS
    for bad_default in (float("nan"), float("inf"), float("-inf"), -5.0):
        wait = _parse_retry_after(None, default=bad_default)
        assert math.isfinite(wait)
        assert 0.0 <= wait <= _S2_MAX_RETRY_DELAY_SECONDS


def test_search_semantic_scholar_429_then_success_sleeps_only_the_bounded_delay():
    rate_limited = _fake_response(429, headers={"Retry-After": "3600"})
    ok = _fake_response(200, {"data": []})
    with patch("research_agent.ingestion.requests.get", side_effect=[rate_limited, ok]), \
         patch("research_agent.ingestion.time.sleep") as mock_sleep:
        result = search_semantic_scholar("test query", max_retries=3)
    assert result == []
    assert mock_sleep.call_count == 1
    (slept,), _ = mock_sleep.call_args
    assert slept == _S2_MAX_RETRY_DELAY_SECONDS


def test_search_semantic_scholar_final_failed_attempt_does_not_sleep_afterward():
    rate_limited = _fake_response(429, headers={"Retry-After": "1"})
    with patch("research_agent.ingestion.requests.get", return_value=rate_limited), \
         patch("research_agent.ingestion.time.sleep") as mock_sleep:
        result = search_semantic_scholar("test query", max_retries=3)
    assert result == []
    # Sleeps after attempts 1 and 2 only, never after the last attempt.
    assert mock_sleep.call_count == 2


def test_search_semantic_scholar_records_one_child_call_per_http_attempt():
    rate_limited = _fake_response(429, headers={})
    ok = _fake_response(200, {"data": []})
    with patch("research_agent.ingestion.requests.get", side_effect=[rate_limited, ok]), \
         patch("research_agent.ingestion.time.sleep"), \
         patch("research_agent.ingestion.telemetry.record_child_call") as mock_rec:
        search_semantic_scholar("test query", max_retries=3)
    assert mock_rec.call_count == 2
    assert mock_rec.call_args_list[0].kwargs["outcome"] == "error"
    assert mock_rec.call_args_list[0].kwargs["error_type"] == "RateLimited"
    assert mock_rec.call_args_list[1].kwargs["outcome"] == "success"


if __name__ == "__main__":
    test_clean_venue_accepts_plausible_venue_names()
    test_clean_venue_rejects_citation_dumps()
    test_clean_abstract_collapses_whitespace_and_normalizes_empty()
    test_search_semantic_scholar_happy_path_parses_normal_response()
    test_search_semantic_scholar_handles_malformed_json_body_gracefully()
    test_parse_retry_after_accepts_plain_seconds()
    test_parse_retry_after_accepts_http_date()
    test_parse_retry_after_falls_back_to_default_on_garbage()
    test_parse_retry_after_falls_back_to_default_when_missing()
    test_search_semantic_scholar_retry_after_date_header_does_not_crash()
    test_parse_retry_after_small_numeric_delay_is_unchanged()
    test_parse_retry_after_caps_a_huge_numeric_delay()
    test_parse_retry_after_caps_a_far_future_http_date()
    test_parse_retry_after_near_future_http_date_stays_approximately_correct()
    test_parse_retry_after_past_http_date_becomes_zero()
    test_parse_retry_after_missing_and_malformed_use_a_bounded_fallback()
    test_parse_retry_after_non_finite_and_negative_never_yield_invalid_delays()
    test_search_semantic_scholar_429_then_success_sleeps_only_the_bounded_delay()
    test_search_semantic_scholar_final_failed_attempt_does_not_sleep_afterward()
    test_search_semantic_scholar_records_one_child_call_per_http_attempt()
    print("All ingestion tests passed.")
