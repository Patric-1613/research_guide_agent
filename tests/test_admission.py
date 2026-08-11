"""Usage Protection M2.1 Part B: tests for research_agent/admission.py.

Same isolation convention as tests/test_telemetry.py: the autouse
`usage_db_path` fixture redirects USAGE_DB_PATH to a fresh
tmp_path-scoped, already-initialized file before every test body runs,
and `test_real_usage_db_path_untouched` at the bottom proves the real
project DB was untouched by this file.

M2.1b: research_agent/admission.py does `from research_agent.telemetry
import USAGE_DB_PATH` -- a by-value import that creates its own,
independent `research_agent.admission.USAGE_DB_PATH` module attribute.
Patching only `telemetry.USAGE_DB_PATH` does NOT redirect admission.py's
own default (`path=None`) resolution -- every call in this file already
passes `path=usage_db_path` explicitly so that gap was never actually
exercised, but the fixture below also patches `admission.USAGE_DB_PATH`
directly so a future test that omits `path=` still fails closed into
the tmp file, never the real one.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import research_agent.admission as admission
import research_agent.telemetry as telemetry
from research_agent.admission import (
    check_global_paid_action_window,
    check_session_daily_budget,
    check_session_hourly_budget,
)
from research_agent.config import get_usage_policy
from research_agent.telemetry import init_usage_db
from tests._usage_db_fingerprint import fingerprint_usage_db

_REAL_USAGE_DB_PATH = telemetry.USAGE_DB_PATH
_REAL_USAGE_DB_FINGERPRINT_BEFORE = fingerprint_usage_db(_REAL_USAGE_DB_PATH)


@pytest.fixture(autouse=True)
def usage_db_path(tmp_path, monkeypatch):
    db_path = tmp_path / "usage.sqlite"
    monkeypatch.setattr(telemetry, "USAGE_DB_PATH", db_path)
    monkeypatch.setattr(admission, "USAGE_DB_PATH", db_path)
    init_usage_db(path=db_path).close()
    return db_path


def _seed_action(db_path, *, subject_type="session", subject_id="s1", started_at, outcome="success", action_type="search"):
    telemetry._write_paid_action(
        action_id=f"a-{started_at}-{subject_id}-{outcome}-{os.urandom(4).hex()}",
        action_type=action_type, request_id=None, subject_type=subject_type, subject_id=subject_id,
        outcome=outcome, started_at=started_at, ended_at=started_at, latency_ms=1.0,
        input_tokens=None, output_tokens=None, total_tokens=None, total_call_count=1,
        child_calls_json="[]", path=db_path,
    )


NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class TestHourlyBudget:
    def test_under_limit_is_allowed(self, usage_db_path):
        policy = get_usage_policy()
        for i in range(policy.max_paid_actions_per_session_per_hour - 1):
            _seed_action(usage_db_path, started_at=_iso(NOW - timedelta(minutes=1)), subject_id="s1")
        decision = check_session_hourly_budget("session", "s1", policy, now=NOW, path=usage_db_path)
        assert decision.allowed is True
        assert decision.reason_code == "ok"
        assert decision.current_count == policy.max_paid_actions_per_session_per_hour - 1
        assert decision.retry_after_seconds is None

    def test_limit_reached_is_rejected_with_positive_retry_after(self, usage_db_path):
        policy = get_usage_policy()
        oldest = NOW - timedelta(minutes=59)
        _seed_action(usage_db_path, started_at=_iso(oldest), subject_id="s1")
        for i in range(policy.max_paid_actions_per_session_per_hour - 1):
            _seed_action(usage_db_path, started_at=_iso(NOW - timedelta(minutes=1)), subject_id="s1")

        decision = check_session_hourly_budget("session", "s1", policy, now=NOW, path=usage_db_path)
        assert decision.allowed is False
        assert decision.reason_code == "session_hourly_limit_reached"
        assert decision.limit == policy.max_paid_actions_per_session_per_hour
        assert decision.current_count == policy.max_paid_actions_per_session_per_hour
        # Retry-After derived from the oldest counted action: it ages out
        # of the 1-hour window 60 minutes after it started, i.e. 1 minute
        # (60s) from `now` in this test.
        assert decision.retry_after_seconds is not None
        assert 0 < decision.retry_after_seconds <= 60

    def test_boundary_timestamp_exactly_at_window_edge_still_counts(self, usage_db_path):
        """started_at == now - window is inside the (inclusive) window --
        the check uses >=, so a row exactly on the boundary still counts
        against the budget."""
        policy = get_usage_policy()
        boundary = NOW - timedelta(hours=1)
        for i in range(policy.max_paid_actions_per_session_per_hour):
            _seed_action(usage_db_path, started_at=_iso(boundary), subject_id="s1")
        decision = check_session_hourly_budget("session", "s1", policy, now=NOW, path=usage_db_path)
        assert decision.allowed is False

    def test_just_outside_window_does_not_count(self, usage_db_path):
        policy = get_usage_policy()
        just_outside = NOW - timedelta(hours=1, seconds=1)
        for i in range(policy.max_paid_actions_per_session_per_hour):
            _seed_action(usage_db_path, started_at=_iso(just_outside), subject_id="s1")
        decision = check_session_hourly_budget("session", "s1", policy, now=NOW, path=usage_db_path)
        assert decision.allowed is True
        assert decision.current_count == 0

    def test_different_subjects_are_isolated(self, usage_db_path):
        policy = get_usage_policy()
        for i in range(policy.max_paid_actions_per_session_per_hour):
            _seed_action(usage_db_path, started_at=_iso(NOW - timedelta(minutes=1)), subject_id="s1")
        decision_s1 = check_session_hourly_budget("session", "s1", policy, now=NOW, path=usage_db_path)
        decision_s2 = check_session_hourly_budget("session", "s2", policy, now=NOW, path=usage_db_path)
        assert decision_s1.allowed is False
        assert decision_s2.allowed is True
        assert decision_s2.current_count == 0

    def test_counts_every_outcome_not_only_success(self, usage_db_path):
        policy = get_usage_policy()
        _seed_action(usage_db_path, started_at=_iso(NOW - timedelta(minutes=1)), subject_id="s1", outcome="error")
        _seed_action(usage_db_path, started_at=_iso(NOW - timedelta(minutes=1)), subject_id="s1", outcome="cancelled")
        decision = check_session_hourly_budget("session", "s1", policy, now=NOW, path=usage_db_path)
        assert decision.current_count == 2


class TestDailyBudget:
    def test_limit_reached(self, usage_db_path):
        policy = get_usage_policy()
        for i in range(policy.max_paid_actions_per_session_per_day):
            _seed_action(usage_db_path, started_at=_iso(NOW - timedelta(hours=1)), subject_id="s1")
        decision = check_session_daily_budget("session", "s1", policy, now=NOW, path=usage_db_path)
        assert decision.allowed is False
        assert decision.reason_code == "session_daily_limit_reached"

    def test_hourly_and_daily_are_independent_windows(self, usage_db_path):
        """A subject can be under the hourly limit but at the daily one
        (spread-out activity), or the reverse never happens since daily
        is a superset window -- this test locks in the independence of
        the two checks, not any ordering relationship."""
        policy = get_usage_policy()
        for i in range(policy.max_paid_actions_per_session_per_day):
            _seed_action(usage_db_path, started_at=_iso(NOW - timedelta(hours=20)), subject_id="s1")
        hourly = check_session_hourly_budget("session", "s1", policy, now=NOW, path=usage_db_path)
        daily = check_session_daily_budget("session", "s1", policy, now=NOW, path=usage_db_path)
        assert hourly.allowed is True
        assert daily.allowed is False


class TestGlobalWindow:
    def test_limit_reached_across_subjects(self, usage_db_path):
        policy = get_usage_policy()
        for i in range(policy.global_paid_action_limit):
            _seed_action(usage_db_path, started_at=_iso(NOW - timedelta(minutes=1)), subject_id=f"s{i}")
        decision = check_global_paid_action_window(policy, now=NOW, path=usage_db_path)
        assert decision.allowed is False
        assert decision.reason_code == "global_window_limit_reached"

    def test_under_limit_allowed(self, usage_db_path):
        policy = get_usage_policy()
        decision = check_global_paid_action_window(policy, now=NOW, path=usage_db_path)
        assert decision.allowed is True
        assert decision.current_count == 0


class TestFailClosed:
    def test_storage_error_is_rejected_not_allowed(self, usage_db_path):
        policy = get_usage_policy()
        # Simulate "cannot confirm this is safe": point at a path that
        # exists but is not a valid SQLite DB with the expected schema.
        broken_path = usage_db_path.parent / "not-a-real-db.sqlite"
        broken_path.write_text("this is not a sqlite database")
        decision = check_session_hourly_budget("session", "s1", policy, now=NOW, path=broken_path)
        assert decision.allowed is False
        assert decision.reason_code == "storage_unavailable"

    def test_missing_directory_is_rejected_not_allowed(self, usage_db_path, tmp_path):
        policy = get_usage_policy()
        missing = tmp_path / "does" / "not" / "exist" / "usage.sqlite"
        decision = check_session_hourly_budget("session", "s1", policy, now=NOW, path=missing)
        assert decision.allowed is False
        assert decision.reason_code == "storage_unavailable"


def test_real_usage_db_path_untouched():
    """Does NOT assert nonexistence -- a legitimate local
    usage_telemetry.sqlite from real dev-server use is normal, valid
    state. Proves nothing in this file's test run created, deleted, or
    modified it (or its -wal/-shm sidecars)."""
    assert fingerprint_usage_db(_REAL_USAGE_DB_PATH) == _REAL_USAGE_DB_FINGERPRINT_BEFORE
