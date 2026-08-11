"""Usage Protection M2.2A Part E: tests for research_agent/usage_guard.py
-- the reusable admission+lease context manager composed on top of
research_agent/admission.py and research_agent/leases.py (each already
covered on their own in tests/test_admission.py and tests/test_leases.py).
This file tests the COMPOSITION: rejection short-circuits before any
paid work runs, no paid_actions row is created on rejection, real
same-session concurrency admits exactly one caller, and the lease
releases correctly on every exit path.

Same isolation convention as tests/test_admission.py/test_leases.py:
admission.py and leases.py each import USAGE_DB_PATH by value from
telemetry.py (independent module attribute, confirmed in M2.1b), so the
fixture below patches all three bindings, not just telemetry's.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import research_agent.admission as admission
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.telemetry import init_usage_db
from research_agent.usage_guard import UsageGuardRejection, guard_paid_action
from tests._usage_db_fingerprint import fingerprint_usage_db

_REAL_USAGE_DB_PATH = telemetry.USAGE_DB_PATH
_REAL_USAGE_DB_FINGERPRINT_BEFORE = fingerprint_usage_db(_REAL_USAGE_DB_PATH)


@pytest.fixture(autouse=True)
def usage_db_path(tmp_path, monkeypatch):
    db_path = tmp_path / "usage.sqlite"
    monkeypatch.setattr(telemetry, "USAGE_DB_PATH", db_path)
    monkeypatch.setattr(admission, "USAGE_DB_PATH", db_path)
    monkeypatch.setattr(leases, "USAGE_DB_PATH", db_path)
    init_usage_db(path=db_path).close()
    return db_path


def _rows(db_path, table):
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}").fetchall()]
    finally:
        conn.close()


def _seed_paid_actions(db_path, subject_type, subject_id, count, started_at):
    for i in range(count):
        telemetry._write_paid_action(
            action_id=f"seed-{subject_type}-{subject_id}-{i}-{os.urandom(4).hex()}",
            action_type="search", request_id=None, subject_type=subject_type, subject_id=subject_id,
            outcome="success", started_at=started_at, ended_at=started_at, latency_ms=1.0,
            input_tokens=None, output_tokens=None, total_tokens=None, total_call_count=1,
            child_calls_json="[]", path=db_path,
        )


class TestAdmissionThroughGuard:
    def test_under_budget_action_proceeds(self, usage_db_path):
        ran = []
        with guard_paid_action("search"):
            ran.append(True)
        assert ran == [True]
        assert len(_rows(usage_db_path, "paid_actions")) == 1

    def test_hourly_limit_raises_429_with_retry_after(self, usage_db_path):
        from research_agent.config import get_usage_policy

        policy = get_usage_policy()
        now = datetime.now(timezone.utc)
        _seed_paid_actions(usage_db_path, "session", "s1", policy.max_paid_actions_per_session_per_hour, now.isoformat())

        with pytest.raises(UsageGuardRejection) as exc_info:
            with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
                pass
        assert exc_info.value.reason_code == "session_hourly_limit_reached"
        assert exc_info.value.retry_after_seconds is not None
        assert exc_info.value.retry_after_seconds > 0

    def test_daily_limit_raises_429_with_retry_after(self, usage_db_path):
        from research_agent.config import get_usage_policy

        policy = get_usage_policy()
        # Outside the hourly window but inside the daily one, so this
        # exercises the daily check specifically, not the hourly one.
        started_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        _seed_paid_actions(usage_db_path, "session", "s1", policy.max_paid_actions_per_session_per_day, started_at)

        with pytest.raises(UsageGuardRejection) as exc_info:
            with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
                pass
        assert exc_info.value.reason_code == "session_daily_limit_reached"
        assert exc_info.value.retry_after_seconds is not None

    def test_global_limit_raises_429(self, usage_db_path):
        from research_agent.config import get_usage_policy

        policy = get_usage_policy()
        now = datetime.now(timezone.utc).isoformat()
        for i in range(policy.global_paid_action_limit):
            _seed_paid_actions(usage_db_path, "session", f"other-{i}", 1, now)

        with pytest.raises(UsageGuardRejection) as exc_info:
            with guard_paid_action("search"):
                pass
        assert exc_info.value.reason_code == "global_window_limit_reached"
        assert exc_info.value.retry_after_seconds is not None

    def test_admission_storage_failure_raises_usage_protection_unavailable(self, usage_db_path, tmp_path, monkeypatch):
        broken = tmp_path / "does" / "not" / "exist" / "usage.sqlite"
        monkeypatch.setattr(admission, "USAGE_DB_PATH", broken)

        with pytest.raises(UsageGuardRejection) as exc_info:
            with guard_paid_action("curation_chat", subject=("session", "s1")):
                pass
        assert exc_info.value.reason_code == "usage_protection_unavailable"

    def test_rejected_action_performs_zero_operations(self, usage_db_path):
        from research_agent.config import get_usage_policy

        policy = get_usage_policy()
        now = datetime.now(timezone.utc).isoformat()
        _seed_paid_actions(usage_db_path, "session", "s1", policy.max_paid_actions_per_session_per_hour, now)

        calls = []
        with pytest.raises(UsageGuardRejection):
            with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
                calls.append("provider call")  # must never execute
        assert calls == []

    def test_rejected_action_creates_no_paid_actions_row(self, usage_db_path):
        from research_agent.config import get_usage_policy

        policy = get_usage_policy()
        now = datetime.now(timezone.utc).isoformat()
        _seed_paid_actions(usage_db_path, "session", "s1", policy.max_paid_actions_per_session_per_hour, now)
        before = len(_rows(usage_db_path, "paid_actions"))

        with pytest.raises(UsageGuardRejection):
            with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
                pass

        after = len(_rows(usage_db_path, "paid_actions"))
        assert after == before  # no new row from the rejected attempt

    def test_different_subjects_are_isolated(self, usage_db_path, monkeypatch):
        from research_agent.config import get_usage_policy

        # Raised well above the hourly limit being tested so seeding s1's
        # own hourly budget full doesn't also trip the coarse GLOBAL check
        # this same seed data counts against -- this test is isolating the
        # per-subject hourly check specifically, not the global one.
        monkeypatch.setenv("USAGE_GLOBAL_PAID_ACTION_LIMIT", "1000")
        policy = get_usage_policy()
        now = datetime.now(timezone.utc).isoformat()
        _seed_paid_actions(usage_db_path, "session", "s1", policy.max_paid_actions_per_session_per_hour, now)

        with pytest.raises(UsageGuardRejection) as exc_info:
            with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
                pass
        assert exc_info.value.reason_code == "session_hourly_limit_reached"

        ran = []
        with guard_paid_action("curation_chat", subject=("session", "s2"), use_lease=True):
            ran.append(True)
        assert ran == [True]


class TestLeaseThroughGuard:
    def test_lease_released_after_success(self, usage_db_path):
        with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
            pass
        # Released -- a second attempt now succeeds.
        with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
            pass

    def test_lease_released_after_normal_exception(self, usage_db_path):
        with pytest.raises(RuntimeError):
            with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
                raise RuntimeError("boom")
        with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
            pass

    def test_lease_released_after_cancellation(self, usage_db_path):
        """The context manager's own finally must release on ANY
        exception type propagating through the `with` block, including
        asyncio.CancelledError -- Python's contextmanager finally
        doesn't distinguish exception types. Real HTTP-level
        cancellation of a sync, threadpool-executed FastAPI route
        cannot interrupt a running worker thread mid-body (confirmed
        during inspection: the thread runs to natural completion
        regardless of client disconnect), so this is the correct,
        honest scope for a "cancellation" test at this layer."""
        import asyncio

        with pytest.raises(asyncio.CancelledError):
            with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
                raise asyncio.CancelledError()
        with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
            pass

    def test_stale_lease_can_be_replaced_after_expiry(self, usage_db_path, monkeypatch):
        import time

        monkeypatch.setenv("USAGE_EXPENSIVE_ACTION_LEASE_TTL_SECONDS", "1")
        with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
            pass
        time.sleep(1.05)
        # Expired -- a fresh acquisition succeeds without waiting for release.
        with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
            pass

    def test_retry_after_reflects_remaining_lease_duration(self, usage_db_path, monkeypatch):
        monkeypatch.setenv("USAGE_EXPENSIVE_ACTION_LEASE_TTL_SECONDS", "120")
        with pytest.raises(UsageGuardRejection) as exc_info:
            with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
                with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
                    pass
        assert exc_info.value.reason_code == "action_in_progress"
        assert 0 < exc_info.value.retry_after_seconds <= 120

    def test_rejected_nested_attempt_does_not_release_the_active_lease(self, usage_db_path):
        """A rejected acquisition never holds a real token, so its own
        (no-op) release call cannot free the lease out from under the
        request that's still legitimately holding it."""
        with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
            with pytest.raises(UsageGuardRejection):
                with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
                    pass
            # Still held by the outer guard -- a third concurrent attempt
            # (still nested inside the outer's own `with`) must also fail.
            with pytest.raises(UsageGuardRejection):
                with guard_paid_action("curation_chat", subject=("session", "s1"), use_lease=True):
                    pass

    def test_different_sessions_proceed_concurrently_real_threads(self, usage_db_path):
        n_threads = 8
        results: dict[int, bool] = {}
        barrier = threading.Barrier(n_threads)

        def attempt(i):
            barrier.wait()
            try:
                with guard_paid_action("curation_chat", subject=("session", f"s{i}"), use_lease=True):
                    results[i] = True
            except UsageGuardRejection:
                results[i] = False

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results.values())

    def test_two_concurrent_same_session_requests_admit_exactly_one(self, usage_db_path):
        """Real OS threads racing for the same session's lease -- not
        sequential calls dressed up as a concurrency test. Whichever
        thread wins holds the lease open (blocked on `release_event`)
        for the whole race window, so every other thread's own
        acquisition attempt genuinely overlaps with an actively-held
        lease -- without this, a winner that acquires-then-immediately-
        releases lets a second thread win sequentially afterward, which
        would prove nothing about same-instant exclusion."""
        n_threads = 12
        admitted: list[bool] = [False] * n_threads
        start_barrier = threading.Barrier(n_threads)
        release_event = threading.Event()

        def attempt(i):
            start_barrier.wait()
            try:
                with guard_paid_action("curation_chat", subject=("session", "contested"), use_lease=True):
                    admitted[i] = True
                    release_event.wait(timeout=5)
            except UsageGuardRejection:
                admitted[i] = False

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        time.sleep(0.3)  # let every thread reach and lose its own attempt
        release_event.set()
        for t in threads:
            t.join()

        assert sum(admitted) == 1


def test_real_usage_db_path_untouched():
    assert fingerprint_usage_db(_REAL_USAGE_DB_PATH) == _REAL_USAGE_DB_FINGERPRINT_BEFORE
