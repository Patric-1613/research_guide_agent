"""Usage Protection M2.1 Part C: tests for research_agent/leases.py.

Concurrency correctness is proven with real OS threads racing against a
real (tmp_path-scoped) SQLite file, not with sequential calls or mocks --
that is the whole point of the module's atomicity claim.

M2.1b: research_agent/leases.py does `from research_agent.telemetry
import USAGE_DB_PATH` -- a by-value import that creates its own,
independent `research_agent.leases.USAGE_DB_PATH` module attribute.
Patching only `telemetry.USAGE_DB_PATH` does NOT redirect leases.py's
own default (`path=None`) resolution -- every call in this file already
passes `path=usage_db_path` explicitly so that gap was never actually
exercised, but the fixture below also patches `leases.USAGE_DB_PATH`
directly so a future test that omits `path=` still fails closed into
the tmp file, never the real one.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.leases import (
    EXPENSIVE_ACTION_GROUP,
    acquire_lease,
    release_lease,
    session_lease,
)
from research_agent.telemetry import init_usage_db
from tests._usage_db_fingerprint import fingerprint_usage_db

_REAL_USAGE_DB_PATH = telemetry.USAGE_DB_PATH
_REAL_USAGE_DB_FINGERPRINT_BEFORE = fingerprint_usage_db(_REAL_USAGE_DB_PATH)


@pytest.fixture(autouse=True)
def usage_db_path(tmp_path, monkeypatch):
    db_path = tmp_path / "usage.sqlite"
    monkeypatch.setattr(telemetry, "USAGE_DB_PATH", db_path)
    monkeypatch.setattr(leases, "USAGE_DB_PATH", db_path)
    init_usage_db(path=db_path).close()
    return db_path


class TestAcquireRelease:
    def test_first_acquisition_succeeds(self, usage_db_path):
        decision = acquire_lease("session", "s1", path=usage_db_path)
        assert decision.acquired is True
        assert decision.reason_code == "ok"
        assert decision.token is not None

    def test_second_acquisition_while_active_is_rejected(self, usage_db_path):
        first = acquire_lease("session", "s1", path=usage_db_path)
        second = acquire_lease("session", "s1", path=usage_db_path)
        assert first.acquired is True
        assert second.acquired is False
        assert second.reason_code == "lease_held"
        assert second.token is None

    def test_different_sessions_do_not_block_each_other(self, usage_db_path):
        first = acquire_lease("session", "s1", path=usage_db_path)
        second = acquire_lease("session", "s2", path=usage_db_path)
        assert first.acquired is True
        assert second.acquired is True

    def test_different_action_groups_do_not_collide(self, usage_db_path):
        first = acquire_lease("session", "s1", "report_generate", path=usage_db_path)
        second = acquire_lease("session", "s1", "chat", path=usage_db_path)
        assert first.acquired is True
        assert second.acquired is True

    def test_expired_lease_is_replaced(self, usage_db_path):
        first = acquire_lease("session", "s1", ttl_seconds=0, path=usage_db_path)
        assert first.acquired is True
        # ttl_seconds=0 means expires_at == acquired_at, already in the
        # past relative to any later "now" -- a fresh acquisition attempt
        # should treat it as expired and win.
        time.sleep(0.01)
        second = acquire_lease("session", "s1", ttl_seconds=60, path=usage_db_path)
        assert second.acquired is True
        assert second.token != first.token

    def test_wrong_token_release_does_nothing(self, usage_db_path):
        first = acquire_lease("session", "s1", path=usage_db_path)
        ok = release_lease("session", "s1", EXPENSIVE_ACTION_GROUP, "not-the-real-token", path=usage_db_path)
        assert ok is True  # storage succeeded; it just matched nothing
        # Lease is still held -- a second acquisition attempt still loses.
        second = acquire_lease("session", "s1", path=usage_db_path)
        assert second.acquired is False

    def test_correct_release_permits_reacquisition(self, usage_db_path):
        first = acquire_lease("session", "s1", path=usage_db_path)
        release_lease("session", "s1", EXPENSIVE_ACTION_GROUP, first.token, path=usage_db_path)
        second = acquire_lease("session", "s1", path=usage_db_path)
        assert second.acquired is True

    def test_release_is_idempotent(self, usage_db_path):
        first = acquire_lease("session", "s1", path=usage_db_path)
        assert release_lease("session", "s1", EXPENSIVE_ACTION_GROUP, first.token, path=usage_db_path) is True
        # Releasing again (already gone) must not error.
        assert release_lease("session", "s1", EXPENSIVE_ACTION_GROUP, first.token, path=usage_db_path) is True

    def test_release_of_nonexistent_lease_is_a_safe_noop(self, usage_db_path):
        assert release_lease("session", "never-acquired", EXPENSIVE_ACTION_GROUP, "some-token", path=usage_db_path) is True


class TestContextManager:
    def test_releases_after_success(self, usage_db_path):
        with session_lease("session", "s1", path=usage_db_path) as decision:
            assert decision.acquired is True
        # Released -- a fresh acquisition now succeeds.
        assert acquire_lease("session", "s1", path=usage_db_path).acquired is True

    def test_releases_after_exception(self, usage_db_path):
        with pytest.raises(RuntimeError):
            with session_lease("session", "s1", path=usage_db_path) as decision:
                assert decision.acquired is True
                raise RuntimeError("boom")
        assert acquire_lease("session", "s1", path=usage_db_path).acquired is True

    def test_does_not_release_a_lease_it_never_acquired(self, usage_db_path):
        held = acquire_lease("session", "s1", path=usage_db_path)
        assert held.acquired is True
        with session_lease("session", "s1", path=usage_db_path) as decision:
            assert decision.acquired is False
        # The ORIGINAL holder's lease must still be intact -- the failed
        # context manager attempt must not have released someone else's lease.
        still_blocked = acquire_lease("session", "s1", path=usage_db_path)
        assert still_blocked.acquired is False

    def test_does_not_raise_on_rejection(self, usage_db_path):
        acquire_lease("session", "s1", path=usage_db_path)
        # Must not raise -- an ordinary rejection is a structured decision.
        with session_lease("session", "s1", path=usage_db_path) as decision:
            assert decision.acquired is False
            assert decision.reason_code == "lease_held"


class TestRealConcurrency:
    def test_genuine_concurrent_acquisition_permits_exactly_one_winner(self, usage_db_path):
        """Real OS threads, real SQLite file, real race -- not sequential
        calls dressed up as a concurrency test."""
        n_threads = 16
        results: list[bool] = [False] * n_threads
        barrier = threading.Barrier(n_threads)

        def attempt(i):
            barrier.wait()  # maximize actual overlap
            decision = acquire_lease("session", "contested", path=usage_db_path)
            results[i] = decision.acquired

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(results) == 1

    def test_concurrent_acquisition_across_distinct_sessions_all_win(self, usage_db_path):
        n_threads = 8
        results: list[bool] = [False] * n_threads
        barrier = threading.Barrier(n_threads)

        def attempt(i):
            barrier.wait()
            decision = acquire_lease("session", f"s{i}", path=usage_db_path)
            results[i] = decision.acquired

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)


class TestFailClosed:
    def test_storage_error_on_acquire_is_rejected(self, usage_db_path):
        broken_path = usage_db_path.parent / "not-a-real-db.sqlite"
        broken_path.write_text("this is not a sqlite database")
        decision = acquire_lease("session", "s1", path=broken_path)
        assert decision.acquired is False
        assert decision.reason_code == "storage_unavailable"

    def test_missing_directory_on_acquire_is_rejected(self, usage_db_path, tmp_path):
        missing = tmp_path / "does" / "not" / "exist" / "usage.sqlite"
        decision = acquire_lease("session", "s1", path=missing)
        assert decision.acquired is False
        assert decision.reason_code == "storage_unavailable"

    def test_storage_error_on_release_returns_false(self, usage_db_path, tmp_path):
        missing = tmp_path / "does" / "not" / "exist" / "usage.sqlite"
        assert release_lease("session", "s1", EXPENSIVE_ACTION_GROUP, "tok", path=missing) is False


def test_real_usage_db_path_untouched():
    """Does NOT assert nonexistence -- a legitimate local
    usage_telemetry.sqlite from real dev-server use is normal, valid
    state. Proves nothing in this file's test run created, deleted, or
    modified it (or its -wal/-shm sidecars)."""
    assert fingerprint_usage_db(_REAL_USAGE_DB_PATH) == _REAL_USAGE_DB_FINGERPRINT_BEFORE
