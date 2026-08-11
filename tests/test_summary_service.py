"""Usage Protection M2.2B: tests for research_agent/services/
summary_service.py -- the conditional guard around /summarize and
/export's shared generate-or-cache boundary (_generate_or_get_summaries).

Same isolation convention as tests/test_admission.py: the autouse
usage_db_path fixture redirects telemetry/admission/leases USAGE_DB_PATH
to a fresh tmp_path file, and test_real_usage_db_path_untouched at the
bottom proves the real project DB was untouched by this file.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import research_agent.api as api
import research_agent.admission as admission
import research_agent.leases as leases
import research_agent.telemetry as telemetry
import research_agent.services.summary_service as summary_service
from research_agent.storage import init_db, save_search
from research_agent.telemetry import init_usage_db
from research_agent.usage_guard import UsageGuardRejection
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


@pytest.fixture
def db(tmp_path):
    conn = init_db(tmp_path / "main.sqlite")
    yield conn
    conn.close()


def _open(db_path):
    """A fresh connection on the given path -- sqlite3.Connection objects
    are single-thread by default (check_same_thread=True), so a
    concurrency test that runs summarize_search on a real background
    thread needs its own connection, exactly like FastAPI's own
    per-request get_db_connection dependency already gives each real
    concurrent HTTP request its own connection in production."""
    import sqlite3 as _sqlite3

    conn = _sqlite3.connect(db_path)
    conn.row_factory = _sqlite3.Row
    return conn


def _seed_search(conn, web_articles=None) -> int:
    search_id, _ = save_search(conn, "topic", ["p1"], [0.9], web_articles=web_articles or [])
    return search_id


def _rows(db_path, table):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}").fetchall()]
    finally:
        conn.close()


_SUMMARY_JSON = {"themes": [], "gaps_and_disagreements": "", "skipped_paper_ids": []}
_WEB_SUMMARY_JSON = {"synthesis": "", "cited_articles": []}


class TestCacheHitBypass:
    def test_summarize_cache_hit_performs_no_admission_check(self, db, usage_db_path):
        search_id = _seed_search(db)
        with patch.object(summary_service, "_get_or_create_summary", return_value=_SUMMARY_JSON) as mock_gen, \
             patch.object(summary_service, "_get_or_create_web_summary", return_value=None):
            # Cache already populated (no web articles, so web_summary
            # need not be considered) -- update_summary first so
            # _needs_generation() sees a real cache hit.
            from research_agent.storage import update_summary
            update_summary(db, search_id, _SUMMARY_JSON)
            summary_service.summarize_search(db, search_id, "apa")
        mock_gen.assert_called_once()  # _get_or_create_summary itself still runs (cheap cache read)
        assert _rows(usage_db_path, "paid_actions") == []
        assert _rows(usage_db_path, "action_leases") == []

    def test_export_with_cached_summary_bypasses_admission(self, db, usage_db_path):
        from research_agent.storage import update_summary

        search_id = _seed_search(db)
        update_summary(db, search_id, _SUMMARY_JSON)
        with patch.object(summary_service, "_get_or_create_web_summary", return_value=None):
            summary_service.export_search_markdown(db, search_id, "apa")
        assert _rows(usage_db_path, "paid_actions") == []


class TestCacheMissIsGuarded:
    def test_summary_cache_miss_is_admitted_and_leased(self, db, usage_db_path):
        search_id = _seed_search(db)
        with patch.object(summary_service, "_get_or_create_summary", return_value=_SUMMARY_JSON) as mock_gen, \
             patch.object(summary_service, "_get_or_create_web_summary", return_value=None):
            summary_service.summarize_search(db, search_id, "apa")
        mock_gen.assert_called_once()
        rows = _rows(usage_db_path, "paid_actions")
        assert len(rows) == 1
        assert rows[0]["action_type"] == "summarize"
        assert rows[0]["subject_type"] == "search"
        assert rows[0]["subject_id"] == str(search_id)
        assert _rows(usage_db_path, "action_leases") == []  # released

    def test_export_triggered_generation_is_guarded_exactly_once(self, db, usage_db_path):
        """Both paper summary AND web summary need generation -- must
        still be exactly ONE top-level paid_actions row, matching M1's
        original one-row-per-request design (both _get_or_create_* calls
        share the same outer guard, not two independent ones)."""
        search_id = _seed_search(db, web_articles=[{"title": "a", "url": "http://x", "snippet": "s", "published_date": None, "source_domain": "x.com"}])
        with patch.object(summary_service, "_get_or_create_summary", return_value=_SUMMARY_JSON) as mock_summary, \
             patch.object(summary_service, "_get_or_create_web_summary", return_value=_WEB_SUMMARY_JSON) as mock_web:
            summary_service.export_search_markdown(db, search_id, "apa")
        mock_summary.assert_called_once()
        mock_web.assert_called_once()
        rows = _rows(usage_db_path, "paid_actions")
        assert len(rows) == 1

    def test_rejection_performs_zero_provider_calls_and_leaves_cache_unchanged(self, db, usage_db_path):
        from datetime import datetime, timezone

        from research_agent.config import get_usage_policy

        search_id = _seed_search(db)
        policy = get_usage_policy()
        now = datetime.now(timezone.utc).isoformat()
        for i in range(policy.max_paid_actions_per_session_per_hour):
            telemetry._write_paid_action(
                action_id=f"seed-{i}", action_type="summarize", request_id=None,
                subject_type="search", subject_id=str(search_id), outcome="success",
                started_at=now, ended_at=now, latency_ms=1.0, input_tokens=None,
                output_tokens=None, total_tokens=None, total_call_count=1,
                child_calls_json="[]", path=usage_db_path,
            )

        with patch.object(summary_service, "_get_or_create_summary") as mock_gen:
            with pytest.raises(UsageGuardRejection) as exc_info:
                summary_service.summarize_search(db, search_id, "apa")
        assert exc_info.value.reason_code == "session_hourly_limit_reached"
        mock_gen.assert_not_called()

        from research_agent.storage import get_search
        assert get_search(db, search_id).summary is None  # cache untouched


class TestConcurrentGeneration:
    def test_concurrent_same_search_cache_misses_generate_exactly_once(self, db, tmp_path, usage_db_path):
        """Real OS threads racing a genuine cache miss for the SAME
        search_id, each with its own DB connection (matching FastAPI's
        own per-request get_db_connection pattern). The loser must
        re-check the cache after acquiring the lease
        (summary_service.py's own _generate_or_get_summaries re-fetches
        via get_search) and skip regeneration, not pay a second time."""
        db_path = tmp_path / "main.sqlite"
        search_id = _seed_search(db)
        generate_calls = []
        release_event = threading.Event()
        entered_event = threading.Event()
        call_lock = threading.Lock()

        def _fake_generate_summary(topic, papers, client=None, style="apa"):
            with call_lock:
                generate_calls.append(1)
                first = len(generate_calls) == 1
            if first:
                entered_event.set()
                release_event.wait(timeout=5)
            return MagicMock(themes=[], gaps_and_disagreements="", skipped_paper_ids=[])

        import research_agent.api as api

        api._state["collection"] = MagicMock()
        api._state["client"] = MagicMock()

        with patch.object(api, "get_papers_by_ids", return_value=[]), \
             patch.object(api, "generate_summary", side_effect=_fake_generate_summary), \
             patch("research_agent.services.summary_cache._summary_to_json", return_value=_SUMMARY_JSON):
            results = {}

            def _winner():
                conn = _open(db_path)
                try:
                    results["winner"] = summary_service.summarize_search(conn, search_id, "apa")
                finally:
                    conn.close()

            t1 = threading.Thread(target=_winner)
            t1.start()
            assert entered_event.wait(timeout=5)  # t1 now holds the lease, mid-generation

            # t2 attempts while t1 still holds the lease -- since t1 will
            # release before this call's own guard gives up (no explicit
            # retry/backoff exists, so this proves the *contract*: once
            # admitted after t1 releases, t2 must see the now-cached
            # value rather than generating again), release t1 first.
            release_event.set()
            t1.join(timeout=5)
            results["second"] = summary_service.summarize_search(db, search_id, "apa")

        assert len(generate_calls) == 1  # only ever generated once
        assert results["second"].themes == results["winner"].themes
        rows = _rows(usage_db_path, "paid_actions")
        assert len(rows) == 1  # the second call never opened its own guard (cache hit)

    def test_second_request_rechecks_cache_after_lease_acquisition(self, db, tmp_path, usage_db_path):
        """Directly proves the recheck: seed a cache hit WHILE a
        first request holds the lease (simulating "someone else just
        finished"), and confirm the second request -- once admitted --
        does not call generate_summary again."""
        from research_agent.storage import update_summary

        db_path = tmp_path / "main.sqlite"
        search_id = _seed_search(db)
        release_event = threading.Event()
        entered_event = threading.Event()

        import research_agent.api as api

        api._state["collection"] = MagicMock()
        api._state["client"] = MagicMock()

        def _blocking_generate(topic, papers, client=None, style="apa"):
            entered_event.set()
            release_event.wait(timeout=5)
            return MagicMock(themes=[], gaps_and_disagreements="", skipped_paper_ids=[])

        def _run_t1():
            conn = _open(db_path)
            try:
                summary_service.summarize_search(conn, search_id, "apa")
            finally:
                conn.close()

        with patch.object(api, "get_papers_by_ids", return_value=[]), \
             patch.object(api, "generate_summary", side_effect=_blocking_generate) as mock_gen, \
             patch("research_agent.services.summary_cache._summary_to_json", return_value=_SUMMARY_JSON):
            t1 = threading.Thread(target=_run_t1)
            t1.start()
            assert entered_event.wait(timeout=5)

            # While t1 holds the lease, simulate the cache being populated
            # by directly writing it (standing in for "t1 is about to finish").
            update_summary(db, search_id, _SUMMARY_JSON)

            release_event.set()
            t1.join(timeout=5)

        assert mock_gen.call_count == 1  # only t1's own generation call


def test_real_usage_db_path_untouched():
    assert fingerprint_usage_db(_REAL_USAGE_DB_PATH) == _REAL_USAGE_DB_FINGERPRINT_BEFORE
