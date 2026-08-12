"""Usage Protection M4.2A Part F: integration tests for
`services/curation_chat_service.py::stream_answer_curation_chat` --
calling the SERVICE function directly (not through HTTP/TestClient),
consuming the returned `StreamingResponse`'s own `body_iterator`
manually. Complements tests/test_curation_chat_stream_api.py (full
HTTP/preflight contracts) and tests/test_curation_chat_streaming.py
(the domain generator in isolation, no guard/telemetry at all) by
proving the real lease + `telemetry.paid_action` lifecycle end to end
through the actual service/router-adjacent code path, including
cancellation.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import research_agent.admission as admission
import research_agent.api as api
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.api_app.schemas import CurationChatRequest
from research_agent.chat_streaming import build_started_event
from research_agent.curation_chat_streaming import HandledStreamFailure
from research_agent.curation_session import save_curation_session
from research_agent.qa import sqlite_checkpointer
from research_agent.query_expansion import PaperPoolSession
from research_agent.services.curation_chat_service import stream_answer_curation_chat
from research_agent.telemetry import init_usage_db
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


@pytest.fixture
def session_setup(tmp_path):
    cp_db_path = Path(tmp_path) / "checkpoints.sqlite"
    session_id = "svc-test-session"
    session = PaperPoolSession(topic="AI safety", stage="synthesize")
    with sqlite_checkpointer(cp_db_path) as cp:
        save_curation_session(session, session_id, cp)
        yield session_id, cp


async def _consume(response) -> list[str]:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return chunks


def test_success_records_one_paid_action_row_and_releases_lease(usage_db_path, session_setup):
    session_id, cp = session_setup
    req = CurationChatRequest(message="anything?")

    async def scenario():
        response = stream_answer_curation_chat(session_id, req, cp)
        return await _consume(response)

    with patch.object(api, "_state", {"client": object(), "async_client": object()}):
        chunks = asyncio.run(scenario())

    assert any("event: started" in c for c in chunks)
    assert any("event: completed" in c for c in chunks)
    assert any("event: done" in c for c in chunks)

    rows = _rows(usage_db_path, "paid_actions")
    assert len(rows) == 1
    assert rows[0]["action_type"] == "curation_chat"
    assert rows[0]["outcome"] == "success"
    assert len(_rows(usage_db_path, "action_leases")) == 0


def test_cancellation_reraises_records_cancelled_outcome_and_releases_lease(usage_db_path, session_setup):
    session_id, cp = session_setup
    req = CurationChatRequest(message="what does the web say?")

    async def scenario():
        response = stream_answer_curation_chat(session_id, req, cp)
        with pytest.raises(asyncio.CancelledError):
            await _consume(response)

    with patch.object(api, "_state", {"client": object(), "async_client": object()}), \
         patch("research_agent.curation_chat_streaming.stream_chat_answer") as mock_stream, \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)):
        # A session with no papers/web articles reaches the no_sources
        # early-return branch before ever calling stream_chat_answer --
        # force the "ready to generate" branch isn't needed here since
        # what's under test is cancellation SURFACING correctly, which
        # this test drives directly by raising CancelledError from
        # within stream_answer_curation_chat's own event_stream body via
        # a monkeypatched stream_curation_chat_turn instead, for a
        # dependency-free, deterministic trigger.
        async def _raise_cancelled(*args, **kwargs):
            yield build_started_event()
            raise asyncio.CancelledError()

        with patch("research_agent.services.curation_chat_service.stream_curation_chat_turn", _raise_cancelled):
            asyncio.run(scenario())

    rows = _rows(usage_db_path, "paid_actions")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "cancelled"
    assert len(_rows(usage_db_path, "action_leases")) == 0


def _drive_with_handled_failure(session_id, cp, req, reason_code, message):
    """Shared driver for the two "handled failure -> outcome=error"
    tests below: forces stream_curation_chat_turn to raise HandledStream
    Failure (the exact typed signal both the model-adapter-error and
    persistence-error real code paths now raise -- see tests/test_
    curation_chat_streaming.py for proof those real paths raise this
    with these exact reason codes) immediately after `started`, then
    consumes the SERVICE's own real event_stream(), which must catch it
    OUTSIDE the telemetry.paid_action block and convert it to safe SSE
    frames -- this is what's under test here, not where the exception
    originates."""

    async def _raise_handled_failure(*args, **kwargs):
        yield build_started_event()
        raise HandledStreamFailure(reason_code, message)

    async def scenario():
        with patch.object(api, "_state", {"client": object(), "async_client": object()}), \
             patch("research_agent.services.curation_chat_service.stream_curation_chat_turn", _raise_handled_failure):
            response = stream_answer_curation_chat(session_id, req, cp)
            return await _consume(response)

    return asyncio.run(scenario())


def test_handled_provider_failure_records_top_level_outcome_error(usage_db_path, session_setup):
    session_id, cp = session_setup
    req = CurationChatRequest(message="anything?")

    chunks = _drive_with_handled_failure(
        session_id, cp, req, "provider_error", "The model provider returned an error.",
    )

    rows = _rows(usage_db_path, "paid_actions")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "error"
    assert len(_rows(usage_db_path, "action_leases")) == 0


def test_handled_persistence_failure_records_top_level_outcome_error(usage_db_path, session_setup):
    session_id, cp = session_setup
    req = CurationChatRequest(message="anything?")

    chunks = _drive_with_handled_failure(
        session_id, cp, req, "persistence_failed", "Failed to save the conversation.",
    )

    rows = _rows(usage_db_path, "paid_actions")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "error"
    assert len(_rows(usage_db_path, "action_leases")) == 0


def test_handled_failure_yields_safe_error_then_done_never_completed(usage_db_path, session_setup):
    session_id, cp = session_setup
    req = CurationChatRequest(message="anything?")

    chunks = _drive_with_handled_failure(
        session_id, cp, req, "provider_error", "The model provider returned an error.",
    )
    joined = "".join(chunks)

    assert "event: started" in joined
    assert "event: error" in joined
    assert '"reason_code":"provider_error"' in joined
    assert '"message":"The model provider returned an error."' in joined
    assert "event: done" in joined
    assert "event: completed" not in joined
    # Only ever the safe, fixed message -- no exception class name/text
    # of any kind (this test's own driver never raises a raw exception,
    # but this also guards against a future regression that starts
    # str()-ing something unsafe into the payload).
    assert "Traceback" not in joined
    assert "RuntimeError" not in joined


def test_lease_remains_held_through_persistence_and_releases_only_after_cancellation_settles(usage_db_path, session_setup):
    """Lifecycle-hardening checkpoint: the core proof that a shielded
    persistence task cannot outlive the lease. A real, controllable,
    slow save runs in its own OS thread; the consuming Task is
    cancelled WHILE that save is genuinely still in flight; the lease
    must still be held at that moment, the save must still run to full
    completion (never aborted), `completed` must never be emitted, and
    only once the save has settled may the lease actually be released."""
    session_id, cp = session_setup
    req = CurationChatRequest(message="anything?")

    started = threading.Event()
    proceed = threading.Event()
    finished = threading.Event()

    def _slow_save(session, sid, checkpointer):
        started.set()
        assert proceed.wait(timeout=5), "test driver never released the save -- deadlock"
        save_curation_session(session, sid, checkpointer)
        finished.set()

    collected: list[str] = []

    async def consume():
        response = stream_answer_curation_chat(session_id, req, cp)
        async for chunk in response.body_iterator:
            collected.append(chunk)

    async def scenario():
        with patch.object(api, "_state", {"client": object(), "async_client": object()}), \
             patch("research_agent.curation_chat_streaming.save_curation_session", _slow_save), \
             patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)):
            task = asyncio.create_task(consume())

            while not started.is_set():
                await asyncio.sleep(0.005)

            task.cancel()

            # Give the cancellation time to reach _persist_and_complete's
            # own await asyncio.shield(save_task) and enter its cleanup
            # -- it must be blocked waiting on the save task, NOT past
            # the caller's own lease-release finally, at this point.
            await asyncio.sleep(0.05)
            assert not finished.is_set(), "the save settled before the assertion window -- test is not exercising the race"
            assert len(_rows(usage_db_path, "action_leases")) == 1, "lease released while persistence was still in flight"

            proceed.set()  # let the real, retained save task actually finish

            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())

    assert finished.is_set()  # the save genuinely ran to completion, never aborted
    assert len(_rows(usage_db_path, "action_leases")) == 0  # released only afterward
    rows = _rows(usage_db_path, "paid_actions")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "cancelled"
    assert not any("event: completed" in c for c in collected)


def test_real_usage_db_path_untouched():
    assert fingerprint_usage_db(_REAL_USAGE_DB_PATH) == _REAL_USAGE_DB_FINGERPRINT_BEFORE
