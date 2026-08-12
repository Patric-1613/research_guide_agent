"""Usage Protection M4.2A Part B: focused lifecycle tests for
research_agent/usage_guard.py's `open_admission_and_lease_for_streaming`/
`StreamingLeaseHandle` -- written BEFORE the streaming route was wired
up, per this phase's own required ordering. Same isolation convention
as tests/test_usage_guard.py (patches USAGE_DB_PATH on telemetry/
admission/leases together).

No network/provider calls anywhere in this file -- these tests only
exercise the guard's own admission/lease composition and the
recommended `telemetry.paid_action` usage pattern, never a real OpenAI
stream.

This file also DELIBERATELY includes a regression test
(`test_lease_handle_safely_crosses_a_real_task_boundary_unlike_
telemetry_paid_action`) proving the exact bug an earlier version of
this primitive had: driving `telemetry.paid_action`'s own `__enter__`/
`__exit__` by hand across a genuine asyncio Task boundary raises
`ValueError: ... Token ... was created in a different Context` --
confirmed directly against real Starlette behavior during this phase's
own API-level testing (Starlette's `TestClient`/httpx `ASGITransport`
takes the `spec_version < (2, 4)` fallback path, which runs
`StreamingResponse.stream_response` in a separate `anyio` Task). The
lease alone has no such hazard; `telemetry.paid_action` must instead be
opened and closed from a single, uninterrupted execution context (see
`research_agent/curation_chat_streaming.py` and `research_agent/
services/curation_chat_service.py::stream_answer_curation_chat`'s own
docstrings for the production pattern this proves is safe).
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import research_agent.admission as admission
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.telemetry import init_usage_db
from research_agent.usage_guard import UsageGuardRejection, open_admission_and_lease_for_streaming
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


def test_rejected_admission_raises_before_any_lease_is_acquired(usage_db_path):
    from datetime import datetime, timezone

    from research_agent.config import get_usage_policy

    policy = get_usage_policy()
    _seed_paid_actions(
        usage_db_path, "session", "s1", policy.max_paid_actions_per_session_per_hour,
        datetime.now(timezone.utc).isoformat(),
    )
    with pytest.raises(UsageGuardRejection) as exc_info:
        open_admission_and_lease_for_streaming("curation_chat", subject=("session", "s1"), use_lease=True)
    assert exc_info.value.reason_code == "session_hourly_limit_reached"
    assert len(_rows(usage_db_path, "action_leases")) == 0


def test_successful_open_acquires_a_lease(usage_db_path):
    handle = open_admission_and_lease_for_streaming("curation_chat", subject=("session", "s2"), use_lease=True)
    assert len(_rows(usage_db_path, "action_leases")) == 1
    handle.release()
    assert len(_rows(usage_db_path, "action_leases")) == 0


def test_lease_held_until_release_then_freed(usage_db_path):
    handle = open_admission_and_lease_for_streaming("curation_chat", subject=("session", "s3"), use_lease=True)

    with pytest.raises(UsageGuardRejection) as exc_info:
        open_admission_and_lease_for_streaming("curation_chat", subject=("session", "s3"), use_lease=True)
    assert exc_info.value.reason_code == "action_in_progress"

    handle.release()

    handle2 = open_admission_and_lease_for_streaming("curation_chat", subject=("session", "s3"), use_lease=True)
    handle2.release()
    assert len(_rows(usage_db_path, "action_leases")) == 0


def test_release_is_idempotent(usage_db_path):
    handle = open_admission_and_lease_for_streaming("curation_chat", subject=("session", "s4"), use_lease=True)
    handle.release()
    handle.release()  # second call must be a no-op, not a double-release error
    assert len(_rows(usage_db_path, "action_leases")) == 0


def test_use_lease_false_acquires_no_lease_and_release_is_a_safe_no_op(usage_db_path):
    handle = open_admission_and_lease_for_streaming("curation_chat", subject=("session", "s5"), use_lease=False)
    assert len(_rows(usage_db_path, "action_leases")) == 0
    handle.release()  # must not raise
    assert len(_rows(usage_db_path, "action_leases")) == 0


def test_lease_handle_safely_crosses_a_real_task_boundary_unlike_telemetry_paid_action(usage_db_path):
    """The regression test for the bug this design was changed to avoid
    -- acquires the lease in one Task (mirroring the sync route
    handler), releases it from a DIFFERENT, separately-spawned Task
    (mirroring Starlette's own `spec_version < (2, 4)` fallback, which
    runs `StreamingResponse.stream_response` -- and therefore this
    project's own streaming generator -- in a separate `anyio`/asyncio
    Task from the one that returned the response object). Must not
    raise. `telemetry.paid_action`, opened and closed entirely WITHIN
    the spawned task's own body (the actual production pattern), must
    also correctly attach a child call and persist exactly one row --
    proving the recommended pattern is genuinely safe across this exact
    boundary, not just theoretically."""

    async def scenario():
        handle = open_admission_and_lease_for_streaming("curation_chat", subject=("session", "s6"), use_lease=True)

        async def body_in_a_different_task():
            with telemetry.paid_action("curation_chat", subject_type="session", subject_id="s6"):
                telemetry.record_child_call(call_type="generate_answer", provider="openai", latency_ms=1.0, outcome="success")
                await asyncio.sleep(0)
            handle.release()

        await asyncio.create_task(body_in_a_different_task())

    asyncio.run(scenario())

    rows = _rows(usage_db_path, "paid_actions")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "success"
    assert rows[0]["total_call_count"] == 1
    assert len(_rows(usage_db_path, "action_leases")) == 0


def test_real_usage_db_path_untouched():
    assert fingerprint_usage_db(_REAL_USAGE_DB_PATH) == _REAL_USAGE_DB_FINGERPRINT_BEFORE
