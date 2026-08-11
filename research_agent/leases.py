"""Usage Protection M2.1 Part C: atomic, opaque session-action leases.

A lease is a concurrency guard, not a rate limit -- it answers "is this
subject already running an expensive action right now?", independently
of `research_agent/admission.py`'s hourly/daily/global budget checks.
Nothing in this module is wired into any report/chat/API route yet;
that is M2.2.

**Why not paid_actions rows.** Per M1's own design, a `paid_actions` row
is written exactly once, at action *completion*
(`telemetry.py::_finalize_and_persist`, called from `paid_action`'s own
`finally` block) -- there is no in-flight row to use as a lock, by
design. This module adds a dedicated `action_leases` table (schema in
`telemetry.py::init_usage_db`) instead.

**Atomicity.** `acquire_lease` uses a single
`INSERT ... ON CONFLICT (subject_type, subject_id, action_group)
DO UPDATE ... WHERE action_leases.expires_at < excluded.acquired_at`
statement. SQLite serializes writers even under WAL (one writer at a
time, `busy_timeout` handles the wait), so two concurrent callers can
never both believe their conditional UPDATE fired: whichever writer's
statement actually executes first either creates the row (wins) or
finds a still-valid lease and its own `WHERE` clause evaluates to
false, so its write becomes a no-op (loses). The immediately-following
`SELECT` in the same connection then simply reads back whichever token
is now stored to determine which side that caller was on -- it is not
itself part of the atomicity guarantee, the UPSERT already is.

**Fail-closed**, same posture as `research_agent/admission.py` and for
the same reason: a lease exists to prevent overlapping expensive work,
so an inability to confirm lease state must be treated as "cannot
confirm this is safe", not "assume no lease exists".
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Literal

from research_agent.telemetry import USAGE_DB_PATH

logger = logging.getLogger(__name__)

# The one generic action group M2.1 defines. A future phase could add
# per-action-type groups (e.g. "report_generate" vs "search") if a
# session should be allowed to run one of each concurrently; M2.1 does
# not need that distinction yet, so everything expensive shares this
# group and a session may hold exactly one lease at a time.
EXPENSIVE_ACTION_GROUP = "expensive_action"

# Provisional: long enough to cover a slow paid action (multi-tool agent
# run, report generation), short enough that a crashed/killed worker
# doesn't block a session for long. Not derived from real latency
# telemetry yet.
DEFAULT_LEASE_TTL_SECONDS = 300

ReasonCode = Literal["ok", "lease_held", "storage_unavailable"]


@dataclass(frozen=True)
class LeaseDecision:
    """Structured result, not an exception -- an ordinary "someone else
    already holds this lease" rejection is expected, routine behavior,
    not an error. `token` is the opaque credential the holder must
    present to release the lease; it is None whenever `acquired` is
    False."""

    acquired: bool
    reason_code: ReasonCode
    token: str | None = None
    expires_at: str | None = None


def _connect(path: Path | None) -> sqlite3.Connection:
    resolved = path if path is not None else USAGE_DB_PATH
    conn = sqlite3.connect(resolved)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def acquire_lease(
    subject_type: str,
    subject_id: str,
    action_group: str = EXPENSIVE_ACTION_GROUP,
    *,
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    path: Path | None = None,
) -> LeaseDecision:
    token = uuid.uuid4().hex
    lease_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    acquired_at = now.isoformat()
    expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()

    # sqlite3.connect() (inside _connect) can raise sqlite3.OperationalError
    # (not just OSError) for an unreachable path -- e.g. a missing parent
    # directory -- so connect and the query below share one fail-closed net.
    conn = None
    try:
        conn = _connect(path)
        conn.execute(
            """
            INSERT INTO action_leases
                (lease_id, subject_type, subject_id, action_group, lease_token, acquired_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (subject_type, subject_id, action_group) DO UPDATE SET
                lease_id = excluded.lease_id,
                lease_token = excluded.lease_token,
                acquired_at = excluded.acquired_at,
                expires_at = excluded.expires_at
            WHERE action_leases.expires_at < excluded.acquired_at
            """,
            (lease_id, subject_type, subject_id, action_group, token, acquired_at, expires_at),
        )
        conn.commit()
        row = conn.execute(
            "SELECT lease_token FROM action_leases "
            "WHERE subject_type = ? AND subject_id = ? AND action_group = ?",
            (subject_type, subject_id, action_group),
        ).fetchone()
    except (sqlite3.Error, OSError):
        logger.error("leases: acquisition failed", exc_info=True)
        return LeaseDecision(acquired=False, reason_code="storage_unavailable")
    finally:
        if conn is not None:
            conn.close()

    if row is not None and row[0] == token:
        return LeaseDecision(acquired=True, reason_code="ok", token=token, expires_at=expires_at)
    return LeaseDecision(acquired=False, reason_code="lease_held")


def release_lease(
    subject_type: str,
    subject_id: str,
    action_group: str,
    token: str,
    *,
    path: Path | None = None,
) -> bool:
    """Idempotent and token-gated: deletes the row only if it still
    holds this exact token, so releasing an already-released lease, or
    presenting the wrong token, is a safe no-op rather than an error.
    Returns False only on a genuine storage error (logged); the caller
    cannot distinguish "already released" from "just released" by
    design, since both are the same safe outcome."""
    conn = None
    try:
        conn = _connect(path)
        conn.execute(
            "DELETE FROM action_leases "
            "WHERE subject_type = ? AND subject_id = ? AND action_group = ? AND lease_token = ?",
            (subject_type, subject_id, action_group, token),
        )
        conn.commit()
        return True
    except (sqlite3.Error, OSError):
        logger.error("leases: release failed", exc_info=True)
        return False
    finally:
        if conn is not None:
            conn.close()


@contextmanager
def session_lease(
    subject_type: str,
    subject_id: str,
    action_group: str = EXPENSIVE_ACTION_GROUP,
    *,
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    path: Path | None = None,
) -> Iterator[LeaseDecision]:
    """Acquires (or fails to acquire) a lease and yields the resulting
    `LeaseDecision` -- it does not raise when acquisition is rejected;
    the caller inspects `.acquired` and decides what to do, same as
    `research_agent/admission.py`'s decisions. Releases in `finally`
    whenever this call actually won the lease, on both the normal and
    the exception exit path, so a crashed caller still frees the
    session for the next attempt (bounded, worst case, by the lease's
    own `ttl_seconds` expiry even if release itself fails)."""
    decision = acquire_lease(subject_type, subject_id, action_group, ttl_seconds=ttl_seconds, path=path)
    try:
        yield decision
    finally:
        if decision.acquired and decision.token is not None:
            release_lease(subject_type, subject_id, action_group, decision.token, path=path)
