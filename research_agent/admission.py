"""Usage Protection M2.1: admission decisions for paid-action budgets.

This module answers one question -- "is this subject allowed to start
another paid action right now?" -- against three independent windows:
a per-session hourly budget, a per-session daily budget, and a coarse
global window shared by every subject. It does not enforce anything by
itself; nothing in this module is wired into a route yet (M2.2 will do
that). It only provides a deterministic, typed decision a caller can
act on.

**Fail-closed, unlike telemetry.py.** `research_agent/telemetry.py` is
deliberately fail-open: a telemetry write failure must never break the
real request it's describing. Admission is the opposite kind of code --
its entire job is to say no under specific conditions -- so a storage
error here (a locked database, a missing file, a permissions problem)
is treated as "cannot confirm this is safe to allow" and rejected with
`reason_code="storage_unavailable"`, not silently treated as "allow".
Only genuine storage errors (`sqlite3.Error`, `OSError`) are caught this
way; a programming bug (e.g. a bad argument) still raises normally
rather than being masked as a rejection.

**Counting rule.** All three checks count *persisted* `paid_actions`
rows with `started_at` inside the relevant window, regardless of
`outcome` (`success`, `error`, or `cancelled` all count). This matches
what the task means by "paid-action attempts": an action that started
and was later persisted as `error`/`cancelled` still consumed a model
call and counts against the budget, exactly like a `success` one. Per
M1's own design, `paid_actions` rows are written only once, at action
completion (see `telemetry.py::_finalize_and_persist`) -- there is no
"in-flight" row to count or exclude, so this rule has no ambiguity
about partially-written actions.

**No private content.** Every function here takes only the existing
opaque `subject_type`/`subject_id` identifiers (already established by
M1 telemetry) plus a `UsagePolicy`. Nothing here accepts or persists
prompts, messages, report text, URLs, headers, API keys, or exception
text -- an `AdmissionDecision` carries only operational metadata.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from research_agent.config import UsagePolicy
from research_agent.telemetry import USAGE_DB_PATH

logger = logging.getLogger(__name__)

ReasonCode = Literal[
    "ok",
    "session_hourly_limit_reached",
    "session_daily_limit_reached",
    "global_window_limit_reached",
    "storage_unavailable",
]


@dataclass(frozen=True)
class AdmissionDecision:
    """Operational metadata only -- see module docstring. `retry_after_seconds`
    is populated only when `allowed` is False and a meaningful retry time
    could be computed; it is always a positive integer when present."""

    allowed: bool
    reason_code: ReasonCode
    retry_after_seconds: int | None = None
    limit: int | None = None
    current_count: int | None = None


def _retry_after_seconds(oldest_started_at: str, window: timedelta, now: datetime) -> int:
    """Retry-After is derived from the oldest counted action in the
    window: once it ages out, the sliding window's count drops by one
    and (absent new activity) the caller is admitted again. Rounded up
    (never down) and floored at 1 second, so a caller is never told to
    retry at a moment that would still be rejected."""
    oldest = datetime.fromisoformat(oldest_started_at)
    frees_at = oldest + window
    seconds = (frees_at - now).total_seconds()
    return max(1, math.ceil(seconds))


def _evaluate(
    *, query: str, params: tuple, limit: int, window: timedelta,
    reason_code: ReasonCode, now: datetime, path: Path | None,
) -> AdmissionDecision:
    resolved = path if path is not None else USAGE_DB_PATH
    # sqlite3.connect() itself can raise sqlite3.OperationalError (not just
    # OSError) for an unreachable path -- e.g. a missing parent directory --
    # so both connect() and the query below share one fail-closed net.
    conn = None
    try:
        conn = sqlite3.connect(resolved)
        conn.execute("PRAGMA busy_timeout=5000")
        count, oldest_started_at = conn.execute(query, params).fetchone()
    except (sqlite3.Error, OSError):
        logger.error("admission: budget query failed", exc_info=True)
        return AdmissionDecision(allowed=False, reason_code="storage_unavailable")
    finally:
        if conn is not None:
            conn.close()

    count = count or 0
    if count < limit:
        return AdmissionDecision(allowed=True, reason_code="ok", limit=limit, current_count=count)

    retry_after = (
        _retry_after_seconds(oldest_started_at, window, now)
        if oldest_started_at is not None
        else int(window.total_seconds())
    )
    return AdmissionDecision(
        allowed=False, reason_code=reason_code, retry_after_seconds=retry_after,
        limit=limit, current_count=count,
    )


def _window_check(
    subject_type: str, subject_id: str, *, limit: int, window: timedelta,
    reason_code: ReasonCode, now: datetime | None, path: Path | None,
) -> AdmissionDecision:
    now = now if now is not None else datetime.now(timezone.utc)
    cutoff = (now - window).isoformat()
    return _evaluate(
        query=(
            "SELECT COUNT(*), MIN(started_at) FROM paid_actions "
            "WHERE subject_type = ? AND subject_id = ? AND started_at >= ?"
        ),
        params=(subject_type, subject_id, cutoff),
        limit=limit, window=window, reason_code=reason_code, now=now, path=path,
    )


def check_session_hourly_budget(
    subject_type: str, subject_id: str, policy: UsagePolicy, *,
    now: datetime | None = None, path: Path | None = None,
) -> AdmissionDecision:
    """Sliding one-hour window, scoped to a single subject."""
    return _window_check(
        subject_type, subject_id,
        limit=policy.max_paid_actions_per_session_per_hour, window=timedelta(hours=1),
        reason_code="session_hourly_limit_reached", now=now, path=path,
    )


def check_session_daily_budget(
    subject_type: str, subject_id: str, policy: UsagePolicy, *,
    now: datetime | None = None, path: Path | None = None,
) -> AdmissionDecision:
    """Sliding 24-hour window, scoped to a single subject."""
    return _window_check(
        subject_type, subject_id,
        limit=policy.max_paid_actions_per_session_per_day, window=timedelta(days=1),
        reason_code="session_daily_limit_reached", now=now, path=path,
    )


def check_global_paid_action_window(
    policy: UsagePolicy, *, now: datetime | None = None, path: Path | None = None,
) -> AdmissionDecision:
    """Coarse window shared across every subject -- a blunt, global
    circuit breaker independent of any one session's own budget."""
    now = now if now is not None else datetime.now(timezone.utc)
    window = timedelta(minutes=policy.global_paid_action_window_minutes)
    cutoff = (now - window).isoformat()
    return _evaluate(
        query="SELECT COUNT(*), MIN(started_at) FROM paid_actions WHERE started_at >= ?",
        params=(cutoff,),
        limit=policy.global_paid_action_limit, window=window,
        reason_code="global_window_limit_reached", now=now, path=path,
    )
