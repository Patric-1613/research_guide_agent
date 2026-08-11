"""Usage Protection M2.2A: the reusable production admission+lease guard.

Composes M2.1's admission checks (research_agent/admission.py) and
leases (research_agent/leases.py) with M1's paid_action(...) telemetry
(research_agent/telemetry.py) into ONE reusable context manager, so
routers/services enforce usage protection consistently instead of
hand-rolling budget/lease checks at each paid call site.

Guarded lifecycle:
1. Existing input/session validation -- happens BEFORE this guard opens.
   Every call site below only opens the guard once session_id/etc are
   already confirmed good (`if session is None: raise ServiceError(404,
   ...)` always runs first, exactly as it did before this guard existed).
2. Completion-based budget admission: per-session hourly, per-session
   daily, then the coarse global window -- always in that order.
3. Atomic lease acquisition (research_agent/leases.py), only when the
   caller asked for one via `use_lease=True`.
4. The existing `telemetry.paid_action(...)` context -- unchanged,
   still fail-open, still "first active action wins" nesting.
5. The caller's own `with` block body (the actual paid operation).
6. Lease release in `finally` (via `leases.session_lease`'s own
   already-tested release-on-success/exception/cancellation behavior).

Rejections are the typed `UsageGuardRejection` exception, not an
`AdmissionDecision`/`LeaseDecision` the caller has to check -- this lets
every router stay guard-agnostic. `research_agent/api_app/app.py`
registers ONE centralized FastAPI exception handler for it (see that
module) that maps it to 429/409/503, rather than every router adding
its own try/except for it the way `ServiceError` is handled today.

**Fail-closed, like admission.py/leases.py, unlike telemetry.py.** A
storage error anywhere in this guard's own admission/lease checks
raises `UsageGuardRejection("usage_protection_unavailable", ...)` --
"cannot confirm this is safe" is treated as "no". The `paid_action(...)`
telemetry this guard wraps is untouched and stays fail-open exactly as
M1 defined it: once past this guard, a telemetry write failure still
never breaks the real request.

**No child-call admission.** This guard is meant to wrap exactly ONE
top-level HTTP action per call site -- the same "first active action
wins" boundary `paid_action` itself already establishes. Nothing calls
this guard a second time for an internal/nested provider call; if a
guarded workflow fans out to multiple providers internally (e.g.
curation chat's nested web-offer search), that is still one admitted
user action with multiple child-call telemetry records, exactly as
before this guard existed.

**Rejected requests create no paid_actions row and make no provider
call.** Every check in steps 2-3 runs and can raise before step 4 ever
opens `paid_action(...)`, so a reject never reaches the operation this
guard wraps.

**Coarse global counter caveat.** `check_global_paid_action_window`
(research_agent/admission.py) counts completed, persisted rows -- like
every M2.1 admission check, it has no visibility into requests that are
still in flight, so under genuine cross-session concurrency right at
the edge of the global limit, more than `global_paid_action_limit`
requests can legitimately be admitted in the same window before any of
them complete and change the count. This is a deliberate, documented
limit of a completion-based counter in a single-process SQLite phase,
not a bug -- a real distributed rate limiter (token buckets, in-flight
reservation counting) is out of scope here. No IP address or other new
identity signal is introduced to work around it.
"""

from __future__ import annotations

import logging
import math
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from typing import Iterator, Literal

import research_agent.telemetry as telemetry
from research_agent.admission import (
    AdmissionDecision,
    check_global_paid_action_window,
    check_session_daily_budget,
    check_session_hourly_budget,
)
from research_agent.config import UsagePolicy, get_usage_policy
from research_agent.leases import LeaseDecision, session_lease

logger = logging.getLogger(__name__)

GuardReasonCode = Literal[
    "session_hourly_limit_reached",
    "session_daily_limit_reached",
    "global_window_limit_reached",
    "action_in_progress",
    "usage_protection_unavailable",
]

# HTTP status + a concise, user-safe message per reason code -- consulted
# only by research_agent/api_app/app.py's centralized exception handler
# for UsageGuardRejection (kept here, next to the reason codes themselves,
# so the two stay in sync). Messages are deliberately generic: no
# exception text, filesystem path, SQL, or any request content ever goes
# into an HTTP response for a rejection.
GUARD_REASON_HTTP_STATUS: dict[GuardReasonCode, int] = {
    "session_hourly_limit_reached": 429,
    "session_daily_limit_reached": 429,
    "global_window_limit_reached": 429,
    "action_in_progress": 409,
    "usage_protection_unavailable": 503,
}

GUARD_REASON_MESSAGE: dict[GuardReasonCode, str] = {
    "session_hourly_limit_reached": "Hourly usage limit reached for this session. Please try again later.",
    "session_daily_limit_reached": "Daily usage limit reached for this session. Please try again later.",
    "global_window_limit_reached": "The service is at capacity right now. Please try again shortly.",
    "action_in_progress": "Another request is already in progress for this session. Please wait for it to finish.",
    "usage_protection_unavailable": "Usage protection is temporarily unavailable. Please try again shortly.",
}


class UsageGuardRejection(Exception):
    """Raised by `guard_paid_action` when admission or the lease reject
    the request before any paid work starts. Carries only operational
    metadata (a reason code and, when meaningful, a positive
    Retry-After in seconds) -- never prompt/message/report/source text,
    matching `AdmissionDecision`/`LeaseDecision`'s own privacy posture."""

    def __init__(self, reason_code: GuardReasonCode, *, retry_after_seconds: int | None = None):
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.retry_after_seconds = retry_after_seconds


def _raise_if_rejected(decision: AdmissionDecision) -> None:
    if decision.allowed:
        return
    if decision.reason_code == "storage_unavailable":
        raise UsageGuardRejection("usage_protection_unavailable")
    raise UsageGuardRejection(decision.reason_code, retry_after_seconds=decision.retry_after_seconds)  # type: ignore[arg-type]


def _lease_retry_after(lease: LeaseDecision, *, now: datetime | None = None) -> int | None:
    if lease.expires_at is None:
        return None
    now = now if now is not None else datetime.now(timezone.utc)
    frees_at = datetime.fromisoformat(lease.expires_at)
    seconds = (frees_at - now).total_seconds()
    return max(1, math.ceil(seconds))


def check_admission(subject: tuple[str, str] | None, policy: UsagePolicy) -> None:
    """Steps 2 of the guarded lifecycle, split out so a caller that only
    needs the budget check (no telemetry/lease) -- none exist yet in
    M2.2A, but this keeps the composition seams honest -- can reuse it.
    Raises UsageGuardRejection; returns None when everything is allowed."""
    if subject is not None:
        subject_type, subject_id = subject
        _raise_if_rejected(check_session_hourly_budget(subject_type, subject_id, policy))
        _raise_if_rejected(check_session_daily_budget(subject_type, subject_id, policy))
    _raise_if_rejected(check_global_paid_action_window(policy))


@contextmanager
def guard_paid_action(
    action_type: str,
    *,
    subject: tuple[str, str] | None = None,
    use_lease: bool = False,
    discard_if_empty: bool = False,
) -> Iterator[None]:
    """Drop-in replacement for `with telemetry.paid_action(action_type,
    subject_type=..., subject_id=..., discard_if_empty=...):` at a
    top-level paid HTTP call site, adding admission + an optional lease
    in front of it.

    `subject`: `(subject_type, subject_id)`, or `None` for the small set
    of actions that genuinely have no stable identifier yet at the
    point they start (e.g. /search and /curation/start open their
    action before a search_id/session_id is minted) -- those receive
    ONLY the coarse global admission check, never a per-subject
    hourly/daily check or a lease (a lease needs a subject to scope
    to). Never pass an invented IP-based or otherwise synthetic
    identity here.

    `use_lease`: acquires the shared expensive-action lease for
    `subject` before opening `paid_action`, released in `finally`
    regardless of how the `with` block exits (success, exception, or
    cancellation) -- see `leases.session_lease`. Requires `subject`.

    Raises `UsageGuardRejection` before any paid work begins if
    admission or the lease reject the request.
    """
    policy = get_usage_policy()
    check_admission(subject, policy)

    with ExitStack() as stack:
        if use_lease:
            if subject is None:
                raise ValueError("guard_paid_action: use_lease=True requires a subject")
            subject_type, subject_id = subject
            lease_decision = stack.enter_context(
                session_lease(subject_type, subject_id, ttl_seconds=policy.expensive_action_lease_ttl_seconds)
            )
            if not lease_decision.acquired:
                if lease_decision.reason_code == "storage_unavailable":
                    raise UsageGuardRejection("usage_protection_unavailable")
                raise UsageGuardRejection("action_in_progress", retry_after_seconds=_lease_retry_after(lease_decision))

        subject_kwargs: dict[str, str] = {}
        if subject is not None:
            subject_kwargs = {"subject_type": subject[0], "subject_id": subject[1]}
        with telemetry.paid_action(action_type, discard_if_empty=discard_if_empty, **subject_kwargs):
            yield
