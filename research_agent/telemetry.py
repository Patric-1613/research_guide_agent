"""Usage Protection M1.1: privacy-safe production usage telemetry
foundation -- observe-only, two layers, neither wired into any
domain/service call site yet (that is M1.2's job).

**HTTP request context** (`RequestTelemetryMiddleware`): a pure ASGI
middleware -- not `BaseHTTPMiddleware`, deliberately, so a future
streaming response is never buffered or altered on its way through
this layer, and so `contextvars` propagation stays exactly as
predictable as every other ASGI middleware in this app (CORS included).
Records one `http_requests` row per HTTP request: route TEMPLATE (never
the raw path -- see `_route_template`), method, status, outcome,
timestamps, latency. Stamps `X-Request-ID` on the real response when
one was actually sent.

**Semantic paid-action context** (`paid_action`/`record_child_call`):
a `contextvars.ContextVar`-based collector, the same mechanism
`research_agent/ingestion.py`'s own `_rate_limit_tracker` already
proves works across this project's synchronous FastAPI/service call
chains (see that module's own docstring for why a plain ContextVar,
not a Langfuse mechanism, is the right tool here too -- the Langfuse
SDK has no public way to update an ancestor span from a descendant
after the fact, and this project needs exactly that "roll nested work
up into one top-level record" shape). `paid_action(...)` opens (or, if
one is already active, simply attaches to) one action's collector;
`record_child_call(...)` appends a typed, content-free child-call
record to whichever action is currently active, and is a no-op when
none is. Nothing in this module ever inspects "is this an eval/CLI
invocation" -- eval/CLI code (confirmed to call `research_agent.report`/
`research_agent.qa` functions directly, e.g. `research_agent/evals/
r6d4_capture.py`) never passes through `RequestTelemetryMiddleware`,
so it never has an active action context, so `record_child_call`
there is structurally a no-op. The exclusion is a property of how
context gets established, not a maintained allowlist anywhere in this
file.

**Fail-open, everywhere.** Every persistence attempt (HTTP row, action
row) is wrapped so a telemetry failure is logged and swallowed --
never allowed to change the real request/action's own outcome or
propagate to its caller.

**Privacy**: the typed shapes below have no field for a prompt,
message, report/chat text, abstract, URL, query string, header value,
API key, or raw exception message (`error_type` is a class name only,
e.g. `"OpenAIError"`, never `str(exc)`). This is enforced structurally
by every public function here taking fixed, named parameters -- no
`**kwargs`, so there is no channel through which an arbitrary field
could ever reach a persisted row.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

logger = logging.getLogger(__name__)

USAGE_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "usage_telemetry.sqlite"

# Frozen per the M1 architecture audit -- a compact, user-visible-operation
# enum, never one action type per function. Not enforced at runtime here
# (see this module's own docstring on why validation stays out of the
# fail-open write path) -- exposed for M1.2's own instrumentation to import
# rather than re-typing these strings.
ACTION_TYPES = (
    "search", "summarize", "search_chat", "curation_start", "curation_refill",
    "curation_chat", "report_generate", "report_regenerate",
)

HttpOutcome = Literal["success", "error", "cancelled"]
ActionOutcome = Literal["success", "error", "cancelled", "rejected"]  # "rejected" is reserved for a later rate-limiting phase; nothing in M1 ever sets it.
SubjectType = Literal["session", "search"]

_request_id_var: ContextVar[str | None] = ContextVar("_request_id_var", default=None)
_action_var: ContextVar["_ActionCollector | None"] = ContextVar("_action_var", default=None)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Schema + persistence --------------------------------------------------

def init_usage_db(path: Path | None = None) -> sqlite3.Connection:
    """Creates the parent directory and both tables/indexes if they
    don't already exist, and returns an open connection (caller closes
    it) -- same convention as `storage.py::init_db`/`embeddings.py::
    _init_cache_db`. Called once at FastAPI startup (`lifespan()`).

    `path` defaults to `None`, resolved to the CURRENT value of
    `USAGE_DB_PATH` *inside this function body* -- deliberately not a
    bound default parameter (`path: Path = USAGE_DB_PATH`), which would
    capture `USAGE_DB_PATH`'s value once at function-definition time
    and never see a later `monkeypatch.setattr(telemetry, "USAGE_DB_PATH",
    ...)`. Every path-accepting function in this module follows this
    same resolve-inside-the-body pattern for exactly this reason --
    tests redirect `USAGE_DB_PATH` and every call site (including ones
    deep inside the middleware/`paid_action`, which have no FastAPI
    `Depends()` seam to override) picks it up correctly.
    """
    resolved = path if path is not None else USAGE_DB_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved)
    # Same WAL + busy_timeout convention as storage.py::init_db -- every
    # request writes here, so overlapping writers under FastAPI's
    # multi-threaded request handling are the normal case, not an edge one.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS http_requests (
            request_id TEXT PRIMARY KEY,
            route_template TEXT,
            method TEXT,
            status_code INTEGER,
            outcome TEXT,
            started_at TEXT,
            ended_at TEXT,
            latency_ms REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paid_actions (
            action_id TEXT PRIMARY KEY,
            action_type TEXT,
            request_id TEXT,
            subject_type TEXT,
            subject_id TEXT,
            outcome TEXT,
            started_at TEXT,
            ended_at TEXT,
            latency_ms REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            total_call_count INTEGER,
            child_calls_json TEXT
        )
        """
    )
    # request_id is a plain indexed correlation field, never a foreign key --
    # an action can finish and persist before the outer middleware writes its
    # own http_requests row (the action's own try/finally runs first, nested
    # inside the request), so a FK would be a real, expected-to-fire
    # constraint violation, not a safety net.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_http_requests_started_at ON http_requests (started_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paid_actions_request_id ON paid_actions (request_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paid_actions_action_type ON paid_actions (action_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paid_actions_started_at ON paid_actions (started_at)")
    conn.commit()
    return conn


def _write_http_request(
    *, request_id: str, route_template: str, method: str, status_code: int | None,
    outcome: HttpOutcome, started_at: str, ended_at: str, latency_ms: float,
    path: Path | None = None,
) -> None:
    """One short-lived connection per write -- opened, used, closed --
    never a shared/cached connection, so a write here is never blocked
    behind (or blocking) some other long-lived connection elsewhere in
    the app."""
    resolved = path if path is not None else USAGE_DB_PATH
    conn = sqlite3.connect(resolved)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(
            "INSERT INTO http_requests "
            "(request_id, route_template, method, status_code, outcome, started_at, ended_at, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (request_id, route_template, method, status_code, outcome, started_at, ended_at, latency_ms),
        )
        conn.commit()
    finally:
        conn.close()


def _write_paid_action(
    *, action_id: str, action_type: str, request_id: str | None, subject_type: str | None,
    subject_id: str | None, outcome: ActionOutcome, started_at: str, ended_at: str, latency_ms: float,
    input_tokens: int | None, output_tokens: int | None, total_tokens: int | None,
    total_call_count: int, child_calls_json: str, path: Path | None = None,
) -> None:
    resolved = path if path is not None else USAGE_DB_PATH
    conn = sqlite3.connect(resolved)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(
            "INSERT INTO paid_actions "
            "(action_id, action_type, request_id, subject_type, subject_id, outcome, started_at, ended_at, "
            "latency_ms, input_tokens, output_tokens, total_tokens, total_call_count, child_calls_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                action_id, action_type, request_id, subject_type, subject_id, outcome, started_at, ended_at,
                latency_ms, input_tokens, output_tokens, total_tokens, total_call_count, child_calls_json,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _safe(write_fn, /, *args: Any, **kwargs: Any) -> None:
    """Fail-open wrapper: any exception from a write attempt (a locked
    file, a permissions error, a serialization bug) is logged and
    swallowed here -- the one and only place in this module allowed to
    catch a bare `Exception` without re-raising, since telemetry must
    never be able to fail the real request/action it's describing."""
    try:
        write_fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 -- deliberate: telemetry failures are observe-only, never propagated.
        logger.warning("telemetry: failed to persist a %s record", getattr(write_fn, "__name__", "?"), exc_info=True)


# --- Request context ---------------------------------------------------

def get_current_request_id() -> str | None:
    """The request_id set by `RequestTelemetryMiddleware` for whichever
    HTTP request is currently being handled on this task/thread, or
    `None` outside any request (eval/CLI code, a raw script, a test
    that never entered the middleware)."""
    return _request_id_var.get()


# --- Semantic paid-action context ---------------------------------------

@dataclass
class _ActionCollector:
    """Mutable, in-memory only -- never itself serialized wholesale;
    `_finalize_and_persist` below reads it once at the end to build the
    actual row. `child_calls` entries are already the exact dicts that
    go into `child_calls_json` (typed, content-free -- see
    `record_child_call`)."""

    action_id: str
    action_type: str
    request_id: str | None
    subject_type: str | None
    subject_id: str | None
    started_monotonic: float
    started_at: str
    child_calls: list[dict[str, Any]] = field(default_factory=list)


def _sum_nullable(values: list[int | None]) -> int | None:
    """Sums the non-`None` entries; returns `None` (never `0`) if every
    entry is `None` -- a provider that reports no usage information at
    all must never look identical to "billed exactly zero tokens"."""
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def _finalize_and_persist(collector: "_ActionCollector", outcome: ActionOutcome) -> None:
    def _build_and_write() -> None:
        # Deliberately INSIDE the closure _safe() below wraps -- not run
        # eagerly before it -- so a JSON-serialization failure (a future
        # M1.2 call site accidentally passing something json.dumps can't
        # handle) is just as fail-open as a raw SQL write failure. This
        # closure raising for ANY reason must never propagate past _safe().
        ended_at = _utcnow_iso()
        latency_ms = (time.monotonic() - collector.started_monotonic) * 1000
        input_tokens = _sum_nullable([c["input_tokens"] for c in collector.child_calls])
        output_tokens = _sum_nullable([c["output_tokens"] for c in collector.child_calls])
        total_tokens = _sum_nullable([c["total_tokens"] for c in collector.child_calls])
        # sort_keys=True: deterministic key order within each child-call dict;
        # list ORDER (call order) is preserved as-is, since that's real
        # information (which call happened first), not something to normalize away.
        child_calls_json = json.dumps(collector.child_calls, sort_keys=True)
        _write_paid_action(
            action_id=collector.action_id, action_type=collector.action_type,
            request_id=collector.request_id, subject_type=collector.subject_type,
            subject_id=collector.subject_id, outcome=outcome,
            started_at=collector.started_at, ended_at=ended_at, latency_ms=latency_ms,
            input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens,
            total_call_count=len(collector.child_calls), child_calls_json=child_calls_json,
        )

    _safe(_build_and_write)


@contextmanager
def paid_action(
    action_type: str, *, subject_type: SubjectType | None = None, subject_id: str | None = None,
) -> Iterator[None]:
    """Opens one semantic paid-action record for the duration of the
    `with` block, correlated to whatever HTTP request is currently
    active (`get_current_request_id()`) if any.

    **First active action wins.** If a `paid_action` is already open on
    this task/thread (a nested call -- e.g. a service function that
    calls another service function which also opens one), this call
    does NOT open a second top-level row: it's a pure pass-through,
    attaching nothing of its own, persisting nothing on its own exit.
    Everything nested underneath -- however many `paid_action`/`record_
    child_call` calls happen inside -- becomes child-call records on
    the ONE outermost action. This is deliberate, not a missing
    feature: Part 4 of the M1 architecture audit is explicit that
    action_type is a small, user-visible-operation enum, and a nested
    `_accept_web_offer`-style call making a second internal `ask_in_
    session` call must never be double-counted as a second `search_
    chat` action.

    Persists exactly once, on the way out of the OUTERMOST call only --
    on success, on an exception (any `Exception` subclass), or on
    `asyncio.CancelledError` -- and always re-raises whatever it caught,
    unchanged (no `raise ... from`, no exception substitution). A
    persistence or JSON-serialization failure at that point is logged
    and swallowed (`_safe`/`_finalize_and_persist`) -- it can never
    surface as, or replace, the real exception from the `with` block.
    """
    existing = _action_var.get()
    if existing is not None:
        yield
        return

    collector = _ActionCollector(
        action_id=uuid.uuid4().hex, action_type=action_type,
        request_id=get_current_request_id(), subject_type=subject_type, subject_id=subject_id,
        started_monotonic=time.monotonic(), started_at=_utcnow_iso(),
    )
    token = _action_var.set(collector)
    outcome: ActionOutcome = "success"
    try:
        yield
    except asyncio.CancelledError:
        # CancelledError is a BaseException (not Exception) as of Python
        # 3.8+ -- that's what makes this branch distinguishable from the
        # except Exception branch below at all, not an arbitrary choice.
        outcome = "cancelled"
        raise
    except Exception:
        outcome = "error"
        raise
    finally:
        _action_var.reset(token)
        _finalize_and_persist(collector, outcome)


def record_child_call(
    call_type: str, provider: str, *, model: str | None = None,
    input_tokens: int | None = None, output_tokens: int | None = None, total_tokens: int | None = None,
    cache_hit: bool | None = None, latency_ms: float, outcome: str, error_type: str | None = None,
) -> None:
    """Appends one typed, content-free child-call record to whichever
    `paid_action` is currently active. **No-op (returns immediately,
    never raises) when no action is active** -- this is what makes an
    eval/CLI invocation of a shared domain function (e.g. `research_
    agent.evals.r6d4_capture` calling `research_agent.report.generate_
    report_for_session` directly, confirmed by the M1 audit to happen
    for real in this codebase) silently record nothing: there is no
    active `RequestTelemetryMiddleware`-established action context to
    attach to, and this function never tries to invent one.

    `error_type` must be an exception CLASS NAME only (e.g.
    `"OpenAIError"`) -- never `str(exc)` or any other exception detail;
    there is no field here for one. Every parameter is fixed and named
    (no `**kwargs`) so there is no channel through which a prompt,
    message, URL, or other forbidden field could ever reach a
    persisted row via this function.
    """
    collector = _action_var.get()
    if collector is None:
        return
    collector.child_calls.append({
        "call_type": call_type, "provider": provider, "model": model,
        "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens,
        "cache_hit": cache_hit, "latency_ms": latency_ms, "outcome": outcome, "error_type": error_type,
    })


# --- HTTP request context: pure ASGI middleware -----------------------

def _route_template(scope: dict[str, Any]) -> str:
    """Reads the path TEMPLATE (e.g. "/curation/{session_id}/report"),
    never the raw resolved path (which would contain a real session/
    search id). FastAPI's own `APIRoute.matches` (confirmed directly
    against the installed fastapi==0.139.0: `child_scope["route"] =
    self`) stamps the matched route object onto `scope["route"]` during
    routing, downstream of this middleware -- since `scope` is the same
    mutable dict threaded through the whole ASGI call chain, that
    mutation is visible here once `self.app(...)` returns. `"route"`
    is absent (never matched -- a 404, or routing never got far enough
    to set it) or its `.path` is missing/empty in some edge case ->
    `"<unresolved>"`, confirmed empirically (a genuinely unmatched path
    leaves `scope.get("route")` as `None`)."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) and path else "<unresolved>"


class RequestTelemetryMiddleware:
    """Pure ASGI middleware -- a plain `(app) -> ASGI callable` class,
    deliberately NOT `starlette.middleware.base.BaseHTTPMiddleware`.
    `BaseHTTPMiddleware` fully buffers a streamed response into an
    in-memory iterator before this app ever sees it and runs the
    downstream app in a separate task (its own well-known `contextvars`
    propagation caveats) -- both are exactly wrong for a foundation
    meant to sit underneath a future streaming endpoint without ever
    buffering or breaking it. This class instead wraps `send` directly:
    every message (`http.response.start`, each `http.response.body`
    chunk) passes through unmodified except the one `http.response.
    start` message, which gets one extra response header appended.

    Records one `http_requests` row per real HTTP request (`scope["type"]
    == "http"` only -- `"lifespan"`/`"websocket"` scopes pass straight
    through, untouched). Never records or persists the query string,
    request/response bodies, or any header value.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        method = scope.get("method", "")
        started_monotonic = time.monotonic()
        started_at = _utcnow_iso()
        status_code: int | None = None
        response_started = False

        token = _request_id_var.set(request_id)

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                # Appended, never replacing whatever headers the app already
                # set (CORS's own headers included, regardless of which
                # middleware layer runs closer to the network boundary).
                message["headers"] = [*message.get("headers", []), (b"x-request-id", request_id.encode("latin-1"))]
            # Every message -- start AND every body chunk -- passes through
            # exactly as given otherwise: no buffering, no re-chunking, no
            # inspection of body content.
            await send(message)

        outcome: HttpOutcome = "success"
        try:
            await self.app(scope, receive, send_wrapper)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception:
            outcome = "error"
            # No http.response.start ever passed through send_wrapper in this
            # branch's common case (an unhandled exception that reaches this
            # middleware has, by construction, already escaped Starlette's own
            # ExceptionMiddleware without producing a response) -- there is no
            # real status code to report, so this records the status the
            # outer ServerErrorMiddleware will actually send. If a response
            # HAD already started (a failure mid-stream, after headers were
            # sent), the real status it used is kept instead of being
            # overwritten.
            if not response_started:
                status_code = 500
            raise
        finally:
            _request_id_var.reset(token)
            ended_at = _utcnow_iso()
            latency_ms = (time.monotonic() - started_monotonic) * 1000
            _safe(
                _write_http_request,
                request_id=request_id, route_template=_route_template(scope), method=method,
                status_code=status_code, outcome=outcome,
                started_at=started_at, ended_at=ended_at, latency_ms=latency_ms,
            )
