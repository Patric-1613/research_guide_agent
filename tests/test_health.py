"""Day 1 (public multi-user deployment foundation, see
docs/plans/public-multi-user-deployment-review.md), Part C: proves the
`GET /health` route contract is unchanged after converting its handler
from `def health()` to `async def health()`.

The behavioral contract this file protects (all pre-existing, unchanged
by the Day 1 signature change): the route path, the exact response body,
the 200 status code, the public-GET auth-gate exemption
(`research_agent/auth_middleware.py`'s allowlist is keyed on
method+path, not on whether the handler is sync or async), and that the
handler touches no dependency (no DB connection, no checkpointer, no
Chroma collection, no OpenAI client) -- confirmed here by inspecting the
route's own dependants rather than merely asserting a status code.

Isolation: `_client()` below is the same "fresh temp SQLite files +
mocked OpenAI/Chroma" pattern `tests/test_api.py`'s own `_client()`
fixture establishes -- `research_agent.api.app`'s real `lifespan()`
otherwise touches the real `data/history.sqlite`/
`data/usage_telemetry.sqlite`/`data/chroma_db/` unconditionally on
startup, which this file must never do.
"""

from __future__ import annotations

import inspect
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import research_agent.admission as admission
import research_agent.api as api
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.api_app.routers.health import health
from research_agent.storage import init_db as real_init_db


@contextmanager
def _client():
    """Same isolation contract as tests/test_api.py's own `_client()`:
    a fresh temp SQLite file for storage.py, redirected telemetry/
    admission/leases usage DBs, a mocked OpenAI client, and a mocked
    Chroma collection -- so nothing in this file ever opens a real data
    file."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.sqlite"
        usage_db_path = Path(tmp) / "usage_telemetry.sqlite"
        with patch.object(api, "init_db", lambda: real_init_db(db_path)), \
             patch.object(telemetry, "USAGE_DB_PATH", usage_db_path), \
             patch.object(admission, "USAGE_DB_PATH", usage_db_path), \
             patch.object(leases, "USAGE_DB_PATH", usage_db_path), \
             patch.object(api, "OpenAI", return_value=MagicMock()), \
             patch.object(api, "get_chroma_collection", return_value=MagicMock(name="fake_chroma_collection")) as chroma_spy:
            with TestClient(api.app) as client:
                yield client, chroma_spy


def test_health_handler_is_a_coroutine_function():
    """Guards the Day 1 change itself: a future accidental revert to a
    plain `def` would not be caught by any status/body assertion below
    (both sync and async handlers produce identical HTTP responses), so
    this is the one test that would actually catch a silent regression
    back to a threadpool-dispatched handler."""
    assert inspect.iscoroutinefunction(health)


def test_health_route_has_no_dependencies():
    """The handler declares zero `Depends(...)` parameters -- confirmed
    directly from its signature, not inferred from behavior. This is what
    makes `/health` structurally incapable of touching a DB connection,
    the curation checkpointer, or the Chroma collection: there is no
    dependency-injection channel through which one could reach it."""
    assert inspect.signature(health).parameters == {}


def test_health_returns_the_unchanged_contract():
    """Status code and response body, byte-for-byte the pre-Day-1
    contract."""
    with _client() as (client, _chroma_spy):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_is_still_public_with_no_credentials_supplied():
    """A GET to /health with no Authorization header at all must still
    succeed -- the one route the auth gate allowlists regardless of
    whether it's enabled. This is a same-process regression guard
    alongside tests/test_auth_middleware.py's own dedicated BasicAuth
    coverage (that file exercises the middleware's allowlist directly;
    this one exercises the real app end to end)."""
    with _client() as (client, _chroma_spy):
        response = client.get("/health", headers={})
    assert response.status_code == 200


def test_health_never_calls_the_chroma_dependency_beyond_startup():
    """A request-scoped proof, not just a signature inspection: the
    Chroma-collection factory is spied on; its call count does not change
    as a result of a `/health` request (it is called at most once already,
    during app-startup `lifespan()`, independent of any request -- this
    test proves `/health` adds no ADDITIONAL call on top of that fixed
    startup cost)."""
    with _client() as (client, chroma_spy):
        calls_before = chroma_spy.call_count
        response = client.get("/health")
        calls_after = chroma_spy.call_count
    assert response.status_code == 200
    assert calls_after == calls_before
