"""Day 1 (public multi-user deployment foundation, see
docs/plans/public-multi-user-deployment-review.md), Part D/Part E:
deterministic proof of the explicit AnyIO thread-pool ceiling
(`research_agent/config/settings.py`'s `get_thread_pool_config()`,
applied once in `research_agent/api_app/app.py`'s `lifespan()`).

Covers, one test function per item, using events/barriers rather than
timing-only assertions wherever a genuine concurrency claim is being
proven:

1. `/health` remains responsive while blocking-worker capacity is
   occupied.
2. The configured AnyIO limit is applied during application lifespan.
3. Invalid configuration fails clearly at startup.
4. Existing authentication behavior for `/health` is unchanged.
5. No provider, database, or Chroma operation is invoked by `/health`
   -- already proven independently in `tests/test_health.py`
   (`test_health_route_has_no_dependencies`,
   `test_health_never_calls_the_chroma_dependency_beyond_startup`); not
   duplicated here.
6. Application shutdown remains clean.

No other route is converted to `async def` in this checkpoint -- item 1's
synthetic `/block` route below exists ONLY to occupy AnyIO worker-thread
capacity deterministically; it is not part of the real application.

Isolation: any test here that issues a REAL HTTP request through a
`create_app()`-built app (items 3's integration variant, 4, 6) goes
through `_isolated_app()`, which redirects `telemetry.USAGE_DB_PATH`/
`admission.USAGE_DB_PATH`/`leases.USAGE_DB_PATH` to a temp file in
addition to mocking `api.init_db`/`telemetry.init_usage_db`/`api.OpenAI`/
`api.get_chroma_collection` -- `RequestTelemetryMiddleware` writes one
real `http_requests` row per request regardless of route or auth outcome,
so any real request through an unpatched `create_app()` app touches the
real `data/usage_telemetry.sqlite`, exactly as `tests/test_usage_guard.py`
and `tests/test_usage_guard_streaming.py`'s own
`test_real_usage_db_path_untouched` regression tests exist to catch.
"""

from __future__ import annotations

import os
import tempfile
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import research_agent.admission as admission
import research_agent.api as api
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.api_app.app import create_app, lifespan
from research_agent.api_app.routers.health import router as health_router
from research_agent.config.settings import ThreadPoolConfig, get_thread_pool_config


@contextmanager
def _isolated_app(env: dict[str, str]):
    """Builds a fresh app via create_app() under the given env, with
    every real-data boundary lifespan()/RequestTelemetryMiddleware can
    reach redirected to temp files -- the same isolation contract
    tests/test_auth_middleware.py's own `_client_with_env` establishes,
    reused here rather than re-derived, since this file makes real HTTP
    requests against real create_app() apps for items 3/4/6."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.sqlite"
        usage_db_path = Path(tmp) / "usage_telemetry.sqlite"
        with patch.dict(os.environ, env), \
             patch.object(api, "init_db", return_value=MagicMock()), \
             patch.object(telemetry, "init_usage_db", return_value=MagicMock()), \
             patch.object(telemetry, "USAGE_DB_PATH", usage_db_path), \
             patch.object(admission, "USAGE_DB_PATH", usage_db_path), \
             patch.object(leases, "USAGE_DB_PATH", usage_db_path), \
             patch.object(api, "OpenAI", return_value=MagicMock()), \
             patch("research_agent.api_app.app.default_async_openai_client", return_value=MagicMock()), \
             patch.object(api, "get_chroma_collection", return_value=MagicMock(name="fake_chroma_collection")):
            app = create_app()
            yield app


def _patched_lifespan_dependencies():
    """The lifespan()-level dependencies used by tests below that never
    issue a real HTTP request (so RequestTelemetryMiddleware never fires
    and the real usage DB is never at risk) -- direct patches, not the
    full `_isolated_app()` temp-file isolation, since no request means
    no telemetry write to redirect."""
    return (
        patch.object(api, "init_db", return_value=MagicMock()),
        patch.object(telemetry, "init_usage_db", return_value=MagicMock()),
        patch.object(api, "OpenAI", return_value=MagicMock()),
        patch.object(api, "get_chroma_collection", return_value=MagicMock(name="fake_chroma_collection")),
    )


# --- Item 1: /health stays responsive while worker-thread capacity is fully occupied ---

def test_health_not_starved_when_thread_pool_capacity_is_saturated():
    """A minimal, isolated ASGI app (the REAL /health router, plus one
    synthetic synchronous /block route that exists only for this test)
    with the ceiling set to exactly 2 -- small on purpose, so saturating
    it needs only 2 concurrent blocked requests, not 40. Deliberately not
    built via create_app() (no RequestTelemetryMiddleware, no real-DB
    surface at all) -- this app never touches any real data file.

    Determinism, no sleeping: a `threading.Barrier(3)` (2 blocker threads
    + this test's own main thread) only releases once BOTH blocker
    requests are genuinely, simultaneously executing inside the handler
    -- proving the 2-token ceiling is fully saturated by real concurrent
    execution, not inferred from a delay. Only after that proof does the
    test issue the /health request; the blockers are held open by a
    `threading.Event` this test controls and never sets until AFTER the
    /health assertion has already succeeded -- so a regression that made
    /health queue behind the sync-route ceiling would hang this test
    (caught by a bounded `Thread.join(timeout=...)` safety net below, an
    explicit deadlock guard, never the pass condition itself), not
    silently pass.
    """
    release_event = threading.Event()
    saturation_barrier = threading.Barrier(3)

    @asynccontextmanager
    async def minimal_lifespan(app: FastAPI):
        # current_default_thread_limiter() must be called from WITHIN the
        # event loop it applies to -- TestClient runs this app's loop on
        # its own background thread, so this must happen in this app's
        # own lifespan (exactly where the real api_app/app.py applies it
        # too), never from the test's calling thread after the fact.
        anyio.to_thread.current_default_thread_limiter().total_tokens = get_thread_pool_config().limit
        yield

    app = FastAPI(lifespan=minimal_lifespan)
    app.include_router(health_router)

    @app.get("/block")
    def block() -> dict:
        saturation_barrier.wait(timeout=10)
        release_event.wait(timeout=10)
        return {"blocked": True}

    results: dict[int, int] = {}

    def call_block(index: int, client: TestClient) -> None:
        response = client.get("/block")
        results[index] = response.status_code

    health_result: dict[str, int] = {}

    def call_health(client: TestClient) -> None:
        response = client.get("/health")
        health_result["status_code"] = response.status_code

    with patch.dict(os.environ, {"ANYIO_THREAD_LIMIT": "2"}):
        with TestClient(app) as client:
            blocker_threads = [
                threading.Thread(target=call_block, args=(i, client)) for i in range(2)
            ]
            for t in blocker_threads:
                t.start()

            # Deterministic proof of saturation: releases only once both
            # blocker threads AND this main thread are all inside wait()
            # simultaneously -- i.e., both of the 2 available tokens are
            # genuinely held by an in-flight, still-blocked request.
            saturation_barrier.wait(timeout=10)

            health_thread = threading.Thread(target=call_health, args=(client,))
            health_thread.start()
            health_thread.join(timeout=5)
            assert not health_thread.is_alive(), (
                "/health did not respond within the bounded join window while "
                "thread-pool capacity was fully saturated -- it appears to have "
                "queued behind the blocked synchronous routes."
            )
            assert health_result.get("status_code") == 200

            release_event.set()
            for t in blocker_threads:
                t.join(timeout=10)
                assert not t.is_alive()

    assert results == {0: 200, 1: 200}


# --- Item 2: the configured limit is applied during the REAL app's lifespan ---

def test_configured_thread_limit_is_applied_during_real_lifespan():
    """Drives `research_agent.api_app.app.lifespan()` directly (not via
    TestClient, and not the minimal reproduction above, and no HTTP
    request is ever made -- so no telemetry write is possible) -- this is
    the proof that the actual production wiring, not just the underlying
    mechanism, applies the configured value. `anyio` is patched at the
    `research_agent.api_app.app` module reference `lifespan()` itself
    calls, so this asserts exactly what that function does, not a
    reimplementation of it."""
    fake_app = MagicMock()
    mock_limiter = MagicMock()

    patches = _patched_lifespan_dependencies()
    with patches[0], patches[1], patches[2], patches[3], \
         patch("research_agent.api_app.app.default_async_openai_client", return_value=MagicMock()), \
         patch("research_agent.api_app.app.get_thread_pool_config", return_value=ThreadPoolConfig(limit=5)), \
         patch("research_agent.api_app.app.anyio") as mock_anyio:
        mock_anyio.to_thread.current_default_thread_limiter.return_value = mock_limiter

        import asyncio

        async def _run() -> None:
            async with lifespan(fake_app):
                pass

        asyncio.run(_run())

    assert mock_limiter.total_tokens == 5


def test_default_thread_limit_used_when_env_var_is_unset():
    """Same proof as above (no HTTP request, no telemetry surface), but
    confirms the DOCUMENTED DEFAULT (16, not AnyIO's own implicit 40) is
    what gets applied when ANYIO_THREAD_LIMIT is genuinely unset --
    exercising the real get_thread_pool_config() call inside lifespan(),
    not a mocked return value this time."""
    fake_app = MagicMock()
    mock_limiter = MagicMock()

    patches = _patched_lifespan_dependencies()
    with patch.dict(os.environ, {}, clear=False), \
         patches[0], patches[1], patches[2], patches[3], \
         patch("research_agent.api_app.app.default_async_openai_client", return_value=MagicMock()), \
         patch("research_agent.api_app.app.anyio") as mock_anyio:
        os.environ.pop("ANYIO_THREAD_LIMIT", None)
        mock_anyio.to_thread.current_default_thread_limiter.return_value = mock_limiter

        import asyncio

        async def _run() -> None:
            async with lifespan(fake_app):
                pass

        asyncio.run(_run())

    assert mock_limiter.total_tokens == 16


# --- Item 3: invalid configuration fails clearly at startup ---

@pytest.mark.parametrize(
    "raw_value",
    ["0", "-1", "not-a-number", "3.5", "9999"],
)
def test_invalid_thread_limit_raises_clearly(raw_value: str):
    """Unit-level proof, every documented rejection case: get_thread_pool_
    config() raises ValueError (never returns a partially-valid config,
    never silently falls back to the default) for a zero, negative,
    non-integer, float-string, or over-the-conservative-ceiling value."""
    with patch.dict(os.environ, {"ANYIO_THREAD_LIMIT": raw_value}):
        with pytest.raises(ValueError, match="ANYIO_THREAD_LIMIT"):
            get_thread_pool_config()


def test_invalid_thread_limit_aborts_real_application_startup():
    """Integration-level proof: a malformed ANYIO_THREAD_LIMIT doesn't
    just make the helper function raise in isolation -- it stops the
    REAL application from starting at all, the same fail-loud contract
    get_auth_config()/get_cors_config() already establish for their own
    env vars. The raise happens inside lifespan(), BEFORE `yield` --
    i.e. before the app can ever accept a request -- so no HTTP call is
    made here and RequestTelemetryMiddleware never fires; `_isolated_app`
    is still used for consistency with the rest of this file and because
    create_app() itself must be built under the target env."""
    with _isolated_app({"ANYIO_THREAD_LIMIT": "garbage"}) as app:
        with pytest.raises(ValueError, match="ANYIO_THREAD_LIMIT"):
            with TestClient(app):
                pass  # pragma: no cover -- startup must raise before this line


# --- Item 4: existing /health authentication behavior is unchanged ---

def test_health_stays_public_with_the_real_auth_gate_enabled():
    """Same integration proof tests/test_auth_middleware.py's own
    `test_integration_health_public_and_minimal_with_gate_enabled`
    already establishes, repeated here explicitly as one of this
    checkpoint's own required deterministic proofs (Part E, item 4) --
    a real production-shaped auth-enabled app, /health still requires no
    credentials. Goes through `_isolated_app` since this DOES issue a
    real HTTP request (RequestTelemetryMiddleware fires regardless of the
    auth outcome for the exempted /health route)."""
    protected_env = {
        "APP_ENV": "production",
        "AUTH_ENABLED": "true",
        "AUTH_USERNAME": "tester",
        "AUTH_PASSWORD": "a-sufficiently-long-password-16",
    }
    with _isolated_app(protected_env) as app:
        with TestClient(app) as client:
            response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- Item 6: application shutdown remains clean ---

def test_application_startup_and_shutdown_cycle_is_clean_and_repeatable():
    """Two consecutive full startup/shutdown cycles of the real app (with
    a non-default thread limit configured, to prove the limiter mutation
    itself leaves no state that breaks a subsequent cycle) -- each
    TestClient context-manager exit runs the ASGI shutdown path; a
    leaked resource or an exception during either shutdown would fail
    this test via the context manager's own exception propagation, not
    via a separate assertion. Goes through `_isolated_app` since each
    cycle issues a real /health request."""
    with _isolated_app({"ANYIO_THREAD_LIMIT": "4"}) as app:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
    # A second, independent cycle against a FRESH isolated app -- proves
    # the first cycle's shutdown didn't leave any process-global state
    # (the default thread limiter included) in a condition that breaks a
    # later startup.
    with _isolated_app({"ANYIO_THREAD_LIMIT": "4"}) as app2:
        with TestClient(app2) as client2:
            assert client2.get("/health").status_code == 200
