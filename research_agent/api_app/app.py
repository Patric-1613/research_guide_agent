"""FastAPI app composition: lifespan, CORS, and router registration.

Moved out of api.py (Phase 10) so app construction has a real,
independent home. `create_app()` reaches every patch-targeted name
(`init_db`, `OpenAI`, `get_chroma_collection`) and `_state` via `import
research_agent.api as api`, at call time only (inside `lifespan()`, which
only runs once uvicorn actually starts the app, long after both modules
have finished loading) — the same safe circular pattern every other
api_app/services module has relied on since Phase 2/6.

`research_agent.api:app` remains the public ASGI entrypoint — this
module intentionally does not construct its own module-level `app`, to
avoid ever having two live FastAPI instances (and two lifespans) around.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import research_agent.api as api
from research_agent.auth_middleware import BasicAuthMiddleware
from research_agent.config import get_auth_config, get_settings, get_usage_policy
from research_agent.provider_clients import default_async_openai_client
from research_agent.request_limits import RequestBodyLimitMiddleware
from research_agent.session_limits import SessionCapacityError
from research_agent.telemetry import RequestTelemetryMiddleware, init_usage_db
from research_agent.usage_guard import GUARD_REASON_HTTP_STATUS, GUARD_REASON_MESSAGE, UsageGuardRejection


async def _handle_usage_guard_rejection(request: Request, exc: UsageGuardRejection) -> JSONResponse:
    """Usage Protection M2.2A: the ONE centralized mapping from a
    UsageGuardRejection (research_agent/usage_guard.py) to an HTTP
    response, registered once here rather than repeated as a try/except
    in every router that uses the guard -- ServiceError's own
    per-router try/except (see every routers/*.py file) stays exactly
    as it was; this is a second, independent exception type with its
    own centralized handler, not a replacement for that convention.

    Body carries only a stable, machine-readable reason_code and a
    generic, user-safe message -- never exception text, a filesystem
    path, SQL, or any request content. Retry-After is set whenever the
    rejection carries a meaningful positive retry_after_seconds
    (budget rejections always have one; a lease conflict has one only
    when the current holder's expiry was itself retrievable)."""
    status_code = GUARD_REASON_HTTP_STATUS[exc.reason_code]
    body = {"reason_code": exc.reason_code, "message": GUARD_REASON_MESSAGE[exc.reason_code]}
    headers = {}
    if exc.retry_after_seconds is not None:
        headers["Retry-After"] = str(exc.retry_after_seconds)
    return JSONResponse(status_code=status_code, content={"detail": body}, headers=headers)


async def _handle_session_capacity_error(request: Request, exc: SessionCapacityError) -> JSONResponse:
    """Usage Protection M2.2C Part G: the centralized handler for the two
    new business-capacity errors (research_agent/session_limits.py) --
    selected_paper_limit_reached and chat_turn_limit_reached. Always HTTP
    409 (a capacity conflict, not a rate limit), never a Retry-After
    header (unlike UsageGuardRejection's 429/409 above, more capacity
    only appears via deselecting/deleting, not by waiting). Body carries
    only the reason code and exc's own already-generic message -- never
    session contents or request text."""
    return JSONResponse(
        status_code=409,
        content={"detail": {"reason_code": exc.reason_code, "message": exc.message}},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema creation/migration only needs to happen once, on a throwaway
    # connection — every request after this opens its own via get_db_connection
    # (a FastAPI dependency, safe for the multi-threaded request handling a
    # single shared connection was not).
    api.init_db().close()
    # Usage Protection M1.1: same one-throwaway-connection-at-startup
    # convention as api.init_db() above — every later write opens its own
    # short-lived connection (research_agent/telemetry.py's own
    # _write_http_request/_write_paid_action), never a shared one.
    init_usage_db().close()
    # Usage Protection M2.2C Part E: still api.OpenAI(...), not the shared
    # provider_clients.default_openai_client() factory used elsewhere --
    # many existing tests patch.object(api, "OpenAI", return_value=...),
    # relying on it staying the real, patchable class reference. Adding
    # the timeout= kwarg directly here preserves that while still
    # applying the same centralized, provisional provider timeout.
    api._state["client"] = api.OpenAI(timeout=get_usage_policy().provider_timeout_seconds)
    # Usage Protection M4.2A: the async counterpart, used ONLY by the
    # curation-chat streaming endpoint's own model-generation call
    # (research_agent/chat_streaming.py::stream_chat_answer) -- every
    # other provider call anywhere in this app still uses the sync
    # client above, unchanged.
    api._state["async_client"] = default_async_openai_client()
    api._state["collection"] = api.get_chroma_collection()
    yield


def create_app() -> FastAPI:
    # PR2B: read and validate the access-gate configuration FIRST, before
    # anything else in this function runs -- get_auth_config() raises on
    # any invalid production configuration (see its own docstring), and
    # that must abort application construction outright, not just fail to
    # register a middleware. research_agent.api's module-level
    # `app = create_app()` means this raise aborts import of that module
    # entirely, which is exactly what should happen if `uvicorn
    # research_agent.api:app` is ever launched with a broken production
    # auth configuration.
    auth_config = get_auth_config()

    app = FastAPI(title="Research Paper Summarizer API", lifespan=lifespan)

    # curation-api-and-ui Phase 6c: the React frontend runs as its own Vite
    # dev-server process -- a genuinely separate origin from this API, so
    # CORS is required for its browser-side fetch calls. FRONTEND_ORIGIN lets
    # a non-default dev-server port/deployed origin be configured without a
    # code change, same convention as this file's other env-var-driven config
    # (e.g. SEMANTIC_SCHOLAR_API_KEY).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[get_settings().frontend_origin, "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Usage Protection M1.1: a pure ASGI middleware (not BaseHTTPMiddleware
    # — see RequestTelemetryMiddleware's own docstring for why), observe-only.
    # Records one http_requests row per request; does not touch route
    # behavior, request/response bodies, or any header other than adding
    # X-Request-ID to the real response.
    app.add_middleware(RequestTelemetryMiddleware)

    # Usage Protection M2.2C Part A: added AFTER RequestTelemetryMiddleware
    # so it is the OUTERMOST layer (FastAPI/Starlette's add_middleware
    # makes the most-recently-added middleware run first) -- an oversized
    # body is rejected before even the request-telemetry middleware sees
    # it, not just before the route. max_bytes read once at app
    # construction time (the same "settled at process start" tradeoff
    # every Pydantic schema constraint in this phase makes -- see
    # api_app/schemas.py) via the same centralized, provisional
    # UsagePolicy every other M2 limit reads.
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=get_usage_policy().max_request_body_bytes)

    # PR2B: added LAST, so it is the OUTERMOST layer of all four
    # (add_middleware's most-recently-added-runs-first convention, same
    # as the RequestBodyLimitMiddleware comment above) -- ahead of CORS,
    # request telemetry, and the body-size limit. An unauthorized request
    # is rejected here, before any of those ever see it: no telemetry
    # row, no body read, no route/service code reached. See
    # auth_middleware.py's own docstring for the full design.
    app.add_middleware(BasicAuthMiddleware, auth_config=auth_config)

    # Usage Protection M2.2A: one centralized handler for every guarded
    # route's rejection, instead of a try/except in each router.
    app.add_exception_handler(UsageGuardRejection, _handle_usage_guard_rejection)

    # Usage Protection M2.2C Part G: same centralized-handler convention
    # as UsageGuardRejection above, for the two new capacity errors.
    app.add_exception_handler(SessionCapacityError, _handle_session_capacity_error)

    from research_agent.api_app.routers.health import router as health_router

    app.include_router(health_router)

    from research_agent.api_app.routers.search import router as search_router

    app.include_router(search_router)

    from research_agent.api_app.routers.summarize import router as summarize_router

    app.include_router(summarize_router)

    from research_agent.api_app.routers.chat import router as chat_router

    app.include_router(chat_router)

    from research_agent.api_app.routers.export import router as export_router

    app.include_router(export_router)

    from research_agent.api_app.routers.library import router as library_router

    app.include_router(library_router)

    # =============================================================================
    # curation-api-and-ui Phase 6a: HTTP exposure for the curation/report/chat
    # backend built in Phases 1-5. Purely additive — nothing above this line is
    # touched. A curation session lives in its own checkpointer-backed store
    # (qa.py's sqlite_checkpointer/QA_CHECKPOINT_DB_PATH, the same file
    # curation_session.py already used), addressed by a server-issued
    # session_id (a uuid4 hex string), not storage.py's SQLite search_id — a
    # genuinely different persistence mechanism from /search's, reused as-is
    # rather than adapted to fit the existing one.
    #
    # The interrupt/resume HTTP shape (the brief's "least obvious" part): both
    # /curation/start and /curation/{id}/picks return the SAME
    # CurationTurnResponse shape — either a fresh `batch` to present (still
    # curating) or a `stop_reason` (curation finished, batch always []). A
    # client never needs to special-case "first turn" vs. "a later turn"; the
    # response shape alone tells it what to render next.
    # =============================================================================

    from research_agent.api_app.routers.curation_core import router as curation_core_router

    app.include_router(curation_core_router)

    # Registered BEFORE curation_history/curation_reports/curation_chat below
    # — not load-bearing for those, but this router (curation_sessions.py)
    # itself registers GET /curation/reviews BEFORE GET /curation/{session_id}
    # internally, which is load-bearing: Starlette matches routes in
    # registration order, so /curation/reviews must come first or a request
    # for it would match {session_id}="reviews" instead.
    from research_agent.api_app.routers.curation_sessions import router as curation_sessions_router

    app.include_router(curation_sessions_router)

    from research_agent.api_app.routers.curation_history import router as curation_history_router
    from research_agent.api_app.routers.curation_reports import router as curation_reports_router

    app.include_router(curation_history_router)
    app.include_router(curation_reports_router)

    from research_agent.api_app.routers.curation_chat import router as curation_chat_router

    app.include_router(curation_chat_router)

    return app
