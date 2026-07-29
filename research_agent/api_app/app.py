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

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import research_agent.api as api


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema creation/migration only needs to happen once, on a throwaway
    # connection — every request after this opens its own via get_db_connection
    # (a FastAPI dependency, safe for the multi-threaded request handling a
    # single shared connection was not).
    api.init_db().close()
    api._state["client"] = api.OpenAI()
    api._state["collection"] = api.get_chroma_collection()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Research Paper Summarizer API", lifespan=lifespan)

    # curation-api-and-ui Phase 6c: the React frontend runs as its own Vite
    # dev-server process -- a genuinely separate origin from this API, so
    # CORS is required for its browser-side fetch calls. FRONTEND_ORIGIN lets
    # a non-default dev-server port/deployed origin be configured without a
    # code change, same convention as this file's other env-var-driven config
    # (e.g. SEMANTIC_SCHOLAR_API_KEY).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"), "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
