"""PR3: mounts the built React frontend (frontend/dist) onto the same
FastAPI app that serves the API, so one process/one origin can serve
both. A small, standalone module -- not inlined into create_app() --
with exactly one entry point, mount_frontend(app, dist_dir), callable
with an injected temp directory in tests instead of the real
frontend/dist.

**Registration order is load-bearing.** mount_frontend() must be called
LAST in api_app/app.py's create_app(), after every app.include_router(...)
call (and after FastAPI's own /docs, /openapi.json routes, which are
already registered by FastAPI(...)'s own __init__ by that point) --
Starlette tries routes/mounts in registration order and returns the
first structural match, so every real API route must already exist
before the catch-all SPA route below is added, or it would shadow them.

**No dependency added.** Uses only starlette.staticfiles.StaticFiles (for
the hashed .../assets/ bundle, which gets correct Content-Type/ETag/
Range handling for free) and fastapi.responses.FileResponse (for
everything else) -- both already part of this project's existing
FastAPI/Starlette dependency.

**Auth is inherited, not reimplemented here.** BasicAuthMiddleware
(research_agent/auth_middleware.py) is registered as the OUTERMOST ASGI
middleware in create_app(), independent of route registration order --
it wraps every route this module adds exactly like every other route,
with no special-casing needed here. /health stays the only public route
because auth_middleware.py's own allowlist is unchanged by this module.

**Local dev/test compatibility.** mount_frontend() is a complete no-op
whenever dist_dir does not exist (the default state of a fresh checkout
before `npm run build` has ever run) -- no route is added, `create_app()`
never fails, and the existing backend test suite/local `uv run uvicorn`
workflow needs zero frontend build step, exactly as before this module
existed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles

# frontend/dist relative to the repo root. This file lives at
# research_agent/api_app/static_frontend.py -- three parents up is the
# repo root.
DEFAULT_FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

# FastAPI's own automatic docs routes -- stable, public defaults
# (docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json";
# create_app() never overrides any of the three), not derived from
# internal introspection.
_FASTAPI_BUILTIN_SEGMENTS = frozenset({"docs", "redoc", "openapi.json"})


def _reserved_top_level_segments(routers: Iterable[APIRouter]) -> frozenset[str]:
    """Every top-level path segment already claimed by a real API route
    -- every router create_app() includes, plus FastAPI's own built-in
    docs routes above.

    Deliberately reads each router's OWN `.routes` (a plain, public,
    stable APIRouter attribute, unrelated to the FastAPI app's own
    internal post-`include_router()` route representation, which this
    project's installed FastAPI version wraps in a private
    `_IncludedRouter`/`effective_candidates` structure with no direct
    per-route `.path` -- introspecting THAT would be a fragile
    dependency on FastAPI internals, exactly what this function avoids)
    -- so a future router added to create_app()'s own `routers` list is
    automatically excluded from the SPA fallback below without anyone
    needing to update a list here (see this module's own "no broad
    catch-all that shadows API prefixes" requirement)."""
    segments: set[str] = set(_FASTAPI_BUILTIN_SEGMENTS)
    for router in routers:
        for route in router.routes:
            path = getattr(route, "path", None)
            if not path:
                continue
            first = path.strip("/").split("/", 1)[0]
            if first and not first.startswith("{"):
                segments.add(first)
    return frozenset(segments)


def mount_frontend(app: FastAPI, routers: Iterable[APIRouter], dist_dir: Path | None = None) -> None:
    """No-op when dist_dir doesn't exist -- see this module's own
    docstring on local dev/test compatibility. Otherwise:

    - mounts /assets on dist_dir/assets via StaticFiles, when that
      directory exists (Vite's hashed JS/CSS bundle output);
    - registers one catch-all GET route that serves a real, non-hashed
      file at the dist root directly (e.g. favicon.svg), serves
      index.html for '/' and every unmatched path NOT under a reserved
      API-prefix segment (the SPA deep-link fallback), and returns a
      genuine 404 -- never index.html -- for an unmatched path that IS
      under a reserved API-prefix segment (e.g. a typo'd curation
      sub-route).

    `routers` must be every router `app` already had `include_router(...)`
    called on -- see this module's own "registration order is load-
    bearing" note above.

    `dist_dir` defaults to None, resolved to DEFAULT_FRONTEND_DIST_DIR
    HERE, inside the function body -- deliberately NOT a `= DEFAULT_
    FRONTEND_DIST_DIR` default argument value, which Python evaluates
    exactly once at function-DEFINITION time. A default argument would
    permanently bake in whatever DEFAULT_FRONTEND_DIST_DIR resolved to at
    import time, making it impossible for a test to override via
    `patch.object(static_frontend, "DEFAULT_FRONTEND_DIST_DIR", tmp_path)`
    -- resolving it inside the function body instead means every call
    re-reads the current module-level value.
    """
    if dist_dir is None:
        dist_dir = DEFAULT_FRONTEND_DIST_DIR
    if not dist_dir.is_dir():
        return

    index_path = dist_dir / "index.html"
    resolved_dist_dir = dist_dir.resolve()

    assets_dir = dist_dir / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    reserved = _reserved_top_level_segments(routers)

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str) -> FileResponse:
        first_segment = full_path.split("/", 1)[0]
        if first_segment in reserved:
            # An unmatched path under a REAL api prefix (a typo'd sub-
            # route, a deleted endpoint) must still 404 as a normal API
            # response, never silently become the SPA HTML shell.
            raise HTTPException(status_code=404)

        # Resolved and containment-checked before ever touching the
        # filesystem -- full_path comes straight from the URL, so a
        # request for e.g. '../../../../etc/passwd' must never be able
        # to escape dist_dir (the same defense Starlette's own
        # StaticFiles applies internally for the /assets mount above).
        candidate = (dist_dir / full_path).resolve()
        if full_path and candidate.is_relative_to(resolved_dist_dir) and candidate.is_file():
            return FileResponse(candidate)

        # Everything else -- '/' and every real SPA deep link -- gets
        # the SPA shell; the client-side router takes it from there.
        return FileResponse(index_path)
