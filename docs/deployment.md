# Deployment

**Status: single-user deployment foundation complete. No cloud deployment
has occurred yet.** This document records what PR2A through PR3.2 actually
built — a protected, same-origin, single-process production package this
app can be deployed as, once a hosting platform is chosen — and lists
exactly what real deployment still requires. It is a separate, smaller,
already-completed track from `specs/production-readiness-roadmap.md`'s own
Phase 19–27 plan (OAuth, PostgreSQL, multi-user, tenant isolation) — that
roadmap remains fully deferred and undecided; nothing below substitutes
for it. See that document for the multi-user path, if/when it's needed.

## What exists today

### 1. Fail-closed single-user HTTP Basic Auth

`research_agent/auth_middleware.py`'s `BasicAuthMiddleware`, registered as
the **outermost** ASGI middleware in `research_agent/api_app/app.py`'s
`create_app()` (added last — Starlette runs most-recently-added
middleware first) — an unauthorized request never reaches CORS, request
telemetry, the body-size limit, or any route/service/provider code.

**Only `GET /health` is public.** Every other route — every API route,
every curation/session route, both chat/report SSE streams, the report
export route, `/docs`, `/openapi.json`, the frontend and its static
assets — requires valid credentials whenever the gate is enabled. This is
a default-deny allowlist, not a denylist: a future router is protected
automatically.

Configuration contract (`research_agent/config/settings.py`'s
`get_auth_config()`, read once at application construction — an invalid
production config raises and aborts `import research_agent.api`, i.e.
`uvicorn` never starts):

| Variable | Meaning | Default |
|---|---|---|
| `APP_ENV` | `local` or `production` | `local` |
| `AUTH_ENABLED` | `true`/`false` | `false` |
| `AUTH_USERNAME` | required, non-empty, no `:` (RFC 7617 delimiter), whenever `AUTH_ENABLED=true` | — |
| `AUTH_PASSWORD` | required, ≥16 characters, whenever `AUTH_ENABLED=true`; **may** contain `:` | — |

`APP_ENV=production` with `AUTH_ENABLED` not `true` always raises at
startup — **there is no production auth-disable override.** A broken gate
is fixed by restoring a previous known-good image/commit, never by
disabling auth. Local/test mode may leave the gate disabled (the current,
unauthenticated default every existing test already relies on).

CORS preflight is exempted narrowly: only an `OPTIONS` request carrying
*both* `Origin` and `Access-Control-Request-Method` bypasses the gate — a
bare `OPTIONS` still requires credentials (PR2B.1 correction; the original
PR2B version exempted every `OPTIONS` request, which was too broad).

Credentials are compared with `hmac.compare_digest` over the full
`"username:password"` representation in one call (never two sequential
compares, which would reintroduce a timing signal). The 401 response is
generic, carries `WWW-Authenticate` and `Cache-Control: no-store`, and
never reveals whether the username or password was wrong.

### 2. Same-origin frontend/backend production package

`research_agent/api_app/static_frontend.py`'s `mount_frontend()` serves
the built React app (`frontend/dist`) from the same FastAPI process and
origin as the API — registered **last** in `create_app()`, after every
`app.include_router(...)` call, so real API routes, `/docs`, and
`/openapi.json` always win over the SPA fallback.

- `/assets` is mounted via Starlette's own `StaticFiles` (correct
  Content-Type/ETag/Range handling for Vite's hashed JS/CSS bundles).
- A single catch-all `GET` route serves a real dist-root file directly
  (e.g. `favicon.svg`), serves `index.html` for `/` and any real SPA deep
  link, and returns a genuine 404 — never `index.html` — for an unmatched
  path whose first segment is a reserved API prefix (derived from each
  router's own routes, not a hand-maintained list).
- Path traversal and symlink escape are both prevented: every candidate
  path is `Path.resolve()`d (which follows symlinks to their real target)
  and checked against the resolved `dist_dir` before ever being served.
- A no-op when `frontend/dist` doesn't exist — local backend dev/tests
  never need a frontend build step.
- `BasicAuthMiddleware` protects `/`, every SPA path, and `/assets/*`
  automatically, since it wraps the whole app regardless of route
  registration order.

`frontend/src/lib/api/client.ts`'s `baseUrl()` picks the right target
without any code path change between environments: an explicit
`VITE_API_BASE_URL` always wins (trailing slashes stripped); a production
build with no override defaults to same-origin (`''`); local Vite dev
keeps defaulting to `http://localhost:8000`. Chat/report SSE and the
report-export link all share this one function.

### 3. Container packaging

`Dockerfile` (multi-stage):

1. `node:20-slim` — `npm ci` (against the committed `package-lock.json`)
   + `npm run build` → `frontend/dist`.
2. `python:3.12-slim` + `uv sync --frozen --no-dev` (against the committed
   `uv.lock`, dev-only group excluded) → `.venv`.
3. `python:3.12-slim` runtime — copies only `.venv`, `research_agent/`,
   and `frontend/dist`; no `tests/`, `.git`, `pyproject.toml`, or
   `uv.lock` in the final image.

Runtime properties:

- **Non-root**: a dedicated system user `app` (confirmed UID 999 in every
  smoke test in this session).
- **Exactly one uvicorn worker, no `--reload`**:
  `uvicorn research_agent.api:app --host 0.0.0.0 --port ${PORT:-8000}`,
  shell-form `CMD` with `exec` so uvicorn (not `sh`) receives signals
  directly for clean shutdown.
- **`ghcr.io/astral-sh/uv:0.11.28`** — pinned, not `latest` (PR3.1
  correction). Chosen as the exact version this repo's local `uv
  --version` reports and `uv.lock` is maintained against; confirmed
  published for both `linux/amd64` and `linux/arm64` via `docker manifest
  inspect` before pinning.
- **`GET /health` `HEALTHCHECK`** via stdlib `urllib` (no curl dependency
  added) — works without credentials, confirmed with `AUTH_ENABLED=true`
  in every smoke test.
- **`/app/data`** is created and `chown`'d to the runtime user — writable,
  but **not yet a persistent volume**: a container restart with no bind
  mount loses all state. Wiring an actual persistent volume is explicitly
  the next piece of open work (see below), not done here.
- No `.env`, real data, Chroma store, caches, eval artifacts,
  `node_modules`, or Git metadata ever enter an image layer
  (`.dockerignore`). Secrets (`AUTH_USERNAME`/`AUTH_PASSWORD`/
  `OPENAI_API_KEY`/etc.) are runtime-injected environment variables only,
  never baked into any layer.

### 4. Concurrency correctness fix (PR2.6B)

An independent review (PR2.6A) reproduced a real lost-update race in
`research_agent/agent.py`: `search_arxiv_tool` and
`search_semantic_scholar_tool` both did an unprotected
`session.papers = deduplicate(session.papers + papers)`. LangGraph's
`ToolNode` runs same-turn tool calls on a real thread pool, so two calls
could read the same stale `session.papers` snapshot and whichever wrote
last silently discarded the other's papers — reproduced in **20 of 20**
barrier-controlled parallel runs.

**Fix**: two dedicated `threading.Lock`s on `ResearchSession`
(`_papers_lock`, `_web_articles_lock` — deliberately *not* the existing
`_suggested_titles_lock`, which protects a different invariant), each
held only across a small, pure, in-memory read/deduplicate/write —
**never across the network call** that produces the new results. The
same class of bug was independently audited and fixed in the separate
`web_articles` pool. **No latency improvement is claimed by this fix** —
it corrects silent data loss while preserving existing concurrency;
arXiv and Semantic Scholar searches still run fully concurrently (proven
directly: a dedicated test has both mocked search functions block on a
shared barrier, which would deadlock/timeout if the lock incorrectly
covered the network wait, and it passes).

`tests/test_agent_concurrency.py` uses real `threading.Thread` +
`threading.Barrier`/lock-instrumented counters to force deterministic
interleavings — never a `sleep`-only race assertion, which would be
flaky by construction. The rest of this project's test suite (including
every other test file) runs normally sequentially; only this file's
tests deliberately exercise real multi-threaded execution.

## Validation evidence (this checkpoint, PR3.2)

- Docker image built from the current `Dockerfile` (public base images
  only, no secrets, no provider keys).
- Container run with `APP_ENV=production`, `AUTH_ENABLED=true`, a
  temporary username/password (≥16 chars), a temporary local directory
  bind-mounted at `/app/data`, one worker, an unused local port.
- Smoke-tested (zero paid/provider calls — only `/health`, `/`, `/docs`,
  `/openapi.json` were ever requested): unauthenticated `GET /health` →
  200; unauthenticated `GET /` → 401; authenticated `GET /` → frontend
  HTML; authenticated `GET /docs` → 200; authenticated `GET /openapi.json`
  → 200 (24 real paths); confirmed non-root runtime user (UID 999, one
  process); confirmed `/app/data` writable from inside the container, and
  that the app's own SQLite/Chroma files land in the bind-mounted host
  directory, not inside the image; restarted the container against the
  **same** mounted directory and confirmed a clean second startup.
- Focused regression suite: `tests/test_auth_middleware.py`,
  `tests/test_config_settings.py`, `tests/test_static_frontend.py`,
  `tests/test_agent.py`, `tests/test_agent_concurrency.py` — 104 passed.
  `frontend/src/lib/api/client.test.ts` — 33 passed. Frontend production
  build — clean. `git diff --check` — clean.
- Real `data/history.sqlite`/`data/usage_telemetry.sqlite`/
  `data/qa_checkpoints.sqlite` were not mutated by any of the above (the
  container used an isolated temporary bind mount, not the repo's real
  `data/` directory).

## Open deployment work (not started)

- **Hosting-platform selection** — no target chosen yet (Fly.io,
  Render, Railway, a VPS, etc.); PR1's constraints (one process, one
  Uvicorn worker, persistent local state, long-lived SSE streams) rule
  out a serverless backend.
- **HTTPS termination** — this container serves plain HTTP; a real
  deployment needs the platform's or a reverse proxy's TLS layer in
  front of it.
- **Secret injection** — `AUTH_USERNAME`/`AUTH_PASSWORD`/
  `OPENAI_API_KEY`/etc. need to be provisioned through the actual
  hosting platform's secret store; nothing beyond local, temporary,
  throwaway credentials has been used anywhere in this track.
- **Persistent-volume provisioning** — `/app/data` is writable but not
  yet backed by a real platform volume; every restart today (without an
  explicit bind mount) loses all SQLite/Chroma/checkpoint state.
- **Backup/restore drill** — no backup mechanism or restore rehearsal
  exists yet.
- **Staging deployment and one bounded live journey** — no real
  deployment has occurred; the one paid, real end-to-end journey this
  project's own PR-phase plan calls for is still pending and requires
  separate, explicit approval when it happens.
- **Monitoring and operational alerts** — nothing beyond the existing
  Langfuse tracing and the local `usage_telemetry.sqlite` table exists;
  no external alerting is wired up.
- **SQLite/Chroma limitations for multi-worker or multi-instance
  deployment** — this package is explicitly single-process/single-worker
  by design (PR1's own constraint). SQLite's per-request-connection
  pattern and Chroma's local `PersistentClient` are not safe to run
  behind more than one worker or more than one instance without a real
  redesign (a second Postgres/Chroma-server-mode/shared-storage
  migration, all still out of scope) — this is a hard architectural
  ceiling, not a tuning knob.

## Related documents

`specs/production-readiness-roadmap.md` (the separate, still fully
deferred multi-user/OAuth/PostgreSQL plan) · `specs/backend-backlog.md`
§4 (explicitly deferred platform work) · `docs/architecture.md` (current
system architecture, including this deployment layer)
