# syntax=docker/dockerfile:1
#
# PR3: single production container. FastAPI (research_agent.api:app)
# serves the API AND the built React frontend from the same origin --
# see research_agent/api_app/static_frontend.py for the serving logic
# this image just needs to ship the inputs for (frontend/dist, the
# locked Python production environment).
#
# One process, one uvicorn worker, no --reload -- matches this
# project's PR1/PR2 "single-instance" deployment constraint. Persistent-
# volume configuration, backup/restore, and platform wiring are all
# explicitly PR4/PR5 scope, not this file's.

# ---- Stage 1: frontend build ----------------------------------------
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
# package.json + the lockfile only, first -- so `npm ci` is its own
# cached layer and doesn't get invalidated by an unrelated source change.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build
# Produces /app/frontend/dist -- the exact directory
# research_agent/api_app/static_frontend.py looks for by default.


# ---- Stage 2: locked Python production dependencies -------------------
FROM python:3.12-slim AS python-deps
# The official, minimal way to get the `uv` binary into a plain Python
# base image without adding it as a project dependency -- Astral's own
# documented pattern, no curl/pip install needed. PR3.1: pinned to
# 0.11.28 (not `latest`) -- the exact version this repo's local `uv
# --version` reports and uv.lock was generated/is maintained against;
# confirmed published for both linux/amd64 and linux/arm64 via `docker
# manifest inspect ghcr.io/astral-sh/uv:0.11.28`. `latest` is a floating
# tag: two builds on different days (or a compromised push to that tag)
# could silently install a different uv binary with no diff in this file
# to show it -- `--frozen` below only protects lockfile RESOLUTION from
# drifting, not which uv binary performs the sync.
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /usr/local/bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
# --frozen: fail rather than silently re-resolve if uv.lock is stale.
# --no-dev: excludes the dev-only dependency group (pytest) -- see
# pyproject.toml's [dependency-groups]. This project's pyproject.toml
# has no [build-system]/package config of its own (confirmed: `uv sync`
# never installs `research_agent` itself into site-packages, only its
# dependencies) -- research_agent/ is copied in and imported the exact
# same CWD/PYTHONPATH-relative way `uv run uvicorn research_agent.api:app`
# already does in local dev, not via a wheel/editable install.
RUN uv sync --frozen --no-dev


# ---- Stage 3: runtime --------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root: a dedicated system user/group, no login shell, no home-dir
# surprises.
RUN groupadd --system app && useradd --system --gid app --home /app --no-create-home app

WORKDIR /app

COPY --from=python-deps /app/.venv /app/.venv
COPY research_agent/ /app/research_agent/
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# PYTHONPATH=/app: belt-and-suspenders alongside uvicorn's own CWD-based
# module resolution (uvicorn inserts the current working directory onto
# sys.path when loading an "module:attr" app string) -- makes
# `import research_agent` resolve deterministically regardless of
# exactly how uvicorn is invoked. PYTHONUNBUFFERED: container logs
# appear immediately, not only at process exit/buffer-flush.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1

# A writable data directory -- PR4's job to wire an actual persistent
# volume onto this path; created here only so the app has somewhere
# writable to start SQLite/Chroma/checkpoint files without a volume
# mounted (e.g. a first local `docker run` smoke test).
RUN mkdir -p /app/data && chown -R app:app /app

USER app

EXPOSE 8000

# GET /health is the one route that stays public even when
# AUTH_ENABLED=true (research_agent/auth_middleware.py's own allowlist)
# -- calling it here needs no credentials and no extra runtime
# dependency (plain stdlib urllib, not curl -- not installed in the
# slim base image and deliberately not added just for this).
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/health', timeout=2)"]

# Secrets (APP_ENV, AUTH_ENABLED, AUTH_USERNAME, AUTH_PASSWORD,
# OPENAI_API_KEY, and every other provider key) are runtime-injected
# environment variables only -- nothing is baked into any layer above.
# Shell form (not exec-array) so ${PORT:-8000} actually expands; `exec`
# replaces the shell process so uvicorn (not `sh`) is PID 1 and receives
# signals directly. Exactly one uvicorn worker, no --reload.
CMD ["sh", "-c", "exec uvicorn research_agent.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
