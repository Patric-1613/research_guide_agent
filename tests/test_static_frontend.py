"""PR3 Part B: tests for research_agent/api_app/static_frontend.py --
the same-origin frontend-serving helper. Two layers:

1. Unit tests against mount_frontend() directly on a bare FastAPI()
   instance with a couple of fake routers -- fast, no lifespan/DB
   isolation needed, proves the helper's own routing/traversal/reserved-
   segment logic in isolation, matching this project's own "small,
   separately testable helper" preference.
2. Integration tests against the real create_app() with a temp
   frontend/dist injected via patch.object(static_frontend,
   "DEFAULT_FRONTEND_DIST_DIR", ...) -- proves the helper is actually
   wired in correctly, registered after every real router, and inherits
   BasicAuthMiddleware protection end to end.

Every integration fixture here isolates the same real, gitignored
resources tests/test_api.py's and tests/test_auth_middleware.py's own
fixtures isolate (temp SQLite, temp checkpointer, temp usage-telemetry
DB, mocked OpenAI/Chroma) -- this file never opens data/history.sqlite,
data/usage_telemetry.sqlite, data/qa_checkpoints.sqlite, or
data/chroma_db/.
"""

from __future__ import annotations

import base64
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

import research_agent.admission as admission
import research_agent.api as api
import research_agent.api_app.static_frontend as static_frontend
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.api_app.static_frontend import mount_frontend
from research_agent.qa import sqlite_checkpointer
from research_agent.storage import init_db as real_init_db


def _write_frontend(dist_dir: Path) -> None:
    (dist_dir / "assets").mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html><body>SPA SHELL</body></html>")
    (dist_dir / "favicon.svg").write_text("<svg>fav</svg>")
    (dist_dir / "assets" / "app-abc123.js").write_text("console.log(1)")
    (dist_dir / "assets" / "app-abc123.css").write_text("body{}")


# --- Unit tests: mount_frontend() on a bare FastAPI() app ---

def _bare_app_with_frontend(dist_dir: Path) -> tuple[FastAPI, APIRouter]:
    app = FastAPI()
    fake_router = APIRouter()

    @fake_router.get("/widgets")
    def list_widgets():
        return []

    @fake_router.get("/widgets/{widget_id}")
    def get_widget(widget_id: str):
        return {"id": widget_id}

    app.include_router(fake_router)
    mount_frontend(app, [fake_router], dist_dir=dist_dir)
    return app, fake_router


def test_reserved_top_level_segments_includes_router_paths_and_fastapi_builtins():
    router = APIRouter()

    @router.get("/widgets")
    def _list():
        return []

    @router.get("/widgets/{widget_id}")
    def _get(widget_id: str):
        return {}

    segments = static_frontend._reserved_top_level_segments([router])

    assert segments == frozenset({"widgets", "docs", "redoc", "openapi.json"})


def test_mount_frontend_is_a_no_op_when_dist_dir_does_not_exist(tmp_path):
    app = FastAPI()
    missing = tmp_path / "does-not-exist"

    mount_frontend(app, [], dist_dir=missing)

    client = TestClient(app)
    assert client.get("/").status_code == 404  # no route was ever added


def test_root_returns_index_html(tmp_path):
    _write_frontend(tmp_path)
    app, _ = _bare_app_with_frontend(tmp_path)

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.text == "<html><body>SPA SHELL</body></html>"


def test_spa_deep_link_returns_index_html(tmp_path):
    _write_frontend(tmp_path)
    app, _ = _bare_app_with_frontend(tmp_path)

    response = TestClient(app).get("/curate/some-session-id")

    assert response.status_code == 200
    assert response.text == "<html><body>SPA SHELL</body></html>"


def test_api_route_registered_before_the_frontend_takes_priority(tmp_path):
    """Proves 'API routers before frontend fallback' at the routing
    level: a real, concretely-registered route always wins over the
    catch-all, regardless of the catch-all matching the same URL shape."""
    _write_frontend(tmp_path)
    app, _ = _bare_app_with_frontend(tmp_path)

    response = TestClient(app).get("/widgets")

    assert response.status_code == 200
    assert response.json() == []


def test_unknown_nested_path_under_a_reserved_api_prefix_404s_not_index_html(tmp_path):
    """/widgets/{widget_id} only matches ONE segment after /widgets, so
    a 3-segment path falls through to the catch-all -- which must still
    404 (never serve the SPA shell) because 'widgets' is reserved."""
    _write_frontend(tmp_path)
    app, _ = _bare_app_with_frontend(tmp_path)

    response = TestClient(app).get("/widgets/a/b/c")

    assert response.status_code == 404
    assert response.text != "<html><body>SPA SHELL</body></html>"


def test_hashed_assets_are_served_from_the_assets_mount(tmp_path):
    _write_frontend(tmp_path)
    app, _ = _bare_app_with_frontend(tmp_path)
    client = TestClient(app)

    js = client.get("/assets/app-abc123.js")
    css = client.get("/assets/app-abc123.css")

    assert js.status_code == 200
    assert js.text == "console.log(1)"
    assert css.status_code == 200
    assert css.text == "body{}"


def test_missing_asset_404s(tmp_path):
    _write_frontend(tmp_path)
    app, _ = _bare_app_with_frontend(tmp_path)

    response = TestClient(app).get("/assets/does-not-exist.js")

    assert response.status_code == 404


def test_non_hashed_root_static_file_is_served_directly(tmp_path):
    _write_frontend(tmp_path)
    app, _ = _bare_app_with_frontend(tmp_path)

    response = TestClient(app).get("/favicon.svg")

    assert response.status_code == 200
    assert response.text == "<svg>fav</svg>"


def test_path_traversal_cannot_escape_dist_dir(tmp_path):
    """full_path comes straight from the URL -- a request percent-
    encoding '..' must never be able to read a file outside dist_dir; it
    must fall through to the ordinary SPA-shell response instead."""
    _write_frontend(tmp_path)
    outside_secret = tmp_path.parent / "outside-secret.txt"
    outside_secret.write_text("SECRET-OUTSIDE-DIST")
    app, _ = _bare_app_with_frontend(tmp_path)

    response = TestClient(app).get("/%2e%2e/outside-secret.txt")

    assert "SECRET-OUTSIDE-DIST" not in response.text
    assert response.text == "<html><body>SPA SHELL</body></html>"


def test_symlink_inside_dist_cannot_escape_to_an_outside_file(tmp_path):
    """PR3.1: the same containment check that defeats '..'-based
    traversal above (Path.resolve() followed by is_relative_to(
    resolved_dist_dir)) must also defeat a symlink PLANTED INSIDE
    dist_dir that points OUTSIDE it -- resolve() follows a symlink to
    its real target before the containment check ever runs, so a
    request for the symlink's own path resolves to the outside file,
    fails containment, and falls through to the ordinary safe SPA
    shell -- never the linked file's contents."""
    _write_frontend(tmp_path)
    outside_secret = tmp_path.parent / "outside-symlink-secret.txt"
    outside_secret.write_text("SECRET-BEHIND-SYMLINK")
    symlink_path = tmp_path / "escape-link"
    symlink_path.symlink_to(outside_secret)
    app, _ = _bare_app_with_frontend(tmp_path)
    client = TestClient(app)

    response = client.get("/escape-link")

    # The outside file's contents are never returned, and the symlink's
    # own target is never served -- the response is byte-for-byte the
    # ordinary SPA shell, the same safe fallback the '..' case above
    # produces, not a 500/crash from an unhandled symlink.
    assert "SECRET-BEHIND-SYMLINK" not in response.text
    assert response.status_code == 200
    assert response.text == "<html><body>SPA SHELL</body></html>"

    # The symlink's mere presence elsewhere in the tree doesn't disturb
    # normal asset serving or real-API-route precedence.
    favicon = client.get("/favicon.svg")
    assert favicon.status_code == 200
    assert favicon.text == "<svg>fav</svg>"
    widgets = client.get("/widgets")
    assert widgets.status_code == 200
    assert widgets.json() == []


# --- Integration tests: the real create_app(), a temp frontend, and
# BasicAuthMiddleware wired in exactly as production would build it. ---

def _make_test_db_override(db_path: Path):
    def _override():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    return _override


def _make_test_checkpointer_override(cp_db_path: Path):
    def _override():
        with sqlite_checkpointer(cp_db_path) as cp:
            yield cp

    return _override


@contextmanager
def _client_with_env_and_frontend(env: dict[str, str], dist_dir: Path):
    """Same isolation approach as tests/test_auth_middleware.py's own
    _client_with_env -- a FRESH app via create_app(), NOT research_agent.
    api.app (already built at import time under a different frontend/dist
    state) -- plus static_frontend.DEFAULT_FRONTEND_DIST_DIR patched to
    the given temp directory. clear=False (patch.dict's default): keeps
    the real environment's OPENAI_API_KEY so the unpatched async client
    lifespan() constructs doesn't fail to build (see that file's own
    comment on this exact point)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.sqlite"
        usage_db_path = Path(tmp) / "usage_telemetry.sqlite"
        cp_db_path = Path(tmp) / "qa_checkpoints.sqlite"
        with patch.dict(os.environ, env), \
             patch.object(api, "init_db", lambda: real_init_db(db_path)), \
             patch.object(telemetry, "USAGE_DB_PATH", usage_db_path), \
             patch.object(admission, "USAGE_DB_PATH", usage_db_path), \
             patch.object(leases, "USAGE_DB_PATH", usage_db_path), \
             patch.object(api, "search_web", return_value=[]), \
             patch.object(api, "OpenAI", return_value=MagicMock()), \
             patch.object(api, "get_chroma_collection", return_value=MagicMock(name="fake_chroma_collection")), \
             patch.object(static_frontend, "DEFAULT_FRONTEND_DIST_DIR", dist_dir):
            from research_agent.api_app.app import create_app
            fresh_app = create_app()
            fresh_app.dependency_overrides[api.get_db_connection] = _make_test_db_override(db_path)
            fresh_app.dependency_overrides[api.get_curation_checkpointer] = _make_test_checkpointer_override(cp_db_path)
            try:
                with TestClient(fresh_app) as client:
                    yield client
            finally:
                fresh_app.dependency_overrides.clear()


def _auth_header(username: str = "alice", password: str = "s3curePlatformSecret!") -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _protected_env(**extra: str) -> dict[str, str]:
    env = {"APP_ENV": "local", "AUTH_ENABLED": "true", "AUTH_USERNAME": "alice", "AUTH_PASSWORD": "s3curePlatformSecret!"}
    env.update(extra)
    return env


def test_integration_app_starts_and_health_stays_public_without_frontend_dist(tmp_path):
    missing = tmp_path / "does-not-exist"
    with _client_with_env_and_frontend({}, missing) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_integration_root_returns_frontend_html_when_auth_disabled(tmp_path):
    _write_frontend(tmp_path)
    with _client_with_env_and_frontend({}, tmp_path) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.text == "<html><body>SPA SHELL</body></html>"


def test_integration_unauthenticated_root_and_spa_and_asset_require_credentials(tmp_path):
    _write_frontend(tmp_path)
    with _client_with_env_and_frontend(_protected_env(), tmp_path) as client:
        assert client.get("/").status_code == 401
        assert client.get("/curate/some-session").status_code == 401
        assert client.get("/assets/app-abc123.js").status_code == 401


def test_integration_authenticated_root_and_spa_and_asset_succeed(tmp_path):
    _write_frontend(tmp_path)
    with _client_with_env_and_frontend(_protected_env(), tmp_path) as client:
        headers = _auth_header()
        root = client.get("/", headers=headers)
        deep_link = client.get("/curate/some-session", headers=headers)
        asset = client.get("/assets/app-abc123.js", headers=headers)
    assert root.status_code == 200
    assert root.text == "<html><body>SPA SHELL</body></html>"
    assert deep_link.status_code == 200
    assert deep_link.text == "<html><body>SPA SHELL</body></html>"
    assert asset.status_code == 200
    assert asset.text == "console.log(1)"


def test_integration_health_stays_public_with_frontend_mounted_and_auth_enabled(tmp_path):
    _write_frontend(tmp_path)
    with _client_with_env_and_frontend(_protected_env(), tmp_path) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_integration_docs_and_openapi_json_unchanged_and_protected(tmp_path):
    """/docs and /openapi.json still 401 unauthenticated and 200
    authenticated (auth_middleware.py's own behavior, untouched by this
    module), and the real API schema is still what's served -- the
    frontend catch-all is include_in_schema=False, so it never pollutes
    /openapi.json's own path list."""
    _write_frontend(tmp_path)
    with _client_with_env_and_frontend(_protected_env(), tmp_path) as client:
        unauth_docs = client.get("/docs")
        unauth_openapi = client.get("/openapi.json")
        headers = _auth_header()
        auth_docs = client.get("/docs", headers=headers)
        auth_openapi = client.get("/openapi.json", headers=headers)
    assert unauth_docs.status_code == 401
    assert unauth_openapi.status_code == 401
    assert auth_docs.status_code == 200
    assert auth_openapi.status_code == 200
    schema = auth_openapi.json()
    assert "/health" in schema["paths"]
    assert "/curation/reviews" in schema["paths"]
    assert "/{full_path}" not in schema["paths"]  # the catch-all never leaks into the schema


def test_integration_unknown_path_under_curation_prefix_404s_not_frontend_html(tmp_path):
    """A path with 'curation' as its first segment but no matching real
    route (three segments deep, no route shaped that way) must still
    404 as a normal response through the FULL app, not silently become
    the SPA shell."""
    _write_frontend(tmp_path)
    with _client_with_env_and_frontend(_protected_env(), tmp_path) as client:
        response = client.get("/curation/some-id/deeply/nested/unknown", headers=_auth_header())
    assert response.status_code == 404
    assert response.text != "<html><body>SPA SHELL</body></html>"


def test_integration_export_route_not_shadowed_by_frontend_fallback(tmp_path):
    """The report-export route is registered as a real API route before
    the frontend catch-all -- a request to it must reach the REAL
    handler (which then 404s for a nonexistent session, a real
    ServiceError-mapped response), never the SPA shell."""
    _write_frontend(tmp_path)
    with _client_with_env_and_frontend(_protected_env(), tmp_path) as client:
        response = client.get("/curation/does-not-exist/report/export", headers=_auth_header())
    assert response.status_code != 401
    assert response.text != "<html><body>SPA SHELL</body></html>"


def test_integration_chat_stream_route_not_shadowed_by_frontend_fallback(tmp_path):
    """The frontend catch-all is GET-only, so a POST to the chat-stream
    route can never match it structurally -- proven here by confirming
    the checkpointer dependency (which only the REAL route resolves) is
    reached for an authenticated request."""
    _write_frontend(tmp_path)

    def _boom():
        raise AssertionError("checkpointer dependency should be resolved for this authenticated request")

    with _client_with_env_and_frontend(_protected_env(), tmp_path) as client:
        # Only override AFTER the fixture's own real checkpointer override
        # is in place, so we can prove it's this route -- not the
        # frontend fallback -- that gets reached.
        client.app.dependency_overrides[api.get_curation_checkpointer] = _boom
        try:
            with pytest.raises(AssertionError):
                client.post("/curation/some-session/chat/stream", json={"message": "hi"}, headers=_auth_header())
        finally:
            client.app.dependency_overrides.pop(api.get_curation_checkpointer, None)
