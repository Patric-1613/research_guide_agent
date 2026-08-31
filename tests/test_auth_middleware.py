"""PR2B: tests for research_agent/auth_middleware.py -- the single-user
HTTP Basic Auth access gate. Two layers, mirroring tests/
test_request_limits.py's own split for the sibling pure-ASGI middleware:

1. ASGI-level unit tests (raw scope/receive/send via asyncio.run(), no
   FastAPI/TestClient) -- proves BasicAuthMiddleware's own parsing/
   comparison/passthrough logic in isolation, including that a rejected
   request never reaches the wrapped app at all.
2. Integration tests against a REAL, freshly-built app (create_app(),
   NOT the already-constructed research_agent.api.app -- see
   _client_with_env's own docstring for why) -- proves the gate is
   actually wired in as the outermost middleware ahead of real routers,
   /docs, /openapi.json, CORS, the request-body limit, and the
   chat/report SSE streaming routes.

Every fixture here isolates the same three real, gitignored resources
tests/test_api.py's own _client() fixture isolates (temp SQLite,
temp usage-telemetry DB, mocked OpenAI/Chroma) -- this file never opens
data/history.sqlite, data/usage_telemetry.sqlite, or data/chroma_db/.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import research_agent.admission as admission
import research_agent.api as api
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.auth_middleware import BasicAuthMiddleware
from research_agent.config.settings import AuthConfig
from research_agent.qa import sqlite_checkpointer
from research_agent.storage import init_db as real_init_db

ENABLED = AuthConfig(enabled=True, username="alice", password="s3curePlatformSecret!")
DISABLED = AuthConfig(enabled=False, username=None, password=None)


# --- ASGI-level unit tests ---

def _http_scope(method: str = "GET", path: str = "/x", headers: list[tuple[bytes, bytes]] | None = None) -> dict:
    return {"type": "http", "method": method, "path": path, "headers": headers or []}


class _RecordingApp:
    """Mirrors test_request_limits.py's own _RecordingApp -- records
    whether it was ever invoked, so a rejected request can be proven to
    have never reached it (i.e. never reached routing/services/provider
    calls)."""

    def __init__(self):
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"ok": true}'})


def _run(mw, scope: dict) -> list[dict]:
    sent: list[dict] = []

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(mw(scope, receive, send))
    return sent


def _status_of(sent: list[dict]) -> int | None:
    for m in sent:
        if m["type"] == "http.response.start":
            return m["status"]
    return None


def _headers_of(sent: list[dict]) -> dict[bytes, bytes]:
    for m in sent:
        if m["type"] == "http.response.start":
            return {name.lower(): value for name, value in m["headers"]}
    return {}


def _body_of(sent: list[dict]) -> bytes:
    return b"".join(m["body"] for m in sent if m["type"] == "http.response.body")


def _basic_header(username: str, password: str) -> bytes:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8"))
    return b"Basic " + token


def test_disabled_gate_passes_every_request_through_unchanged():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=DISABLED)

    sent = _run(mw, _http_scope(path="/curation/reviews"))

    assert app.called is True
    assert _status_of(sent) == 200


def test_missing_authorization_header_rejects_401_before_reaching_app():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)

    sent = _run(mw, _http_scope(path="/search"))

    assert app.called is False
    assert _status_of(sent) == 401
    headers = _headers_of(sent)
    assert headers[b"www-authenticate"] == b'Basic realm="research-agent", charset="UTF-8"'
    assert headers[b"cache-control"] == b"no-store"
    assert json.loads(_body_of(sent)) == {
        "detail": {"reason_code": "unauthorized", "message": "Authentication required."},
    }


def test_wrong_scheme_rejects_401():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)
    header = b"Bearer " + base64.b64encode(b"alice:s3curePlatformSecret!")

    sent = _run(mw, _http_scope(headers=[(b"authorization", header)]))

    assert app.called is False
    assert _status_of(sent) == 401


def test_malformed_base64_rejects_401_not_500():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)

    sent = _run(mw, _http_scope(headers=[(b"authorization", b"Basic ###not-base64###")]))

    assert app.called is False
    assert _status_of(sent) == 401


def test_non_utf8_decoded_bytes_rejects_401():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)
    invalid_utf8 = base64.b64encode(b"\xff\xfe:bad")

    sent = _run(mw, _http_scope(headers=[(b"authorization", b"Basic " + invalid_utf8)]))

    assert app.called is False
    assert _status_of(sent) == 401


def test_missing_colon_rejects_401():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)
    header = b"Basic " + base64.b64encode(b"no-colon-here")

    sent = _run(mw, _http_scope(headers=[(b"authorization", header)]))

    assert app.called is False
    assert _status_of(sent) == 401


def test_empty_authorization_header_rejects_401():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)

    sent = _run(mw, _http_scope(headers=[(b"authorization", b"")]))

    assert app.called is False
    assert _status_of(sent) == 401


def test_duplicate_authorization_headers_rejects_401():
    """A duplicate header is ambiguous and rejected outright, even when
    both copies carry the SAME, otherwise-correct credential -- never
    guessed at by taking the first/last."""
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)
    good = _basic_header("alice", "s3curePlatformSecret!")

    sent = _run(mw, _http_scope(headers=[(b"authorization", good), (b"authorization", good)]))

    assert app.called is False
    assert _status_of(sent) == 401


def test_wrong_username_and_wrong_password_get_identical_401():
    app1, app2 = _RecordingApp(), _RecordingApp()
    mw1 = BasicAuthMiddleware(app1, auth_config=ENABLED)
    mw2 = BasicAuthMiddleware(app2, auth_config=ENABLED)

    sent1 = _run(mw1, _http_scope(headers=[(b"authorization", _basic_header("mallory", "s3curePlatformSecret!"))]))
    sent2 = _run(mw2, _http_scope(headers=[(b"authorization", _basic_header("alice", "wrong-password-value"))]))

    assert app1.called is False and app2.called is False
    assert _status_of(sent1) == 401 == _status_of(sent2)
    assert _body_of(sent1) == _body_of(sent2)  # never reveals which field was wrong


def test_correct_credentials_pass_through():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)

    sent = _run(mw, _http_scope(headers=[(b"authorization", _basic_header("alice", "s3curePlatformSecret!"))]))

    assert app.called is True
    assert _status_of(sent) == 200


def test_password_containing_colon_authenticates_successfully():
    """PR2B.1: AUTH_USERNAME may never contain ':' (rejected at
    get_auth_config()), but AUTH_PASSWORD still may -- _parse_basic_
    credentials only ever splits the decoded header on the FIRST colon,
    so a colon anywhere in the password is unambiguous end-to-end."""
    app = _RecordingApp()
    config_with_colon_password = AuthConfig(enabled=True, username="alice", password="s3cure:Platform:Secret!")
    mw = BasicAuthMiddleware(app, auth_config=config_with_colon_password)

    sent = _run(mw, _http_scope(headers=[(b"authorization", _basic_header("alice", "s3cure:Platform:Secret!"))]))

    assert app.called is True
    assert _status_of(sent) == 200


def test_health_get_is_public_even_with_gate_enabled():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)

    sent = _run(mw, _http_scope(method="GET", path="/health"))

    assert app.called is True
    assert _status_of(sent) == 200


def test_health_post_is_not_exempted():
    """Only GET /health is public -- the allowlist is a (method, path)
    pair, not a bare path."""
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)

    sent = _run(mw, _http_scope(method="POST", path="/health"))

    assert app.called is False
    assert _status_of(sent) == 401


# --- PR2B.1: the OPTIONS exemption is narrow, not blanket -- only a
# genuine CORS preflight (BOTH Origin AND Access-Control-Request-Method
# present) bypasses the gate. A bare OPTIONS, or one missing either
# header, is not a preflight and must be challenged like any other
# request.

def test_genuine_cors_preflight_is_not_challenged():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)
    headers = [
        (b"origin", b"http://localhost:5173"),
        (b"access-control-request-method", b"POST"),
    ]

    sent = _run(mw, _http_scope(method="OPTIONS", path="/curation/reviews", headers=headers))

    assert app.called is True
    assert _status_of(sent) == 200


def test_bare_options_is_challenged():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)

    sent = _run(mw, _http_scope(method="OPTIONS", path="/curation/reviews"))

    assert app.called is False
    assert _status_of(sent) == 401


def test_options_with_only_origin_is_challenged():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)
    headers = [(b"origin", b"http://localhost:5173")]

    sent = _run(mw, _http_scope(method="OPTIONS", path="/curation/reviews", headers=headers))

    assert app.called is False
    assert _status_of(sent) == 401


def test_options_with_only_access_control_request_method_is_challenged():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)
    headers = [(b"access-control-request-method", b"POST")]

    sent = _run(mw, _http_scope(method="OPTIONS", path="/curation/reviews", headers=headers))

    assert app.called is False
    assert _status_of(sent) == 401


def test_authenticated_bare_options_reaches_normal_routing():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)
    headers = [(b"authorization", _basic_header("alice", "s3curePlatformSecret!"))]

    sent = _run(mw, _http_scope(method="OPTIONS", path="/curation/reviews", headers=headers))

    assert app.called is True
    assert _status_of(sent) == 200


def test_non_http_scope_passes_through_unchanged():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)

    sent = _run(mw, {"type": "lifespan"})

    assert app.called is True


def test_no_credential_ever_appears_in_the_401_response():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED)

    sent = _run(mw, _http_scope(headers=[(b"authorization", _basic_header("alice", "wrong-password-value"))]))

    response_text = (_body_of(sent) + b"".join(v for _, v in _headers_of(sent).items())).decode(
        "utf-8", errors="replace",
    )
    assert "wrong-password-value" not in response_text
    assert "s3curePlatformSecret!" not in response_text  # the real configured secret
    assert "alice" not in response_text


# --- the 401 carries credentialed-CORS headers for an allowed origin ---

_ALLOWED = ("http://localhost:5173", "https://research.example.com")


def test_401_from_an_allowed_origin_carries_credentialed_cors_headers():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED, allowed_origins=_ALLOWED)

    sent = _run(mw, _http_scope(path="/search", headers=[(b"origin", b"https://research.example.com")]))

    assert app.called is False
    assert _status_of(sent) == 401
    headers = _headers_of(sent)
    # the CORS headers CORSMiddleware (inner) would have added
    assert headers[b"access-control-allow-origin"] == b"https://research.example.com"
    assert headers[b"access-control-allow-credentials"] == b"true"
    assert headers[b"vary"] == b"Origin"
    # the existing 401 headers/body are untouched
    assert headers[b"www-authenticate"] == b'Basic realm="research-agent", charset="UTF-8"'
    assert headers[b"cache-control"] == b"no-store"
    assert json.loads(_body_of(sent))["detail"]["reason_code"] == "unauthorized"


def test_401_echoes_the_exact_matched_origin_never_a_wildcard():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED, allowed_origins=_ALLOWED)

    sent = _run(mw, _http_scope(headers=[(b"origin", b"http://localhost:5173")]))

    headers = _headers_of(sent)
    assert headers[b"access-control-allow-origin"] == b"http://localhost:5173"
    assert headers[b"access-control-allow-origin"] != b"*"


def test_401_from_a_disallowed_origin_carries_no_cors_headers():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED, allowed_origins=_ALLOWED)

    sent = _run(mw, _http_scope(headers=[(b"origin", b"https://evil.example.com")]))

    assert _status_of(sent) == 401
    headers = _headers_of(sent)
    assert b"access-control-allow-origin" not in headers
    assert b"access-control-allow-credentials" not in headers


def test_401_with_no_origin_header_carries_no_cors_headers():
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED, allowed_origins=_ALLOWED)

    sent = _run(mw, _http_scope(path="/search"))

    assert _status_of(sent) == 401
    headers = _headers_of(sent)
    assert b"access-control-allow-origin" not in headers


def test_401_carries_no_cors_headers_when_no_origins_are_configured():
    """Same-origin production (get_cors_config -> allowed_origins == ()):
    even a request bearing an Origin gets a plain 401."""
    app = _RecordingApp()
    mw = BasicAuthMiddleware(app, auth_config=ENABLED, allowed_origins=())

    sent = _run(mw, _http_scope(headers=[(b"origin", b"http://localhost:5173")]))

    assert _status_of(sent) == 401
    assert b"access-control-allow-origin" not in _headers_of(sent)


def test_default_allowed_origins_is_empty_so_pre_h1_construction_still_works():
    """BasicAuthMiddleware(app, auth_config=...) with no 3rd arg -- every
    existing unit test constructs it this way -- still behaves as before."""
    mw = BasicAuthMiddleware(_RecordingApp(), auth_config=ENABLED)
    assert mw.allowed_origins == frozenset()


# --- Integration tests against a real, freshly-built app ---

def _make_test_db_override(db_path: Path):
    """Same shape as tests/test_api.py's own _make_test_db_override --
    duplicated here rather than imported cross-file, matching this
    project's existing convention of each test file owning its own
    isolation fixture (see K5D.2c/K5D.2d fixing this same gap
    independently in tests/test_curation_api.py and tests/test_api.py)."""
    def _override():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    return _override


def _make_test_checkpointer_override(cp_db_path: Path):
    """Same shape as tests/test_curation_api.py's own
    _make_test_checkpointer_override -- every curation/session/chat/
    report route (including both SSE streams and /curation/reviews)
    depends on api.get_curation_checkpointer, which otherwise opens the
    real, gitignored data/qa_checkpoints.sqlite."""
    def _override():
        with sqlite_checkpointer(cp_db_path) as cp:
            yield cp

    return _override


@contextmanager
def _client_with_env(env: dict[str, str]):
    """Builds a FRESH app via create_app() under the given env -- NOT
    research_agent.api.app, which is already constructed at import time
    under whatever env was present then (research_agent/api.py's
    module-level `app = create_app()`), so it cannot be used to test a
    DIFFERENT auth configuration within the same test process.

    Applies the exact same lifespan-isolation patches tests/test_api.py's
    own _client() fixture uses -- they target research_agent.api's shared
    module attributes, which api_app/app.py's lifespan() reads via
    `import research_agent.api as api` at call time, so they take effect
    identically for a freshly-built app as for the real one. Never opens
    the real data/history.sqlite, data/usage_telemetry.sqlite, or
    data/chroma_db/.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.sqlite"
        usage_db_path = Path(tmp) / "usage_telemetry.sqlite"
        # clear=False (patch.dict's default): overlays only the given
        # APP_ENV/AUTH_* keys on top of the real environment, same as
        # every other test file in this project -- NOT clear=True, which
        # would also wipe OPENAI_API_KEY out of os.environ and break
        # lifespan()'s unpatched default_async_openai_client() call
        # (api.OpenAI, the SYNC client, is patched below; the async one
        # is not, since no existing test file patches it either -- it
        # constructs fine as long as a real-looking key is present, and
        # nothing in this file ever actually calls it).
        with patch.dict(os.environ, env), \
             patch.object(api, "init_db", lambda: real_init_db(db_path)), \
             patch.object(telemetry, "USAGE_DB_PATH", usage_db_path), \
             patch.object(admission, "USAGE_DB_PATH", usage_db_path), \
             patch.object(leases, "USAGE_DB_PATH", usage_db_path), \
             patch.object(api, "search_web", return_value=[]), \
             patch.object(api, "OpenAI", return_value=MagicMock()), \
             patch.object(api, "get_chroma_collection", return_value=MagicMock(name="fake_chroma_collection")):
            cp_db_path = Path(tmp) / "qa_checkpoints.sqlite"
            from research_agent.api_app.app import create_app
            fresh_app = create_app()
            fresh_app.dependency_overrides[api.get_db_connection] = _make_test_db_override(db_path)
            fresh_app.dependency_overrides[api.get_curation_checkpointer] = _make_test_checkpointer_override(cp_db_path)
            try:
                with TestClient(fresh_app) as client:
                    yield client
            finally:
                fresh_app.dependency_overrides.clear()


def _protected_env(**extra: str) -> dict[str, str]:
    env = {"APP_ENV": "local", "AUTH_ENABLED": "true", "AUTH_USERNAME": "alice", "AUTH_PASSWORD": "s3curePlatformSecret!"}
    env.update(extra)
    return env


def _auth_header(username: str = "alice", password: str = "s3curePlatformSecret!") -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def test_integration_disabled_gate_matches_current_unauthenticated_behavior():
    with _client_with_env({}) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_integration_health_public_and_minimal_with_gate_enabled():
    with _client_with_env(_protected_env()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_integration_docs_protected():
    with _client_with_env(_protected_env()) as client:
        unauthorized = client.get("/docs")
        authorized = client.get("/docs", headers=_auth_header())
    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_integration_openapi_json_protected():
    with _client_with_env(_protected_env()) as client:
        unauthorized = client.get("/openapi.json")
        authorized = client.get("/openapi.json", headers=_auth_header())
    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert "paths" in authorized.json()


def test_integration_wrong_credentials_on_openapi_json_get_401():
    with _client_with_env(_protected_env()) as client:
        response = client.get("/openapi.json", headers=_auth_header(password="not-the-real-password"))
    assert response.status_code == 401


def test_integration_curation_route_protected_before_handler_executes():
    """Proves the auth gate rejects BEFORE FastAPI's dependency
    injection -- and therefore the real route handler/service -- ever
    runs: the checkpointer dependency every curation route depends on is
    overridden to raise AssertionError if it is ever resolved. An
    unauthorized request must never trigger it."""
    def _boom():
        raise AssertionError("checkpointer dependency must not be resolved for an unauthorized request")

    with _client_with_env(_protected_env()) as client:
        client.app.dependency_overrides[api.get_curation_checkpointer] = _boom
        try:
            response = client.get("/curation/reviews")
        finally:
            client.app.dependency_overrides.pop(api.get_curation_checkpointer, None)

    assert response.status_code == 401


def test_integration_chat_stream_route_protected_without_provider_calls():
    """The chat/report SSE routes share the same checkpointer dependency
    as every other curation route -- an unauthorized POST to the stream
    endpoint must never resolve it, which also proves no provider call
    (only reachable deeper inside the stream, after cp is resolved) ever
    fires."""
    def _boom():
        raise AssertionError("checkpointer dependency must not be resolved for an unauthorized request")

    with _client_with_env(_protected_env()) as client:
        client.app.dependency_overrides[api.get_curation_checkpointer] = _boom
        try:
            response = client.post("/curation/some-session-id/chat/stream", json={"message": "hello"})
        finally:
            client.app.dependency_overrides.pop(api.get_curation_checkpointer, None)

    assert response.status_code == 401


def test_integration_report_stream_route_protected_without_provider_calls():
    def _boom():
        raise AssertionError("checkpointer dependency must not be resolved for an unauthorized request")

    with _client_with_env(_protected_env()) as client:
        client.app.dependency_overrides[api.get_curation_checkpointer] = _boom
        try:
            response = client.post("/curation/some-session-id/report/stream", json={})
        finally:
            client.app.dependency_overrides.pop(api.get_curation_checkpointer, None)

    assert response.status_code == 401


def test_integration_export_route_reaches_handler_with_valid_credentials():
    """Proves the auth boundary itself works correctly for the one route
    consumed as a plain browser download link (`<a href download>`, not
    a JS fetch() call) rather than fabricating a real curation session
    (out of scope for this PR -- no real session mutation). Without
    credentials: 401, never reaching the handler. With valid, browser-
    style credentials on a nonexistent session id: the request clears
    the auth boundary and reaches the real handler, which then correctly
    404s for "no such session" -- proving auth, not session lookup,
    is what the unauthorized case was blocked by."""
    with _client_with_env(_protected_env()) as client:
        unauthorized = client.get("/curation/does-not-exist/report/export")
        authorized = client.get("/curation/does-not-exist/report/export", headers=_auth_header())

    assert unauthorized.status_code == 401
    assert authorized.status_code != 401


def test_integration_normal_route_returns_200_with_correct_credentials():
    with _client_with_env(_protected_env()) as client:
        response = client.get("/curation/reviews", headers=_auth_header())
    assert response.status_code == 200
    assert response.json() == []


def test_integration_no_paid_action_or_http_request_telemetry_for_unauthorized_requests():
    """The auth gate is registered as the OUTERMOST middleware -- an
    unauthorized request must never reach RequestTelemetryMiddleware at
    all, so it writes zero http_requests rows (and, transitively, zero
    paid_actions rows, since those are only ever opened from inside a
    route/service that runs strictly after telemetry)."""
    with _client_with_env(_protected_env()) as client:
        for _ in range(3):
            client.get("/curation/reviews")  # no Authorization header
            client.post("/search", json={"topic": "quantum computing"})

        usage_db_path = telemetry.USAGE_DB_PATH
        conn = sqlite3.connect(usage_db_path)
        try:
            http_request_count = conn.execute("SELECT COUNT(*) FROM http_requests").fetchone()[0]
            paid_action_count = conn.execute("SELECT COUNT(*) FROM paid_actions").fetchone()[0]
        finally:
            conn.close()

    assert http_request_count == 0
    assert paid_action_count == 0


def test_integration_genuine_cors_preflight_not_challenged_and_still_gets_cors_headers():
    """PR2B.1: a genuine preflight (both Origin and Access-Control-
    Request-Method present) must clear the auth gate AND still get
    CORSMiddleware's own response/headers -- proves the narrowed OPTIONS
    exemption didn't just avoid a 401, but left CORSMiddleware's real
    preflight-handling behavior completely intact."""
    with _client_with_env(_protected_env()) as client:
        response = client.options(
            "/search",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
        )
    assert response.status_code == 200
    assert "access-control-allow-origin" in {k.lower() for k in response.headers.keys()}


def test_integration_bare_options_is_challenged_even_against_a_real_route():
    with _client_with_env(_protected_env()) as client:
        response = client.options("/search")
    assert response.status_code == 401


def test_integration_existing_body_limit_middleware_still_enforced_when_gate_enabled():
    with _client_with_env(_protected_env()) as client:
        oversized_topic = "x" * (1024 * 1024)  # 1 MiB, far above the 64 KiB provisional limit
        response = client.post("/search", json={"topic": oversized_topic}, headers=_auth_header())
    assert response.status_code == 413


def test_integration_unauthorized_request_from_allowed_origin_gets_readable_401_with_cors_headers():
    """An unauthenticated cross-origin request from a PERMITTED origin
    must come back as a plain, readable 401 -- with the credentialed-CORS
    headers a browser needs to actually read it -- AND still never reach
    telemetry / body parsing / a route. Local mode's default allow-list
    includes http://localhost:5173."""
    with _client_with_env(_protected_env()) as client:
        response = client.post(
            "/search",
            json={"topic": "quantum computing"},
            headers={"Origin": "http://localhost:5173"},  # no Authorization
        )

        assert response.status_code == 401
        lower = {k.lower(): v for k, v in response.headers.items()}
        assert lower["access-control-allow-origin"] == "http://localhost:5173"
        assert lower["access-control-allow-credentials"] == "true"
        assert "origin" in lower["vary"].lower()
        assert lower["www-authenticate"].startswith("Basic ")
        assert lower["cache-control"] == "no-store"
        assert response.json()["detail"]["reason_code"] == "unauthorized"

        # requirement 2: still rejected before telemetry
        conn = sqlite3.connect(telemetry.USAGE_DB_PATH)
        try:
            assert conn.execute("SELECT COUNT(*) FROM http_requests").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM paid_actions").fetchone()[0] == 0
        finally:
            conn.close()


def test_integration_unauthorized_request_from_a_disallowed_origin_gets_a_plain_401():
    with _client_with_env(_protected_env()) as client:
        response = client.post(
            "/search", json={"topic": "x"}, headers={"Origin": "https://not-my-frontend.example.com"},
        )
    assert response.status_code == 401
    lower = {k.lower() for k in response.headers.keys()}
    assert "access-control-allow-origin" not in lower


def test_integration_split_origin_production_401_carries_the_configured_origin():
    """A configured split-origin production deployment: FRONTEND_ORIGIN is
    the only allowed origin, and an unauthenticated request from it still
    gets a readable 401."""
    env = {"APP_ENV": "production", "AUTH_ENABLED": "true", "AUTH_USERNAME": "alice",
           "AUTH_PASSWORD": "s3curePlatformSecret!", "FRONTEND_ORIGIN": "https://research.example.com"}
    with _client_with_env(env) as client:
        response = client.post(
            "/search", json={"topic": "x"}, headers={"Origin": "https://research.example.com"},
        )
    assert response.status_code == 401
    lower = {k.lower(): v for k, v in response.headers.items()}
    assert lower["access-control-allow-origin"] == "https://research.example.com"
    assert lower["access-control-allow-credentials"] == "true"


def test_integration_authenticated_cross_origin_response_carries_allow_credentials():
    """The success path: once credentials are correct, CORSMiddleware
    (allow_credentials=True, explicit allow_origins) adds the credentialed
    CORS headers to the real response -- so credentials: "include" round
    trips."""
    with _client_with_env(_protected_env()) as client:
        response = client.get(
            "/curation/reviews",
            headers={**_auth_header(), "Origin": "http://localhost:5173"},
        )
    assert response.status_code == 200
    lower = {k.lower(): v for k, v in response.headers.items()}
    assert lower["access-control-allow-origin"] == "http://localhost:5173"
    assert lower["access-control-allow-credentials"] == "true"


def test_integration_production_same_origin_default_allows_no_cross_origin():
    """FRONTEND_ORIGIN unset + APP_ENV=production: a cross-origin request
    gets NO Access-Control-Allow-Origin from anywhere -- correct for a
    same-origin deployment."""
    env = {"APP_ENV": "production", "AUTH_ENABLED": "true", "AUTH_USERNAME": "alice",
           "AUTH_PASSWORD": "s3curePlatformSecret!"}
    with _client_with_env(env) as client:
        r_401 = client.get("/curation/reviews", headers={"Origin": "https://research.example.com"})
        r_ok = client.get("/curation/reviews", headers={**_auth_header(), "Origin": "https://research.example.com"})
    assert r_401.status_code == 401
    assert r_ok.status_code == 200
    for r in (r_401, r_ok):
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers.keys()}
