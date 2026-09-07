"""Day 3, Parts C/D/E: the Firebase token-verification boundary, the
request-identity resolution, and the FirebaseAuthMiddleware integration.

Token verification is fully mocked here (Part G: "Mock token
verification for ordinary unit tests") -- no real Firebase project, no
real token, no network. The emulator path uses crafted unsigned JWTs.
The Postgres-backed upsert behavior lives in
tests/test_firebase_identity_postgres.py.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research_agent.admission as admission
import research_agent.api as api
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.config.settings import AuthConfig
from research_agent.firebase_auth import (
    FirebaseTokenError,
    VerifiedFirebaseToken,
    extract_bearer_token,
    verify_firebase_id_token,
)
from research_agent.identity import (
    IdentityDenied,
    IdentityUnavailable,
    RequestIdentity,
    build_firebase_authenticator,
    resolve_request_identity,
)
from research_agent.storage import init_db as real_init_db

_PROJECT_ID = "my-app-12345"
_FIREBASE_CFG = AuthConfig(mode="firebase", firebase_project_id=_PROJECT_ID)
_EMULATOR_CFG = AuthConfig(mode="firebase", firebase_project_id=_PROJECT_ID, firebase_emulator_host="127.0.0.1:9099")


def _b64(obj: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


def _unsigned_jwt(payload: dict) -> str:
    return f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{_b64(payload)}."


def _claims(*, project_id: str = _PROJECT_ID, uid: str = "firebase-uid-123", **overrides) -> dict:
    now = int(time.time())
    c = {
        "iss": f"https://securetoken.google.com/{project_id}",
        "aud": project_id,
        "sub": uid,
        "exp": now + 3600,
        "iat": now - 10,
        "auth_time": now - 10,
        "email": "user@example.com",
        "name": "Test User",
    }
    c.update(overrides)
    return c


# --- extract_bearer_token: header shapes (Part G tests 1-4) ---

def test_no_authorization_header_raises():
    with pytest.raises(FirebaseTokenError):
        extract_bearer_token([])


def test_duplicate_authorization_headers_raises():
    with pytest.raises(FirebaseTokenError):
        extract_bearer_token(["Bearer a", "Bearer b"])


@pytest.mark.parametrize("value", ["Basic abc123", "Token abc123", "abc123", "bearerabc"])
def test_wrong_scheme_raises(value):
    with pytest.raises(FirebaseTokenError):
        extract_bearer_token([value])


@pytest.mark.parametrize("value", ["Bearer ", "Bearer    ", "bearer "])
def test_empty_bearer_token_raises(value):
    with pytest.raises(FirebaseTokenError):
        extract_bearer_token([value])


def test_well_formed_bearer_token_is_extracted():
    assert extract_bearer_token(["Bearer eyJ.abc.def"]) == "eyJ.abc.def"
    assert extract_bearer_token(["bearer   eyJ.abc.def  "]) == "eyJ.abc.def"


# --- verify_firebase_id_token: emulator path (unsigned) ---

def test_emulator_valid_token_returns_claims():
    result = verify_firebase_id_token(_unsigned_jwt(_claims()), _EMULATOR_CFG)
    assert result == VerifiedFirebaseToken(
        firebase_uid="firebase-uid-123", email="user@example.com", display_name="Test User",
    )


def test_emulator_token_missing_email_and_name_yields_none_metadata():
    claims = _claims()
    del claims["email"]
    del claims["name"]
    result = verify_firebase_id_token(_unsigned_jwt(claims), _EMULATOR_CFG)
    assert result.email is None
    assert result.display_name is None
    assert result.firebase_uid == "firebase-uid-123"


@pytest.mark.parametrize(
    "overrides",
    [
        {"iss": "https://securetoken.google.com/some-other-project"},   # wrong issuer (Part G test 8)
        {"iss": "https://session.firebase.google.com/my-app-12345"},    # a session cookie, not an ID token
        {"aud": "some-other-project"},                                  # wrong audience (Part G test 7)
        {"exp": int(time.time()) - 10_000},                             # expired (Part G test 6)
        {"iat": int(time.time()) + 10_000},                             # issued in the future
        {"sub": ""},                                                    # missing uid (Part G test 10)
        {"sub": 12345},                                                 # non-string uid
        {"sub": "x" * 200},                                             # uid too long
    ],
)
def test_emulator_invalid_claims_raise_generic_error(overrides):
    with pytest.raises(FirebaseTokenError):
        verify_firebase_id_token(_unsigned_jwt(_claims(**overrides)), _EMULATOR_CFG)


@pytest.mark.parametrize("token", ["not-a-jwt", "only.two", "a.b.c.d", "header..sig", "..", ""])
def test_emulator_malformed_token_raises_generic_error(token):
    with pytest.raises(FirebaseTokenError):
        verify_firebase_id_token(token, _EMULATOR_CFG)


# --- verify_firebase_id_token: real path (google-auth mocked) ---

@contextmanager
def _mock_google_verify(*, returns=None, raises=None):
    with patch("google.oauth2.id_token.verify_firebase_token") as m:
        if raises is not None:
            m.side_effect = raises
        else:
            m.return_value = returns
        yield m


def test_real_path_valid_claims_return_verified_token():
    with _mock_google_verify(returns=_claims()):
        result = verify_firebase_id_token("real.looking.token", _FIREBASE_CFG)
    assert result.firebase_uid == "firebase-uid-123"
    assert result.email == "user@example.com"


def test_real_path_passes_project_id_as_audience():
    with _mock_google_verify(returns=_claims()) as m:
        verify_firebase_id_token("real.looking.token", _FIREBASE_CFG)
    assert m.call_args.kwargs["audience"] == _PROJECT_ID


@pytest.mark.parametrize(
    "overrides",
    [
        {"iss": "https://securetoken.google.com/other"},                # wrong issuer / custom-token-shaped
        {"iss": "https://identitytoolkit.googleapis.com/..."},          # a Firebase custom token's audience-as-iss shape
        {"sub": ""},                                                    # no uid (Part G test 10)
        {"sub": None},
    ],
)
def test_real_path_rejects_bad_claims_even_when_signature_check_passed(overrides):
    with _mock_google_verify(returns=_claims(**overrides)):
        with pytest.raises(FirebaseTokenError):
            verify_firebase_id_token("real.looking.token", _FIREBASE_CFG)


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("Token signature is invalid"),      # bad signature / unsupported alg (Part G tests 5, 9)
        ValueError("Token has expired"),                # (Part G test 6)
        ValueError("Token has wrong audience"),         # (Part G test 7)
        RuntimeError("Could not fetch certificates"),   # network failure fetching certs
        KeyError("kid"),
    ],
)
def test_real_path_maps_every_verifier_exception_to_generic_error(exc):
    with _mock_google_verify(raises=exc):
        with pytest.raises(FirebaseTokenError) as ei:
            verify_firebase_id_token("real.looking.token", _FIREBASE_CFG)
    # the generic error never carries the verifier's own message
    assert str(exc) not in str(ei.value)
    assert "signature" not in str(ei.value).lower()


# --- resolve_request_identity + build_firebase_authenticator ---

class _FakeRepo:
    def __init__(self, *, user=None, raises=None):
        self._user = user
        self._raises = raises
        self.calls = []

    def sync_user_from_identity(self, *, firebase_uid, email, display_name):
        self.calls.append((firebase_uid, email, display_name))
        if self._raises is not None:
            raise self._raises
        return self._user


def _user_record(**over):
    from research_agent.db.ownership_repository import UserRecord

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    d = dict(
        id=uuid.uuid4(), firebase_uid="firebase-uid-123", email="user@example.com",
        display_name="Test User", approved=False, disabled=False, created_at=now, updated_at=now,
    )
    d.update(over)
    return UserRecord(**d)


def test_resolve_request_identity_maps_a_user_record_to_a_request_identity():
    user = _user_record()
    identity = resolve_request_identity(
        VerifiedFirebaseToken("firebase-uid-123", "user@example.com", "Test User"), _FakeRepo(user=user),
    )
    assert isinstance(identity, RequestIdentity)
    assert identity.user_id == user.id
    assert identity.firebase_uid == "firebase-uid-123"
    assert identity.approved is False
    assert identity.disabled is False


def test_resolve_request_identity_denies_a_disabled_user():
    with pytest.raises(IdentityDenied):
        resolve_request_identity(
            VerifiedFirebaseToken("firebase-uid-123", None, None), _FakeRepo(user=_user_record(disabled=True)),
        )


def test_authenticator_fails_closed_when_no_db_pool_is_initialized():
    from research_agent.api_app.runtime import _state

    _state.pop("db_pool", None)
    authenticate = build_firebase_authenticator(_EMULATOR_CFG)
    with pytest.raises(IdentityUnavailable):
        authenticate(_unsigned_jwt(_claims()))


def test_authenticator_fails_closed_when_the_pool_raises_a_psycopg_error():
    import psycopg

    from research_agent.api_app.runtime import _state

    class _BoomPool:
        def connection(self):
            raise psycopg.OperationalError("connection refused")

    _state["db_pool"] = _BoomPool()
    try:
        authenticate = build_firebase_authenticator(_EMULATOR_CFG)
        with pytest.raises(IdentityUnavailable):
            authenticate(_unsigned_jwt(_claims()))
    finally:
        _state.pop("db_pool", None)


def test_authenticator_propagates_identity_denied_not_masked_as_unavailable():
    from research_agent.api_app.runtime import _state

    class _Pool:
        @contextmanager
        def connection(self):
            yield MagicMock()

    with patch("research_agent.identity.resolve_request_identity", side_effect=IdentityDenied("disabled")):
        _state["db_pool"] = _Pool()
        try:
            authenticate = build_firebase_authenticator(_EMULATOR_CFG)
            with pytest.raises(IdentityDenied):
                authenticate(_unsigned_jwt(_claims()))
        finally:
            _state.pop("db_pool", None)


# --- FirebaseAuthMiddleware integration (create_app, fake authenticator) ---

@contextmanager
def _firebase_client(*, authenticate, extra_env=None):
    """A real create_app() in AUTH_MODE=firebase, with the identity
    authenticator replaced by `authenticate` and the DB pool build
    mocked (the fake authenticator never touches a real pool)."""
    import tempfile
    from pathlib import Path

    env = {
        "APP_ENV": "local",
        "AUTH_MODE": "firebase",
        "FIREBASE_PROJECT_ID": _PROJECT_ID,
        "DATABASE_URL": "postgresql://u:p@localhost:5432/x",
        "AUTH_ENABLED": "",
        "AUTH_USERNAME": "",
        "AUTH_PASSWORD": "",
    }
    if extra_env:
        env.update(extra_env)

    with tempfile.TemporaryDirectory() as tmp:
        usage_db = Path(tmp) / "usage.sqlite"
        with patch.dict(os.environ, env), \
             patch.object(api, "init_db", lambda: real_init_db(Path(tmp) / "h.sqlite")), \
             patch.object(telemetry, "USAGE_DB_PATH", usage_db), \
             patch.object(admission, "USAGE_DB_PATH", usage_db), \
             patch.object(leases, "USAGE_DB_PATH", usage_db), \
             patch.object(api, "OpenAI", return_value=MagicMock()), \
             patch("research_agent.api_app.app.default_async_openai_client", return_value=MagicMock()), \
             patch("research_agent.api_app.app.build_connection_pool", return_value=MagicMock()), \
             patch("research_agent.api_app.app.build_firebase_authenticator", return_value=authenticate), \
             patch.object(api, "get_chroma_collection", return_value=MagicMock()):
            from research_agent.api_app.app import create_app

            app = create_app()
            with TestClient(app) as client:
                yield client


def _ok_identity(*, approved=False, disabled=False):
    return RequestIdentity(
        user_id=uuid.uuid4(), firebase_uid="firebase-uid-123", email="user@example.com",
        display_name="Test User", approved=approved, disabled=disabled,
    )


def test_health_is_public_in_firebase_mode():
    with _firebase_client(authenticate=lambda t: _ok_identity()) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_me_requires_a_token_and_works_for_an_unapproved_user():
    identity = _ok_identity(approved=False)
    with _firebase_client(authenticate=lambda t: identity) as client:
        unauth = client.get("/me")
        ok = client.get("/me", headers={"Authorization": "Bearer good-token"})
    assert unauth.status_code == 401
    assert ok.status_code == 200
    body = ok.json()
    assert body == {
        "user_id": str(identity.user_id),
        "email": "user@example.com",
        "display_name": "Test User",
        "approved": False,
        "disabled": False,
    }
    assert "firebase_uid" not in body
    assert "token" not in body


@pytest.mark.parametrize(
    "headers",
    [
        {},                                                    # missing (Part G test 1)
        {"Authorization": "Basic dXNlcjpwYXNz"},                # wrong scheme (Part G test 2)
        {"Authorization": "Bearer "},                           # empty token (Part G test 3)
    ],
)
def test_missing_or_malformed_auth_returns_generic_401(headers):
    def _authenticate(_token):
        raise AssertionError("authenticate must not run for a header that never parses")

    with _firebase_client(authenticate=_authenticate) as client:
        r = client.get("/me", headers=headers)
    assert r.status_code == 401
    assert r.json() == {"detail": {"reason_code": "unauthorized", "message": "Authentication required."}}
    assert r.headers["www-authenticate"] == "Bearer"
    assert r.headers["cache-control"] == "no-store"


def test_bad_token_returns_generic_401_with_no_verifier_detail():
    with _firebase_client(authenticate=lambda t: (_ for _ in ()).throw(FirebaseTokenError("signature is invalid: RS256"))) as client:
        r = client.get("/me", headers={"Authorization": "Bearer bad"})
    assert r.status_code == 401
    text = r.text
    assert "signature" not in text
    assert "RS256" not in text
    assert "bad" not in r.json()["detail"]["message"]


def test_disabled_user_gets_the_same_generic_401():
    with _firebase_client(authenticate=lambda t: (_ for _ in ()).throw(IdentityDenied("user is disabled"))) as client:
        r = client.get("/me", headers={"Authorization": "Bearer good"})
    assert r.status_code == 401
    assert r.json() == {"detail": {"reason_code": "unauthorized", "message": "Authentication required."}}


def test_database_unavailable_denies_access_with_503():
    with _firebase_client(authenticate=lambda t: (_ for _ in ()).throw(IdentityUnavailable("db down"))) as client:
        r = client.get("/me", headers={"Authorization": "Bearer good"})
    assert r.status_code == 503
    assert r.json()["detail"]["reason_code"] == "identity_store_unavailable"
    assert "db down" not in r.text


def test_protected_route_is_unreachable_before_verification_and_makes_zero_provider_calls():
    """A curation route (and its checkpointer dependency, and therefore
    any provider call deeper inside) must never be reached by an
    unauthorized firebase-mode request."""
    def _boom_cp():
        raise AssertionError("checkpointer dependency must not resolve for an unauthorized request")

    with _firebase_client(authenticate=lambda t: (_ for _ in ()).throw(FirebaseTokenError("nope"))) as client:
        client.app.dependency_overrides[api.get_curation_checkpointer] = _boom_cp
        try:
            r1 = client.post("/curation/some-id/chat/stream", json={"message": "hi"})
            r2 = client.post("/curation/some-id/report/stream", json={})
            r3 = client.get("/curation/reviews")
        finally:
            client.app.dependency_overrides.pop(api.get_curation_checkpointer, None)
    assert r1.status_code == 401
    assert r2.status_code == 401
    assert r3.status_code == 401


def test_docs_and_openapi_are_protected_in_firebase_mode():
    with _firebase_client(authenticate=lambda t: _ok_identity()) as client:
        assert client.get("/docs").status_code == 401
        assert client.get("/openapi.json").status_code == 401
        assert client.get("/docs", headers={"Authorization": "Bearer good"}).status_code == 200


def test_401_from_an_allowed_origin_carries_credentialed_cors_headers():
    identity = _ok_identity()
    with _firebase_client(
        authenticate=lambda t: (_ for _ in ()).throw(FirebaseTokenError("nope")),
        extra_env={"FRONTEND_ORIGIN": "https://app.example.com"},
    ) as client:
        r = client.get("/me", headers={"Origin": "https://app.example.com"})
    assert r.status_code == 401
    assert r.headers["access-control-allow-origin"] == "https://app.example.com"
    assert r.headers["access-control-allow-credentials"] == "true"
    assert r.headers["vary"] == "Origin"


def test_401_from_a_disallowed_origin_carries_no_cors_headers():
    with _firebase_client(
        authenticate=lambda t: (_ for _ in ()).throw(FirebaseTokenError("nope")),
        extra_env={"FRONTEND_ORIGIN": "https://app.example.com"},
    ) as client:
        r = client.get("/me", headers={"Origin": "https://evil.example.com"})
    assert r.status_code == 401
    assert "access-control-allow-origin" not in r.headers


def test_503_from_an_allowed_origin_also_carries_credentialed_cors_headers():
    with _firebase_client(
        authenticate=lambda t: (_ for _ in ()).throw(IdentityUnavailable("db down")),
        extra_env={"FRONTEND_ORIGIN": "https://app.example.com"},
    ) as client:
        r = client.get("/me", headers={"Origin": "https://app.example.com", "Authorization": "Bearer good"})
    assert r.status_code == 503
    assert r.headers["access-control-allow-origin"] == "https://app.example.com"


def test_firebase_mode_without_database_url_refuses_to_start():
    with patch.dict(os.environ, {
        "APP_ENV": "local", "AUTH_MODE": "firebase", "FIREBASE_PROJECT_ID": _PROJECT_ID,
        "DATABASE_URL": "", "AUTH_ENABLED": "", "AUTH_USERNAME": "", "AUTH_PASSWORD": "",
    }):
        from research_agent.api_app.app import create_app

        with pytest.raises(RuntimeError, match="AUTH_MODE=firebase requires DATABASE_URL"):
            create_app()
