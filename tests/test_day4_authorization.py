"""Day 4 (public multi-user deployment foundation, see
docs/plans/public-multi-user-deployment-review.md): approval + per-user
resource ownership enforcement, tested at the HTTP boundary.

Token verification and the ownership database are both faked here (Part
E: no real Firebase, no real PostgreSQL, no provider calls). The fake
ownership repo is a plain dict keyed by session_id -> owner uuid, wired
in exactly where `api_app/access.ownership_repo()` would build a
`PostgresOwnershipRepository`. The Postgres-backed equivalents of the
ownership SQL itself live in tests/test_curation_ownership_postgres.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import psycopg_pool

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research_agent.admission as admission
import research_agent.api as api
import research_agent.db.ownership_repository as ownership_repository
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.identity import IdentityDenied, RequestIdentity
from research_agent.qa import sqlite_checkpointer
from research_agent.schema import Paper
from research_agent.storage import init_db as real_init_db

_PROJECT_ID = "my-app-12345"
_OWNER_A = uuid.uuid4()
_OWNER_B = uuid.uuid4()


def _paper(pid: str) -> Paper:
    return Paper(
        title=f"Paper {pid}", authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract="an abstract", url=f"http://arxiv.org/abs/{pid}", doi=None,
        citation_count=None, source="arxiv", paper_id=pid,
    )


def _ranked(papers):
    return [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)]


# --- fake ownership store: one dict shared across per-request instances ---

class _FakeOwnershipRepo:
    owners: dict[str, uuid.UUID] = {}

    def __init__(self, conn=None):
        pass

    @classmethod
    def reset(cls):
        cls.owners = {}

    def get_owner_row(self, session_id: str):
        owner_id = _FakeOwnershipRepo.owners.get(session_id)
        if owner_id is None:
            return None
        return SimpleNamespace(session_id=session_id, owner_id=owner_id, topic="t", stage="curate")

    def create_owner_row(self, *, session_id, owner_id, topic, display_title, stage):
        _FakeOwnershipRepo.owners[session_id] = owner_id
        return SimpleNamespace(session_id=session_id, owner_id=owner_id)

    def delete_owner_row(self, session_id: str) -> bool:
        return _FakeOwnershipRepo.owners.pop(session_id, None) is not None

    def list_owner_sessions(self, owner_id, *, limit: int = 100):
        return [
            SimpleNamespace(session_id=sid)
            for sid, oid in _FakeOwnershipRepo.owners.items()
            if oid == owner_id
        ]


class _FakePool:
    @contextmanager
    def connection(self):
        yield object()


class _BoomPool:
    def connection(self):
        # psycopg_pool raises PoolTimeout / PoolClosed (both psycopg.Error
        # subclasses) when a pooled connection cannot be handed out.
        raise psycopg_pool.PoolTimeout("pool exhausted")


_IDENTITIES: dict[str, RequestIdentity] = {}


def _identity(token: str) -> RequestIdentity:
    if token not in _IDENTITIES:
        raise AssertionError(f"test used an unregistered token: {token!r}")
    identity = _IDENTITIES[token]
    # Mirror build_firebase_authenticator: a disabled user is denied at
    # the middleware (generic 401), never handed to a route.
    if identity.disabled:
        raise IdentityDenied("user is disabled")
    return identity


def _register(token: str, *, owner_id: uuid.UUID, approved: bool, disabled: bool = False) -> str:
    _IDENTITIES[token] = RequestIdentity(
        user_id=owner_id, firebase_uid=f"fb-{owner_id}", email="u@example.com",
        display_name="U", approved=approved, disabled=disabled,
    )
    return token


@contextmanager
def _firebase_app(*, pool=None):
    """A real create_app() in AUTH_MODE=firebase with faked verification,
    a faked ownership pool/repo, and the checkpointer + history DB on temp
    files. Providers (OpenAI / Chroma) are mocked as hard tripwires."""
    _FakeOwnershipRepo.reset()
    _IDENTITIES.clear()
    pool = pool if pool is not None else _FakePool()

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "history.sqlite"
        cp_path = Path(tmp) / "checkpoints.sqlite"
        usage_db = Path(tmp) / "usage.sqlite"
        env = {
            "APP_ENV": "local", "AUTH_MODE": "firebase", "FIREBASE_PROJECT_ID": _PROJECT_ID,
            "DATABASE_URL": "postgresql://u:p@localhost:5432/x",
            "AUTH_ENABLED": "", "AUTH_USERNAME": "", "AUTH_PASSWORD": "",
        }
        with patch.dict(os.environ, env), \
             patch.object(api, "init_db", lambda: real_init_db(db_path)), \
             patch.object(telemetry, "USAGE_DB_PATH", usage_db), \
             patch.object(admission, "USAGE_DB_PATH", usage_db), \
             patch.object(leases, "USAGE_DB_PATH", usage_db), \
             patch.object(api, "OpenAI", return_value=MagicMock()), \
             patch("research_agent.api_app.app.default_async_openai_client", return_value=MagicMock()), \
             patch("research_agent.api_app.app.build_connection_pool", return_value=MagicMock()), \
             patch("research_agent.api_app.app.build_firebase_authenticator", return_value=_identity), \
             patch.object(api, "get_chroma_collection", return_value=MagicMock()), \
             patch.object(api, "canonicalize_topic", side_effect=lambda topic, client=None: topic), \
             patch.object(ownership_repository, "PostgresOwnershipRepository", _FakeOwnershipRepo):
            from research_agent.api_app.app import create_app

            app = create_app()

            def _cp_override():
                with sqlite_checkpointer(cp_path) as cp:
                    yield cp

            app.dependency_overrides[api.get_curation_checkpointer] = _cp_override
            with TestClient(app) as client:
                api._state["db_pool"] = pool
                try:
                    yield client
                finally:
                    api._state.pop("db_pool", None)
                    app.dependency_overrides.clear()


def _call(client, method: str, path: str, headers: dict, body=None):
    if method == "get":
        return client.get(path, headers=headers)
    if method == "delete":
        return client.request("DELETE", path, headers=headers)
    return client.post(path, json=(body if body is not None else {}), headers=headers)


def _start_a_curation(client, token: str, topic: str = "topic one") -> str:
    papers = [_paper(f"p{i}") for i in range(12)]
    with patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        r = client.post(
            "/curation/start", json={"topic": topic, "target_count": 5},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


# --- Part E.1: unapproved user -- /me works, product routes 403 ---

def test_unapproved_user_can_read_me_but_not_product_routes():
    with _firebase_app() as client:
        _register("unapproved", owner_id=_OWNER_A, approved=False)
        h = {"Authorization": "Bearer unapproved"}

        me = client.get("/me", headers=h)
        assert me.status_code == 200
        assert me.json()["approved"] is False

        for method, path, body in [
            ("post", "/curation/start", {"topic": "t"}),
            ("get", "/curation/reviews", None),
            ("get", "/curation/capabilities", None),
            ("post", "/curation/lanes/suggest", {"topic": "t"}),
        ]:
            r = _call(client, method, path, h, body)
            assert r.status_code == 403, (path, r.status_code)
            assert r.json()["detail"]["reason_code"] == "account_not_approved"


def test_unapproved_user_product_route_makes_zero_provider_calls():
    with _firebase_app() as client:
        _register("unapproved", owner_id=_OWNER_A, approved=False)
        with patch.object(api, "build_candidate_pool", side_effect=AssertionError("provider must not run")):
            r = client.post(
                "/curation/start", json={"topic": "t"},
                headers={"Authorization": "Bearer unapproved"},
            )
    assert r.status_code == 403


# --- Part E.2: approved user can create and access their own work ---

def test_approved_user_creates_and_reads_own_session():
    with _firebase_app() as client:
        _register("a", owner_id=_OWNER_A, approved=True)
        sid = _start_a_curation(client, "a")

        assert _FakeOwnershipRepo.owners[sid] == _OWNER_A

        state = client.get(f"/curation/{sid}", headers={"Authorization": "Bearer a"})
        assert state.status_code == 200
        assert state.json()["session_id"] == sid

        reviews = client.get("/curation/reviews", headers={"Authorization": "Bearer a"})
        assert [row["session_id"] for row in reviews.json()] == [sid]


# --- Part E.3 + E.4: cross-owner isolation, indistinguishable 404 ---

def test_owner_b_cannot_touch_owner_a_session_and_gets_the_same_404_as_missing():
    with _firebase_app() as client:
        _register("a", owner_id=_OWNER_A, approved=True)
        _register("b", owner_id=_OWNER_B, approved=True)
        sid = _start_a_curation(client, "a")
        nonexistent = uuid.uuid4().hex

        hb = {"Authorization": "Bearer b"}
        cross = client.get(f"/curation/{sid}", headers=hb)
        missing = client.get(f"/curation/{nonexistent}", headers=hb)
        assert cross.status_code == missing.status_code == 404
        assert cross.json() == missing.json()

        # b sees none of a's sessions in the listing
        assert client.get("/curation/reviews", headers=hb).json() == []

        # every session-scoped verb is blocked identically
        for method, path, body in [
            ("post", f"/curation/{sid}/picks", {"picked_paper_ids": []}),
            ("post", f"/curation/{sid}/chat", {"message": "hi"}),
            ("post", f"/curation/{sid}/report", {}),
            ("get", f"/curation/{sid}/report/export", None),
            ("delete", f"/curation/{sid}", None),
            ("post", f"/curation/{sid}/reopen", None),
        ]:
            r = _call(client, method, path, hb, body)
            assert r.status_code == 404, (path, r.status_code)
            assert r.json()["detail"] == "session_id not found"

        # ... and a's session still exists afterwards
        assert client.get(f"/curation/{sid}", headers={"Authorization": "Bearer a"}).status_code == 200


def test_cross_owner_mutation_and_stream_make_zero_provider_and_paid_action_work():
    with _firebase_app() as client:
        _register("a", owner_id=_OWNER_A, approved=True)
        _register("b", owner_id=_OWNER_B, approved=True)
        sid = _start_a_curation(client, "a")

        with patch.object(api, "build_candidate_pool", side_effect=AssertionError("no provider")), \
             patch("research_agent.services.curation_chat_service.answer_curation_chat",
                   side_effect=AssertionError("service must not run")):
            hb = {"Authorization": "Bearer b"}
            r1 = client.post(f"/curation/{sid}/chat", json={"message": "x"}, headers=hb)
            r2 = client.post(f"/curation/{sid}/chat/stream", json={"message": "x"}, headers=hb)
            r3 = client.post(f"/curation/{sid}/report/stream", json={}, headers=hb)
        assert r1.status_code == r2.status_code == r3.status_code == 404


# --- Part E.7: ownership DB failure -> 503, never a fall-through ---

def test_ownership_database_failure_denies_with_503():
    with _firebase_app(pool=_BoomPool()) as client:
        _register("a", owner_id=_OWNER_A, approved=True)
        _FakeOwnershipRepo.owners["sess-x"] = _OWNER_A

        r = client.get("/curation/sess-x", headers={"Authorization": "Bearer a"})
        assert r.status_code == 503
        assert r.json()["detail"]["reason_code"] == "identity_store_unavailable"

        listing = client.get("/curation/reviews", headers={"Authorization": "Bearer a"})
        assert listing.status_code == 503


def test_ownership_pool_missing_denies_with_503():
    with _firebase_app() as client:
        _register("a", owner_id=_OWNER_A, approved=True)
        api._state.pop("db_pool", None)
        r = client.post("/curation/start", json={"topic": "t"}, headers={"Authorization": "Bearer a"})
    assert r.status_code == 503


# --- Part E.8: client-supplied owner fields are ignored ---

def test_client_supplied_owner_fields_are_ignored():
    with _firebase_app() as client:
        _register("a", owner_id=_OWNER_A, approved=True)
        _register("b", owner_id=_OWNER_B, approved=True)
        papers = [_paper(f"p{i}") for i in range(12)]
        with patch.object(api, "build_candidate_pool", return_value=papers), \
             patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
            r = client.post(
                "/curation/start",
                json={"topic": "t", "target_count": 5, "owner_id": str(_OWNER_B),
                      "firebase_uid": "fb-b", "approved": True},
                headers={"Authorization": "Bearer a"},
            )
        assert r.status_code == 200
        sid = r.json()["session_id"]
        # ownership recorded from the token, not the body
        assert _FakeOwnershipRepo.owners[sid] == _OWNER_A
        assert client.get(f"/curation/{sid}", headers={"Authorization": "Bearer b"}).status_code == 404


# --- Part E.10: ownership check precedes checkpoint loading ---

def test_ownership_is_checked_before_the_checkpoint_is_read():
    with _firebase_app() as client:
        _register("a", owner_id=_OWNER_A, approved=True)
        _register("b", owner_id=_OWNER_B, approved=True)
        sid = _start_a_curation(client, "a")

        with patch("research_agent.services.curation_session_service.get_curation_state",
                   side_effect=AssertionError("checkpoint must not be read for a non-owner")):
            r = client.get(f"/curation/{sid}", headers={"Authorization": "Bearer b"})
        assert r.status_code == 404


# --- Part E.1 (disabled) : a disabled account is denied everywhere ---

def test_disabled_account_is_denied():
    with _firebase_app() as client:
        _register("d", owner_id=_OWNER_A, approved=True, disabled=True)
        h = {"Authorization": "Bearer d"}
        assert client.get("/me", headers=h).status_code == 401
        assert client.post("/curation/start", json={"topic": "t"}, headers=h).status_code == 401


# --- Part D: the legacy single-user search family is refused in firebase mode ---

def test_legacy_search_family_is_refused_in_firebase_mode():
    with _firebase_app() as client:
        _register("a", owner_id=_OWNER_A, approved=True)
        h = {"Authorization": "Bearer a"}
        for method, path, body in [
            ("post", "/search", {"topic": "t"}),
            ("get", "/library", None),
            ("get", "/library/1", None),
            ("post", "/summarize", {"search_id": 1}),
            ("post", "/chat", {"search_id": 1, "question": "q"}),
            ("get", "/export/1", None),
        ]:
            r = _call(client, method, path, h, body)
            assert r.status_code == 403, (path, r.status_code)
            assert r.json()["detail"]["reason_code"] == "unavailable_in_multiuser_mode"


def test_legacy_search_refusal_runs_no_provider_or_db_work():
    with _firebase_app() as client:
        _register("a", owner_id=_OWNER_A, approved=True)
        with patch.object(api, "build_candidate_pool", side_effect=AssertionError("provider must not run")):
            r = client.post("/search", json={"topic": "t"}, headers={"Authorization": "Bearer a"})
    assert r.status_code == 403


# --- Part E.9: basic / disabled modes keep the pre-Day-4 behaviour ---

def test_basic_and_disabled_modes_are_untouched_by_the_access_dependency():
    """The access dependencies are pass-throughs unless mode == firebase:
    the module-level api.app (created with no auth env -> disabled mode)
    still serves /curation/reviews and the legacy /library with no
    identity at all. (The full curation/library suites are the broader
    regression proof; this is the explicit spot check.)"""
    with tempfile.TemporaryDirectory() as tmp:
        usage_db = Path(tmp) / "usage.sqlite"
        with patch.object(api, "init_db", lambda: real_init_db(Path(tmp) / "h.sqlite")), \
             patch.object(telemetry, "USAGE_DB_PATH", usage_db), \
             patch.object(admission, "USAGE_DB_PATH", usage_db), \
             patch.object(leases, "USAGE_DB_PATH", usage_db), \
             patch.object(api, "get_chroma_collection", return_value=MagicMock()), \
             patch.object(api, "OpenAI", return_value=MagicMock()):
            api.app.dependency_overrides[api.get_curation_checkpointer] = _disabled_cp(tmp)
            api.app.dependency_overrides[api.get_db_connection] = _disabled_db(tmp)
            try:
                with TestClient(api.app) as client:
                    assert client.get("/curation/reviews").status_code == 200
                    assert client.get("/library").status_code == 200
            finally:
                api.app.dependency_overrides.clear()


def _disabled_cp(tmp):
    def _override():
        with sqlite_checkpointer(Path(tmp) / "cp.sqlite") as cp:
            yield cp
    return _override


def _disabled_db(tmp):
    def _override():
        import sqlite3
        conn = sqlite3.connect(Path(tmp) / "h.sqlite")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    return _override
