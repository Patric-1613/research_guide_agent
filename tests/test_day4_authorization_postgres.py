"""Day 4, Part E.11: approval + ownership enforcement exercised through
the real HTTP layer against a real PostgreSQL ownership store.

Requires `TEST_DATABASE_URL` (the disposable Day 2 container boundary) --
skipped, never failed, when unset. Token verification is still faked
(no real Firebase); only the ownership tables and their SQL are real.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research_agent.admission as admission
import research_agent.api as api
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.db.migrations import run_migrations
from research_agent.db.ownership_repository import PostgresOwnershipRepository
from research_agent.db.pool import build_connection_pool
from research_agent.config.settings import DatabaseConfig
from research_agent.identity import RequestIdentity
from research_agent.qa import sqlite_checkpointer
from research_agent.schema import Paper
from research_agent.storage import init_db as real_init_db

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is not set -- Postgres-dependent Day 4 tests are skipped, not failed.",
)

_PROJECT_ID = "my-app-12345"


def _paper(pid: str) -> Paper:
    return Paper(
        title=f"Paper {pid}", authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract="an abstract", url=f"http://arxiv.org/abs/{pid}", doi=None,
        citation_count=None, source="arxiv", paper_id=pid,
    )


@pytest.fixture()
def seeded_users():
    """Two approved users + one unapproved, freshly inserted. Returns
    their internal uuids."""
    conn = psycopg.connect(TEST_DATABASE_URL)
    run_migrations(conn)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE saved_searches, curation_owners, users RESTART IDENTITY CASCADE")
    conn.commit()
    repo = PostgresOwnershipRepository(conn)
    a = repo.create_user(firebase_uid="fb-a", email="a@example.com")
    b = repo.create_user(firebase_uid="fb-b", email="b@example.com")
    u = repo.create_user(firebase_uid="fb-u", email="u@example.com")
    with conn.cursor() as cur:
        cur.execute("UPDATE users SET approved = true WHERE id IN (%s, %s)", (a.id, b.id))
    conn.commit()
    conn.close()
    return {"a": a.id, "b": b.id, "unapproved": u.id}


_IDENTITIES: dict[str, RequestIdentity] = {}


def _identity(token: str) -> RequestIdentity:
    return _IDENTITIES[token]


@contextmanager
def _client(seeded_users):
    _IDENTITIES.clear()
    for name, uid in seeded_users.items():
        _IDENTITIES[name] = RequestIdentity(
            user_id=uid, firebase_uid=f"fb-{name}", email=f"{name}@example.com",
            display_name=name, approved=(name != "unapproved"), disabled=False,
        )
    pool = build_connection_pool(
        DatabaseConfig(configured=True, url=TEST_DATABASE_URL, pool_min_size=1, pool_max_size=3,
                       redacted_url="postgresql://***"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "history.sqlite"
        cp_path = Path(tmp) / "checkpoints.sqlite"
        usage_db = Path(tmp) / "usage.sqlite"
        env = {
            "APP_ENV": "local", "AUTH_MODE": "firebase", "FIREBASE_PROJECT_ID": _PROJECT_ID,
            "DATABASE_URL": TEST_DATABASE_URL, "AUTH_ENABLED": "", "AUTH_USERNAME": "", "AUTH_PASSWORD": "",
        }
        with patch.dict(os.environ, env), \
             patch.object(api, "init_db", lambda: real_init_db(db_path)), \
             patch.object(telemetry, "USAGE_DB_PATH", usage_db), \
             patch.object(admission, "USAGE_DB_PATH", usage_db), \
             patch.object(leases, "USAGE_DB_PATH", usage_db), \
             patch.object(api, "OpenAI", return_value=MagicMock()), \
             patch("research_agent.api_app.app.default_async_openai_client", return_value=MagicMock()), \
             patch("research_agent.api_app.app.build_connection_pool", return_value=pool), \
             patch("research_agent.api_app.app.build_firebase_authenticator", return_value=_identity), \
             patch.object(api, "get_chroma_collection", return_value=MagicMock()), \
             patch.object(api, "canonicalize_topic", side_effect=lambda topic, client=None: topic):
            from research_agent.api_app.app import create_app

            app = create_app()

            def _cp_override():
                with sqlite_checkpointer(cp_path) as cp:
                    yield cp

            app.dependency_overrides[api.get_curation_checkpointer] = _cp_override
            with TestClient(app) as client:
                try:
                    yield client
                finally:
                    app.dependency_overrides.clear()
    pool.close()


def _start(client, token: str, topic: str = "t") -> str:
    papers = [_paper(f"p{i}") for i in range(12)]
    with patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=([(p, 1.0) for p in papers], {})):
        r = client.post("/curation/start", json={"topic": topic, "target_count": 5},
                        headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def test_full_ownership_lifecycle_against_real_postgres(seeded_users):
    with _client(seeded_users) as client:
        # unapproved -> 403 on a product route, 200 on /me
        assert client.get("/me", headers={"Authorization": "Bearer unapproved"}).status_code == 200
        assert client.post("/curation/start", json={"topic": "t"},
                           headers={"Authorization": "Bearer unapproved"}).status_code == 403

        sid = _start(client, "a")

        # the real curation_owners row exists and is owned by a
        conn = psycopg.connect(TEST_DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("SELECT owner_id FROM curation_owners WHERE session_id = %s", (sid,))
            assert cur.fetchone()[0] == seeded_users["a"]
        conn.close()

        # a reads it; b gets the same 404 as for a random id
        assert client.get(f"/curation/{sid}", headers={"Authorization": "Bearer a"}).status_code == 200
        cross = client.get(f"/curation/{sid}", headers={"Authorization": "Bearer b"})
        missing = client.get(f"/curation/{uuid.uuid4().hex}", headers={"Authorization": "Bearer b"})
        assert cross.status_code == missing.status_code == 404
        assert cross.json() == missing.json()

        # listings are per-owner
        assert [r["session_id"] for r in client.get("/curation/reviews",
                headers={"Authorization": "Bearer a"}).json()] == [sid]
        assert client.get("/curation/reviews", headers={"Authorization": "Bearer b"}).json() == []

        # b cannot delete a's session; a can, and it removes the owner row
        assert client.request("DELETE", f"/curation/{sid}",
                              headers={"Authorization": "Bearer b"}).status_code == 404
        assert client.request("DELETE", f"/curation/{sid}",
                              headers={"Authorization": "Bearer a"}).status_code == 200

        conn = psycopg.connect(TEST_DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM curation_owners WHERE session_id = %s", (sid,))
            assert cur.fetchone()[0] == 0
        conn.close()


def test_two_owners_are_fully_isolated_against_real_postgres(seeded_users):
    with _client(seeded_users) as client:
        sid_a = _start(client, "a", "topic a")
        sid_b = _start(client, "b", "topic b")
        assert sid_a != sid_b

        a_reviews = {r["session_id"] for r in client.get(
            "/curation/reviews", headers={"Authorization": "Bearer a"}).json()}
        b_reviews = {r["session_id"] for r in client.get(
            "/curation/reviews", headers={"Authorization": "Bearer b"}).json()}
        assert a_reviews == {sid_a}
        assert b_reviews == {sid_b}
