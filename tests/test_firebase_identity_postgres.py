"""Day 3, Part D: the verified-token -> internal-user upsert, against a
real PostgreSQL container.

Requires TEST_DATABASE_URL (skipped, never failed, when unset -- same
convention as tests/test_curation_ownership_postgres.py). Concurrency
proof uses a real threading.Barrier, not a sleep.
"""

from __future__ import annotations

import os
import sys
import threading
import uuid

import psycopg
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.db.migrations import run_migrations
from research_agent.db.ownership_repository import PostgresOwnershipRepository
from research_agent.firebase_auth import VerifiedFirebaseToken
from research_agent.identity import IdentityDenied, resolve_request_identity

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is not set -- Postgres-dependent Day 3 identity tests are skipped, not failed.",
)


@pytest.fixture()
def conn():
    connection = psycopg.connect(TEST_DATABASE_URL)
    run_migrations(connection)
    with connection.cursor() as cur:
        cur.execute("TRUNCATE saved_searches, curation_owners, users RESTART IDENTITY CASCADE")
    connection.commit()
    yield connection
    connection.close()


@pytest.fixture()
def repo(conn):
    return PostgresOwnershipRepository(conn)


def _verified(uid="fb-uid-1", email="user@example.com", name="Test User") -> VerifiedFirebaseToken:
    return VerifiedFirebaseToken(firebase_uid=uid, email=email, display_name=name)


# --- Part G test 11: a valid verified token creates exactly one internal user ---

def test_first_sign_in_creates_exactly_one_user(repo, conn):
    identity = resolve_request_identity(_verified(), repo)
    assert identity.firebase_uid == "fb-uid-1"
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM users")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT id FROM users WHERE firebase_uid = %s", ("fb-uid-1",))
        assert cur.fetchone()[0] == identity.user_id


# --- Part G test 15: new users default approved=false ---

def test_first_sign_in_user_defaults_to_not_approved_not_disabled(repo):
    identity = resolve_request_identity(_verified(), repo)
    assert identity.approved is False
    assert identity.disabled is False


# --- Part G test 12: repeated sign-in preserves internal user_id ---

def test_repeated_sign_in_returns_the_same_internal_user_id(repo):
    first = resolve_request_identity(_verified(), repo)
    second = resolve_request_identity(_verified(), repo)
    third = resolve_request_identity(_verified(), repo)
    assert first.user_id == second.user_id == third.user_id


# --- Part G test 13: changed email/name update metadata, not the ownership id ---

def test_changed_email_and_display_name_update_metadata_only(repo, conn):
    first = resolve_request_identity(_verified(email="old@example.com", name="Old Name"), repo)
    updated = resolve_request_identity(_verified(email="new@example.com", name="New Name"), repo)

    assert updated.user_id == first.user_id  # ownership identity unchanged
    assert updated.email == "new@example.com"
    assert updated.display_name == "New Name"
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM users")
        assert cur.fetchone()[0] == 1  # still one row, not a new user


def test_unchanged_metadata_second_sign_in_does_no_write(repo, conn):
    resolve_request_identity(_verified(), repo)
    with conn.cursor() as cur:
        cur.execute("SELECT updated_at FROM users WHERE firebase_uid = %s", ("fb-uid-1",))
        first_updated_at = cur.fetchone()[0]
    resolve_request_identity(_verified(), repo)  # identical claims
    with conn.cursor() as cur:
        cur.execute("SELECT updated_at FROM users WHERE firebase_uid = %s", ("fb-uid-1",))
        assert cur.fetchone()[0] == first_updated_at  # no UPDATE fired


def test_token_without_an_email_creates_a_user_with_null_email(repo):
    identity = resolve_request_identity(_verified(email=None, name=None), repo)
    assert identity.email is None
    assert identity.display_name is None
    assert identity.user_id is not None


# --- Part G test 14: concurrent first sign-ins create one user ---

def test_two_concurrent_first_sign_ins_resolve_to_one_user_row():
    """Two threads, each with its own connection, call
    sync_user_from_identity for the same firebase_uid at the same instant
    (threading.Barrier). The INSERT ... ON CONFLICT (firebase_uid) DO
    UPDATE path must converge them on a single row with a single id."""
    barrier = threading.Barrier(2)
    results: dict[int, uuid.UUID] = {}
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            c = psycopg.connect(TEST_DATABASE_URL)
            try:
                r = PostgresOwnershipRepository(c)
                barrier.wait(timeout=10)
                user = r.sync_user_from_identity(
                    firebase_uid="fb-race", email=f"e{index}@example.com", display_name=f"n{index}",
                )
                results[index] = user.id
            finally:
                c.close()
        except BaseException as exc:  # noqa: BLE001 -- captured for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
        assert not t.is_alive()

    assert errors == [], f"concurrent first sign-ins raised: {errors}"
    assert len(results) == 2
    assert results[0] == results[1], "both threads must converge on one internal user_id"

    verify = psycopg.connect(TEST_DATABASE_URL)
    try:
        with verify.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM users WHERE firebase_uid = %s", ("fb-race",))
            assert cur.fetchone()[0] == 1
    finally:
        verify.close()


# --- Part G test 16: disabled users are denied ---

def test_disabled_user_is_denied(repo, conn):
    identity = resolve_request_identity(_verified(), repo)
    with conn.cursor() as cur:
        cur.execute("UPDATE users SET disabled = true WHERE id = %s", (identity.user_id,))
    conn.commit()

    with pytest.raises(IdentityDenied):
        resolve_request_identity(_verified(), repo)


# --- Part G test 17: PostgreSQL failure denies access (end to end via the authenticator) ---

def test_authenticator_over_a_closed_real_pool_fails_closed():
    """A real pool that has been closed -> pool.connection() raises ->
    build_firebase_authenticator's closure maps it to IdentityUnavailable,
    never a fall-through and never an unowned identity."""
    import base64
    import json
    import time

    from research_agent.api_app.runtime import _state
    from research_agent.config.settings import AuthConfig, DatabaseConfig
    from research_agent.db.pool import build_connection_pool
    from research_agent.identity import IdentityUnavailable, build_firebase_authenticator

    cfg = AuthConfig(
        mode="firebase", firebase_project_id="my-app-12345", firebase_emulator_host="127.0.0.1:9099",
    )
    db_cfg = DatabaseConfig(
        configured=True, url=TEST_DATABASE_URL, pool_min_size=1, pool_max_size=2,
        redacted_url="postgresql://***",
    )
    pool = build_connection_pool(db_cfg)
    pool.close()
    _state["db_pool"] = pool
    try:
        now = int(time.time())
        claims = {
            "iss": "https://securetoken.google.com/my-app-12345", "aud": "my-app-12345",
            "sub": "fb-uid-x", "exp": now + 3600, "iat": now - 10,
        }

        def b64(obj):
            return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

        token = f"{b64({'alg': 'none'})}.{b64(claims)}."

        authenticate = build_firebase_authenticator(cfg)
        with pytest.raises(IdentityUnavailable):
            authenticate(token)
    finally:
        _state.pop("db_pool", None)
