"""Day 2 (public multi-user deployment foundation, see
docs/plans/public-multi-user-deployment-review.md), Part G: deterministic
tests for the PostgreSQL ownership layer, the cross-store consistency
coordinator, and the reconciliation script.

**Requires a real PostgreSQL reachable via the `TEST_DATABASE_URL`
environment variable.** Every test in this file is skipped (never
failed) when it's unset -- a normal `uv run pytest` stays completely
Docker/Postgres-free, exactly as before this checkpoint. This file must
never be pointed at a real production database: `conn` below truncates
`saved_searches`/`curation_owners`/`users` before every test for
isolation, which would be destructive against real data.

No timing-only race proofs: test_two_concurrent_creates_mint_independent_
session_ids uses a real `threading.Barrier` to force genuinely
simultaneous execution before asserting independence, not a sleep-based
inference.
"""

from __future__ import annotations

import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
import pytest

from research_agent.curation_ownership import (
    create_owned_curation_session,
    delete_owned_curation_session,
    get_owned_curation_state,
    get_owner_id_for_session,
)
from research_agent.db.migrations import run_migrations
from research_agent.db.ownership_repository import DuplicateFirebaseUidError, PostgresOwnershipRepository
from research_agent.db.saved_search_repository import PostgresSavedSearchRepository
from research_agent.qa import sqlite_checkpointer
from research_agent.query_expansion import PaperPoolSession
from scripts.reconcile_curation_ownership import run_reconciliation

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is not set -- Postgres-dependent Day 2 tests are skipped, not failed.",
)


@pytest.fixture()
def conn():
    connection = psycopg.connect(TEST_DATABASE_URL)
    run_migrations(connection)  # idempotent -- safe even if already applied
    with connection.cursor() as cur:
        # FK-safe truncate order: saved_searches/curation_owners both
        # reference users. RESTART IDENTITY resets saved_searches' own
        # serial id so each test's assertions about ids stay simple.
        cur.execute("TRUNCATE saved_searches, curation_owners, users RESTART IDENTITY CASCADE")
    connection.commit()
    yield connection
    connection.close()


@pytest.fixture()
def repo(conn):
    return PostgresOwnershipRepository(conn)


@pytest.fixture()
def checkpointer():
    with tempfile.TemporaryDirectory() as tmp:
        with sqlite_checkpointer(Path(tmp) / "qa_checkpoints.sqlite") as cp:
            yield cp


def _make_session(topic: str = "test topic") -> PaperPoolSession:
    return PaperPoolSession(
        topic=topic, reserve=[], cursor=0, seen_paper_ids=set(), seen_titles=set(),
        stage="curate", target_count=10,
    )


# --- users: uniqueness / email / approval defaults (Part G items 11-13) ---

def test_firebase_uid_is_unique(repo):
    repo.create_user(firebase_uid="fb-dup", email="a@example.com")
    with pytest.raises(DuplicateFirebaseUidError):
        repo.create_user(firebase_uid="fb-dup", email="b@example.com")


def test_email_is_not_the_ownership_key(repo):
    """Two DIFFERENT users may share the same email -- firebase_uid, not
    email, is what identifies a user. This is deliberate (see
    0001_initial.sql's own comment: no uniqueness constraint on email, to
    support a future account-deletion-then-re-signup flow without a
    schema change)."""
    user_a = repo.create_user(firebase_uid="fb-shared-email-a", email="shared@example.com")
    user_b = repo.create_user(firebase_uid="fb-shared-email-b", email="shared@example.com")
    assert user_a.id != user_b.id
    assert user_a.email == user_b.email == "shared@example.com"


def test_new_users_default_to_not_approved_and_not_disabled(repo):
    user = repo.create_user(firebase_uid="fb-new", email="new@example.com")
    assert user.approved is False
    assert user.disabled is False


# --- saved searches: owner-filtered at the repository level (Part G item 14) ---

def test_postgres_saved_searches_are_owner_filtered_at_the_repository_level(repo, conn):
    user_a = repo.create_user(firebase_uid="fb-ss-a", email="ssa@example.com")
    user_b = repo.create_user(firebase_uid="fb-ss-b", email="ssb@example.com")
    ss_repo = PostgresSavedSearchRepository(conn)

    ss_repo.save(owner_id=str(user_a.id), topic="a1", paper_ids=["p1"], scores=[0.9])
    ss_repo.save(owner_id=str(user_b.id), topic="b1", paper_ids=["p2"], scores=[0.8])
    ss_repo.save(owner_id=str(user_a.id), topic="a2", paper_ids=["p3"], scores=[0.7])

    a_results = ss_repo.list_by_owner(str(user_a.id))
    assert {r.topic for r in a_results} == {"a1", "a2"}
    assert all(r.owner_id == str(user_a.id) for r in a_results)

    b_results = ss_repo.list_by_owner(str(user_b.id))
    assert {r.topic for r in b_results} == {"b1"}

    # Newest-first ordering, same contract as the SQLite side.
    assert a_results[0].topic == "a2"


# --- coordinator: creation ordering and failure behavior (Part G items 1-3) ---

def test_owner_write_failure_prevents_checkpoint_creation(repo, checkpointer, conn, monkeypatch):
    """Item 1: if the Postgres owner-insert itself fails, the checkpoint
    step must never be reached at all."""
    checkpoint_calls = []
    import research_agent.curation_ownership as ownership_module

    def _spy_save(session, session_id, cp):
        checkpoint_calls.append(session_id)

    monkeypatch.setattr(ownership_module, "save_curation_session", _spy_save)

    user = repo.create_user(firebase_uid="fb-owner-fail", email="ownerfail@example.com")

    class _FailingRepo:
        def create_owner_row(self, **kwargs):
            raise psycopg.errors.UndefinedTable("simulated owner-insert failure")

    with pytest.raises(psycopg.errors.UndefinedTable):
        create_owned_curation_session(
            owner_id=user.id, session=_make_session(), topic="t", display_title=None,
            stage="curate", ownership_repo=_FailingRepo(), checkpointer=checkpointer,
        )
    assert checkpoint_calls == [], "checkpoint creation must never be invoked when the owner-insert fails"


def test_checkpoint_write_failure_leaves_a_fail_closed_incomplete_owner_record(repo, checkpointer, monkeypatch):
    """Item 2: if the checkpoint write fails AFTER the owner row was
    already inserted, the owner row must remain (fail-closed incomplete
    -- never deleted/repaired by this function itself)."""
    import research_agent.curation_ownership as ownership_module

    def _failing_save(session, session_id, cp):
        raise RuntimeError("simulated checkpoint-write failure")

    monkeypatch.setattr(ownership_module, "save_curation_session", _failing_save)

    user = repo.create_user(firebase_uid="fb-cp-fail", email="cpfail@example.com")

    with pytest.raises(RuntimeError, match="simulated checkpoint-write failure"):
        create_owned_curation_session(
            owner_id=user.id, session=_make_session(), topic="t", display_title=None,
            stage="curate", ownership_repo=repo, checkpointer=checkpointer,
        )

    # The owner row must still exist -- this is the "fail-closed
    # incomplete" state, not an error this function silently cleans up.
    remaining = repo.list_owner_sessions(user.id)
    assert len(remaining) == 1


def test_missing_checkpoint_state_is_never_returned_as_a_usable_session(repo, checkpointer):
    """Item 3: an owner row with no matching checkpoint (simulating the
    aftermath of item 2's failure) must read back as None, never as a
    partially-constructed session."""
    user = repo.create_user(firebase_uid="fb-missing-cp", email="missingcp@example.com")
    repo.create_owner_row(session_id="orphan-session-no-cp", owner_id=user.id, topic="t", display_title=None, stage="curate")

    result = get_owned_curation_state("orphan-session-no-cp", repo, checkpointer)
    assert result is None
    # The 404-equivalent owner lookup, by contrast, correctly still finds
    # the (incomplete) owner row -- these are two deliberately different
    # questions ("does anyone own this" vs. "is this a usable session").
    assert get_owner_id_for_session("orphan-session-no-cp", repo) == user.id


def test_retry_after_a_failed_create_mints_a_new_session_id_never_reuses_the_failed_one(repo, checkpointer, monkeypatch):
    """Explicit proof of the plan's own 'retries mint a new session_id
    rather than repairing the failed one' policy."""
    import research_agent.curation_ownership as ownership_module

    calls = {"n": 0}
    real_save = ownership_module.save_curation_session

    def _fail_once_then_succeed(session, session_id, cp):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated transient failure")
        real_save(session, session_id, cp)

    monkeypatch.setattr(ownership_module, "save_curation_session", _fail_once_then_succeed)

    user = repo.create_user(firebase_uid="fb-retry", email="retry@example.com")

    with pytest.raises(RuntimeError):
        create_owned_curation_session(
            owner_id=user.id, session=_make_session(), topic="attempt-1", display_title=None,
            stage="curate", ownership_repo=repo, checkpointer=checkpointer,
        )
    failed_session_ids = [r.session_id for r in repo.list_owner_sessions(user.id)]
    assert len(failed_session_ids) == 1

    # A second, independent call -- never passed the failed session_id --
    # mints a brand-new one and succeeds cleanly.
    new_session_id = create_owned_curation_session(
        owner_id=user.id, session=_make_session(), topic="attempt-2", display_title=None,
        stage="curate", ownership_repo=repo, checkpointer=checkpointer,
    )
    assert new_session_id != failed_session_ids[0]
    state = get_owned_curation_state(new_session_id, repo, checkpointer)
    assert state is not None


# --- coordinator: deletion ordering and failure behavior (Part G items 4-5) ---

def test_normal_deletion_removes_both_ownership_and_checkpoint_state(repo, checkpointer):
    """Item 4."""
    user = repo.create_user(firebase_uid="fb-del", email="del@example.com")
    session_id = create_owned_curation_session(
        owner_id=user.id, session=_make_session(), topic="to-delete", display_title=None,
        stage="curate", ownership_repo=repo, checkpointer=checkpointer,
    )
    assert get_owned_curation_state(session_id, repo, checkpointer) is not None

    deleted = delete_owned_curation_session(session_id=session_id, ownership_repo=repo, checkpointer=checkpointer)
    assert deleted is True
    assert get_owner_id_for_session(session_id, repo) is None
    assert get_owned_curation_state(session_id, repo, checkpointer) is None


def test_checkpoint_delete_failure_leaves_the_session_unreachable(repo, checkpointer, monkeypatch):
    """Item 5: ownership is deleted FIRST -- even if the checkpoint
    delete step then fails, the session must already be unreachable
    through this module's own read path."""
    user = repo.create_user(firebase_uid="fb-cp-del-fail", email="cpdelfail@example.com")
    session_id = create_owned_curation_session(
        owner_id=user.id, session=_make_session(), topic="t", display_title=None,
        stage="curate", ownership_repo=repo, checkpointer=checkpointer,
    )

    import research_agent.curation_ownership as ownership_module

    def _failing_delete(sid, cp):
        raise RuntimeError("simulated checkpoint-delete failure")

    monkeypatch.setattr(ownership_module, "delete_curation_session", _failing_delete)

    with pytest.raises(RuntimeError, match="simulated checkpoint-delete failure"):
        delete_owned_curation_session(session_id=session_id, ownership_repo=repo, checkpointer=checkpointer)

    # Ownership is already gone -- the session is unreachable regardless
    # of the checkpoint-store failure that happened second.
    assert get_owner_id_for_session(session_id, repo) is None
    assert repo.get_owner_row(session_id) is None


# --- concurrency: independent session IDs under real simultaneous execution (Part G item 10) ---

def test_two_concurrent_creates_mint_independent_session_ids(repo, checkpointer):
    """A `threading.Barrier(2)` forces both threads to mint their
    session_id at the same instant (both wait, then both proceed
    together) -- a genuine concurrency proof, not two sequential calls
    that merely happen not to collide."""
    barrier = threading.Barrier(2)
    results: dict[int, str] = {}
    errors: list[BaseException] = []

    user = repo.create_user(firebase_uid="fb-concurrent", email="concurrent@example.com")

    import research_agent.curation_ownership as ownership_module
    real_mint = ownership_module.mint_session_id

    def _synchronized_mint() -> str:
        barrier.wait(timeout=10)
        return real_mint()

    def worker(index: int) -> None:
        try:
            sid = create_owned_curation_session(
                owner_id=user.id, session=_make_session(f"topic-{index}"), topic=f"topic-{index}",
                display_title=None, stage="curate", ownership_repo=repo, checkpointer=checkpointer,
            )
            results[index] = sid
        except BaseException as exc:  # noqa: BLE001 -- captured for the assertion below
            errors.append(exc)

    # Patch the module-level mint function so BOTH threads' session_id
    # generation is forced through the same barrier -- proving the
    # independence holds even when both truly overlap in time, not just
    # when they happen to run one after another.
    import unittest.mock as mock
    with mock.patch.object(ownership_module, "mint_session_id", _synchronized_mint):
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
            assert not t.is_alive()

    assert errors == [], f"concurrent creates raised: {errors}"
    assert len(results) == 2
    assert results[0] != results[1]
    # Both sessions independently exist and are correctly owned.
    for sid in results.values():
        assert get_owner_id_for_session(sid, repo) == user.id


# --- reconciliation (Part G items 6-9, 18) ---

def test_reconciliation_removes_an_expired_owner_without_checkpoint_orphan(repo, checkpointer, conn):
    """Item 6."""
    user = repo.create_user(firebase_uid="fb-recon-6", email="recon6@example.com")
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO curation_owners (session_id, owner_id, topic, stage, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            ("expired-owner-orphan", user.id, "topic", "curate", old_time, old_time),
        )
    conn.commit()

    report = run_reconciliation(
        conn, checkpointer, apply=True,
        owner_orphan_safety_window=timedelta(hours=1),
        checkpoint_orphan_safety_window=timedelta(hours=24),
    )
    assert report.owner_without_checkpoint_eligible == ["expired-owner-orphan"]
    assert report.removed_owner_rows == 1
    assert repo.get_owner_row("expired-owner-orphan") is None


def test_reconciliation_removes_an_expired_checkpoint_without_owner_orphan(repo, checkpointer, conn):
    """Item 7."""
    session = _make_session()
    from research_agent.curation_session import save_curation_session
    save_curation_session(session, "expired-checkpoint-orphan", checkpointer)

    # Force this checkpoint's own timestamp to read as far in the past --
    # simulated by passing a `now` far enough ahead, rather than actually
    # waiting real wall-clock time (no timing-only test).
    far_future = datetime.now(timezone.utc) + timedelta(days=2)
    report = run_reconciliation(
        conn, checkpointer, apply=True,
        owner_orphan_safety_window=timedelta(hours=1),
        checkpoint_orphan_safety_window=timedelta(hours=24),
        now=far_future,
    )
    assert report.checkpoint_without_owner_eligible == ["expired-checkpoint-orphan"]
    assert report.removed_checkpoints == 1

    from research_agent.curation_session import load_curation_session
    assert load_curation_session("expired-checkpoint-orphan", checkpointer) is None


def test_reconciliation_leaves_recent_in_flight_records_untouched(repo, checkpointer, conn):
    """Item 8: a freshly-created owner-without-checkpoint AND a
    freshly-created checkpoint-without-owner both survive a reconciliation
    run at the DEFAULT safety windows, evaluated at "now" (no artificial
    time travel) -- exactly the in-flight-create/delete race window this
    module's own docstring says must never be swept."""
    user = repo.create_user(firebase_uid="fb-recon-8", email="recon8@example.com")
    repo.create_owner_row(session_id="fresh-owner-orphan", owner_id=user.id, topic="t", display_title=None, stage="curate")

    session = _make_session()
    from research_agent.curation_session import save_curation_session
    save_curation_session(session, "fresh-checkpoint-orphan", checkpointer)

    report = run_reconciliation(
        conn, checkpointer, apply=True,
        owner_orphan_safety_window=timedelta(hours=1),
        checkpoint_orphan_safety_window=timedelta(hours=24),
    )
    assert report.owner_without_checkpoint_eligible == []
    assert report.checkpoint_without_owner_eligible == []
    assert report.removed_owner_rows == 0
    assert report.removed_checkpoints == 0
    # Both records genuinely still exist.
    assert repo.get_owner_row("fresh-owner-orphan") is not None
    from research_agent.curation_session import load_curation_session
    assert load_curation_session("fresh-checkpoint-orphan", checkpointer) is not None


def test_reconciliation_never_assigns_a_new_owner(repo, checkpointer, conn):
    """Item 9: after reconciliation removes a checkpoint-without-owner
    orphan, no curation_owners row is ever created for that session_id
    -- the orphan is deleted, never adopted."""
    session = _make_session()
    from research_agent.curation_session import save_curation_session
    save_curation_session(session, "never-adopt-me", checkpointer)

    far_future = datetime.now(timezone.utc) + timedelta(days=2)
    run_reconciliation(
        conn, checkpointer, apply=True,
        owner_orphan_safety_window=timedelta(hours=1),
        checkpoint_orphan_safety_window=timedelta(hours=24),
        now=far_future,
    )
    assert repo.get_owner_row("never-adopt-me") is None
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM curation_owners")
        (count,) = cur.fetchone()
    assert count == 0


def test_dry_run_reconciliation_writes_nothing(repo, checkpointer, conn):
    """Item 18: an eligible orphan of BOTH classes survives a dry-run
    untouched."""
    user = repo.create_user(firebase_uid="fb-dry-run", email="dryrun@example.com")
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO curation_owners (session_id, owner_id, topic, stage, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            ("dry-run-owner-orphan", user.id, "topic", "curate", old_time, old_time),
        )
    conn.commit()
    session = _make_session()
    from research_agent.curation_session import save_curation_session, load_curation_session
    save_curation_session(session, "dry-run-checkpoint-orphan", checkpointer)

    far_future = datetime.now(timezone.utc) + timedelta(days=2)
    report = run_reconciliation(
        conn, checkpointer, apply=False,
        owner_orphan_safety_window=timedelta(hours=1),
        checkpoint_orphan_safety_window=timedelta(hours=24),
        now=far_future,
    )
    assert report.mode == "dry-run"
    assert report.owner_without_checkpoint_eligible == ["dry-run-owner-orphan"]
    assert report.checkpoint_without_owner_eligible == ["dry-run-checkpoint-orphan"]
    assert report.removed_owner_rows == 0
    assert report.removed_checkpoints == 0

    # Nothing was actually removed.
    assert repo.get_owner_row("dry-run-owner-orphan") is not None
    assert load_curation_session("dry-run-checkpoint-orphan", checkpointer) is not None


# --- Part G item 17: database credentials never appear in errors, at the connection level ---

def test_connection_failure_error_output_never_contains_the_password(capsys):
    """A bad --database-url (unreachable host, real-looking credentials)
    must never echo the password back in the script's own error output."""
    import scripts.reconcile_curation_ownership as reconcile_module

    bad_url = "postgresql://realuser:realsecretpw123@127.0.0.1:1/nonexistent"
    with tempfile.TemporaryDirectory() as tmp:
        exit_code = reconcile_module.main([
            "--database-url", bad_url,
            "--checkpoint-db-path", str(Path(tmp) / "cp.sqlite"),
        ])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "realsecretpw123" not in captured.err
    assert "realsecretpw123" not in captured.out
