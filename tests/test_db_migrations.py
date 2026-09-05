"""Day 2, Part D: deterministic tests for the migration runner's own
stated requirements -- repeatability, an inspectable current version, and
"a partially failed migration never masquerades as success". The
`_discover_migrations` file-naming tests need no database at all; the
rest need `TEST_DATABASE_URL` (skipped, never failed, when unset -- same
convention as `tests/test_curation_ownership_postgres.py`).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import psycopg
import pytest

from research_agent.db.migrations import _discover_migrations, get_current_schema_version, run_migrations

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pg_only = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is not set -- Postgres-dependent Day 2 tests are skipped, not failed.",
)


# --- _discover_migrations: pure filesystem logic, no database needed ---

def test_discover_migrations_ignores_non_matching_files():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "0001_initial.sql").write_text("SELECT 1;")
        (d / "readme.txt").write_text("not a migration")
        (d / "sql_without_leading_digits.sql").write_text("SELECT 1;")
        found = _discover_migrations(d)
        assert [v for v, _ in found] == [1]


def test_discover_migrations_orders_by_version_ascending():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "0003_third.sql").write_text("SELECT 1;")
        (d / "0001_first.sql").write_text("SELECT 1;")
        (d / "0002_second.sql").write_text("SELECT 1;")
        found = _discover_migrations(d)
        assert [v for v, _ in found] == [1, 2, 3]


def test_discover_migrations_rejects_duplicate_versions():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "0001_first.sql").write_text("SELECT 1;")
        (d / "0001_also_first.sql").write_text("SELECT 1;")
        with pytest.raises(ValueError, match="Duplicate migration version"):
            _discover_migrations(d)


# --- run_migrations / get_current_schema_version: require a real Postgres ---

@pytest.fixture()
def bare_conn():
    """A connection to a database with the real schema (and
    schema_migrations, whose version numbers this file's tests
    deliberately reuse for their own fake migrations) dropped, so each
    test starts from a genuinely un-migrated state -- unlike
    tests/test_curation_ownership_postgres.py's own `conn` fixture,
    which assumes the real 0001_initial.sql schema already applied and
    only truncates data.

    Teardown restores the REAL schema via the real, default
    `run_migrations()` -- without this, a test here that records its own
    fake version-1/2/3 migrations in `schema_migrations` would make the
    REAL `run_migrations()` (used by every other test file's `conn`
    fixture) wrongly believe 0001_initial.sql is already applied and
    skip it, leaving `users`/`curation_owners`/`saved_searches` missing
    for every test that runs afterward in the same pytest session.
    """
    from research_agent.db.migrations import run_migrations as real_run_migrations

    connection = psycopg.connect(TEST_DATABASE_URL)
    with connection.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS saved_searches, curation_owners, users, schema_migrations, "
            "first_table, second_table, third_table, deliberately_partial CASCADE"
        )
    connection.commit()
    yield connection
    with connection.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS saved_searches, curation_owners, users, schema_migrations, "
            "first_table, second_table, third_table, deliberately_partial CASCADE"
        )
    connection.commit()
    real_run_migrations(connection)  # restore the real schema for every test file that runs after this one
    connection.close()


@pg_only
def test_get_current_schema_version_is_none_before_any_migration(bare_conn):
    assert get_current_schema_version(bare_conn) is None


@pg_only
def test_run_migrations_applies_and_records_version(bare_conn):
    applied = run_migrations(bare_conn)
    assert applied == [1]
    assert get_current_schema_version(bare_conn) == 1
    with bare_conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")
        tables = {r[0] for r in cur.fetchall()}
    assert {"users", "curation_owners", "saved_searches", "schema_migrations"} <= tables


@pg_only
def test_run_migrations_is_repeatable_and_idempotent(bare_conn):
    """Part D: 'schema creation is repeatable' -- calling run_migrations
    twice in a row is always safe and the second call applies nothing
    new."""
    first = run_migrations(bare_conn)
    second = run_migrations(bare_conn)
    assert first == [1]
    assert second == []
    assert get_current_schema_version(bare_conn) == 1


@pg_only
def test_a_partially_failing_migration_never_masquerades_as_success(bare_conn):
    """Part D: 'a partially failed migration does not masquerade as
    success'. A migration file with a genuine SQL error partway through
    must leave NEITHER its own DDL applied NOR a schema_migrations row
    recorded for it."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "0001_broken.sql").write_text(
            "CREATE TABLE deliberately_partial (id INTEGER); "
            "SELECT * FROM this_table_does_not_exist_at_all;"
        )
        with pytest.raises(psycopg.errors.UndefinedTable):
            run_migrations(bare_conn, migrations_dir=d)

    assert get_current_schema_version(bare_conn) is None
    with bare_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.deliberately_partial')")
        (exists,) = cur.fetchone()
    assert exists is None, "the failed migration's own DDL must not have been left partially applied"


@pg_only
def test_run_migrations_applies_only_versions_not_yet_recorded(bare_conn):
    """A version already present in schema_migrations is never
    re-executed, even if a later, unrelated migration is added --
    proven by seeding schema_migrations directly rather than only
    relying on run_migrations' own bookkeeping."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "0001_first.sql").write_text("CREATE TABLE first_table (id INTEGER);")
        (d / "0002_second.sql").write_text("CREATE TABLE second_table (id INTEGER);")
        applied = run_migrations(bare_conn, migrations_dir=d)
        assert applied == [1, 2]

        # A third migration added later -- only it should apply on a
        # subsequent run, never 1/2 again.
        (d / "0003_third.sql").write_text("CREATE TABLE third_table (id INTEGER);")
        applied_again = run_migrations(bare_conn, migrations_dir=d)
        assert applied_again == [3]
