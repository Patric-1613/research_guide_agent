"""Day 2: the idempotent, versioned SQL migration runner.

Deliberately NOT Alembic -- this project has no ORM (raw parameterized
SQL throughout `storage.py`/`telemetry.py`/etc.), and the schema this
checkpoint needs (three tables, a handful of indexes) does not justify
adopting a new migration framework and its own dependency/config surface.
This is the smallest maintainable approach: plain `.sql` files under
`research_agent/db/migrations/`, named `NNNN_description.sql`, applied in
ascending numeric order, each inside its own transaction, tracked in a
`schema_migrations` table this module itself creates.

**Explicit, not automatic.** `run_migrations()` is never called from
`api_app/app.py`'s `lifespan()` or from any request path -- it runs only
when a caller (a script, or a test fixture standing up an isolated
database) calls it deliberately. This is intentional: "migrations run
explicitly, not unexpectedly on every request" is one of this
checkpoint's own stated requirements, not an oversight to fix later.

**A partially failed migration never masquerades as success.** Each
migration file's entire SQL text is executed inside ONE transaction that
also records the `schema_migrations` row -- either the whole file's DDL
AND its version-tracking row commit together, or (on any error) the
whole transaction rolls back and `run_migrations()` re-raises. There is
no path where a `schema_migrations` row exists for a version whose DDL
didn't fully apply, and no path where DDL applied but the tracking row is
missing.

**Rollback, documented honestly.** No down-migrations exist. This module
has no way to undo a migration once applied. In production (Cloud SQL),
"rollback" means restoring a point-in-time-recovery snapshot taken before
the migration ran -- exactly what
`docs/plans/public-multi-user-deployment-review.md` already establishes
for the rest of this project's data. This is a real, stated limitation,
not a gap this module tries to paper over.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_SCHEMA_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL
)
"""


def _discover_migrations(migrations_dir: Path) -> list[tuple[int, Path]]:
    """Every `NNNN_*.sql` file in `migrations_dir`, as `(version, path)`
    pairs sorted by version ascending. A filename not matching
    `NNNN_...sql` (four-or-more leading digits, an underscore) is
    ignored -- never guessed at or coerced -- so a stray non-migration
    file dropped into this directory can never be silently applied.
    Raises `ValueError` if two files claim the same version number
    (ambiguous application order is a real authoring error, not
    something to resolve by picking one arbitrarily)."""
    found: dict[int, Path] = {}
    for path in sorted(migrations_dir.glob("*.sql")):
        stem = path.stem
        digits = ""
        for ch in stem:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits or not stem[len(digits):].startswith("_"):
            continue
        version = int(digits)
        if version in found:
            raise ValueError(
                f"Duplicate migration version {version}: {found[version].name!r} and {path.name!r} "
                "both claim it -- migration versions must be unique."
            )
        found[version] = path
    return sorted(found.items())


def get_current_schema_version(conn: "psycopg.Connection") -> int | None:
    """The highest applied migration version, or None if
    `schema_migrations` doesn't exist yet or is empty -- never raises for
    either of those two "nothing applied yet" cases, only for a genuine
    connection/query error."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_regclass('public.schema_migrations') IS NOT NULL"
        )
        (table_exists,) = cur.fetchone()
        if not table_exists:
            return None
        cur.execute("SELECT MAX(version) FROM schema_migrations")
        (version,) = cur.fetchone()
        return version


def run_migrations(conn: "psycopg.Connection", migrations_dir: Path = MIGRATIONS_DIR) -> list[int]:
    """Applies every migration in `migrations_dir` whose version is not
    already present in `schema_migrations`, in ascending order. Returns
    the list of newly applied version numbers (empty if the schema was
    already current -- calling this repeatedly is always safe).

    Each migration is applied in its own transaction (this function
    manages `conn`'s transaction boundaries explicitly via
    `conn.commit()`/`conn.rollback()` rather than relying on `conn`'s
    ambient autocommit state, so this works whether the caller's
    connection has autocommit on or off) -- a failure partway through one
    migration's SQL rolls back that migration's own DDL and its own
    tracking-row insert together, and this function then re-raises
    immediately without attempting any later migration. Migrations
    already recorded as applied before the failure are untouched and
    stay applied.
    """
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_MIGRATIONS_TABLE_SQL)
    conn.commit()

    applied: list[int] = []
    for version, path in _discover_migrations(migrations_dir):
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM schema_migrations WHERE version = %s", (version,))
            already_applied = cur.fetchone() is not None
        if already_applied:
            continue
        sql = path.read_text()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (%s, %s)",
                    (version, datetime.now(timezone.utc)),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        applied.append(version)
    return applied
