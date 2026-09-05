"""Day 2: PostgreSQL `users` / `curation_owners` repository.

`PostgresOwnershipRepository` is a thin, direct wrapper over the schema
`research_agent/db/migrations/0001_initial.sql` creates -- no ORM, raw
parameterized SQL, matching this project's own established convention
(`research_agent/storage.py`, `research_agent/telemetry.py`). Every
method takes an already-open `psycopg.Connection` (from a
`psycopg_pool.ConnectionPool`, see `pool.py`) rather than owning
connection lifecycle itself -- the caller (a service, a script, a test)
decides transaction boundaries, matching `storage.py`'s own
per-request-connection convention.

**Never invents ownership.** No method here ever assigns a
`curation_owners` row a NEW `owner_id` for a session that didn't already
have one -- see `research_agent/curation_ownership.py`'s own module
docstring for the create/delete ordering this repository is a building
block for, and `scripts/reconcile_curation_ownership.py` for the
orphan-cleanup policy that explicitly never repairs an orphan by
adopting it.

**`id`/`owner_id` are Python-generated UUIDs** (`uuid.uuid4()`), not a
database-side default -- see `0001_initial.sql`'s own comment for why.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

import psycopg


class DuplicateFirebaseUidError(Exception):
    """Raised by `create_user` when `firebase_uid` already exists --
    `firebase_uid` is the immutable provider identity and must be unique
    (see 0001_initial.sql's own unique index); this is a distinct,
    catchable exception rather than letting a raw `psycopg.errors.
    UniqueViolation` leak past this module's own boundary."""


@dataclass(frozen=True)
class UserRecord:
    id: uuid.UUID
    firebase_uid: str
    email: str
    display_name: str | None
    approved: bool
    disabled: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class CurationOwnerRecord:
    session_id: str
    owner_id: uuid.UUID
    topic: str
    display_title: str | None
    stage: str
    created_at: datetime
    updated_at: datetime


_USER_COLUMNS = "id, firebase_uid, email, display_name, approved, disabled, created_at, updated_at"
_OWNER_COLUMNS = "session_id, owner_id, topic, display_title, stage, created_at, updated_at"


def _row_to_user(row: tuple) -> UserRecord:
    return UserRecord(
        id=row[0], firebase_uid=row[1], email=row[2], display_name=row[3],
        approved=row[4], disabled=row[5], created_at=row[6], updated_at=row[7],
    )


def _row_to_owner(row: tuple) -> CurationOwnerRecord:
    return CurationOwnerRecord(
        session_id=row[0], owner_id=row[1], topic=row[2], display_title=row[3],
        stage=row[4], created_at=row[5], updated_at=row[6],
    )


class PostgresOwnershipRepository:
    """Every method commits its own connection immediately after its own
    write -- matching `research_agent/storage.py`'s own
    one-write-one-commit convention -- rather than assuming the caller
    manages a transaction spanning multiple repository calls. A caller
    that genuinely needs several of these calls to succeed or fail
    together (the create/delete coordinator in
    `research_agent/curation_ownership.py`) composes them at that higher
    layer instead, exactly as documented in this module's own docstring.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    # --- users -----------------------------------------------------

    def create_user(
        self, *, firebase_uid: str, email: str, display_name: str | None = None,
    ) -> UserRecord:
        """New users always start `approved=False, disabled=False` --
        the schema's own column defaults, never overridable through this
        method's signature (there is deliberately no `approved=` kwarg
        here: approval is a separate, later, explicit administrative
        action, never something a caller can grant at creation time)."""
        user_id = uuid.uuid4()
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO users (id, firebase_uid, email, display_name) "
                    f"VALUES (%s, %s, %s, %s) RETURNING {_USER_COLUMNS}",
                    (user_id, firebase_uid, email, display_name),
                )
                row = cur.fetchone()
            self._conn.commit()
        except psycopg.errors.UniqueViolation:
            self._conn.rollback()
            raise DuplicateFirebaseUidError(firebase_uid) from None
        return _row_to_user(row)

    def get_user_by_firebase_uid(self, firebase_uid: str) -> UserRecord | None:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE firebase_uid = %s", (firebase_uid,))
            row = cur.fetchone()
        return _row_to_user(row) if row else None

    def get_user_by_id(self, user_id: uuid.UUID) -> UserRecord | None:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
        return _row_to_user(row) if row else None

    # --- curation_owners --------------------------------------------

    def create_owner_row(
        self, *, session_id: str, owner_id: uuid.UUID, topic: str,
        display_title: str | None, stage: str,
    ) -> CurationOwnerRecord:
        """`session_id` is caller-supplied (minted by
        `research_agent.curation_ownership`'s coordinator, never by this
        repository) -- this method only ever inserts the row it's given;
        it never generates a session_id of its own."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO curation_owners (session_id, owner_id, topic, display_title, stage) "
                f"VALUES (%s, %s, %s, %s, %s) RETURNING {_OWNER_COLUMNS}",
                (session_id, owner_id, topic, display_title, stage),
            )
            row = cur.fetchone()
        self._conn.commit()
        return _row_to_owner(row)

    def get_owner_row(self, session_id: str) -> CurationOwnerRecord | None:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT {_OWNER_COLUMNS} FROM curation_owners WHERE session_id = %s", (session_id,))
            row = cur.fetchone()
        return _row_to_owner(row) if row else None

    def delete_owner_row(self, session_id: str) -> bool:
        """Idempotent: deleting an already-absent session_id is a safe
        no-op (returns False), not an error -- same "already released is
        the same safe outcome" posture `research_agent/leases.py`'s own
        `release_lease` establishes."""
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM curation_owners WHERE session_id = %s", (session_id,))
            deleted = cur.rowcount > 0
        self._conn.commit()
        return deleted

    def list_owner_sessions(self, owner_id: uuid.UUID, *, limit: int = 100) -> list[CurationOwnerRecord]:
        """Newest-first, `id`-free stable tie-break: `session_id` itself
        (a uuid4 hex, so this is an arbitrary-but-deterministic
        tie-break, same role `storage.py`'s own `id DESC` tie-break
        plays for same-second saves)."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT {_OWNER_COLUMNS} FROM curation_owners "
                "WHERE owner_id = %s ORDER BY created_at DESC, session_id DESC LIMIT %s",
                (owner_id, limit),
            )
            rows = cur.fetchall()
        return [_row_to_owner(r) for r in rows]

    def count_owner_sessions(self, owner_id: uuid.UUID, *, stage: str | None = None) -> int:
        """Count of this owner's sessions, optionally scoped to one
        `stage` -- the "active session" business definition (which
        stage(s) count as active) is deliberately NOT decided here; a
        caller passes whichever `stage` value it means, or omits it to
        count every session regardless of stage."""
        with self._conn.cursor() as cur:
            if stage is None:
                cur.execute("SELECT COUNT(*) FROM curation_owners WHERE owner_id = %s", (owner_id,))
            else:
                cur.execute(
                    "SELECT COUNT(*) FROM curation_owners WHERE owner_id = %s AND stage = %s",
                    (owner_id, stage),
                )
            (count,) = cur.fetchone()
        return count
