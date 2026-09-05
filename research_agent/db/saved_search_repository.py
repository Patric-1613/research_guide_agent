"""Day 2: the saved-search repository boundary.

`SavedSearchRepository` is a `typing.Protocol` two implementations
satisfy identically: `SqliteSavedSearchRepository` (a thin wrapper over
the existing, unmodified `research_agent/storage.py` functions -- local
development and tests) and `PostgresSavedSearchRepository` (a fresh
Postgres table, production only). Neither implementation is wired into
any existing route or service in this checkpoint -- see this package's
own `__init__.py` docstring and
`research_agent/curation_ownership.py`'s module docstring for why that
wiring is deferred, not an oversight.

Both implementations return `research_agent.storage.SavedSearch`
instances -- one shared result shape regardless of backend, so a future
caller that switches which repository it uses never has to branch on the
result type.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Protocol

import psycopg
from psycopg.types.json import Jsonb

import research_agent.storage as storage
from research_agent.storage import SavedSearch


class SavedSearchRepository(Protocol):
    def save(
        self, *, owner_id: str, topic: str, paper_ids: list[str], scores: list[float],
        web_articles: list[dict] | None = None,
    ) -> SavedSearch: ...

    def list_by_owner(self, owner_id: str, *, limit: int = storage.DEFAULT_LIST_LIMIT) -> list[SavedSearch]: ...


class SqliteSavedSearchRepository:
    """Local-development / test backend -- delegates entirely to
    `research_agent.storage`'s existing, unmodified functions. `conn` is
    a caller-supplied `sqlite3.Connection` (matching every other SQLite
    call site's own per-request-connection convention); this class owns
    no connection lifecycle of its own."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def save(
        self, *, owner_id: str, topic: str, paper_ids: list[str], scores: list[float],
        web_articles: list[dict] | None = None,
    ) -> SavedSearch:
        search_id, _created_at = storage.save_search(
            self._conn, topic, paper_ids, scores, web_articles=web_articles, owner_id=owner_id,
        )
        result = storage.get_search(self._conn, search_id)
        assert result is not None  # just inserted under the same connection/transaction
        return result

    def list_by_owner(self, owner_id: str, *, limit: int = storage.DEFAULT_LIST_LIMIT) -> list[SavedSearch]:
        return storage.list_searches_by_owner(self._conn, owner_id, limit=limit)


_SAVED_SEARCH_SELECT_COLUMNS = (
    "id, owner_id, topic, created_at, paper_ids, scores, summary, web_articles, web_summary"
)


def _pg_row_to_saved_search(row: tuple) -> SavedSearch:
    id_, owner_id, topic, created_at, paper_ids, scores, summary, web_articles, web_summary = row
    return SavedSearch(
        id=id_,
        topic=topic,
        # psycopg returns a real datetime for TIMESTAMPTZ; isoformat()
        # matches the ISO-8601 string shape storage.py's SQLite side
        # already returns from its own TEXT column, so SavedSearch.
        # created_at is the same type (str) regardless of backend.
        created_at=created_at.isoformat() if isinstance(created_at, datetime) else created_at,
        # JSONB columns are already decoded to native Python objects by
        # psycopg -- no json.loads() needed here, unlike the SQLite side
        # (whose TEXT columns genuinely hold JSON strings).
        paper_ids=paper_ids,
        scores=scores,
        summary=summary,
        web_articles=web_articles if web_articles is not None else [],
        web_summary=web_summary,
        owner_id=str(owner_id),
    )


class PostgresSavedSearchRepository:
    """Production backend -- a fresh `saved_searches` Postgres table
    (`research_agent/db/migrations/0001_initial.sql`), never the
    migrated/converted SQLite `searches` table. `conn` is a
    caller-supplied `psycopg.Connection` (from a
    `psycopg_pool.ConnectionPool`); this class owns no connection
    lifecycle of its own, matching `SqliteSavedSearchRepository`'s
    convention exactly.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def save(
        self, *, owner_id: str, topic: str, paper_ids: list[str], scores: list[float],
        web_articles: list[dict] | None = None,
    ) -> SavedSearch:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO saved_searches (owner_id, topic, paper_ids, scores, web_articles) "
                "VALUES (%s, %s, %s, %s, %s) "
                f"RETURNING {_SAVED_SEARCH_SELECT_COLUMNS}",
                (
                    uuid.UUID(owner_id), topic, Jsonb(paper_ids), Jsonb(scores),
                    Jsonb(web_articles) if web_articles is not None else None,
                ),
            )
            row = cur.fetchone()
        self._conn.commit()
        return _pg_row_to_saved_search(row)

    def list_by_owner(self, owner_id: str, *, limit: int = storage.DEFAULT_LIST_LIMIT) -> list[SavedSearch]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT {_SAVED_SEARCH_SELECT_COLUMNS} FROM saved_searches "
                "WHERE owner_id = %s ORDER BY created_at DESC, id DESC LIMIT %s",
                (uuid.UUID(owner_id), limit),
            )
            rows = cur.fetchall()
        return [_pg_row_to_saved_search(r) for r in rows]
