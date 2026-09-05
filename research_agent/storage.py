"""Phase 7: SQLite persistence for saved searches.

Only the search's identity is stored here (topic, timestamp, which
paper_ids belong to it, their relevance scores, and the generated summary
once /summarize has run) — not the papers' own content. Paper content
(title, abstract, authors, ...) is already persisted in Chroma as of phase
3; duplicating it here would just be a second copy to keep in sync. This
table's paper_ids are the join key back to Chroma at read time.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.sqlite"

# The columns _row_to_saved_search() needs, named explicitly instead of
# `SELECT *` so row reconstruction does not silently depend on the table's
# column order or on columns added later for unrelated reasons.
_SAVED_SEARCH_COLUMNS = (
    "id, topic, created_at, paper_ids, scores, summary, web_articles, web_summary, owner_id"
)

# Bounds for a saved-search listing. The listing is unpaginated, so an
# unbounded query would grow without limit as history accumulates; 100 is
# a generous default for a single-user library view and 500 a hard cap.
DEFAULT_LIST_LIMIT = 100
MAX_LIST_LIMIT = 500


def init_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # WAL lets concurrent readers proceed while a writer holds the lock,
    # instead of the default rollback-journal mode's whole-file lock — the
    # right complement to a per-request connection pattern under FastAPI's
    # multi-threaded request handling, where overlapping requests are the
    # normal case, not an edge case.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            created_at TEXT NOT NULL,
            paper_ids TEXT NOT NULL,
            scores TEXT NOT NULL,
            summary TEXT,
            web_articles TEXT,
            web_summary TEXT
        )
        """
    )
    # Round-2 enhancement 5: CREATE TABLE IF NOT EXISTS only applies the new
    # columns to a brand-new table — a database file created before this
    # enhancement already has a `searches` table without them. SQLite has no
    # "ADD COLUMN IF NOT EXISTS", so check first rather than relying on
    # catching the duplicate-column error.
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(searches)")}
    for column in ("web_articles", "web_summary"):
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE searches ADD COLUMN {column} TEXT")
    # Day 2 (public multi-user deployment foundation, see
    # docs/plans/public-multi-user-deployment-review.md): additive,
    # nullable owner_id -- schema PARITY with the production Postgres
    # saved_searches table, not a behavior change. NULL means "no owner
    # recorded" (every row saved before this column existed, and every
    # row saved via the existing, unmodified save_search() call sites
    # that don't pass owner_id) -- never backfilled, never treated as
    # "belongs to everyone." Same idempotent-ALTER-TABLE pattern as
    # web_articles/web_summary above, applied identically to a database
    # created before or after this column existed.
    if "owner_id" not in existing_columns:
        conn.execute("ALTER TABLE searches ADD COLUMN owner_id TEXT")
    # Backs list_searches_by_owner()'s `WHERE owner_id = ? ORDER BY
    # created_at DESC, id DESC` -- same "don't fall back to a full scan
    # as history grows" reasoning as idx_searches_created_at_id below.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_searches_owner_created_at_id "
        "ON searches(owner_id, created_at DESC, id DESC)"
    )
    # Backs the listing's `ORDER BY created_at DESC, id DESC LIMIT ?` so it
    # does not fall back to a full scan and sort as history grows. IF NOT
    # EXISTS keeps this idempotent and safe on a database created before
    # the index existed; no data is rewritten.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_searches_created_at_id "
        "ON searches(created_at DESC, id DESC)"
    )
    conn.commit()
    return conn


def get_db_connection(path: Path = DB_PATH):
    """FastAPI dependency: opens a fresh connection for one request and
    closes it when the request finishes, instead of every request sharing
    a single long-lived connection object across FastAPI's threadpool.
    Schema creation/migration (init_db) must already have run once at
    startup — this intentionally skips re-running CREATE TABLE/ALTER TABLE
    checks on every request.
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # With WAL, concurrent writers no longer corrupt each other, but one can
    # still momentarily block another — busy_timeout makes a blocked writer
    # wait and retry internally instead of immediately raising
    # "database is locked" back to the request.
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
    finally:
        conn.close()


@dataclass
class SavedSearch:
    id: int
    topic: str
    created_at: str
    paper_ids: list[str]
    scores: list[float]
    summary: dict | None
    web_articles: list[dict]
    web_summary: dict | None
    # Day 2: additive, nullable -- see init_db()'s own comment on the
    # ALTER TABLE that adds this column. None for every row saved before
    # this field existed, or saved without an owner_id argument.
    owner_id: str | None = None


def _row_to_saved_search(row: sqlite3.Row) -> SavedSearch:
    return SavedSearch(
        id=row["id"],
        topic=row["topic"],
        created_at=row["created_at"],
        paper_ids=json.loads(row["paper_ids"]),
        scores=json.loads(row["scores"]),
        summary=json.loads(row["summary"]) if row["summary"] else None,
        web_articles=json.loads(row["web_articles"]) if row["web_articles"] else [],
        web_summary=json.loads(row["web_summary"]) if row["web_summary"] else None,
        # sqlite3.Row supports `in` since it behaves like a mapping over
        # its own column names; a row selected via a column list that
        # predates this field (impossible today, since
        # _SAVED_SEARCH_COLUMNS already includes it, but kept defensive
        # for any future direct SELECT that doesn't) falls back to None
        # rather than raising.
        owner_id=row["owner_id"] if "owner_id" in row.keys() else None,
    )


def save_search(
    conn: sqlite3.Connection,
    topic: str,
    paper_ids: list[str],
    scores: list[float],
    web_articles: list[dict] | None = None,
    owner_id: str | None = None,
) -> tuple[int, str]:
    """`owner_id` is optional and defaults to None (unowned) -- every
    existing call site that doesn't pass it keeps storing NULL, exactly
    the pre-Day-2 behavior. This is the ONLY place a saved search's
    owner is recorded; there is no separate "claim an existing search"
    path."""
    created_at = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO searches (topic, created_at, paper_ids, scores, summary, web_articles, web_summary, owner_id) "
        "VALUES (?, ?, ?, ?, NULL, ?, NULL, ?)",
        (topic, created_at, json.dumps(paper_ids), json.dumps(scores), json.dumps(web_articles or []), owner_id),
    )
    conn.commit()
    return cur.lastrowid, created_at


def update_summary(conn: sqlite3.Connection, search_id: int, summary: dict) -> None:
    conn.execute("UPDATE searches SET summary = ? WHERE id = ?", (json.dumps(summary), search_id))
    conn.commit()


def update_web_summary(conn: sqlite3.Connection, search_id: int, web_summary: dict) -> None:
    conn.execute("UPDATE searches SET web_summary = ? WHERE id = ?", (json.dumps(web_summary), search_id))
    conn.commit()


def get_search(conn: sqlite3.Connection, search_id: int) -> SavedSearch | None:
    row = conn.execute(
        f"SELECT {_SAVED_SEARCH_COLUMNS} FROM searches WHERE id = ?", (search_id,)
    ).fetchone()
    return _row_to_saved_search(row) if row else None


def list_searches(
    conn: sqlite3.Connection, limit: int = DEFAULT_LIST_LIMIT
) -> list[SavedSearch]:
    """Newest saved searches first, at most `limit` rows. `id DESC` is the
    tie-break so rows sharing a `created_at` (same-second saves) have a
    stable, deterministic order rather than whatever the scan yields.

    Unfiltered by owner -- this is the existing, unmodified pre-Day-2
    behavior (every saved search, regardless of owner_id), kept exactly
    as-is since no existing call site is switched to owner-scoped
    listing in this checkpoint. See list_searches_by_owner() below for
    the new, additive, owner-scoped counterpart.
    """
    rows = conn.execute(
        f"SELECT {_SAVED_SEARCH_COLUMNS} FROM searches "
        "ORDER BY created_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_saved_search(r) for r in rows]


def list_searches_by_owner(
    conn: sqlite3.Connection, owner_id: str, limit: int = DEFAULT_LIST_LIMIT
) -> list[SavedSearch]:
    """Day 2: the owner-scoped counterpart to list_searches() above --
    additive, not a replacement; no existing call site uses this yet
    (route/service wiring is later work, see
    docs/plans/public-multi-user-deployment-review.md's Day 4). Same
    newest-first, id-tie-broken ordering and the same DEFAULT_LIST_LIMIT/
    MAX_LIST_LIMIT contract as list_searches(). A row with owner_id IS
    NULL never matches any real owner_id value in SQL, so unowned
    (legacy, or saved without an owner) rows are correctly excluded
    without needing a special-cased `IS NULL` branch here.
    """
    rows = conn.execute(
        f"SELECT {_SAVED_SEARCH_COLUMNS} FROM searches "
        "WHERE owner_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
        (owner_id, limit),
    ).fetchall()
    return [_row_to_saved_search(r) for r in rows]
