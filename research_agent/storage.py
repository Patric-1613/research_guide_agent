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


def init_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
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

    # Round 3, phase 4: bags are a separate persistence concept from the
    # round-1/2 `searches` table above — a bag is the *basket* from an
    # interactive triage session (research_agent/session.py), not a single
    # search's full ranked result set, and additionally carries the round
    # history (keywords_used per round) for provenance. Kept as its own
    # table rather than reusing `searches`: the two rows mean genuinely
    # different things (one ranked search vs. a curated, multi-round pick
    # set), and conflating them would force one or the other's columns to
    # be meaningless for half of the rows.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            topic TEXT NOT NULL,
            created_at TEXT NOT NULL,
            paper_ids TEXT NOT NULL,
            web_articles TEXT NOT NULL,
            rounds TEXT NOT NULL,
            summary TEXT NOT NULL,
            web_summary TEXT
        )
        """
    )
    conn.commit()
    return conn


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
    )


def save_search(
    conn: sqlite3.Connection,
    topic: str,
    paper_ids: list[str],
    scores: list[float],
    web_articles: list[dict] | None = None,
) -> tuple[int, str]:
    created_at = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO searches (topic, created_at, paper_ids, scores, summary, web_articles, web_summary) "
        "VALUES (?, ?, ?, ?, NULL, ?, NULL)",
        (topic, created_at, json.dumps(paper_ids), json.dumps(scores), json.dumps(web_articles or [])),
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
    row = conn.execute("SELECT * FROM searches WHERE id = ?", (search_id,)).fetchone()
    return _row_to_saved_search(row) if row else None


def list_searches(conn: sqlite3.Connection) -> list[SavedSearch]:
    rows = conn.execute("SELECT * FROM searches ORDER BY created_at DESC").fetchall()
    return [_row_to_saved_search(r) for r in rows]


@dataclass
class Bag:
    id: int
    name: str
    topic: str
    created_at: str
    paper_ids: list[str]
    web_articles: list[dict]
    rounds: list[dict]
    summary: dict
    web_summary: dict | None


def _row_to_bag(row: sqlite3.Row) -> Bag:
    return Bag(
        id=row["id"],
        name=row["name"],
        topic=row["topic"],
        created_at=row["created_at"],
        paper_ids=json.loads(row["paper_ids"]),
        web_articles=json.loads(row["web_articles"]),
        rounds=json.loads(row["rounds"]),
        summary=json.loads(row["summary"]),
        web_summary=json.loads(row["web_summary"]) if row["web_summary"] else None,
    )


def save_bag(
    conn: sqlite3.Connection,
    name: str,
    topic: str,
    paper_ids: list[str],
    web_articles: list[dict],
    rounds: list[dict],
    summary: dict,
    web_summary: dict | None = None,
) -> tuple[int, str]:
    """Persist a bag: the round-3 triage flow's basket, its round history
    (for provenance), and the summaries already generated for it at
    Summarize time (research_agent/session.py, api.py's /triage/summarize).
    Only called once the user explicitly chooses to save rather than
    discard — nothing here runs automatically."""
    created_at = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO bags (name, topic, created_at, paper_ids, web_articles, rounds, summary, web_summary) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            name, topic, created_at, json.dumps(paper_ids), json.dumps(web_articles),
            json.dumps(rounds), json.dumps(summary), json.dumps(web_summary) if web_summary is not None else None,
        ),
    )
    conn.commit()
    return cur.lastrowid, created_at


def get_bag(conn: sqlite3.Connection, bag_id: int) -> Bag | None:
    row = conn.execute("SELECT * FROM bags WHERE id = ?", (bag_id,)).fetchone()
    return _row_to_bag(row) if row else None


def list_bags(conn: sqlite3.Connection) -> list[Bag]:
    rows = conn.execute("SELECT * FROM bags ORDER BY created_at DESC").fetchall()
    return [_row_to_bag(r) for r in rows]


def delete_bag(conn: sqlite3.Connection, bag_id: int) -> bool:
    """SQLite-only half of bag deletion — returns whether a row was
    actually deleted. Callers (api.py) are responsible for also cleaning
    up the corresponding Chroma vectors; see
    paper_ids_referenced_by_other_bags below for why that isn't done
    unconditionally."""
    cur = conn.execute("DELETE FROM bags WHERE id = ?", (bag_id,))
    conn.commit()
    return cur.rowcount > 0


def paper_ids_referenced_by_other_bags(
    conn: sqlite3.Connection, paper_ids: list[str], exclude_bag_id: int | None = None
) -> set[str]:
    """Which of `paper_ids` still belong to some OTHER saved bag.

    The same real-world paper can end up in more than one bag (picked in
    two different sessions, or resurfacing under a different keyword).
    Before deleting a bag's Chroma vectors, callers must check this first —
    otherwise deleting bag A could silently corrupt bag B if they happen to
    share a paper, by ripping out the vector B's own /library-style lookup
    still depends on. Only ids with zero remaining references anywhere
    should actually have their Chroma vectors removed.
    """
    if not paper_ids:
        return set()
    candidates = set(paper_ids)
    referenced: set[str] = set()
    for row in conn.execute("SELECT id, paper_ids FROM bags").fetchall():
        if exclude_bag_id is not None and row["id"] == exclude_bag_id:
            continue
        referenced.update(json.loads(row["paper_ids"]))
        if candidates <= referenced:
            break  # every candidate already accounted for, no need to keep scanning
    return candidates & referenced
