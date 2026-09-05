"""Deterministic tests for the SQLite persistence layer — no mocking needed,
just a temp DB file per test."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3

from research_agent.storage import (
    DEFAULT_LIST_LIMIT,
    MAX_LIST_LIMIT,
    _SAVED_SEARCH_COLUMNS,
    get_db_connection,
    get_search,
    init_db,
    list_searches,
    list_searches_by_owner,
    save_search,
    update_summary,
    update_web_summary,
)


def _index_names(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA index_list(searches)")}


def test_save_and_get_search_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        search_id, created_at = save_search(conn, "test topic", ["p1", "p2"], [0.9, 0.8])

        saved = get_search(conn, search_id)
        assert saved.topic == "test topic"
        assert saved.created_at == created_at
        assert saved.paper_ids == ["p1", "p2"]
        assert saved.scores == [0.9, 0.8]
        assert saved.summary is None


def test_get_search_missing_id_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        assert get_search(conn, 999) is None


def test_update_summary_persists():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        search_id, _ = save_search(conn, "topic", ["p1"], [0.5])

        summary = {"themes": [], "gaps_and_disagreements": "none", "skipped_paper_ids": []}
        update_summary(conn, search_id, summary)

        saved = get_search(conn, search_id)
        assert saved.summary == summary


def test_save_search_without_web_articles_defaults_to_empty_list():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        search_id, _ = save_search(conn, "topic", ["p1"], [0.5])  # no web_articles arg — old call shape
        saved = get_search(conn, search_id)
        assert saved.web_articles == []
        assert saved.web_summary is None


def test_save_search_persists_web_articles():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        articles = [{"title": "A", "url": "https://x.com/a", "snippet": "s", "published_date": None, "source_domain": "x.com"}]
        search_id, _ = save_search(conn, "topic", ["p1"], [0.5], web_articles=articles)
        saved = get_search(conn, search_id)
        assert saved.web_articles == articles


def test_update_web_summary_persists():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        search_id, _ = save_search(conn, "topic", ["p1"], [0.5])
        web_summary = {"synthesis": "some synthesis", "cited_urls": ["https://x.com/a"]}
        update_web_summary(conn, search_id, web_summary)
        saved = get_search(conn, search_id)
        assert saved.web_summary == web_summary


def test_init_db_migrates_pre_existing_database_missing_web_columns():
    """A database file created before round-2 enhancement 5 has a
    `searches` table without web_articles/web_summary — init_db must add
    them (via ALTER TABLE) rather than erroring on the next save_search."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.sqlite"
        legacy_conn = sqlite3.connect(path)
        legacy_conn.execute(
            """
            CREATE TABLE searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                created_at TEXT NOT NULL,
                paper_ids TEXT NOT NULL,
                scores TEXT NOT NULL,
                summary TEXT
            )
            """
        )
        legacy_conn.commit()
        legacy_conn.close()

        conn = init_db(path)  # must not raise
        search_id, _ = save_search(conn, "topic", ["p1"], [0.5], web_articles=[{"title": "A"}])
        saved = get_search(conn, search_id)
        assert saved.web_articles == [{"title": "A"}]


def test_list_searches_orders_newest_first():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        id1, _ = save_search(conn, "first", ["a"], [1.0])
        id2, _ = save_search(conn, "second", ["b"], [1.0])

        results = list_searches(conn)
        assert [s.id for s in results] == [id2, id1]


def test_list_searches_never_returns_more_than_limit():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        ids = [save_search(conn, f"t{i}", ["a"], [1.0])[0] for i in range(5)]

        results = list_searches(conn, limit=3)
        assert [s.id for s in results] == list(reversed(ids))[:3]  # 3 newest, newest first


def test_list_searches_tie_breaks_equal_timestamps_by_descending_id():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        # Force an identical created_at across rows (same-second saves) so
        # only the id DESC tie-break decides the order.
        ts = "2026-01-01T00:00:00+00:00"
        for topic in ("a", "b", "c"):
            conn.execute(
                "INSERT INTO searches (topic, created_at, paper_ids, scores) VALUES (?, ?, ?, ?)",
                (topic, ts, "[]", "[]"),
            )
        conn.commit()

        results = list_searches(conn)
        assert [s.id for s in results] == [3, 2, 1]


def test_list_searches_default_limit_constant_is_100():
    assert DEFAULT_LIST_LIMIT == 100
    assert MAX_LIST_LIMIT == 500


def test_get_search_and_list_searches_ignore_an_unrelated_extra_column():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        search_id, _ = save_search(conn, "topic", ["p1"], [0.5])
        # A column added later for an unrelated reason must not change row
        # reconstruction, because the queries name their columns explicitly.
        conn.execute("ALTER TABLE searches ADD COLUMN unrelated_extra TEXT")
        conn.commit()

        assert get_search(conn, search_id).topic == "topic"
        assert [s.id for s in list_searches(conn)] == [search_id]


def test_init_db_creates_the_created_at_index_on_a_new_database():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        assert "idx_searches_created_at_id" in _index_names(conn)


def test_init_db_creates_the_created_at_index_on_a_legacy_database():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.sqlite"
        legacy = sqlite3.connect(path)
        legacy.execute(
            """
            CREATE TABLE searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                created_at TEXT NOT NULL,
                paper_ids TEXT NOT NULL,
                scores TEXT NOT NULL,
                summary TEXT
            )
            """
        )
        legacy.execute(
            "INSERT INTO searches (topic, created_at, paper_ids, scores) VALUES (?, ?, ?, ?)",
            ("old", "2020-01-01T00:00:00+00:00", '["p1"]', "[0.5]"),
        )
        legacy.commit()
        legacy.close()

        conn = init_db(path)
        assert "idx_searches_created_at_id" in _index_names(conn)


def test_init_db_is_idempotent_across_repeated_calls():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.sqlite"
        init_db(path).close()
        init_db(path).close()
        conn = init_db(path)  # third call must not raise
        assert list(_index_names(conn)).count("idx_searches_created_at_id") == 1


def test_legacy_rows_deserialize_with_web_field_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.sqlite"
        legacy = sqlite3.connect(path)
        legacy.execute(
            """
            CREATE TABLE searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                created_at TEXT NOT NULL,
                paper_ids TEXT NOT NULL,
                scores TEXT NOT NULL,
                summary TEXT
            )
            """
        )
        legacy.execute(
            "INSERT INTO searches (topic, created_at, paper_ids, scores) VALUES (?, ?, ?, ?)",
            ("old", "2020-01-01T00:00:00+00:00", '["p1"]', "[0.5]"),
        )
        legacy.commit()
        legacy.close()

        conn = init_db(path)
        legacy_row = get_search(conn, 1)
        assert legacy_row.web_articles == []
        assert legacy_row.web_summary is None
        assert [s.id for s in list_searches(conn)] == [1]


def test_list_searches_query_plan_uses_the_created_at_index():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        for i in range(5):
            save_search(conn, f"t{i}", ["a"], [1.0])
        plan = "\n".join(
            str(tuple(r))
            for r in conn.execute(
                f"EXPLAIN QUERY PLAN SELECT {_SAVED_SEARCH_COLUMNS} FROM searches "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (10,),
            )
        )
        assert "idx_searches_created_at_id" in plan


def test_get_db_connection_yields_working_connection_and_closes_it_after():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.sqlite"
        init_db(path).close()

        gen = get_db_connection(path)
        conn = next(gen)
        conn.execute("SELECT 1")  # works while the generator is still suspended at yield

        try:
            next(gen)  # drives the generator past yield, running the finally: conn.close()
            assert False, "expected StopIteration once the generator is exhausted"
        except StopIteration:
            pass

        try:
            conn.execute("SELECT 1")
            assert False, "connection should be closed after the generator finished"
        except sqlite3.ProgrammingError:
            pass


def test_concurrent_requests_via_per_request_connections_do_not_corrupt_storage():
    # Simulates FastAPI's threadpool handing each request its own connection
    # (the fix for storage.py's old single-shared-connection pattern) —
    # N threads each open a fresh connection via get_db_connection and write
    # concurrently; every write must land, with no corruption or lost rows.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.sqlite"
        init_db(path).close()

        errors: list[Exception] = []
        n_threads = 20

        def worker(i: int) -> None:
            try:
                gen = get_db_connection(path)
                conn = next(gen)
                save_search(conn, f"topic-{i}", [f"p{i}"], [0.5])
            except Exception as exc:  # noqa: BLE001 - captured for the assertion below
                errors.append(exc)
            finally:
                try:
                    next(gen)
                except StopIteration:
                    pass

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"concurrent writes raised: {errors}"

        conn = init_db(path)
        results = list_searches(conn)
        assert len(results) == n_threads
        assert {s.topic for s in results} == {f"topic-{i}" for i in range(n_threads)}


# --- Day 2 (public multi-user deployment foundation, see
# docs/plans/public-multi-user-deployment-review.md): additive owner_id
# column -- Part G, test 15 ("legacy SQLite rows without owner_id still
# deserialize correctly") ---

def test_init_db_migrates_pre_existing_database_missing_owner_id_column():
    """A database file created before Day 2 has a `searches` table with
    no `owner_id` column at all (not even NULL-valued -- the column
    itself is absent) -- init_db must add it (via ALTER TABLE) rather
    than erroring on the next save_search/list_searches_by_owner call.
    Same template as test_init_db_migrates_pre_existing_database_missing_
    web_columns above, extended to also predate web_articles/web_summary
    -- the oldest possible schema shape."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.sqlite"
        legacy_conn = sqlite3.connect(path)
        legacy_conn.execute(
            """
            CREATE TABLE searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                created_at TEXT NOT NULL,
                paper_ids TEXT NOT NULL,
                scores TEXT NOT NULL,
                summary TEXT
            )
            """
        )
        # A row saved under the pre-Day-2 (and pre-round-2) schema --
        # genuinely no owner_id value was ever written for it, not even
        # NULL, since the column didn't exist at insert time.
        legacy_conn.execute(
            "INSERT INTO searches (topic, created_at, paper_ids, scores, summary) VALUES (?, ?, ?, ?, ?)",
            ("legacy topic", "2020-01-01T00:00:00+00:00", '["p1"]', "[0.5]", None),
        )
        legacy_conn.commit()
        legacy_conn.close()

        conn = init_db(path)  # must not raise
        results = list_searches(conn)
        assert len(results) == 1
        assert results[0].topic == "legacy topic"
        assert results[0].owner_id is None

        # The new column also fully supports the normal write path
        # against this now-migrated database.
        search_id, _ = save_search(conn, "new topic", ["p2"], [0.9], owner_id="owner-x")
        saved = get_search(conn, search_id)
        assert saved.owner_id == "owner-x"


def test_save_search_without_owner_id_defaults_to_none():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        search_id, _ = save_search(conn, "topic", ["p1"], [0.5])
        saved = get_search(conn, search_id)
        assert saved.owner_id is None


def test_save_search_persists_owner_id():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        search_id, _ = save_search(conn, "topic", ["p1"], [0.5], owner_id="owner-a")
        saved = get_search(conn, search_id)
        assert saved.owner_id == "owner-a"


def test_list_searches_by_owner_returns_only_that_owners_rows():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        id_a1, _ = save_search(conn, "a1", ["p1"], [0.5], owner_id="owner-a")
        save_search(conn, "b1", ["p2"], [0.5], owner_id="owner-b")
        id_a2, _ = save_search(conn, "a2", ["p3"], [0.5], owner_id="owner-a")
        save_search(conn, "unowned", ["p4"], [0.5])  # owner_id=None

        a_rows = list_searches_by_owner(conn, "owner-a")
        assert [r.id for r in a_rows] == [id_a2, id_a1]  # newest first
        assert all(r.owner_id == "owner-a" for r in a_rows)

        b_rows = list_searches_by_owner(conn, "owner-b")
        assert len(b_rows) == 1

        none_rows = list_searches_by_owner(conn, "nonexistent-owner")
        assert none_rows == []

        # Unowned (owner_id IS NULL) rows never leak into any real
        # owner's listing -- SQL's NULL != 'owner-a' semantics already
        # guarantee this, confirmed here rather than only assumed.
        all_a_and_b_ids = {r.id for r in a_rows} | {r.id for r in b_rows}
        assert id_a1 in all_a_and_b_ids and id_a2 in all_a_and_b_ids


def test_list_searches_by_owner_respects_limit():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        for i in range(5):
            save_search(conn, f"t{i}", [f"p{i}"], [0.5], owner_id="owner-a")
        limited = list_searches_by_owner(conn, "owner-a", limit=2)
        assert len(limited) == 2


if __name__ == "__main__":
    test_save_and_get_search_roundtrip()
    test_get_search_missing_id_returns_none()
    test_update_summary_persists()
    test_save_search_without_web_articles_defaults_to_empty_list()
    test_save_search_persists_web_articles()
    test_update_web_summary_persists()
    test_init_db_migrates_pre_existing_database_missing_web_columns()
    test_list_searches_orders_newest_first()
    test_list_searches_never_returns_more_than_limit()
    test_list_searches_tie_breaks_equal_timestamps_by_descending_id()
    test_list_searches_default_limit_constant_is_100()
    test_get_search_and_list_searches_ignore_an_unrelated_extra_column()
    test_init_db_creates_the_created_at_index_on_a_new_database()
    test_init_db_creates_the_created_at_index_on_a_legacy_database()
    test_init_db_is_idempotent_across_repeated_calls()
    test_legacy_rows_deserialize_with_web_field_defaults()
    test_list_searches_query_plan_uses_the_created_at_index()
    test_get_db_connection_yields_working_connection_and_closes_it_after()
    test_concurrent_requests_via_per_request_connections_do_not_corrupt_storage()
    test_init_db_migrates_pre_existing_database_missing_owner_id_column()
    test_save_search_without_owner_id_defaults_to_none()
    test_save_search_persists_owner_id()
    test_list_searches_by_owner_returns_only_that_owners_rows()
    test_list_searches_by_owner_respects_limit()
    print("All storage tests passed.")
