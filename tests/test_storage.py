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
    get_db_connection,
    get_search,
    init_db,
    list_searches,
    save_search,
    update_summary,
    update_web_summary,
)


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


if __name__ == "__main__":
    test_save_and_get_search_roundtrip()
    test_get_search_missing_id_returns_none()
    test_update_summary_persists()
    test_save_search_without_web_articles_defaults_to_empty_list()
    test_save_search_persists_web_articles()
    test_update_web_summary_persists()
    test_init_db_migrates_pre_existing_database_missing_web_columns()
    test_list_searches_orders_newest_first()
    test_get_db_connection_yields_working_connection_and_closes_it_after()
    test_concurrent_requests_via_per_request_connections_do_not_corrupt_storage()
    print("All storage tests passed.")
