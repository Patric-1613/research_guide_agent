"""Deterministic tests for the SQLite persistence layer — no mocking needed,
just a temp DB file per test."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3

from research_agent.storage import (
    delete_bag,
    get_bag,
    get_search,
    init_db,
    list_bags,
    list_searches,
    paper_ids_referenced_by_other_bags,
    save_bag,
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


def _summary_fixture() -> dict:
    return {"themes": [{"theme_name": "Theme", "papers": []}], "gaps_and_disagreements": "none", "skipped_paper_ids": []}


def test_save_and_get_bag_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        rounds = [{"round_number": 1, "keywords_used": ["PEFT"], "timestamp": "t", "paper_ids_found": ["p1"],
                   "new_paper_ids": ["p1"], "web_urls_found": [], "new_web_urls": []}]
        bag_id, created_at = save_bag(
            conn, "My PEFT Bag", "PEFT", ["p1", "p2"], [], rounds, _summary_fixture(),
        )

        bag = get_bag(conn, bag_id)
        assert bag.name == "My PEFT Bag"
        assert bag.topic == "PEFT"
        assert bag.created_at == created_at
        assert bag.paper_ids == ["p1", "p2"]
        assert bag.rounds == rounds
        assert bag.summary == _summary_fixture()
        assert bag.web_summary is None


def test_get_bag_missing_id_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        assert get_bag(conn, 999) is None


def test_save_bag_persists_web_articles_and_web_summary():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        articles = [{"title": "A", "url": "https://x.com/a", "snippet": "s", "published_date": None, "source_domain": "x.com"}]
        web_summary = {"synthesis": "X", "cited_articles": articles}
        bag_id, _ = save_bag(conn, "Bag", "topic", [], articles, [], _summary_fixture(), web_summary)
        bag = get_bag(conn, bag_id)
        assert bag.web_articles == articles
        assert bag.web_summary == web_summary


def test_list_bags_orders_newest_first():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        id1, _ = save_bag(conn, "First", "t1", ["a"], [], [], _summary_fixture())
        id2, _ = save_bag(conn, "Second", "t2", ["b"], [], [], _summary_fixture())

        results = list_bags(conn)
        assert [b.id for b in results] == [id2, id1]


def test_delete_bag_removes_row_and_reports_success():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        bag_id, _ = save_bag(conn, "Bag", "topic", ["p1"], [], [], _summary_fixture())

        assert delete_bag(conn, bag_id) is True
        assert get_bag(conn, bag_id) is None
        assert delete_bag(conn, bag_id) is False  # already gone


def test_paper_ids_referenced_by_other_bags_detects_shared_paper():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        bag_a_id, _ = save_bag(conn, "Bag A", "t", ["shared", "only_in_a"], [], [], _summary_fixture())
        save_bag(conn, "Bag B", "t", ["shared", "only_in_b"], [], [], _summary_fixture())

        # "shared" is referenced by bag B even after excluding bag A itself.
        result = paper_ids_referenced_by_other_bags(conn, ["shared", "only_in_a"], exclude_bag_id=bag_a_id)
        assert result == {"shared"}


def test_paper_ids_referenced_by_other_bags_empty_when_no_other_bags():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        bag_id, _ = save_bag(conn, "Bag", "t", ["p1"], [], [], _summary_fixture())
        assert paper_ids_referenced_by_other_bags(conn, ["p1"], exclude_bag_id=bag_id) == set()


def test_init_db_creates_bags_table_for_legacy_database_missing_it():
    """A database file from round 1/2 has `searches` but no `bags` table at
    all — init_db must create it without disturbing the existing table."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.sqlite"
        legacy_conn = sqlite3.connect(path)
        legacy_conn.execute(
            """
            CREATE TABLE searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, created_at TEXT NOT NULL,
                paper_ids TEXT NOT NULL, scores TEXT NOT NULL, summary TEXT
            )
            """
        )
        legacy_conn.commit()
        legacy_conn.close()

        conn = init_db(path)  # must not raise
        bag_id, _ = save_bag(conn, "Bag", "t", ["p1"], [], [], _summary_fixture())
        assert get_bag(conn, bag_id) is not None


if __name__ == "__main__":
    test_save_and_get_search_roundtrip()
    test_get_search_missing_id_returns_none()
    test_update_summary_persists()
    test_save_search_without_web_articles_defaults_to_empty_list()
    test_save_search_persists_web_articles()
    test_update_web_summary_persists()
    test_init_db_migrates_pre_existing_database_missing_web_columns()
    test_list_searches_orders_newest_first()
    test_save_and_get_bag_roundtrip()
    test_get_bag_missing_id_returns_none()
    test_save_bag_persists_web_articles_and_web_summary()
    test_list_bags_orders_newest_first()
    test_delete_bag_removes_row_and_reports_success()
    test_paper_ids_referenced_by_other_bags_detects_shared_paper()
    test_paper_ids_referenced_by_other_bags_empty_when_no_other_bags()
    test_init_db_creates_bags_table_for_legacy_database_missing_it()
    print("All storage tests passed.")
