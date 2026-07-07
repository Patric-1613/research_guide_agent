"""Deterministic tests for the SQLite persistence layer — no mocking needed,
just a temp DB file per test."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.storage import get_search, init_db, list_searches, save_search, update_summary


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


def test_list_searches_orders_newest_first():
    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(Path(tmp) / "test.sqlite")
        id1, _ = save_search(conn, "first", ["a"], [1.0])
        id2, _ = save_search(conn, "second", ["b"], [1.0])

        results = list_searches(conn)
        assert [s.id for s in results] == [id2, id1]


if __name__ == "__main__":
    test_save_and_get_search_roundtrip()
    test_get_search_missing_id_returns_none()
    test_update_summary_persists()
    test_list_searches_orders_newest_first()
    print("All storage tests passed.")
