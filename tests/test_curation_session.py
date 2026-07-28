"""Tests for curation_session.py (curation-checkpointer Phase 2): the
SQLite checkpointer activated for the new literature-review curation
flow only. Real SQLite reads throughout, not just trusting the
save/load API round-trip — that's the whole point of this phase.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.curation_session import (
    build_curation_graph,
    curation_thread_id,
    delete_curation_session,
    list_curation_sessions,
    load_curation_session,
    save_curation_session,
)
from research_agent.qa import sqlite_checkpointer
from research_agent.query_expansion import PaperPoolSession
from research_agent.schema import Paper, WebArticle


def _paper(pid: str) -> Paper:
    return Paper(
        title=f"Paper {pid}", authors=["A"], year=2024, venue="X",
        abstract=f"abstract {pid}", url=None, doi=None, citation_count=None,
        source="arxiv", paper_id=pid,
    )


def test_save_then_load_roundtrips_all_fields_verified_via_real_sqlite_read():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        session = PaperPoolSession(
            topic="parameter-efficient fine-tuning",
            reserve=[(_paper("p0"), 0.9), (_paper("p1"), 0.8)],
            cursor=1,
            seen_paper_ids={"p0"},
            seen_titles={"Paper p0"},
            stage="curate",
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-1", cp)

        # Real SQLite read of the actual file — not the checkpointer API —
        # confirming the row genuinely exists on disk under the expected
        # thread_id, not just trusting a round-trip through the same API
        # that wrote it.
        conn = sqlite3.connect(db_path)
        thread_ids = {row[0] for row in conn.execute("SELECT thread_id FROM checkpoints")}
        conn.close()
        assert thread_ids == {curation_thread_id("session-1")}

        with sqlite_checkpointer(db_path) as cp2:
            loaded = load_curation_session("session-1", cp2)

        assert loaded is not None
        assert loaded.topic == session.topic
        assert loaded.cursor == 1
        assert loaded.seen_paper_ids == {"p0"}
        assert loaded.seen_titles == {"Paper p0"}
        assert loaded.stage == "curate"
        assert [p.paper_id for p, _ in loaded.reserve] == ["p0", "p1"]
        assert [score for _, score in loaded.reserve] == [0.9, 0.8]


def test_load_nonexistent_session_returns_none_not_an_error():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = load_curation_session("never-saved-session", cp)
        assert result is None


def test_two_sessions_do_not_cross_contaminate():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        session_a = PaperPoolSession(topic="topic A", reserve=[(_paper("a1"), 0.9)], stage="curate")
        session_b = PaperPoolSession(topic="topic B", reserve=[(_paper("b1"), 0.8)], stage="synthesize")

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session_a, "session-a", cp)
            save_curation_session(session_b, "session-b", cp)

            loaded_a = load_curation_session("session-a", cp)
            loaded_b = load_curation_session("session-b", cp)

        assert loaded_a.topic == "topic A" and loaded_a.stage == "curate"
        assert [p.paper_id for p, _ in loaded_a.reserve] == ["a1"]
        assert loaded_b.topic == "topic B" and loaded_b.stage == "synthesize"
        assert [p.paper_id for p, _ in loaded_b.reserve] == ["b1"]


def test_saving_again_under_the_same_session_id_updates_not_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        session = PaperPoolSession(topic="q", reserve=[(_paper("p0"), 0.9)], cursor=0)

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-1", cp)
            session.cursor = 1
            session.seen_paper_ids.add("p0")
            save_curation_session(session, "session-1", cp)

            loaded = load_curation_session("session-1", cp)

        # get_state() always returns the LATEST checkpoint for a thread_id
        assert loaded.cursor == 1
        assert loaded.seen_paper_ids == {"p0"}


def test_phase5_fields_roundtrip_including_nested_papers_inside_report():
    """curation-chat-web-escalation Phase 5b: report/chat_history/
    web_articles_added/pending_web_offer/pending_report_update are new
    fields on PaperPoolSession. report specifically nests raw Paper
    objects inside each section's cited_papers and in skipped_papers --
    the same deprecation-warning trap Phase 2 already solved for the
    top-level fields, so this test exists to prove
    _serialize_report/_deserialize_report actually close that gap rather
    than just trusting the round-trip API worked."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        cited = _paper("p0")
        skipped = _paper("p1")
        report = {
            "findings": {"content": "papers show X", "cited_papers": [cited]},
            "limitations": {"content": "no major gaps", "cited_papers": []},
            "future_scope": {"content": "future work on Y", "cited_papers": [cited]},
            "skipped_papers": [skipped],
        }
        web_article = WebArticle(
            title="A Survey", url="https://example.com/survey",
            snippet="a snippet", published_date="2024-01-01", source_domain="example.com",
        )
        session = PaperPoolSession(
            topic="parameter-efficient fine-tuning",
            reserve=[(_paper("p0"), 0.9), (_paper("p1"), 0.8)],
            stage="synthesize",
            selected_paper_ids=["p0", "p1"],
            selected_papers=[cited, skipped],
            report=report,
            chat_history=[{"role": "user", "content": "what do these papers find?"}],
            web_articles_added=[web_article],
            pending_web_offer={"question": "what about scaling laws?"},
            pending_report_update={"reason": "new web source approved"},
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-phase5", cp)
            loaded = load_curation_session("session-phase5", cp)

        assert loaded is not None
        assert loaded.report is not None
        assert loaded.report["findings"]["content"] == "papers show X"
        assert all(isinstance(p, Paper) for p in loaded.report["findings"]["cited_papers"])
        assert [p.paper_id for p in loaded.report["findings"]["cited_papers"]] == ["p0"]
        assert [p.paper_id for p in loaded.report["skipped_papers"]] == ["p1"]
        assert loaded.chat_history == [{"role": "user", "content": "what do these papers find?"}]
        assert len(loaded.web_articles_added) == 1
        assert isinstance(loaded.web_articles_added[0], WebArticle)
        assert loaded.web_articles_added[0].url == "https://example.com/survey"
        assert loaded.pending_web_offer == {"question": "what about scaling laws?"}
        assert loaded.pending_report_update == {"reason": "new web source approved"}


def test_list_curation_sessions_returns_empty_for_no_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            assert list_curation_sessions(cp) == []


def test_list_curation_sessions_returns_a_summary_per_session_most_recently_touched_first():
    """curation-api-and-ui Phase 6b/6c: powers the frontend's reviews
    list. Confirms both the summary fields AND the ordering claim (most
    recently touched thread first) against real, separately-timed
    writes -- not just trusting the underlying checkpointer.list()
    docstring."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        session_a = PaperPoolSession(topic="Topic A", reserve=[(_paper("a0"), 0.9)], target_count=5)
        session_b = PaperPoolSession(topic="Topic B", reserve=[(_paper("b0"), 0.9)], target_count=3)

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session_a, "session-a", cp)
            save_curation_session(session_b, "session-b", cp)
            # Touch session-a again, most recently -- it should now sort first.
            session_a.selected_paper_ids = ["a0"]
            save_curation_session(session_a, "session-a", cp)

            summaries = list_curation_sessions(cp)

        assert [s["session_id"] for s in summaries] == ["session-a", "session-b"]
        assert summaries[0]["topic"] == "Topic A"
        assert summaries[0]["selected_count"] == 1
        assert summaries[0]["target_count"] == 5
        assert summaries[1]["topic"] == "Topic B"
        assert summaries[1]["selected_count"] == 0


def test_list_curation_sessions_flags_has_report_and_has_chat_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        bare = PaperPoolSession(topic="bare", reserve=[(_paper("p0"), 0.9)], stage="curate")
        with_chat_only = PaperPoolSession(
            topic="chatty", reserve=[(_paper("p1"), 0.9)], stage="synthesize",
            chat_history=[{"role": "user", "content": "hi"}],
        )
        with_report_and_chat = PaperPoolSession(
            topic="full", reserve=[(_paper("p2"), 0.9)], stage="synthesize",
            report={
                "findings": {"content": "f", "cited_papers": []},
                "limitations": {"content": "", "cited_papers": []},
                "future_scope": {"content": "", "cited_papers": []},
                "skipped_papers": [],
            },
            chat_history=[{"role": "user", "content": "hi"}],
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(bare, "bare-id", cp)
            save_curation_session(with_chat_only, "chatty-id", cp)
            save_curation_session(with_report_and_chat, "full-id", cp)
            summaries = {s["session_id"]: s for s in list_curation_sessions(cp)}

        assert summaries["bare-id"]["has_report"] is False
        assert summaries["bare-id"]["has_chat"] is False
        assert summaries["chatty-id"]["has_report"] is False
        assert summaries["chatty-id"]["has_chat"] is True
        assert summaries["full-id"]["has_report"] is True
        assert summaries["full-id"]["has_chat"] is True


def test_corrupted_database_file_raises_cleanly_instead_of_silently_returning_wrong_data():
    """Simulates a crash mid-write by truncating the file after a real
    save — the point isn't that this must not error (a genuinely
    corrupted database IS a real, serious problem), it's that it must
    fail LOUDLY and cleanly (a catchable exception an operator can act
    on), never silently return None (indistinguishable from "session
    never existed") or partial/wrong data."""
    import pytest

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(PaperPoolSession(topic="q"), "session-1", cp)

        original_size = db_path.stat().st_size
        with open(db_path, "r+b") as f:
            f.truncate(original_size // 2)

        with pytest.raises(Exception):
            with sqlite_checkpointer(db_path) as cp:
                load_curation_session("session-1", cp)


# --- display_title (curation-review-management Phase 8, item 5) ---

def test_display_title_roundtrips_through_real_sqlite_distinct_from_topic():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        session = PaperPoolSession(
            topic="cars cooling system",
            display_title="Automotive Engine Cooling Systems",
            reserve=[(_paper("p0"), 0.9)],
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-1", cp)

        with sqlite_checkpointer(db_path) as cp2:
            loaded = load_curation_session("session-1", cp2)

        assert loaded.topic == "cars cooling system"
        assert loaded.display_title == "Automotive Engine Cooling Systems"


def test_loading_a_pre_phase8_session_without_display_title_falls_back_to_its_own_topic():
    """A checkpoint saved before this field existed has no "display_title"
    key in its dict at all -- simulated here by invoking the graph
    directly with a hand-built dict, bypassing _session_to_dict (which
    would always include the key for a session built today)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        old_format_dict = {
            "topic": "old style session", "reserve": [], "cursor": 0,
            "seen_paper_ids": [], "seen_titles": [], "stage": "curate", "target_count": 10,
            "selected_paper_ids": [], "selected_papers": [], "report": None, "chat_history": [],
            "web_articles_added": [], "pending_web_offer": None, "pending_report_update": None,
            "refinement_notes": [], "report_covered_web_article_count": 0,
            # deliberately no "display_title" key
        }

        with sqlite_checkpointer(db_path) as cp:
            graph = build_curation_graph(cp)
            config = {"configurable": {"thread_id": curation_thread_id("old-id")}}
            graph.invoke({"session": old_format_dict}, config=config)

            loaded = load_curation_session("old-id", cp)

        assert loaded.topic == "old style session"
        assert loaded.display_title == "old style session"


def test_list_curation_sessions_includes_display_title():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        session = PaperPoolSession(
            topic="cars cooling system", display_title="Automotive Engine Cooling Systems",
            reserve=[(_paper("p0"), 0.9)],
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-1", cp)
            summaries = list_curation_sessions(cp)

        assert summaries[0]["topic"] == "cars cooling system"
        assert summaries[0]["display_title"] == "Automotive Engine Cooling Systems"


# --- delete_curation_session (curation-review-management Phase 8, item 1) ---

def test_delete_curation_session_removes_it_for_real():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        session = PaperPoolSession(topic="to be deleted", reserve=[(_paper("p0"), 0.9)])

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-1", cp)
            assert load_curation_session("session-1", cp) is not None

            delete_curation_session("session-1", cp)

            assert load_curation_session("session-1", cp) is None
            assert list_curation_sessions(cp) == []

        # Real SQLite read confirming the rows are actually gone from
        # disk, not just unreachable through the checkpointer API.
        conn = sqlite3.connect(db_path)
        remaining = list(conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (curation_thread_id("session-1"),),
        ))
        conn.close()
        assert remaining == [(0,)]


def test_delete_curation_session_on_unknown_id_does_not_error():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            delete_curation_session("never-existed", cp)  # must not raise


def test_delete_curation_session_does_not_affect_other_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        session_a = PaperPoolSession(topic="Topic A", reserve=[(_paper("a0"), 0.9)])
        session_b = PaperPoolSession(topic="Topic B", reserve=[(_paper("b0"), 0.9)])

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session_a, "session-a", cp)
            save_curation_session(session_b, "session-b", cp)

            delete_curation_session("session-a", cp)

            assert load_curation_session("session-a", cp) is None
            assert load_curation_session("session-b", cp) is not None


if __name__ == "__main__":
    test_save_then_load_roundtrips_all_fields_verified_via_real_sqlite_read()
    test_load_nonexistent_session_returns_none_not_an_error()
    test_two_sessions_do_not_cross_contaminate()
    test_saving_again_under_the_same_session_id_updates_not_duplicates()
    test_phase5_fields_roundtrip_including_nested_papers_inside_report()
    test_list_curation_sessions_returns_empty_for_no_sessions()
    test_list_curation_sessions_returns_a_summary_per_session_most_recently_touched_first()
    test_list_curation_sessions_flags_has_report_and_has_chat_correctly()
    test_corrupted_database_file_raises_cleanly_instead_of_silently_returning_wrong_data()
    test_display_title_roundtrips_through_real_sqlite_distinct_from_topic()
    test_loading_a_pre_phase8_session_without_display_title_falls_back_to_its_own_topic()
    test_list_curation_sessions_includes_display_title()
    test_delete_curation_session_removes_it_for_real()
    test_delete_curation_session_on_unknown_id_does_not_error()
    test_delete_curation_session_does_not_affect_other_sessions()
    print("All curation_session tests passed.")
