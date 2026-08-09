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

from research_agent.api_app.serializers import _report_to_out
from research_agent.curation_session import (
    build_curation_graph,
    curation_thread_id,
    delete_curation_session,
    list_curation_sessions,
    load_curation_session,
    reopen_curation_session,
    save_curation_session,
    select_paper_from_history,
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


def test_web_article_provenance_by_url_round_trips_through_real_sqlite():
    """R7E.2: web_article_provenance_by_url is plain JSON-native (str ->
    {str: str}) -- no Paper/WebArticle objects nested in it, so this
    proves the opaque pass-through in _session_to_dict/_dict_to_session
    actually preserves it through a real save/load, not just that the
    dataclass default survives (which would pass trivially even if the
    field were silently dropped from serialization)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        web_article = WebArticle(
            title="A Survey", url="https://example.com/survey",
            snippet="a snippet", published_date="2024-01-01", source_domain="example.com",
        )
        session = PaperPoolSession(
            topic="parameter-efficient fine-tuning",
            stage="synthesize",
            web_articles_added=[web_article],
            web_article_provenance_by_url={
                "https://example.com/survey": {
                    "source_query": "PEFT survey papers", "added_at": "2026-08-09T12:00:00+00:00",
                },
            },
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-provenance", cp)
            loaded = load_curation_session("session-provenance", cp)

        assert loaded is not None
        assert loaded.web_article_provenance_by_url == {
            "https://example.com/survey": {
                "source_query": "PEFT survey papers", "added_at": "2026-08-09T12:00:00+00:00",
            },
        }


def test_report_cited_web_articles_and_references_survive_serialize_deserialize():
    """report-quality Phase R1 bug fix: cited_web_articles used to be
    dropped entirely by _serialize_report/_deserialize_report -- only
    content/cited_papers ever survived a save/load round trip. This
    proves a section's web citations, its reference_numbers, and the
    top-level references list all round-trip through real SQLite intact."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        cited_paper = _paper("p0")
        web_article = WebArticle(
            title="A Survey", url="https://example.com/survey",
            snippet="a snippet", published_date="2024-01-01", source_domain="example.com",
        )
        report = {
            "findings": {
                "content": "Per [1] and [2], X.", "cited_papers": [cited_paper],
                "cited_web_articles": [web_article], "reference_numbers": [1, 2],
            },
            "limitations": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "future_scope": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "skipped_papers": [],
            "references": [
                {"number": 1, "kind": "paper", "paper_id": "p0", "url": None, "title": "Paper p0",
                 "formatted": "A. (2024). Paper p0. X.", "link_url": None},
                {"number": 2, "kind": "web", "paper_id": None, "url": "https://example.com/survey",
                 "title": "A Survey", "formatted": "A Survey. example.com. https://example.com/survey",
                 "link_url": "https://example.com/survey"},
            ],
        }
        session = PaperPoolSession(
            topic="parameter-efficient fine-tuning", stage="synthesize",
            selected_paper_ids=["p0"], selected_papers=[cited_paper], report=report,
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-report-web-refs", cp)
            loaded = load_curation_session("session-report-web-refs", cp)

        assert loaded is not None
        assert loaded.report is not None
        findings = loaded.report["findings"]
        assert len(findings["cited_web_articles"]) == 1
        assert isinstance(findings["cited_web_articles"][0], WebArticle)
        assert findings["cited_web_articles"][0].url == "https://example.com/survey"
        assert findings["reference_numbers"] == [1, 2]
        assert [r["number"] for r in loaded.report["references"]] == [1, 2]
        assert loaded.report["references"][1]["kind"] == "web"


def test_report_without_references_or_web_citations_still_loads_old_shape():
    """A pre-R1 persisted report has neither cited_web_articles nor
    reference_numbers/references at all -- must still load cleanly,
    not crash, and simply lack those keys (the API serializer, not this
    module, is what derives them on the fly -- see
    test_report_to_out_derives_references_for_an_old_shape_report_dict
    in test_curation_api.py)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        cited_paper = _paper("p0")
        old_report = {
            "findings": {"content": "Old prose, no markers.", "cited_papers": [cited_paper]},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [],
        }
        session = PaperPoolSession(
            topic="q", stage="synthesize", selected_papers=[cited_paper], report=old_report,
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-old-report", cp)
            loaded = load_curation_session("session-old-report", cp)

        assert loaded is not None
        assert loaded.report is not None
        assert loaded.report["findings"]["content"] == "Old prose, no markers."
        assert loaded.report["findings"]["cited_web_articles"] == []
        assert loaded.report["findings"]["reference_numbers"] == []
        assert "references" not in loaded.report
        assert "sections" not in loaded.report


def test_report_sections_survive_serialize_deserialize_when_present():
    """report-quality Phase R2A: `sections` is opaquely round-tripped the
    same way `references` already is -- a report dict that already
    carries a `sections` list (as a future real multi-section generation
    would produce) must survive a save/load cycle through real SQLite
    intact, in order."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        cited_paper = _paper("p0")
        report = {
            "findings": {"content": "F", "cited_papers": [cited_paper], "cited_web_articles": [], "reference_numbers": [1]},
            "limitations": {"content": "L", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "future_scope": {"content": "S", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "skipped_papers": [],
            "references": [
                {"number": 1, "kind": "paper", "paper_id": "p0", "url": None, "title": "Paper p0",
                 "formatted": "A. (2024). Paper p0. X.", "link_url": None},
            ],
            "sections": [
                {"key": "findings", "title": "Findings", "content": "F", "reference_numbers": [1]},
                {"key": "limitations", "title": "Limitations", "content": "L", "reference_numbers": []},
                {"key": "future_scope", "title": "Future Scope", "content": "S", "reference_numbers": []},
            ],
        }
        session = PaperPoolSession(
            topic="q", stage="synthesize", selected_paper_ids=["p0"], selected_papers=[cited_paper], report=report,
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-report-sections", cp)
            loaded = load_curation_session("session-report-sections", cp)

        assert loaded is not None
        assert loaded.report is not None
        assert [s["key"] for s in loaded.report["sections"]] == ["findings", "limitations", "future_scope"]
        assert loaded.report["sections"][0]["content"] == "F"
        assert loaded.report["sections"][0]["reference_numbers"] == [1]


# --- report-quality Phase R2C: report_template + 8-section round-trip fix ---

def test_report_template_survives_serialize_deserialize():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        cited_paper = _paper("p0")
        report = {
            "findings": {"content": "F", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "limitations": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "future_scope": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "skipped_papers": [],
            "report_template": "expert",
        }
        session = PaperPoolSession(
            topic="q", stage="synthesize", selected_paper_ids=["p0"], selected_papers=[cited_paper], report=report,
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-report-template", cp)
            loaded = load_curation_session("session-report-template", cp)

        assert loaded is not None
        assert loaded.report is not None
        assert loaded.report["report_template"] == "expert"


def test_report_refinement_metadata_survives_serialize_deserialize():
    """report-quality Phase R4.1 bug fix: refinement was silently
    dropped on the very first save/load round trip -- _serialize_report/
    _deserialize_report built their output from an explicit, hardcoded
    key list that never accounted for the new top-level "refinement"
    key, so it was present in the immediate POST response (built from
    the in-memory report dict) but gone from every subsequent GET
    /curation/{id} (which reads the persisted, round-tripped copy) --
    the same class of gap R2C's own section-key round-trip bug was.

    report-quality Phase R4.2: uses the FULL refinement shape (issues/
    revision_instructions/section_scores included, not just R4.1's
    original 4 keys) -- proves the opaque, whole-dict pass-through the
    R4.1 bug fix introduced covers the new fields with zero further
    changes to this module, exactly as designed."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        cited_paper = _paper("p0")
        report = {
            "findings": {"content": "F", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "limitations": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "future_scope": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "skipped_papers": [],
            "report_template": "analytical",
            "refinement": {
                "enabled": True, "rounds": 1, "initial_score": 40, "final_score": None,
                "issues": ["too shallow", "missing comparison"], "revision_instructions": "add more depth",
                "section_scores": {"thematic_findings": 35, "conclusion": 50},
            },
        }
        session = PaperPoolSession(
            topic="q", stage="synthesize", selected_paper_ids=["p0"], selected_papers=[cited_paper], report=report,
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-report-refinement", cp)
            loaded = load_curation_session("session-report-refinement", cp)

        assert loaded is not None
        assert loaded.report is not None
        assert loaded.report["refinement"] == {
            "enabled": True, "rounds": 1, "initial_score": 40, "final_score": None,
            "issues": ["too shallow", "missing comparison"], "revision_instructions": "add more depth",
            "section_scores": {"thematic_findings": 35, "conclusion": 50},
        }


def test_report_without_refinement_still_loads_cleanly():
    """A report refinement was never requested for (no "refinement" key
    at all -- the overwhelming common case, including every report
    generated before R4.1) round-trips with no "refinement" key added,
    not a fabricated {"enabled": False, ...} placeholder."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        cited_paper = _paper("p0")
        report = {
            "findings": {"content": "F", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "limitations": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "future_scope": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        session = PaperPoolSession(
            topic="q", stage="synthesize", selected_paper_ids=["p0"], selected_papers=[cited_paper], report=report,
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-report-no-refinement", cp)
            loaded = load_curation_session("session-report-no-refinement", cp)

        assert loaded is not None
        assert "refinement" not in loaded.report


def test_report_version_body_preserves_refinement_metadata():
    """report-quality Phase R4.1: _serialize_report_version/
    _deserialize_report_version reuse _serialize_report/_deserialize_
    report unchanged, so a specific version's own refinement metadata
    (not just session.report's) survives the same round trip -- proven
    here directly against report_versions[0], not just session.report."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        cited_paper = _paper("p0")
        report = {
            "findings": {"content": "F", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "limitations": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "future_scope": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
            "skipped_papers": [],
            "report_template": "analytical",
            "refinement": {"enabled": True, "rounds": 0, "initial_score": 88, "final_score": 88},
        }
        version = {
            "version_id": "v1", "version_number": 1, "created_at": "2026-08-06T00:00:00+00:00",
            "report_template": "analytical", "generation_reason": "initial", "report": report,
        }
        session = PaperPoolSession(
            topic="q", stage="synthesize", selected_paper_ids=["p0"], selected_papers=[cited_paper],
            report=report, report_versions=[version], active_report_version_id="v1",
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-report-version-refinement", cp)
            loaded = load_curation_session("session-report-version-refinement", cp)

        assert loaded is not None
        assert loaded.report_versions[0]["report"]["refinement"] == {
            "enabled": True, "rounds": 0, "initial_score": 88, "final_score": 88,
        }


def test_report_to_out_includes_full_r42_refinement_fields():
    """report-quality Phase R4.2: _report_to_out's existing
    ReportRefinementOut(**report["refinement"]) construction (unchanged
    since R4.1) correctly surfaces issues/revision_instructions/
    section_scores once they're present in the report dict -- no
    serializer code change was needed for R4.2, this proves it."""
    report = {
        "findings": {"content": "F", "cited_papers": []},
        "limitations": {"content": "", "cited_papers": []},
        "future_scope": {"content": "", "cited_papers": []},
        "skipped_paper_ids": [], "skipped_papers": [],
        "refinement": {
            "enabled": True, "rounds": 1, "initial_score": 40, "final_score": None,
            "issues": ["too shallow"], "revision_instructions": "add more depth",
            "section_scores": {"thematic_findings": 35},
        },
    }

    out = _report_to_out(report)

    assert out.refinement is not None
    assert out.refinement.issues == ["too shallow"]
    assert out.refinement.revision_instructions == "add more depth"
    assert out.refinement.section_scores == {"thematic_findings": 35}


def test_report_to_out_defaults_new_fields_for_an_r41_only_refinement_dict():
    """report-quality Phase R4.2 compatibility: a report persisted
    before R4.2 shipped has a "refinement" dict with only R4.1's
    original 4 keys (enabled/rounds/initial_score/final_score) -- no
    issues/revision_instructions/section_scores at all. _report_to_out
    must still construct a valid ReportRefinementOut, with Pydantic's
    own field defaults filling the gap -- no migration, no special-
    casing needed in the serializer itself."""
    report = {
        "findings": {"content": "F", "cited_papers": []},
        "limitations": {"content": "", "cited_papers": []},
        "future_scope": {"content": "", "cited_papers": []},
        "skipped_paper_ids": [], "skipped_papers": [],
        "refinement": {"enabled": True, "rounds": 0, "initial_score": 88, "final_score": 88},
    }

    out = _report_to_out(report)

    assert out.refinement is not None
    assert out.refinement.enabled is True
    assert out.refinement.initial_score == 88
    assert out.refinement.issues == []
    assert out.refinement.revision_instructions == ""
    assert out.refinement.section_scores is None


def test_report_all_eight_analytical_sections_survive_serialize_deserialize_with_cited_papers_and_web_articles():
    """report-quality Phase R2C serialization fix: previously only the 3
    legacy keys (findings/limitations/future_scope) got their full
    {content, cited_papers, cited_web_articles, reference_numbers} dict
    reconstructed on load -- the 5 non-legacy-mapped Analytical keys
    (executive_summary, introduction_scope, methodology_landscape,
    gap_analysis, conclusion) silently vanished as top-level dict keys
    after a save/load round trip, which meant report.py's own citation-
    preservation helpers could never find prior citations for them on
    any regeneration reached through the real HTTP API (every request
    reloads the session fresh from the checkpointer -- there is no
    in-process cache). This proves all 8 keys, not just the 3 legacy
    ones, now round-trip intact -- including a web citation on one of
    the previously-dropped keys -- through real SQLite."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        p0, p1 = _paper("p0"), _paper("p1")
        web_article = WebArticle(
            title="A Survey", url="https://example.com/survey",
            snippet="a snippet", published_date="2024-01-01", source_domain="example.com",
        )

        def section(content, papers, articles=None, refs=None):
            return {
                "content": content, "cited_papers": papers,
                "cited_web_articles": articles or [], "reference_numbers": refs or [],
            }

        report = {
            "executive_summary": section("Exec.", [p0], refs=[1]),
            "introduction_scope": section("Intro.", []),
            "thematic_findings": section("Findings.", [p0, p1], refs=[1, 2]),
            "methodology_landscape": section("Methods.", []),
            "contradictions_open_debates": section("Debates.", []),
            "gap_analysis": section("Gaps.", [p1], [web_article], refs=[2, 3]),
            "future_research_directions": section("Future.", []),
            "conclusion": section("Done.", []),
            "findings": section("Findings.", [p0, p1], refs=[1, 2]),
            "limitations": section("Debates.", []),
            "future_scope": section("Future.", []),
            "skipped_papers": [],
            "references": [
                {"number": 1, "kind": "paper", "paper_id": "p0", "url": None, "title": "Paper p0", "formatted": "x", "link_url": None},
                {"number": 2, "kind": "paper", "paper_id": "p1", "url": None, "title": "Paper p1", "formatted": "x", "link_url": None},
                {"number": 3, "kind": "web", "paper_id": None, "url": "https://example.com/survey", "title": "A Survey", "formatted": "x", "link_url": "https://example.com/survey"},
            ],
            "report_template": "analytical",
        }
        session = PaperPoolSession(
            topic="q", stage="synthesize", selected_paper_ids=["p0", "p1"], selected_papers=[p0, p1], report=report,
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-full-report", cp)
            loaded = load_curation_session("session-full-report", cp)

        assert loaded is not None
        r = loaded.report
        assert r is not None
        for key in ("executive_summary", "introduction_scope", "methodology_landscape", "gap_analysis", "conclusion"):
            assert key in r, f"{key} missing from deserialized report"
        assert [p.paper_id for p in r["executive_summary"]["cited_papers"]] == ["p0"]
        assert r["executive_summary"]["reference_numbers"] == [1]
        assert [p.paper_id for p in r["gap_analysis"]["cited_papers"]] == ["p1"]
        assert len(r["gap_analysis"]["cited_web_articles"]) == 1
        assert isinstance(r["gap_analysis"]["cited_web_articles"][0], WebArticle)
        assert r["gap_analysis"]["cited_web_articles"][0].url == "https://example.com/survey"
        assert r["gap_analysis"]["reference_numbers"] == [2, 3]
        # Legacy fields still present too -- the fix is additive, never a
        # replacement for the pre-existing legacy round-trip.
        assert [p.paper_id for p in r["findings"]["cited_papers"]] == ["p0", "p1"]
        assert r["report_template"] == "analytical"


def test_report_without_report_template_still_loads():
    """A pre-R2C persisted report has no report_template key at all --
    must still load cleanly, not crash, and simply lack the key (the
    API serializer, not this module, is what defaults it to "analytical"
    on the way out -- see api_app/serializers.py's _report_to_out)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        cited_paper = _paper("p0")
        old_report = {
            "findings": {"content": "Old prose.", "cited_papers": [cited_paper]},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [],
        }
        session = PaperPoolSession(topic="q", stage="synthesize", selected_papers=[cited_paper], report=old_report)

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-old-report-no-template", cp)
            loaded = load_curation_session("session-old-report-no-template", cp)

        assert loaded is not None
        assert loaded.report is not None
        assert "report_template" not in loaded.report


# --- report-quality Phase R3: report versioning ---

def test_old_session_with_only_report_loads_as_implicit_version_1():
    """A checkpoint saved before R3 has a `report` but no `report_
    versions`/`active_report_version_id` keys at all -- simulated here
    by invoking the graph directly with a hand-built dict, same
    established pattern as test_loading_a_pre_phase8_session_without_
    display_title_falls_back_to_its_own_topic above. Must load as
    exactly ONE implicit version, generation_reason="initial", with
    active_report_version_id pointing at it -- never crash, never an
    empty version list despite a real report being present."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        old_report = {
            "findings": {"content": "Old findings.", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [],
            "report_template": "expert",
        }
        old_format_dict = {
            "topic": "old style session", "reserve": [], "cursor": 0,
            "seen_paper_ids": [], "seen_titles": [], "stage": "synthesize", "target_count": 10,
            "selected_paper_ids": [], "selected_papers": [], "report": old_report, "chat_history": [],
            "web_articles_added": [], "pending_web_offer": None, "pending_report_update": None,
            "refinement_notes": [], "report_covered_web_article_count": 0,
            # deliberately no "report_versions"/"active_report_version_id" keys
        }

        with sqlite_checkpointer(db_path) as cp:
            graph = build_curation_graph(cp)
            config = {"configurable": {"thread_id": curation_thread_id("old-report-id")}}
            graph.invoke({"session": old_format_dict}, config=config)

            loaded = load_curation_session("old-report-id", cp)

        assert loaded is not None
        assert len(loaded.report_versions) == 1
        version = loaded.report_versions[0]
        assert version["version_number"] == 1
        assert version["generation_reason"] == "initial"
        assert version["created_at"] is None  # real creation time was never recorded, never fabricated
        assert version["report_template"] == "expert"
        assert version["report"]["findings"]["content"] == "Old findings."
        assert loaded.active_report_version_id == version["version_id"]
        # session.report itself is unaffected by this derivation -- still
        # exactly the same report body, same pre-R3 compatibility field.
        assert loaded.report["findings"]["content"] == "Old findings."


def test_old_session_with_no_report_at_all_has_no_implicit_version():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        old_format_dict = {
            "topic": "old style session", "reserve": [], "cursor": 0,
            "seen_paper_ids": [], "seen_titles": [], "stage": "curate", "target_count": 10,
            "selected_paper_ids": [], "selected_papers": [], "report": None, "chat_history": [],
            "web_articles_added": [], "pending_web_offer": None, "pending_report_update": None,
            "refinement_notes": [], "report_covered_web_article_count": 0,
        }

        with sqlite_checkpointer(db_path) as cp:
            graph = build_curation_graph(cp)
            config = {"configurable": {"thread_id": curation_thread_id("old-no-report-id")}}
            graph.invoke({"session": old_format_dict}, config=config)

            loaded = load_curation_session("old-no-report-id", cp)

        assert loaded is not None
        assert loaded.report_versions == []
        assert loaded.active_report_version_id is None


def test_old_session_without_web_article_provenance_key_loads_with_empty_dict():
    """R7E.2 backward compatibility: a checkpoint saved before this phase
    has no web_article_provenance_by_url key at all -- must load cleanly
    as an empty dict, same convention as report_approved_web_article_urls/
    revoked_web_article_urls above (both also absent from this same
    old_format_dict, and both already load as empty via .get(..., set())
    -- this proves the new field follows the identical pattern, not a
    special case)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        old_format_dict = {
            "topic": "old style session", "reserve": [], "cursor": 0,
            "seen_paper_ids": [], "seen_titles": [], "stage": "curate", "target_count": 10,
            "selected_paper_ids": [], "selected_papers": [], "report": None, "chat_history": [],
            "web_articles_added": [], "pending_web_offer": None, "pending_report_update": None,
            "refinement_notes": [], "report_covered_web_article_count": 0,
            # deliberately no "web_article_provenance_by_url" key
        }

        with sqlite_checkpointer(db_path) as cp:
            graph = build_curation_graph(cp)
            config = {"configurable": {"thread_id": curation_thread_id("old-no-provenance-id")}}
            graph.invoke({"session": old_format_dict}, config=config)

            loaded = load_curation_session("old-no-provenance-id", cp)

        assert loaded is not None
        assert loaded.web_article_provenance_by_url == {}


def test_report_versions_round_trip_through_real_sqlite():
    p0 = _paper("p0")
    v1_report = {
        "findings": {"content": "V1 findings.", "cited_papers": [p0]},
        "limitations": {"content": "", "cited_papers": []},
        "future_scope": {"content": "", "cited_papers": []},
        "skipped_papers": [], "report_template": "analytical",
    }
    v2_report = {
        "findings": {"content": "V2 findings.", "cited_papers": [p0]},
        "limitations": {"content": "", "cited_papers": []},
        "future_scope": {"content": "", "cited_papers": []},
        "skipped_papers": [], "report_template": "expert",
    }
    versions = [
        {
            "version_id": "v1", "version_number": 1, "created_at": "2026-08-05T00:00:00+00:00",
            "report_template": "analytical", "generation_reason": "initial", "report": v1_report,
        },
        {
            "version_id": "v2", "version_number": 2, "created_at": "2026-08-05T00:05:00+00:00",
            "report_template": "expert", "generation_reason": "regenerate", "report": v2_report,
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        session = PaperPoolSession(
            topic="q", stage="synthesize", selected_papers=[p0], report=v2_report,
            report_versions=versions, active_report_version_id="v2",
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-versions", cp)
            loaded = load_curation_session("session-versions", cp)

        assert loaded is not None
        assert len(loaded.report_versions) == 2
        assert [v["version_id"] for v in loaded.report_versions] == ["v1", "v2"]
        assert [v["version_number"] for v in loaded.report_versions] == [1, 2]
        assert [v["generation_reason"] for v in loaded.report_versions] == ["initial", "regenerate"]
        assert [v["report_template"] for v in loaded.report_versions] == ["analytical", "expert"]
        assert loaded.report_versions[0]["created_at"] == "2026-08-05T00:00:00+00:00"


def test_active_report_version_id_round_trips():
    p0 = _paper("p0")
    report = {
        "findings": {"content": "F", "cited_papers": []},
        "limitations": {"content": "", "cited_papers": []},
        "future_scope": {"content": "", "cited_papers": []},
        "skipped_papers": [], "report_template": "analytical",
    }
    version = {
        "version_id": "the-active-one", "version_number": 1, "created_at": None,
        "report_template": "analytical", "generation_reason": "initial", "report": report,
    }

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        session = PaperPoolSession(
            topic="q", stage="synthesize", selected_papers=[p0], report=report,
            report_versions=[version], active_report_version_id="the-active-one",
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-active-id", cp)
            loaded = load_curation_session("session-active-id", cp)

        assert loaded is not None
        assert loaded.active_report_version_id == "the-active-one"


def test_report_version_bodies_preserve_sections_references_and_report_template():
    """Each version's own `report` dict must round-trip through the SAME
    machinery session.report already relies on (_serialize_report/
    _deserialize_report reused per-version) -- sections, references, and
    cited_papers/cited_web_articles must all survive intact, not just
    the version envelope fields."""
    p0 = _paper("p0")
    web = WebArticle(title="A Survey", url="https://example.com/survey", snippet="s", published_date=None, source_domain="example.com")
    report = {
        "executive_summary": {
            "content": "Per [1].", "cited_papers": [p0], "cited_web_articles": [web], "reference_numbers": [1, 2],
        },
        "findings": {"content": "Per [1].", "cited_papers": [p0], "cited_web_articles": [], "reference_numbers": [1]},
        "limitations": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
        "future_scope": {"content": "", "cited_papers": [], "cited_web_articles": [], "reference_numbers": []},
        "skipped_papers": [],
        "references": [
            {"number": 1, "kind": "paper", "paper_id": "p0", "url": None, "title": "Paper p0", "formatted": "x", "link_url": None},
            {"number": 2, "kind": "web", "paper_id": None, "url": "https://example.com/survey", "title": "A Survey", "formatted": "x", "link_url": "https://example.com/survey"},
        ],
        "sections": [
            {"key": "executive_summary", "title": "Executive Summary", "content": "Per [1].", "reference_numbers": [1, 2]},
        ],
        "report_template": "foundational",
    }
    version = {
        "version_id": "v1", "version_number": 1, "created_at": None,
        "report_template": "foundational", "generation_reason": "initial", "report": report,
    }

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        session = PaperPoolSession(
            topic="q", stage="synthesize", selected_papers=[p0], report=report,
            report_versions=[version], active_report_version_id="v1",
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-version-body", cp)
            loaded = load_curation_session("session-version-body", cp)

        assert loaded is not None
        loaded_report = loaded.report_versions[0]["report"]
        assert loaded_report["executive_summary"]["content"] == "Per [1]."
        assert [p.paper_id for p in loaded_report["executive_summary"]["cited_papers"]] == ["p0"]
        assert len(loaded_report["executive_summary"]["cited_web_articles"]) == 1
        assert loaded_report["executive_summary"]["cited_web_articles"][0].url == "https://example.com/survey"
        assert [s["key"] for s in loaded_report["sections"]] == ["executive_summary"]
        assert [r["kind"] for r in loaded_report["references"]] == ["paper", "web"]
        assert loaded_report["report_template"] == "foundational"


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


# --- turn_history + stop_reason (curation-turn-history Phase 9b) ---

def test_turn_history_and_stop_reason_roundtrip_through_real_sqlite():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        session = PaperPoolSession(
            topic="q", reserve=[(_paper("p0"), 0.9)], stage="synthesize", stop_reason="user_stopped",
            turn_history=[
                {"turn_number": 1, "batch": [[_paper("p0").to_dict(), 0.9]], "refilled": False},
                {"turn_number": 2, "batch": [[_paper("p1").to_dict(), 0.8]], "refilled": True},
            ],
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-1", cp)

        with sqlite_checkpointer(db_path) as cp2:
            loaded = load_curation_session("session-1", cp2)

        assert loaded.stop_reason == "user_stopped"
        assert len(loaded.turn_history) == 2
        assert loaded.turn_history[0]["turn_number"] == 1
        assert loaded.turn_history[0]["refilled"] is False
        assert loaded.turn_history[1]["refilled"] is True
        assert loaded.turn_history[0]["batch"][0][0]["paper_id"] == "p0"


def test_loading_a_pre_phase9b_session_without_turn_history_falls_back_to_empty():
    """A checkpoint saved before this field existed has neither key at
    all -- simulated the same way the Phase 8 display_title backward-
    compat test does: invoking the graph directly with a hand-built dict."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        old_format_dict = {
            "topic": "old style session", "display_title": "old style session",
            "reserve": [], "cursor": 0, "seen_paper_ids": [], "seen_titles": [], "stage": "curate",
            "target_count": 10, "selected_paper_ids": [], "selected_papers": [], "report": None,
            "chat_history": [], "web_articles_added": [], "pending_web_offer": None,
            "pending_report_update": None, "refinement_notes": [], "report_covered_web_article_count": 0,
            # deliberately no "turn_history" or "stop_reason" keys
        }

        with sqlite_checkpointer(db_path) as cp:
            graph = build_curation_graph(cp)
            config = {"configurable": {"thread_id": curation_thread_id("old-id")}}
            graph.invoke({"session": old_format_dict}, config=config)

            loaded = load_curation_session("old-id", cp)

        assert loaded.turn_history == []
        assert loaded.stop_reason is None


# --- select_paper_from_history (curation-turn-history Phase 9c) ---

def _history_entry(turn_number: int, papers: list[Paper], refilled: bool = False) -> dict:
    return {
        "turn_number": turn_number,
        "batch": [[p.to_dict(), 1.0] for p in papers],
        "refilled": refilled,
    }


def test_select_paper_from_history_adds_it_to_selection_when_synthesize():
    session = PaperPoolSession(
        topic="q", stage="synthesize",
        turn_history=[_history_entry(1, [_paper("p0"), _paper("p1")])],
    )

    select_paper_from_history(session, "p1")

    assert session.selected_paper_ids == ["p1"]
    assert [p.paper_id for p in session.selected_papers] == ["p1"]


def test_select_paper_from_history_refuses_while_still_curating():
    """The architectural core of Phase 9c: this out-of-band mutation is
    only safe once the interrupt loop has actually ended."""
    session = PaperPoolSession(
        topic="q", stage="curate",
        turn_history=[_history_entry(1, [_paper("p0")])],
    )

    import pytest
    with pytest.raises(ValueError, match="not ready"):
        select_paper_from_history(session, "p0")

    assert session.selected_paper_ids == []


def test_select_paper_from_history_refuses_once_a_report_has_been_generated():
    """curation-editable-until-locked bug fix: stage=="synthesize" alone
    no longer means "safe to add" -- a report already exists (built from
    the selection AS IT WAS), so silently growing selected_paper_ids here
    would orphan the report from the current selection."""
    session = PaperPoolSession(
        topic="q", stage="synthesize",
        turn_history=[_history_entry(1, [_paper("p0"), _paper("p1")])],
        report={"findings": {"content": "f", "cited_papers": []}},
    )

    import pytest
    with pytest.raises(ValueError, match="report has already been generated"):
        select_paper_from_history(session, "p1")

    assert session.selected_paper_ids == []


def test_select_paper_from_history_refuses_once_chat_has_started():
    session = PaperPoolSession(
        topic="q", stage="synthesize",
        turn_history=[_history_entry(1, [_paper("p0"), _paper("p1")])],
        chat_history=[{"role": "user", "content": "hi"}],
    )

    import pytest
    with pytest.raises(ValueError, match="chat has already started"):
        select_paper_from_history(session, "p1")

    assert session.selected_paper_ids == []


def test_select_paper_from_history_raises_for_a_paper_never_actually_served():
    session = PaperPoolSession(topic="q", stage="synthesize", turn_history=[_history_entry(1, [_paper("p0")])])

    import pytest
    with pytest.raises(ValueError, match="was not found"):
        select_paper_from_history(session, "never-served")


def test_select_paper_from_history_is_a_silent_no_op_if_already_selected():
    session = PaperPoolSession(
        topic="q", stage="synthesize",
        selected_paper_ids=["p0"], selected_papers=[_paper("p0")],
        turn_history=[_history_entry(1, [_paper("p0"), _paper("p1")])],
    )

    select_paper_from_history(session, "p0")  # already selected -- must not duplicate

    assert session.selected_paper_ids == ["p0"]
    assert len(session.selected_papers) == 1


def test_select_paper_from_history_finds_a_paper_from_an_earlier_turn_not_just_the_latest():
    session = PaperPoolSession(
        topic="q", stage="synthesize",
        turn_history=[
            _history_entry(1, [_paper("p0")]),
            _history_entry(2, [_paper("p1")], refilled=True),
        ],
    )

    select_paper_from_history(session, "p0")  # turn 1, not the most recent turn

    assert session.selected_paper_ids == ["p0"]


def test_select_paper_from_history_can_exceed_target_count():
    """Approved design decision: no cap -- selection is free to exceed
    target_count when browsing history."""
    session = PaperPoolSession(
        topic="q", stage="synthesize", target_count=1,
        selected_paper_ids=["p0"], selected_papers=[_paper("p0")],
        turn_history=[_history_entry(1, [_paper("p0"), _paper("p1")])],
    )

    select_paper_from_history(session, "p1")

    assert session.selected_paper_ids == ["p0", "p1"]
    assert len(session.selected_paper_ids) > session.target_count


def test_select_paper_from_history_persists_through_real_sqlite():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        session = PaperPoolSession(
            topic="q", stage="synthesize",
            turn_history=[_history_entry(1, [_paper("p0"), _paper("p1")])],
        )

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-1", cp)
            reloaded = load_curation_session("session-1", cp)
            select_paper_from_history(reloaded, "p1")
            save_curation_session(reloaded, "session-1", cp)

            final = load_curation_session("session-1", cp)

        assert final.selected_paper_ids == ["p1"]


# --- reopen_curation_session (curation-editable-until-locked Phase 10c) ---

def test_reopen_curation_session_resets_stage_and_stop_reason_when_eligible():
    session = PaperPoolSession(
        topic="q", stage="synthesize", stop_reason="user_stopped",
        selected_paper_ids=["p0"], selected_papers=[_paper("p0")],
    )

    reopen_curation_session(session)

    assert session.stage == "curate"
    assert session.stop_reason is None
    assert session.selected_paper_ids == ["p0"]  # untouched


def test_reopen_curation_session_refuses_while_still_curating():
    session = PaperPoolSession(topic="q", stage="curate")

    import pytest
    with pytest.raises(ValueError, match="nothing to reopen"):
        reopen_curation_session(session)


def test_reopen_curation_session_refuses_if_report_already_generated():
    session = PaperPoolSession(
        topic="q", stage="synthesize",
        report={"findings": {"content": "f", "cited_papers": []}},
    )

    import pytest
    with pytest.raises(ValueError, match="report has already been generated"):
        reopen_curation_session(session)

    assert session.stage == "synthesize"  # unchanged on refusal


def test_reopen_curation_session_refuses_if_chat_already_started():
    session = PaperPoolSession(
        topic="q", stage="synthesize",
        chat_history=[{"role": "user", "content": "hi"}],
    )

    import pytest
    with pytest.raises(ValueError, match="chat has already started"):
        reopen_curation_session(session)

    assert session.stage == "synthesize"  # unchanged on refusal


def test_reopen_curation_session_persists_through_real_sqlite():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        session = PaperPoolSession(topic="q", stage="synthesize", stop_reason="user_stopped")

        with sqlite_checkpointer(db_path) as cp:
            save_curation_session(session, "session-1", cp)
            reloaded = load_curation_session("session-1", cp)
            reopen_curation_session(reloaded)
            save_curation_session(reloaded, "session-1", cp)

            final = load_curation_session("session-1", cp)

        assert final.stage == "curate"
        assert final.stop_reason is None


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
    test_turn_history_and_stop_reason_roundtrip_through_real_sqlite()
    test_loading_a_pre_phase9b_session_without_turn_history_falls_back_to_empty()
    test_select_paper_from_history_adds_it_to_selection_when_synthesize()
    test_select_paper_from_history_refuses_while_still_curating()
    test_select_paper_from_history_refuses_once_a_report_has_been_generated()
    test_select_paper_from_history_refuses_once_chat_has_started()
    test_select_paper_from_history_raises_for_a_paper_never_actually_served()
    test_select_paper_from_history_is_a_silent_no_op_if_already_selected()
    test_select_paper_from_history_finds_a_paper_from_an_earlier_turn_not_just_the_latest()
    test_select_paper_from_history_can_exceed_target_count()
    test_select_paper_from_history_persists_through_real_sqlite()
    test_reopen_curation_session_resets_stage_and_stop_reason_when_eligible()
    test_reopen_curation_session_refuses_while_still_curating()
    test_reopen_curation_session_refuses_if_report_already_generated()
    test_reopen_curation_session_refuses_if_chat_already_started()
    test_reopen_curation_session_persists_through_real_sqlite()
    print("All curation_session tests passed.")
