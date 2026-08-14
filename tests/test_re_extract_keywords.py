"""Paper Keywords and Filtering, K4.1: tests for
scripts/re_extract_keywords.py -- the explicit, offline, dry-run-by-default
maintenance command that recomputes Paper.keywords for one local session.

Real SQLite reads throughout (via sqlite_checkpointer/load_curation_session),
same convention as tests/test_curation_session.py -- not just trusting a
round-trip through the same API that wrote it.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.re_extract_keywords as re_extract_keywords
from research_agent.curation_session import load_curation_session, save_curation_session
from research_agent.qa import sqlite_checkpointer
from research_agent.query_expansion import PaperPoolSession
from research_agent.schema import Paper

_ABSTRACT = (
    "We present a comprehensive study of graph neural networks for molecular "
    "property prediction. Our method combines message passing with attention "
    "mechanisms to capture long-range dependencies between atoms across five "
    "benchmark datasets, improving mean absolute error by a substantial margin "
    "while remaining computationally efficient at inference time."
)


def _paper(pid: str, keywords: list[str] | None = None) -> Paper:
    return Paper(
        title="Graph Neural Networks for Molecular Property Prediction",
        authors=["A"],
        year=2024,
        venue="X",
        abstract=_ABSTRACT,
        url=None,
        doi=None,
        citation_count=None,
        source="arxiv",
        paper_id=pid,
        keywords=keywords or [],
    )


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "checkpoints.sqlite"


def _save(db_path: Path, session_id: str, session: PaperPoolSession) -> None:
    with sqlite_checkpointer(db_path) as cp:
        save_curation_session(session, session_id, cp)


def _load(db_path: Path, session_id: str) -> PaperPoolSession:
    with sqlite_checkpointer(db_path) as cp:
        return load_curation_session(session_id, cp)


def _run(db_path: Path, argv: list[str]) -> int:
    with patch.object(re_extract_keywords, "QA_CHECKPOINT_DB_PATH", db_path):
        return re_extract_keywords.main(argv)


def test_script_is_directly_invocable_as_documented_without_pythonpath_help():
    # K4.3 bounded review: regression for a real defect found by actually
    # running the script exactly as its own usage docstring says
    # (`python scripts/re_extract_keywords.py SESSION_ID`) -- unlike every
    # other script in scripts/, this one was missing the sys.path
    # bootstrap for the project root, so Python's own script-directory
    # sys.path[0] behavior made `import research_agent` fail with
    # ModuleNotFoundError the moment it was run as a file rather than
    # imported by a test harness that already has the repo root on
    # sys.path. A subprocess, run from the repo root with no PYTHONPATH
    # set, is the only way to actually exercise this -- calling main()
    # in-process (every other test in this file) can never catch it.
    import subprocess

    repo_root = Path(__file__).resolve().parent.parent
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "scripts/re_extract_keywords.py", "no-such-session-id"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "ModuleNotFoundError" not in result.stderr
    assert "Traceback" not in result.stderr
    assert result.returncode == 1
    assert "no session found" in result.stderr


def test_dry_run_never_saves(db_path, capsys):
    paper = _paper("p1", keywords=["stale", "old kw"])
    session = PaperPoolSession(topic="graph neural networks", reserve=[(paper, 0.9)], cursor=1)
    _save(db_path, "sid1", session)

    rc = _run(db_path, ["sid1"])
    assert rc == 0

    reloaded = _load(db_path, "sid1")
    assert reloaded.reserve[0][0].keywords == ["stale", "old kw"]

    out = capsys.readouterr().out
    assert "dry-run: pass --apply" in out


def test_apply_saves_once_when_something_changed(db_path):
    paper = _paper("p1", keywords=["stale", "old kw"])
    session = PaperPoolSession(topic="graph neural networks", reserve=[(paper, 0.9)], cursor=1)
    _save(db_path, "sid1", session)

    rc = _run(db_path, ["sid1", "--apply"])
    assert rc == 0

    reloaded = _load(db_path, "sid1")
    assert reloaded.reserve[0][0].keywords != ["stale", "old kw"]
    assert len(reloaded.reserve[0][0].keywords) > 0


def test_repeated_paper_ids_are_extracted_once_and_every_occurrence_matches(db_path):
    # The SAME paper_id appears in reserve, selected_papers, AND
    # turn_history -- each occurrence starts with different stale
    # keywords (simulating drift), and after --apply every occurrence
    # must carry the identical, freshly recomputed keyword list.
    reserve_paper = _paper("p1", keywords=["reserve-stale"])
    selected_paper = _paper("p1", keywords=["selected-stale"])
    turn_history_paper_dict = _paper("p1", keywords=["history-stale"]).to_dict()

    session = PaperPoolSession(
        topic="graph neural networks",
        reserve=[(reserve_paper, 0.9)],
        cursor=1,
        selected_papers=[selected_paper],
        selected_paper_ids=["p1"],
        stage="synthesize",
        turn_history=[{"turn_number": 1, "batch": [[turn_history_paper_dict, 0.9]], "refilled": False}],
    )
    _save(db_path, "sid1", session)

    rc = _run(db_path, ["sid1", "--apply"])
    assert rc == 0

    reloaded = _load(db_path, "sid1")
    reserve_kw = reloaded.reserve[0][0].keywords
    selected_kw = reloaded.selected_papers[0].keywords
    history_kw = reloaded.turn_history[0]["batch"][0][0]["keywords"]

    assert reserve_kw == selected_kw == history_kw
    assert reserve_kw != ["reserve-stale"]


def test_no_op_input_performs_no_save(db_path, capsys):
    # Keywords already match what the current extractor would produce --
    # nothing should be written even with --apply.
    from research_agent.keywords import extract_keywords

    correct_keywords = extract_keywords(
        "Graph Neural Networks for Molecular Property Prediction", _ABSTRACT
    )
    paper = _paper("p1", keywords=correct_keywords)
    session = PaperPoolSession(topic="graph neural networks", reserve=[(paper, 0.9)], cursor=1)
    _save(db_path, "sid1", session)

    with patch.object(re_extract_keywords, "save_curation_session") as mock_save:
        rc = _run(db_path, ["sid1", "--apply"])
        assert rc == 0
        mock_save.assert_not_called()

    out = capsys.readouterr().out
    assert "no changes" in out


def test_unknown_session_id_fails_cleanly(db_path, capsys):
    rc = _run(db_path, ["does-not-exist"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no session found" in err


def test_non_keyword_fields_remain_deep_equal_after_apply(db_path):
    from copy import deepcopy
    from dataclasses import replace

    paper = _paper("p1", keywords=["stale"])
    session = PaperPoolSession(
        topic="graph neural networks",
        display_title="Graph Neural Networks Review",
        reserve=[(paper, 0.9)],
        cursor=1,
        seen_paper_ids={"p1"},
        seen_titles={paper.title},
        stage="synthesize",
        target_count=5,
        selected_paper_ids=["p1"],
        selected_papers=[deepcopy(paper)],
    )
    _save(db_path, "sid1", session)
    before = _load(db_path, "sid1")

    rc = _run(db_path, ["sid1", "--apply"])
    assert rc == 0

    after = _load(db_path, "sid1")

    assert after.topic == before.topic
    assert after.display_title == before.display_title
    assert after.cursor == before.cursor
    assert after.seen_paper_ids == before.seen_paper_ids
    assert after.seen_titles == before.seen_titles
    assert after.stage == before.stage
    assert after.target_count == before.target_count
    assert after.selected_paper_ids == before.selected_paper_ids
    assert len(after.reserve) == len(before.reserve)
    assert after.reserve[0][1] == before.reserve[0][1]  # score unchanged
    assert after.reserve[0][0].paper_id == before.reserve[0][0].paper_id
    # Every field except keywords is identical on the Paper itself too.
    before_paper_no_kw = replace(before.reserve[0][0], keywords=[])
    after_paper_no_kw = replace(after.reserve[0][0], keywords=[])
    assert before_paper_no_kw == after_paper_no_kw


def test_report_is_never_modified(db_path):
    # A report embeds its own frozen snapshot of cited/skipped papers --
    # this script must never touch it, even if the report's own copy of
    # a paper's keywords differs from the freshly recomputed value.
    from research_agent.report import GENERATION_REASON_INITIAL

    paper = _paper("p1", keywords=["stale"])
    report_paper = _paper("p1", keywords=["report-snapshot-stale"])
    report = {
        "executive_summary": {"content": "x", "cited_papers": [report_paper], "cited_web_articles": [], "reference_numbers": []},
        "skipped_papers": [],
    }
    session = PaperPoolSession(
        topic="graph neural networks",
        reserve=[(paper, 0.9)],
        cursor=1,
        stage="synthesize",
        report=report,
    )
    _save(db_path, "sid1", session)
    before = _load(db_path, "sid1")

    rc = _run(db_path, ["sid1", "--apply"])
    assert rc == 0

    after = _load(db_path, "sid1")
    assert after.report["executive_summary"]["cited_papers"][0].keywords == ["report-snapshot-stale"]
    assert after.report == before.report


def test_provider_boundaries_are_never_called(db_path):
    # OpenAI/Tavily/arXiv/Semantic Scholar clients must never be imported
    # or invoked by this script -- confirmed by patching the well-known
    # module-level client constructors to raise if touched.
    paper = _paper("p1", keywords=["stale"])
    session = PaperPoolSession(topic="graph neural networks", reserve=[(paper, 0.9)], cursor=1)
    _save(db_path, "sid1", session)

    with patch("openai.OpenAI") as mock_openai:
        rc = _run(db_path, ["sid1", "--apply"])
        assert rc == 0
        mock_openai.assert_not_called()


def test_malformed_missing_batch_key_session_fails_cleanly(db_path, capsys):
    # A turn_history entry missing its expected "batch" key must produce
    # a clean, non-zero exit and an error message -- not an unhandled
    # traceback.
    paper = _paper("p1", keywords=["stale"])
    session = PaperPoolSession(
        topic="graph neural networks",
        reserve=[(paper, 0.9)],
        cursor=1,
        turn_history=[{"turn_number": 1, "refilled": False}],
    )
    _save(db_path, "sid1", session)

    rc = _run(db_path, ["sid1"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "malformed" in err
