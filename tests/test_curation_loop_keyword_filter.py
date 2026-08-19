"""K5D.2 integration tests: the optional Policy C keyword filter wired
into research_agent.curation_loop._serve_batch_node.

Reuses tests/test_curation_loop.py's own conventions exactly: the
autouse usage_db_path fixture redirects telemetry/admission/leases to a
fresh tmp_path database (no test here ever reads or writes the real
data/usage_telemetry.sqlite), and _paper()/_session() build a
PaperPoolSession without any real search/embedding/provider call.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import research_agent.admission as admission
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent import keyword_filter as kf
from research_agent.config import settings as settings_module
from research_agent.curation_loop import CurationLoopState, resume_curation_turn, start_curation_turn
from research_agent.curation_session import _session_to_dict
from research_agent.qa import sqlite_checkpointer
from research_agent.query_expansion import PaperPoolSession
from research_agent.schema import Paper
from research_agent.telemetry import init_usage_db
from research_agent.usage_guard import UsageGuardRejection
from tests._usage_db_fingerprint import fingerprint_usage_db

_REAL_USAGE_DB_PATH = telemetry.USAGE_DB_PATH
_REAL_USAGE_DB_FINGERPRINT_BEFORE = fingerprint_usage_db(_REAL_USAGE_DB_PATH)
_REAL_KEYWORD_CACHE_PATH = kf.CACHE_DB_PATH
_REAL_KEYWORD_CACHE_FINGERPRINT_BEFORE = fingerprint_usage_db(_REAL_KEYWORD_CACHE_PATH)


@pytest.fixture(autouse=True)
def usage_db_path(tmp_path, monkeypatch):
    db_path = tmp_path / "usage.sqlite"
    monkeypatch.setattr(telemetry, "USAGE_DB_PATH", db_path)
    monkeypatch.setattr(admission, "USAGE_DB_PATH", db_path)
    monkeypatch.setattr(leases, "USAGE_DB_PATH", db_path)
    init_usage_db(path=db_path).close()
    return db_path


@pytest.fixture(autouse=True)
def keyword_filter_cache_path(tmp_path, monkeypatch):
    cache_path = tmp_path / "keyword_filter_cache.sqlite"
    monkeypatch.setattr(kf, "CACHE_DB_PATH", cache_path)
    return cache_path


@pytest.fixture
def enabled(monkeypatch):
    """Enables the flag for this test only, with a small, deterministic
    concurrency bound -- never touches the real environment."""
    monkeypatch.setenv("KEYWORD_FILTER_POLICY_C_ENABLED", "true")
    monkeypatch.setenv("KEYWORD_FILTER_MAX_CONCURRENT_CALLS", "2")


def _paper(pid: str, keywords: list[str] | None = None) -> Paper:
    return Paper(
        title=f"Paper {pid}", authors=["A"], year=2024, venue="X",
        abstract=f"abstract {pid}", url=None, doi=None, citation_count=None,
        source="arxiv", paper_id=pid, keywords=keywords or [],
    )


def _session(papers: list[Paper], target_count: int = 100) -> PaperPoolSession:
    return PaperPoolSession(topic="q", reserve=[(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], target_count=target_count)


class _FakeCompletions:
    """Deterministic: removes the SECOND candidate phrase (sentence_fragment), keeps everything else."""

    def __init__(self):
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"])
        rows = [
            {"candidate_id": c["candidate_id"], "decision": "sentence_fragment" if i == 1 else "keep"}
            for i, c in enumerate(payload["candidates"])
        ]
        parsed = kwargs["response_format"](results=rows)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))], usage=_Usage())


class _Usage:
    prompt_tokens = 12
    completion_tokens = 4
    total_tokens = 16


def _fake_client() -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))


def _paid_action_rows(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = [dict(row) for row in conn.execute("SELECT * FROM paid_actions").fetchall()]
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Disabled path: byte-equivalent to before this feature existed
# ---------------------------------------------------------------------------

def test_disabled_by_default_keywords_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            papers = [_paper(f"p{i}", keywords=["kw one", "kw two"]) for i in range(3)]
            result = start_curation_turn("s1", cp, _session_to_dict(_session(papers)))
        batch = result["__interrupt__"][0].value["batch"]
        for paper_dict, _score in batch:
            assert paper_dict["keywords"] == ["kw one", "kw two"]


def test_disabled_flag_opens_no_paid_action(usage_db_path):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            papers = [_paper(f"p{i}", keywords=["kw one", "kw two"]) for i in range(3)]
            start_curation_turn("s1", cp, _session_to_dict(_session(papers)))
    assert _paid_action_rows(usage_db_path) == []


# ---------------------------------------------------------------------------
# Empty batch: zero side effects even when enabled
# ---------------------------------------------------------------------------

def test_empty_batch_performs_no_admission_cache_or_provider_work(enabled, usage_db_path):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch("research_agent.query_expansion.build_candidate_pool", return_value=[]), \
             patch("research_agent.query_expansion.rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: ([], {})), \
             sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn(
                "s1", cp, _session_to_dict(_session([], target_count=100)), config={"client": _fake_client()},
            )
    assert result["__interrupt__"][0].value["batch"] == []
    # An empty starting reserve legitimately triggers an unrelated
    # curation_refill action (pre-existing _refill_node behavior, out of
    # this feature's scope) -- what THIS test proves is that the empty
    # (post-refill, still-empty) batch never ALSO opens a
    # curation_keyword_filter action.
    assert [row for row in _paid_action_rows(usage_db_path) if row["action_type"] == "curation_keyword_filter"] == []


# ---------------------------------------------------------------------------
# Only displayed papers filtered; reserve/ranking untouched
# ---------------------------------------------------------------------------

def test_only_displayed_papers_are_filtered_reserve_keeps_originals(enabled, usage_db_path):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        papers = [_paper(f"p{i}", keywords=["one", "two"]) for i in range(15)]  # 10 served, 5 held back in reserve
        with sqlite_checkpointer(db_path) as cp:
            client = _fake_client()
            result = start_curation_turn("s1", cp, _session_to_dict(_session(papers)), config={"client": client})
        batch = result["__interrupt__"][0].value["batch"]
        assert len(batch) == 10
        for paper_dict, _score in batch:
            assert paper_dict["keywords"] == ["one"]  # "two" removed (sentence_fragment)

        # The unserved 5 in reserve were never sent to the model at all --
        # only 10 provider calls happened (checked separately below) -- and
        # their own Paper objects (accessible via get_curation_state) must
        # still carry the untouched original keywords.
        from research_agent.curation_loop import get_curation_state
        with sqlite_checkpointer(db_path) as cp:
            state = get_curation_state("s1", cp)
        unserved = state["session"].reserve[state["session"].cursor:]
        assert len(unserved) == 5
        for paper, _score in unserved:
            assert paper.keywords == ["one", "two"]


def test_ranking_order_and_scores_are_unaffected(enabled):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        papers = [_paper(f"p{i}", keywords=["one", "two"]) for i in range(5)]
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(papers)), config={"client": _fake_client()})
        batch = result["__interrupt__"][0].value["batch"]
        assert [p["paper_id"] for p, _score in batch] == [f"p{i}" for i in range(5)]
        assert [round(score, 6) for _p, score in batch] == [round(1.0 - i * 0.01, 6) for i in range(5)]


# ---------------------------------------------------------------------------
# turn_history / current_batch / selected reconstruction stay consistent
# ---------------------------------------------------------------------------

def test_turn_history_current_batch_and_selected_papers_share_the_filtered_keywords(enabled):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        papers = [_paper(f"p{i}", keywords=["one", "two"]) for i in range(3)]
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(papers)), config={"client": _fake_client()})
            current_batch = result["__interrupt__"][0].value["batch"]
            pick_id = current_batch[0][0]["paper_id"]
            result = resume_curation_turn("s1", cp, picked_paper_ids=[pick_id], config={"client": _fake_client()})

        session_dict = result["session"]
        turn_history_batch = session_dict["turn_history"][0]["batch"]
        assert turn_history_batch == current_batch
        selected = next(p for p in session_dict["selected_papers"] if p["paper_id"] == pick_id)
        assert selected["keywords"] == ["one"]


# ---------------------------------------------------------------------------
# Usage protection: cache-only vs. uncached batches
# ---------------------------------------------------------------------------

def test_cache_only_batch_uses_no_admission_lease_or_provider_telemetry(enabled, usage_db_path, keyword_filter_cache_path):
    papers = [_paper(f"p{i}", keywords=["one", "two"]) for i in range(2)]
    for paper in papers:
        plan = kf.plan_paper(paper.keywords, cache_path=keyword_filter_cache_path)
        kf._cache_set(plan.cache_key, {"C0": "keep", "C1": "keep"}, keyword_filter_cache_path)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"

        def _explode(*_a, **_k):
            raise AssertionError("provider must not be called when every paper is a cache hit")

        with patch.object(kf, "_call_provider_for_paper", side_effect=_explode), sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(papers)), config={"client": _fake_client()})

    batch = result["__interrupt__"][0].value["batch"]
    for paper_dict, _score in batch:
        assert paper_dict["keywords"] == ["one", "two"]  # both "keep"
    assert _paid_action_rows(usage_db_path) == []


def test_uncached_batch_opens_exactly_one_paid_action_with_correct_child_call_count(enabled, usage_db_path):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        papers = [_paper(f"p{i}", keywords=["one", "two"]) for i in range(4)]
        client = _fake_client()
        with sqlite_checkpointer(db_path) as cp:
            start_curation_turn("s1", cp, _session_to_dict(_session(papers)), config={"client": client})

    rows = _paid_action_rows(usage_db_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["action_type"] == "curation_keyword_filter"
    assert row["total_call_count"] == 4  # one provider call per uncached paper
    assert len(client.chat.completions.calls) == 4
    child_calls = json.loads(row["child_calls_json"])
    assert len(child_calls) == 4
    assert all(c["call_type"] == "curation_keyword_filter" and c["provider"] == "openai" for c in child_calls)


def test_nested_inside_curation_start_rolls_up_under_first_active_action(enabled, usage_db_path):
    """'First active action wins': curation_start's own guard has no
    subject/lease (per usage_guard.guard_paid_action's own docstring),
    so it's already open when _serve_batch_node runs on the very first
    turn -- the keyword-filter child calls must attach to THAT action,
    not create a second top-level curation_keyword_filter row."""
    from research_agent.usage_guard import guard_paid_action

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        papers = [_paper(f"p{i}", keywords=["one", "two"]) for i in range(3)]
        with guard_paid_action("curation_start"), sqlite_checkpointer(db_path) as cp:
            start_curation_turn("s1", cp, _session_to_dict(_session(papers)), config={"client": _fake_client()})

    rows = _paid_action_rows(usage_db_path)
    assert len(rows) == 1
    assert rows[0]["action_type"] == "curation_start"
    child_calls = json.loads(rows[0]["child_calls_json"])
    assert len(child_calls) == 3
    assert all(c["call_type"] == "curation_keyword_filter" for c in child_calls)


def test_continuation_turn_receives_its_own_admission_and_lease_protection(enabled, usage_db_path):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        # Distinct keywords per paper -- identical phrases across papers
        # would share one content-hashed cache entry, and batch 2 would
        # then be a cache-only (no-provider-call, no-guard) batch, which
        # is correct caching behavior but defeats what THIS test needs to
        # isolate: that a standalone continuation turn is independently
        # guarded, not that caching works (see test_keyword_filter.py's
        # own cache tests for that).
        papers = [_paper(f"p{i}", keywords=[f"kw{i} one", f"kw{i} two"]) for i in range(25)]
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(papers)), config={"client": _fake_client()})
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            # Second turn (continuation, not the initial start) -- no
            # outer guard is open here; this must still be independently
            # admitted and leased, not silently skipped.
            resume_curation_turn("s1", cp, picked_paper_ids=batch1_ids[:1], config={"client": _fake_client()})

    rows = _paid_action_rows(usage_db_path)
    assert len(rows) == 2
    assert [row["action_type"] for row in rows] == ["curation_keyword_filter", "curation_keyword_filter"]
    assert all(row["subject_type"] == "session" and row["subject_id"] == "s1" for row in rows)


def test_lease_excludes_a_concurrent_keyword_filter_call_for_the_same_session(enabled, usage_db_path):
    """Same proof shape as test_curation_loop.py's own concurrent-refill
    lease test: calling _serve_batch_node directly with two threads
    while the flag is enabled and the first call is still mid-provider-
    work, to deterministically exercise guard_paid_action's exclusion."""
    from research_agent.curation_loop import _serve_batch_node

    release_event = threading.Event()
    entered_event = threading.Event()
    papers = [_paper(f"p{i}", keywords=["one", "two"]) for i in range(3)]
    session = _session(papers)
    state: CurationLoopState = {
        "session": _session_to_dict(session), "current_batch": [], "stop_reason": None,
        "should_stop": False, "refilled": False, "force_refill": False,
    }
    config = {"configurable": {"thread_id": "curation-session:s1", "client": _fake_client()}}

    def _blocking_call(client, plan):
        entered_event.set()
        release_event.wait(timeout=5)
        return plan.original_keywords

    with patch.object(kf, "_call_provider_for_paper", side_effect=_blocking_call):
        outcomes = {}

        def _attempt(key, st):
            try:
                _serve_batch_node(st, config)
                outcomes[key] = "admitted"
            except UsageGuardRejection as exc:
                outcomes[key] = exc.reason_code

        t1 = threading.Thread(target=_attempt, args=("t1", state))
        t1.start()
        assert entered_event.wait(timeout=5)

        second_state: CurationLoopState = {
            "session": _session_to_dict(session), "current_batch": [], "stop_reason": None,
            "should_stop": False, "refilled": False, "force_refill": False,
        }
        _attempt("t2", second_state)

        release_event.set()
        t1.join(timeout=5)

    assert outcomes["t1"] == "admitted"
    assert outcomes["t2"] == "action_in_progress"


def test_admission_rejection_is_not_a_provider_failure_and_propagates_truthfully(enabled, usage_db_path):
    """The API's existing 429/409/503 truthfulness must be preserved --
    a usage-protection rejection must never be silently swallowed into a
    fail-open "keep the original keywords and continue" outcome."""
    from research_agent.curation_loop import _serve_batch_node

    papers = [_paper(f"p{i}", keywords=["one", "two"]) for i in range(2)]
    session = _session(papers)
    state: CurationLoopState = {
        "session": _session_to_dict(session), "current_batch": [], "stop_reason": None,
        "should_stop": False, "refilled": False, "force_refill": False,
    }
    config = {"configurable": {"thread_id": "curation-session:s1", "client": _fake_client()}}

    with patch("research_agent.usage_guard.check_admission", side_effect=UsageGuardRejection("session_hourly_limit_reached")):
        with pytest.raises(UsageGuardRejection, match="session_hourly_limit_reached"):
            _serve_batch_node(state, config)


def test_privacy_safe_telemetry_no_phrases_titles_ids_or_prompt_text(enabled, usage_db_path):
    """subject_id legitimately carries the real session_id -- that's the
    pre-existing, sanctioned admission/lease scoping mechanism every
    other action type (curation_refill included) already uses; it is
    NOT what "privacy-safe telemetry" governs here. What must stay
    content-free is the CHILD-CALL record: call_type/provider/model/
    tokens/cache_hit/latency/outcome/error_type only -- no phrase,
    title, abstract, paper ID, or prompt text has a field to go into."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        papers = [_paper("very-secret-paper-id-123", keywords=["a very distinctive secret keyword phrase", "second phrase"])]
        with sqlite_checkpointer(db_path) as cp:
            start_curation_turn("s1", cp, _session_to_dict(_session(papers)), config={"client": _fake_client()})

    rows = _paid_action_rows(usage_db_path)
    assert len(rows) == 1
    child_calls = json.loads(rows[0]["child_calls_json"])
    assert len(child_calls) == 1
    assert set(child_calls[0]) == {
        "call_type", "provider", "model", "input_tokens", "output_tokens", "total_tokens",
        "cache_hit", "latency_ms", "outcome", "error_type",
    }
    child_blob = json.dumps(child_calls)
    for forbidden in ("secret", "very-secret-paper-id-123", "abstract", "distinctive", "second phrase", "Paper very-secret"):
        assert forbidden not in child_blob


# ---------------------------------------------------------------------------
# Regression: no real network call, real DBs untouched
# ---------------------------------------------------------------------------

def test_real_usage_and_keyword_cache_dbs_untouched():
    assert fingerprint_usage_db(_REAL_USAGE_DB_PATH) == _REAL_USAGE_DB_FINGERPRINT_BEFORE
    assert fingerprint_usage_db(_REAL_KEYWORD_CACHE_PATH) == _REAL_KEYWORD_CACHE_FINGERPRINT_BEFORE
