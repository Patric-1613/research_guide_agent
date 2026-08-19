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
# K5D.2a (Codex MEDIUM finding): a prior version of
# test_turn_history_current_batch_and_selected_papers_share_the_filtered_keywords
# exhausted a too-small reserve and fell into a real refill/search/
# embedding path, touching this real, ignored Chroma database. Fixed
# (see that test's own docstring); this fingerprint proves it stays
# untouched by every test in this file going forward.
_REAL_CHROMA_DB_PATH = Path("data/chroma_db/chroma.sqlite3")
_REAL_CHROMA_DB_FINGERPRINT_BEFORE = fingerprint_usage_db(_REAL_CHROMA_DB_PATH)
_REAL_QA_CHECKPOINT_DB_PATH = Path("data/qa_checkpoints.sqlite")
_REAL_QA_CHECKPOINT_DB_FINGERPRINT_BEFORE = fingerprint_usage_db(_REAL_QA_CHECKPOINT_DB_PATH)


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
    """K5D.2a fix (Codex MEDIUM finding): the original version of this
    test served only 3 papers, so resuming (without stopping) exhausted
    the reserve (remaining() == 0) and routed into a REAL refill turn --
    real arXiv/Semantic Scholar HTTP calls and a real embeddings.create
    attempt against the fake client (confirmed by reproducing the exact
    failure before this fix: an AttributeError deep inside
    embeddings.py, with real arxiv request logs above it). Fixed by
    using 15 papers (5 remain in reserve after the first 10-paper batch,
    same margin test_only_displayed_papers_are_filtered_reserve_keeps_
    originals already uses), PLUS hard spies that fail the test
    immediately -- rather than quietly making a real network call again
    -- if refill/search/ranking/embedding is ever reached regardless."""
    from research_agent import query_expansion as qe_module

    def _must_not_be_called(name):
        def _raise(*_a, **_k):
            raise AssertionError(f"{name} must not be called: this test must stay inside the selected-state path")
        return _raise

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        papers = [_paper(f"p{i}", keywords=["one", "two"]) for i in range(15)]
        with patch.object(qe_module, "build_candidate_pool", side_effect=_must_not_be_called("build_candidate_pool")), \
             patch.object(qe_module, "rank_full_pool", side_effect=_must_not_be_called("rank_full_pool")), \
             patch("socket.create_connection", side_effect=AssertionError("network call attempted")), \
             sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(papers)), config={"client": _fake_client()})
            current_batch = result["__interrupt__"][0].value["batch"]
            assert len(current_batch) == 10
            pick_id = current_batch[0][0]["paper_id"]
            result = resume_curation_turn("s1", cp, picked_paper_ids=[pick_id], config={"client": _fake_client()})
            # Confirms the reserve genuinely was not exhausted (no refill
            # branch taken) -- the real proof is the patches above never
            # firing, but this also documents the margin directly.
            assert result["refilled"] is False

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
# Additional focused gaps from the K5D.2a review
# ---------------------------------------------------------------------------

def test_provider_timeout_retains_original_keywords(enabled):
    """A timeout is just one more Exception subclass to _call_provider_
    for_paper's own fail-open try/except -- proven explicitly, by name,
    since the review called it out separately from a generic provider
    error."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        papers = [_paper("p0", keywords=["one", "two"])]

        class _TimingOutCompletions:
            def parse(self, **kwargs):
                raise TimeoutError("simulated provider timeout")

        client = SimpleNamespace(chat=SimpleNamespace(completions=_TimingOutCompletions()))
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(papers)), config={"client": client})
    batch = result["__interrupt__"][0].value["batch"]
    assert batch[0][0]["keywords"] == ["one", "two"]


def test_token_aggregation_sums_only_the_successful_child_call(enabled, usage_db_path):
    """One paper succeeds (real usage numbers), one paper's provider call
    raises (usage=None, per telemetry.timed_child_call's own contract) --
    the paid_actions row's aggregate token totals must reflect only the
    successful call, per telemetry._sum_nullable's documented "sum the
    non-None entries, never treat a missing one as 0" behavior."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        papers = [_paper("p0", keywords=["kw0 one", "kw0 two"]), _paper("p1", keywords=["kw1 one", "kw1 two"])]

        class _MixedCompletions:
            def __init__(self):
                self.calls = 0

            def parse(self, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("simulated failure on the second paper")
                payload = json.loads(kwargs["messages"][1]["content"])
                rows = [{"candidate_id": c["candidate_id"], "decision": "keep"} for c in payload["candidates"]]
                parsed = kwargs["response_format"](results=rows)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))], usage=_Usage())

        client = SimpleNamespace(chat=SimpleNamespace(completions=_MixedCompletions()))
        with sqlite_checkpointer(db_path) as cp:
            start_curation_turn("s1", cp, _session_to_dict(_session(papers)), config={"client": client})

    rows = _paid_action_rows(usage_db_path)
    assert len(rows) == 1
    row = rows[0]
    child_calls = json.loads(row["child_calls_json"])
    assert len(child_calls) == 2
    outcomes = sorted(c["outcome"] for c in child_calls)
    assert outcomes == ["error", "success"]
    # Exactly one successful call's usage (_Usage: 12/4/16) is reflected;
    # the failed call's None usage never contributes a phantom 0.
    assert row["input_tokens"] == 12
    assert row["output_tokens"] == 4
    assert row["total_tokens"] == 16


def test_service_level_admission_rejection_yields_the_truthful_existing_error_not_a_fail_open_bypass(enabled, usage_db_path):
    """Confirms the existing UsageGuardRejection contract already
    correctly covers the NEW curation_keyword_filter guard call site,
    without redesigning anything: a session-scoped admission rejection
    during _serve_batch_node must propagate as UsageGuardRejection all
    the way out of research_agent.services.curation_core_service.
    start_curation (the same exception the API layer's existing
    centralized handler already maps to 429/409/503) -- never silently
    swallowed into a 200-shaped response with unfiltered keywords."""
    import research_agent.api as api  # must import before curation_core_service -- see api.py's own create_app() -> routers -> services import chain
    import research_agent.usage_guard as usage_guard_module
    from research_agent.admission import AdmissionDecision
    from research_agent.api_app.schemas import CurationStartRequest
    from research_agent.services import curation_core_service
    from research_agent.services.curation_helpers import _curation_config

    papers = [_paper(f"p{i}", keywords=["one", "two"]) for i in range(3)]

    def _reject_session_hourly(*_a, **_k):
        return AdmissionDecision(allowed=False, reason_code="session_hourly_limit_reached", retry_after_seconds=60)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        # usage_guard.py does `from research_agent.admission import ...
        # check_session_hourly_budget` (a by-value import, same gotcha
        # test_curation_api.py's own _client() docstring already notes
        # for USAGE_DB_PATH) -- must patch usage_guard's own bound name,
        # not admission.py's, or this silently never fires.
        with patch.object(curation_core_service.api, "build_candidate_pool", return_value=papers), \
             patch.object(curation_core_service.api, "rank_full_pool", return_value=([(p, 1.0) for p in papers], {})), \
             patch.object(curation_core_service.api, "canonicalize_topic", side_effect=lambda topic, client=None: topic), \
             patch.object(usage_guard_module, "check_session_hourly_budget", side_effect=_reject_session_hourly), \
             patch.object(curation_core_service, "_curation_config", side_effect=lambda: {**_curation_config(), "client": _fake_client()}), \
             sqlite_checkpointer(db_path) as cp:
            curation_core_service.api._state["client"] = _fake_client()
            curation_core_service.api._state["collection"] = None
            with pytest.raises(UsageGuardRejection, match="session_hourly_limit_reached"):
                curation_core_service.start_curation(CurationStartRequest(topic="t", target_count=10), cp)

    # The rejected action was never a fail-open "keep going" -- no
    # curation_keyword_filter row was ever persisted for it.
    assert [row for row in _paid_action_rows(usage_db_path) if row["action_type"] == "curation_keyword_filter"] == []


# ---------------------------------------------------------------------------
# Regression: no real network call, real DBs untouched
# ---------------------------------------------------------------------------

def test_real_usage_and_keyword_cache_dbs_untouched():
    assert fingerprint_usage_db(_REAL_USAGE_DB_PATH) == _REAL_USAGE_DB_FINGERPRINT_BEFORE
    assert fingerprint_usage_db(_REAL_KEYWORD_CACHE_PATH) == _REAL_KEYWORD_CACHE_FINGERPRINT_BEFORE


def test_real_qa_checkpoint_db_untouched():
    assert fingerprint_usage_db(_REAL_QA_CHECKPOINT_DB_PATH) == _REAL_QA_CHECKPOINT_DB_FINGERPRINT_BEFORE


def test_real_chroma_db_untouched():
    """Separate from the QA-checkpoint/usage/keyword-cache checks above:
    on a developer machine with a live `uvicorn --reload` process already
    serving research_agent.api:app (a real, independently-running dev
    server -- not started by this test suite, and not something a test
    suite should ever stop/restart), that server's own lifespan holds an
    open real Chroma connection and can legitimately touch this file's
    bytes (observed: identical size, different content/mtime -- consistent
    with routine WAL-checkpoint housekeeping, not new data) independently
    of anything any test here does. This assertion is still meaningful --
    in the common case (no such server racing this file) it's a real,
    exact regression guard -- but a failure here alone, with the other
    three real-DB fingerprint tests in this file passing, points at that
    external process, not at this test suite's own hermeticity. Every
    other proof in this file (the hard _must_not_be_called network/
    refill/rank spies, the disabled/empty-batch no-op tests, the
    cache-only-batch tests) is what actually establishes this test
    suite never reaches Chroma on its own."""
    assert fingerprint_usage_db(_REAL_CHROMA_DB_PATH) == _REAL_CHROMA_DB_FINGERPRINT_BEFORE
