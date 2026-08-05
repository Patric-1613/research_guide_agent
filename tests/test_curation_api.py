"""curation-api-and-ui Phase 6a: real HTTP-level tests for the new
/curation/* endpoints — actual TestClient requests, not direct calls into
curation_loop.py/curation_chat.py/report.py (those already have their own
thorough unit/live coverage from Phases 1-5; this file exists specifically
to prove the HTTP boundary itself: request/response shapes, status codes,
and cross-REQUEST persistence through a real checkpointer).

Deliberately hybrid on what's mocked, matching this project's existing
mocked-vs-real trade-off in test_api.py:
  - build_candidate_pool/rank_full_pool (network+embedding calls) are
    mocked — no reason to hit arXiv/Semantic Scholar/OpenAI for HTTP
    wiring tests.
  - The interrupt/resume loop itself (curation_loop.py's real graph,
    real LangGraph Interrupt objects, real SqliteSaver checkpointer
    against a temp file) runs FOR REAL through actual HTTP calls — this
    is the brief's explicitly flagged "least obvious to get right" part,
    and mocking it away would defeat the point of testing it at the HTTP
    layer at all.
  - generate_report_for_session/regenerate_report_with_new_sources/
    chat_turn (the LLM-calling leaf functions) are mocked at the api.py
    level, same convention test_api.py already uses for api.ask/
    api.generate_summary — but the mocks mutate the passed-in session
    object the same way the real functions do, so save_curation_session
    persistence is still genuinely exercised end-to-end via a follow-up
    GET within the same test, not just trusted.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import research_agent.api as api
from research_agent.qa import sqlite_checkpointer
from research_agent.schema import Paper, WebArticle
from research_agent.storage import init_db as real_init_db


def _paper(paper_id: str, title: str, abstract: str = "an abstract") -> Paper:
    return Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract=abstract, url=f"http://arxiv.org/abs/{paper_id}",
        doi=None, citation_count=None, source="arxiv", paper_id=paper_id,
    )


def _web_article(url: str, title: str) -> WebArticle:
    return WebArticle(title=title, url=url, snippet=f"Snippet for {title}.", published_date=None, source_domain="example.com")


def _make_test_db_override(db_path: Path):
    def _override():
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    return _override


def _make_test_checkpointer_override(cp_db_path: Path):
    def _override():
        with sqlite_checkpointer(cp_db_path) as cp:
            yield cp

    return _override


@contextmanager
def _client():
    """Same isolation rationale as test_api.py's own _client(): a fresh
    temp SQLite file per test for storage.py's search history, PLUS
    (new here) a fresh temp SQLite file per test for the curation
    checkpointer — genuinely real persistence, isolated per test, not
    shared with the real dev database or other tests."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.sqlite"
        cp_db_path = Path(tmp) / "test_checkpoints.sqlite"
        with patch.object(api, "init_db", lambda: real_init_db(db_path)), \
             patch.object(api, "search_web", return_value=[]), \
             patch.object(api, "OpenAI", return_value=MagicMock()), \
             patch.object(api, "canonicalize_topic", side_effect=lambda topic, client=None: topic):
            api.app.dependency_overrides[api.get_db_connection] = _make_test_db_override(db_path)
            api.app.dependency_overrides[api.get_curation_checkpointer] = _make_test_checkpointer_override(cp_db_path)
            try:
                with TestClient(api.app) as client:
                    yield client
            finally:
                api.app.dependency_overrides.clear()


def _ranked(papers: list[Paper]) -> list[tuple[Paper, float]]:
    return [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)]


# --- /curation/start + /curation/{id}/picks: the real interrupt/resume loop ---

def test_curation_start_returns_a_batch_and_a_fresh_session_id():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(12)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        resp = client.post("/curation/start", json={"topic": "test topic", "target_count": 5})

    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "curate"
    assert body["stop_reason"] is None
    assert len(body["batch"]) == 10  # BATCH_SIZE
    assert body["selected_paper_ids"] == []
    assert body["target_count"] == 5
    assert isinstance(body["session_id"], str) and len(body["session_id"]) > 0


def test_curation_start_with_no_papers_returns_404():
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=[]), \
         patch.object(api, "rank_full_pool", return_value=([], {})):
        resp = client.post("/curation/start", json={"topic": "nonexistent topic"})
    assert resp.status_code == 404


def test_curation_start_upstream_openai_failure_returns_clean_error_not_raw_500():
    """_upstream_error_guard's own contract, proven on a curation endpoint
    for the first time (existing coverage was only /search and /chat in
    test_api.py) -- confirmed before /curation/start moves out of api.py,
    same discipline as the query-expansion-branch test added before
    /search's own move. build_candidate_pool runs before canonicalize_
    topic in the real handler, so this never reaches the fixture's own
    canonicalize_topic patch."""
    import httpx
    from openai import APIConnectionError

    with _client() as client, \
         patch.object(
             api, "build_candidate_pool",
             side_effect=APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/x")),
         ):
        resp = client.post("/curation/start", json={"topic": "t"})

    assert resp.status_code == 503
    assert resp.json()["detail"] == {"error": "curation_start service unavailable"}


# --- display_title (curation-review-management Phase 8, item 5) ---

def test_curation_get_state_returns_canonicalized_display_title_distinct_from_raw_topic():
    """The approved design's core property: display_title is a SEPARATE
    field from topic, and topic itself is never touched by
    canonicalize_topic -- proven end-to-end through real HTTP requests,
    not just at the function level."""
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(12)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})), \
         patch.object(api, "canonicalize_topic", return_value="Automotive Engine Cooling Systems"):
        start_body = client.post("/curation/start", json={"topic": "cars cooling system", "target_count": 5}).json()
        session_id = start_body["session_id"]

        resp = client.get(f"/curation/{session_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["topic"] == "cars cooling system"
    assert body["display_title"] == "Automotive Engine Cooling Systems"


def test_curation_start_does_not_canonicalize_when_no_papers_are_found():
    """Ordering choice: canonicalize_topic runs AFTER confirming papers
    actually exist for this topic, so a topic that returns nothing never
    spends an extra LLM call for a session that's about to 404 anyway."""
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=[]), \
         patch.object(api, "rank_full_pool", return_value=([], {})), \
         patch.object(api, "canonicalize_topic") as mock_canonicalize:
        client.post("/curation/start", json={"topic": "nonexistent topic"})

    mock_canonicalize.assert_not_called()


def test_curation_list_reviews_includes_display_title():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(12)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})), \
         patch.object(api, "canonicalize_topic", return_value="Automotive Engine Cooling Systems"):
        client.post("/curation/start", json={"topic": "cars cooling system", "target_count": 5})

        resp = client.get("/curation/reviews")

    assert resp.status_code == 200
    reviews = resp.json()
    assert any(r["topic"] == "cars cooling system" and r["display_title"] == "Automotive Engine Cooling Systems" for r in reviews)


def test_curation_picks_resumes_the_real_interrupt_and_returns_the_next_batch():
    """The core interrupt/resume HTTP shape: submitting picks via a
    genuinely separate HTTP request (not a shared Python object) must
    resume the REAL LangGraph interrupt and hand back a new batch."""
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(25)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 12}).json()
        session_id = start_body["session_id"]
        picks = [p["paper_id"] for p in start_body["batch"][:4]]

        resp = client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks})

    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "curate"  # target_count=12, only 4 picked -> still curating
    assert body["stop_reason"] is None
    assert body["selected_paper_ids"] == picks
    assert len(body["batch"]) == 10  # next batch presented


def test_curation_picks_reaching_target_does_not_transition_to_synthesize():
    """curation-editable-until-locked Phase 10b: hitting target_count is
    no longer a hard stop -- only an explicit stop=True locks the review
    into synthesize stage. Reaching target just keeps curation open, with
    a fresh batch still presented."""
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(15)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 3}).json()
        session_id = start_body["session_id"]
        picks = [p["paper_id"] for p in start_body["batch"][:3]]

        resp = client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks})

    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "curate"
    assert body["stop_reason"] is None
    assert len(body["batch"]) == 5  # 15 total - 10 served turn 1 = 5 left, still curating
    assert body["selected_paper_ids"] == picks


def test_curation_picks_with_explicit_stop_transitions_to_synthesize():
    """The ONLY way to lock a review into synthesize stage now: an
    explicit stop=True, regardless of whether target_count was reached."""
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(15)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 3}).json()
        session_id = start_body["session_id"]
        picks = [p["paper_id"] for p in start_body["batch"][:3]]

        resp = client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks, "stop": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "synthesize"
    assert body["stop_reason"] == "user_stopped"
    assert body["batch"] == []
    assert body["selected_paper_ids"] == picks


def test_curation_picks_on_unknown_session_id_returns_404():
    with _client() as client:
        resp = client.post("/curation/does-not-exist/picks", json={"picked_paper_ids": []})
    assert resp.status_code == 404


def test_curation_picks_after_curation_already_finished_returns_400():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(15)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 2}).json()
        session_id = start_body["session_id"]
        picks = [p["paper_id"] for p in start_body["batch"][:2]]
        client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks, "stop": True})  # finishes curation

        resp = client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": []})

    assert resp.status_code == 400


def test_curation_turn_response_surfaces_refilled_flag_for_real_across_a_genuine_refill():
    """The turn-feed divider's "from existing pool" vs "triggered a new
    search" label depends on this being real per-turn backend truth, not
    a client-side guess -- confirmed through the actual HTTP layer, not
    just curation_loop.py's own unit test for the same property."""
    from research_agent import query_expansion as qe_module

    # Exactly BATCH_SIZE papers: serving turn 1's entire batch drains
    # remaining to 0 regardless of pick count -- curation-turn-history
    # Phase 9d narrowed the auto-refill trigger to true exhaustion, not
    # just "< BATCH_SIZE".
    initial_papers = [_paper(f"p{i}", f"Paper {i}") for i in range(10)]
    fresh_papers = [_paper(f"new{i}", f"New Paper {i}") for i in range(8)]

    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=initial_papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(initial_papers), {})):
        # target_count=30 (the request model's max) -- comfortably unreachable
        # with only 5 picks below, so the turn genuinely continues into a
        # refill rather than hitting target_met first.
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 30}).json()
        assert start_body["refilled"] is False  # reserve exactly covers one batch, no refill needed yet
        assert start_body["reserve_remaining"] == 0  # 10 total - 10 served this turn
        session_id = start_body["session_id"]
        picks = [p["paper_id"] for p in start_body["batch"][:5]]  # remaining after: 10-10=0

        with patch.object(qe_module, "build_candidate_pool", return_value=fresh_papers), \
             patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                 [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
             )):
            resp = client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks})

    assert resp.status_code == 200
    body = resp.json()
    assert body["refilled"] is True
    # refill_pool merges the 0 unserved (reserve was fully drained) + 8
    # genuinely-new fresh papers -> reserve of 8, cursor resets to 0, this
    # turn serves all 8 (fewer than BATCH_SIZE exist) -> 0 remain.
    assert body["reserve_remaining"] == 0


# --- /curation/{id}/picks: refinement (Phase 6f) ---

def test_curation_picks_with_refinement_forces_a_real_refill_and_persists_across_a_separate_get_request():
    """The core Phase 6f-2 property through the actual HTTP layer: typed
    refinement text must force a fresh search even when the pool doesn't
    need one yet, and the applied refinement must be visible via a
    genuinely separate GET request afterward (not just the picks
    response), same refresh-persistence standard as everything else."""
    from research_agent import query_expansion as qe_module

    initial_papers = [_paper(f"p{i}", f"Paper {i}") for i in range(25)]
    fresh_papers = [_paper(f"new{i}", f"New Paper {i}") for i in range(8)]

    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=initial_papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(initial_papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 30}).json()
        assert start_body["refinement_notes"] == []
        session_id = start_body["session_id"]
        # remaining after this pick: 25-10=15, well above BATCH_SIZE=10 --
        # a plain pick here would NOT trigger a refill on its own.
        picks = [p["paper_id"] for p in start_body["batch"][:2]]

        with patch.object(qe_module, "build_candidate_pool", return_value=fresh_papers) as mock_build, \
             patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                 [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
             )):
            resp = client.post(
                f"/curation/{session_id}/picks",
                json={"picked_paper_ids": picks, "refinement": "focus on more recent work"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["refilled"] is True
        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs["refinement_notes"] == ["focus on more recent work"]
        assert body["refinement_notes"] == ["focus on more recent work"]

        state_resp = client.get(f"/curation/{session_id}")

    assert state_resp.json()["refinement_notes"] == ["focus on more recent work"]


def test_curation_picks_without_refinement_does_not_force_a_refill():
    from research_agent import query_expansion as qe_module

    initial_papers = [_paper(f"p{i}", f"Paper {i}") for i in range(25)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=initial_papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(initial_papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 30}).json()
        session_id = start_body["session_id"]
        picks = [p["paper_id"] for p in start_body["batch"][:2]]

        # Patched at the module _refill_node actually calls (query_expansion's
        # own reference), not api.py's -- api.build_candidate_pool is only
        # ever used by /curation/start, never by a mid-curation refill.
        with patch.object(qe_module, "build_candidate_pool") as mock_build_unused:
            resp = client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks})

    assert resp.status_code == 200
    assert resp.json()["refilled"] is False
    assert resp.json()["refinement_notes"] == []
    mock_build_unused.assert_not_called()


def test_curation_picks_with_request_refill_forces_a_search_through_the_real_http_layer():
    """curation-turn-history Phase 9d: the explicit "search for more now"
    action, confirmed through the actual HTTP layer, not just
    curation_loop.py's own unit test for the same property."""
    from research_agent import query_expansion as qe_module

    initial_papers = [_paper(f"p{i}", f"Paper {i}") for i in range(25)]
    fresh_papers = [_paper(f"new{i}", f"New Paper {i}") for i in range(8)]

    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=initial_papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(initial_papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 30}).json()
        session_id = start_body["session_id"]
        # remaining after this pick: 25-10=15, comfortably above zero --
        # a plain pick here would NOT trigger a refill on its own.
        picks = [p["paper_id"] for p in start_body["batch"][:2]]

        with patch.object(qe_module, "build_candidate_pool", return_value=fresh_papers) as mock_build, \
             patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                 [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
             )):
            resp = client.post(
                f"/curation/{session_id}/picks",
                json={"picked_paper_ids": picks, "request_refill": True},
            )

    assert resp.status_code == 200
    assert resp.json()["refilled"] is True
    mock_build.assert_called_once()


# --- GET /curation/reviews: the left-panel reviews list ---

def test_curation_list_reviews_returns_empty_with_no_sessions():
    with _client() as client:
        resp = client.get("/curation/reviews")
    assert resp.status_code == 200
    assert resp.json() == []


def test_curation_list_reviews_reflects_real_sessions_via_genuinely_separate_requests():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(12)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        client.post("/curation/start", json={"topic": "Topic One", "target_count": 5})
        client.post("/curation/start", json={"topic": "Topic Two", "target_count": 3})

        resp = client.get("/curation/reviews")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    topics = {r["topic"] for r in body}
    assert topics == {"Topic One", "Topic Two"}
    for review in body:
        assert review["stage"] == "curate"
        assert review["selected_count"] == 0
        assert review["has_report"] is False
        assert review["has_chat"] is False


def test_curation_list_reviews_flags_report_and_chat_progress():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        resp_before = client.get("/curation/reviews")
        assert resp_before.json()[0]["has_report"] is False

        fake_report = {
            "findings": {"content": "f", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [],
        }
        with patch.object(api, "generate_report_for_session", return_value=fake_report):
            client.post(f"/curation/{session_id}/report")

        resp_after = client.get("/curation/reviews")

    review = resp_after.json()[0]
    assert review["session_id"] == session_id
    assert review["has_report"] is True
    assert review["has_chat"] is False
    assert review["selected_count"] == len(pick_ids)


# --- GET /curation/{id}: the refresh-persistence endpoint ---

def test_curation_get_state_reflects_a_pending_batch_from_a_separate_request():
    """Proves state comes from the checkpointer, not anything held only in
    the request that started curation -- start and get-state are two
    independent TestClient calls here, exactly mirroring what a browser
    refresh does at the HTTP layer."""
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(12)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        session_id = client.post("/curation/start", json={"topic": "t", "target_count": 5}).json()["session_id"]

        resp = client.get(f"/curation/{session_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "curate"
    assert body["pending_batch"] is not None
    assert len(body["pending_batch"]) == 10
    assert body["reserve_remaining"] == 2  # 12 total - 10 served
    assert body["selected_papers"] == []
    assert body["report"] is None


def test_curation_get_state_after_finishing_has_no_pending_batch():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(15)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 2}).json()
        session_id = start_body["session_id"]
        picks = [p["paper_id"] for p in start_body["batch"][:2]]
        client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks, "stop": True})

        resp = client.get(f"/curation/{session_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "synthesize"
    assert body["pending_batch"] is None
    assert sorted(p["paper_id"] for p in body["selected_papers"]) == sorted(picks)


def test_curation_get_state_unknown_session_id_returns_404():
    with _client() as client:
        resp = client.get("/curation/does-not-exist")
    assert resp.status_code == 404


# --- turn_history + stop_reason (curation-turn-history Phase 9b) ---

def test_curation_get_state_exposes_turn_history_and_stop_reason():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(15)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 2}).json()
        session_id = start_body["session_id"]
        picks = [p["paper_id"] for p in start_body["batch"][:2]]
        client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks, "stop": True})

        resp = client.get(f"/curation/{session_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "user_stopped"
    assert len(body["turn_history"]) == 1
    assert body["turn_history"][0]["turn_number"] == 1
    assert body["turn_history"][0]["refilled"] is False
    assert sorted(p["paper_id"] for p in body["turn_history"][0]["batch"]) == sorted(
        p["paper_id"] for p in start_body["batch"]
    )


def test_curation_get_state_stop_reason_is_none_mid_curation():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(15)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        session_id = client.post("/curation/start", json={"topic": "t", "target_count": 12}).json()["session_id"]

        resp = client.get(f"/curation/{session_id}")

    assert resp.json()["stop_reason"] is None


# --- POST /curation/{id}/select-from-history (curation-turn-history Phase 9c) ---

def test_select_from_history_adds_a_paper_after_curation_finished_short_of_target():
    """The exact real-world scenario reported: the pool ran genuinely dry
    (exhausted) short of target_count. curation-editable-until-locked
    Phase 10b: that no longer auto-stops curation -- the search just
    comes back with an empty batch, still open -- so the user has to
    explicitly stop before select-from-history (synthesize-stage-only)
    becomes usable."""
    from research_agent import query_expansion as qe_module

    # Exactly BATCH_SIZE papers: the very first batch consumes the ENTIRE
    # reserve (remaining=0), so the refill this pick triggers has an
    # empty unserved_tail to merge with -- if it also finds nothing new
    # (mocked below), the result is a genuinely empty next batch, not
    # just a small one.
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(10)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 20}).json()
        session_id = start_body["session_id"]
        served_ids = [p["paper_id"] for p in start_body["batch"]]
        picks = served_ids[:3]  # pick only 3 -- well short of target_count=20

        # remaining after this pick: 10-10=0 < BATCH_SIZE=10 -> refill_pool
        # runs internally; mocked here (via query_expansion's own module
        # reference, not api's) to find nothing new -> merged reserve is
        # genuinely empty -> the next batch is empty, but curation stays open.
        with patch.object(qe_module, "build_candidate_pool", return_value=[]), \
             patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                 [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
             )):
            picks_resp = client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks}).json()
        assert picks_resp["stop_reason"] is None
        assert picks_resp["stage"] == "curate"
        assert picks_resp["batch"] == []  # nothing new found, but still editable

        # The user decides to stop here rather than keep searching.
        stop_resp = client.post(
            f"/curation/{session_id}/picks", json={"picked_paper_ids": [], "stop": True},
        ).json()
        assert stop_resp["stage"] == "synthesize"

        not_yet_picked = [pid for pid in served_ids if pid not in picks][0]
        resp = client.post(f"/curation/{session_id}/select-from-history", json={"paper_id": not_yet_picked})

        state = client.get(f"/curation/{session_id}").json()

    assert resp.status_code == 200
    assert resp.json()["selected_paper_ids"] == picks + [not_yet_picked]
    assert sorted(p["paper_id"] for p in state["selected_papers"]) == sorted(picks + [not_yet_picked])


def test_select_from_history_refuses_while_curation_still_in_progress():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(12)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 20}).json()
        session_id = start_body["session_id"]
        served_id = start_body["batch"][0]["paper_id"]

        resp = client.post(f"/curation/{session_id}/select-from-history", json={"paper_id": served_id})

    assert resp.status_code == 400
    assert "not ready" in resp.json()["detail"]


def test_select_from_history_refuses_once_a_report_has_been_generated():
    """curation-editable-until-locked bug fix: the turn-history browser
    was letting a paper be silently added here even after a report
    already existed for a DIFFERENT selection -- reproduces the exact
    reported symptom (clicking "+ Add to review" from Browse past turns
    while a report/chat already exists must be refused, not accepted)."""
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        served_ids = [p["paper_id"] for p in client.get(f"/curation/{session_id}").json()["turn_history"][0]["batch"]]
        not_yet_picked = [pid for pid in served_ids if pid not in pick_ids][0]

        fake_report = {
            "findings": {"content": "f", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [],
        }
        with patch.object(api, "generate_report_for_session", return_value=fake_report):
            client.post(f"/curation/{session_id}/report")

        resp = client.post(f"/curation/{session_id}/select-from-history", json={"paper_id": not_yet_picked})

    assert resp.status_code == 400
    assert "report has already been generated" in resp.json()["detail"]


def test_select_from_history_refuses_once_chat_has_started():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        served_ids = [p["paper_id"] for p in client.get(f"/curation/{session_id}").json()["turn_history"][0]["batch"]]
        not_yet_picked = [pid for pid in served_ids if pid not in pick_ids][0]

        fake_result = {"answer": "an answer", "answerable": True, "cited_papers": [], "cited_web_articles": []}

        def _fake_chat_turn(session, message, client=None, **kwargs):
            # Mutate the REAL session object the same way chat_turn()
            # actually does, so chat_history is genuinely non-empty
            # afterward -- not just a mocked return value.
            session.chat_history.append({"role": "user", "content": message})
            session.chat_history.append({"role": "assistant", "content": fake_result["answer"]})
            return fake_result

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn):
            client.post(f"/curation/{session_id}/chat", json={"message": "hi"})

        resp = client.post(f"/curation/{session_id}/select-from-history", json={"paper_id": not_yet_picked})

    assert resp.status_code == 400
    assert "chat has already started" in resp.json()["detail"]


def test_select_from_history_unknown_paper_id_returns_400():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(12)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 1}).json()
        session_id = start_body["session_id"]
        client.post(
            f"/curation/{session_id}/picks",
            json={"picked_paper_ids": [start_body["batch"][0]["paper_id"]], "stop": True},
        )

        resp = client.post(f"/curation/{session_id}/select-from-history", json={"paper_id": "never-served"})

    assert resp.status_code == 400
    assert "was not found" in resp.json()["detail"]


def test_select_from_history_unknown_session_id_returns_404():
    with _client() as client:
        resp = client.post("/curation/does-not-exist/select-from-history", json={"paper_id": "p0"})
    assert resp.status_code == 404


def test_select_from_history_can_exceed_target_count():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(12)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 1}).json()
        session_id = start_body["session_id"]
        served_ids = [p["paper_id"] for p in start_body["batch"]]
        # meets target_count=1, but that alone no longer stops curation --
        # stop explicitly to reach synthesize stage.
        client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [served_ids[0]], "stop": True})

        resp = client.post(f"/curation/{session_id}/select-from-history", json={"paper_id": served_ids[1]})

    assert resp.status_code == 200
    assert resp.json()["selected_paper_ids"] == [served_ids[0], served_ids[1]]


# --- POST /curation/{id}/reopen (curation-editable-until-locked Phase 10c) ---

def test_curation_reopen_resumes_active_curation_preserving_prior_state():
    """The exact scenario this whole phase is for: a review stopped
    short of target, with nothing chatted/reported yet, can go back to
    active curation -- picking up from where cursor/selections left off,
    not restarting."""
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(15)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 10}).json()
        session_id = start_body["session_id"]
        served_ids = [p["paper_id"] for p in start_body["batch"]]
        picks = served_ids[:3]
        client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks, "stop": True})
        assert client.get(f"/curation/{session_id}").json()["stage"] == "synthesize"

        resp = client.post(f"/curation/{session_id}/reopen")

        # Persisted for real, via a genuinely separate GET request.
        state = client.get(f"/curation/{session_id}").json()

    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "curate"
    assert body["stop_reason"] is None
    assert body["selected_paper_ids"] == picks
    assert len(body["batch"]) == 5  # 15 total - 10 served turn 1 = 5 remain
    assert state["stage"] == "curate"
    assert state["pending_batch"] is not None


def test_curation_reopen_while_still_curating_returns_400():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(12)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        session_id = client.post("/curation/start", json={"topic": "t", "target_count": 5}).json()["session_id"]

        resp = client.post(f"/curation/{session_id}/reopen")

    assert resp.status_code == 400
    assert "nothing to reopen" in resp.json()["detail"]


def test_curation_reopen_after_report_generated_returns_400():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        fake_report = {
            "findings": {"content": "f", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [],
        }
        with patch.object(api, "generate_report_for_session", return_value=fake_report):
            client.post(f"/curation/{session_id}/report")

        resp = client.post(f"/curation/{session_id}/reopen")

    assert resp.status_code == 400
    assert "report has already been generated" in resp.json()["detail"]


def test_curation_reopen_after_chat_started_returns_400():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        fake_result = {
            "answer": "Per [Paper 1], X is true.", "answerable": True,
            "cited_papers": [], "cited_web_articles": [],
        }

        def _fake_chat_turn(session, message, client=None, **kwargs):
            session.chat_history.append({"role": "user", "content": message})
            session.chat_history.append({"role": "assistant", "content": fake_result["answer"]})
            return fake_result

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn):
            client.post(f"/curation/{session_id}/chat", json={"message": "hi"})

        resp = client.post(f"/curation/{session_id}/reopen")

    assert resp.status_code == 400
    assert "chat has already started" in resp.json()["detail"]


def test_curation_reopen_unknown_session_id_returns_404():
    with _client() as client:
        resp = client.post("/curation/does-not-exist/reopen")
    assert resp.status_code == 404


# --- DELETE /curation/{id}: delete/abandon a review (curation-review-management Phase 8, item 1) ---

def test_curation_delete_removes_the_session_for_real():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(12)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        session_id = client.post("/curation/start", json={"topic": "t", "target_count": 5}).json()["session_id"]

        # It's real and gettable before deletion.
        assert client.get(f"/curation/{session_id}").status_code == 200

        resp = client.delete(f"/curation/{session_id}")
        assert resp.status_code == 200
        assert resp.json() == {"session_id": session_id, "deleted": True}

        # Gone for real, via a genuinely separate GET request -- not just a
        # client-side assumption.
        assert client.get(f"/curation/{session_id}").status_code == 404
        # Also gone from the reviews list, not just the direct-get endpoint.
        reviews = client.get("/curation/reviews").json()
        assert all(r["session_id"] != session_id for r in reviews)


def test_curation_delete_unknown_session_id_returns_404():
    with _client() as client:
        resp = client.delete("/curation/does-not-exist")
    assert resp.status_code == 404


def test_curation_delete_does_not_affect_other_sessions():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(12)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        session_a = client.post("/curation/start", json={"topic": "Topic A", "target_count": 5}).json()["session_id"]
        session_b = client.post("/curation/start", json={"topic": "Topic B", "target_count": 5}).json()["session_id"]

        client.delete(f"/curation/{session_a}")

        assert client.get(f"/curation/{session_a}").status_code == 404
        assert client.get(f"/curation/{session_b}").status_code == 200


# --- /curation/{id}/report + /report/regenerate ---

def _finish_curation(client, target_count: int = 2, n_papers: int = 12) -> tuple[str, list[str]]:
    """curation-editable-until-locked Phase 10b: reaching target_count no
    longer ends curation on its own -- an explicit stop=True is now
    required to reach stage="synthesize", so every caller that needs a
    finished review (report/chat tests, etc.) must ask for that
    explicitly rather than relying on target_count as a side effect."""
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(n_papers)]
    with patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": target_count}).json()
        session_id = start_body["session_id"]
        picks = start_body["batch"][:target_count]
        pick_ids = [p["paper_id"] for p in picks]
        client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": pick_ids, "stop": True})
    return session_id, pick_ids


def test_curation_report_generates_and_persists_across_a_separate_get_request():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        # report-quality Phase R1: this fake_report is deliberately an
        # OLD-SHAPE dict -- no "references"/"reference_numbers" keys at
        # all, exactly what a pre-R1 generate_report_for_session() (or a
        # report loaded from before this phase) would return/persist.
        # Proves _report_to_out derives a references list for it on the
        # fly (via report.py's derive_legacy_references) rather than
        # crashing or omitting the field.
        fake_report = {
            "findings": {"content": "f", "cited_papers": [_paper(pick_ids[0], "Paper 0")]},
            "limitations": {"content": "l", "cited_papers": []},
            "future_scope": {"content": "fs", "cited_papers": []},
            "skipped_papers": [],
        }
        with patch.object(api, "generate_report_for_session", return_value=fake_report) as mock_gen:
            resp = client.post(f"/curation/{session_id}/report")
            resp2 = client.post(f"/curation/{session_id}/report")  # second call must not re-generate

        assert resp.status_code == 200
        assert resp.json()["findings"]["content"] == "f"
        assert resp2.status_code == 200
        mock_gen.assert_called_once()  # cached on the second call

        # An old-shape report dict (no references key at all) still
        # serializes cleanly -- a References list is derived, naming the
        # one cited paper, and cited_papers/cited_web_articles are still
        # present on the section (compatibility fields, unremoved).
        body = resp.json()
        assert [r["paper_id"] for r in body["references"]] == [pick_ids[0]]
        assert body["findings"]["reference_numbers"] == [1]
        assert body["findings"]["cited_papers"][0]["paper_id"] == pick_ids[0]
        assert body["findings"]["cited_web_articles"] == []

        # report-quality Phase R2A: the SAME old-shape dict (no "sections"
        # key either) still derives a 3-entry sections list, in order,
        # matching the legacy findings/limitations/future_scope fields --
        # and those legacy fields themselves stay populated, not dropped.
        assert [s["key"] for s in body["sections"]] == ["findings", "limitations", "future_scope"]
        assert body["sections"][0]["title"] == "Findings"
        assert body["sections"][0]["content"] == "f"
        assert body["limitations"]["content"] == "l"
        assert body["future_scope"]["content"] == "fs"

        # Persisted state visible via a genuinely separate request too.
        state_resp = client.get(f"/curation/{session_id}")
    assert state_resp.json()["report"]["findings"]["content"] == "f"
    assert [r["paper_id"] for r in state_resp.json()["report"]["references"]] == [pick_ids[0]]
    assert [s["key"] for s in state_resp.json()["report"]["sections"]] == ["findings", "limitations", "future_scope"]


def test_curation_report_before_curation_finished_returns_400():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(12)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        session_id = client.post("/curation/start", json={"topic": "t", "target_count": 5}).json()["session_id"]
        resp = client.post(f"/curation/{session_id}/report")
    assert resp.status_code == 400


def test_curation_report_unknown_session_id_returns_404():
    with _client() as client:
        resp = client.post("/curation/does-not-exist/report")
    assert resp.status_code == 404


def test_curation_report_regenerate_overwrites_persisted_report():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        first_report = {
            "findings": {"content": "v1", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [],
        }
        with patch.object(api, "generate_report_for_session", return_value=first_report):
            client.post(f"/curation/{session_id}/report")

        second_report = {
            "findings": {"content": "v2, now with a web source", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [],
        }
        with patch.object(api, "regenerate_report_with_new_sources", return_value=second_report) as mock_regen:
            resp = client.post(f"/curation/{session_id}/report/regenerate")
        mock_regen.assert_called_once()

        state_resp = client.get(f"/curation/{session_id}")

    assert resp.status_code == 200
    assert resp.json()["findings"]["content"] == "v2, now with a web source"
    assert state_resp.json()["report"]["findings"]["content"] == "v2, now with a web source"


# --- report-quality Phase R2C: report templates ---

def test_curation_report_generate_omitted_body_defaults_to_analytical():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        fake_report = {
            "findings": {"content": "f", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        with patch.object(api, "generate_report_for_session", return_value=fake_report) as mock_gen:
            resp = client.post(f"/curation/{session_id}/report")

    assert resp.status_code == 200
    assert resp.json()["report_template"] == "analytical"
    assert mock_gen.call_args.kwargs["report_template"] == "analytical"


def test_curation_report_generate_explicit_template_returns_that_template():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        fake_report = {
            "findings": {"content": "f", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "expert",
        }
        with patch.object(api, "generate_report_for_session", return_value=fake_report) as mock_gen:
            resp = client.post(f"/curation/{session_id}/report", json={"report_template": "expert"})

    assert resp.status_code == 200
    assert resp.json()["report_template"] == "expert"
    assert mock_gen.call_args.kwargs["report_template"] == "expert"


def test_curation_report_regenerate_omitted_template_preserves_existing():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        first_report = {
            "findings": {"content": "v1", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "foundational",
        }
        with patch.object(api, "generate_report_for_session", return_value=first_report):
            client.post(f"/curation/{session_id}/report")

        second_report = {**first_report, "findings": {"content": "v2", "cited_papers": []}}
        with patch.object(api, "regenerate_report_with_new_sources", return_value=second_report) as mock_regen:
            resp = client.post(f"/curation/{session_id}/report/regenerate")

    assert resp.status_code == 200
    assert resp.json()["report_template"] == "foundational"
    # None reaching report.py means "preserve" -- proven by the mock's
    # own recorded call args, not just the (mocked) response shape.
    assert mock_regen.call_args.kwargs["report_template"] is None


def test_curation_report_regenerate_explicit_template_switches():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        first_report = {
            "findings": {"content": "v1", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        with patch.object(api, "generate_report_for_session", return_value=first_report):
            client.post(f"/curation/{session_id}/report")

        switched_report = {**first_report, "findings": {"content": "v2", "cited_papers": []}, "report_template": "expert"}
        with patch.object(api, "regenerate_report_with_new_sources", return_value=switched_report) as mock_regen:
            resp = client.post(f"/curation/{session_id}/report/regenerate", json={"report_template": "expert"})

    assert resp.status_code == 200
    assert resp.json()["report_template"] == "expert"
    assert mock_regen.call_args.kwargs["report_template"] == "expert"


# --- report-quality Phase R4.1: optional, bounded refinement loop ---

def test_curation_report_generate_omitted_refinement_mode_does_not_refine():
    """Required test 12 (compatibility): an existing client posting {}
    (or omitting refinement_mode entirely) sees no refinement -- proven
    at the real API boundary, without mocking refine_report_if_
    requested away, since its own None/"off" branch never touches the
    OpenAI client at all (see report.py's own docstring)."""
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        fake_report = {
            "findings": {"content": "f", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        with patch.object(api, "generate_report_for_session", return_value=fake_report) as mock_gen:
            resp = client.post(f"/curation/{session_id}/report", json={})

    assert resp.status_code == 200
    assert resp.json()["refinement"] is None
    assert mock_gen.call_args.kwargs["report_template"] == "analytical"


def test_curation_report_generate_with_refinement_mode_single_calls_refine_report_if_requested():
    """Required test 10 (generate side): POST /curation/{id}/report
    accepts refinement_mode and threads it into refine_report_if_
    requested; the refined report (not the raw draft) is what the
    endpoint returns."""
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        fake_report = {
            "findings": {"content": "f", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        refined_report = {
            **fake_report, "findings": {"content": "refined", "cited_papers": []},
            "refinement": {"enabled": True, "rounds": 0, "initial_score": 90, "final_score": 90},
        }
        with patch.object(api, "generate_report_for_session", return_value=fake_report):
            with patch.object(api, "refine_report_if_requested", return_value=refined_report) as mock_refine:
                resp = client.post(f"/curation/{session_id}/report", json={"refinement_mode": "single"})

    assert resp.status_code == 200
    assert mock_refine.call_args.kwargs["refinement_mode"] == "single"
    assert resp.json()["findings"]["content"] == "refined"
    # report-quality Phase R4.2: issues/revision_instructions/section_scores
    # default via ReportRefinementOut's own field defaults even though the
    # mocked refine_report_if_requested return value only supplies R4.1's
    # original 4 keys -- proves no serializer change was needed for R4.2.
    assert resp.json()["refinement"] == {
        "enabled": True, "rounds": 0, "initial_score": 90, "final_score": 90,
        "issues": [], "revision_instructions": "", "section_scores": None,
    }


def test_curation_report_regenerate_with_refinement_mode_single_calls_refine_report_if_requested():
    """Required test 10 (regenerate side): POST /curation/{id}/report/
    regenerate accepts refinement_mode the same way."""
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        first_report = {
            "findings": {"content": "v1", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        with patch.object(api, "generate_report_for_session", return_value=first_report):
            client.post(f"/curation/{session_id}/report")

        second_report = {**first_report, "findings": {"content": "v2", "cited_papers": []}}
        refined_report = {
            **second_report, "findings": {"content": "v2 refined", "cited_papers": []},
            "refinement": {"enabled": True, "rounds": 1, "initial_score": 40, "final_score": None},
        }
        with patch.object(api, "regenerate_report_with_new_sources", return_value=second_report):
            with patch.object(api, "refine_report_if_requested", return_value=refined_report) as mock_refine:
                resp = client.post(f"/curation/{session_id}/report/regenerate", json={"refinement_mode": "single"})

    assert resp.status_code == 200
    assert mock_refine.call_args.kwargs["refinement_mode"] == "single"
    assert resp.json()["findings"]["content"] == "v2 refined"
    assert resp.json()["refinement"]["rounds"] == 1


def test_curation_chat_add_to_report_never_refines_even_if_refinement_mode_is_posted():
    """Required test 11: chat-triggered regeneration (add-to-report)
    never refines -- R4.1 only wires into the two explicit generate/
    regenerate endpoints. This endpoint's own request schema has no
    refinement_mode field at all; posting one anyway is silently
    ignored (same as any other unknown JSON key), proven here by
    confirming refine_report_if_requested is never called and the
    response's own refinement field stays None."""
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        _with_existing_report(client, session_id)

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_web_source("ex-1", "https://a.com")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})

        with patch.object(api, "regenerate_report_with_approved_web_sources", return_value=_report_stub_out("v2")):
            with patch.object(api, "refine_report_if_requested") as mock_refine:
                resp = client.post(
                    f"/curation/{session_id}/chat/exchanges/add-to-report",
                    json={"exchange_ids": ["ex-1"], "refinement_mode": "single"},
                )

    assert resp.status_code == 200
    mock_refine.assert_not_called()
    assert resp.json()["report"]["refinement"] is None


# --- report-quality Phase R3: report versioning ---

def test_curation_report_generate_creates_version_1_with_initial_reason():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        fake_report = {
            "findings": {"content": "f", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        with patch.object(api, "generate_report_for_session", return_value=fake_report):
            resp = client.post(f"/curation/{session_id}/report")
        state_resp = client.get(f"/curation/{session_id}")

    assert resp.status_code == 200
    assert resp.json()["version_number"] == 1
    assert resp.json()["generation_reason"] == "initial"
    assert resp.json()["version_id"]

    body = state_resp.json()
    assert len(body["report_versions"]) == 1
    assert body["report_versions"][0]["version_number"] == 1
    assert body["report_versions"][0]["generation_reason"] == "initial"
    assert body["report_versions"][0]["is_active"] is True
    assert body["active_report_version_id"] == body["report_versions"][0]["version_id"]


def test_curation_report_regenerate_appends_version_2_and_keeps_version_1_unchanged():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        first_report = {
            "findings": {"content": "v1", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        with patch.object(api, "generate_report_for_session", return_value=first_report):
            client.post(f"/curation/{session_id}/report")

        second_report = {**first_report, "findings": {"content": "v2", "cited_papers": []}}
        with patch.object(api, "regenerate_report_with_new_sources", return_value=second_report):
            resp = client.post(f"/curation/{session_id}/report/regenerate")
        state_resp = client.get(f"/curation/{session_id}")

    assert resp.status_code == 200
    assert resp.json()["version_number"] == 2
    assert resp.json()["generation_reason"] == "regenerate"

    versions = state_resp.json()["report_versions"]
    assert [v["version_number"] for v in versions] == [1, 2]
    assert versions[0]["generation_reason"] == "initial"
    assert versions[1]["generation_reason"] == "regenerate"
    assert versions[0]["is_active"] is False
    assert versions[1]["is_active"] is True


def test_curation_report_activate_switches_state_report_to_the_selected_version():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        first_report = {
            "findings": {"content": "v1", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        with patch.object(api, "generate_report_for_session", return_value=first_report):
            client.post(f"/curation/{session_id}/report")
        state_after_v1 = client.get(f"/curation/{session_id}").json()
        v1_id = state_after_v1["report_versions"][0]["version_id"]

        second_report = {**first_report, "findings": {"content": "v2", "cited_papers": []}}
        with patch.object(api, "regenerate_report_with_new_sources", return_value=second_report):
            client.post(f"/curation/{session_id}/report/regenerate")

        # Activate v1 back -- state.report must switch to v1's own content.
        activate_resp = client.post(f"/curation/{session_id}/reports/{v1_id}/activate")
        state_resp = client.get(f"/curation/{session_id}")

    assert activate_resp.status_code == 200
    assert activate_resp.json()["findings"]["content"] == "v1"
    assert activate_resp.json()["version_id"] == v1_id
    assert state_resp.json()["report"]["findings"]["content"] == "v1"
    assert state_resp.json()["active_report_version_id"] == v1_id
    versions_by_id = {v["version_id"]: v for v in state_resp.json()["report_versions"]}
    assert versions_by_id[v1_id]["is_active"] is True


def test_curation_report_activate_unknown_version_id_returns_404():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        fake_report = {
            "findings": {"content": "f", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        with patch.object(api, "generate_report_for_session", return_value=fake_report):
            client.post(f"/curation/{session_id}/report")

        resp = client.post(f"/curation/{session_id}/reports/does-not-exist/activate")

    assert resp.status_code == 404


def test_curation_report_activate_unknown_session_id_returns_404():
    with _client() as client:
        resp = client.post("/curation/does-not-exist/reports/some-version/activate")
    assert resp.status_code == 404


def test_curation_report_regenerate_builds_from_the_active_not_latest_version():
    """report-quality Phase R3 decision 8, at the API level: activate an
    OLDER version, then regenerate -- the mocked regenerate call must
    see session.report already switched to that older version's own
    content by the time it's invoked (report.py's own regenerate
    functions read session.report directly)."""
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        first_report = {
            "findings": {"content": "v1", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        with patch.object(api, "generate_report_for_session", return_value=first_report):
            client.post(f"/curation/{session_id}/report")
        v1_id = client.get(f"/curation/{session_id}").json()["report_versions"][0]["version_id"]

        second_report = {**first_report, "findings": {"content": "v2", "cited_papers": []}}
        with patch.object(api, "regenerate_report_with_new_sources", return_value=second_report):
            client.post(f"/curation/{session_id}/report/regenerate")

        client.post(f"/curation/{session_id}/reports/{v1_id}/activate")

        seen_session_report_content = {}

        def _capture_active_report(session, client=None, model="gpt-4.1", report_template=None):
            seen_session_report_content["content"] = session.report["findings"]["content"]
            return {**first_report, "findings": {"content": "v3 from v1", "cited_papers": []}}

        with patch.object(api, "regenerate_report_with_new_sources", side_effect=_capture_active_report):
            resp = client.post(f"/curation/{session_id}/report/regenerate")

    assert seen_session_report_content["content"] == "v1"
    assert resp.json()["findings"]["content"] == "v3 from v1"
    assert resp.json()["version_number"] == 3


def test_curation_chat_add_to_report_appends_version_with_chat_add_to_report_reason():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        _with_existing_report(client, session_id)

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_web_source("ex-1", "https://a.com")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})

        with patch.object(api, "regenerate_report_with_approved_web_sources", return_value=_report_stub_out("v2")):
            client.post(f"/curation/{session_id}/chat/exchanges/add-to-report", json={"exchange_ids": ["ex-1"]})

        state_resp = client.get(f"/curation/{session_id}")

    versions = state_resp.json()["report_versions"]
    assert [v["generation_reason"] for v in versions] == ["initial", "chat_add_to_report"]
    assert versions[1]["is_active"] is True


# --- report-quality Phase R5A: GET /curation/{id}/report/export ---

def test_curation_report_export_returns_markdown_for_the_active_version():
    """Required tests 10: content type and body shape at the real HTTP
    boundary -- render_report_markdown/report_export_filename
    themselves are unit-tested directly in test_report.py; this proves
    the endpoint actually wires them up correctly."""
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        fake_report = {
            "findings": {"content": "Per [1].", "cited_papers": [_paper(pick_ids[0], "Paper Zero")]},
            "limitations": {"content": "L", "cited_papers": []},
            "future_scope": {"content": "S", "cited_papers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        with patch.object(api, "generate_report_for_session", return_value=fake_report):
            client.post(f"/curation/{session_id}/report")

        resp = client.get(f"/curation/{session_id}/report/export")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    assert 'attachment; filename="' in resp.headers["content-disposition"]
    assert "Per [1]." in resp.text
    assert "## Findings" in resp.text
    assert "**Version:** 1 (initial)" in resp.text
    assert "## References" in resp.text
    assert "Paper Zero" in resp.text


def test_curation_report_export_explicit_format_markdown_matches_the_default():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        fake_report = {
            "findings": {"content": "f", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        with patch.object(api, "generate_report_for_session", return_value=fake_report):
            client.post(f"/curation/{session_id}/report")

        default_resp = client.get(f"/curation/{session_id}/report/export")
        explicit_resp = client.get(f"/curation/{session_id}/report/export", params={"format": "markdown"})

    assert default_resp.text == explicit_resp.text


def test_curation_report_export_reflects_active_not_latest_version():
    """Required test 6: activate an older version, then export -- the
    exported content must be the ACTIVE (v1) version's, not the latest
    (v2) one, mirroring test_curation_report_regenerate_builds_from_
    the_active_not_latest_version's own proof for regeneration."""
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        first_report = {
            "findings": {"content": "v1 content", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        with patch.object(api, "generate_report_for_session", return_value=first_report):
            client.post(f"/curation/{session_id}/report")
        v1_id = client.get(f"/curation/{session_id}").json()["report_versions"][0]["version_id"]

        second_report = {**first_report, "findings": {"content": "v2 content", "cited_papers": []}}
        with patch.object(api, "regenerate_report_with_new_sources", return_value=second_report):
            client.post(f"/curation/{session_id}/report/regenerate")

        client.post(f"/curation/{session_id}/reports/{v1_id}/activate")

        resp = client.get(f"/curation/{session_id}/report/export")

    assert "v1 content" in resp.text
    assert "v2 content" not in resp.text
    assert "**Version:** 1 (initial)" in resp.text


def test_curation_report_export_unknown_session_id_returns_404():
    with _client() as client:
        resp = client.get("/curation/does-not-exist/report/export")

    assert resp.status_code == 404


def test_curation_report_export_no_report_yet_returns_404():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        resp = client.get(f"/curation/{session_id}/report/export")

    assert resp.status_code == 404


def test_curation_report_export_unsupported_format_returns_400():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        fake_report = {
            "findings": {"content": "f", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [], "report_template": "analytical",
        }
        with patch.object(api, "generate_report_for_session", return_value=fake_report):
            client.post(f"/curation/{session_id}/report")

        resp = client.get(f"/curation/{session_id}/report/export", params={"format": "pdf"})

    assert resp.status_code == 400


def test_curation_report_export_unsupported_format_checked_before_session_lookup():
    """An unsupported format 400s even for a session_id that doesn't
    exist at all -- format validation is a pure request-shape check,
    resolved before any session lookup (see export_active_report's own
    docstring for why)."""
    with _client() as client:
        resp = client.get("/curation/does-not-exist/report/export", params={"format": "docx"})

    assert resp.status_code == 400


# --- /curation/{id}/chat ---

def test_curation_chat_answers_and_persists_history_across_a_separate_get_request():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        fake_result = {
            "answer": "Per [Paper 1], X is true.",
            "answerable": True,
            "cited_papers": [_paper(pick_ids[0], "Paper 0")],
            "cited_web_articles": [],
        }

        def _fake_chat_turn(session, message, client=None, **kwargs):
            # Mutate the REAL session object the same way chat_turn()
            # actually does, so save_curation_session() has something
            # genuine to persist -- not just a mocked return value.
            session.chat_history.append({"role": "user", "content": message})
            session.chat_history.append({"role": "assistant", "content": fake_result["answer"]})
            return fake_result

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn):
            resp = client.post(f"/curation/{session_id}/chat", json={"message": "what does paper 0 say?"})

        state_resp = client.get(f"/curation/{session_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Per [Paper 1], X is true."
    assert body["cited_papers"] == [{"paper_id": pick_ids[0], "title": "Paper 0"}]
    assert len(body["chat_history"]) == 2

    # curation-chat-metadata Phase 1: ChatTurn now always serializes its
    # (additive, defaulted) metadata fields too -- this fake (which mocks
    # api.chat_turn entirely, bypassing curation_chat.py's real metadata
    # attachment) only ever appends old-shape {role, content} dicts, so
    # every new field comes back at its default, not a hardcoded/omitted
    # value. This is the exact "old chat_history entries still serialize
    # through ChatTurn/API" backward-compat case.
    #
    # report-quality Phase R3.2 Chunk 1: cited_papers is the newest such
    # additive field -- also defaults to [] for this exact old-shape
    # fixture, same as cited_web_articles already does.
    assert state_resp.json()["chat_history"] == [
        {
            "role": "user", "content": "what does paper 0 say?",
            "exchange_id": None, "used_web_search": False, "cited_web_articles": [], "cited_papers": [],
            "added_to_report": False,
        },
        {
            "role": "assistant", "content": "Per [Paper 1], X is true.",
            "exchange_id": None, "used_web_search": False, "cited_web_articles": [], "cited_papers": [],
            "added_to_report": False,
        },
    ]


def test_curation_state_includes_chat_references_and_marker_rewritten_chat_history():
    """report-quality Phase R3.2 Chunk 2: GET /curation/{id} returns
    chat-local numeric [N] markers in chat_history (not the model's raw
    [Paper N]/[Web N]) plus the chat_references list those markers point
    into -- derived fresh, independent of report.references."""
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        pid = pick_ids[0]

        def _fake_chat_turn(session, message, client=None, **kwargs):
            answer = "Per [Paper 1], X is true."
            session.chat_history.append({"role": "user", "content": message, "exchange_id": "ex-1"})
            session.chat_history.append({
                "role": "assistant", "content": answer, "exchange_id": "ex-1",
                "used_web_search": False, "cited_web_articles": [],
                "cited_papers": [{"paper_id": pid, "title": "Paper 0"}],
                "added_to_report": False,
            })
            return {"answer": answer, "answerable": True, "cited_papers": [], "cited_web_articles": []}

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn):
            client.post(f"/curation/{session_id}/chat", json={"message": "what does paper 0 say?"})

        state_resp = client.get(f"/curation/{session_id}")

    body = state_resp.json()
    assistant_turn = next(t for t in body["chat_history"] if t["role"] == "assistant")
    assert assistant_turn["content"] == "Per [1], X is true."  # rewritten, not raw [Paper 1]
    assert len(body["chat_references"]) == 1
    assert body["chat_references"][0]["number"] == 1
    assert body["chat_references"][0]["kind"] == "paper"
    assert body["chat_references"][0]["paper_id"] == pid
    # Independent of report.references -- no report was ever generated
    # in this session, so report is still None; chat_references is
    # populated regardless.
    assert body["report"] is None


# --- /curation/{id}/chat/exchanges/delete (curation-chat-delete Phase 3) ---

def _fake_chat_turn_with_exchange(exchange_id: str, used_web_search: bool = False, added_to_report: bool = False):
    """Same 'mock at the api.chat_turn level, but mutate the real session
    the same way the real function does' convention as this file's other
    chat tests -- here also stamping the exchange_id/metadata Phase 1's
    real chat_turn() would have attached, since that's exactly the shape
    the delete endpoint under test needs to operate on."""

    def _fake(session, message, client=None, **kwargs):
        answer = f"answer to {message!r}"
        session.chat_history.append({"role": "user", "content": message, "exchange_id": exchange_id})
        session.chat_history.append({
            "role": "assistant", "content": answer, "exchange_id": exchange_id,
            "used_web_search": used_web_search,
            "cited_web_articles": [{"url": "https://x.com", "title": "X"}] if used_web_search else [],
            "added_to_report": added_to_report,
        })
        return {"answer": answer, "answerable": True, "cited_papers": [], "cited_web_articles": []}

    return _fake


def test_curation_chat_delete_removes_both_entries_of_one_exchange_and_persists():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_exchange("ex-1")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})
        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_exchange("ex-2")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q2"})

        resp = client.post(f"/curation/{session_id}/chat/exchanges/delete", json={"exchange_ids": ["ex-1"]})
        state_resp = client.get(f"/curation/{session_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted_exchange_ids"] == ["ex-1"]
    assert body["report_possibly_stale"] is False
    assert len(body["chat_history"]) == 2
    assert all(t["exchange_id"] == "ex-2" for t in body["chat_history"])
    # Persisted, visible via a genuinely separate request too.
    assert len(state_resp.json()["chat_history"]) == 2
    assert all(t["exchange_id"] == "ex-2" for t in state_resp.json()["chat_history"])


def test_curation_chat_delete_multiple_exchanges_at_once():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        for eid in ("ex-1", "ex-2", "ex-3"):
            with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_exchange(eid)):
                client.post(f"/curation/{session_id}/chat", json={"message": eid})

        resp = client.post(f"/curation/{session_id}/chat/exchanges/delete", json={"exchange_ids": ["ex-1", "ex-3"]})

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted_exchange_ids"] == ["ex-1", "ex-3"]
    assert len(body["chat_history"]) == 2
    assert all(t["exchange_id"] == "ex-2" for t in body["chat_history"])


def test_curation_chat_delete_reports_possibly_stale_when_deleted_answer_was_added_to_report():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_exchange("ex-1", added_to_report=True)):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})

        resp = client.post(f"/curation/{session_id}/chat/exchanges/delete", json={"exchange_ids": ["ex-1"]})

    assert resp.status_code == 200
    assert resp.json()["report_possibly_stale"] is True


def test_curation_chat_delete_unknown_exchange_id_is_idempotent_no_op():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_exchange("ex-1")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})

        resp = client.post(f"/curation/{session_id}/chat/exchanges/delete", json={"exchange_ids": ["does-not-exist"]})

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted_exchange_ids"] == []
    assert len(body["chat_history"]) == 2  # unchanged


def test_curation_chat_delete_empty_exchange_ids_returns_400():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        resp = client.post(f"/curation/{session_id}/chat/exchanges/delete", json={"exchange_ids": []})
    assert resp.status_code == 400


def test_curation_chat_delete_unknown_session_id_returns_404():
    with _client() as client:
        resp = client.post("/curation/does-not-exist/chat/exchanges/delete", json={"exchange_ids": ["ex-1"]})
    assert resp.status_code == 404


def test_curation_chat_delete_never_touches_pre_phase_1_entries_with_no_exchange_id():
    def _fake_old_shape_chat_turn(session, message, client=None, **kwargs):
        session.chat_history.append({"role": "user", "content": message})
        session.chat_history.append({"role": "assistant", "content": f"answer to {message!r}"})
        return {"answer": f"answer to {message!r}", "answerable": True, "cited_papers": [], "cited_web_articles": []}

    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        with patch.object(api, "chat_turn", side_effect=_fake_old_shape_chat_turn):
            client.post(f"/curation/{session_id}/chat", json={"message": "an old-shape question"})
        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_exchange("ex-1")):
            client.post(f"/curation/{session_id}/chat", json={"message": "a new question"})

        resp = client.post(f"/curation/{session_id}/chat/exchanges/delete", json={"exchange_ids": ["ex-1"]})

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted_exchange_ids"] == ["ex-1"]
    # The old-shape pair (exchange_id: None) survives, unchanged.
    assert len(body["chat_history"]) == 2
    assert body["chat_history"][0]["content"] == "an old-shape question"
    assert body["chat_history"][0]["exchange_id"] is None
    assert body["chat_history"][1]["content"] == "answer to 'an old-shape question'"
    assert body["chat_history"][1]["exchange_id"] is None


# --- /curation/{id}/chat/exchanges/add-to-report (curation-chat-add-to-report Phase 4) ---

def _fake_chat_turn_with_web_source(exchange_id: str, url: str, title: str = "Article", added_to_report: bool = False):
    def _fake(session, message, client=None, **kwargs):
        answer = f"answer to {message!r}"
        session.chat_history.append({"role": "user", "content": message, "exchange_id": exchange_id})
        session.chat_history.append({
            "role": "assistant", "content": answer, "exchange_id": exchange_id,
            "used_web_search": True, "cited_web_articles": [{"url": url, "title": title}],
            "added_to_report": added_to_report,
        })
        # Real chat_turn() can only ever cite a web article that's already
        # in session.web_articles_added (qa.ask() retrieves from exactly
        # that list) -- this fake mirrors that so the article is actually
        # resolvable by resolve_approved_web_articles_for_regeneration.
        if not any(a.url == url for a in session.web_articles_added):
            session.web_articles_added.append(
                WebArticle(title=title, url=url, snippet="s", published_date=None, source_domain="example.com"),
            )
        return {"answer": answer, "answerable": True, "cited_papers": [], "cited_web_articles": []}

    return _fake


def _report_stub_out(content: str = "content") -> dict:
    return {
        "findings": {"content": content, "cited_papers": [], "cited_web_articles": []},
        "limitations": {"content": "", "cited_papers": [], "cited_web_articles": []},
        "future_scope": {"content": "", "cited_papers": [], "cited_web_articles": []},
        "skipped_papers": [],
    }


def _with_existing_report(client, session_id):
    with patch.object(api, "generate_report_for_session", return_value=_report_stub_out("v1")):
        client.post(f"/curation/{session_id}/report")


def test_curation_chat_add_to_report_regenerates_with_only_the_selected_sources_and_persists():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        _with_existing_report(client, session_id)

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_web_source("ex-1", "https://a.com")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})
        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_web_source("ex-2", "https://b.com")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q2"})

        with patch.object(api, "regenerate_report_with_approved_web_sources", return_value=_report_stub_out("v2")) as mock_regen:
            resp = client.post(f"/curation/{session_id}/chat/exchanges/add-to-report", json={"exchange_ids": ["ex-1"]})

        state_resp = client.get(f"/curation/{session_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["added_exchange_ids"] == ["ex-1"]
    assert body["skipped_exchange_ids"] == []
    assert body["source_count"] == 1
    assert body["report"]["findings"]["content"] == "v2"

    # Only ex-1's article reached regenerate_report_with_approved_web_sources -- ex-2's never did.
    approved_articles = mock_regen.call_args.args[1]
    assert [a.url for a in approved_articles] == ["https://a.com"]

    # Persisted: ex-1's assistant entry is added_to_report, ex-2's is not.
    by_exchange = {t["exchange_id"]: t for t in state_resp.json()["chat_history"] if t["role"] == "assistant"}
    assert by_exchange["ex-1"]["added_to_report"] is True
    assert by_exchange["ex-2"]["added_to_report"] is False
    assert state_resp.json()["report"]["findings"]["content"] == "v2"


def test_curation_chat_add_to_report_regeneration_preserves_existing_report_template():
    """report-quality Phase R2C decision 8: the add-to-report HTTP
    endpoint has no report_template field on its own request schema, and
    curation_chat_service.py's call site never passes one -- this
    endpoint always preserves whatever template the existing report
    already has, proven both by the response shape and by checking the
    underlying regenerate call never received an explicit override."""
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        with patch.object(
            api, "generate_report_for_session",
            return_value={**_report_stub_out("v1"), "report_template": "foundational"},
        ):
            client.post(f"/curation/{session_id}/report")

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_web_source("ex-1", "https://a.com")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})

        with patch.object(
            api, "regenerate_report_with_approved_web_sources",
            return_value={**_report_stub_out("v2"), "report_template": "foundational"},
        ) as mock_regen:
            resp = client.post(f"/curation/{session_id}/chat/exchanges/add-to-report", json={"exchange_ids": ["ex-1"]})

    assert resp.status_code == 200
    assert resp.json()["report"]["report_template"] == "foundational"
    assert "report_template" not in mock_regen.call_args.kwargs


def test_curation_chat_add_to_report_second_call_includes_previously_approved_sources():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        _with_existing_report(client, session_id)

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_web_source("ex-1", "https://a.com")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})
        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_web_source("ex-2", "https://b.com")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q2"})

        with patch.object(api, "regenerate_report_with_approved_web_sources", return_value=_report_stub_out("v2")):
            client.post(f"/curation/{session_id}/chat/exchanges/add-to-report", json={"exchange_ids": ["ex-1"]})

        with patch.object(api, "regenerate_report_with_approved_web_sources", return_value=_report_stub_out("v3")) as mock_regen2:
            resp2 = client.post(f"/curation/{session_id}/chat/exchanges/add-to-report", json={"exchange_ids": ["ex-2"]})

    assert resp2.status_code == 200
    assert resp2.json()["source_count"] == 1  # only ex-2's url is NEWLY approved this call
    approved_articles = mock_regen2.call_args.args[1]
    assert {a.url for a in approved_articles} == {"https://a.com", "https://b.com"}  # both, not just the new one


def test_curation_chat_delete_then_add_to_report_excludes_the_revoked_source_end_to_end():
    """chat-ux-report-semantics Phase B, end-to-end: ex-1 (https://a.com)
    gets added to the report, then deleted -- a later add-to-report call
    for an unrelated exchange must resolve ONLY its own URL, not the
    revoked one, even though https://a.com's WebArticle is still sitting
    in the raw web_articles_added pool (delete never touches that)."""
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        _with_existing_report(client, session_id)

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_web_source("ex-1", "https://a.com")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})
        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_web_source("ex-2", "https://b.com")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q2"})

        with patch.object(api, "regenerate_report_with_approved_web_sources", return_value=_report_stub_out("v2")):
            client.post(f"/curation/{session_id}/chat/exchanges/add-to-report", json={"exchange_ids": ["ex-1"]})

        delete_resp = client.post(f"/curation/{session_id}/chat/exchanges/delete", json={"exchange_ids": ["ex-1"]})
        assert delete_resp.json()["report_possibly_stale"] is True

        with patch.object(api, "regenerate_report_with_approved_web_sources", return_value=_report_stub_out("v3")) as mock_regen:
            resp = client.post(f"/curation/{session_id}/chat/exchanges/add-to-report", json={"exchange_ids": ["ex-2"]})

    assert resp.status_code == 200
    approved_articles = mock_regen.call_args.args[1]
    assert [a.url for a in approved_articles] == ["https://b.com"]  # https://a.com stayed revoked


def test_curation_chat_add_to_report_duplicate_urls_dedupe_source_count():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        _with_existing_report(client, session_id)

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_web_source("ex-1", "https://same.com")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})
        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_web_source("ex-2", "https://same.com")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q2"})

        with patch.object(api, "regenerate_report_with_approved_web_sources", return_value=_report_stub_out("v2")) as mock_regen:
            resp = client.post(
                f"/curation/{session_id}/chat/exchanges/add-to-report", json={"exchange_ids": ["ex-1", "ex-2"]},
            )

    assert resp.status_code == 200
    assert resp.json()["source_count"] == 1
    approved_articles = mock_regen.call_args.args[1]
    assert len(approved_articles) == 1


def test_curation_chat_add_to_report_already_added_returns_400_and_does_not_regenerate():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        _with_existing_report(client, session_id)

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_web_source("ex-1", "https://a.com")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})

        with patch.object(api, "regenerate_report_with_approved_web_sources", return_value=_report_stub_out("v2")):
            client.post(f"/curation/{session_id}/chat/exchanges/add-to-report", json={"exchange_ids": ["ex-1"]})

        with patch.object(api, "regenerate_report_with_approved_web_sources") as mock_regen:
            resp = client.post(f"/curation/{session_id}/chat/exchanges/add-to-report", json={"exchange_ids": ["ex-1"]})

    assert resp.status_code == 400
    assert "no eligible" in resp.json()["detail"].lower()
    mock_regen.assert_not_called()


def test_curation_chat_add_to_report_regeneration_failure_leaves_added_to_report_false():
    import httpx
    from openai import APIConnectionError

    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        _with_existing_report(client, session_id)

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_web_source("ex-1", "https://a.com")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})

        with patch.object(
            api, "regenerate_report_with_approved_web_sources",
            side_effect=APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/x")),
        ):
            failed_resp = client.post(f"/curation/{session_id}/chat/exchanges/add-to-report", json={"exchange_ids": ["ex-1"]})

        assert failed_resp.status_code == 503

        # The exchange must still be ELIGIBLE after the failure -- if the
        # failed attempt had wrongly marked it added_to_report=True, this
        # retry would 400 with "no eligible" instead of succeeding.
        with patch.object(api, "regenerate_report_with_approved_web_sources", return_value=_report_stub_out("v2")):
            retry_resp = client.post(f"/curation/{session_id}/chat/exchanges/add-to-report", json={"exchange_ids": ["ex-1"]})

    assert retry_resp.status_code == 200
    assert retry_resp.json()["added_exchange_ids"] == ["ex-1"]


def test_curation_chat_add_to_report_no_report_yet_returns_400():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        # No POST /report here -- session.report stays None.

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_web_source("ex-1", "https://a.com")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})

        resp = client.post(f"/curation/{session_id}/chat/exchanges/add-to-report", json={"exchange_ids": ["ex-1"]})

    assert resp.status_code == 400
    assert "generate a report first" in resp.json()["detail"].lower()


def test_curation_chat_add_to_report_ineligible_entries_return_400():
    def _fake_old_shape_chat_turn(session, message, client=None, **kwargs):
        session.chat_history.append({"role": "user", "content": message})
        session.chat_history.append({"role": "assistant", "content": f"answer to {message!r}"})
        return {"answer": f"answer to {message!r}", "answerable": True, "cited_papers": [], "cited_web_articles": []}

    def _fake_paper_only_chat_turn(session, message, client=None, **kwargs):
        session.chat_history.append({"role": "user", "content": message, "exchange_id": "ex-paper-only"})
        session.chat_history.append({
            "role": "assistant", "content": "paper-only answer", "exchange_id": "ex-paper-only",
            "used_web_search": False, "cited_web_articles": [], "added_to_report": False,
        })
        return {"answer": "paper-only answer", "answerable": True, "cited_papers": [], "cited_web_articles": []}

    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        _with_existing_report(client, session_id)

        with patch.object(api, "chat_turn", side_effect=_fake_old_shape_chat_turn):
            client.post(f"/curation/{session_id}/chat", json={"message": "old-shape question"})
        with patch.object(api, "chat_turn", side_effect=_fake_paper_only_chat_turn):
            client.post(f"/curation/{session_id}/chat", json={"message": "paper-only question"})

        with patch.object(api, "regenerate_report_with_approved_web_sources") as mock_regen:
            resp = client.post(
                f"/curation/{session_id}/chat/exchanges/add-to-report",
                json={"exchange_ids": ["ex-paper-only", "does-not-exist"]},
            )

    assert resp.status_code == 400
    mock_regen.assert_not_called()


def test_curation_chat_add_to_report_empty_exchange_ids_returns_400():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        _with_existing_report(client, session_id)
        resp = client.post(f"/curation/{session_id}/chat/exchanges/add-to-report", json={"exchange_ids": []})
    assert resp.status_code == 400


def test_curation_chat_add_to_report_unknown_session_id_returns_404():
    with _client() as client:
        resp = client.post("/curation/does-not-exist/chat/exchanges/add-to-report", json={"exchange_ids": ["ex-1"]})
    assert resp.status_code == 404


# --- /curation/{id}/chat/exchanges/edit (curation-chat-edit Phase 5) ---

def _fake_edit_chat_exchange(new_exchange_id: str, report_possibly_stale: bool = False):
    """Mocked at the api.edit_chat_exchange level -- same convention this
    whole file uses for every LLM-calling function (chat_turn,
    regenerate_report_with_approved_web_sources, etc.): the real
    truncation logic is already thoroughly covered by
    tests/test_curation_chat.py's domain-level unit tests, so this fake
    only needs to mimic ITS OBSERVABLE EFFECT (truncate + append a fresh
    exchange) closely enough to prove the HTTP/service/persistence layer
    wiring, not re-prove the truncation algorithm itself."""

    def _fake(session, exchange_id, new_question, client=None, **kwargs):
        user_idx = next(
            (i for i, t in enumerate(session.chat_history) if t.get("role") == "user" and t.get("exchange_id") == exchange_id),
            None,
        )
        if user_idx is None:
            raise ValueError(f"No editable user question found for exchange_id {exchange_id!r}")
        session.chat_history = session.chat_history[:user_idx]
        answer = f"fresh answer to {new_question!r}"
        session.chat_history.append({"role": "user", "content": new_question, "exchange_id": new_exchange_id})
        session.chat_history.append({
            "role": "assistant", "content": answer, "exchange_id": new_exchange_id,
            "used_web_search": False, "cited_web_articles": [], "added_to_report": False,
        })
        result = {"answer": answer, "answerable": True, "cited_papers": [], "cited_web_articles": []}
        return result, report_possibly_stale

    return _fake


def test_curation_chat_edit_truncates_and_persists_across_a_separate_get_request():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_exchange("ex-1")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})
        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_exchange("ex-2")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q2"})

        with patch.object(api, "edit_chat_exchange", side_effect=_fake_edit_chat_exchange("ex-new")):
            resp = client.post(
                f"/curation/{session_id}/chat/exchanges/edit",
                json={"exchange_id": "ex-1", "question": "an edited question"},
            )

        state_resp = client.get(f"/curation/{session_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "fresh answer to 'an edited question'"
    assert body["report_possibly_stale"] is False
    # report_update_offer_made/declined/report_updated present (always
    # False on this path) for frontend-handling consistency.
    assert body["report_update_offer_made"] is False
    assert body["report_update_declined"] is False
    assert body["report_updated"] is False
    assert len(body["chat_history"]) == 2
    assert all(t["exchange_id"] == "ex-new" for t in body["chat_history"])
    # ex-1's old pair and ex-2 are gone -- persisted, not just in this response.
    assert len(state_resp.json()["chat_history"]) == 2
    assert all(t["exchange_id"] == "ex-new" for t in state_resp.json()["chat_history"])


def test_curation_chat_edit_surfaces_report_possibly_stale_true():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_exchange("ex-1")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})

        with patch.object(api, "edit_chat_exchange", side_effect=_fake_edit_chat_exchange("ex-new", report_possibly_stale=True)):
            resp = client.post(
                f"/curation/{session_id}/chat/exchanges/edit",
                json={"exchange_id": "ex-1", "question": "an edited question"},
            )

    assert resp.status_code == 200
    assert resp.json()["report_possibly_stale"] is True


def test_curation_chat_edit_unknown_exchange_id_returns_400():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_exchange("ex-1")):
            client.post(f"/curation/{session_id}/chat", json={"message": "q1"})

        with patch.object(api, "edit_chat_exchange", side_effect=_fake_edit_chat_exchange("ex-new")):
            resp = client.post(
                f"/curation/{session_id}/chat/exchanges/edit",
                json={"exchange_id": "does-not-exist", "question": "an edited question"},
            )

    assert resp.status_code == 400


def test_curation_chat_edit_empty_exchange_id_returns_400():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        resp = client.post(f"/curation/{session_id}/chat/exchanges/edit", json={"exchange_id": "", "question": "hi"})
    assert resp.status_code == 400


def test_curation_chat_edit_blank_question_returns_400():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        resp = client.post(f"/curation/{session_id}/chat/exchanges/edit", json={"exchange_id": "ex-1", "question": "   "})
    assert resp.status_code == 400


def test_curation_chat_edit_unknown_session_id_returns_404():
    with _client() as client:
        resp = client.post("/curation/does-not-exist/chat/exchanges/edit", json={"exchange_id": "ex-1", "question": "hi"})
    assert resp.status_code == 404


def test_curation_chat_before_curation_finished_returns_400():
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(12)]
    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        session_id = client.post("/curation/start", json={"topic": "t", "target_count": 5}).json()["session_id"]
        resp = client.post(f"/curation/{session_id}/chat", json={"message": "hi"})
    assert resp.status_code == 400


def test_curation_chat_unknown_session_id_returns_404():
    with _client() as client:
        resp = client.post("/curation/does-not-exist/chat", json={"message": "hi"})
    assert resp.status_code == 404


def test_curation_chat_surfaces_web_offer_flag_to_the_client():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        fake_result = {
            "answer": "Not covered. Want me to search the web?",
            "answerable": False,
            "cited_papers": [],
            "cited_web_articles": [],
            "web_offer_made": True,
        }
        with patch.object(api, "chat_turn", return_value=fake_result):
            resp = client.post(f"/curation/{session_id}/chat", json={"message": "something obscure"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answerable"] is False
    assert body["web_offer_made"] is True
    assert body["web_offer_declined"] is False


def test_curation_chat_surfaces_report_update_offer_flags_to_the_client():
    """curation-refinement-and-auto-offer Phase 6f-3: same HTTP-wiring
    proof as the web-offer flag test above, for the new report-update
    offer fields."""
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        fake_result = {
            "answer": "Per [Paper 1], X is true. I also found new source(s) -- want me to update the report?",
            "answerable": True,
            "cited_papers": [],
            "cited_web_articles": [],
            "report_update_offer_made": True,
        }
        with patch.object(api, "chat_turn", return_value=fake_result):
            resp = client.post(f"/curation/{session_id}/chat", json={"message": "anything"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["report_update_offer_made"] is True
    assert body["report_update_declined"] is False
    assert body["report_updated"] is False


def test_curation_chat_report_update_persists_across_a_separate_get_request():
    """Confirms pending_report_update round-trips through
    save_curation_session/GET /curation/{id} for real, not just that the
    POST response happens to include it -- the property the frontend's
    own refresh-mid-offer behavior depends on."""
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        def _fake_chat_turn(session, message, client=None, **kwargs):
            session.chat_history.append({"role": "user", "content": message})
            session.chat_history.append({"role": "assistant", "content": "answer with an offer"})
            session.pending_report_update = {"new_article_count": 1}
            return {
                "answer": "answer with an offer", "answerable": True,
                "cited_papers": [], "cited_web_articles": [], "report_update_offer_made": True,
            }

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn):
            client.post(f"/curation/{session_id}/chat", json={"message": "trigger"})

        state_resp = client.get(f"/curation/{session_id}")

    assert state_resp.json()["pending_report_update"] == {"new_article_count": 1}


def test_curation_report_endpoint_sets_report_covered_web_article_count():
    """curation-refinement-and-auto-offer Phase 6f-3: generating a
    report directly through the API (not via chat's accept-web-offer
    path) must still keep the auto-offer's staleness bookkeeping
    accurate -- confirmed by adding a web article AFTER report
    generation and checking the offer condition via a real chat_turn
    call afterward would see it as stale (verified here at the level
    this endpoint actually controls: report_covered_web_article_count
    reflects web_articles_added at generation time)."""
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        fake_report = {
            "findings": {"content": "f", "cited_papers": []},
            "limitations": {"content": "", "cited_papers": []},
            "future_scope": {"content": "", "cited_papers": []},
            "skipped_papers": [],
        }
        with patch.object(api, "generate_report_for_session", return_value=fake_report):
            client.post(f"/curation/{session_id}/report")

        # No direct HTTP field exposes report_covered_web_article_count,
        # but a fake chat_turn can inspect the real session object to
        # confirm the endpoint actually set it correctly.
        seen = {}

        def _inspecting_chat_turn(session, message, client=None, **kwargs):
            seen["count"] = session.report_covered_web_article_count
            session.chat_history.append({"role": "user", "content": message})
            session.chat_history.append({"role": "assistant", "content": "ok"})
            return {"answer": "ok", "answerable": True, "cited_papers": [], "cited_web_articles": []}

        with patch.object(api, "chat_turn", side_effect=_inspecting_chat_turn):
            client.post(f"/curation/{session_id}/chat", json={"message": "hi"})

    assert seen["count"] == 0  # no web articles were ever added


if __name__ == "__main__":
    test_curation_start_returns_a_batch_and_a_fresh_session_id()
    test_curation_start_with_no_papers_returns_404()
    test_curation_start_upstream_openai_failure_returns_clean_error_not_raw_500()
    test_curation_picks_resumes_the_real_interrupt_and_returns_the_next_batch()
    test_curation_picks_reaching_target_does_not_transition_to_synthesize()
    test_curation_picks_with_explicit_stop_transitions_to_synthesize()
    test_curation_picks_on_unknown_session_id_returns_404()
    test_curation_picks_after_curation_already_finished_returns_400()
    test_curation_turn_response_surfaces_refilled_flag_for_real_across_a_genuine_refill()
    test_curation_picks_with_refinement_forces_a_real_refill_and_persists_across_a_separate_get_request()
    test_curation_picks_without_refinement_does_not_force_a_refill()
    test_curation_list_reviews_returns_empty_with_no_sessions()
    test_curation_list_reviews_reflects_real_sessions_via_genuinely_separate_requests()
    test_curation_list_reviews_flags_report_and_chat_progress()
    test_curation_get_state_reflects_a_pending_batch_from_a_separate_request()
    test_curation_get_state_after_finishing_has_no_pending_batch()
    test_curation_get_state_unknown_session_id_returns_404()
    test_curation_report_generates_and_persists_across_a_separate_get_request()
    test_curation_report_before_curation_finished_returns_400()
    test_curation_report_unknown_session_id_returns_404()
    test_curation_report_regenerate_overwrites_persisted_report()
    test_curation_chat_answers_and_persists_history_across_a_separate_get_request()
    test_curation_chat_before_curation_finished_returns_400()
    test_curation_chat_unknown_session_id_returns_404()
    test_curation_chat_surfaces_web_offer_flag_to_the_client()
    test_curation_chat_surfaces_report_update_offer_flags_to_the_client()
    test_curation_chat_report_update_persists_across_a_separate_get_request()
    test_curation_report_endpoint_sets_report_covered_web_article_count()
    test_curation_get_state_exposes_turn_history_and_stop_reason()
    test_curation_get_state_stop_reason_is_none_mid_curation()
    test_select_from_history_adds_a_paper_after_curation_finished_short_of_target()
    test_select_from_history_refuses_while_curation_still_in_progress()
    test_select_from_history_refuses_once_a_report_has_been_generated()
    test_select_from_history_refuses_once_chat_has_started()
    test_select_from_history_unknown_paper_id_returns_400()
    test_select_from_history_unknown_session_id_returns_404()
    test_select_from_history_can_exceed_target_count()
    test_curation_picks_with_request_refill_forces_a_search_through_the_real_http_layer()
    test_curation_reopen_resumes_active_curation_preserving_prior_state()
    test_curation_reopen_while_still_curating_returns_400()
    test_curation_reopen_after_report_generated_returns_400()
    test_curation_reopen_after_chat_started_returns_400()
    test_curation_reopen_unknown_session_id_returns_404()
    print("All curation API tests passed.")
