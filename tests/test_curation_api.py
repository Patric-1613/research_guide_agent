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
             patch.object(api, "OpenAI", return_value=MagicMock()):
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


def test_curation_picks_reaching_target_transitions_to_synthesize():
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
    assert body["stage"] == "synthesize"
    assert body["stop_reason"] == "target_met"
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
        client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks})  # finishes curation

        resp = client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": []})

    assert resp.status_code == 400


def test_curation_turn_response_surfaces_refilled_flag_for_real_across_a_genuine_refill():
    """The turn-feed divider's "from existing pool" vs "triggered a new
    search" label depends on this being real per-turn backend truth, not
    a client-side guess -- confirmed through the actual HTTP layer, not
    just curation_loop.py's own unit test for the same property."""
    from research_agent import query_expansion as qe_module

    initial_papers = [_paper(f"p{i}", f"Paper {i}") for i in range(15)]
    fresh_papers = [_paper(f"new{i}", f"New Paper {i}") for i in range(8)]

    with _client() as client, \
         patch.object(api, "build_candidate_pool", return_value=initial_papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(initial_papers), {})):
        # target_count=30 (the request model's max) -- comfortably unreachable
        # with only 5 picks below, so the turn genuinely continues into a
        # refill rather than hitting target_met first.
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": 30}).json()
        assert start_body["refilled"] is False  # 15 papers >= BATCH_SIZE, no refill needed yet
        assert start_body["reserve_remaining"] == 5  # 15 total - 10 served this turn
        session_id = start_body["session_id"]
        picks = [p["paper_id"] for p in start_body["batch"][:5]]  # remaining after: 15-10=5 < BATCH_SIZE

        with patch.object(qe_module, "build_candidate_pool", return_value=fresh_papers), \
             patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                 [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
             )):
            resp = client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks})

    assert resp.status_code == 200
    body = resp.json()
    assert body["refilled"] is True
    # refill_pool merges the 5 unserved + 8 genuinely-new fresh papers ->
    # reserve of 13, cursor resets to 0, this turn serves 10 -> 3 remain.
    assert body["reserve_remaining"] == 3


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
        client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks})

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


# --- /curation/{id}/report + /report/regenerate ---

def _finish_curation(client, target_count: int = 2, n_papers: int = 12) -> tuple[str, list[str]]:
    papers = [_paper(f"p{i}", f"Paper {i}") for i in range(n_papers)]
    with patch.object(api, "build_candidate_pool", return_value=papers), \
         patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
        start_body = client.post("/curation/start", json={"topic": "t", "target_count": target_count}).json()
        session_id = start_body["session_id"]
        picks = start_body["batch"][:target_count]
        pick_ids = [p["paper_id"] for p in picks]
        client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": pick_ids})
    return session_id, pick_ids


def test_curation_report_generates_and_persists_across_a_separate_get_request():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

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

        # Persisted state visible via a genuinely separate request too.
        state_resp = client.get(f"/curation/{session_id}")
    assert state_resp.json()["report"]["findings"]["content"] == "f"


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

    assert state_resp.json()["chat_history"] == [
        {"role": "user", "content": "what does paper 0 say?"},
        {"role": "assistant", "content": "Per [Paper 1], X is true."},
    ]


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


if __name__ == "__main__":
    test_curation_start_returns_a_batch_and_a_fresh_session_id()
    test_curation_start_with_no_papers_returns_404()
    test_curation_picks_resumes_the_real_interrupt_and_returns_the_next_batch()
    test_curation_picks_reaching_target_transitions_to_synthesize()
    test_curation_picks_on_unknown_session_id_returns_404()
    test_curation_picks_after_curation_already_finished_returns_400()
    test_curation_turn_response_surfaces_refilled_flag_for_real_across_a_genuine_refill()
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
    print("All curation API tests passed.")
