"""Deterministic tests for the FastAPI endpoints. The expensive/LLM-backed
functions (run_research_agent, generate_summary, ask, embedding calls) are
mocked so these run without network access or billing — live end-to-end
behavior is covered separately by scripts/test_api.py. Each test gets an
isolated temp SQLite file so tests can't see each other's rows or pollute
the real dev database.
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
from research_agent.schema import Paper
from research_agent.storage import init_db as real_init_db


def _paper(paper_id: str, title: str, abstract: str = "an abstract") -> Paper:
    return Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract=abstract, url=f"http://arxiv.org/abs/{paper_id}",
        doi=None, citation_count=None, source="arxiv", paper_id=paper_id,
    )


@contextmanager
def _client():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.sqlite"
        with patch.object(api, "init_db", lambda: real_init_db(db_path)):
            with TestClient(api.app) as client:
                yield client


def test_search_success_persists_and_returns_ranked_papers():
    papers = [_paper("p1", "Paper One"), _paper("p2", "Paper Two")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9), (papers[1], 0.7)])

    with _client() as client, patch.object(api, "run_research_agent", return_value=fake_session):
        resp = client.post("/search", json={"topic": "test topic"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["topic"] == "test topic"
    assert [p["title"] for p in body["papers"]] == ["Paper One", "Paper Two"]
    assert body["papers"][0]["score"] == 0.9


def test_search_with_no_papers_returns_404():
    fake_session = MagicMock(papers=[], ranked=[])

    with _client() as client, patch.object(api, "run_research_agent", return_value=fake_session):
        resp = client.post("/search", json={"topic": "nothing found"})

    assert resp.status_code == 404


def test_search_falls_back_to_server_side_rerank_if_agent_skipped_it():
    papers = [_paper("p1", "Paper One")]
    # Agent gathered papers but (for whatever reason) never called its rerank tool.
    fake_session = MagicMock(papers=papers, ranked=[])

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "embed_and_index_papers"), \
         patch.object(api, "semantic_search", return_value=[(papers[0], 0.42)]):
        resp = client.post("/search", json={"topic": "fallback test"})

    assert resp.status_code == 200
    assert resp.json()["papers"][0]["score"] == 0.42


def test_summarize_reuses_cached_summary_without_recalling_llm():
    papers = [_paper("p1", "Paper One")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)])
    fake_summary_result = {
        "themes": [{"theme_name": "Only Theme", "papers": [
            {"paper": papers[0], "summary": "grounded summary", "apa_citation": "cite", "bibtex": "@misc{x,}"}
        ]}],
        "gaps_and_disagreements": "none",
        "skipped_papers": [],
    }

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "get_papers_by_ids", return_value=papers), \
         patch.object(api, "generate_summary", return_value=fake_summary_result) as mock_gen:
        search_id = client.post("/search", json={"topic": "t"}).json()["search_id"]

        first = client.post("/summarize", json={"search_id": search_id})
        second = client.post("/summarize", json={"search_id": search_id})

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    mock_gen.assert_called_once()  # second call must reuse the persisted summary, not re-bill


def test_summarize_missing_search_id_returns_404():
    with _client() as client:
        resp = client.post("/summarize", json={"search_id": 999})
    assert resp.status_code == 404


def test_chat_roundtrip_returns_answer_and_history():
    papers = [_paper("p1", "Paper One")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)])
    fake_ask_result = {
        "answer": "Here's the answer [1].",
        "answerable": True,
        "cited_papers": [papers[0]],
        "retrieved_papers": papers,
    }

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "get_papers_by_ids", return_value=papers), \
         patch.object(api, "ask", return_value=fake_ask_result):
        search_id = client.post("/search", json={"topic": "t"}).json()["search_id"]
        resp = client.post("/chat", json={"search_id": search_id, "question": "What is this?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Here's the answer [1]."
    assert body["cited_papers"] == [{"paper_id": "p1", "title": "Paper One"}]


def test_export_returns_markdown_with_citations():
    papers = [_paper("p1", "Paper One")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)])
    fake_summary_result = {
        "themes": [{"theme_name": "Only Theme", "papers": [
            {"paper": papers[0], "summary": "grounded summary", "apa_citation": "Author (2024). Paper One.", "bibtex": "@misc{x,}"}
        ]}],
        "gaps_and_disagreements": "none",
        "skipped_papers": [],
    }

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "get_papers_by_ids", return_value=papers), \
         patch.object(api, "generate_summary", return_value=fake_summary_result):
        search_id = client.post("/search", json={"topic": "t"}).json()["search_id"]
        resp = client.get(f"/export/{search_id}")

    assert resp.status_code == 200
    assert "# Literature Summary: t" in resp.text
    assert "Author (2024). Paper One." in resp.text
    assert "@misc{x,}" in resp.text


def test_library_list_and_detail():
    papers = [_paper("p1", "Paper One")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)])

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "get_papers_by_ids", return_value=papers):
        search_id = client.post("/search", json={"topic": "library test"}).json()["search_id"]

        listing = client.get("/library").json()
        assert any(item["search_id"] == search_id and item["topic"] == "library test" for item in listing)
        assert next(item for item in listing if item["search_id"] == search_id)["has_summary"] is False

        detail = client.get(f"/library/{search_id}").json()
        assert detail["topic"] == "library test"
        assert detail["papers"][0]["title"] == "Paper One"


if __name__ == "__main__":
    test_search_success_persists_and_returns_ranked_papers()
    test_search_with_no_papers_returns_404()
    test_search_falls_back_to_server_side_rerank_if_agent_skipped_it()
    test_summarize_reuses_cached_summary_without_recalling_llm()
    test_summarize_missing_search_id_returns_404()
    test_chat_roundtrip_returns_answer_and_history()
    test_export_returns_markdown_with_citations()
    test_library_list_and_detail()
    print("All API tests passed.")
