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


@contextmanager
def _client():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.sqlite"
        # Default search_web to a no-op here so every test that doesn't care
        # about web search (i.e. doesn't patch it itself) stays isolated from
        # the network/Tavily billing, per this file's module docstring —
        # otherwise api.py's server-side web-search fallback (added alongside
        # the top_k-style guarantee) would fire a real call for any test
        # whose fake session has fewer web_articles than web_max_results.
        with patch.object(api, "init_db", lambda: real_init_db(db_path)), \
             patch.object(api, "search_web", return_value=[]):
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


def test_search_returns_web_articles_as_separate_section_from_papers():
    papers = [_paper("p1", "Paper One")]
    web_articles = [_web_article("https://x.com/a", "Article A")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)], web_articles=web_articles)

    with _client() as client, patch.object(api, "run_research_agent", return_value=fake_session):
        resp = client.post("/search", json={"topic": "t"})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["papers"]) == 1
    assert len(body["web_articles"]) == 1
    assert body["web_articles"][0]["url"] == "https://x.com/a"
    # never interleaved with or counted in the papers list
    assert all("url" not in p or "abstract" in p for p in body["papers"])


def test_search_truncates_web_articles_to_requested_web_max_results():
    papers = [_paper("p1", "Paper One")]
    web_articles = [_web_article(f"https://x.com/{i}", f"Article {i}") for i in range(6)]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)], web_articles=web_articles)

    with _client() as client, patch.object(api, "run_research_agent", return_value=fake_session):
        resp = client.post("/search", json={"topic": "t", "web_max_results": 2})

    assert resp.status_code == 200
    assert len(resp.json()["web_articles"]) == 2


def test_search_degrades_gracefully_with_no_web_articles():
    papers = [_paper("p1", "Paper One")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)], web_articles=[])

    with _client() as client, patch.object(api, "run_research_agent", return_value=fake_session):
        resp = client.post("/search", json={"topic": "t"})

    assert resp.status_code == 200
    assert resp.json()["web_articles"] == []


def test_search_falls_back_to_direct_web_search_when_agent_found_none():
    # Whether the agent calls its own search_web_tool is a per-topic
    # judgment call (agent.py's system prompt) that it can skip entirely —
    # but web_max_results is a user-set request parameter like top_k, so the
    # user should still get web context whenever any exists, not only when
    # the model happened to decide to look.
    papers = [_paper("p1", "Paper One")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)], web_articles=[])
    fallback_articles = [_web_article("https://x.com/fallback", "Fallback Article")]

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "search_web", return_value=fallback_articles) as mock_search_web:
        resp = client.post("/search", json={"topic": "t", "web_max_results": 3})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["web_articles"]) == 1
    assert body["web_articles"][0]["url"] == "https://x.com/fallback"
    mock_search_web.assert_called_once_with("t", max_results=3)


def test_search_skips_web_fallback_when_agent_already_met_web_max_results():
    papers = [_paper("p1", "Paper One")]
    web_articles = [_web_article(f"https://x.com/{i}", f"Article {i}") for i in range(3)]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)], web_articles=web_articles)

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "search_web") as mock_search_web:
        resp = client.post("/search", json={"topic": "t", "web_max_results": 3})

    assert resp.status_code == 200
    assert len(resp.json()["web_articles"]) == 3
    mock_search_web.assert_not_called()


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


def test_search_reranks_serverside_when_agent_ignored_requested_top_k():
    # 3 papers gathered, but the agent's rerank result only has 2 — as if it
    # ignored the top_k=3 the user asked for. api.py must not trust that
    # count silently; it should re-rank server-side to honor top_k.
    papers = [_paper("p1", "Paper One"), _paper("p2", "Paper Two"), _paper("p3", "Paper Three")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9), (papers[1], 0.8)])
    corrected = [(papers[0], 0.9), (papers[1], 0.8), (papers[2], 0.7)]

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "embed_and_index_papers") as mock_embed, \
         patch.object(api, "semantic_search", return_value=corrected) as mock_search:
        resp = client.post("/search", json={"topic": "top_k test", "top_k": 3})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["papers"]) == 3
    mock_embed.assert_called_once()
    mock_search.assert_called_once()
    assert mock_search.call_args.kwargs["top_k"] == 3


def test_search_keeps_agent_ranking_when_count_already_matches_top_k():
    # Agent already returned exactly top_k results — no need to re-rank
    # server-side (would just re-bill an embedding call for nothing).
    papers = [_paper("p1", "Paper One"), _paper("p2", "Paper Two"), _paper("p3", "Paper Three")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9), (papers[1], 0.7), (papers[2], 0.6)])

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "embed_and_index_papers") as mock_embed, \
         patch.object(api, "semantic_search") as mock_search:
        resp = client.post("/search", json={"topic": "t", "top_k": 3})

    assert resp.status_code == 200
    assert len(resp.json()["papers"]) == 3
    mock_embed.assert_not_called()
    mock_search.assert_not_called()


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


def test_summarize_different_styles_produce_different_citations_without_recalling_llm():
    papers = [_paper("p1", "Paper One")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)])
    fake_summary_result = {
        "themes": [{"theme_name": "Only Theme", "papers": [
            {
                "paper": papers[0], "summary": "grounded summary",
                "apa_citation": "APA_CITE", "harvard_citation": "HARVARD_CITE", "bibtex": "@misc{x,}",
            }
        ]}],
        "gaps_and_disagreements": "none",
        "skipped_papers": [],
    }

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "get_papers_by_ids", return_value=papers), \
         patch.object(api, "generate_summary", return_value=fake_summary_result) as mock_gen:
        search_id = client.post("/search", json={"topic": "t"}).json()["search_id"]

        apa_resp = client.post("/summarize", json={"search_id": search_id, "style": "apa"})
        harvard_resp = client.post("/summarize", json={"search_id": search_id, "style": "harvard"})

    assert apa_resp.status_code == harvard_resp.status_code == 200
    apa_citation = apa_resp.json()["themes"][0]["papers"][0]["citation"]
    harvard_citation = harvard_resp.json()["themes"][0]["papers"][0]["citation"]
    assert apa_citation == "APA_CITE"
    assert harvard_citation == "HARVARD_CITE"
    assert harvard_resp.json()["style"] == "harvard"
    # Second call used a different style but must still reuse the cached
    # summary rather than re-billing the LLM — citation re-selection is free.
    mock_gen.assert_called_once()


def test_export_uses_selected_citation_style_in_references_section():
    papers = [_paper("p1", "Paper One")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)])
    fake_summary_result = {
        "themes": [{"theme_name": "Only Theme", "papers": [
            {
                "paper": papers[0], "summary": "grounded summary",
                "apa_citation": "APA_CITE", "harvard_citation": "HARVARD_CITE", "bibtex": "@misc{x,}",
            }
        ]}],
        "gaps_and_disagreements": "none",
        "skipped_papers": [],
    }

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "get_papers_by_ids", return_value=papers), \
         patch.object(api, "generate_summary", return_value=fake_summary_result):
        search_id = client.post("/search", json={"topic": "t"}).json()["search_id"]

        apa_export = client.get(f"/export/{search_id}")
        harvard_export = client.get(f"/export/{search_id}?style=harvard")

    assert "## References (APA)" in apa_export.text and "APA_CITE" in apa_export.text
    assert "## References (Harvard)" in harvard_export.text and "HARVARD_CITE" in harvard_export.text
    assert "## BibTeX" in apa_export.text and "## BibTeX" in harvard_export.text


def test_summarize_missing_search_id_returns_404():
    with _client() as client:
        resp = client.post("/summarize", json={"search_id": 999})
    assert resp.status_code == 404


def test_summarize_includes_web_summary_block_when_web_articles_present():
    papers = [_paper("p1", "Paper One")]
    web_articles = [_web_article("https://x.com/a", "Article A")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)], web_articles=web_articles)
    fake_summary_result = {
        "themes": [{"theme_name": "Only Theme", "papers": [
            {"paper": papers[0], "summary": "grounded summary", "apa_citation": "cite", "bibtex": "@misc{x,}"}
        ]}],
        "gaps_and_disagreements": "none",
        "skipped_papers": [],
    }
    fake_web_summary_result = {"synthesis": "Web articles say X.", "cited_articles": [web_articles[0]]}

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "get_papers_by_ids", return_value=papers), \
         patch.object(api, "generate_summary", return_value=fake_summary_result), \
         patch.object(api, "generate_web_summary", return_value=fake_web_summary_result) as mock_web_gen:
        search_id = client.post("/search", json={"topic": "t"}).json()["search_id"]
        resp = client.post("/summarize", json={"search_id": search_id})

    assert resp.status_code == 200
    web_summary = resp.json()["web_summary"]
    assert web_summary is not None
    assert web_summary["synthesis"] == "Web articles say X."
    assert web_summary["cited_articles"][0]["url"] == "https://x.com/a"
    mock_web_gen.assert_called_once()


def test_summarize_omits_web_summary_when_no_web_articles():
    papers = [_paper("p1", "Paper One")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)], web_articles=[])
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
         patch.object(api, "generate_summary", return_value=fake_summary_result), \
         patch.object(api, "generate_web_summary") as mock_web_gen:
        search_id = client.post("/search", json={"topic": "t"}).json()["search_id"]
        resp = client.post("/summarize", json={"search_id": search_id})

    assert resp.status_code == 200
    assert resp.json()["web_summary"] is None
    mock_web_gen.assert_not_called()  # never billed when there's nothing to summarize


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
    assert body["cited_web_articles"] == []  # old-style mock result predates this field — must default cleanly


def test_chat_returns_cited_web_articles_distinguishable_from_cited_papers():
    papers = [_paper("p1", "Paper One")]
    web_articles = [_web_article("https://x.com/a", "Article A")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)], web_articles=web_articles)
    fake_ask_result = {
        "answer": "Per [Paper 1] and [Web 1], X is true.",
        "answerable": True,
        "cited_papers": [papers[0]],
        "retrieved_papers": papers,
        "cited_web_articles": [web_articles[0]],
        "retrieved_web_articles": web_articles,
    }

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "get_papers_by_ids", return_value=papers), \
         patch.object(api, "ask", return_value=fake_ask_result):
        search_id = client.post("/search", json={"topic": "t"}).json()["search_id"]
        resp = client.post("/chat", json={"search_id": search_id, "question": "What is this?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["cited_papers"] == [{"paper_id": "p1", "title": "Paper One"}]
    assert body["cited_web_articles"] == [{"url": "https://x.com/a", "title": "Article A"}]


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


def test_export_includes_separate_web_context_section():
    papers = [_paper("p1", "Paper One")]
    web_articles = [_web_article("https://x.com/a", "Article A")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)], web_articles=web_articles)
    fake_summary_result = {
        "themes": [{"theme_name": "Only Theme", "papers": [
            {"paper": papers[0], "summary": "grounded summary", "apa_citation": "Author (2024). Paper One.", "bibtex": "@misc{x,}"}
        ]}],
        "gaps_and_disagreements": "none",
        "skipped_papers": [],
    }
    fake_web_summary_result = {"synthesis": "Web says X.", "cited_articles": [web_articles[0]]}

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "get_papers_by_ids", return_value=papers), \
         patch.object(api, "generate_summary", return_value=fake_summary_result), \
         patch.object(api, "generate_web_summary", return_value=fake_web_summary_result):
        search_id = client.post("/search", json={"topic": "t"}).json()["search_id"]
        resp = client.get(f"/export/{search_id}")

    assert resp.status_code == 200
    assert "## Web Context" in resp.text
    assert "Web says X." in resp.text
    assert "Article A" in resp.text
    # web context section is distinct from the paper references, not merged into it
    assert resp.text.index("## Web Context") > resp.text.index("## Gaps and Disagreements")


def test_round_search_first_round_marks_papers_new_and_does_not_touch_llm_agent():
    with _client() as client, \
         patch.object(api, "search_arxiv", return_value=[_paper("a1", "Paper A")]) as mock_arxiv, \
         patch.object(api, "search_semantic_scholar", return_value=[]) as mock_s2, \
         patch.object(api, "run_research_agent") as mock_agent:
        resp = client.post("/round_search", json={"topic": "PEFT", "keyword": "PEFT", "include_web": False})

    assert resp.status_code == 200
    body = resp.json()
    assert body["round"]["round_number"] == 1
    assert body["round"]["new_paper_ids"] == ["a1"]
    assert body["session_state"]["all_papers"]["a1"]["title"] == "Paper A"
    mock_arxiv.assert_called_once()
    mock_s2.assert_called_once()
    mock_agent.assert_not_called()  # code-driven round search, no LLM tool-loop involved


def test_round_search_second_round_resurfaces_paper_as_seen_not_new():
    with _client() as client, \
         patch.object(api, "search_semantic_scholar", return_value=[]):
        with patch.object(api, "search_arxiv", return_value=[_paper("arxiv:1", "LoRA: Low-Rank Adaptation")]):
            first = client.post(
                "/round_search",
                json={"topic": "PEFT", "keyword": "PEFT", "include_web": False},
            ).json()

        with patch.object(api, "search_arxiv", return_value=[_paper("s2:1", "LoRA: Low-Rank Adaptation")]):
            second = client.post(
                "/round_search",
                json={
                    "topic": "PEFT", "keyword": "low-rank adaptation",
                    "session_state": first["session_state"], "include_web": False,
                },
            ).json()

    assert second["round"]["round_number"] == 2
    merged_id = next(iter(second["session_state"]["all_papers"]))
    assert merged_id in second["round"]["paper_ids_found"]
    assert merged_id not in second["round"]["new_paper_ids"]
    # exactly one merged record, not two
    assert len(second["session_state"]["all_papers"]) == 1


def test_round_search_basket_status_survives_a_rename_across_rounds():
    with _client() as client, patch.object(api, "search_semantic_scholar", return_value=[]):
        with patch.object(api, "search_arxiv", return_value=[_paper("arxiv:1", "LoRA: Low-Rank Adaptation")]):
            first = client.post(
                "/round_search",
                json={"topic": "PEFT", "keyword": "PEFT", "include_web": False},
            ).json()

        session_state = first["session_state"]
        session_state["basket_paper_ids"] = ["arxiv:1"]

        with patch.object(api, "search_arxiv", return_value=[_paper("s2:1", "LoRA: Low-Rank Adaptation")]):
            second = client.post(
                "/round_search",
                json={
                    "topic": "PEFT", "keyword": "low-rank adaptation",
                    "session_state": session_state, "include_web": False,
                },
            ).json()

    merged_id = next(iter(second["session_state"]["all_papers"]))
    assert second["session_state"]["basket_paper_ids"] == [merged_id]


def test_round_search_skips_web_search_when_include_web_false():
    with _client() as client, \
         patch.object(api, "search_arxiv", return_value=[]), \
         patch.object(api, "search_semantic_scholar", return_value=[]), \
         patch.object(api, "search_web") as mock_web:
        resp = client.post("/round_search", json={"topic": "t", "keyword": "kw", "include_web": False})

    assert resp.status_code == 200
    mock_web.assert_not_called()


def _fake_session_state(all_papers: list[Paper], basket_paper_ids: list[str], topic: str = "t",
                         all_web_articles: list[WebArticle] | None = None, basket_web_urls: list[str] | None = None) -> dict:
    return {
        "topic": topic,
        "rounds": [],
        "all_papers": {p.paper_id: p.to_dict() for p in all_papers},
        "all_web_articles": {a.url: a.to_dict() for a in (all_web_articles or [])},
        "basket_paper_ids": basket_paper_ids,
        "basket_web_urls": basket_web_urls or [],
    }


def test_round_search_never_embeds_during_browsing():
    # Phase-3 success criterion: embed_and_index_papers must never be
    # called by a round search, however many rounds run — only /triage/summarize
    # is allowed to call it, and only for the basket.
    with _client() as client, \
         patch.object(api, "search_arxiv", return_value=[_paper("a1", "Paper A")]), \
         patch.object(api, "search_semantic_scholar", return_value=[]), \
         patch.object(api, "embed_and_index_papers") as mock_embed, \
         patch.object(api, "enrich_missing_abstracts") as mock_enrich:
        client.post("/round_search", json={"topic": "t", "keyword": "kw1", "include_web": False})
        client.post("/round_search", json={"topic": "t", "keyword": "kw2", "include_web": False})

    mock_embed.assert_not_called()
    mock_enrich.assert_not_called()


def test_triage_summarize_embeds_only_basket_not_full_pool():
    all_papers = [_paper("a1", "Paper A"), _paper("a2", "Paper B"), _paper("a3", "Paper C")]
    session_state = _fake_session_state(all_papers, basket_paper_ids=["a2"])
    fake_summary_result = {
        "themes": [{"theme_name": "Theme", "papers": [
            {"paper": all_papers[1], "summary": "s", "apa_citation": "c", "bibtex": "@misc{x,}"}
        ]}],
        "gaps_and_disagreements": "none",
        "skipped_papers": [],
    }

    with _client() as client, \
         patch.object(api, "embed_and_index_papers", return_value={
             "cache_hits": 0, "cache_misses": 1, "tokens_billed": 12, "estimated_cost_usd": 0.0001,
         }) as mock_embed, \
         patch.object(api, "generate_summary", return_value=fake_summary_result):
        resp = client.post("/triage/summarize", json={"session_state": session_state})

    assert resp.status_code == 200
    mock_embed.assert_called_once()
    embedded_papers = mock_embed.call_args.args[0]
    assert [p.paper_id for p in embedded_papers] == ["a2"]  # basket only, not all 3
    body = resp.json()
    assert body["basket_paper_count"] == 1
    assert body["embed_stats"]["cache_misses"] == 1


def test_triage_summarize_returns_400_on_empty_basket():
    session_state = _fake_session_state([_paper("a1", "Paper A")], basket_paper_ids=[])
    with _client() as client:
        resp = client.post("/triage/summarize", json={"session_state": session_state})
    assert resp.status_code == 400


def test_triage_summarize_skips_embedding_when_basket_has_only_web_articles():
    web_articles = [_web_article("https://x.com/a", "Article A")]
    session_state = _fake_session_state(
        [], basket_paper_ids=[], all_web_articles=web_articles, basket_web_urls=["https://x.com/a"],
    )
    fake_web_summary_result = {"synthesis": "X.", "cited_articles": [web_articles[0]]}

    with _client() as client, \
         patch.object(api, "embed_and_index_papers") as mock_embed, \
         patch.object(api, "enrich_missing_abstracts") as mock_enrich, \
         patch.object(api, "generate_web_summary", return_value=fake_web_summary_result):
        resp = client.post("/triage/summarize", json={"session_state": session_state})

    assert resp.status_code == 200
    mock_embed.assert_not_called()
    mock_enrich.assert_not_called()
    body = resp.json()
    assert body["basket_paper_count"] == 0
    assert body["basket_web_article_count"] == 1
    assert body["web_summary"]["synthesis"] == "X."
    assert body["embed_stats"] == {"cache_hits": 0, "cache_misses": 0, "tokens_billed": 0, "estimated_cost_usd": 0.0}


def test_triage_save_bag_persists_basket_and_reload_shows_it_organized():
    all_papers = [_paper("a1", "Paper A"), _paper("a2", "Paper B")]
    session_state = _fake_session_state(
        all_papers, basket_paper_ids=["a1"], topic="PEFT",
    )
    session_state["rounds"] = [{
        "round_number": 1, "keywords_used": ["PEFT"], "timestamp": "t",
        "paper_ids_found": ["a1", "a2"], "new_paper_ids": ["a1", "a2"],
        "web_urls_found": [], "new_web_urls": [],
    }]
    summary = {
        "themes": [{"theme_name": "Theme", "papers": [
            {"paper_id": "a1", "title": "Paper A", "summary": "s", "apa_citation": "c",
             "harvard_citation": "c", "bibtex": "@misc{x,}", "citation": "c"},
        ]}],
        "gaps_and_disagreements": "none", "skipped_paper_ids": [],
    }

    with _client() as client:
        save_resp = client.post("/triage/save_bag", json={
            "name": "My PEFT Bag", "session_state": session_state, "summary": summary,
        })
        assert save_resp.status_code == 200
        bag_id = save_resp.json()["bag_id"]
        assert save_resp.json()["keywords"] == ["PEFT"]

        # "reloading the app" == a fresh GET /bags call, exactly what the sidebar does on every rerun.
        listing = client.get("/bags").json()
        item = next(b for b in listing if b["bag_id"] == bag_id)
        assert item["name"] == "My PEFT Bag"
        assert item["paper_count"] == 1
        assert item["year"] == save_resp.json()["year"]

        with patch.object(api, "get_papers_by_ids", return_value=[all_papers[0]]):
            detail = client.get(f"/bags/{bag_id}").json()
        assert detail["themes"][0]["papers"][0]["title"] == "Paper A"
        assert detail["rounds"][0]["keywords_used"] == ["PEFT"]


def test_triage_save_bag_returns_400_on_empty_basket():
    session_state = _fake_session_state([_paper("a1", "Paper A")], basket_paper_ids=[])
    with _client() as client:
        resp = client.post("/triage/save_bag", json={"name": "x", "session_state": session_state, "summary": {}})
    assert resp.status_code == 400


def test_triage_discard_calls_chroma_delete_for_basket_papers():
    with _client() as client:
        # _state["collection"] is set up by the app's lifespan; swap in a mock directly.
        mock_collection = MagicMock()
        api._state["collection"] = mock_collection
        resp = client.post("/triage/discard", json={"paper_ids": ["a1", "a2"]})

    assert resp.status_code == 200
    assert resp.json()["chroma_ids_removed"] == ["a1", "a2"]
    mock_collection.delete.assert_called_once_with(ids=["a1", "a2"])


def test_triage_discard_never_deletes_papers_still_referenced_by_a_saved_bag():
    all_papers = [_paper("shared", "Shared Paper")]
    session_state = _fake_session_state(all_papers, basket_paper_ids=["shared"])
    summary = {"themes": [], "gaps_and_disagreements": "none", "skipped_paper_ids": []}

    with _client() as client:
        client.post("/triage/save_bag", json={"name": "Existing Bag", "session_state": session_state, "summary": summary})

        mock_collection = MagicMock()
        api._state["collection"] = mock_collection
        resp = client.post("/triage/discard", json={"paper_ids": ["shared", "not_shared"]})

    assert resp.status_code == 200
    # "shared" is still needed by the saved bag — only "not_shared" gets removed from Chroma.
    assert resp.json()["chroma_ids_removed"] == ["not_shared"]
    mock_collection.delete.assert_called_once_with(ids=["not_shared"])


def test_delete_bag_removes_sqlite_row_and_chroma_vectors_leaving_zero_trace():
    all_papers = [_paper("a1", "Paper A")]
    session_state = _fake_session_state(all_papers, basket_paper_ids=["a1"])
    summary = {"themes": [], "gaps_and_disagreements": "none", "skipped_paper_ids": []}

    with _client() as client:
        bag_id = client.post(
            "/triage/save_bag", json={"name": "Bag", "session_state": session_state, "summary": summary}
        ).json()["bag_id"]

        mock_collection = MagicMock()
        api._state["collection"] = mock_collection
        del_resp = client.delete(f"/bags/{bag_id}")

        assert del_resp.status_code == 200
        assert del_resp.json()["chroma_ids_removed"] == ["a1"]
        mock_collection.delete.assert_called_once_with(ids=["a1"])

        # Zero trace in SQLite: gone from both detail and list.
        assert client.get(f"/bags/{bag_id}").status_code == 404
        assert bag_id not in [b["bag_id"] for b in client.get("/bags").json()]


def test_delete_bag_keeps_chroma_vector_shared_with_another_bag():
    shared_paper = [_paper("shared", "Shared Paper")]
    only_a = [_paper("only_a", "Only In A")]
    summary = {"themes": [], "gaps_and_disagreements": "none", "skipped_paper_ids": []}

    with _client() as client:
        state_a = _fake_session_state(shared_paper + only_a, basket_paper_ids=["shared", "only_a"])
        bag_a_id = client.post("/triage/save_bag", json={"name": "A", "session_state": state_a, "summary": summary}).json()["bag_id"]

        state_b = _fake_session_state(shared_paper, basket_paper_ids=["shared"])
        client.post("/triage/save_bag", json={"name": "B", "session_state": state_b, "summary": summary})

        mock_collection = MagicMock()
        api._state["collection"] = mock_collection
        del_resp = client.delete(f"/bags/{bag_a_id}")

    # "shared" must survive in Chroma since bag B still references it — only "only_a" is removed.
    assert del_resp.json()["chroma_ids_removed"] == ["only_a"]
    mock_collection.delete.assert_called_once_with(ids=["only_a"])


def test_delete_bag_missing_id_returns_404():
    with _client() as client:
        resp = client.delete("/bags/999")
    assert resp.status_code == 404


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


def test_library_reports_web_article_count():
    papers = [_paper("p1", "Paper One")]
    web_articles = [_web_article("https://x.com/a", "Article A"), _web_article("https://x.com/b", "Article B")]
    fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)], web_articles=web_articles)

    with _client() as client, \
         patch.object(api, "run_research_agent", return_value=fake_session), \
         patch.object(api, "get_papers_by_ids", return_value=papers):
        search_id = client.post("/search", json={"topic": "t"}).json()["search_id"]

        listing = client.get("/library").json()
        item = next(i for i in listing if i["search_id"] == search_id)
        assert item["web_article_count"] == 2

        detail = client.get(f"/library/{search_id}").json()
        assert len(detail["web_articles"]) == 2


if __name__ == "__main__":
    test_search_success_persists_and_returns_ranked_papers()
    test_search_returns_web_articles_as_separate_section_from_papers()
    test_search_truncates_web_articles_to_requested_web_max_results()
    test_search_degrades_gracefully_with_no_web_articles()
    test_search_falls_back_to_direct_web_search_when_agent_found_none()
    test_search_skips_web_fallback_when_agent_already_met_web_max_results()
    test_search_with_no_papers_returns_404()
    test_search_falls_back_to_server_side_rerank_if_agent_skipped_it()
    test_search_reranks_serverside_when_agent_ignored_requested_top_k()
    test_search_keeps_agent_ranking_when_count_already_matches_top_k()
    test_summarize_reuses_cached_summary_without_recalling_llm()
    test_summarize_different_styles_produce_different_citations_without_recalling_llm()
    test_export_uses_selected_citation_style_in_references_section()
    test_summarize_missing_search_id_returns_404()
    test_summarize_includes_web_summary_block_when_web_articles_present()
    test_summarize_omits_web_summary_when_no_web_articles()
    test_chat_roundtrip_returns_answer_and_history()
    test_chat_returns_cited_web_articles_distinguishable_from_cited_papers()
    test_export_returns_markdown_with_citations()
    test_export_includes_separate_web_context_section()
    test_library_list_and_detail()
    test_library_reports_web_article_count()
    test_round_search_first_round_marks_papers_new_and_does_not_touch_llm_agent()
    test_round_search_second_round_resurfaces_paper_as_seen_not_new()
    test_round_search_basket_status_survives_a_rename_across_rounds()
    test_round_search_skips_web_search_when_include_web_false()
    test_round_search_never_embeds_during_browsing()
    test_triage_summarize_embeds_only_basket_not_full_pool()
    test_triage_summarize_returns_400_on_empty_basket()
    test_triage_summarize_skips_embedding_when_basket_has_only_web_articles()
    test_triage_save_bag_persists_basket_and_reload_shows_it_organized()
    test_triage_save_bag_returns_400_on_empty_basket()
    test_triage_discard_calls_chroma_delete_for_basket_papers()
    test_triage_discard_never_deletes_papers_still_referenced_by_a_saved_bag()
    test_delete_bag_removes_sqlite_row_and_chroma_vectors_leaving_zero_trace()
    test_delete_bag_keeps_chroma_vector_shared_with_another_bag()
    test_delete_bag_missing_id_returns_404()
    print("All API tests passed.")
