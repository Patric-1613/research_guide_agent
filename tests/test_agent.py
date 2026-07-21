"""Deterministic tests for the agent's tool-building logic — no live LLM
call. Verifies the tools correctly accumulate/dedup the session's working
pool, which the agent relies on regardless of which tool-call sequence the
model chooses on a given run.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch

from research_agent.agent import ResearchSession, build_tools
from research_agent.schema import Paper, WebArticle


def _paper(title: str, source: str, paper_id: str) -> Paper:
    return Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract="An abstract about " + title, url=f"http://example.com/{paper_id}",
        doi=None, citation_count=None, source=source, paper_id=paper_id,
    )


def test_search_tools_accumulate_and_dedup_into_session():
    session = ResearchSession()
    arxiv_tool, s2_tool, rerank_tool, _web_tool = build_tools(session)

    with patch("research_agent.agent.search_arxiv") as mock_arxiv:
        mock_arxiv.return_value = [_paper("Same Paper", "arxiv", "1111.1111")]
        arxiv_tool.invoke({"query": "test"})

    assert len(session.papers) == 1

    with patch("research_agent.agent.search_semantic_scholar") as mock_s2:
        # Same title from a different source -> should merge, not duplicate.
        mock_s2.return_value = [_paper("Same Paper", "semantic_scholar", "abc123")]
        s2_tool.invoke({"query": "test"})

    assert len(session.papers) == 1
    assert session.papers[0].source == "arxiv+semantic_scholar"


def test_rerank_tool_reports_empty_pool_without_crashing():
    session = ResearchSession()
    _, _, rerank_tool, _web_tool = build_tools(session)
    result = rerank_tool.invoke({"query": "anything", "top_k": 5})
    assert "No papers collected" in result


def _web_article(url: str, title: str) -> WebArticle:
    return WebArticle(title=title, url=url, snippet="a snippet", published_date=None, source_domain="example.com")


def test_search_web_tool_accumulates_into_session_web_articles():
    session = ResearchSession()
    _, _, _, web_tool = build_tools(session)

    with patch("research_agent.agent.search_web") as mock_search_web:
        mock_search_web.return_value = [_web_article("https://x.com/a", "Article A")]
        web_tool.invoke({"query": "test", "max_results": 4})

    assert len(session.web_articles) == 1
    assert session.papers == []  # web articles never touch the paper pool


def test_search_web_tool_dedups_by_url_across_calls():
    session = ResearchSession()
    _, _, _, web_tool = build_tools(session)

    with patch("research_agent.agent.search_web") as mock_search_web:
        mock_search_web.return_value = [_web_article("https://x.com/a", "Article A")]
        web_tool.invoke({"query": "test", "max_results": 4})
        # Same URL returned again on a second call (different query) — must not duplicate.
        mock_search_web.return_value = [_web_article("https://x.com/a", "Article A (reworded title)")]
        web_tool.invoke({"query": "test again", "max_results": 4})

    assert len(session.web_articles) == 1


def test_search_arxiv_tool_survives_underlying_exception_and_pool_stays_usable():
    # A tool failure (e.g. an unexpected error search_arxiv's own defensive
    # handling didn't catch) must not propagate and kill the whole agent
    # run — it should come back as a normal tool observation, and the
    # session must remain usable for a retry or a different tool.
    session = ResearchSession()
    arxiv_tool, s2_tool, _rerank_tool, _web_tool = build_tools(session)

    with patch("research_agent.agent.search_arxiv", side_effect=RuntimeError("arXiv is down")):
        result = arxiv_tool.invoke({"query": "test"})

    assert "arXiv is down" in result
    assert "failed" in result.lower()
    assert session.papers == []  # untouched by the failed call, not corrupted

    # The run continues with partial results: a subsequent successful call
    # on a different tool still works normally, no retry of the failed step
    # required first.
    with patch("research_agent.agent.search_semantic_scholar") as mock_s2:
        mock_s2.return_value = [_paper("Recovered Paper", "semantic_scholar", "xyz")]
        s2_tool.invoke({"query": "test"})

    assert len(session.papers) == 1
    assert session.papers[0].title == "Recovered Paper"


def test_rerank_tool_survives_underlying_exception_and_keeps_collected_papers():
    session = ResearchSession()
    _arxiv_tool, _s2_tool, rerank_tool, _web_tool = build_tools(session)
    session.papers = [_paper("Some Paper", "arxiv", "1111.1111")]

    # Mock the OpenAI() client construction itself too — otherwise this test's
    # outcome would depend on whether a real API key happens to be present
    # in the environment (it constructs a real client before the patched
    # embed_and_index_papers call ever runs), which isn't the failure this
    # test is about.
    with patch("research_agent.agent.OpenAI"), \
         patch("research_agent.agent.embed_and_index_papers", side_effect=RuntimeError("OpenAI is down")):
        result = rerank_tool.invoke({"query": "test", "top_k": 5})

    assert "OpenAI is down" in result
    assert "failed" in result.lower()
    # The already-collected papers are not lost just because reranking failed.
    assert len(session.papers) == 1
    assert session.ranked == []


def test_search_web_tool_survives_underlying_exception():
    session = ResearchSession()
    _, _, _, web_tool = build_tools(session)

    with patch("research_agent.agent.search_web", side_effect=RuntimeError("Tavily is down")):
        result = web_tool.invoke({"query": "test", "max_results": 4})

    assert "Tavily is down" in result
    assert session.web_articles == []


def test_search_web_tool_degrades_gracefully_when_no_results():
    session = ResearchSession()
    _, _, _, web_tool = build_tools(session)

    with patch("research_agent.agent.search_web", return_value=[]):
        result = web_tool.invoke({"query": "anything", "max_results": 4})

    assert session.web_articles == []
    assert "0 article(s)" in result


if __name__ == "__main__":
    test_search_tools_accumulate_and_dedup_into_session()
    test_rerank_tool_reports_empty_pool_without_crashing()
    test_search_arxiv_tool_survives_underlying_exception_and_pool_stays_usable()
    test_rerank_tool_survives_underlying_exception_and_keeps_collected_papers()
    test_search_web_tool_survives_underlying_exception()
    test_search_web_tool_accumulates_into_session_web_articles()
    test_search_web_tool_dedups_by_url_across_calls()
    test_search_web_tool_degrades_gracefully_when_no_results()
    print("All agent tests passed.")
