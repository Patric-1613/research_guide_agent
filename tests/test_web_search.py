"""Deterministic tests for round-2 enhancement 5's web search wrapper.
langchain_tavily.TavilySearch is mocked throughout — a live Tavily call is
covered by scripts/test_web_search.py.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.web_search import _source_domain, search_web


def test_source_domain_strips_www():
    assert _source_domain("https://www.example.com/article") == "example.com"
    assert _source_domain("https://blog.example.com/post") == "blog.example.com"


def test_search_web_empty_query_returns_empty_without_calling_tavily():
    with patch("langchain_tavily.TavilySearch") as mock_cls:
        assert search_web("   ") == []
        mock_cls.assert_not_called()


def test_search_web_returns_empty_when_no_api_key_configured():
    with patch.dict(os.environ, {}, clear=True), \
         patch("langchain_tavily.TavilySearch") as mock_cls:
        assert search_web("some query") == []
        mock_cls.assert_not_called()


def test_search_web_parses_results_into_web_articles():
    fake_response = {
        "results": [
            {
                "url": "https://www.example.com/a",
                "title": "Article A",
                "content": "Snippet A",
                "published_date": "2026-01-01",
            },
            {
                "url": "https://other.org/b",
                "title": "Article B",
                "content": "Snippet B",
            },
        ]
    }
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = fake_response

    with patch.dict(os.environ, {"TAVILY_API_KEY": "fake-key"}), \
         patch("langchain_tavily.TavilySearch", return_value=mock_tool):
        articles = search_web("some query", max_results=2)

    assert len(articles) == 2
    assert articles[0].title == "Article A"
    assert articles[0].url == "https://www.example.com/a"
    assert articles[0].snippet == "Snippet A"
    assert articles[0].published_date == "2026-01-01"
    assert articles[0].source_domain == "example.com"
    assert articles[1].published_date is None  # not every result carries one


def test_search_web_skips_malformed_items_missing_url_or_title():
    fake_response = {"results": [{"content": "no url or title here"}, {"url": "https://x.com", "title": ""}]}
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = fake_response

    with patch.dict(os.environ, {"TAVILY_API_KEY": "fake-key"}), \
         patch("langchain_tavily.TavilySearch", return_value=mock_tool):
        assert search_web("q") == []


def test_search_web_handles_api_errors_gracefully():
    mock_tool = MagicMock()
    mock_tool.invoke.side_effect = RuntimeError("Tavily rate limited us")

    with patch.dict(os.environ, {"TAVILY_API_KEY": "fake-key"}), \
         patch("langchain_tavily.TavilySearch", return_value=mock_tool):
        assert search_web("q") == []


def test_search_web_handles_zero_results_gracefully():
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = {"results": []}

    with patch.dict(os.environ, {"TAVILY_API_KEY": "fake-key"}), \
         patch("langchain_tavily.TavilySearch", return_value=mock_tool):
        assert search_web("q") == []


if __name__ == "__main__":
    test_source_domain_strips_www()
    test_search_web_empty_query_returns_empty_without_calling_tavily()
    test_search_web_returns_empty_when_no_api_key_configured()
    test_search_web_parses_results_into_web_articles()
    test_search_web_skips_malformed_items_missing_url_or_title()
    test_search_web_handles_api_errors_gracefully()
    test_search_web_handles_zero_results_gracefully()
    print("All web_search tests passed.")
