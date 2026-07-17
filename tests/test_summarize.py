"""Deterministic tests for summarize.py's non-LLM logic: schema construction,
citation attachment, and skipped-paper detection. The OpenAI call itself is
mocked (covered live instead by scripts/test_summarize.py) so these run
without network access or billing.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.schema import Paper, WebArticle
from research_agent.summarize import _build_response_schema, _build_web_response_schema, generate_summary, generate_web_summary


def _paper(paper_id: str, title: str) -> Paper:
    return Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract=f"Abstract for {title}.", url=f"http://arxiv.org/abs/{paper_id}",
        doi=None, citation_count=None, source="arxiv", paper_id=paper_id,
    )


def _web_article(url: str, title: str) -> WebArticle:
    return WebArticle(title=title, url=url, snippet=f"Snippet for {title}.", published_date=None, source_domain="example.com")


def test_response_schema_rejects_unknown_paper_id():
    schema = _build_response_schema(["a", "b"])
    theme_cls = schema.model_fields["themes"].annotation.__args__[0]
    summary_cls = theme_cls.model_fields["papers"].annotation.__args__[0]

    summary_cls(paper_id="a", summary="fine")  # known id: should not raise
    try:
        summary_cls(paper_id="not-a-real-id", summary="fabricated")
        assert False, "expected a validation error for an unknown paper_id"
    except Exception:
        pass


def test_generate_summary_attaches_citations_and_flags_skipped():
    papers = [_paper("1111", "Paper One"), _paper("2222", "Paper Two")]
    schema = _build_response_schema([p.paper_id for p in papers])
    theme_cls = schema.model_fields["themes"].annotation.__args__[0]
    summary_cls = theme_cls.model_fields["papers"].annotation.__args__[0]

    # Model only references Paper One — Paper Two should show up as skipped.
    parsed = schema(
        themes=[theme_cls(theme_name="Only Theme", papers=[summary_cls(paper_id="1111", summary="grounded summary")])],
        gaps_and_disagreements="No notable gaps or disagreements observed among the retrieved papers.",
    )
    mock_message = MagicMock(parsed=parsed, refusal=None)
    mock_usage = MagicMock(total_tokens=100, prompt_tokens=80, completion_tokens=20)
    mock_response = MagicMock(usage=mock_usage)
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = mock_response

    result = generate_summary("some topic", papers, client=mock_client)

    assert len(result["themes"]) == 1
    entry = result["themes"][0]["papers"][0]
    assert entry["paper"].paper_id == "1111"
    assert "apa_citation" in entry and "bibtex" in entry

    assert len(result["skipped_papers"]) == 1
    assert result["skipped_papers"][0].paper_id == "2222"


def test_generate_summary_drops_duplicate_paper_id_across_themes():
    papers = [_paper("1111", "Paper One"), _paper("2222", "Paper Two")]
    schema = _build_response_schema([p.paper_id for p in papers])
    theme_cls = schema.model_fields["themes"].annotation.__args__[0]
    summary_cls = theme_cls.model_fields["papers"].annotation.__args__[0]

    # The model places paper_id "1111" in two different themes — the Literal
    # grounding permits this (each reference is individually a real paper_id),
    # but generate_summary must keep only the first occurrence.
    parsed = schema(
        themes=[
            theme_cls(theme_name="First Theme", papers=[summary_cls(paper_id="1111", summary="first occurrence")]),
            theme_cls(theme_name="Second Theme", papers=[
                summary_cls(paper_id="1111", summary="duplicate occurrence"),
                summary_cls(paper_id="2222", summary="fine, only referenced once"),
            ]),
        ],
        gaps_and_disagreements="No notable gaps or disagreements observed among the retrieved papers.",
    )
    mock_message = MagicMock(parsed=parsed, refusal=None)
    mock_usage = MagicMock(total_tokens=100, prompt_tokens=80, completion_tokens=20)
    mock_response = MagicMock(usage=mock_usage)
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = mock_response

    result = generate_summary("some topic", papers, client=mock_client)

    all_paper_ids = [p["paper"].paper_id for theme in result["themes"] for p in theme["papers"]]
    assert all_paper_ids == ["1111", "2222"]  # "1111" appears exactly once, in its first theme

    first_theme_summary = result["themes"][0]["papers"][0]["summary"]
    assert first_theme_summary == "first occurrence"  # first occurrence wins, not overwritten

    assert result["skipped_papers"] == []  # every input paper was referenced at least once


def test_generate_summary_returns_empty_for_no_papers():
    result = generate_summary("topic", [], client=MagicMock())
    assert result == {"themes": [], "gaps_and_disagreements": "", "skipped_papers": []}


def test_web_response_schema_rejects_unknown_url():
    schema = _build_web_response_schema(["https://a.com", "https://b.com"])
    schema(synthesis="fine", cited_urls=["https://a.com"])  # known url: should not raise
    try:
        schema(synthesis="fabricated", cited_urls=["https://not-retrieved.com"])
        assert False, "expected a validation error for an unretrieved URL"
    except Exception:
        pass


def test_generate_web_summary_returns_cited_articles_only():
    articles = [_web_article("https://a.com", "Article A"), _web_article("https://b.com", "Article B")]
    schema = _build_web_response_schema([a.url for a in articles])

    parsed = schema(synthesis="Article A discusses recent tooling.", cited_urls=["https://a.com"])
    mock_message = MagicMock(parsed=parsed, refusal=None)
    mock_usage = MagicMock(total_tokens=50, prompt_tokens=40, completion_tokens=10)
    mock_response = MagicMock(usage=mock_usage)
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = mock_response

    result = generate_web_summary("some topic", articles, client=mock_client)

    assert result["synthesis"] == "Article A discusses recent tooling."
    assert len(result["cited_articles"]) == 1
    assert result["cited_articles"][0].url == "https://a.com"


def test_generate_web_summary_returns_empty_for_no_articles():
    result = generate_web_summary("topic", [], client=MagicMock())
    assert result == {"synthesis": "", "cited_articles": []}


if __name__ == "__main__":
    test_response_schema_rejects_unknown_paper_id()
    test_generate_summary_attaches_citations_and_flags_skipped()
    test_generate_summary_drops_duplicate_paper_id_across_themes()
    test_generate_summary_returns_empty_for_no_papers()
    test_web_response_schema_rejects_unknown_url()
    test_generate_web_summary_returns_cited_articles_only()
    test_generate_web_summary_returns_empty_for_no_articles()
    print("All summarize tests passed.")
