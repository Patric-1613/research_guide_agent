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
from research_agent.schema import Paper


def _paper(title: str, source: str, paper_id: str) -> Paper:
    return Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract="An abstract about " + title, url=f"http://example.com/{paper_id}",
        doi=None, citation_count=None, source=source, paper_id=paper_id,
    )


def test_search_tools_accumulate_and_dedup_into_session():
    session = ResearchSession()
    arxiv_tool, s2_tool, rerank_tool = build_tools(session)

    with patch("research_agent.agent.search_arxiv") as mock_arxiv:
        mock_arxiv.return_value = [_paper("Same Paper", "arxiv", "1111.1111")]
        arxiv_tool.invoke({"query": "test", "max_results": 5})

    assert len(session.papers) == 1

    with patch("research_agent.agent.search_semantic_scholar") as mock_s2:
        # Same title from a different source -> should merge, not duplicate.
        mock_s2.return_value = [_paper("Same Paper", "semantic_scholar", "abc123")]
        s2_tool.invoke({"query": "test", "max_results": 5})

    assert len(session.papers) == 1
    assert session.papers[0].source == "arxiv+semantic_scholar"


def test_rerank_tool_reports_empty_pool_without_crashing():
    session = ResearchSession()
    _, _, rerank_tool = build_tools(session)
    result = rerank_tool.invoke({"query": "anything", "top_k": 5})
    assert "No papers collected" in result


if __name__ == "__main__":
    test_search_tools_accumulate_and_dedup_into_session()
    test_rerank_tool_reports_empty_pool_without_crashing()
    print("All agent tests passed.")
