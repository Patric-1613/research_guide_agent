"""Deterministic tests for qa.py's non-LLM logic: schema constraints, the
defensive "no citations if unanswerable" override, empty-pool handling, and
condense-question being skipped on the first turn. The OpenAI calls are
mocked so these run without network access or billing.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.qa import ChatSession, _build_answer_schema, _condense_question, ask
from research_agent.schema import Paper


def _paper(paper_id: str, title: str) -> Paper:
    return Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract=f"Abstract for {title}.", url=f"http://arxiv.org/abs/{paper_id}",
        doi=None, citation_count=None, source="arxiv", paper_id=paper_id,
    )


def _mock_parse_response(schema_cls, **kwargs):
    parsed = schema_cls(**kwargs)
    mock_message = MagicMock(parsed=parsed, refusal=None)
    mock_usage = MagicMock(total_tokens=100, prompt_tokens=80, completion_tokens=20)
    mock_response = MagicMock(usage=mock_usage)
    mock_response.choices = [MagicMock(message=mock_message)]
    return mock_response


def test_answer_schema_rejects_unknown_paper_id():
    schema = _build_answer_schema(["a", "b"])
    schema(answerable=True, answer="fine [1]", cited_paper_ids=["a"])
    try:
        schema(answerable=True, answer="bad", cited_paper_ids=["not-real"])
        assert False, "expected a validation error for an unknown paper_id"
    except Exception:
        pass


def test_ask_with_no_papers_short_circuits_without_calling_client():
    session = ChatSession(papers=[])
    mock_client = MagicMock()
    result = ask(session, "anything?", client=mock_client)
    assert result["answerable"] is False
    assert result["cited_papers"] == []
    mock_client.chat.completions.parse.assert_not_called()
    # still logs the turn so a caller building a transcript sees the refusal
    assert session.history[-2] == {"role": "user", "content": "anything?"}


def test_condense_question_skips_llm_call_on_first_turn():
    mock_client = MagicMock()
    result = _condense_question([], "what about it?", mock_client)
    assert result == "what about it?"
    mock_client.chat.completions.create.assert_not_called()


def test_ask_forces_empty_citations_when_model_marks_unanswerable():
    """Defensive check: even if the model violates instructions and returns
    cited_paper_ids alongside answerable=False, ask() must not surface a
    citation on a claim the model itself says it can't support."""
    papers = [_paper("1111", "Paper One")]
    session = ChatSession(papers=papers)
    schema = _build_answer_schema(["1111"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=False, answer="I can't answer this.", cited_paper_ids=["1111"],
    )

    from unittest.mock import patch
    with patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = ask(session, "unanswerable question", client=mock_client)

    assert result["answerable"] is False
    assert result["cited_papers"] == []  # forced empty despite model returning an id


if __name__ == "__main__":
    test_answer_schema_rejects_unknown_paper_id()
    test_ask_with_no_papers_short_circuits_without_calling_client()
    test_condense_question_skips_llm_call_on_first_turn()
    test_ask_forces_empty_citations_when_model_marks_unanswerable()
    print("All qa tests passed.")
