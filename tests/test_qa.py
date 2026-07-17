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

from unittest.mock import patch

from research_agent.qa import MAX_HISTORY_TURNS, ChatSession, _build_answer_schema, _condense_question, _recent_history, ask
from research_agent.schema import Paper, WebArticle


def _paper(paper_id: str, title: str) -> Paper:
    return Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract=f"Abstract for {title}.", url=f"http://arxiv.org/abs/{paper_id}",
        doi=None, citation_count=None, source="arxiv", paper_id=paper_id,
    )


def _web_article(url: str, title: str) -> WebArticle:
    return WebArticle(title=title, url=url, snippet=f"Snippet for {title}.", published_date=None, source_domain="example.com")


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


def test_answer_schema_without_web_urls_has_no_cited_web_urls_field():
    schema = _build_answer_schema(["a"])  # no web_urls -> field must not exist at all
    assert "cited_web_urls" not in schema.model_fields


def test_answer_schema_rejects_unknown_web_url():
    schema = _build_answer_schema(["a"], ["https://real.com"])
    schema(answerable=True, answer="fine [Paper 1] [Web 1]", cited_paper_ids=["a"], cited_web_urls=["https://real.com"])
    try:
        schema(answerable=True, answer="bad", cited_paper_ids=[], cited_web_urls=["https://not-retrieved.com"])
        assert False, "expected a validation error for an unretrieved URL"
    except Exception:
        pass


def test_ask_with_only_web_articles_no_papers_still_answers():
    session = ChatSession(papers=[], web_articles=[_web_article("https://x.com/a", "Article A")])
    schema = _build_answer_schema([], ["https://x.com/a"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Per [Web 1], X is true.", cited_paper_ids=[], cited_web_urls=["https://x.com/a"],
    )

    result = ask(session, "what does the web say?", client=mock_client)

    assert result["answerable"] is True
    assert result["cited_papers"] == []
    assert len(result["cited_web_articles"]) == 1
    assert result["cited_web_articles"][0].url == "https://x.com/a"
    mock_client.chat.completions.parse.assert_called_once()


def test_ask_forces_empty_web_citations_when_model_marks_unanswerable():
    session = ChatSession(papers=[], web_articles=[_web_article("https://x.com/a", "Article A")])
    schema = _build_answer_schema([], ["https://x.com/a"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=False, answer="Can't answer.", cited_paper_ids=[], cited_web_urls=["https://x.com/a"],
    )

    result = ask(session, "unanswerable", client=mock_client)

    assert result["answerable"] is False
    assert result["cited_web_articles"] == []  # forced empty despite model returning a url


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

    with patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = ask(session, "unanswerable question", client=mock_client)

    assert result["answerable"] is False
    assert result["cited_papers"] == []  # forced empty despite model returning an id


def test_recent_history_keeps_only_last_n_turns():
    history = []
    for i in range(12):
        history.append({"role": "user", "content": f"question {i}"})
        history.append({"role": "assistant", "content": f"answer {i}"})

    capped = _recent_history(history, max_turns=3)
    assert capped == [
        {"role": "user", "content": "question 9"},
        {"role": "assistant", "content": "answer 9"},
        {"role": "user", "content": "question 10"},
        {"role": "assistant", "content": "answer 10"},
        {"role": "user", "content": "question 11"},
        {"role": "assistant", "content": "answer 11"},
    ]


def test_recent_history_is_a_no_op_below_the_cap():
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert _recent_history(history, max_turns=8) == history


def test_ask_caps_history_to_last_n_turns_in_prompt_sent_to_model():
    # A long simulated conversation (12 prior turns, more than the 8-turn
    # cap) — only the last MAX_HISTORY_TURNS turns should reach the actual
    # prompt sent to the model; older ones must be dropped.
    papers = [_paper("1111", "Paper One")]
    session = ChatSession(papers=papers)
    for i in range(12):
        session.history.append({"role": "user", "content": f"question {i}"})
        session.history.append({"role": "assistant", "content": f"answer {i}"})

    schema = _build_answer_schema(["1111"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Final answer [Paper 1].", cited_paper_ids=["1111"],
    )

    with patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        ask(session, "new question", client=mock_client)

    sent_messages = mock_client.chat.completions.parse.call_args.kwargs["messages"]
    # [0] is the system prompt, [-1] is the new question + context, and
    # everything in between should be exactly the last MAX_HISTORY_TURNS
    # turns (2 messages each) — no more, no less.
    history_in_prompt = sent_messages[1:-1]
    assert len(history_in_prompt) == 2 * MAX_HISTORY_TURNS
    assert history_in_prompt[0] == {"role": "user", "content": "question 4"}  # oldest turn kept
    assert history_in_prompt[-1] == {"role": "assistant", "content": "answer 11"}  # newest prior turn

    sent_contents = [m["content"] for m in sent_messages if isinstance(m.get("content"), str)]
    assert "question 0" not in sent_contents  # dropped: older than the cap
    assert "question 3" not in sent_contents  # dropped: older than the cap

    # The full, uncapped history is still preserved on the session object —
    # only the prompt sent to the model is capped, not what's stored (a
    # caller building a UI transcript still sees every turn).
    assert len(session.history) == 12 * 2 + 2


if __name__ == "__main__":
    test_answer_schema_rejects_unknown_paper_id()
    test_ask_with_no_papers_short_circuits_without_calling_client()
    test_condense_question_skips_llm_call_on_first_turn()
    test_answer_schema_without_web_urls_has_no_cited_web_urls_field()
    test_answer_schema_rejects_unknown_web_url()
    test_ask_with_only_web_articles_no_papers_still_answers()
    test_ask_forces_empty_web_citations_when_model_marks_unanswerable()
    test_ask_forces_empty_citations_when_model_marks_unanswerable()
    test_recent_history_keeps_only_last_n_turns()
    test_recent_history_is_a_no_op_below_the_cap()
    test_ask_caps_history_to_last_n_turns_in_prompt_sent_to_model()
    print("All qa tests passed.")
