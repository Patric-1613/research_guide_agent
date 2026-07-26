"""Deterministic tests for qa.py's non-LLM logic: schema constraints, the
defensive "no citations if unanswerable" override, empty-pool handling,
condense-question being skipped on the first turn, and the
classify_message non-substantive-message gate (semantic-classify-message:
embedding similarity, not exact-match). The OpenAI calls are mocked so
these run without network access or billing — real embedding similarity
scores for the adversarial cases live in
scripts/test_semantic_classify_live.py instead, since that needs a real
embedding call to mean anything.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch

from research_agent.qa import MAX_HISTORY_TURNS, ChatSession, _build_answer_schema, _classify_non_substantive, _condense_question, _recent_history, ask
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

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)):
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

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = ask(session, "unanswerable question", client=mock_client)

    assert result["answerable"] is False
    assert result["cited_papers"] == []  # forced empty despite model returning an id


# --- classify_message: semantic non-substantive-message gate ---
#
# Two layers of tests. The gating tests below (question mark / word count
# / empty) are pure, deterministic, and must short-circuit BEFORE any
# embedding call is made — asserted directly via a MagicMock client whose
# .embeddings.create is never touched. The threshold-decision tests mock
# _get_reference_embeddings/_embed_with_cache with small hand-picked
# vectors so the >=threshold boundary itself is deterministically
# testable without a real embedding call. Real similarity scores for the
# actual adversarial phrases ("hii", "thnak you", "thx", ...) are in
# scripts/test_semantic_classify_live.py — a mocked vector can prove the
# threshold LOGIC is correct, but only a real embedding call can prove
# real typos land above it.

def test_classify_non_substantive_question_mark_veto_short_circuits_before_any_embedding_call():
    """The canonical trap: a real follow-up question that happens to open
    with a closer phrase. The question mark alone must be enough to save
    it — and must never even reach the embedding call."""
    mock_client = MagicMock()
    is_skip, category, score = _classify_non_substantive("Thanks, but what about its limitations?", mock_client)
    assert (is_skip, category, score) == (False, None, 0.0)
    mock_client.embeddings.create.assert_not_called()


def test_classify_non_substantive_length_gate_short_circuits_before_any_embedding_call():
    """Isolates the word-count gate: no question mark present, so this
    relies entirely on the message being too long to be a bare closer —
    and, same as the question-mark veto, must never reach the embedding
    call (a real follow-up shouldn't cost an embedding call it can't use)."""
    mock_client = MagicMock()
    for message in ["thanks but can you also compare it to BERT", "ok what does the paper say about this"]:
        is_skip, category, score = _classify_non_substantive(message, mock_client)
        assert (is_skip, category, score) == (False, None, 0.0), f"expected {message!r} to fail the length gate"
    mock_client.embeddings.create.assert_not_called()


def test_classify_non_substantive_content_override_word_short_circuits_before_any_embedding_call():
    """Isolates the wh-word/imperative-verb guard specifically: short
    (<=5 words), no question mark — so neither of the other two guards
    would catch these — real measured evidence showed similarity alone
    scores these HIGHER than genuine closers ("thanks so explain" 0.6104
    vs. "sup" 0.5406), so this guard is the only thing protecting them."""
    mock_client = MagicMock()
    for message in ["Thanks explain", "Ok cool but why", "Hi define overfitting", "Great but how"]:
        is_skip, category, score = _classify_non_substantive(message, mock_client)
        assert (is_skip, category, score) == (False, None, 0.0), f"expected {message!r} to be caught by the content-override guard"
    mock_client.embeddings.create.assert_not_called()


def test_classify_non_substantive_empty_message_does_not_skip_and_skips_embedding_call():
    mock_client = MagicMock()
    for message in ["", "   "]:
        is_skip, category, score = _classify_non_substantive(message, mock_client)
        assert (is_skip, category, score) == (False, None, 0.0)
    mock_client.embeddings.create.assert_not_called()


def test_classify_non_substantive_similarity_at_or_above_threshold_skips_with_matched_category():
    mock_client = MagicMock()
    # [0.45, 0.8930] is unit-length (0.45^2 + 0.8930^2 ~= 1), so its cosine
    # similarity against [1.0, 0.0] is exactly ~0.45 — the real threshold
    # itself (see _NON_SUBSTANTIVE_SIMILARITY_THRESHOLD's own comment for
    # how that value was derived from real scores), deliberately chosen to
    # test the boundary (>=), not just comfortably above it.
    with patch("research_agent.qa._get_reference_embeddings", return_value={"acknowledgment": [("thanks", [0.45, 0.8930])]}), \
         patch("research_agent.qa._embed_with_cache", return_value=[1.0, 0.0]):
        is_skip, category, score = _classify_non_substantive("thnx", mock_client)
    assert score == pytest.approx(0.45, abs=1e-3)
    assert category == "acknowledgment"
    assert is_skip is True


def test_classify_non_substantive_similarity_below_threshold_does_not_skip():
    mock_client = MagicMock()
    with patch("research_agent.qa._get_reference_embeddings", return_value={"greeting": [("hi", [1.0, 0.0])]}), \
         patch("research_agent.qa._embed_with_cache", return_value=[0.0, 1.0]):  # orthogonal -> similarity 0.0
        is_skip, category, score = _classify_non_substantive("what is transfer learning", mock_client)
    assert score == pytest.approx(0.0)
    assert category is None
    assert is_skip is False


def test_classify_non_substantive_picks_the_best_scoring_category_across_multiple():
    mock_client = MagicMock()
    with patch("research_agent.qa._get_reference_embeddings", return_value={
        "greeting": [("hi", [0.0, 1.0])],
        "acknowledgment": [("thanks", [1.0, 0.0])],
    }), patch("research_agent.qa._embed_with_cache", return_value=[1.0, 0.0]):
        is_skip, category, score = _classify_non_substantive("thanks", mock_client)
    assert category == "acknowledgment"  # exact match (similarity 1.0), not greeting (similarity 0.0)
    assert score == pytest.approx(1.0)
    assert is_skip is True


def test_ask_short_circuits_on_non_substantive_message_without_any_llm_or_retrieval_call():
    """Mocks _classify_non_substantive directly rather than the embedding
    call it makes internally — this test's job is proving ask() reacts
    correctly to a "skip" verdict, not re-proving the classifier's own
    similarity math (that's the dedicated _classify_non_substantive tests
    above, plus real scores in scripts/test_semantic_classify_live.py)."""
    papers = [_paper("1111", "Paper One")]
    session = ChatSession(papers=papers, history=[
        {"role": "user", "content": "what is RoCoFT?"},
        {"role": "assistant", "content": "RoCoFT is a parameter-efficient fine-tuning method."},
    ])
    mock_client = MagicMock()

    with patch("research_agent.qa._classify_non_substantive", return_value=(True, "acknowledgment", 0.91)), \
         patch("research_agent.qa.embed_and_index_papers") as mock_embed, \
         patch("research_agent.qa.get_chroma_collection") as mock_collection, \
         patch("research_agent.qa.semantic_search") as mock_search:
        result = ask(session, "Thanks!", client=mock_client)

    assert result["answerable"] is True
    assert result["answer"] == "You're welcome! Let me know if you have any more questions about these papers."
    mock_client.chat.completions.create.assert_not_called()  # condense_question never ran
    mock_client.chat.completions.parse.assert_not_called()  # generate_answer never ran
    mock_embed.assert_not_called()
    mock_collection.assert_not_called()
    mock_search.assert_not_called()
    assert session.history[-2:] == [
        {"role": "user", "content": "Thanks!"},
        {"role": "assistant", "content": "You're welcome! Let me know if you have any more questions about these papers."},
    ]


def test_ask_does_not_short_circuit_the_trap_case_thanks_but_a_real_question():
    """The real regression test: a message opening with a closer phrase but
    containing an actual follow-up question must go through the full
    condense/retrieve/generate pipeline, not the canned response."""
    papers = [_paper("1111", "Paper One")]
    session = ChatSession(papers=papers, history=[
        {"role": "user", "content": "what is RoCoFT?"},
        {"role": "assistant", "content": "RoCoFT is a parameter-efficient fine-tuning method."},
    ])
    schema = _build_answer_schema(["1111"])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="What are RoCoFT's limitations?"))],
        usage=MagicMock(total_tokens=50, prompt_tokens=40, completion_tokens=10),
    )
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="RoCoFT's main limitation is X [Paper 1].", cited_paper_ids=["1111"],
    )

    with patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = ask(session, "Thanks, but what about its limitations?", client=mock_client)

    assert result["answerable"] is True
    assert len(result["cited_papers"]) == 1
    mock_client.chat.completions.create.assert_called_once()  # condense_question DID run
    mock_client.chat.completions.parse.assert_called_once()  # generate_answer DID run


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

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
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
    test_classify_non_substantive_question_mark_veto_short_circuits_before_any_embedding_call()
    test_classify_non_substantive_length_gate_short_circuits_before_any_embedding_call()
    test_classify_non_substantive_content_override_word_short_circuits_before_any_embedding_call()
    test_classify_non_substantive_empty_message_does_not_skip_and_skips_embedding_call()
    test_classify_non_substantive_similarity_at_or_above_threshold_skips_with_matched_category()
    test_classify_non_substantive_similarity_below_threshold_does_not_skip()
    test_classify_non_substantive_picks_the_best_scoring_category_across_multiple()
    test_ask_short_circuits_on_non_substantive_message_without_any_llm_or_retrieval_call()
    test_ask_does_not_short_circuit_the_trap_case_thanks_but_a_real_question()
    test_recent_history_keeps_only_last_n_turns()
    test_recent_history_is_a_no_op_below_the_cap()
    test_ask_caps_history_to_last_n_turns_in_prompt_sent_to_model()
    print("All qa tests passed.")
