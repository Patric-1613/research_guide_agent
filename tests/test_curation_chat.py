"""Tests for curation_chat.py (curation-chat-web-escalation Phase 5b/5c):
5b baseline chat grounded in a curation session's selected_papers, plus
5c's offer-and-decide web escalation mechanism on top. The OpenAI call
is mocked (same convention as test_qa.py) since these exist to prove
the wiring — that ask_in_session()/chat_turn() correctly build session
state and delegate to qa.ask()/search_web() unmodified — not to
re-prove qa.py's own grounding guarantee, which test_qa.py already
covers. Real live output (including the offer-response classifier
against real ambiguous/unrelated messages) is verified separately in
scripts/test_curation_chat.py.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.curation_chat import _OfferResponseIntent, _classify_offer_response, ask_in_session, chat_turn
from research_agent.qa import _build_answer_schema
from research_agent.query_expansion import PaperPoolSession
from research_agent.schema import Paper, WebArticle


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


def test_ask_in_session_grounds_in_selected_papers_and_updates_chat_history():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="parameter-efficient fine-tuning",
        selected_paper_ids=["p1"],
        selected_papers=papers,
        stage="synthesize",
    )
    schema = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="RoCoFT updates rows/columns [Paper 1].", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = ask_in_session(session, "what does RoCoFT update?", client=mock_client)

    assert result["answerable"] is True
    assert [p.paper_id for p in result["cited_papers"]] == ["p1"]
    # session.chat_history (not some private copy) reflects the new turn --
    # this is the actual state a caller would persist via curation_session.py
    assert session.chat_history == [
        {"role": "user", "content": "what does RoCoFT update?"},
        {"role": "assistant", "content": "RoCoFT updates rows/columns [Paper 1]."},
    ]


def test_ask_in_session_second_turn_sees_first_turns_history():
    """Proves chat_history round-trips into the NEXT call, not just that
    ask_in_session updates it once -- the whole point of persisting
    chat_history on the session is multi-turn continuity across calls
    (and, eventually, across process restarts via curation_session.py)."""
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="parameter-efficient fine-tuning",
        selected_paper_ids=["p1"],
        selected_papers=papers,
        stage="synthesize",
        chat_history=[
            {"role": "user", "content": "what is RoCoFT?"},
            {"role": "assistant", "content": "RoCoFT is a PEFT method [Paper 1]."},
        ],
    )
    schema = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="what does it update?"))],
        usage=MagicMock(total_tokens=50, prompt_tokens=40, completion_tokens=10),
    )
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Rows and columns [Paper 1].", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        ask_in_session(session, "what does it update?", client=mock_client)

    # condense_question was actually invoked with the pre-existing history
    mock_client.chat.completions.create.assert_called_once()
    condense_call_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assert "what is RoCoFT?" in condense_call_messages[1]["content"]
    assert len(session.chat_history) == 4


def test_ask_in_session_with_no_selected_papers_or_web_articles_short_circuits():
    session = PaperPoolSession(topic="empty session", stage="synthesize")
    mock_client = MagicMock()

    result = ask_in_session(session, "anything?", client=mock_client)

    assert result["answerable"] is False
    mock_client.chat.completions.parse.assert_not_called()
    assert session.chat_history[-2] == {"role": "user", "content": "anything?"}


# --- chat_turn(): Phase 5c offer-and-decide web escalation ---

def _mock_intent_response(intent: str):
    parsed = _OfferResponseIntent(intent=intent)
    mock_message = MagicMock(parsed=parsed, refusal=None)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    return mock_response


def test_chat_turn_sets_pending_web_offer_when_unanswerable():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize")
    schema = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=False, answer="Not covered by the selected papers.", cited_paper_ids=[],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = chat_turn(session, "what's the SOTA on GLUE in 2026?", client=mock_client)

    assert result["web_offer_made"] is True
    assert result["answer"].endswith("Would you like me to search the web for more on this?")
    assert session.pending_web_offer == {"question": "what's the SOTA on GLUE in 2026?"}
    # the offer text landed in chat_history too, not just the returned dict
    assert session.chat_history[-1]["content"] == result["answer"]


def test_chat_turn_does_not_set_pending_web_offer_when_answerable():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize")
    schema = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Fine [Paper 1].", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = chat_turn(session, "what is RoCoFT?", client=mock_client)

    assert "web_offer_made" not in result
    assert session.pending_web_offer is None


def test_chat_turn_decline_never_rearms_a_new_offer_even_if_the_decline_text_itself_comes_back_unanswerable():
    """Regression test for a real bug caught by scripts/test_curation_chat_offer.py:
    "nah, papers are enough" wasn't recognized as non-substantive by
    qa.py's own gate, fell through to a real answer attempt, and (since
    it isn't a real question) came back answerable=False -- which used
    to re-arm pending_web_offer from the decline text itself, a
    nonsensical loop. decline must never re-trigger _maybe_set_web_offer,
    regardless of what ask_in_session returns for it."""
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        pending_web_offer={"question": "what's new in 2026?"},
    )
    schema = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("decline"),
        _mock_parse_response(schema, answerable=False, answer="That's not a question I can answer.", cited_paper_ids=[]),
    ]

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = chat_turn(session, "nah, papers are enough", client=mock_client)

    assert result["web_offer_declined"] is True
    assert "web_offer_made" not in result
    assert session.pending_web_offer is None  # not re-armed from the decline text


def test_chat_turn_accept_triggers_web_search_adds_articles_and_clears_offer():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        pending_web_offer={"question": "what's new in 2026?"},
    )
    new_article = WebArticle(
        title="2026 roundup", url="https://x.com/roundup", snippet="s",
        published_date=None, source_domain="x.com",
    )
    schema = _build_answer_schema(["p1"], ["https://x.com/roundup"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("accept"),
        _mock_parse_response(
            schema, answerable=True, answer="Per [Web 1], ...", cited_paper_ids=[], cited_web_urls=["https://x.com/roundup"],
        ),
    ]

    with patch("research_agent.curation_chat.search_web", return_value=[new_article]) as mock_search, \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = chat_turn(session, "yes please", client=mock_client)

    mock_search.assert_called_once_with("what's new in 2026?")
    assert session.pending_web_offer is None
    assert session.web_articles_added == [new_article]
    assert result["web_search_used"] is True
    assert result["new_web_articles_found"] == 1


def test_chat_turn_accept_records_a_curated_search_label_not_a_repeated_question_or_a_bare_yes():
    """chat-ux-fixes bug 4 (second pass): accepting a web offer re-asks the
    ORIGINAL question internally (needed for real retrieval/generation
    grounding -- confirmed via mock_search's own assert above), but the
    transcript must show neither that question repeated verbatim NOR a
    context-free "yes please" -- a curated label naming the actual
    resolved search query instead. No prior chat_history here, so
    condense_question skips its LLM call and the query is the question
    unchanged (see the dedicated condensing test below for the case
    where history genuinely changes it)."""
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        pending_web_offer={"question": "what's new in 2026?"},
    )
    new_article = WebArticle(
        title="2026 roundup", url="https://x.com/roundup", snippet="s",
        published_date=None, source_domain="x.com",
    )
    schema = _build_answer_schema(["p1"], ["https://x.com/roundup"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("accept"),
        _mock_parse_response(
            schema, answerable=True, answer="Per [Web 1], ...", cited_paper_ids=[], cited_web_urls=["https://x.com/roundup"],
        ),
    ]

    with patch("research_agent.curation_chat.search_web", return_value=[new_article]), \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        chat_turn(session, "yes please", client=mock_client)

    # curation-chat-metadata Phase 1: exact dict equality no longer holds
    # (both entries now also carry a real, randomly-generated exchange_id)
    # -- stable fields checked exactly, exchange_id checked for shape/
    # sharedness instead of a hardcoded value.
    user_turn, assistant_turn = session.chat_history[-2], session.chat_history[-1]
    assert user_turn["role"] == "user"
    assert user_turn["content"] == 'Search the web for: "what\'s new in 2026?"'
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["content"] == "Per [Web 1], ..."
    assert user_turn["exchange_id"]  # non-empty
    assert user_turn["exchange_id"] == assistant_turn["exchange_id"]
    assert not any(turn["content"] == "what's new in 2026?" for turn in session.chat_history)
    assert not any(turn["content"] == "yes please" for turn in session.chat_history)


def _mock_create_response(content: str):
    mock_message = MagicMock(content=content)
    mock_usage = MagicMock(prompt_tokens=50, completion_tokens=10, total_tokens=60)
    mock_response = MagicMock(usage=mock_usage)
    mock_response.choices = [MagicMock(message=mock_message)]
    return mock_response


def test_chat_turn_accept_resolves_a_follow_up_fragment_into_a_standalone_search_query():
    """chat-ux-fixes bug 2 (second pass): the root cause behind "web
    search finds nothing new" on follow-ups -- the RAW per-turn message
    that triggered the offer (a pronoun-heavy fragment relying on
    earlier turns for context) must not be searched verbatim. qa.py's
    own condense_question is reused to resolve it into a real standalone
    query first, using the conversation history that came before the
    offer -- confirmed here by asserting search_web receives the
    CONDENSED text, not the fragment, and that the transcript label
    names the condensed query too."""
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        chat_history=[
            {"role": "user", "content": "What are the reasons invasive species are in the UK?"},
            {"role": "assistant", "content": "Several ecological and human factors contribute."},
        ],
        pending_web_offer={"question": "I mean very recent like in the 2026?"},
    )
    new_article = WebArticle(
        title="2026 sightings roundup", url="https://x.com/2026", snippet="s",
        published_date=None, source_domain="x.com",
    )
    schema = _build_answer_schema(["p1"], ["https://x.com/2026"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("accept"),
        _mock_parse_response(
            schema, answerable=True, answer="Per [Web 1], nothing recent is reported.",
            cited_paper_ids=[], cited_web_urls=["https://x.com/2026"],
        ),
    ]
    mock_client.chat.completions.create.return_value = _mock_create_response(
        "Recent invasive species sightings in the UK in 2026",
    )

    with patch("research_agent.curation_chat.search_web", return_value=[new_article]) as mock_search, \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        chat_turn(session, "yes", client=mock_client)

    mock_search.assert_called_once_with("Recent invasive species sightings in the UK in 2026")
    user_turn = session.chat_history[-2]
    assert user_turn["role"] == "user"
    assert user_turn["content"] == 'Search the web for: "Recent invasive species sightings in the UK in 2026"'
    assert user_turn["exchange_id"]  # curation-chat-metadata Phase 1: real, non-empty id
    assert not any(turn["content"] == "I mean very recent like in the 2026?" for turn in session.chat_history)


def test_chat_turn_accept_falls_back_to_the_raw_question_if_condensing_fails():
    """Defensive: an external-call failure while resolving the search
    query must degrade to searching the raw question, not blow up the
    whole accept flow -- same posture search_web's own guard already
    has for its own external call."""
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        chat_history=[
            {"role": "user", "content": "What are the reasons invasive species are in the UK?"},
            {"role": "assistant", "content": "Several ecological and human factors contribute."},
        ],
        pending_web_offer={"question": "I mean very recent like in the 2026?"},
    )
    schema = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("accept"),
        _mock_parse_response(schema, answerable=True, answer="Still nothing recent.", cited_paper_ids=[]),
    ]
    # Only the FIRST condense call (this fix's own, for the search query)
    # is meant to fail here -- ask_in_session's own internal condense call
    # (qa.py's pre-existing retrieval-condensing, unrelated to this fix
    # and not in this phase's scope) must still succeed normally, so this
    # test isolates the one guard actually being tested.
    mock_client.chat.completions.create.side_effect = [
        RuntimeError("condense call failed"),
        _mock_create_response("Recent invasive species sightings in the UK in 2026"),
    ]

    with patch("research_agent.curation_chat.search_web", return_value=[]) as mock_search, \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        chat_turn(session, "yes", client=mock_client)

    mock_search.assert_called_once_with("I mean very recent like in the 2026?")


def test_chat_turn_decline_clears_offer_without_web_search():
    """A genuine, content-free decline ("no thanks") still short-circuits
    -- not via a canned reply this module invents, but via qa.ask()'s own
    classify_message gate treating it as non-substantive (mocked here the
    same way test_qa.py mocks it, to avoid a real embedding call)."""
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        pending_web_offer={"question": "what's new in 2026?"},
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_intent_response("decline")

    with patch("research_agent.curation_chat.search_web") as mock_search, \
         patch("research_agent.qa._classify_non_substantive", return_value=(True, "acknowledgment", 0.9)):
        result = chat_turn(session, "no thanks", client=mock_client)

    mock_search.assert_not_called()
    assert session.pending_web_offer is None
    assert result["web_offer_declined"] is True
    assert result["answerable"] is True  # qa.py's own non-substantive path marks these answerable=True
    user_turn = session.chat_history[-2]
    assert user_turn["role"] == "user"
    assert user_turn["content"] == "no thanks"
    assert user_turn["exchange_id"]  # curation-chat-metadata Phase 1: real, non-empty id


def test_chat_turn_decline_with_trailing_real_question_does_not_silently_drop_it():
    """The gap this test exists to close: a message classified as
    "decline" is not guaranteed to be JUST a decline. "no wait, what
    about vector databases instead?" starts exactly like a decline but
    carries a real, unrelated question -- confirms it is actually
    answered (via qa.ask()'s own classify_message gate correctly NOT
    treating it as non-substantive, thanks to the question mark + "what"
    content-override word), not silently swallowed by a canned ack."""
    papers = [_paper("p1", "RoCoFT"), _paper("p2", "Vector DB Survey")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1", "p2"], selected_papers=papers, stage="synthesize",
        pending_web_offer={"question": "what's new in 2026?"},
    )
    schema = _build_answer_schema(["p1", "p2"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("decline"),
        _mock_parse_response(schema, answerable=True, answer="Vector DBs are covered in [Paper 2].", cited_paper_ids=["p2"]),
    ]

    with patch("research_agent.curation_chat.search_web") as mock_search, \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[1], 0.9)]):
        result = chat_turn(session, "no wait, what about vector databases instead?", client=mock_client)

    mock_search.assert_not_called()  # still a decline of the ORIGINAL offer -- no web search
    assert session.pending_web_offer is None
    assert result["web_offer_declined"] is True
    assert result["answerable"] is True
    assert [p.paper_id for p in result["cited_papers"]] == ["p2"]


def test_chat_turn_unrelated_message_while_offer_pending_clears_offer_and_answers_new_question():
    """The user's explicit Phase 5c requirement: a next message that
    neither clearly accepts nor declines a pending offer (here, a
    completely different question) must still clear pending_web_offer --
    not leave it lingering to be misread as a yes/no on some later,
    unrelated turn."""
    papers = [_paper("p1", "RoCoFT"), _paper("p2", "LoRA")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1", "p2"], selected_papers=papers, stage="synthesize",
        pending_web_offer={"question": "what's the 2026 SOTA on GLUE?"},
    )
    schema = _build_answer_schema(["p1", "p2"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("other"),
        _mock_parse_response(schema, answerable=True, answer="LoRA injects low-rank matrices [Paper 2].", cited_paper_ids=["p2"]),
    ]

    with patch("research_agent.curation_chat.search_web") as mock_search, \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[1], 0.9)]):
        result = chat_turn(session, "actually, what does LoRA inject into each layer?", client=mock_client)

    mock_search.assert_not_called()
    assert session.pending_web_offer is None
    assert result["answerable"] is True
    assert [p.paper_id for p in result["cited_papers"]] == ["p2"]
    assert "web_offer_made" not in result


def test_offer_classifier_refusal_falls_back_to_other_not_accept_or_decline():
    """If the classifier itself can't decide, the safe fallback is
    "other" (clear + answer as a fresh question) -- never "accept"
    (would trigger an unrequested web search) or "decline" (would
    silently swallow a real pending offer)."""
    mock_client = MagicMock()
    mock_message = MagicMock(parsed=None, refusal="cannot classify")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.parse.return_value = mock_response

    intent = _classify_offer_response("some question", "hmm", mock_client)
    assert intent == "other"


def test_accept_with_no_new_web_results_does_not_immediately_reoffer():
    """search_web() degrades to [] on no-results/failure rather than
    raising (see web_search.py) -- confirms that case doesn't cause an
    infinite offer loop by re-triggering _maybe_set_web_offer right
    after the user just accepted the same search."""
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        pending_web_offer={"question": "obscure question"},
    )
    schema = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("accept"),
        _mock_parse_response(schema, answerable=False, answer="Still not covered.", cited_paper_ids=[]),
    ]

    with patch("research_agent.curation_chat.search_web", return_value=[]), \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = chat_turn(session, "yes", client=mock_client)

    assert result["new_web_articles_found"] == 0
    assert session.pending_web_offer is None  # not re-set even though still unanswerable
    assert "web_offer_made" not in result


# --- Phase 5e: web search flakiness + the chat-readiness stage guard ---

def test_accept_when_search_web_raises_unexpectedly_does_not_crash_or_corrupt_offer_state():
    """search_web()'s own docstring promises it never raises, but this
    project has hit real, recurring flakiness from external search APIs
    all session (arXiv, Semantic Scholar) -- trusting that contract alone
    would be exactly the kind of assumption this project's discipline
    says not to make. Confirms chat_turn() degrades gracefully (no crash)
    and pending_web_offer ends up cleanly cleared, not corrupted, even
    when search_web violates its own contract outright."""
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        pending_web_offer={"question": "obscure question"},
    )
    schema = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("accept"),
        _mock_parse_response(schema, answerable=False, answer="Still not covered.", cited_paper_ids=[]),
    ]

    with patch("research_agent.curation_chat.search_web", side_effect=RuntimeError("simulated Tavily hard failure")), \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = chat_turn(session, "yes", client=mock_client)  # must not raise

    assert result["web_search_used"] is True
    assert result["new_web_articles_found"] == 0
    assert session.web_articles_added == []
    assert session.pending_web_offer is None  # cleared cleanly, not left corrupted


def test_chat_turn_refuses_when_session_has_not_finished_curation():
    """Same guard pattern as report.py's generate_report_for_session():
    a session still in "curate" hasn't finished picking papers yet, so
    there's nothing stable to ground chat in."""
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(topic="peft", selected_papers=papers, stage="curate")
    mock_client = MagicMock()

    try:
        chat_turn(session, "what is RoCoFT?", client=mock_client)
        assert False, "expected a ValueError for a session not yet ready for chat"
    except ValueError as e:
        assert "curate" in str(e)
    mock_client.chat.completions.parse.assert_not_called()


# --- curation-refinement-and-auto-offer Phase 6f-3: automatic report-update offer ---
# Same rigor as Phase 5c's web-offer testing above, including the exact
# bug classes found there (an ignored offer must not persist; a
# decline-shaped message with a real trailing question must not be
# silently dropped).

def _report_stub(cited_papers: list[Paper] | None = None) -> dict:
    section = {"content": "content", "cited_papers": cited_papers or []}
    return {
        "findings": section, "limitations": {"content": "", "cited_papers": []},
        "future_scope": {"content": "", "cited_papers": []}, "skipped_papers": [],
    }


def test_accept_web_offer_sets_pending_report_update_when_report_becomes_stale():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        pending_web_offer={"question": "what's new in 2026?"},
        report=_report_stub(), report_covered_web_article_count=0,
    )
    new_article = WebArticle(title="2026 roundup", url="https://x.com/roundup", snippet="s", published_date=None, source_domain="x.com")
    schema = _build_answer_schema(["p1"], ["https://x.com/roundup"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("accept"),
        _mock_parse_response(schema, answerable=True, answer="Per [Web 1], ...", cited_paper_ids=[], cited_web_urls=["https://x.com/roundup"]),
    ]

    with patch("research_agent.curation_chat.search_web", return_value=[new_article]), \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = chat_turn(session, "yes please", client=mock_client)

    assert result["report_update_offer_made"] is True
    assert result["answer"].endswith("want me to update it to include them?")
    assert session.pending_report_update == {"new_article_count": 1}
    assert session.chat_history[-1]["content"] == result["answer"]


def test_accept_web_offer_does_not_offer_report_update_when_no_report_exists():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        pending_web_offer={"question": "what's new in 2026?"}, report=None,
    )
    new_article = WebArticle(title="2026 roundup", url="https://x.com/roundup", snippet="s", published_date=None, source_domain="x.com")
    schema = _build_answer_schema(["p1"], ["https://x.com/roundup"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("accept"),
        _mock_parse_response(schema, answerable=True, answer="Per [Web 1], ...", cited_paper_ids=[], cited_web_urls=["https://x.com/roundup"]),
    ]

    with patch("research_agent.curation_chat.search_web", return_value=[new_article]), \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = chat_turn(session, "yes please", client=mock_client)

    assert "report_update_offer_made" not in result
    assert session.pending_report_update is None


def test_accept_web_offer_does_not_offer_report_update_when_report_already_covers_current_articles():
    """search_web() finds nothing genuinely NEW (already-known URL) --
    web_articles_added doesn't grow past what the report already
    covers, so no offer should fire even though a report exists."""
    papers = [_paper("p1", "RoCoFT")]
    existing_article = WebArticle(title="Old roundup", url="https://x.com/old", snippet="s", published_date=None, source_domain="x.com")
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        pending_web_offer={"question": "what's new in 2026?"},
        report=_report_stub(), report_covered_web_article_count=1,
        web_articles_added=[existing_article],
    )
    schema = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("accept"),
        _mock_parse_response(schema, answerable=False, answer="Still not covered.", cited_paper_ids=[]),
    ]

    with patch("research_agent.curation_chat.search_web", return_value=[]), \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = chat_turn(session, "yes please", client=mock_client)

    assert "report_update_offer_made" not in result
    assert session.pending_report_update is None


def test_chat_turn_report_update_accept_regenerates_and_clears_offer():
    """chat-ux-fixes bug 4 (second pass): the transcript shows a curated
    label ("Update the report to include N new source(s)") instead of
    the literal accept message -- same reasoning as the web-offer accept
    label: every "yes" in a conversation must not look identical."""
    papers = [_paper("p1", "RoCoFT")]
    new_article = WebArticle(title="2026 roundup", url="https://x.com/roundup", snippet="s", published_date=None, source_domain="x.com")
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        report=_report_stub(), report_covered_web_article_count=0,
        web_articles_added=[new_article],
        pending_report_update={"new_article_count": 1},
    )
    updated_report = _report_stub(cited_papers=[papers[0]])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_intent_response("accept")

    with patch("research_agent.curation_chat.regenerate_report_with_new_sources", return_value=updated_report) as mock_regen:
        result = chat_turn(session, "yes, please update it", client=mock_client)

    mock_regen.assert_called_once_with(session, client=mock_client)
    assert session.pending_report_update is None
    assert session.report is updated_report
    assert session.report_covered_web_article_count == 1
    assert result["report_updated"] is True
    user_turn, assistant_turn = session.chat_history[-2], session.chat_history[-1]
    assert user_turn["role"] == "user"
    assert user_turn["content"] == "Update the report to include 1 new source"
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["content"] == result["answer"]
    assert user_turn["exchange_id"]  # curation-chat-metadata Phase 1: real, non-empty id
    assert user_turn["exchange_id"] == assistant_turn["exchange_id"]
    # This answer carries no citations at all -- not a web-search answer.
    assert assistant_turn["used_web_search"] is False
    assert assistant_turn["cited_web_articles"] == []
    assert assistant_turn["added_to_report"] is False


def test_chat_turn_report_update_decline_clears_offer_without_regenerating():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        report=_report_stub(), report_covered_web_article_count=0,
        pending_report_update={"new_article_count": 1},
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_intent_response("decline")

    with patch("research_agent.curation_chat.regenerate_report_with_new_sources") as mock_regen, \
         patch("research_agent.qa._classify_non_substantive", return_value=(True, "acknowledgment", 0.9)):
        result = chat_turn(session, "no thanks", client=mock_client)

    mock_regen.assert_not_called()
    assert session.pending_report_update is None
    assert result["report_update_declined"] is True
    assert session.report_covered_web_article_count == 0  # unchanged -- report is still the old one


def test_chat_turn_report_update_decline_with_trailing_real_question_does_not_silently_drop_it():
    """Reapplies the EXACT Phase 5c fix to the report-update offer: a
    message that reads as "decline" is not guaranteed to be JUST a
    decline."""
    papers = [_paper("p1", "RoCoFT"), _paper("p2", "Vector DB Survey")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1", "p2"], selected_papers=papers, stage="synthesize",
        report=_report_stub(), report_covered_web_article_count=0,
        pending_report_update={"new_article_count": 1},
    )
    schema = _build_answer_schema(["p1", "p2"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("decline"),
        _mock_parse_response(schema, answerable=True, answer="Vector DBs are covered in [Paper 2].", cited_paper_ids=["p2"]),
    ]

    with patch("research_agent.curation_chat.regenerate_report_with_new_sources") as mock_regen, \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[1], 0.9)]):
        result = chat_turn(session, "no wait, what about vector databases instead?", client=mock_client)

    mock_regen.assert_not_called()
    assert session.pending_report_update is None
    assert result["report_update_declined"] is True
    assert result["answerable"] is True
    assert [p.paper_id for p in result["cited_papers"]] == ["p2"]


def test_chat_turn_report_update_decline_never_rearms_any_offer_even_if_the_decline_text_comes_back_unanswerable():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        report=_report_stub(), report_covered_web_article_count=0,
        pending_report_update={"new_article_count": 1},
    )
    schema = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("decline"),
        _mock_parse_response(schema, answerable=False, answer="Not a question I can answer.", cited_paper_ids=[]),
    ]

    with patch("research_agent.curation_chat.regenerate_report_with_new_sources") as mock_regen, \
         patch("research_agent.curation_chat.search_web") as mock_search, \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        result = chat_turn(session, "nah, don't bother", client=mock_client)

    mock_regen.assert_not_called()
    mock_search.assert_not_called()
    assert session.pending_report_update is None
    assert session.pending_web_offer is None  # not re-armed from the decline text either
    assert "report_update_offer_made" not in result
    assert "web_offer_made" not in result


def test_chat_turn_report_update_unrelated_message_clears_offer_and_answers_new_question():
    """The exact Phase 6f constraint 2 requirement: a next message that
    neither clearly accepts nor declines a pending report-update offer
    must still clear pending_report_update -- not leave it lingering to
    be misread as a yes/no on some later, unrelated turn. Mirrors Phase
    5c's own web-offer "other" test exactly."""
    papers = [_paper("p1", "RoCoFT"), _paper("p2", "LoRA")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1", "p2"], selected_papers=papers, stage="synthesize",
        report=_report_stub(), report_covered_web_article_count=0,
        pending_report_update={"new_article_count": 1},
    )
    schema = _build_answer_schema(["p1", "p2"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_intent_response("other"),
        _mock_parse_response(schema, answerable=True, answer="LoRA injects low-rank matrices [Paper 2].", cited_paper_ids=["p2"]),
    ]

    with patch("research_agent.curation_chat.regenerate_report_with_new_sources") as mock_regen, \
         patch("research_agent.curation_chat.search_web") as mock_search, \
         patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[1], 0.9)]):
        result = chat_turn(session, "actually, what does LoRA inject into each layer?", client=mock_client)

    mock_regen.assert_not_called()
    mock_search.assert_not_called()
    assert session.pending_report_update is None
    assert result["answerable"] is True
    assert [p.paper_id for p in result["cited_papers"]] == ["p2"]
    assert "report_update_offer_made" not in result
    assert "report_update_declined" not in result


# --- curation-chat-metadata Phase 1: persisted per-exchange metadata ---

def test_chat_turn_attaches_a_shared_exchange_id_to_both_entries_of_a_plain_answer():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize")
    schema = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Fine [Paper 1].", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        chat_turn(session, "what is RoCoFT?", client=mock_client)

    assert len(session.chat_history) == 2
    user_turn, assistant_turn = session.chat_history
    assert user_turn["exchange_id"] is not None
    assert isinstance(user_turn["exchange_id"], str) and user_turn["exchange_id"] != ""
    assert user_turn["exchange_id"] == assistant_turn["exchange_id"]


def test_chat_turn_assistant_entry_marks_used_web_search_and_cites_when_web_sources_were_actually_cited():
    """retrieved_web_articles is just session.web_articles (qa.py's own
    _retrieve_node) -- pre-populating web_articles_added is enough to make
    a web citation reachable without going through the offer/accept flow."""
    papers = [_paper("p1", "RoCoFT")]
    web_article = WebArticle(
        title="2026 roundup", url="https://x.com/roundup", snippet="s",
        published_date=None, source_domain="x.com",
    )
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        web_articles_added=[web_article],
    )
    schema = _build_answer_schema(["p1"], ["https://x.com/roundup"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Per [Web 1], ...", cited_paper_ids=[], cited_web_urls=["https://x.com/roundup"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[]):
        chat_turn(session, "what's new?", client=mock_client)

    assistant_turn = session.chat_history[-1]
    assert assistant_turn["used_web_search"] is True
    assert assistant_turn["cited_web_articles"] == [{"url": "https://x.com/roundup", "title": "2026 roundup"}]
    assert assistant_turn["added_to_report"] is False  # Phase 1: never set True by any code path yet


def test_chat_turn_paper_only_answer_has_used_web_search_false_and_no_cited_web_articles():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize")
    schema = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Fine [Paper 1].", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        chat_turn(session, "what is RoCoFT?", client=mock_client)

    assistant_turn = session.chat_history[-1]
    assert assistant_turn["used_web_search"] is False
    assert assistant_turn["cited_web_articles"] == []
    assert assistant_turn["added_to_report"] is False


def test_second_chat_turn_after_enriched_first_turn_sends_only_role_and_content_to_openai():
    """The actual sanitization regression test: after a first turn leaves
    an enriched (exchange_id/used_web_search/cited_web_articles/
    added_to_report) assistant entry in session.chat_history, a SECOND
    turn must still work (proving capped_history's new stripping doesn't
    break the multi-turn path at all) -- and, critically, every message
    dict actually sent to the model on that second turn must contain
    ONLY {role, content}, proving the enriched history never reaches the
    OpenAI call."""
    papers = [_paper("p1", "RoCoFT")]
    web_article = WebArticle(
        title="2026 roundup", url="https://x.com/roundup", snippet="s",
        published_date=None, source_domain="x.com",
    )
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        web_articles_added=[web_article],
    )
    schema_with_web = _build_answer_schema(["p1"], ["https://x.com/roundup"])
    schema_paper_only = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_parse_response(
            schema_with_web, answerable=True, answer="Per [Web 1], ...",
            cited_paper_ids=[], cited_web_urls=["https://x.com/roundup"],
        ),
        _mock_parse_response(
            schema_paper_only, answerable=True, answer="Rows and columns [Paper 1].", cited_paper_ids=["p1"],
        ),
    ]
    # condense_question's own call -- exercised on the second turn only,
    # since the first turn has no prior history to condense against.
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="what does RoCoFT update?"))],
        usage=MagicMock(total_tokens=50, prompt_tokens=40, completion_tokens=10),
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        chat_turn(session, "what's new?", client=mock_client)
        # First turn's assistant entry is now enriched (used_web_search=True,
        # cited_web_articles populated) -- confirm before the second call.
        assert session.chat_history[-1]["used_web_search"] is True

        chat_turn(session, "what does it update?", client=mock_client)

    assert len(session.chat_history) == 4

    # The SECOND generate_answer call's messages -- must be clean.
    second_generate_messages = mock_client.chat.completions.parse.call_args_list[-1].kwargs["messages"]
    for message in second_generate_messages:
        assert set(message.keys()) == {"role", "content"}, f"unexpected keys in {message!r}"

    # condense_question's own call also only ever received role/content
    # (built from a transcript string, not the raw dicts -- confirmed
    # here too rather than assumed).
    condense_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    for message in condense_messages:
        assert set(message.keys()) == {"role", "content"}

    # The first turn's persisted history STILL has its full metadata --
    # sanitization only strips the LLM-bound copy, never the original.
    assert session.chat_history[1]["used_web_search"] is True
    assert session.chat_history[1]["cited_web_articles"] == [{"url": "https://x.com/roundup", "title": "2026 roundup"}]
