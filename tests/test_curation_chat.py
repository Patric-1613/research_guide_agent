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

from research_agent.curation_chat import (
    _OfferResponseIntent,
    _classify_offer_response,
    approve_web_article_urls,
    approved_web_article_urls_from_added_to_report_entries,
    ask_in_session,
    chat_turn,
    cited_web_article_urls_for_exchanges,
    delete_chat_exchanges,
    edit_chat_exchange,
    live_cited_web_article_urls,
    mark_exchanges_added_to_report,
    prune_report_approved_web_article_urls,
    resolve_approved_web_articles_for_regeneration,
    select_eligible_exchanges_for_report,
)
from research_agent.qa import _build_answer_schema
from research_agent.query_expansion import PaperPoolSession
from research_agent.report import GENERATION_REASON_INITIAL, append_report_version
from research_agent.schema import Paper, WebArticle


def _paper(paper_id: str, title: str) -> Paper:
    return Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract=f"Abstract for {title}.", url=f"http://arxiv.org/abs/{paper_id}",
        doi=None, citation_count=None, source="arxiv", paper_id=paper_id,
    )


def _web_article(url: str, title: str) -> WebArticle:
    return WebArticle(title=title, url=url, snippet="s", published_date=None, source_domain="example.com")


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
    # report-quality Phase R3: the easiest of the four report-mutation
    # call sites to miss, since it lives here rather than in either
    # report service module -- must append a real version, not just
    # reassign session.report directly.
    assert len(session.report_versions) == 1
    assert session.report_versions[0]["generation_reason"] == "chat_auto_update"
    assert session.report_versions[0]["report"] is updated_report
    assert session.active_report_version_id == session.report_versions[0]["version_id"]
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


def test_chat_turn_report_update_accept_appends_version_2_when_an_initial_version_already_exists():
    """The realistic case: session.report_versions already has a real
    version 1 (appended via append_report_version, the same way every
    production code path builds it) -- the auto-update accept must
    append version 2, not silently restart numbering at 1, and must
    leave version 1 completely untouched."""
    papers = [_paper("p1", "RoCoFT")]
    new_article = WebArticle(title="2026 roundup", url="https://x.com/roundup", snippet="s", published_date=None, source_domain="x.com")
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        report_covered_web_article_count=0, web_articles_added=[new_article],
        pending_report_update={"new_article_count": 1},
    )
    v1 = append_report_version(session, _report_stub(), GENERATION_REASON_INITIAL)
    updated_report = _report_stub(cited_papers=[papers[0]])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_intent_response("accept")

    with patch("research_agent.curation_chat.regenerate_report_with_new_sources", return_value=updated_report):
        chat_turn(session, "yes, please update it", client=mock_client)

    assert len(session.report_versions) == 2
    assert session.report_versions[0] is v1
    assert session.report_versions[0]["report"] is v1["report"]  # untouched
    assert session.report_versions[1]["version_number"] == 2
    assert session.report_versions[1]["generation_reason"] == "chat_auto_update"
    assert session.active_report_version_id == session.report_versions[1]["version_id"]


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


def test_chat_turn_assistant_entry_attaches_cited_papers_from_the_qa_ask_result():
    """report-quality Phase R3.2 Chunk 1: the raw material (qa.ask()'s
    own result["cited_papers"], real Paper objects) already existed at
    this exact point -- it was just discarded before this fix. Stored as
    lightweight {paper_id, title} dicts, NOT full Paper objects."""
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
    assert assistant_turn["cited_papers"] == [{"paper_id": "p1", "title": "RoCoFT"}]


def test_chat_turn_unanswerable_message_has_no_cited_papers():
    """Mirrors _generate_node's own defensive "empty if not answerable"
    enforcement -- an unanswerable turn's cited_papers must be [], not
    whatever the model may have hallucinated onto an unused field."""
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize")
    schema = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=False, answer="I can't answer that from the selected papers.", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        chat_turn(session, "something unrelated?", client=mock_client)

    assistant_turn = session.chat_history[-1]
    assert assistant_turn["cited_papers"] == []


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


# --- curation-chat-web-relevance: stale web_articles_added no longer
# leaks into a later, unrelated turn ---

def test_chat_turn_used_web_search_false_when_stale_web_pool_exists_but_is_filtered_out():
    """The direct regression test for the reported bug: a session with a
    NON-EMPTY web_articles_added (from an earlier, unrelated accepted
    search) must not tag a later, unrelated answer as used_web_search=True
    just because the stale pool exists -- qa.py's filter_web_relevance
    node (patched here at its own seam, not the embedding math) is what
    keeps it out of this turn's cited_web_articles."""
    papers = [_paper("p1", "AI Risk Tiering")]
    stale = WebArticle(title="Stale roundup", url="https://stale.com", snippet="s", published_date=None, source_domain="x.com")
    session = PaperPoolSession(
        topic="ai risk", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        web_articles_added=[stale],
    )
    schema = _build_answer_schema(["p1"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Per [Paper 1], ...", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]), \
         patch("research_agent.qa._filter_relevant_web_articles", return_value=[]):
        chat_turn(session, "is jailbreaking covered?", client=mock_client)

    assistant_turn = session.chat_history[-1]
    assert assistant_turn["used_web_search"] is False
    assert assistant_turn["cited_web_articles"] == []


def test_chat_turn_web_search_offer_retriggers_when_stale_web_pool_filtered_out_and_papers_insufficient():
    """The other direct regression test: once the stale web pool no
    longer props up a false answerable=True, a genuinely unanswerable
    follow-up correctly re-arms the web-search offer -- proving the
    offer-and-decide mechanism (unchanged in this phase) starts working
    again as a side effect of cleaner input, not because
    _maybe_set_web_offer itself was touched."""
    papers = [_paper("p1", "AI Risk Tiering")]
    stale = WebArticle(title="Stale roundup", url="https://stale.com", snippet="s", published_date=None, source_domain="x.com")
    session = PaperPoolSession(
        topic="ai risk", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        web_articles_added=[stale],
    )
    schema = _build_answer_schema(["p1"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=False, answer="Not covered by the retrieved papers.", cited_paper_ids=[],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]), \
         patch("research_agent.qa._filter_relevant_web_articles", return_value=[]):
        result = chat_turn(session, "is jailbreaking covered?", client=mock_client)

    assert session.pending_web_offer == {"question": "is jailbreaking covered?"}
    assert result["web_offer_made"] is True
    assert result["answer"].endswith("Would you like me to search the web for more on this?")


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


# --- curation-chat-delete Phase 3: delete_chat_exchanges() ---

def _exchange(exchange_id: str, added_to_report: bool = False, used_web_search: bool = False) -> list[dict]:
    return [
        {"role": "user", "content": f"question for {exchange_id}", "exchange_id": exchange_id},
        {
            "role": "assistant", "content": f"answer for {exchange_id}", "exchange_id": exchange_id,
            "used_web_search": used_web_search,
            "cited_web_articles": [{"url": "https://x.com", "title": "X"}] if used_web_search else [],
            "added_to_report": added_to_report,
        },
    ]


def test_delete_chat_exchanges_removes_both_entries_of_a_matching_exchange():
    session = PaperPoolSession(topic="peft", stage="synthesize", chat_history=[*_exchange("ex-1"), *_exchange("ex-2")])

    deleted_ids, report_possibly_stale = delete_chat_exchanges(session, ["ex-1"])

    assert deleted_ids == ["ex-1"]
    assert report_possibly_stale is False
    assert len(session.chat_history) == 2
    assert all(turn["exchange_id"] == "ex-2" for turn in session.chat_history)


def test_delete_chat_exchanges_leaves_pre_phase_1_entries_with_no_exchange_id_untouched():
    old_entries = [{"role": "user", "content": "old question"}, {"role": "assistant", "content": "old answer"}]
    session = PaperPoolSession(topic="peft", stage="synthesize", chat_history=[*old_entries, *_exchange("ex-1")])

    deleted_ids, _ = delete_chat_exchanges(session, ["ex-1"])

    assert deleted_ids == ["ex-1"]
    assert session.chat_history == old_entries  # old pair survives, byte-for-byte, in original order


def test_delete_chat_exchanges_can_delete_multiple_exchanges_at_once():
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[*_exchange("ex-1"), *_exchange("ex-2"), *_exchange("ex-3")],
    )

    deleted_ids, _ = delete_chat_exchanges(session, ["ex-1", "ex-3"])

    assert deleted_ids == ["ex-1", "ex-3"]
    assert len(session.chat_history) == 2
    assert all(turn["exchange_id"] == "ex-2" for turn in session.chat_history)


def test_delete_chat_exchanges_unknown_id_is_an_idempotent_no_op():
    session = PaperPoolSession(topic="peft", stage="synthesize", chat_history=[*_exchange("ex-1")])

    deleted_ids, report_possibly_stale = delete_chat_exchanges(session, ["does-not-exist"])

    assert deleted_ids == []
    assert report_possibly_stale is False
    assert len(session.chat_history) == 2  # unchanged


def test_delete_chat_exchanges_empty_request_is_a_no_op():
    session = PaperPoolSession(topic="peft", stage="synthesize", chat_history=[*_exchange("ex-1")])

    deleted_ids, report_possibly_stale = delete_chat_exchanges(session, [])

    assert deleted_ids == []
    assert report_possibly_stale is False
    assert len(session.chat_history) == 2


def test_delete_chat_exchanges_reports_possibly_stale_when_a_deleted_answer_was_added_to_report():
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[*_exchange("ex-1", added_to_report=True), *_exchange("ex-2", added_to_report=False)],
    )

    _, report_possibly_stale = delete_chat_exchanges(session, ["ex-1"])
    assert report_possibly_stale is True

    # Deleting the NOT-added-to-report one alone must not report stale.
    session2 = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[*_exchange("ex-1", added_to_report=True), *_exchange("ex-2", added_to_report=False)],
    )
    _, report_possibly_stale2 = delete_chat_exchanges(session2, ["ex-2"])
    assert report_possibly_stale2 is False


# --- chat-ux-report-semantics Phase B: approved-URL pruning on delete ---

def test_delete_chat_exchanges_prunes_approved_url_when_no_remaining_exchange_cites_it():
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[*_web_exchange("ex-1", "https://a.com", added_to_report=True)],
        report_approved_web_article_urls={"https://a.com"},
    )

    _, report_possibly_stale = delete_chat_exchanges(session, ["ex-1"])

    assert report_possibly_stale is True
    assert session.report_approved_web_article_urls == set()


def test_delete_chat_exchanges_keeps_approved_url_still_cited_by_another_remaining_added_exchange():
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[
            *_web_exchange("ex-1", "https://a.com", added_to_report=True),
            *_web_exchange("ex-2", "https://a.com", added_to_report=True),  # same URL, different exchange
        ],
        report_approved_web_article_urls={"https://a.com"},
    )

    _, report_possibly_stale = delete_chat_exchanges(session, ["ex-1"])

    assert report_possibly_stale is True  # ex-1 itself was added_to_report
    # ex-2 is still there, still added_to_report, still citing the URL.
    assert session.report_approved_web_article_urls == {"https://a.com"}


def test_delete_chat_exchanges_of_non_report_added_exchange_leaves_approved_set_unchanged():
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[
            *_web_exchange("ex-1", "https://a.com", added_to_report=True),
            *_web_exchange("ex-2", "https://b.com", added_to_report=False),
        ],
        report_approved_web_article_urls={"https://a.com"},
    )

    _, report_possibly_stale = delete_chat_exchanges(session, ["ex-2"])

    assert report_possibly_stale is False
    assert session.report_approved_web_article_urls == {"https://a.com"}


def test_delete_chat_exchanges_does_not_touch_the_raw_web_articles_added_pool():
    approved_article = _web_article("https://a.com", "A")
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[*_web_exchange("ex-1", "https://a.com", added_to_report=True)],
        report_approved_web_article_urls={"https://a.com"},
        web_articles_added=[approved_article],
    )

    delete_chat_exchanges(session, ["ex-1"])

    assert session.web_articles_added == [approved_article]  # raw pool, unaffected by pruning


def test_delete_chat_exchanges_pruned_url_is_excluded_from_a_future_selective_regeneration():
    approved_article = _web_article("https://a.com", "A")
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[*_web_exchange("ex-1", "https://a.com", added_to_report=True)],
        report_approved_web_article_urls={"https://a.com"},
        web_articles_added=[approved_article],
    )

    delete_chat_exchanges(session, ["ex-1"])

    resolved = resolve_approved_web_articles_for_regeneration(session, set())
    assert resolved == []  # the revoked URL no longer resolves to anything


# --- report-quality Phase R2D: persistent web-citation revocation tracking ---

def test_live_cited_web_article_urls_includes_entries_never_added_to_report():
    """live_cited_web_article_urls is deliberately BROADER than
    approved_web_article_urls_from_added_to_report_entries -- it counts
    every assistant entry's cited_web_articles regardless of
    added_to_report, since a source never formally added to the report
    can still be revoked by chat deleting its only backing exchange."""
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[*_web_exchange("ex-1", "https://a.com", added_to_report=False)],
    )
    assert live_cited_web_article_urls(session) == {"https://a.com"}


def test_delete_chat_exchanges_marks_revoked_when_no_remaining_exchange_backs_it():
    """The general case: a web source's only backing exchange is
    deleted, regardless of whether it was ever added_to_report -- must
    land in revoked_web_article_urls so a later whole-pool regeneration
    (report.py's regenerate_report_with_new_sources) permanently
    excludes it, even though web_articles_added itself is never pruned."""
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[*_web_exchange("ex-1", "https://a.com", added_to_report=False)],
    )

    delete_chat_exchanges(session, ["ex-1"])

    assert session.revoked_web_article_urls == {"https://a.com"}


def test_delete_chat_exchanges_does_not_revoke_a_url_still_backed_by_another_exchange():
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[
            *_web_exchange("ex-1", "https://a.com"),
            *_web_exchange("ex-2", "https://a.com"),  # same URL, different exchange
        ],
    )

    delete_chat_exchanges(session, ["ex-1"])

    assert session.revoked_web_article_urls == set()


def test_delete_chat_exchanges_un_revokes_a_url_that_becomes_live_again():
    """A URL previously revoked by an earlier delete, then re-cited by a
    later exchange (before this delete call), must not stay marked
    revoked just because THIS delete also happens to touch chat_history
    -- _sync_revoked_web_article_urls's own "subtract what's live now"
    half."""
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[
            *_web_exchange("ex-1", "https://a.com"),
            *_web_exchange("ex-2", "https://b.com"),
        ],
        revoked_web_article_urls={"https://a.com"},  # stale revocation from some earlier operation
    )

    delete_chat_exchanges(session, ["ex-2"])

    # https://a.com is still live via ex-1 (untouched by this delete) --
    # un-revoked even though this call never re-cited it itself. https://
    # b.com is newly dead (its only exchange, ex-2, was just removed) --
    # correctly becomes revoked by this same call.
    assert session.revoked_web_article_urls == {"https://b.com"}


def test_edit_chat_exchange_marks_revoked_when_truncation_removes_the_only_backing_exchange():
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[
            {"role": "user", "content": "q1", "exchange_id": "ex-1"},
            {
                "role": "assistant", "content": "a1", "exchange_id": "ex-1",
                "used_web_search": True, "cited_web_articles": [{"url": "https://a.com", "title": "A"}],
                "added_to_report": False,
            },
        ],
    )
    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_parse_response(_build_answer_schema([]), answerable=False, answer="No sources for this.", cited_paper_ids=[]),
    ]

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[]), \
         patch("research_agent.curation_chat.search_web", return_value=[]):
        edit_chat_exchange(session, "ex-1", "a different question", client=mock_client)

    assert session.revoked_web_article_urls == {"https://a.com"}


def test_deleted_exchange_content_is_excluded_from_the_next_chat_turns_openai_messages():
    """curation-chat-delete Phase 3, requirement 4: the deleted exchange's
    content must not survive into a LATER turn's context -- confirmed by
    inspecting the actual messages sent to the mocked OpenAI client for
    the follow-up turn, not just that chat_history looks right."""
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize")
    schema = _build_answer_schema(["p1"])

    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_parse_response(schema, answerable=True, answer="RoCoFT is unusual [Paper 1].", cited_paper_ids=["p1"]),
        _mock_parse_response(schema, answerable=True, answer="It updates rows and columns [Paper 1].", cited_paper_ids=["p1"]),
    ]
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="what does it update?"))],
        usage=MagicMock(total_tokens=50, prompt_tokens=40, completion_tokens=10),
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        chat_turn(session, "is RoCoFT unusual in some specific way?", client=mock_client)

        exchange_id = session.chat_history[0]["exchange_id"]
        deleted_ids, _ = delete_chat_exchanges(session, [exchange_id])
        assert deleted_ids == [exchange_id]
        assert session.chat_history == []  # the only exchange so far, now gone

        chat_turn(session, "what does it update?", client=mock_client)

    second_generate_messages = mock_client.chat.completions.parse.call_args_list[-1].kwargs["messages"]
    joined = " ".join(m["content"] for m in second_generate_messages)
    assert "unusual in some specific way" not in joined
    assert "RoCoFT is unusual" not in joined


# --- curation-chat-add-to-report Phase 4: domain helpers ---

def _web_exchange(exchange_id: str, url: str, title: str = "Article", added_to_report: bool = False) -> list[dict]:
    return [
        {"role": "user", "content": f"question for {exchange_id}", "exchange_id": exchange_id},
        {
            "role": "assistant", "content": f"answer for {exchange_id}", "exchange_id": exchange_id,
            "used_web_search": True,
            "cited_web_articles": [{"url": url, "title": title}],
            "added_to_report": added_to_report,
        },
    ]


def test_select_eligible_exchanges_for_report_partitions_eligible_and_skipped():
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[
            *_web_exchange("ex-web", "https://a.com"),
            *_exchange("ex-paper-only"),  # used_web_search=False by _exchange's own default
            *_web_exchange("ex-already-added", "https://b.com", added_to_report=True),
        ],
    )

    eligible, skipped = select_eligible_exchanges_for_report(
        session, ["ex-web", "ex-paper-only", "ex-already-added", "does-not-exist"],
    )

    assert eligible == ["ex-web"]
    assert skipped == ["ex-paper-only", "ex-already-added", "does-not-exist"]


def test_select_eligible_exchanges_for_report_ignores_old_entries_with_no_exchange_id():
    old_entries = [{"role": "user", "content": "old q"}, {"role": "assistant", "content": "old a"}]
    session = PaperPoolSession(topic="peft", stage="synthesize", chat_history=old_entries)

    eligible, skipped = select_eligible_exchanges_for_report(session, [""])
    assert eligible == []
    assert skipped == []  # an empty/falsy id is dropped outright, not even counted as skipped


def test_select_eligible_exchanges_for_report_dedupes_requested_ids():
    session = PaperPoolSession(topic="peft", stage="synthesize", chat_history=_web_exchange("ex-1", "https://a.com"))

    eligible, skipped = select_eligible_exchanges_for_report(session, ["ex-1", "ex-1", "ex-1"])

    assert eligible == ["ex-1"]
    assert skipped == []


def test_cited_web_article_urls_for_exchanges_unions_and_dedupes():
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[
            *_web_exchange("ex-1", "https://a.com"),
            *_web_exchange("ex-2", "https://a.com"),  # same URL as ex-1
            *_web_exchange("ex-3", "https://b.com"),
        ],
    )

    urls = cited_web_article_urls_for_exchanges(session, ["ex-1", "ex-2", "ex-3"])

    assert urls == {"https://a.com", "https://b.com"}


def test_resolve_approved_web_articles_for_regeneration_excludes_unapproved_pool_articles():
    """The core Phase 4 correctness guarantee: an article sitting in the
    raw web_articles_added pool that was never approved must NOT appear
    in the resolved list, even though it's fully valid, real session
    data."""
    approved_article = _web_article("https://approved.com", "Approved")
    unapproved_article = _web_article("https://unapproved.com", "Unapproved")
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        web_articles_added=[approved_article, unapproved_article],
    )

    resolved = resolve_approved_web_articles_for_regeneration(session, {"https://approved.com"})

    assert resolved == [approved_article]
    assert unapproved_article not in resolved


def test_resolve_approved_web_articles_for_regeneration_includes_previously_approved_plus_newly_approved():
    article_1 = _web_article("https://one.com", "One")
    article_2 = _web_article("https://two.com", "Two")
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        web_articles_added=[article_1, article_2],
        report_approved_web_article_urls={"https://one.com"},
    )

    resolved = resolve_approved_web_articles_for_regeneration(session, {"https://two.com"})

    assert {a.url for a in resolved} == {"https://one.com", "https://two.com"}


def test_approve_web_article_urls_unions_into_the_existing_approved_set():
    session = PaperPoolSession(topic="peft", stage="synthesize", report_approved_web_article_urls={"https://one.com"})

    approve_web_article_urls(session, {"https://two.com"})

    assert session.report_approved_web_article_urls == {"https://one.com", "https://two.com"}


def test_mark_exchanges_added_to_report_flips_only_the_targeted_assistant_entries():
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[*_web_exchange("ex-1", "https://a.com"), *_web_exchange("ex-2", "https://b.com")],
    )

    mark_exchanges_added_to_report(session, ["ex-1"])

    by_id = {t["exchange_id"]: t for t in session.chat_history if t["role"] == "assistant"}
    assert by_id["ex-1"]["added_to_report"] is True
    assert by_id["ex-2"]["added_to_report"] is False


# --- chat-ux-report-semantics Phase B: the shared invariant helper ---

def test_approved_web_article_urls_from_added_to_report_entries_only_counts_added_ones():
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[
            *_web_exchange("ex-1", "https://a.com", added_to_report=True),
            *_web_exchange("ex-2", "https://b.com", added_to_report=False),
        ],
    )

    assert approved_web_article_urls_from_added_to_report_entries(session) == {"https://a.com"}


def test_prune_report_approved_web_article_urls_recomputes_from_scratch_not_intersect():
    # Approved set has a URL nothing in chat_history cites at all (e.g.
    # stale from a code path outside this test's scope) -- recompute
    # replaces it wholesale rather than merely narrowing it.
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[*_web_exchange("ex-1", "https://a.com", added_to_report=True)],
        report_approved_web_article_urls={"https://a.com", "https://orphaned.com"},
    )

    prune_report_approved_web_article_urls(session)

    assert session.report_approved_web_article_urls == {"https://a.com"}


# --- curation-chat-edit Phase 5: edit_chat_exchange() ---

def test_edit_chat_exchange_truncates_target_and_later_exchanges_then_appends_a_fresh_exchange():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        chat_history=[*_exchange("ex-0"), *_exchange("ex-1"), *_exchange("ex-2")],
    )
    schema = _build_answer_schema(["p1"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Fresh answer [Paper 1].", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        edit_chat_exchange(session, "ex-1", "an edited question", client=mock_client)

    # ex-0 (before the edited exchange) survives untouched; ex-1's old
    # pair and ex-2 (after it) are both gone, replaced by exactly one
    # fresh exchange.
    assert len(session.chat_history) == 4  # ex-0's pair + the fresh pair
    assert session.chat_history[0]["exchange_id"] == "ex-0"
    assert session.chat_history[1]["exchange_id"] == "ex-0"
    assert session.chat_history[2]["role"] == "user"
    assert session.chat_history[2]["content"] == "an edited question"
    assert session.chat_history[3]["role"] == "assistant"
    assert session.chat_history[3]["content"] == "Fresh answer [Paper 1]."
    assert not any(t.get("exchange_id") == "ex-1" for t in session.chat_history)
    assert not any(t.get("exchange_id") == "ex-2" for t in session.chat_history)


def test_edit_chat_exchange_fresh_exchange_gets_a_new_exchange_id():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        chat_history=[*_exchange("ex-1")],
    )
    schema = _build_answer_schema(["p1"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Fresh answer.", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        edit_chat_exchange(session, "ex-1", "edited question", client=mock_client)

    new_exchange_id = session.chat_history[0]["exchange_id"]
    assert new_exchange_id is not None
    assert new_exchange_id != "ex-1"
    assert session.chat_history[1]["exchange_id"] == new_exchange_id


def test_edit_chat_exchange_future_context_excludes_truncated_content():
    """Mirrors the analogous Phase 3 delete regression test: content from
    the edited-away exchange and everything after it must not survive
    into a LATER turn's context."""
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        chat_history=[
            {"role": "user", "content": "the original question that will be edited away", "exchange_id": "ex-1"},
            {
                "role": "assistant", "content": "the original answer that will be edited away", "exchange_id": "ex-1",
                "used_web_search": False, "cited_web_articles": [], "added_to_report": False,
            },
        ],
    )
    schema = _build_answer_schema(["p1"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = [
        _mock_parse_response(schema, answerable=True, answer="Fresh answer.", cited_paper_ids=["p1"]),
        _mock_parse_response(schema, answerable=True, answer="Follow-up answer.", cited_paper_ids=["p1"]),
    ]
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="a follow-up question"))],
        usage=MagicMock(total_tokens=50, prompt_tokens=40, completion_tokens=10),
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        edit_chat_exchange(session, "ex-1", "the edited question", client=mock_client)
        chat_turn(session, "a follow-up question", client=mock_client)

    second_generate_messages = mock_client.chat.completions.parse.call_args_list[-1].kwargs["messages"]
    joined = " ".join(m["content"] for m in second_generate_messages)
    assert "will be edited away" not in joined


def test_edit_chat_exchange_report_possibly_stale_true_when_the_edited_exchange_itself_was_added_to_report():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        chat_history=[*_web_exchange("ex-1", "https://a.com", added_to_report=True)],
    )
    schema = _build_answer_schema(["p1"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Fresh answer.", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        _, report_possibly_stale = edit_chat_exchange(session, "ex-1", "edited question", client=mock_client)

    assert report_possibly_stale is True


def test_edit_chat_exchange_report_possibly_stale_true_when_a_later_truncated_exchange_was_added_to_report():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        chat_history=[*_exchange("ex-1"), *_web_exchange("ex-2", "https://a.com", added_to_report=True)],
    )
    schema = _build_answer_schema(["p1"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Fresh answer.", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        _, report_possibly_stale = edit_chat_exchange(session, "ex-1", "edited question", client=mock_client)

    assert report_possibly_stale is True


def test_edit_chat_exchange_report_possibly_stale_false_when_nothing_truncated_was_added_to_report():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        chat_history=[*_web_exchange("ex-1", "https://a.com", added_to_report=False)],
    )
    schema = _build_answer_schema(["p1"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Fresh answer.", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        _, report_possibly_stale = edit_chat_exchange(session, "ex-1", "edited question", client=mock_client)

    assert report_possibly_stale is False


def test_edit_chat_exchange_prunes_approved_url_only_cited_by_the_truncated_exchange():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        chat_history=[*_web_exchange("ex-1", "https://a.com", added_to_report=True)],
        report_approved_web_article_urls={"https://a.com"},
    )
    schema = _build_answer_schema(["p1"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Fresh answer.", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        _, report_possibly_stale = edit_chat_exchange(session, "ex-1", "edited question", client=mock_client)

    # chat-ux-report-semantics Phase B: superseded the old "Option A, never
    # pruned" behavior -- editing away the only added-to-report exchange
    # that cited this URL revokes it, so a future selective regeneration
    # can't resurrect it.
    assert report_possibly_stale is True
    assert session.report_approved_web_article_urls == set()


def test_edit_chat_exchange_keeps_approved_url_cited_by_a_remaining_added_exchange_before_the_edit_point():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        chat_history=[
            *_web_exchange("ex-0", "https://a.com", added_to_report=True),  # before the edit point, untouched
            *_web_exchange("ex-1", "https://a.com", added_to_report=True),  # will be truncated away
        ],
        report_approved_web_article_urls={"https://a.com"},
    )
    schema = _build_answer_schema(["p1"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Fresh answer.", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        _, report_possibly_stale = edit_chat_exchange(session, "ex-1", "edited question", client=mock_client)

    assert report_possibly_stale is True  # ex-1 itself was added_to_report
    # ex-0 survives the truncation, still added_to_report, still citing it.
    assert session.report_approved_web_article_urls == {"https://a.com"}


def test_edit_chat_exchange_of_non_report_added_exchange_leaves_approved_set_unchanged():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        chat_history=[
            *_web_exchange("ex-0", "https://a.com", added_to_report=True),
            *_exchange("ex-1"),  # paper-only, not added_to_report -- the one being edited
        ],
        report_approved_web_article_urls={"https://a.com"},
    )
    schema = _build_answer_schema(["p1"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Fresh answer.", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        _, report_possibly_stale = edit_chat_exchange(session, "ex-1", "edited question", client=mock_client)

    assert report_possibly_stale is False
    assert session.report_approved_web_article_urls == {"https://a.com"}


def test_edit_chat_exchange_does_not_touch_the_raw_web_articles_added_pool():
    papers = [_paper("p1", "RoCoFT")]
    approved_article = _web_article("https://a.com", "A")
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        chat_history=[*_web_exchange("ex-1", "https://a.com", added_to_report=True)],
        report_approved_web_article_urls={"https://a.com"},
        web_articles_added=[approved_article],
    )
    schema = _build_answer_schema(["p1"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Fresh answer.", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        edit_chat_exchange(session, "ex-1", "edited question", client=mock_client)

    assert session.web_articles_added == [approved_article]  # raw pool, unaffected by pruning


def test_edit_chat_exchange_clears_pending_offer_state_even_though_it_was_set():
    papers = [_paper("p1", "RoCoFT")]
    session = PaperPoolSession(
        topic="peft", selected_paper_ids=["p1"], selected_papers=papers, stage="synthesize",
        chat_history=[*_exchange("ex-1")],
        pending_web_offer={"question": "some stale offer"},
        pending_report_update={"new_article_count": 1},
    )
    schema = _build_answer_schema(["p1"])
    mock_client = MagicMock()
    mock_client.chat.completions.parse.return_value = _mock_parse_response(
        schema, answerable=True, answer="Fresh answer.", cited_paper_ids=["p1"],
    )

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
         patch("research_agent.qa.embed_and_index_papers"), \
         patch("research_agent.qa.get_chroma_collection"), \
         patch("research_agent.qa.semantic_search", return_value=[(papers[0], 0.9)]):
        edit_chat_exchange(session, "ex-1", "edited question", client=mock_client)

    assert session.pending_web_offer is None
    assert session.pending_report_update is None


def test_edit_chat_exchange_raises_for_unknown_exchange_id():
    session = PaperPoolSession(topic="peft", stage="synthesize", chat_history=[*_exchange("ex-1")])

    try:
        edit_chat_exchange(session, "does-not-exist", "edited question", client=MagicMock())
        assert False, "expected a ValueError for an unknown exchange_id"
    except ValueError as e:
        assert "does-not-exist" in str(e)


def test_edit_chat_exchange_raises_when_exchange_id_only_matches_an_assistant_entry():
    """Structurally shouldn't happen given the one-user-one-assistant
    invariant, but confirmed defensively: searching only role=="user"
    means an id that somehow only resolves to an assistant entry is
    correctly reported as not editable, not silently mismatched."""
    session = PaperPoolSession(
        topic="peft", stage="synthesize",
        chat_history=[{"role": "assistant", "content": "orphan answer", "exchange_id": "ex-orphan"}],
    )

    try:
        edit_chat_exchange(session, "ex-orphan", "edited question", client=MagicMock())
        assert False, "expected a ValueError -- no user entry carries this exchange_id"
    except ValueError:
        pass


def test_edit_chat_exchange_raises_for_pre_phase_1_entries_with_no_exchange_id():
    old_entries = [{"role": "user", "content": "old question"}, {"role": "assistant", "content": "old answer"}]
    session = PaperPoolSession(topic="peft", stage="synthesize", chat_history=old_entries)

    try:
        edit_chat_exchange(session, "", "edited question", client=MagicMock())
        assert False, "expected a ValueError -- no entry has a real (non-None) exchange_id to match"
    except ValueError:
        pass
    assert session.chat_history == old_entries  # untouched
