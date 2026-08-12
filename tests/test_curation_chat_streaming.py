"""Usage Protection M4.2A Part D/F: tests for
research_agent/curation_chat_streaming.py -- the production
curation-chat streaming domain service. No admission/lease/HTTP
concerns here (that's tests/test_curation_chat_stream_api.py, Part E)
-- this file drives `stream_curation_chat_turn` directly, mocking
every provider-facing seam (`stream_chat_answer`, `chat_turn`,
`_classify_offer_response`, `save_curation_session`) so nothing here
makes a real network/provider call or touches a real database.

Same `asyncio.run()`-per-test convention as tests/test_chat_streaming.py
(no pytest-asyncio dependency needed).
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from research_agent.chat_streaming import AnswerCompleted, AnswerDelta, ChatAnswerStreamError, StreamedChatAnswer
from research_agent.curation_chat_streaming import stream_curation_chat_turn
from research_agent.query_expansion import PaperPoolSession
from research_agent.schema import Paper, WebArticle


def _web_article(url: str, title: str = "W") -> WebArticle:
    return WebArticle(title=title, url=url, snippet="a snippet", published_date=None, source_domain="example.com")


def _session(**kwargs) -> PaperPoolSession:
    defaults = dict(topic="AI safety", stage="synthesize")
    defaults.update(kwargs)
    return PaperPoolSession(**defaults)


async def _fake_stream_chat_answer(items):
    for item in items:
        yield item


def _run(coro):
    return asyncio.run(coro)


async def _collect(session, message, **kwargs):
    events = []
    async for event in stream_curation_chat_turn(
        session, message,
        session_id=kwargs.pop("session_id", "sess-1"),
        checkpointer=kwargs.pop("checkpointer", MagicMock()),
        sync_client=kwargs.pop("sync_client", MagicMock()),
        async_client=kwargs.pop("async_client", MagicMock()),
        top_k=kwargs.pop("top_k", 5),
    ):
        events.append((event.type, event.data))
    return events


def _types(events):
    return [t for t, _ in events]


@patch("research_agent.curation_chat_streaming.save_curation_session")
@patch("research_agent.curation_chat_streaming.stream_chat_answer")
@patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0))
def test_plain_fallback_success_event_order_and_persistence(mock_classify, mock_stream, mock_save):
    session = _session(web_articles_added=[_web_article("https://x.com/a", "Article A")])
    completed = AnswerCompleted(result=StreamedChatAnswer(
        answer="Hello world", answerable=True, cited_papers=[], cited_web_articles=[],
    ))
    mock_stream.return_value = _fake_stream_chat_answer(
        [AnswerDelta(text="Hello"), AnswerDelta(text=" world"), completed],
    )

    events = _run(_collect(session, "what does the web say?"))

    assert _types(events) == [
        "started", "phase", "phase", "phase", "delta", "delta", "phase", "completed", "done",
    ]
    phases = [data["phase"] for t, data in events if t == "phase"]
    assert phases == ["preparing_context", "checking_relevance", "generating", "saving"]
    assert [data["text"] for t, data in events if t == "delta"] == ["Hello", " world"]
    completed_payload = next(data for t, data in events if t == "completed")
    assert completed_payload["answer"] == "Hello world"
    mock_save.assert_called_once()
    assert session.chat_history[-2] == {"role": "user", "content": "what does the web say?", "exchange_id": session.chat_history[-2]["exchange_id"]}
    assert session.chat_history[-1]["content"] == "Hello world"


@patch("research_agent.curation_chat_streaming.save_curation_session")
def test_zero_delta_final_only_completion_for_no_sources_early_return(mock_save):
    session = _session()  # no papers, no web articles at all

    events = _run(_collect(session, "anything?"))

    assert _types(events) == ["started", "phase", "completed", "done"]
    assert events[1][1]["phase"] == "saving"
    assert events[2][1]["answerable"] is False


@patch("research_agent.curation_chat_streaming.save_curation_session")
@patch("research_agent.curation_chat_streaming.stream_chat_answer")
@patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0))
def test_handled_stream_error_emits_error_then_done_never_completed(mock_classify, mock_stream, mock_save):
    session = _session(web_articles_added=[_web_article("https://x.com/a", "Article A")])

    async def _raising():
        yield AnswerDelta(text="partial")
        raise ChatAnswerStreamError("provider_error", "The model provider returned an error.")

    mock_stream.return_value = _raising()

    events = _run(_collect(session, "what does the web say?"))

    assert _types(events)[-2:] == ["error", "done"]
    assert "completed" not in _types(events)
    error_payload = next(data for t, data in events if t == "error")
    assert error_payload == {"reason_code": "provider_error", "message": "The model provider returned an error."}
    mock_save.assert_not_called()
    assert session.chat_history == []  # never appended -- no terminal validation happened


@patch("research_agent.curation_chat_streaming.stream_chat_answer")
@patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0))
def test_cancelled_reason_code_reraises_asyncio_cancelled_error(mock_classify, mock_stream):
    session = _session(web_articles_added=[_web_article("https://x.com/a", "Article A")])

    async def _cancelled():
        yield AnswerDelta(text="partial")
        raise ChatAnswerStreamError("cancelled", "The request was cancelled.")

    mock_stream.return_value = _cancelled()

    async def scenario():
        events = []
        async for event in stream_curation_chat_turn(
            session, "q", session_id="s", checkpointer=MagicMock(), sync_client=MagicMock(), async_client=MagicMock(),
        ):
            events.append(event.type)
        return events

    with pytest.raises(asyncio.CancelledError):
        _run(scenario())


@patch("research_agent.curation_chat_streaming.save_curation_session")
@patch("research_agent.curation_chat_streaming.stream_chat_answer")
@patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0))
def test_persistence_failure_emits_error_never_completed(mock_classify, mock_stream, mock_save):
    session = _session(web_articles_added=[_web_article("https://x.com/a", "Article A")])
    completed = AnswerCompleted(result=StreamedChatAnswer(answer="Hi", answerable=True, cited_papers=[], cited_web_articles=[]))
    mock_stream.return_value = _fake_stream_chat_answer([completed])
    mock_save.side_effect = RuntimeError("disk full")

    events = _run(_collect(session, "q"))

    assert _types(events)[-2:] == ["error", "done"]
    assert "completed" not in _types(events)
    error_payload = next(data for t, data in events if t == "error")
    assert error_payload["reason_code"] == "persistence_failed"
    assert "disk full" not in str(error_payload)


@patch("research_agent.curation_chat_streaming.save_curation_session")
@patch("research_agent.curation_chat_streaming.chat_turn")
@patch("research_agent.curation_chat_streaming._classify_offer_response", return_value="accept")
def test_web_offer_accept_uses_sync_fallback_single_delta_no_streaming_adapter(mock_classify, mock_chat_turn, mock_save):
    session = _session(pending_web_offer={"question": "what about limitations?"})
    mock_chat_turn.return_value = {
        "answer": "Full answer via search.", "answerable": True, "cited_papers": [], "cited_web_articles": [],
        "web_search_used": True,
    }

    with patch("research_agent.curation_chat_streaming.stream_chat_answer") as mock_stream:
        events = _run(_collect(session, "yes please"))
        mock_stream.assert_not_called()

    assert _types(events) == ["started", "phase", "phase", "delta", "phase", "completed", "done"]
    phases = [data["phase"] for t, data in events if t == "phase"]
    assert phases == ["searching_web", "generating", "saving"]
    assert [data["text"] for t, data in events if t == "delta"] == ["Full answer via search."]
    mock_chat_turn.assert_called_once()
    mock_save.assert_called_once()


@patch("research_agent.curation_chat_streaming.save_curation_session")
@patch("research_agent.curation_chat_streaming.chat_turn")
@patch("research_agent.curation_chat_streaming._classify_offer_response", return_value="accept")
def test_report_update_offer_accept_has_only_generating_phase(mock_classify, mock_chat_turn, mock_save):
    session = _session(pending_report_update={"new_article_count": 2})
    mock_chat_turn.return_value = {
        "answer": "Report updated.", "answerable": True, "cited_papers": [], "cited_web_articles": [],
        "report_updated": True,
    }

    events = _run(_collect(session, "yes update it"))

    phases = [data["phase"] for t, data in events if t == "phase"]
    assert phases == ["generating", "saving"]


@patch("research_agent.curation_chat_streaming.save_curation_session")
@patch("research_agent.curation_chat_streaming.stream_chat_answer")
@patch("research_agent.curation_chat_streaming._classify_offer_response", return_value="decline")
def test_web_offer_decline_skips_maybe_set_web_offer(mock_classify, mock_stream, mock_save):
    session = _session(
        pending_web_offer={"question": "old question"},
        web_articles_added=[_web_article("https://x.com/a", "Article A")],
    )
    # Unanswerable -- if _maybe_set_web_offer ran, it would append a
    # web-search-offer suffix and re-set pending_web_offer; decline must
    # skip that entirely.
    completed = AnswerCompleted(result=StreamedChatAnswer(answer="Can't answer.", answerable=False, cited_papers=[], cited_web_articles=[]))
    mock_stream.return_value = _fake_stream_chat_answer([completed])

    with patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)):
        events = _run(_collect(session, "no thanks, but what about X?"))

    completed_payload = next(data for t, data in events if t == "completed")
    assert completed_payload["answer"] == "Can't answer."  # no offer suffix appended
    assert session.pending_web_offer is None


@patch("research_agent.curation_chat_streaming._classify_offer_response", side_effect=RuntimeError("boom"))
def test_unexpected_exception_becomes_safe_generic_error_event(mock_classify):
    session = _session(pending_web_offer={"question": "q"})

    events = _run(_collect(session, "hi"))

    assert _types(events) == ["started", "error", "done"]
    error_payload = next(data for t, data in events if t == "error")
    assert error_payload == {"reason_code": "provider_error", "message": "The model provider returned an error."}
    assert "boom" not in str(error_payload)
