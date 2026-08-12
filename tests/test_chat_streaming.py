"""Usage Protection M4.1: tests for research_agent/chat_streaming.py --
the chat-streaming event vocabulary (Part A) and the one-call
structured-answer streaming adapter (Part C).

No real network/provider call anywhere in this file. `stream_chat_
answer` receives a fully synthetic, in-memory fake standing in for
`AsyncOpenAI().chat.completions.stream(...)` -- built from the exact
public shape (`async with ... as stream: async for event in stream:
...; await stream.get_final_completion()`) this project's own real
usage will call, confirmed directly against the installed
`openai==2.44.0` SDK before writing this file (see chat_streaming.py's
own module docstring for the full evidence trail from real, offline
`ChatCompletionStreamState` experiments this test suite's synthetic
event shapes are modeled on).
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from openai import APIConnectionError, OpenAIError
from unittest.mock import MagicMock

from research_agent.chat_streaming import (
    AnswerCompleted,
    AnswerDelta,
    ChatAnswerStreamError,
    StreamedChatAnswer,
    build_completed_event,
    build_delta_event,
    build_done_event,
    build_error_event,
    build_phase_event,
    build_started_event,
    stream_chat_answer,
)
from research_agent.qa import _build_answer_schema
from research_agent.schema import Paper, WebArticle

# --- shared fakes/helpers ---------------------------------------------

def _paper(pid: str, title: str = "T") -> Paper:
    return Paper(
        title=title, authors=["A"], year=2024, venue="X", abstract="an abstract",
        url=None, doi=None, citation_count=None, source="arxiv", paper_id=pid,
    )


def _web(url: str, title: str = "W") -> WebArticle:
    return WebArticle(title=title, url=url, snippet="a snippet", published_date=None, source_domain="example.com")


class _FakeEvent:
    def __init__(self, type_: str, parsed=None):
        self.type = type_
        self.parsed = parsed


class _FakeParsed:
    """Stands in for the SDK's own partial-parse snapshot object --
    only ever exposes `.answer`, matching exactly what stream_chat_
    answer itself ever reads off an intermediate event."""

    def __init__(self, answer):
        self.answer = answer


class _FakeStream:
    def __init__(self, events, final=None, final_exc: Exception | None = None):
        self._events = events
        self._final = final
        self._final_exc = final_exc

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for e in self._events:
            yield e

    async def get_final_completion(self):
        if self._final_exc is not None:
            raise self._final_exc
        return self._final


class _FakeStreamCtx:
    def __init__(self, events, final=None, final_exc=None, enter_exc: Exception | None = None):
        self._stream = _FakeStream(events, final, final_exc)
        self._enter_exc = enter_exc

    async def __aenter__(self):
        if self._enter_exc is not None:
            raise self._enter_exc
        return self._stream

    async def __aexit__(self, *args):
        return False


def _make_client(events, final=None, final_exc=None, enter_exc=None):
    client = MagicMock()
    client.chat.completions.stream = MagicMock(
        return_value=_FakeStreamCtx(events, final=final, final_exc=final_exc, enter_exc=enter_exc),
    )
    return client


def _make_final(schema, answer_text, answerable=True, cited_paper_ids=None, cited_web_urls=None):
    kwargs = {"answerable": answerable, "answer": answer_text}
    if "cited_paper_ids" in schema.model_fields:
        kwargs["cited_paper_ids"] = cited_paper_ids if cited_paper_ids is not None else []
    if "cited_web_urls" in schema.model_fields:
        kwargs["cited_web_urls"] = cited_web_urls if cited_web_urls is not None else []
    parsed = schema(**kwargs)
    message = MagicMock(parsed=parsed)
    choice = MagicMock(message=message)
    return MagicMock(choices=[choice])


def _run(coro):
    return asyncio.run(coro)


async def _collect(gen):
    deltas: list[str] = []
    completed: AnswerCompleted | None = None
    async for item in gen:
        if isinstance(item, AnswerDelta):
            deltas.append(item.text)
        elif isinstance(item, AnswerCompleted):
            completed = item
        else:
            raise AssertionError(f"unexpected yielded item type: {type(item)}")
    return deltas, completed


# --- Part A: event builders --------------------------------------------

class TestEventBuilders:
    def test_started_event_has_no_domain_payload(self):
        event = build_started_event()
        assert event.type == "started"
        assert event.data == {}

    def test_phase_event_carries_only_the_phase_field(self):
        event = build_phase_event("generating")
        assert event.data == {"phase": "generating"}

    def test_delta_event_carries_only_text(self):
        event = build_delta_event("hello")
        assert event.data == {"text": "hello"}

    def test_completed_event_carries_final_result_and_citations(self):
        result = StreamedChatAnswer(
            answer="LoRA reduces params [Paper 1].", answerable=True,
            cited_papers=[_paper("p1", "LoRA Paper")], cited_web_articles=[_web("https://x.com", "X Article")],
        )
        event = build_completed_event(result)
        assert event.data["answer"] == "LoRA reduces params [Paper 1]."
        assert event.data["answerable"] is True
        assert event.data["cited_papers"] == [{"paper_id": "p1", "title": "LoRA Paper"}]
        assert event.data["cited_web_articles"] == [{"url": "https://x.com", "title": "X Article"}]

    def test_completed_event_never_includes_abstract_or_snippet(self):
        result = StreamedChatAnswer(
            answer="ok", answerable=True, cited_papers=[_paper("p1")], cited_web_articles=[],
        )
        event = build_completed_event(result)
        serialized = str(event.data)
        assert "an abstract" not in serialized

    def test_error_event_carries_reason_code_and_message_only(self):
        event = build_error_event("provider_error", "The model provider returned an error.")
        assert event.data == {"reason_code": "provider_error", "message": "The model provider returned an error."}

    def test_done_event_has_no_domain_payload(self):
        event = build_done_event()
        assert event.data == {}

    def test_every_event_serializes_to_a_valid_sse_frame(self):
        for event in [
            build_started_event(), build_phase_event("generating"), build_delta_event("x"),
            build_completed_event(StreamedChatAnswer(answer="x", answerable=True, cited_papers=[], cited_web_articles=[])),
            build_error_event("refused", "x"), build_done_event(),
        ]:
            frame = event.to_sse()
            assert frame.startswith(f"event: {event.type}\n")
            assert frame.endswith("\n\n")


# --- Part C: streaming adapter -------------------------------------------

class TestStreamChatAnswer:
    def test_final_only_structured_output_zero_useful_deltas_until_the_end(self):
        """The realistic, empirically-observed case for this schema
        shape (see chat_streaming.py's own module docstring): parsed
        stays None until the closing quote, then jumps to the full
        string in one event."""
        schema = _build_answer_schema(["p1"])
        events = [
            _FakeEvent("content.delta", parsed=None),
            _FakeEvent("content.delta", parsed=None),
            _FakeEvent("content.delta", parsed=_FakeParsed("LoRA reduces trainable parameters.")),
        ]
        final = _make_final(schema, "LoRA reduces trainable parameters.", cited_paper_ids=["p1"])
        client = _make_client(events, final=final)

        deltas, completed = _run(_collect(stream_chat_answer(
            client, [{"role": "user", "content": "hi"}], schema, {"p1": _paper("p1")}, {},
        )))

        assert "".join(deltas) == completed.result.answer == "LoRA reduces trainable parameters."
        assert client.chat.completions.stream.call_count == 1

    def test_multiple_monotonically_growing_snapshots_produce_exact_suffix_deltas(self):
        schema = _build_answer_schema(["p1"])
        events = [
            _FakeEvent("content.delta", parsed=_FakeParsed("LoRA")),
            _FakeEvent("content.delta", parsed=_FakeParsed("LoRA reduces")),
            _FakeEvent("content.delta", parsed=_FakeParsed("LoRA reduces params.")),
        ]
        final = _make_final(schema, "LoRA reduces params.")
        client = _make_client(events, final=final)

        deltas, completed = _run(_collect(stream_chat_answer(
            client, [{"role": "user", "content": "hi"}], schema, {}, {},
        )))

        assert deltas == ["LoRA", " reduces", " params."]
        assert "".join(deltas) == completed.result.answer

    def test_repeated_identical_snapshots_produce_no_duplicate_text(self):
        schema = _build_answer_schema(["p1"])
        events = [
            _FakeEvent("content.delta", parsed=_FakeParsed("LoRA")),
            _FakeEvent("content.delta", parsed=_FakeParsed("LoRA")),  # exact repeat
            _FakeEvent("content.delta", parsed=_FakeParsed("LoRA")),  # exact repeat again
            _FakeEvent("content.delta", parsed=_FakeParsed("LoRA reduces.")),
        ]
        final = _make_final(schema, "LoRA reduces.")
        client = _make_client(events, final=final)

        deltas, completed = _run(_collect(stream_chat_answer(
            client, [{"role": "user", "content": "hi"}], schema, {}, {},
        )))

        assert deltas == ["LoRA", " reduces."]  # no duplicate "LoRA" entries
        assert "".join(deltas) == completed.result.answer

    def test_raw_json_provider_chunks_are_never_exposed_as_deltas(self):
        """Even if event.parsed happens to be unavailable (None) for a
        chunk, the raw JSON-fragment text (e.g. from a hypothetical
        `.delta`/`.snapshot` attribute) must never leak into an
        AnswerDelta -- only .parsed.answer is ever read."""
        schema = _build_answer_schema(["p1"])

        class _RawLeakEvent:
            type = "content.delta"
            parsed = None
            delta = '{"answerable": true, "ans'  # raw JSON fragment -- must never surface
            snapshot = '{"answerable": true, "ans'

        events = [_RawLeakEvent(), _FakeEvent("content.delta", parsed=_FakeParsed("Full answer."))]
        final = _make_final(schema, "Full answer.")
        client = _make_client(events, final=final)

        deltas, completed = _run(_collect(stream_chat_answer(
            client, [{"role": "user", "content": "hi"}], schema, {}, {},
        )))

        for d in deltas:
            assert "answerable" not in d
            assert "{" not in d and "}" not in d
        assert "".join(deltas) == completed.result.answer

    def test_final_delta_concatenation_exactly_equals_validated_final_answer(self):
        schema = _build_answer_schema(["p1", "p2"])
        events = [
            _FakeEvent("content.delta", parsed=_FakeParsed("Part one, ")),
            _FakeEvent("content.delta", parsed=_FakeParsed("Part one, part two, ")),
            _FakeEvent("content.delta", parsed=_FakeParsed("Part one, part two, part three.")),
        ]
        final = _make_final(schema, "Part one, part two, part three.", cited_paper_ids=["p1", "p2"])
        client = _make_client(events, final=final)

        deltas, completed = _run(_collect(stream_chat_answer(
            client, [{"role": "user", "content": "hi"}], schema, {"p1": _paper("p1"), "p2": _paper("p2")}, {},
        )))

        assert "".join(deltas) == completed.result.answer
        assert len(completed.result.cited_papers) == 2

    def test_citations_appear_only_in_the_final_result_never_mid_stream(self):
        schema = _build_answer_schema(["p1"], ["https://x.com"])
        events = [
            _FakeEvent("content.delta", parsed=_FakeParsed("Answer text.")),
        ]
        final = _make_final(schema, "Answer text.", cited_paper_ids=["p1"], cited_web_urls=["https://x.com"])
        papers_by_id = {"p1": _paper("p1")}
        web_by_url = {"https://x.com": _web("https://x.com")}
        client = _make_client(events, final=final)

        deltas, completed = _run(_collect(stream_chat_answer(
            client, [{"role": "user", "content": "hi"}], schema, papers_by_id, web_by_url,
        )))

        for d in deltas:
            assert "p1" not in d and "x.com" not in d
        assert completed.result.cited_papers[0].paper_id == "p1"
        assert completed.result.cited_web_articles[0].url == "https://x.com"

    def test_answerable_false_yields_no_citations_even_if_model_supplied_some(self):
        """Mirrors _generate_node's own defensive 'don't trust the
        model to honor empty-if-not-answerable' rule."""
        schema = _build_answer_schema(["p1"])
        final = _make_final(schema, "I don't know.", answerable=False, cited_paper_ids=["p1"])
        client = _make_client([], final=final)

        _, completed = _run(_collect(stream_chat_answer(
            client, [{"role": "user", "content": "hi"}], schema, {"p1": _paper("p1")}, {},
        )))

        assert completed.result.cited_papers == []

    def test_refusal_produces_no_completed_result(self):
        schema = _build_answer_schema(["p1"])
        message = MagicMock(parsed=None, refusal="cannot comply")
        final = MagicMock(choices=[MagicMock(message=message)])
        client = _make_client([], final=final)

        with pytest.raises(ChatAnswerStreamError) as exc_info:
            _run(_collect(stream_chat_answer(client, [{"role": "user", "content": "hi"}], schema, {}, {})))
        assert exc_info.value.reason_code == "refused"

    def test_malformed_final_output_produces_no_completed_result(self):
        """The final completion's own message has no usable .parsed at
        all and no .refusal either -- a genuinely malformed shape."""
        schema = _build_answer_schema(["p1"])
        message = MagicMock(parsed=None, refusal=None)
        final = MagicMock(choices=[MagicMock(message=message)])
        client = _make_client([], final=final)

        with pytest.raises(ChatAnswerStreamError) as exc_info:
            _run(_collect(stream_chat_answer(client, [{"role": "user", "content": "hi"}], schema, {}, {})))
        assert exc_info.value.reason_code == "refused"

    def test_non_monotonic_snapshot_fails_safely_with_no_completed_result(self):
        schema = _build_answer_schema(["p1"])
        events = [
            _FakeEvent("content.delta", parsed=_FakeParsed("The answer is X.")),
            _FakeEvent("content.delta", parsed=_FakeParsed("Something totally different.")),  # not an extension
        ]
        client = _make_client(events, final=_make_final(schema, "irrelevant"))

        with pytest.raises(ChatAnswerStreamError) as exc_info:
            _run(_collect(stream_chat_answer(client, [{"role": "user", "content": "hi"}], schema, {}, {})))
        assert exc_info.value.reason_code == "non_monotonic_answer"

    def test_non_monotonic_snapshot_shrinking_also_fails_safely(self):
        schema = _build_answer_schema(["p1"])
        events = [
            _FakeEvent("content.delta", parsed=_FakeParsed("A long partial answer here")),
            _FakeEvent("content.delta", parsed=_FakeParsed("A long")),  # shrank
        ]
        client = _make_client(events, final=_make_final(schema, "irrelevant"))

        with pytest.raises(ChatAnswerStreamError) as exc_info:
            _run(_collect(stream_chat_answer(client, [{"role": "user", "content": "hi"}], schema, {}, {})))
        assert exc_info.value.reason_code == "non_monotonic_answer"

    def test_final_answer_inconsistent_with_streamed_prefix_fails_safely(self):
        """Even if every intermediate snapshot was internally
        monotonic, a final answer that doesn't extend the last emitted
        text (a hypothetical SDK/model inconsistency) must still fail
        closed, never silently substitute the final value."""
        schema = _build_answer_schema(["p1"])
        events = [_FakeEvent("content.delta", parsed=_FakeParsed("Streamed prefix text"))]
        final = _make_final(schema, "A completely different final answer")
        client = _make_client(events, final=final)

        with pytest.raises(ChatAnswerStreamError) as exc_info:
            _run(_collect(stream_chat_answer(client, [{"role": "user", "content": "hi"}], schema, {}, {})))
        assert exc_info.value.reason_code == "non_monotonic_answer"

    def test_provider_exception_during_streaming_propagates_as_stable_outcome(self):
        schema = _build_answer_schema(["p1"])
        request = MagicMock()
        client = _make_client([], enter_exc=APIConnectionError(request=request))

        with pytest.raises(ChatAnswerStreamError) as exc_info:
            _run(_collect(stream_chat_answer(client, [{"role": "user", "content": "hi"}], schema, {}, {})))
        assert exc_info.value.reason_code == "provider_error"
        # Never the raw exception text.
        assert "APIConnectionError" not in exc_info.value.message

    def test_provider_exception_during_final_completion_propagates_as_stable_outcome(self):
        schema = _build_answer_schema(["p1"])
        events = [_FakeEvent("content.delta", parsed=_FakeParsed("partial"))]
        client = _make_client(events, final_exc=OpenAIError("upstream failure"))

        with pytest.raises(ChatAnswerStreamError) as exc_info:
            _run(_collect(stream_chat_answer(client, [{"role": "user", "content": "hi"}], schema, {}, {})))
        assert exc_info.value.reason_code == "provider_error"
        assert "upstream failure" not in exc_info.value.message

    def test_cancellation_propagates_as_a_stable_internal_outcome(self):
        schema = _build_answer_schema(["p1"])

        class _CancellingStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise asyncio.CancelledError()

            async def get_final_completion(self):
                raise AssertionError("should never be reached after cancellation")

        class _CancellingCtx:
            async def __aenter__(self):
                return _CancellingStream()

            async def __aexit__(self, *a):
                return False

        client = MagicMock()
        client.chat.completions.stream = MagicMock(return_value=_CancellingCtx())

        with pytest.raises(ChatAnswerStreamError) as exc_info:
            _run(_collect(stream_chat_answer(client, [{"role": "user", "content": "hi"}], schema, {}, {})))
        assert exc_info.value.reason_code == "cancelled"

    def test_no_session_history_or_persistence_mutation(self):
        """stream_chat_answer takes no session/history argument at all
        -- structurally cannot mutate one. This test documents/locks
        that contract via the function's own signature."""
        import inspect
        params = inspect.signature(stream_chat_answer).parameters
        assert "session" not in params
        assert "history" not in params
        assert "cp" not in params  # no checkpointer parameter either

    def test_exactly_one_provider_stream_invocation_on_success(self):
        schema = _build_answer_schema(["p1"])
        client = _make_client(
            [_FakeEvent("content.delta", parsed=_FakeParsed("ok"))],
            final=_make_final(schema, "ok"),
        )
        _run(_collect(stream_chat_answer(client, [{"role": "user", "content": "hi"}], schema, {}, {})))
        client.chat.completions.stream.assert_called_once()

    def test_exactly_one_provider_stream_invocation_on_failure_no_fallback_call(self):
        schema = _build_answer_schema(["p1"])
        client = _make_client([], enter_exc=OpenAIError("down"))
        with pytest.raises(ChatAnswerStreamError):
            _run(_collect(stream_chat_answer(client, [{"role": "user", "content": "hi"}], schema, {}, {})))
        client.chat.completions.stream.assert_called_once()
        # No parse()/create() fallback call of any kind.
        client.chat.completions.parse.assert_not_called()
        client.chat.completions.create.assert_not_called()
