"""Usage Protection M4.1: chat-streaming event vocabulary (Part A) and
the one-call structured-answer streaming adapter (Part C).

**Foundation only.** Nothing in this module is wired into a live
FastAPI route, no `StreamingResponse` exists yet, and nothing here
touches telemetry, admission, leases, or session/report persistence --
that is M4.2's job. `stream_chat_answer` below never mutates
`session.chat_history`, never calls `save_curation_session`, and opens
no `guard_paid_action`.

**Why not LangGraph `interrupt_before`.** M4's own planning phase
considered pausing `qa.py`'s compiled graph with
`interrupt_before=["generate_answer"]` so the existing
classify/condense/retrieve/filter nodes could run unchanged and only
the final, streaming-incompatible generation step would be replaced.
Tested directly against LangGraph's own `Pregel`/`CompiledStateGraph`
(not assumed): a graph compiled with `interrupt_before=[...]` and
`checkpointer=None` (`qa.py`'s own `_DEFAULT_GRAPH`'s exact
configuration) gives `.invoke()` no signal at all that it stopped early
(no `__interrupt__` key, no exception) -- `.get_state(...)` raises
`ValueError: No checkpointer set`, and naively "resuming" by feeding
the paused state back into `.invoke()` does NOT continue from the
interrupt point at all: it reruns the graph from `START` a second time,
re-executing every preceding node (confirmed directly: a toy two-node
graph's first node ran twice, its side effect applied twice). Applied
to `qa.py`'s real graph, this would mean `classify_message`/
`condense_question`/`retrieve`/`filter_web_relevance` -- including a
real embedding call and a real `condense_question` OpenAI call --
would silently run AGAIN on "resume," a serious hidden-cost/
correctness bug, not a safe pause/resume. Adding a real checkpointer +
thread-id configuration just to make this safe would be new
persistence semantics, explicitly out of scope for M4.1. **Rejected.**

Instead: `qa.prepare_answer_generation` (a plain, pure function
extracted from `_generate_node`, Part B) is called directly by both the
existing synchronous graph node and this module's streaming adapter --
no graph interruption, no new checkpointer, the existing
classify/condense/retrieve/filter graph is completely untouched and
still runs exactly as `_DEFAULT_GRAPH.invoke(...)` always has.

**Why not smooth, token-by-token `answer` text.** Verified directly
against the installed `openai==2.44.0` SDK (not assumed): the public
`client.chat.completions.stream(response_format=ChatAnswer)` API's own
`ContentDeltaEvent.parsed` is built from `jiter.from_json(snapshot,
partial_mode=True)` -- confirmed via direct source read of
`openai.lib.streaming.chat._completions.ChatCompletionStreamState.
_accumulate_chunk`. Tested this exact mechanism offline (synthetic
`ChatCompletionChunk` objects fed through the SDK's own
`ChatCompletionStreamState`, no network call): for a `str` field,
`jiter`'s `partial_mode=True` OMITS an in-progress (unterminated)
string value entirely rather than including its truncated prefix --
confirmed at every tested truncation point from just after the
opening quote through one character before the close. The `answer`
field's parsed value is therefore `None` throughout almost the entire
stream and then appears COMPLETE, in one jump, the instant its closing
quote is received -- never a smoothly growing prefix. (`jiter` does
support a `partial_mode="trailing-strings"` mode that WOULD include
the in-progress string -- confirmed directly -- but this is hardcoded
to `partial_mode=True` inside the SDK's own accumulator with no public
parameter to change it; using it would mean bypassing the SDK's own
event/accumulation pipeline entirely, exactly the kind of custom
plumbing this phase avoids building.)

**What this means for the adapter below**: it is real, correct, and
provably safe (one model call, exact delta/final-answer equality,
citations never provisional) -- but for `ChatAnswer`'s current
single-string-field shape, its own honestly-observed behavior is
"typically zero or very few deltas, often one large final delta,
followed by `completed`," not smooth token-by-token reveal. This is
disclosed explicitly, not hidden. See this project's own M4 planning
report and this module's test suite for the full evidence trail.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel

from research_agent.qa import ANSWER_MODEL
from research_agent.schema import Paper, WebArticle
from research_agent.sse import format_sse_event

# --- Part A: chat-streaming event vocabulary --------------------------

ChatStreamEventType = Literal["started", "phase", "delta", "completed", "error", "done"]

# The only phase values this module's own docstring/tests reference --
# a small, fixed, safe enum (never free text), matching this project's
# own established "safe fixed vocabulary, never raw internal detail"
# convention (M2.3's own reason-code message mapping). Usage Protection
# M4.2A: expanded from M4.1's original 3-value set to the full,
# grounded vocabulary the production streaming service
# (research_agent/curation_chat_streaming.py) actually emits --
# `summarizing_history` (M3 bounded-history summarization, only when it
# genuinely triggers), `searching_web` (a web-search-offer accept turn
# only), and `saving` (session persistence, every successful turn) --
# no phase here is emitted unless that operation genuinely occurs; see
# that module's own docstring for exactly when each one fires.
ChatStreamPhase = Literal[
    "preparing_context", "summarizing_history", "checking_relevance", "searching_web", "generating", "saving",
]


@dataclass
class ChatStreamEvent:
    """One event in the chat-streaming protocol. `data` must already be
    the complete, bounded, approved payload for `type` -- see each
    `build_*_event` helper below for the one-and-only place each event
    type's payload shape is decided. Never constructed with an
    arbitrary/unvalidated `data` dict from outside this module."""

    type: ChatStreamEventType
    data: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        return format_sse_event(self.type, self.data)


def build_started_event() -> ChatStreamEvent:
    return ChatStreamEvent(type="started", data={})


def build_phase_event(phase: ChatStreamPhase) -> ChatStreamEvent:
    return ChatStreamEvent(type="phase", data={"phase": phase})


def build_delta_event(text: str) -> ChatStreamEvent:
    """`text` is visible answer prose ONLY -- never a raw provider
    chunk, never JSON syntax, never citation-list content. See
    `stream_chat_answer` below for the one place this invariant is
    actually enforced (the monotonic-suffix algorithm), not this
    builder, which only frames whatever string it is given."""
    return ChatStreamEvent(type="delta", data={"text": text})


def build_completed_event(result: "StreamedChatAnswer") -> ChatStreamEvent:
    """The final canonical chat result -- citation metadata appears
    here, and ONLY here, never in any `delta`/`phase` payload. Mirrors
    `api_app/schemas.py`'s own `CitedPaperOut`/`CitedWebArticleOut`
    shape (`{paper_id, title}` / `{url, title}`) rather than a full
    `Paper`/`WebArticle` object -- no abstract, no snippet, same
    "response models are deliberately narrower than the domain object"
    convention every existing `CurationChatResponse`-family schema
    already uses."""
    return ChatStreamEvent(
        type="completed",
        data={
            "answer": result.answer,
            "answerable": result.answerable,
            "cited_papers": [{"paper_id": p.paper_id, "title": p.title} for p in result.cited_papers],
            "cited_web_articles": [{"url": a.url, "title": a.title} for a in result.cited_web_articles],
        },
    )


def build_error_event(reason_code: str, message: str) -> ChatStreamEvent:
    """`message` must always be a small, fixed, safe string -- never
    `str(exc)`, never a provider's own error body, never a path/SQL
    fragment. `stream_chat_answer`'s own `ChatAnswerStreamError` is the
    only source this module's own (future) caller should ever build
    this event from -- see that exception's own docstring for the
    closed set of reason codes it can carry."""
    return ChatStreamEvent(type="error", data={"reason_code": reason_code, "message": message})


def build_done_event() -> ChatStreamEvent:
    return ChatStreamEvent(type="done", data={})


# --- Part C: one-call structured-answer streaming adapter ---------------

@dataclass
class StreamedChatAnswer:
    """The adapter's own terminal result. Deliberately NOT the raw SDK
    `ParsedChatCompletion` object (never expose that upward) and
    deliberately NOT citation-marker-renumbered (`qa._renumber_
    citation_markers` is intentionally NOT applied here -- see this
    class's own field docstring below) -- both scope boundaries are
    explicit, not oversights.
    """

    # The raw, validated `answer` field exactly as the model produced
    # it -- NOT passed through `qa._renumber_citation_markers`.
    # Renumbering is a whole-string, non-streaming-safe transform (it
    # needs the complete text to decide final marker numbers), so
    # applying it here would break this adapter's own central proof
    # requirement (`"".join(deltas) == result.answer` holding EXACTLY,
    # not approximately). A future M4.2 integration applies renumbering
    # exactly where `_generate_node` already does today -- once, to
    # this terminal value, after streaming has finished -- never to
    # this adapter's own emitted deltas.
    answer: str
    answerable: bool
    cited_papers: list[Paper]
    cited_web_articles: list[WebArticle]


@dataclass
class AnswerDelta:
    """A new, non-empty suffix of visible answer text -- never a
    repeat, never non-monotonic (see `stream_chat_answer`'s own
    docstring for the exact guarantee)."""

    text: str


@dataclass
class AnswerCompleted:
    """Terminal item -- always the last thing `stream_chat_answer`
    yields on a successful run. Never yielded more than once."""

    result: StreamedChatAnswer


class ChatAnswerStreamError(Exception):
    """The ONE exception type `stream_chat_answer` ever raises --
    every recognized failure mode (refusal, malformed/incomplete final
    output, a non-monotonic partial-answer snapshot, a provider
    exception, cancellation) is normalized to one of this class's own
    fixed `reason_code` values below. Carries only a stable internal
    reason code and a small, safe, fixed message -- never raw provider
    exception text, never a stack trace, never request/response
    content. A future caller (M4.2) maps this directly to
    `build_error_event(exc.reason_code, exc.message)`.

    Reason codes this module raises, exhaustively:
      - "non_monotonic_answer": a later parsed `answer` snapshot was
        not an extension of the last emitted value (shrank, or
        diverged) -- the stream is abandoned immediately; nothing is
        fabricated or "corrected."
      - "refused": the model completed without producing a parsed
        `ChatAnswer` at all (a genuine refusal, or the SDK could not
        validate the final structured output against the schema).
      - "provider_error": a network/API-level failure from the
        provider (a raised `OpenAIError` or any other exception from
        inside the `async with ... stream(...)` block not already
        covered by a more specific reason code above).
      - "cancelled": an `asyncio.CancelledError` was observed while
        this adapter was awaiting provider output. See this class's
        own note on cancellation semantics below.
    """

    def __init__(self, reason_code: str, message: str):
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.message = message


async def stream_chat_answer(
    client: AsyncOpenAI,
    messages: list[dict],
    schema: type[BaseModel],
    papers_by_id: dict[str, Paper],
    web_by_url: dict[str, WebArticle],
    model: str = ANSWER_MODEL,
) -> AsyncIterator[AnswerDelta | AnswerCompleted]:
    """Usage Protection M4.1 Part C: exactly ONE OpenAI structured
    streaming call (`client.chat.completions.stream(...)`, the SDK's
    own public structured-output streaming API -- never `.create()`
    with a second, separate parse pass, never a second `.parse()`
    call). `messages`/`schema`/`papers_by_id`/`web_by_url` are expected
    to come straight from `qa.prepare_answer_generation` -- this
    function performs NO retrieval, filtering, or prompt construction
    of its own, matching Part B's own "don't duplicate this logic"
    requirement.

    Yields `AnswerDelta` (a new, non-empty, monotonic suffix of visible
    answer text -- see the algorithm note below) zero or more times,
    then exactly one terminal `AnswerCompleted` on success. Raises
    `ChatAnswerStreamError` (never a raw provider exception, never a
    bare `asyncio.CancelledError`) on every failure path -- no item is
    yielded after a raise.

    **Delta algorithm** (the adapter's own central, tested guarantee):
    tracks `last_emitted_answer`, starting at `""`. For each `content.
    delta` event, reads `event.parsed.answer` (a `str | None` --
    `None` whenever the SDK's own partial-JSON accumulation hasn't
    resolved a complete `answer` string value yet; see this module's
    own top-level docstring for why this is `None` far more often than
    it is a smoothly growing string, for this schema shape). A `None`
    or non-`str` snapshot is silently skipped (not an error -- it
    simply means "no new information yet"). An identical repeat of
    `last_emitted_answer` is silently skipped (no duplicate delta). A
    snapshot that does not `.startswith(last_emitted_answer)` is
    treated as `"non_monotonic_answer"` -- the stream ends immediately,
    nothing is persisted or fabricated. Otherwise the NEW SUFFIX ONLY
    (`candidate[len(last_emitted_answer):]`) is yielded and
    `last_emitted_answer` is updated to the full new value -- never the
    whole snapshot re-sent, never JSON syntax, never a raw provider
    chunk. After the stream's own final event, `get_final_completion()`
    supplies the fully validated `ChatAnswer`; if its own `.answer`
    extends beyond `last_emitted_answer`, the remaining suffix is
    emitted as one last `AnswerDelta` before `AnswerCompleted` --
    this is what makes the class's own central invariant hold exactly:
    concatenating every `AnswerDelta.text` this function ever yields
    equals `AnswerCompleted.result.answer` character-for-character,
    proven directly in this module's own test suite, including the
    realistic case (confirmed against this project's own real, offline
    SDK experiments) where zero useful deltas appear until that final
    suffix.

    Citation fields (`cited_papers`/`cited_web_articles`) are read
    ONLY from the terminal, fully-validated completion -- never from
    any intermediate `event.parsed` -- structurally guaranteed by this
    function's own control flow (nothing before `get_final_completion()`
    ever touches `cited_paper_ids`/`cited_web_urls` at all).

    Makes NO database/session mutation, appends nothing to any chat
    history, opens no telemetry/admission/lease context -- purely a
    provider-facing adapter. Exactly one `client.chat.completions.
    stream(...)` invocation per call, confirmed by this module's own
    test suite via call-count assertions; never falls back to a second,
    non-streaming completion call under any failure path.
    """
    last_emitted_answer = ""

    try:
        async with client.chat.completions.stream(model=model, messages=messages, response_format=schema) as stream:
            async for event in stream:
                if event.type != "content.delta":
                    continue
                candidate = getattr(event.parsed, "answer", None) if event.parsed is not None else None
                if candidate is None or not isinstance(candidate, str):
                    continue
                if candidate == last_emitted_answer:
                    continue
                if not candidate.startswith(last_emitted_answer):
                    raise ChatAnswerStreamError(
                        "non_monotonic_answer", "The model's streamed answer changed unexpectedly.",
                    )
                suffix = candidate[len(last_emitted_answer):]
                last_emitted_answer = candidate
                yield AnswerDelta(text=suffix)

            final = await stream.get_final_completion()
    except ChatAnswerStreamError:
        raise
    except asyncio.CancelledError:
        # Usage Protection M4.1: converted to this module's own stable
        # internal outcome, per this adapter's documented error
        # contract (every failure normalizes to ChatAnswerStreamError).
        # A future M4.2 caller that also needs the raw asyncio-level
        # cancellation to propagate further (e.g. so Starlette's own
        # disconnect handling correctly tears down the enclosing
        # request) must check `reason_code == "cancelled"` and act on
        # it independently -- converting the exception TYPE here does
        # not, by itself, keep the surrounding asyncio Task marked
        # cancelled. Documented here deliberately, not silently glossed
        # over.
        raise ChatAnswerStreamError("cancelled", "The request was cancelled.") from None
    except OpenAIError as exc:
        raise ChatAnswerStreamError("provider_error", "The model provider returned an error.") from exc
    except Exception as exc:
        raise ChatAnswerStreamError("provider_error", "The model provider returned an error.") from exc

    message = final.choices[0].message
    parsed_final = message.parsed
    if parsed_final is None:
        raise ChatAnswerStreamError("refused", "The model declined to answer.")

    final_answer_text = parsed_final.answer
    if not isinstance(final_answer_text, str) or not final_answer_text.startswith(last_emitted_answer):
        raise ChatAnswerStreamError(
            "non_monotonic_answer", "The model's final answer did not match its streamed text.",
        )
    remaining_suffix = final_answer_text[len(last_emitted_answer):]
    if remaining_suffix:
        yield AnswerDelta(text=remaining_suffix)

    cited_paper_ids = list(getattr(parsed_final, "cited_paper_ids", [])) if parsed_final.answerable else []
    cited_papers = [papers_by_id[pid] for pid in cited_paper_ids]
    cited_web_urls = list(getattr(parsed_final, "cited_web_urls", [])) if parsed_final.answerable else []
    cited_web_articles = [web_by_url[url] for url in cited_web_urls]

    yield AnswerCompleted(
        result=StreamedChatAnswer(
            answer=final_answer_text, answerable=parsed_final.answerable,
            cited_papers=cited_papers, cited_web_articles=cited_web_articles,
        ),
    )
