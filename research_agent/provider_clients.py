"""Usage Protection M2.2C Part E: the one shared factory for this
project's default (lazily-constructed) OpenAI client.

Every production call site that needs an OpenAI client already follows
the same `client = client or OpenAI()` fallback -- an explicit client is
used when the caller passed one (tests, mostly), otherwise a bare
`OpenAI()` is constructed on the spot. This factory is a drop-in
replacement for that bare `OpenAI()` call, adding the centralized,
provisional provider timeout via the SDK's own supported `timeout=`
constructor option -- not a broader refactor of how clients are
threaded through this codebase, which stays exactly as it was.

`api.OpenAI` (the plain class, re-exported from the `openai` package)
is deliberately NOT replaced by this factory -- many existing tests
`patch.object(api, "OpenAI", return_value=MagicMock())`, relying on it
being the real, patchable class reference. Called only at the specific
call sites that previously did a bare `OpenAI()`.
"""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAI

from research_agent.config import get_usage_policy


def default_openai_client() -> OpenAI:
    return OpenAI(timeout=get_usage_policy().provider_timeout_seconds)


def default_async_openai_client() -> AsyncOpenAI:
    """Usage Protection M4.2A: the async counterpart to
    `default_openai_client` above -- needed once a real production
    caller (the curation-chat streaming endpoint) constructs its own
    `AsyncOpenAI` client rather than receiving one pre-built (M4.1's
    `chat_streaming.stream_chat_answer` only ever accepted one as a
    parameter, never constructed one itself). Same centralized,
    provisional provider timeout convention as the sync factory --
    one real source of truth for both."""
    return AsyncOpenAI(timeout=get_usage_policy().provider_timeout_seconds)
