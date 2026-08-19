"""Centralized, typed access to this project's environment-variable-driven
configuration.

Deliberately covers only the env vars our own code reads directly today:
`SEMANTIC_SCHOLAR_API_KEY`, `UNPAYWALL_EMAIL`, `TAVILY_API_KEY`,
`FRONTEND_ORIGIN`, `OPENALEX_MAILTO` — same names, same defaults, same
falsy-value-becomes-None handling as every call site being replaced. No
env var was renamed and no default changed.

**Intentionally NOT covered here, and why:**

- `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/
  `LANGFUSE_BASE_URL` — nothing in this codebase reads these directly.
  The OpenAI SDK's bare `OpenAI()` constructor call
  (`api_app/app.py`'s `lifespan()`, reached as the patch target
  `api.OpenAI`) and the Langfuse SDK's `get_client()` (`tracing.py`)
  both read these from `os.environ` internally. Routing them through
  this module would mean explicitly passing credentials into
  constructors that currently read them implicitly — a real behavior
  touchpoint on sensitive, central code paths, deferred rather than
  folded into this first, low-risk pass.
- Model-name constants (`EMBEDDING_MODEL`, `SUMMARY_MODEL`,
  `AGENT_MODEL`, etc.) and data/cache/Chroma paths (`DATA_DIR`,
  `DB_PATH`, `CHROMA_PERSIST_DIR`, `QA_CHECKPOINT_DB_PATH`) — none of
  these are read from the environment today; they're plain Python
  literals. Centralizing them would mean changing import structure
  across every domain module that defines one, for zero behavior
  difference — a separately-scoped step, not this one.

`get_settings()` re-reads `os.environ` on every call — deliberately
uncached, so existing tests that wrap a single call in
`unittest.mock.patch.dict(os.environ, ...)` keep working exactly as
before, and so a `.env` change takes effect without restarting anything
that imports this module lazily.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    semantic_scholar_api_key: str | None
    unpaywall_email: str | None
    tavily_api_key: str | None
    frontend_origin: str
    openalex_mailto: str | None
    # K5D.2: off-by-default product behavior flags belong here, not in
    # UsagePolicy -- UsagePolicy is exclusively safety/threshold
    # configuration (request limits, admission budgets, lease TTLs; see
    # its own module docstring), never a feature on/off switch. Disabling
    # this restores plain YAKE-v2 for every FUTURE served batch
    # immediately (research_agent/curation_loop.py's _serve_batch_node
    # reads it fresh every turn, same uncached-per-call convention as
    # every other value here) -- it does NOT retroactively rewrite
    # keywords already persisted into a session's turn_history/
    # selected_papers from while the flag was on; those stay exactly as
    # served. That's a real, stated pilot limitation, not a bug -- see
    # research_agent/keyword_filter.py's own module docstring and
    # tests/test_keyword_filter.py's rollback test for the same point
    # proven in code rather than only asserted here.
    #
    # K5D.2a fix (Codex MEDIUM finding): this is the ONLY keyword_filter
    # field left on Settings -- a cheap, always-safe os.getenv() + strict
    # bool parse. KEYWORD_FILTER_MAX_CONCURRENT_CALLS deliberately does
    # NOT live here (see get_keyword_filter_max_concurrent_calls below):
    # computing it required get_usage_policy().provider_fan_out_limit,
    # which meant EVERY call to get_settings() -- including every
    # request on the old, disabled curation path, which has nothing to
    # do with keyword filtering -- silently depended on UsagePolicy
    # parsing successfully, and a malformed/out-of-range
    # KEYWORD_FILTER_MAX_CONCURRENT_CALLS could break curation even
    # while this flag was False. get_settings() now stays a lightweight
    # "is the feature even on" check; the full filter-only configuration
    # is loaded lazily, only by a caller that already confirmed this
    # field is True.
    keyword_filter_policy_c_enabled: bool


def _strict_bool(name: str, default: bool) -> bool:
    """Same fail-loud posture as config/limits.py's `_positive_int`: an
    unset/empty var falls back to `default`, but a SET, unparseable value
    raises rather than silently defaulting -- a typo'd
    KEYWORD_FILTER_POLICY_C_ENABLED=disabled must never be silently read
    as `default` (currently False) and quietly enable/disable production
    behavior differently than the operator intended."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0"):
        return False
    raise ValueError(f"{name}={raw!r} is not a valid boolean (use true/false or 1/0)")


def _positive_int(name: str, default: int) -> int:
    """Fail-loud like config/limits.py's own `_positive_int`: unset/empty
    falls back to `default`; a SET but non-integer or non-positive value
    always raises. Deliberately has NO upper bound of its own -- clamping
    against provider_fan_out_limit is
    get_keyword_filter_max_concurrent_calls's job below, kept separate so
    parsing this env var never needs UsagePolicy."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid integer") from None
    if value <= 0:
        raise ValueError(f"{name}={value} must be a positive integer")
    return value


def get_keyword_filter_max_concurrent_calls(fan_out_limit: int) -> int:
    """K5D.2a fix (Codex MEDIUM finding): the effective, ready-to-use
    concurrency bound for keyword_filter.py's per-paper provider calls --
    deliberately NOT part of Settings/get_settings() (see that
    dataclass's own field comment for why). Callers (research_agent.
    curation_loop's _serve_batch_node) must only call this after already
    confirming Settings.keyword_filter_policy_c_enabled is True, and
    must supply `fan_out_limit` themselves (typically
    get_usage_policy().provider_fan_out_limit) -- fetching UsagePolicy
    is the caller's decision to make on the already-enabled path, never
    something this module reaches for on its own.

    KEYWORD_FILTER_MAX_CONCURRENT_CALLS: malformed or <= 0 always raises
    clearly -- a real misconfiguration must surface, not silently
    default. A valid positive value (explicit, or the provisional
    default of 3, same "conservative given this project's own rate-
    limiting history" judgment call as query_expansion.py's
    _MAX_CONCURRENT_TITLE_PAIRS) that EXCEEDS `fan_out_limit` is CLAMPED
    down to it, never rejected -- fan_out_limit is this project's one
    general cross-feature concurrency ceiling, not a per-feature
    configuration error to reject a perfectly valid request over. The
    result is always >= 1.
    """
    requested = _positive_int("KEYWORD_FILTER_MAX_CONCURRENT_CALLS", 3)
    return max(1, min(requested, fan_out_limit))


def get_settings() -> Settings:
    """Deliberately lightweight and UsagePolicy-free -- called on every
    request regardless of whether the keyword-filter feature is on, so
    it must never be able to fail (or do meaningfully more work) because
    of a keyword-filter-only misconfiguration while the feature is off.
    See get_keyword_filter_max_concurrent_calls above for the
    concurrency setting this deliberately excludes."""
    return Settings(
        semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None,
        unpaywall_email=os.getenv("UNPAYWALL_EMAIL") or None,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
        openalex_mailto=os.getenv("OPENALEX_MAILTO") or None,
        keyword_filter_policy_c_enabled=_strict_bool("KEYWORD_FILTER_POLICY_C_ENABLED", False),
    )
