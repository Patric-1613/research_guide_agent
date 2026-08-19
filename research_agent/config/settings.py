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

from research_agent.config.limits import get_usage_policy

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
    keyword_filter_policy_c_enabled: bool
    # Bounded concurrency for keyword_filter.py's per-paper provider
    # calls within one served batch. Clamped at read time (not merely
    # documented) to [1, provider_fan_out_limit] -- provider_fan_out_limit
    # already exists in UsagePolicy as this project's one general
    # cross-feature fan-out ceiling, so this setting can never silently
    # authorize more concurrency than that policy already allows,
    # without this module needing to own or duplicate that ceiling
    # itself. A further per-batch clamp (never more workers than
    # uncached papers in that specific turn, at most 10) happens in
    # research_agent/keyword_filter.py, since that number isn't known
    # until a real batch exists.
    keyword_filter_max_concurrent_calls: int


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


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """Same fail-loud posture as `_positive_int` in config/limits.py: a
    set, invalid, or out-of-[minimum, maximum] value raises rather than
    being silently clamped -- clamping would let a real misconfiguration
    (e.g. asking for more concurrency than provider_fan_out_limit
    permits) pass silently instead of surfacing at startup."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{name}={raw!r} is not a valid integer") from None
    if value < minimum or value > maximum:
        raise ValueError(f"{name}={value} must be between {minimum} and {maximum}")
    return value


def get_settings() -> Settings:
    fan_out_limit = get_usage_policy().provider_fan_out_limit
    return Settings(
        semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None,
        unpaywall_email=os.getenv("UNPAYWALL_EMAIL") or None,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
        openalex_mailto=os.getenv("OPENALEX_MAILTO") or None,
        keyword_filter_policy_c_enabled=_strict_bool("KEYWORD_FILTER_POLICY_C_ENABLED", False),
        # Provisional default of 3, same "conservative given this
        # project's own rate-limiting history" judgment call as
        # query_expansion.py's _MAX_CONCURRENT_TITLE_PAIRS, well under
        # provider_fan_out_limit's own default of 20.
        keyword_filter_max_concurrent_calls=_bounded_int(
            "KEYWORD_FILTER_MAX_CONCURRENT_CALLS", 3, 1, fan_out_limit,
        ),
    )
