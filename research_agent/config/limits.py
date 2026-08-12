"""Usage Protection M2.1: the centralized usage-policy configuration.

This module defines every threshold M2 will eventually enforce. **M2.1
itself enforces none of them** except the two that are wired directly
into the standalone agent in `research_agent/agent.py`
(`agent_model_call_limit_per_run`, `agent_tool_call_limit_per_run`,
`agent_recursion_limit`) and the ones the new admission/lease modules
(`research_agent/admission.py`, `research_agent/leases.py`) read to
evaluate — but not yet apply from any route — a decision. Request-body,
text-length, paper-count, chat-turn, and provider-fan-out limits are
defined here so their names and values are settled, but no code path
reads them yet; wiring them into routes is M2.2.

Every default below is **provisional**: a conservative starting point
chosen by inspecting the current agent topology (four tools, a handful
of search/rerank turns per run) and this project's existing test
conventions, not a value calibrated against real production telemetry.
Once M1's telemetry has real traffic to look at, these should be
revisited. Treat every number here as configurable and disposable, not
as an industry-standard fact.

Same conventions as `research_agent/config/settings.py`: a frozen
dataclass plus an uncached `get_usage_policy()` that re-reads
`os.environ` on every call, so tests using
`unittest.mock.patch.dict(os.environ, ...)` work without needing a
cache-invalidation mechanism.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class UsagePolicy:
    # --- Defined but NOT enforced in M2.1 (M2.2 applies these to routes) ---
    max_text_length: int
    max_picked_ids_per_mutation: int
    max_selected_papers_per_session: int
    max_request_body_bytes: int
    max_chat_turns_per_session: int
    provider_timeout_seconds: int
    provider_fan_out_limit: int

    # --- Read by research_agent/admission.py (M2.1), not yet applied by any route ---
    max_paid_actions_per_session_per_hour: int
    max_paid_actions_per_session_per_day: int
    global_paid_action_limit: int
    global_paid_action_window_minutes: int

    # --- Read by research_agent/leases.py (M2.1), wired into routes in M2.2A ---
    max_concurrent_expensive_actions_per_session: int
    expensive_action_lease_ttl_seconds: int

    # --- Enforced in M2.1, directly in agent.py's create_agent()/stream() ---
    agent_model_call_limit_per_run: int
    agent_tool_call_limit_per_run: int
    agent_recursion_limit: int

    # --- M3.1: read by research_agent/chat_summarization.py's pure helpers
    # only -- nothing on the live curation-chat request path calls any of
    # that module yet (see its own module docstring). Independent of
    # max_chat_turns_per_session above: that's a storage-capacity ceiling
    # (how many turns a session may ever hold, M2.2C), this is a
    # model-context-cost policy (how much of that stored history gets sent
    # to the model on any one turn) -- the two intentionally never share a
    # constant, so recalibrating one is never assumed to imply the other.
    chat_summary_trigger_tokens: int
    # In USER TURNS/EXCHANGES (one user+assistant pair), not raw
    # chat_history entries -- see chat_summarization.py's own exchange-
    # boundary helper for why a raw-entry count would be the wrong unit
    # here (a pair must never be split across the retained/summarized
    # boundary).
    chat_summary_keep_recent_turns: int
    chat_summary_max_output_tokens: int
    # Same turn/exchange unit as chat_summary_keep_recent_turns above --
    # the minimum number of NEWLY eligible turns that must have
    # accumulated since the current summary's own coverage before a
    # second (or later) summarization pass is allowed to fire again, even
    # if the token trigger alone would otherwise re-fire every turn.
    chat_summary_min_new_turns: int


def _positive_int(name: str, default: int) -> int:
    """Reads an env var as a positive int, or falls back to `default`.
    Every threshold in this module must be a positive integer -- zero or
    negative would mean "always reject"/"no budget", which is never
    what an operator setting this env var actually wants; they should
    unset it (falls back to the provisional default) rather than being
    able to silently misconfigure a policy field into a footgun."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid integer") from None
    if value <= 0:
        raise ValueError(f"{name}={raw!r} must be a positive integer")
    return value


def get_usage_policy() -> UsagePolicy:
    return UsagePolicy(
        max_text_length=_positive_int("USAGE_MAX_TEXT_LENGTH", 2_000),
        max_picked_ids_per_mutation=_positive_int("USAGE_MAX_PICKED_IDS_PER_MUTATION", 30),
        max_selected_papers_per_session=_positive_int("USAGE_MAX_SELECTED_PAPERS_PER_SESSION", 60),
        max_request_body_bytes=_positive_int("USAGE_MAX_REQUEST_BODY_BYTES", 64 * 1024),
        max_chat_turns_per_session=_positive_int("USAGE_MAX_CHAT_TURNS_PER_SESSION", 100),
        provider_timeout_seconds=_positive_int("USAGE_PROVIDER_TIMEOUT_SECONDS", 60),
        provider_fan_out_limit=_positive_int("USAGE_PROVIDER_FAN_OUT_LIMIT", 20),
        max_paid_actions_per_session_per_hour=_positive_int("USAGE_MAX_PAID_ACTIONS_PER_SESSION_PER_HOUR", 30),
        max_paid_actions_per_session_per_day=_positive_int("USAGE_MAX_PAID_ACTIONS_PER_SESSION_PER_DAY", 150),
        global_paid_action_limit=_positive_int("USAGE_GLOBAL_PAID_ACTION_LIMIT", 20),
        global_paid_action_window_minutes=_positive_int("USAGE_GLOBAL_PAID_ACTION_WINDOW_MINUTES", 10),
        max_concurrent_expensive_actions_per_session=_positive_int(
            "USAGE_MAX_CONCURRENT_EXPENSIVE_ACTIONS_PER_SESSION", 1
        ),
        # M2.2A: a crash-recovery ceiling, not a request timeout -- the TTL
        # a lease survives for if the worker holding it dies without
        # releasing it (process kill, uncaught interpreter-level crash).
        # Deliberately NOT derived from provider_timeout_seconds above (60s):
        # that bounds one provider call, but a guarded action (e.g. report
        # generation's draft -> evaluate -> revise refinement loop, or the
        # standalone agent's up-to-10-model-call run) is several provider
        # calls in sequence, so a single provider timeout would expire the
        # lease out from under a still-healthy, still-working request. 900s
        # (15 minutes) is a provisional judgment call sized for the slowest
        # realistic guarded workflow (multi-pass report refinement) with
        # real headroom, not a calibrated figure -- revisit once real
        # latency telemetry exists, same as every other provisional value
        # in this module.
        expensive_action_lease_ttl_seconds=_positive_int("USAGE_EXPENSIVE_ACTION_LEASE_TTL_SECONDS", 900),
        # 10 / 10 / 15 (M2.1, reaffirmed unchanged in M2.1b): configurable
        # PROVISIONAL safeguards, not calibrated or permanent values. Chosen
        # after inspecting agent.py's topology alone -- build_tools()
        # exposes 4 tools (arxiv/semantic_scholar/rerank/web search) and a
        # typical run alternates a handful of model turns with (possibly
        # parallel) tool calls before producing a final answer, so 10/10
        # gives real headroom for a normal multi-tool research turn while
        # still bounding a runaway loop. No real run-length telemetry has
        # been consulted to set these; revisit once it exists. Every field
        # here is overridable via its env var (see below) without a code
        # change, precisely so a recalibration is a config change, not a
        # redesign.
        agent_model_call_limit_per_run=_positive_int("USAGE_AGENT_MODEL_CALL_LIMIT_PER_RUN", 10),
        agent_tool_call_limit_per_run=_positive_int("USAGE_AGENT_TOOL_CALL_LIMIT_PER_RUN", 10),
        # Matches the task-specified default exactly; also passed as
        # LangGraph's own `recursion_limit` graph configuration (not
        # middleware) at the `agent.stream(...)` call site. Same provisional
        # status as the two limits above -- not derived from real traffic.
        agent_recursion_limit=_positive_int("USAGE_AGENT_RECURSION_LIMIT", 15),
        # M3.1: provisional, uncalibrated -- chosen the same way as every
        # other threshold in this module (topology inspection: today's
        # MAX_HISTORY_TURNS=8 cap already implies a typical uncompressed
        # recent-history window of roughly a few thousand tokens; 6000
        # gives real headroom above that before compression kicks in, not
        # a calibrated figure). Revisit once usage_telemetry.sqlite has
        # real curation-chat token data -- same caveat as every M1/M2
        # value already carries.
        chat_summary_trigger_tokens=_positive_int("USAGE_CHAT_SUMMARY_TRIGGER_TOKENS", 6_000),
        # Deliberately reuses qa.MAX_HISTORY_TURNS's own already-reasoned
        # value ("confirmed with the project owner: coherence rarely
        # depends on more than a handful of recent turns here") rather
        # than inventing a second, silently-different "how much recent
        # context matters" number in the same codebase.
        chat_summary_keep_recent_turns=_positive_int("USAGE_CHAT_SUMMARY_KEEP_RECENT_TURNS", 8),
        chat_summary_max_output_tokens=_positive_int("USAGE_CHAT_SUMMARY_MAX_OUTPUT_TOKENS", 800),
        chat_summary_min_new_turns=_positive_int("USAGE_CHAT_SUMMARY_MIN_NEW_TURNS", 4),
    )
