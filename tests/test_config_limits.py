"""Usage Protection M2.1 Part A: tests for research_agent/config/limits.py.
Nothing here touches a real database, network, or OpenAI call -- pure
env-var-to-dataclass parsing.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from research_agent.config import UsagePolicy, get_usage_policy


def test_defaults_match_the_provisional_spec():
    policy = get_usage_policy()
    assert policy == UsagePolicy(
        max_text_length=2_000,
        max_picked_ids_per_mutation=30,
        max_selected_papers_per_session=60,
        max_request_body_bytes=64 * 1024,
        max_chat_turns_per_session=100,
        provider_timeout_seconds=60,
        provider_fan_out_limit=20,
        max_paid_actions_per_session_per_hour=30,
        max_paid_actions_per_session_per_day=150,
        global_paid_action_limit=20,
        global_paid_action_window_minutes=10,
        max_concurrent_expensive_actions_per_session=1,
        expensive_action_lease_ttl_seconds=900,
        agent_model_call_limit_per_run=10,
        agent_tool_call_limit_per_run=10,
        agent_recursion_limit=15,
    )


def test_every_field_is_individually_overridable(monkeypatch):
    monkeypatch.setenv("USAGE_MAX_TEXT_LENGTH", "500")
    monkeypatch.setenv("USAGE_AGENT_MODEL_CALL_LIMIT_PER_RUN", "3")
    monkeypatch.setenv("USAGE_AGENT_RECURSION_LIMIT", "7")
    policy = get_usage_policy()
    assert policy.max_text_length == 500
    assert policy.agent_model_call_limit_per_run == 3
    assert policy.agent_recursion_limit == 7
    # Untouched fields keep their provisional defaults.
    assert policy.max_selected_papers_per_session == 60


def test_uncached_reflects_env_change_between_calls(monkeypatch):
    """Same uncached-per-call convention as get_settings() -- a single
    process must be able to see a changed env var without a cache
    invalidation mechanism, which matters for both real .env reloads
    and this suite's own monkeypatch-based isolation."""
    assert get_usage_policy().max_chat_turns_per_session == 100
    monkeypatch.setenv("USAGE_MAX_CHAT_TURNS_PER_SESSION", "5")
    assert get_usage_policy().max_chat_turns_per_session == 5


@pytest.mark.parametrize("bad_value", ["0", "-1", "not-a-number", "1.5", ""])
def test_invalid_values_are_rejected_or_fall_back_consistently(monkeypatch, bad_value):
    monkeypatch.setenv("USAGE_MAX_TEXT_LENGTH", bad_value)
    if bad_value == "":
        # Empty string is treated the same as "unset" -- falls back to
        # the provisional default, consistent with os.getenv(...) or None
        # handling used throughout this project's config modules.
        assert get_usage_policy().max_text_length == 2_000
    else:
        with pytest.raises(ValueError):
            get_usage_policy()


def test_invalid_value_error_names_the_offending_env_var(monkeypatch):
    monkeypatch.setenv("USAGE_AGENT_TOOL_CALL_LIMIT_PER_RUN", "-5")
    with pytest.raises(ValueError, match="USAGE_AGENT_TOOL_CALL_LIMIT_PER_RUN"):
        get_usage_policy()
