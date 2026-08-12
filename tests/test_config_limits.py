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
        chat_summary_trigger_tokens=6_000,
        chat_summary_keep_recent_turns=8,
        chat_summary_max_output_tokens=800,
        chat_summary_min_new_turns=4,
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


# --- M3.1: chat-summarization policy fields ---

def test_chat_summary_defaults_are_6000_8_800_4():
    policy = get_usage_policy()
    assert policy.chat_summary_trigger_tokens == 6_000
    assert policy.chat_summary_keep_recent_turns == 8
    assert policy.chat_summary_max_output_tokens == 800
    assert policy.chat_summary_min_new_turns == 4


def test_chat_summary_fields_are_individually_env_overridable(monkeypatch):
    monkeypatch.setenv("USAGE_CHAT_SUMMARY_TRIGGER_TOKENS", "3000")
    monkeypatch.setenv("USAGE_CHAT_SUMMARY_KEEP_RECENT_TURNS", "5")
    monkeypatch.setenv("USAGE_CHAT_SUMMARY_MAX_OUTPUT_TOKENS", "400")
    monkeypatch.setenv("USAGE_CHAT_SUMMARY_MIN_NEW_TURNS", "2")
    policy = get_usage_policy()
    assert policy.chat_summary_trigger_tokens == 3000
    assert policy.chat_summary_keep_recent_turns == 5
    assert policy.chat_summary_max_output_tokens == 400
    assert policy.chat_summary_min_new_turns == 2
    # Untouched M1/M2 fields keep their own provisional defaults --
    # confirms the new fields are independent, not a reinterpretation of
    # an existing field.
    assert policy.max_chat_turns_per_session == 100


@pytest.mark.parametrize(
    "env_var",
    [
        "USAGE_CHAT_SUMMARY_TRIGGER_TOKENS",
        "USAGE_CHAT_SUMMARY_KEEP_RECENT_TURNS",
        "USAGE_CHAT_SUMMARY_MAX_OUTPUT_TOKENS",
        "USAGE_CHAT_SUMMARY_MIN_NEW_TURNS",
    ],
)
@pytest.mark.parametrize("bad_value", ["0", "-1", "not-a-number", "1.5"])
def test_chat_summary_fields_reject_invalid_values_same_as_every_other_field(monkeypatch, env_var, bad_value):
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValueError, match=env_var):
        get_usage_policy()


def test_chat_summary_policy_is_independent_of_m2_storage_ceiling(monkeypatch):
    """M3.1's own field docstring claim, verified: chat_summary_trigger_tokens
    (a model-context-cost policy) and max_chat_turns_per_session (M2.2C's
    storage-capacity ceiling) never share a constant or move together."""
    monkeypatch.setenv("USAGE_MAX_CHAT_TURNS_PER_SESSION", "5")
    policy = get_usage_policy()
    assert policy.max_chat_turns_per_session == 5
    assert policy.chat_summary_trigger_tokens == 6_000
    assert policy.chat_summary_keep_recent_turns == 8


def test_existing_m1_m2_defaults_are_unchanged_by_this_addition():
    """M3.1 must not change any existing M1/M2 policy default -- spot-checks
    a representative field from each prior chunk (M1's chat-turn cap doesn't
    exist -- M2.2C's does; M2.1's agent limits; M2.2A's budgets)."""
    policy = get_usage_policy()
    assert policy.max_text_length == 2_000
    assert policy.max_chat_turns_per_session == 100
    assert policy.agent_model_call_limit_per_run == 10
    assert policy.agent_recursion_limit == 15
    assert policy.max_paid_actions_per_session_per_hour == 30
    assert policy.expensive_action_lease_ttl_seconds == 900
