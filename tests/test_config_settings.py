"""Deterministic tests for research_agent/config/settings.py — the
centralized, typed read of every env var the backend reads directly
today. get_settings() is deliberately uncached (see its own module
docstring), so every test below wraps a single call in
patch.dict(os.environ, ..., clear=True) to control exactly what it sees.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.config import get_settings
from research_agent.config.settings import Settings


def test_defaults_with_no_env_vars_set():
    with patch.dict(os.environ, {}, clear=True):
        settings = get_settings()

    assert settings == Settings(
        semantic_scholar_api_key=None,
        unpaywall_email=None,
        tavily_api_key=None,
        frontend_origin="http://localhost:5173",
        openalex_mailto=None,
        keyword_filter_policy_c_enabled=False,
        keyword_filter_max_concurrent_calls=3,
    )


def test_empty_string_env_vars_are_treated_as_unset():
    # Matches every original os.getenv(...) or None call site's behavior —
    # an explicitly-empty env var is the same as an absent one.
    with patch.dict(os.environ, {
        "SEMANTIC_SCHOLAR_API_KEY": "", "UNPAYWALL_EMAIL": "", "TAVILY_API_KEY": "", "OPENALEX_MAILTO": "",
    }, clear=True):
        settings = get_settings()

    assert settings.semantic_scholar_api_key is None
    assert settings.unpaywall_email is None
    assert settings.tavily_api_key is None
    assert settings.openalex_mailto is None


def test_every_env_var_override_is_read():
    with patch.dict(os.environ, {
        "SEMANTIC_SCHOLAR_API_KEY": "s2-key",
        "UNPAYWALL_EMAIL": "contact@example.org",
        "TAVILY_API_KEY": "tavily-key",
        "FRONTEND_ORIGIN": "https://app.example.org",
        "OPENALEX_MAILTO": "openalex@example.org",
    }, clear=True):
        settings = get_settings()

    assert settings.semantic_scholar_api_key == "s2-key"
    assert settings.unpaywall_email == "contact@example.org"
    assert settings.tavily_api_key == "tavily-key"
    assert settings.frontend_origin == "https://app.example.org"
    assert settings.openalex_mailto == "openalex@example.org"


def test_get_settings_is_uncached_and_reflects_env_changes_between_calls():
    with patch.dict(os.environ, {}, clear=True):
        assert get_settings().frontend_origin == "http://localhost:5173"
        with patch.dict(os.environ, {"FRONTEND_ORIGIN": "https://changed.example.org"}):
            assert get_settings().frontend_origin == "https://changed.example.org"
        assert get_settings().frontend_origin == "http://localhost:5173"


# --- K5D.2: keyword_filter_policy_c_enabled / keyword_filter_max_concurrent_calls ---

def test_keyword_filter_disabled_by_default():
    with patch.dict(os.environ, {}, clear=True):
        assert get_settings().keyword_filter_policy_c_enabled is False


def test_keyword_filter_enabled_true_false_parsing():
    for truthy in ("true", "True", "TRUE", "1"):
        with patch.dict(os.environ, {"KEYWORD_FILTER_POLICY_C_ENABLED": truthy}, clear=True):
            assert get_settings().keyword_filter_policy_c_enabled is True
    for falsy in ("false", "False", "FALSE", "0"):
        with patch.dict(os.environ, {"KEYWORD_FILTER_POLICY_C_ENABLED": falsy}, clear=True):
            assert get_settings().keyword_filter_policy_c_enabled is False


def test_keyword_filter_enabled_empty_string_is_treated_as_unset():
    with patch.dict(os.environ, {"KEYWORD_FILTER_POLICY_C_ENABLED": ""}, clear=True):
        assert get_settings().keyword_filter_policy_c_enabled is False


def test_keyword_filter_enabled_invalid_value_fails_clearly():
    with patch.dict(os.environ, {"KEYWORD_FILTER_POLICY_C_ENABLED": "enabled-ish"}, clear=True):
        try:
            get_settings()
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "KEYWORD_FILTER_POLICY_C_ENABLED" in str(exc)
            assert "enabled-ish" in str(exc)


def test_keyword_filter_max_concurrent_calls_default_is_three():
    with patch.dict(os.environ, {}, clear=True):
        assert get_settings().keyword_filter_max_concurrent_calls == 3


def test_keyword_filter_max_concurrent_calls_override_within_bounds():
    with patch.dict(os.environ, {"KEYWORD_FILTER_MAX_CONCURRENT_CALLS": "5"}, clear=True):
        assert get_settings().keyword_filter_max_concurrent_calls == 5


def test_keyword_filter_max_concurrent_calls_rejects_below_minimum():
    with patch.dict(os.environ, {"KEYWORD_FILTER_MAX_CONCURRENT_CALLS": "0"}, clear=True):
        try:
            get_settings()
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "between 1" in str(exc)


def test_keyword_filter_max_concurrent_calls_rejects_above_provider_fan_out_limit():
    with patch.dict(os.environ, {
        "KEYWORD_FILTER_MAX_CONCURRENT_CALLS": "21", "USAGE_PROVIDER_FAN_OUT_LIMIT": "20",
    }, clear=True):
        try:
            get_settings()
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "between 1 and 20" in str(exc)


def test_keyword_filter_max_concurrent_calls_clamp_tracks_provider_fan_out_limit():
    with patch.dict(os.environ, {
        "KEYWORD_FILTER_MAX_CONCURRENT_CALLS": "8", "USAGE_PROVIDER_FAN_OUT_LIMIT": "8",
    }, clear=True):
        assert get_settings().keyword_filter_max_concurrent_calls == 8


def test_keyword_filter_max_concurrent_calls_non_integer_fails_clearly():
    with patch.dict(os.environ, {"KEYWORD_FILTER_MAX_CONCURRENT_CALLS": "three"}, clear=True):
        try:
            get_settings()
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "not a valid integer" in str(exc)


if __name__ == "__main__":
    test_defaults_with_no_env_vars_set()
    test_empty_string_env_vars_are_treated_as_unset()
    test_every_env_var_override_is_read()
    test_get_settings_is_uncached_and_reflects_env_changes_between_calls()
    print("All tests passed!")
