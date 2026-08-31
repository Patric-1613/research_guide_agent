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

from research_agent.config import (
    get_auth_config,
    get_cors_config,
    get_keyword_filter_max_concurrent_calls,
    get_settings,
)
from research_agent.config.settings import AuthConfig, CorsConfig, Settings


def test_defaults_with_no_env_vars_set():
    with patch.dict(os.environ, {}, clear=True):
        settings = get_settings()

    assert settings == Settings(
        semantic_scholar_api_key=None,
        unpaywall_email=None,
        tavily_api_key=None,
        openalex_mailto=None,
        keyword_filter_policy_c_enabled=False,
        research_lanes_enabled=False,
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
        "OPENALEX_MAILTO": "openalex@example.org",
    }, clear=True):
        settings = get_settings()

    assert settings.semantic_scholar_api_key == "s2-key"
    assert settings.unpaywall_email == "contact@example.org"
    assert settings.tavily_api_key == "tavily-key"
    assert settings.openalex_mailto == "openalex@example.org"


def test_get_settings_has_no_frontend_origin_field():
    # FRONTEND_ORIGIN lives on the validated, may-raise get_cors_config()
    # -- see below -- not on Settings/get_settings(). There must be no
    # unvalidated raw accessor left to bypass it.
    with patch.dict(os.environ, {}, clear=True):
        assert not hasattr(get_settings(), "frontend_origin")


def test_get_settings_is_uncached_and_reflects_env_changes_between_calls():
    with patch.dict(os.environ, {}, clear=True):
        assert get_settings().semantic_scholar_api_key is None
        with patch.dict(os.environ, {"SEMANTIC_SCHOLAR_API_KEY": "s2-key"}):
            assert get_settings().semantic_scholar_api_key == "s2-key"
        assert get_settings().semantic_scholar_api_key is None


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


# --- Research Lanes (RL1): research_lanes_enabled feature flag ---

def test_research_lanes_disabled_by_default():
    with patch.dict(os.environ, {}, clear=True):
        assert get_settings().research_lanes_enabled is False


def test_research_lanes_enabled_true_false_parsing():
    for truthy in ("true", "True", "TRUE", "1"):
        with patch.dict(os.environ, {"RESEARCH_LANES_ENABLED": truthy}, clear=True):
            assert get_settings().research_lanes_enabled is True
    for falsy in ("false", "False", "FALSE", "0"):
        with patch.dict(os.environ, {"RESEARCH_LANES_ENABLED": falsy}, clear=True):
            assert get_settings().research_lanes_enabled is False


def test_research_lanes_enabled_empty_string_is_treated_as_unset():
    with patch.dict(os.environ, {"RESEARCH_LANES_ENABLED": ""}, clear=True):
        assert get_settings().research_lanes_enabled is False


def test_research_lanes_enabled_invalid_value_fails_clearly():
    with patch.dict(os.environ, {"RESEARCH_LANES_ENABLED": "yes-please"}, clear=True):
        try:
            get_settings()
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "RESEARCH_LANES_ENABLED" in str(exc)
            assert "yes-please" in str(exc)


# --- K5D.2a: get_keyword_filter_max_concurrent_calls is a SEPARATE,
# lazily-called function -- get_settings() itself must never touch
# KEYWORD_FILTER_MAX_CONCURRENT_CALLS or provider_fan_out_limit at all
# (see test_get_settings_never_reads_keyword_filter_concurrency_env_var
# and test_disabled_path_equivalence.py's own proof of the same point at
# the curation_loop integration level).

def test_get_settings_has_no_concurrency_field_at_all():
    with patch.dict(os.environ, {}, clear=True):
        assert not hasattr(get_settings(), "keyword_filter_max_concurrent_calls")


def test_get_settings_never_reads_keyword_filter_concurrency_env_var(monkeypatch):
    """A malformed KEYWORD_FILTER_MAX_CONCURRENT_CALLS must never even be
    LOOKED AT by get_settings() -- proven by making os.getenv raise for
    that one name specifically and confirming get_settings() still
    succeeds."""
    real_getenv = os.environ.get

    def _boom_on_concurrency_var(name, default=None):
        if name == "KEYWORD_FILTER_MAX_CONCURRENT_CALLS":
            raise AssertionError("get_settings() must never read this env var")
        return real_getenv(name, default)

    monkeypatch.setattr(os, "getenv", _boom_on_concurrency_var)
    with patch.dict(os.environ, {"KEYWORD_FILTER_MAX_CONCURRENT_CALLS": "not-an-int"}, clear=True):
        settings = get_settings()
    assert settings.keyword_filter_policy_c_enabled is False


def test_keyword_filter_max_concurrent_calls_default_is_three():
    with patch.dict(os.environ, {}, clear=True):
        assert get_keyword_filter_max_concurrent_calls(fan_out_limit=20) == 3


def test_keyword_filter_max_concurrent_calls_override_within_bounds():
    with patch.dict(os.environ, {"KEYWORD_FILTER_MAX_CONCURRENT_CALLS": "5"}, clear=True):
        assert get_keyword_filter_max_concurrent_calls(fan_out_limit=20) == 5


def test_keyword_filter_max_concurrent_calls_rejects_zero_or_negative():
    for bad in ("0", "-1"):
        with patch.dict(os.environ, {"KEYWORD_FILTER_MAX_CONCURRENT_CALLS": bad}, clear=True):
            try:
                get_keyword_filter_max_concurrent_calls(fan_out_limit=20)
                assert False, "expected ValueError"
            except ValueError as exc:
                assert "positive integer" in str(exc)


def test_keyword_filter_max_concurrent_calls_non_integer_fails_clearly():
    with patch.dict(os.environ, {"KEYWORD_FILTER_MAX_CONCURRENT_CALLS": "three"}, clear=True):
        try:
            get_keyword_filter_max_concurrent_calls(fan_out_limit=20)
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "not a valid integer" in str(exc)


def test_default_concurrency_is_clamped_not_rejected_when_fan_out_limit_is_two():
    """Codex MEDIUM finding, reproduced directly: a low
    provider_fan_out_limit (2) must CLAMP the default requested
    concurrency (3) down to 2, never raise."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_keyword_filter_max_concurrent_calls(fan_out_limit=2) == 2


def test_explicit_valid_concurrency_above_fan_out_limit_is_clamped_not_rejected():
    with patch.dict(os.environ, {"KEYWORD_FILTER_MAX_CONCURRENT_CALLS": "21"}, clear=True):
        assert get_keyword_filter_max_concurrent_calls(fan_out_limit=20) == 20


def test_concurrency_at_or_below_fan_out_limit_is_unchanged():
    with patch.dict(os.environ, {"KEYWORD_FILTER_MAX_CONCURRENT_CALLS": "8"}, clear=True):
        assert get_keyword_filter_max_concurrent_calls(fan_out_limit=8) == 8
        assert get_keyword_filter_max_concurrent_calls(fan_out_limit=100) == 8


def test_effective_concurrency_is_always_at_least_one():
    with patch.dict(os.environ, {}, clear=True):
        assert get_keyword_filter_max_concurrent_calls(fan_out_limit=1) == 1


# --- PR2B: get_auth_config() -- the fail-closed access-gate config ---

_VALID_AUTH_ENV = {
    "AUTH_ENABLED": "true", "AUTH_USERNAME": "alice", "AUTH_PASSWORD": "s3curePlatformSecret!",
}


def test_get_auth_config_disabled_by_default_in_local_mode():
    """No env vars set at all -- the exact current, unauthenticated
    local-dev/test state -- must return a disabled config, never raise."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_auth_config() == AuthConfig(enabled=False, username=None, password=None)


def test_get_auth_config_app_env_defaults_to_local():
    with patch.dict(os.environ, {}, clear=True):
        # Disabled + no raise is only possible if APP_ENV defaulted to
        # "local" -- "production" would have raised (see next test).
        get_auth_config()


def test_get_auth_config_invalid_app_env_raises():
    with patch.dict(os.environ, {"APP_ENV": "prod"}, clear=True):  # "production" misspelled
        try:
            get_auth_config()
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "APP_ENV" in str(exc)


def test_get_auth_config_production_without_auth_enabled_raises():
    """The core PR2A correction: disabling auth is never an acceptable
    production configuration, and there is no override for it."""
    with patch.dict(os.environ, {"APP_ENV": "production"}, clear=True):
        try:
            get_auth_config()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "AUTH_ENABLED" in str(exc)


def test_get_auth_config_production_with_auth_enabled_explicitly_false_still_raises():
    """Same as above, but with AUTH_ENABLED explicitly set to false --
    proves there is no env-var-shaped emergency production auth-disable
    switch, not just that the unset default is rejected."""
    with patch.dict(os.environ, {"APP_ENV": "production", "AUTH_ENABLED": "false"}, clear=True):
        try:
            get_auth_config()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "AUTH_ENABLED" in str(exc)


def test_get_auth_config_production_with_valid_credentials_succeeds():
    with patch.dict(os.environ, {"APP_ENV": "production", **_VALID_AUTH_ENV}, clear=True):
        assert get_auth_config() == AuthConfig(enabled=True, username="alice", password="s3curePlatformSecret!")


def test_get_auth_config_enabled_missing_username_raises():
    env = dict(_VALID_AUTH_ENV)
    del env["AUTH_USERNAME"]
    with patch.dict(os.environ, env, clear=True):
        try:
            get_auth_config()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "AUTH_USERNAME" in str(exc)


# --- PR2B.1: AUTH_USERNAME containing ':' is ambiguous under RFC 7617's
# own username/password delimiter and must be rejected at startup;
# AUTH_PASSWORD may still contain ':' (auth_middleware.py only ever
# splits the decoded credential on the FIRST colon).

def test_get_auth_config_username_with_colon_raises():
    env = {**_VALID_AUTH_ENV, "AUTH_USERNAME": "ali:ce"}
    with patch.dict(os.environ, env, clear=True):
        try:
            get_auth_config()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "AUTH_USERNAME" in str(exc)


def test_get_auth_config_username_with_colon_error_never_includes_actual_username_or_password():
    env = {"AUTH_ENABLED": "true", "AUTH_USERNAME": "ali:ce-the-secret-user", "AUTH_PASSWORD": "s3curePlatformSecret!"}
    with patch.dict(os.environ, env, clear=True):
        try:
            get_auth_config()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            message = str(exc)
            assert "ali:ce-the-secret-user" not in message
            assert "s3curePlatformSecret!" not in message


def test_get_auth_config_password_with_colon_is_accepted():
    env = {"AUTH_ENABLED": "true", "AUTH_USERNAME": "alice", "AUTH_PASSWORD": "s3cure:Platform:Secret!"}
    with patch.dict(os.environ, env, clear=True):
        assert get_auth_config() == AuthConfig(enabled=True, username="alice", password="s3cure:Platform:Secret!")


def test_get_auth_config_enabled_missing_password_raises():
    env = dict(_VALID_AUTH_ENV)
    del env["AUTH_PASSWORD"]
    with patch.dict(os.environ, env, clear=True):
        try:
            get_auth_config()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "AUTH_PASSWORD" in str(exc)


def test_get_auth_config_enabled_short_password_raises():
    env = {**_VALID_AUTH_ENV, "AUTH_PASSWORD": "too-short-1"}
    assert len(env["AUTH_PASSWORD"]) < 16
    with patch.dict(os.environ, env, clear=True):
        try:
            get_auth_config()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "AUTH_PASSWORD" in str(exc)
            assert "too-short-1" not in str(exc)  # never echoes the actual (weak) value


def test_get_auth_config_local_mode_may_stay_disabled():
    with patch.dict(os.environ, {"APP_ENV": "local"}, clear=True):
        assert get_auth_config().enabled is False


def test_get_auth_config_local_mode_enabled_still_validates_credentials():
    """Local mode may enable the gate too (for manually testing it before
    a real deployment) -- and when it does, the same credential
    requirements apply; there is no separate, weaker local validation
    path."""
    with patch.dict(os.environ, {"APP_ENV": "local", "AUTH_ENABLED": "true"}, clear=True):
        try:
            get_auth_config()
            assert False, "expected RuntimeError for missing credentials"
        except RuntimeError:
            pass
    with patch.dict(os.environ, {"APP_ENV": "local", **_VALID_AUTH_ENV}, clear=True):
        assert get_auth_config() == AuthConfig(enabled=True, username="alice", password="s3curePlatformSecret!")


def test_get_auth_config_error_messages_never_include_the_actual_secret_values():
    cases = [
        {"APP_ENV": "production"},
        {**_VALID_AUTH_ENV, "AUTH_USERNAME": ""},
        {**_VALID_AUTH_ENV, "AUTH_USERNAME": "ali:ce"},
        {**_VALID_AUTH_ENV, "AUTH_PASSWORD": "weak"},
    ]
    for env in cases:
        with patch.dict(os.environ, env, clear=True):
            try:
                get_auth_config()
            except (RuntimeError, ValueError) as exc:
                message = str(exc)
                assert "s3curePlatformSecret!" not in message
                assert "alice" not in message


# --- get_cors_config() -- the validated FRONTEND_ORIGIN / credentialed
# CORS contract. Same "read once, allowed to raise" discipline as
# get_auth_config() above; never on the always-succeeds get_settings().

def test_get_cors_config_local_default_is_the_two_dev_origins():
    with patch.dict(os.environ, {}, clear=True):
        assert get_cors_config() == CorsConfig(
            allowed_origins=("http://localhost:5173", "http://127.0.0.1:5173")
        )
    with patch.dict(os.environ, {"APP_ENV": "local"}, clear=True):
        assert get_cors_config().allowed_origins == ("http://localhost:5173", "http://127.0.0.1:5173")


def test_get_cors_config_local_empty_string_is_treated_as_unset():
    with patch.dict(os.environ, {"FRONTEND_ORIGIN": "   "}, clear=True):
        assert get_cors_config().allowed_origins == ("http://localhost:5173", "http://127.0.0.1:5173")


def test_get_cors_config_production_unset_allows_no_cross_origin():
    """Same-origin production (frontend served by this same process) needs
    no CORS entry -- and must not silently get a dev origin."""
    with patch.dict(os.environ, {"APP_ENV": "production"}, clear=True):
        assert get_cors_config() == CorsConfig(allowed_origins=())


def test_get_cors_config_explicit_origin_is_the_only_allowed_one():
    for env_extra in ({}, {"APP_ENV": "local"}):
        with patch.dict(os.environ, {"FRONTEND_ORIGIN": "https://research.example.com", **env_extra}, clear=True):
            assert get_cors_config().allowed_origins == ("https://research.example.com",)


def test_get_cors_config_split_origin_production_requires_explicit_origin():
    with patch.dict(os.environ, {
        "APP_ENV": "production", "FRONTEND_ORIGIN": "https://research.example.com",
    }, clear=True):
        assert get_cors_config().allowed_origins == ("https://research.example.com",)


def test_get_cors_config_production_refuses_a_localhost_origin():
    for bad in (
        "http://localhost:5173", "http://127.0.0.1:5173", "https://localhost", "http://[::1]:5173",
    ):
        with patch.dict(os.environ, {"APP_ENV": "production", "FRONTEND_ORIGIN": bad}, clear=True):
            try:
                get_cors_config()
                assert False, f"expected RuntimeError for {bad!r}"
            except RuntimeError as exc:
                assert "local development origin" in str(exc)


def test_get_cors_config_normalizes_a_lone_trailing_slash_and_case():
    with patch.dict(os.environ, {"FRONTEND_ORIGIN": "HTTPS://App.Example.COM/"}, clear=True):
        assert get_cors_config().allowed_origins == ("https://app.example.com",)
    with patch.dict(os.environ, {"FRONTEND_ORIGIN": "  https://app.example.com:8443/  "}, clear=True):
        assert get_cors_config().allowed_origins == ("https://app.example.com:8443",)


def test_get_cors_config_rejects_invalid_origins():
    cases = {
        "ftp://app.example.com": "http or https",
        "app.example.com": "http or https",           # no scheme
        "//app.example.com": "http or https",           # scheme-relative
        "https://app.example.com/dashboard": "no path",
        "https://app.example.com/?next=/x": "query string or fragment",
        "https://app.example.com/#frag": "query string or fragment",
        "https://user:pass@app.example.com": "embedded credentials",
        "https://*.example.com": "wildcard",
        "*": "wildcard",
        "https://": "must include a host",
    }
    for raw, needle in cases.items():
        with patch.dict(os.environ, {"FRONTEND_ORIGIN": raw}, clear=True):
            try:
                get_cors_config()
                assert False, f"expected RuntimeError for {raw!r}"
            except RuntimeError as exc:
                assert needle in str(exc), f"{raw!r}: {exc}"


def test_get_cors_config_error_never_echoes_a_full_configured_value():
    # An origin isn't a secret, but the message stays a rule statement,
    # never a quote of the raw input (matching get_auth_config()).
    with patch.dict(os.environ, {"FRONTEND_ORIGIN": "https://user:hunter2@evil.example.com/secret"}, clear=True):
        try:
            get_cors_config()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "hunter2" not in str(exc)
            assert "evil.example.com" not in str(exc)


def test_get_cors_config_is_uncached_and_reflects_env_changes_between_calls():
    with patch.dict(os.environ, {}, clear=True):
        assert get_cors_config().allowed_origins == ("http://localhost:5173", "http://127.0.0.1:5173")
        with patch.dict(os.environ, {"FRONTEND_ORIGIN": "https://research.example.com"}):
            assert get_cors_config().allowed_origins == ("https://research.example.com",)
        assert get_cors_config().allowed_origins == ("http://localhost:5173", "http://127.0.0.1:5173")


if __name__ == "__main__":
    test_defaults_with_no_env_vars_set()
    test_empty_string_env_vars_are_treated_as_unset()
    test_every_env_var_override_is_read()
    test_get_settings_is_uncached_and_reflects_env_changes_between_calls()
    print("All tests passed!")
