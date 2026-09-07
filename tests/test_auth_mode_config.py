"""Day 3, Part B: deterministic tests for the AUTH_MODE / legacy
AUTH_ENABLED transition and the firebase-mode config validation in
research_agent/config/settings.py's get_auth_config().

No Firebase or Postgres needed -- pure config parsing.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.config.settings import AuthConfig, get_auth_config

_AUTH_ENV_KEYS = (
    "APP_ENV", "AUTH_MODE", "AUTH_ENABLED", "AUTH_USERNAME", "AUTH_PASSWORD",
    "FIREBASE_PROJECT_ID", "FIREBASE_AUTH_EMULATOR_HOST",
)


def _env(**overrides: str) -> dict[str, str]:
    """A clean auth env: every auth-related key removed, then `overrides`
    applied. Used with clear=False so unrelated vars (OPENAI_API_KEY,
    etc.) survive."""
    base = {k: "" for k in _AUTH_ENV_KEYS}
    base.update(overrides)
    return base


# --- legacy AUTH_ENABLED path stays byte-compatible (Part G test 25) ---

def test_no_auth_env_vars_at_all_is_disabled_mode():
    with patch.dict(os.environ, _env(), clear=False):
        assert get_auth_config() == AuthConfig(mode="disabled")


def test_legacy_auth_enabled_true_resolves_to_basic_mode():
    with patch.dict(os.environ, _env(AUTH_ENABLED="true", AUTH_USERNAME="alice", AUTH_PASSWORD="x" * 16), clear=False):
        cfg = get_auth_config()
    assert cfg.mode == "basic"
    assert cfg.enabled is True
    assert cfg.username == "alice"


def test_legacy_auth_enabled_false_resolves_to_disabled_mode():
    with patch.dict(os.environ, _env(AUTH_ENABLED="false"), clear=False):
        assert get_auth_config().mode == "disabled"


def test_legacy_basic_auth_production_still_works_without_a_database_url():
    """A pre-Day-3 basic-auth production deployment sets no DATABASE_URL
    -- get_auth_config() must not start requiring one."""
    with patch.dict(
        os.environ,
        _env(APP_ENV="production", AUTH_ENABLED="true", AUTH_USERNAME="a", AUTH_PASSWORD="x" * 16),
        clear=False,
    ):
        assert get_auth_config().mode == "basic"


# --- AUTH_MODE is authoritative ---

@pytest.mark.parametrize("mode", ["disabled", "basic", "firebase"])
def test_auth_mode_explicit_values(mode):
    extra = {}
    if mode == "basic":
        extra = {"AUTH_USERNAME": "a", "AUTH_PASSWORD": "x" * 16}
    elif mode == "firebase":
        extra = {"FIREBASE_PROJECT_ID": "my-app-12345"}
    with patch.dict(os.environ, _env(AUTH_MODE=mode, **extra), clear=False):
        assert get_auth_config().mode == mode


def test_invalid_auth_mode_raises():
    with patch.dict(os.environ, _env(AUTH_MODE="on"), clear=False):
        with pytest.raises(ValueError, match="AUTH_MODE"):
            get_auth_config()


# --- no ambiguous AUTH_MODE + AUTH_ENABLED combinations ---

@pytest.mark.parametrize(
    "mode, enabled",
    [("basic", "true"), ("basic", "false"), ("disabled", "true"), ("firebase", "false"), ("firebase", "true")],
)
def test_setting_both_auth_mode_and_auth_enabled_always_raises(mode, enabled):
    extra = {"AUTH_USERNAME": "a", "AUTH_PASSWORD": "x" * 16, "FIREBASE_PROJECT_ID": "my-app-12345"}
    with patch.dict(os.environ, _env(AUTH_MODE=mode, AUTH_ENABLED=enabled, **extra), clear=False):
        with pytest.raises(RuntimeError, match="must not both be set"):
            get_auth_config()


# --- production refuses disabled auth (Part G test 26) ---

def test_production_refuses_disabled_via_auth_mode():
    with patch.dict(os.environ, _env(APP_ENV="production", AUTH_MODE="disabled"), clear=False):
        with pytest.raises(RuntimeError, match="must not be disabled when APP_ENV=production"):
            get_auth_config()


def test_production_refuses_disabled_via_no_config_at_all():
    with patch.dict(os.environ, _env(APP_ENV="production"), clear=False):
        with pytest.raises(RuntimeError, match="must not be disabled when APP_ENV=production"):
            get_auth_config()


# --- firebase mode config validation ---

def test_firebase_mode_requires_project_id():
    with patch.dict(os.environ, _env(AUTH_MODE="firebase"), clear=False):
        with pytest.raises(RuntimeError, match="FIREBASE_PROJECT_ID must be set"):
            get_auth_config()


@pytest.mark.parametrize("bad_id", ["BAD_ID", "x", "a" * 40, "-startswithhyphen", "endswithhyphen-", "has spaces"])
def test_firebase_mode_rejects_invalid_project_id(bad_id):
    with patch.dict(os.environ, _env(AUTH_MODE="firebase", FIREBASE_PROJECT_ID=bad_id), clear=False):
        with pytest.raises(RuntimeError, match="not a valid Google Cloud / Firebase project ID"):
            get_auth_config()


def test_firebase_config_carries_project_id_and_no_credentials():
    with patch.dict(os.environ, _env(AUTH_MODE="firebase", FIREBASE_PROJECT_ID="my-app-12345"), clear=False):
        cfg = get_auth_config()
    assert cfg.mode == "firebase"
    assert cfg.firebase_project_id == "my-app-12345"
    assert cfg.firebase_emulator_host is None
    assert cfg.username is None and cfg.password is None
    assert cfg.enabled is False  # the Basic gate stays a passthrough in firebase mode


# --- Part G test 24: emulator settings rejected in production ---

def test_emulator_host_is_rejected_in_production_regardless_of_mode():
    with patch.dict(
        os.environ,
        _env(APP_ENV="production", AUTH_MODE="firebase", FIREBASE_PROJECT_ID="my-app-12345",
             FIREBASE_AUTH_EMULATOR_HOST="127.0.0.1:9099"),
        clear=False,
    ):
        with pytest.raises(RuntimeError, match="FIREBASE_AUTH_EMULATOR_HOST must not be set when APP_ENV=production"):
            get_auth_config()


def test_emulator_host_is_rejected_in_production_even_in_basic_mode():
    with patch.dict(
        os.environ,
        _env(APP_ENV="production", AUTH_MODE="basic", AUTH_USERNAME="a", AUTH_PASSWORD="x" * 16,
             FIREBASE_AUTH_EMULATOR_HOST="127.0.0.1:9099"),
        clear=False,
    ):
        with pytest.raises(RuntimeError, match="FIREBASE_AUTH_EMULATOR_HOST must not be set when APP_ENV=production"):
            get_auth_config()


def test_emulator_host_allowed_outside_production_in_firebase_mode():
    with patch.dict(
        os.environ,
        _env(AUTH_MODE="firebase", FIREBASE_PROJECT_ID="my-app-12345", FIREBASE_AUTH_EMULATOR_HOST="localhost:9099"),
        clear=False,
    ):
        cfg = get_auth_config()
    assert cfg.firebase_emulator_host == "localhost:9099"


@pytest.mark.parametrize("bad_host", ["nohostport", "host:", ":9099", "host:notaport", "host:99999999"])
def test_emulator_host_must_be_host_colon_port(bad_host):
    with patch.dict(
        os.environ,
        _env(AUTH_MODE="firebase", FIREBASE_PROJECT_ID="my-app-12345", FIREBASE_AUTH_EMULATOR_HOST=bad_host),
        clear=False,
    ):
        with pytest.raises(RuntimeError, match="FIREBASE_AUTH_EMULATOR_HOST must be of the form"):
            get_auth_config()


# --- Part G test 27: no service-account JSON key is required anywhere ---

def test_no_service_account_key_is_referenced_by_any_day3_auth_module():
    """Verification uses only the project ID + Google's public certs.
    No Day-3 module reads GOOGLE_APPLICATION_CREDENTIALS, loads a
    key file, or constructs a firebase_admin.credentials.Certificate."""
    import inspect

    import research_agent.config.settings as settings_mod
    import research_agent.firebase_auth as fb
    import research_agent.firebase_auth_middleware as fb_mw
    import research_agent.identity as identity_mod

    # Code-shaped patterns only -- a docstring may *mention* that a key
    # file is never used (firebase_auth.py's does), which is not a
    # violation.
    forbidden = (
        'getenv("GOOGLE_APPLICATION_CREDENTIALS"',
        "getenv('GOOGLE_APPLICATION_CREDENTIALS'",
        "credentials.Certificate(",
        "firebase_admin",
        "from_service_account_file",
        "from_service_account_json",
    )
    for mod in (settings_mod, fb, fb_mw, identity_mod):
        src = inspect.getsource(mod)
        for needle in forbidden:
            assert needle not in src, f"{mod.__name__} references {needle!r}"
