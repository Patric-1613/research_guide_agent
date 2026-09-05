"""Day 2 (public multi-user deployment foundation, see
docs/plans/public-multi-user-deployment-review.md), Part B/Part G:
deterministic tests for `research_agent.config.settings.get_database_config`
and its URL redaction. No Postgres required -- these are pure
configuration-parsing tests, same style as
`tests/test_config_settings.py`'s own `get_auth_config`/`get_cors_config`
coverage.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from research_agent.config.settings import (
    MAX_DATABASE_POOL_MAX_SIZE,
    _redact_database_url,
    get_database_config,
)


def test_unset_database_url_in_local_env_is_not_configured():
    with patch.dict(os.environ, {"APP_ENV": "local"}, clear=False):
        os.environ.pop("DATABASE_URL", None)
        cfg = get_database_config()
    assert cfg.configured is False
    assert cfg.url is None
    assert cfg.redacted_url is None


def test_default_app_env_with_no_database_url_is_not_configured():
    """APP_ENV itself unset (the overall default) behaves identically to
    APP_ENV=local -- Postgres is entirely optional for local dev."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("APP_ENV", None)
        os.environ.pop("DATABASE_URL", None)
        cfg = get_database_config()
    assert cfg.configured is False


# --- Part G, test 16: production configuration never silently falls back to SQLite ---

def test_production_without_database_url_raises():
    with patch.dict(os.environ, {"APP_ENV": "production"}, clear=False):
        os.environ.pop("DATABASE_URL", None)
        with pytest.raises(RuntimeError, match="DATABASE_URL must be set"):
            get_database_config()


def test_production_with_empty_database_url_raises():
    with patch.dict(os.environ, {"APP_ENV": "production", "DATABASE_URL": "   "}, clear=False):
        with pytest.raises(RuntimeError, match="DATABASE_URL must be set"):
            get_database_config()


def test_configured_database_url_in_local_env_is_honored():
    """Setting DATABASE_URL in local/dev is an explicit opt-in, not
    required -- but when a developer does set it, it's used exactly as
    given, same in either APP_ENV."""
    with patch.dict(os.environ, {"APP_ENV": "local", "DATABASE_URL": "postgresql://u:p@localhost:5432/d"}, clear=False):
        cfg = get_database_config()
    assert cfg.configured is True
    assert cfg.url == "postgresql://u:p@localhost:5432/d"


@pytest.mark.parametrize("bad_scheme", ["mysql://u:p@host/db", "sqlite:///file.db", "http://host/db"])
def test_wrong_scheme_always_raises_regardless_of_app_env(bad_scheme):
    with patch.dict(os.environ, {"APP_ENV": "local", "DATABASE_URL": bad_scheme}, clear=False):
        with pytest.raises(RuntimeError, match="postgresql"):
            get_database_config()


def test_url_with_no_host_raises():
    with patch.dict(os.environ, {"APP_ENV": "local", "DATABASE_URL": "postgresql:///dbname"}, clear=False):
        with pytest.raises(RuntimeError, match="host"):
            get_database_config()


def test_unparseable_url_raises():
    with patch.dict(os.environ, {"APP_ENV": "local", "DATABASE_URL": "postgresql://[::1"}, clear=False):
        with pytest.raises(RuntimeError):
            get_database_config()


@pytest.mark.parametrize("raw_value", ["0", "-1", "not-a-number", "3.5"])
def test_invalid_pool_min_size_raises(raw_value):
    with patch.dict(os.environ, {"DATABASE_POOL_MIN_SIZE": raw_value}, clear=False):
        os.environ.pop("DATABASE_URL", None)
        with pytest.raises(ValueError, match="DATABASE_POOL_MIN_SIZE"):
            get_database_config()


@pytest.mark.parametrize("raw_value", ["0", "-1", "not-a-number", "3.5"])
def test_invalid_pool_max_size_raises(raw_value):
    with patch.dict(os.environ, {"DATABASE_POOL_MAX_SIZE": raw_value}, clear=False):
        os.environ.pop("DATABASE_URL", None)
        with pytest.raises(ValueError, match="DATABASE_POOL_MAX_SIZE"):
            get_database_config()


def test_pool_max_size_over_ceiling_raises():
    with patch.dict(os.environ, {"DATABASE_POOL_MAX_SIZE": str(MAX_DATABASE_POOL_MAX_SIZE + 1)}, clear=False):
        os.environ.pop("DATABASE_URL", None)
        with pytest.raises(ValueError, match="exceeds the conservative upper bound"):
            get_database_config()


def test_pool_min_greater_than_max_raises():
    with patch.dict(os.environ, {"DATABASE_POOL_MIN_SIZE": "10", "DATABASE_POOL_MAX_SIZE": "5"}, clear=False):
        os.environ.pop("DATABASE_URL", None)
        with pytest.raises(ValueError, match="must not exceed"):
            get_database_config()


def test_pool_sizes_default_to_1_and_5():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DATABASE_POOL_MIN_SIZE", None)
        os.environ.pop("DATABASE_POOL_MAX_SIZE", None)
        os.environ.pop("DATABASE_URL", None)
        cfg = get_database_config()
    assert cfg.pool_min_size == 1
    assert cfg.pool_max_size == 5


# --- Part G, test 17: database credentials never appear in logs/errors ---

def test_redacted_url_never_contains_the_password():
    redacted = _redact_database_url("postgresql://myuser:supersecretpassword@dbhost:5432/mydb")
    assert "supersecretpassword" not in redacted
    assert redacted == "postgresql://myuser:***@dbhost:5432/mydb"


def test_configured_database_config_redacted_url_never_contains_the_password():
    with patch.dict(os.environ, {"APP_ENV": "local", "DATABASE_URL": "postgresql://myuser:supersecretpassword@dbhost:5432/mydb"}, clear=False):
        cfg = get_database_config()
    assert "supersecretpassword" not in cfg.redacted_url
    # DatabaseConfig overrides __repr__ (defense in depth) so an
    # accidental `logger.info(cfg)` never leaks the password either --
    # confirmed here, not just in the `.redacted_url` field directly.
    assert "supersecretpassword" not in repr(cfg)
    # cfg.url itself DOES still carry the real password (see
    # DatabaseConfig's own docstring: it exists to actually connect, and
    # is deliberately excluded from __repr__ rather than scrubbed
    # in-place) -- confirming that, and that redaction is genuinely
    # happening rather than a no-op.
    assert "supersecretpassword" in cfg.url
    assert cfg.url != cfg.redacted_url


def test_redact_url_with_no_credentials_omits_the_at_sign():
    redacted = _redact_database_url("postgresql://dbhost:5432/mydb")
    assert redacted == "postgresql://dbhost:5432/mydb"
    assert "@" not in redacted


def test_redact_malformed_url_falls_back_to_a_fixed_placeholder_never_raises():
    redacted = _redact_database_url("not a url at all :::")
    assert redacted == "<database url, redacted>"
