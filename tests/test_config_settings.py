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


if __name__ == "__main__":
    test_defaults_with_no_env_vars_set()
    test_empty_string_env_vars_are_treated_as_unset()
    test_every_env_var_override_is_read()
    test_get_settings_is_uncached_and_reflects_env_changes_between_calls()
    print("All tests passed!")
