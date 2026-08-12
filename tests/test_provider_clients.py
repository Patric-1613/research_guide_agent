"""Usage Protection M2.2C Part E: tests for
research_agent/provider_clients.py -- the shared factory adding the
centralized, provisional provider timeout to every production call site
that previously constructed a bare OpenAI() client. Also regression-
proves the existing, already-stricter provider-specific timeouts
(enrichment.py, ingestion.py) were left unchanged, per this phase's own
"existing stricter timeouts remain unchanged" requirement. No real
network/paid calls anywhere in this file.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.config import get_usage_policy
from research_agent.provider_clients import default_openai_client


def test_default_openai_client_applies_the_centralized_timeout():
    client = default_openai_client()
    assert client.timeout == get_usage_policy().provider_timeout_seconds


def test_default_openai_client_timeout_overridable_via_policy_env(monkeypatch):
    monkeypatch.setenv("USAGE_PROVIDER_TIMEOUT_SECONDS", "5")
    client = default_openai_client()
    assert client.timeout == 5


def test_default_openai_client_does_not_change_default_retry_behavior():
    """No automatic retries added by this factory -- max_retries stays
    at whatever the OpenAI SDK's own default already is, not a value
    this factory sets or overrides."""
    from openai import OpenAI

    plain = OpenAI(api_key="sk-test")
    factory_made = default_openai_client()
    assert factory_made.max_retries == plain.max_retries


def test_enrichment_request_timeout_unchanged_and_stricter_than_provider_default():
    from research_agent.enrichment import _REQUEST_TIMEOUT

    policy = get_usage_policy()
    assert _REQUEST_TIMEOUT == 10
    assert _REQUEST_TIMEOUT < policy.provider_timeout_seconds


def test_ingestion_semantic_scholar_and_openalex_timeouts_unchanged():
    import inspect

    from research_agent import ingestion

    source = inspect.getsource(ingestion)
    # Both direct requests.get() calls already pass an explicit,
    # stricter-than-60s timeout -- unchanged by this phase.
    assert "timeout=15" in source


def test_every_production_openai_construction_site_uses_the_shared_factory():
    """Regression guard: every production module that previously did a
    bare OpenAI() fallback now goes through default_openai_client()
    instead -- catches a future call site quietly reverting to an
    unbounded client."""
    import ast
    from pathlib import Path

    production_modules = [
        "research_agent/query_expansion.py",
        "research_agent/embeddings.py",
        "research_agent/summarize.py",
        "research_agent/agent.py",
        "research_agent/curation_chat.py",
        "research_agent/report.py",
        "research_agent/qa.py",
    ]
    repo_root = Path(__file__).resolve().parent.parent
    for rel_path in production_modules:
        source = (repo_root / rel_path).read_text()
        tree = ast.parse(source)
        bare_openai_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "OpenAI"
        ]
        assert bare_openai_calls == [], f"{rel_path} still constructs OpenAI() directly: {bare_openai_calls}"
