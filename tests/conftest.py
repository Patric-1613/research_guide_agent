"""Pytest-wide setup, applied before any test module (and therefore any
research_agent module) is imported.

Disables Langfuse tracing for the whole suite via its own supported kill
switch (LANGFUSE_TRACING_ENABLED), not a monkeypatch of get_client()/
@observe() — those wrap a real Langfuse client that's a lazy singleton,
constructed on the first actual traced call, not at decoration/import time,
so patching the get_client symbol per test file wouldn't stop the
underlying client the @observe decorator itself already resolved. Setting
the env var here, before pytest imports anything else, guarantees every
Langfuse client constructed during the test run (across dedup.py,
ingestion.py, embeddings.py, query_expansion.py, qa.py, summarize.py, and
any future @observe-decorated code) is inert from the start — the same
"zero real external calls" guarantee the OpenAI mocking already provides
per test file, just via Langfuse's own mechanism since it doesn't need a
client object threaded through every call the way OpenAI does.
"""

from __future__ import annotations

import os

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
