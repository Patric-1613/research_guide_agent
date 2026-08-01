"""Centralized, typed access to this project's environment-variable-driven
configuration.

Deliberately covers only the env vars our own code reads directly today:
`SEMANTIC_SCHOLAR_API_KEY`, `UNPAYWALL_EMAIL`, `TAVILY_API_KEY`,
`FRONTEND_ORIGIN`, `OPENALEX_MAILTO` — same names, same defaults, same
falsy-value-becomes-None handling as every call site being replaced. No
env var was renamed and no default changed.

**Intentionally NOT covered here, and why:**

- `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/
  `LANGFUSE_BASE_URL` — nothing in this codebase reads these directly.
  The OpenAI SDK's bare `OpenAI()` constructor call
  (`api_app/app.py`'s `lifespan()`, reached as the patch target
  `api.OpenAI`) and the Langfuse SDK's `get_client()` (`tracing.py`)
  both read these from `os.environ` internally. Routing them through
  this module would mean explicitly passing credentials into
  constructors that currently read them implicitly — a real behavior
  touchpoint on sensitive, central code paths, deferred rather than
  folded into this first, low-risk pass.
- Model-name constants (`EMBEDDING_MODEL`, `SUMMARY_MODEL`,
  `AGENT_MODEL`, etc.) and data/cache/Chroma paths (`DATA_DIR`,
  `DB_PATH`, `CHROMA_PERSIST_DIR`, `QA_CHECKPOINT_DB_PATH`) — none of
  these are read from the environment today; they're plain Python
  literals. Centralizing them would mean changing import structure
  across every domain module that defines one, for zero behavior
  difference — a separately-scoped step, not this one.

`get_settings()` re-reads `os.environ` on every call — deliberately
uncached, so existing tests that wrap a single call in
`unittest.mock.patch.dict(os.environ, ...)` keep working exactly as
before, and so a `.env` change takes effect without restarting anything
that imports this module lazily.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    semantic_scholar_api_key: str | None
    unpaywall_email: str | None
    tavily_api_key: str | None
    frontend_origin: str
    openalex_mailto: str | None


def get_settings() -> Settings:
    return Settings(
        semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None,
        unpaywall_email=os.getenv("UNPAYWALL_EMAIL") or None,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
        openalex_mailto=os.getenv("OPENALEX_MAILTO") or None,
    )
