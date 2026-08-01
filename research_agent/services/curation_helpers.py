"""Shared curation-graph config builder.

Moved out of api.py (Phase 6) so this has a real, independent home —
api.py re-exports `_curation_config` so `research_agent.api.
_curation_config` keeps working unchanged for anything still reaching it
that way. Reaches `api._state` via `import research_agent.api as api`,
accessed only inside the function body (never at import time).

get_curation_checkpointer and _state itself are NOT moved here — they
stay in api.py (see docs/architecture.md).
"""

from __future__ import annotations

import research_agent.api as api
from research_agent.config import get_settings


def _curation_config() -> dict:
    """refill_pool() (called from inside the graph whenever the reserve
    runs low, on ANY turn, not just the first) requires "client" present
    with direct key access, not .get() — confirmed by hitting a real
    KeyError during Phase 6a's own design verification when it was
    omitted. Built fresh per request rather than cached in _state,
    since s2_api_key/openalex_mailto are read from the environment the
    same way /search already does, not assumed constant."""
    settings = get_settings()
    return {
        "client": api._state["client"],
        "s2_api_key": settings.semantic_scholar_api_key,
        "openalex_mailto": settings.openalex_mailto,
    }
