"""Shared runtime state and the curation checkpointer dependency.

Moved out of api.py (Phase 9) so these have a real, independent home —
api.py re-exports both names so `research_agent.api._state` and
`research_agent.api.get_curation_checkpointer` keep resolving to the
exact same dict/function objects for anything still reaching them that
way. `_state` is a plain mutable dict populated once, in place, by
api.py's own `lifespan()` (`_state["client"] = ...`) — every reader
(routers, services, api.py itself) sees the same object regardless of
which module holds the name.

`get_curation_checkpointer` is NOT wrapped or re-defined in api.py: it is
imported here and re-exported as the literal same function object, so
`app.dependency_overrides[api.get_curation_checkpointer]` (keyed by
callable identity) keeps matching `Depends(api.get_curation_checkpointer)`
in every curation router, unchanged.
"""

from __future__ import annotations

from research_agent.qa import sqlite_checkpointer

_state: dict = {}


def get_curation_checkpointer():
    """FastAPI dependency: a fresh SqliteSaver-backed checkpointer per
    request, same per-request-not-shared rationale as get_db_connection()
    above (a single shared connection is not safe across FastAPI's
    threadpool) — just wrapping sqlite_checkpointer()'s own contextmanager
    instead of a raw sqlite3.connect() call."""
    with sqlite_checkpointer() as cp:
        yield cp
