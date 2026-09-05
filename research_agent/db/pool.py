"""Day 2: PostgreSQL connection-pool construction.

`build_connection_pool` is the ONE place a `psycopg_pool.ConnectionPool`
is ever constructed in this codebase. It does not own a module-level
singleton -- the caller (a script, a test fixture, or a future
`lifespan()`) owns the pool's lifetime and is responsible for closing it
(every call site in this checkpoint uses it as a context manager, which
`ConnectionPool` supports directly).

Pool sizing comes from `research_agent.config.get_database_config()`
only -- never a hand-typed `min_size`/`max_size` at a call site -- so
there is exactly one place `DATABASE_POOL_MIN_SIZE`/
`DATABASE_POOL_MAX_SIZE` are read and validated (see that function's own
docstring for the exact validation rules and the Cloud-SQL-Auth-Proxy
compatibility reasoning behind a small, explicit pool).
"""

from __future__ import annotations

from psycopg_pool import ConnectionPool

from research_agent.config.settings import DatabaseConfig


def build_connection_pool(config: DatabaseConfig, *, open: bool = True) -> ConnectionPool:
    """Builds a `ConnectionPool` for `config.url`. Raises `ValueError`
    if `config.configured` is False -- callers must check
    `DatabaseConfig.configured` themselves before ever reaching this
    function; it is never this function's job to silently no-op or fall
    back to anything.

    `open=True` (the default) opens the pool immediately and blocks
    until at least one connection is ready or `config`'s implicit
    connect timeout elapses, matching `ConnectionPool`'s own recommended
    non-lazy-open usage outside of a running event loop. Tests that want
    to assert on a not-yet-open pool (rare) may pass `open=False`.
    """
    if not config.configured:
        raise ValueError(
            "build_connection_pool() called with an unconfigured DatabaseConfig -- "
            "callers must check .configured first; there is no SQLite fallback here."
        )
    return ConnectionPool(
        conninfo=config.url,
        min_size=config.pool_min_size,
        max_size=config.pool_max_size,
        open=open,
    )
