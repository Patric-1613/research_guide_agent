"""Day 2 (public multi-user deployment foundation, see
docs/plans/public-multi-user-deployment-review.md): the production
PostgreSQL ownership store.

This package is the ONLY place `research_agent` opens a connection to
the production PostgreSQL database. Nothing outside it (and outside
`scripts/reconcile_curation_ownership.py`, which uses it) imports
`psycopg` directly.

Not wired into `research_agent.api`/`api_app/app.py`'s lifespan, any
route, or any existing service in this checkpoint -- see
`research_agent/curation_ownership.py`'s own module docstring for why
that wiring is deliberately deferred to a later day, not an oversight.

Modules:
- `pool.py` -- connection-pool construction (`build_connection_pool`),
  owned by whichever caller constructs it (a script, or a future
  `lifespan()`) -- this package never holds a module-level singleton
  pool, so nothing here can accidentally reuse a pool across an
  unrelated test's isolated database.
- `migrations.py` -- the idempotent, versioned SQL migration runner
  (`run_migrations`, `get_current_schema_version`) and its own honest
  rollback-limitations documentation.
- `migrations/*.sql` -- the actual schema, one file per version.
- `ownership_repository.py` -- `PostgresOwnershipRepository`: `users`
  and `curation_owners` CRUD.
- `saved_search_repository.py` -- the `SavedSearchRepository` protocol
  plus its SQLite and PostgreSQL implementations.
"""

from __future__ import annotations
