# Archive

Historical project artifacts kept for provenance, **not used by the
current runtime**. Nothing in this directory is read by any code, test,
build step, or dependency-resolution tool — it's a record of what was
true at some earlier point in the project's history, retained because it
might be useful context later, not because anything still depends on it.

Current, live sources of truth for what this directory's contents used
to describe:

- **Backend dependencies**: `pyproject.toml`'s `dependencies` (direct)
  and `[tool.uv] constraint-dependencies` (pinned transitive) sections,
  resolved into `uv.lock`. Install with `uv sync`.
- **Frontend dependencies**: `frontend/package.json`.

## What's here

- **`requirements-frozen-baseline.txt`** — a one-time snapshot of the
  working pip environment, captured at the point this project migrated
  its dependency management from pip/`requirements.txt` to `uv`
  (commit `54c613b`, 2026-07-16). `pyproject.toml`'s own `[tool.uv]`
  comment explains its original purpose: `constraint-dependencies` was
  pinned from it once, so `uv lock` would reproduce the same resolved
  versions instead of re-resolving fresh ones — a provenance record, not
  a live input.

  **This file is stale, not just old** — it still lists
  `streamlit==1.59.0` and `altair==6.2.2`, both removed from this
  project's actual dependency set by the `streamlit-removal` branch
  (commit `b9150fa`, 2026-07-28) — twelve days after the pip→uv
  migration this file documents. `pyproject.toml`'s current
  `constraint-dependencies` list is already clean of both. Retained here
  only for history (what the environment looked like at that one moment
  in time), not as a description of anything current.
