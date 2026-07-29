# Migration plan: incremental standardization

Goal: move `research_agent/` from a flat module layout toward the layered
architecture in `docs/architecture.md` (`UI → API → Services → Agents/
Graphs/RAG/Sources → Storage → Database`), without a rewrite, without
breaking any existing behavior, and without losing any existing test
coverage. Every phase below is independently reviewable and independently
revertible.

Working branch: `codex/standardize-project-structure`, off `main`.
Safety tag: `pre-standardization-2026-07-29` (created before any file was
touched — `git checkout pre-standardization-2026-07-29` returns to the exact
pre-migration state if anything here needs to be abandoned).

## Ground rules (apply to every phase)

- No phase changes user-visible behavior or API response shapes.
- No phase deletes existing functionality; where a module moves, the old
  import path keeps working via a re-export until callers are updated.
- One logical change per commit — a router move, a file split, a doc
  addition — not bundled together.
- Run the smallest relevant test subset after each change; run the full
  backend + frontend suite at the end of each phase.
- If a test fails, stop and diagnose before continuing — do not proceed
  past a red test, and do not weaken an assertion to make it pass.
- `frontend.zip` and any pre-existing modified eval CSVs are left exactly as
  found; they are not part of this migration's scope.

## Baseline (captured before Phase 0, 2026-07-29)

- Backend: `uv run pytest -q` → 340 passed.
- Frontend: `npm test` (vitest) → 98 passed, 10 test files.
- Frontend: `npm run build` → succeeds (`tsc -b && vite build`).

Any phase that ends with a different pass count must explain the delta
(expected new coverage vs. an actual regression) before moving on, same
discipline this project has used throughout its history.

## Phase 0 — Safety checkpoint (done)

1. Inspected `git status` — clean relative to `origin/main` except a
   pre-existing, already-known modification to
   `eval_results/retrieval_history.csv` and an untracked `frontend.zip`,
   neither touched by this migration.
2. Confirmed `chat-ux-fixes` (the most recent feature branch) is merged into
   `origin/main`.
3. Created annotated tag `pre-standardization-2026-07-29` on `main`.
4. Created branch `codex/standardize-project-structure` off `main`.

## Phase 1 — Documentation first, no behavior changes (done)

- Added `docs/architecture.md` (current + target architecture).
- Added `docs/project-history.md` (phase-by-phase project history — a new,
  accurate summary; nothing removed from `README.md`).
- Added this file and `specs/test-plan.md`.
- `README.md` is untouched — see "Known gap" below.

**Known gap surfaced during this phase (not fixed here):** `README.md`'s
"Project structure," "Architecture," and "Run the tests" sections describe
the project as it existed before the curation/report/chat system and the
React frontend were built — they still say "148 tests" (currently 340) and
don't mention `curation_loop.py`, `curation_chat.py`, `report.py`, or
`frontend/` at all. This is a real, pre-existing documentation gap, not
something introduced by this migration. `docs/architecture.md` and
`docs/project-history.md` are the accurate references for anything not
covered by `README.md`'s original scope. Updating `README.md` itself is
out of scope for this migration unless explicitly requested — it's core
documented history the instructions for this work say to preserve, not
rewrite.

## Phase 2 — Backend API split (behavior-preserving)

Goal: split `research_agent/api.py` (currently ~1,300 lines, every route in
one file) into a `research_agent/api/` package, one router at a time,
without changing any endpoint's behavior.

Steps:
1. Create `research_agent/api/` with `app.py`, `dependencies.py`,
   `schemas/`, `routers/`.
2. **Import-conflict check first:** Python cannot have both
   `research_agent/api.py` and `research_agent/api/__init__.py` resolve the
   same import path — `research_agent/api.py` must be removed/renamed in
   the same commit that introduces `research_agent/api/__init__.py`, not
   left alongside it. The compatibility requirement (`uvicorn research_
   agent.api:app` keeps working) is satisfied by `research_agent/api/
   __init__.py` re-exporting `app` from `app.py`, not by keeping the old
   file. This is checked for real (a real `uvicorn --help`-level import,
   not assumed) before any router is moved.
3. Move exactly one router at a time (suggested order: `health` → `search`
   → `summarize` → `chat` → `reports` → `curation`, cheapest/lowest-risk
   first). After each move: `uv run pytest tests/test_api.py tests/
   test_curation_api.py -q`, plus a live `uvicorn` boot + `curl /health`
   smoke check.
4. Do not proceed to the next router if a test fails.

Risk: this is the first phase that touches import paths — the phase most
likely to break something subtle (a route registered in the wrong order, a
`Depends()` losing its override in a test). Mitigated by moving one router
per commit and the import-conflict check above happening before any code
moves, not after.

Rollback: revert the single commit for the router that broke something;
every other already-moved router is unaffected since each is its own
commit.

## Phase 3 — Service layer

Extract orchestration currently inline in route handlers into
`research_agent/services/{search,summary,chat,curation,report}_service.py`.
Routers become thin: validate request → call one service function → return.
No algorithm changes. Add focused tests only where a service introduces
logic that wasn't independently testable before (e.g., an orchestration
sequence currently only reachable through the full HTTP round trip).

Test gate: `uv run pytest -q` (full suite) after each service is extracted,
not just the affected router's tests — a service can be called from more
than one route.

## Phase 4 — Agents, graphs, RAG, and sources organization

Move (not rewrite) modules into their layer, per `docs/architecture.md`'s
mapping table. No new agents, no multi-agent architecture — this phase is
purely about where a file lives, not what it does. Every moved module keeps
a compatibility re-export at its old path until nothing imports the old
path anymore (checked via `grep -rn` before removing the shim).

Test gate: `uv run pytest -q` (full suite) after each module move.

## Phase 5 — Config standardization

Introduce `config.yml` + `research_agent/config/settings.py`; migrate
hardcoded constants (model names, `top_k` defaults, Chroma/SQLite paths,
CORS origins, rate-limit/retry settings) gradually, one setting group per
commit. Environment variables keep working — `settings.py` reads them, it
doesn't replace `.env`. `.env.example` is not broken. Add tests asserting
each setting's default value and its env-var override.

## Phase 6 — Evals standardization

Move scattered eval logic toward `research_agent/evals/{datasets,runners,
evaluators}/` + `cli.py`. Existing `scripts/*eval*.py` keep working, either
unchanged or as thin wrappers calling the new location. `eval_results/` and
its historical CSVs are never modified or deleted by this phase. Add a
deterministic, no-API-key test gate for evaluator logic where the
evaluation itself doesn't strictly require a live model call.

## Phase 7 — Frontend structure

Introduce `frontend/src/{pages,lib/api,types}/`; move route-level views into
`pages/` gradually. No UI redesign, no behavior change. Test gate after
each move: `npm test` (vitest) and `npm run build`.

## Phase 8 — Multi-user production readiness (proposal only, not implemented)

To be proposed as its own document once Phases 2–7 are stable, covering:
user/session ownership, auth strategy, SQLite → PostgreSQL migration
path, per-user isolation, audit logs, background jobs for long-running
work, rate limiting/cost controls, Docker Compose deployment, and an
observability/tracing standard on top of the existing Langfuse
integration. Explicitly **not** implemented without separate, explicit
approval — this phase changes real behavior (auth, multi-tenancy) in a way
none of Phases 1–7 do.

## Standing risks across the whole migration

- **SQLite checkpointer file paths are load-bearing.** `data/qa_checkpoints.
  sqlite`'s thread_id namespacing (`curation-session:` prefix) is how
  curation sessions and (inactive) chat persistence coexist in one file —
  any path/config move in Phase 5 must preserve this exactly, verified by
  running `tests/test_curation_session.py` (real SQLite round-trips, not
  mocks) after the change.
- **Two LangGraph checkpointers sharing one thread_id** (`curation_loop.py`'s
  interrupt-based graph and `curation_session.py`'s sync-only graph) is a
  documented, verified-safe interaction (see `curation_session.py`'s own
  docstrings) that depends on neither graph's *shape* changing across a
  refactor — Phase 4's module moves must not change either graph's nodes/
  edges, only their file location.
- **`frontend.zip`** (45MB, untracked) is left alone throughout; it is not
  part of the source tree this migration reorganizes.
