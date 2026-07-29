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

## Phase 2 — Backend API split (behavior-preserving) — DONE (2026-07-29)

Goal: split `research_agent/api.py` (was ~1,300 lines, every route in one
file) into per-endpoint-group routers, without changing any endpoint's
behavior. Completed as designed, with one significant, caught-before-
implementation revision to the original plan below.

**The plan as originally written here called for converting `api.py` into a
`research_agent/api/` package.** A design-note review (before any code
moved) found a real bug in that plan: `unittest.mock.patch.object(api,
"name")` mutates whichever module a function's `__globals__` actually
points to — the module it's *physically defined in* — not whatever
re-exports that name elsewhere. Relocating `api.py`'s content into `api/
app.py` with `api/__init__.py` wildcard-re-exporting the old names would
silently break all ~140 `patch.object(api, "<name>", ...)` calls this test
suite relies on: a wildcard re-export only copies names into the package's
own dict once, at import time, so patching `research_agent.api.<name>`
afterward would leave `app.py`'s own dict (and any handler physically
defined there) unaffected — tests would keep *passing*, just silently
exercising the real function instead of the mock. Confirmed with an
isolated scratch reproduction before accepting the revision, not assumed.
Full reasoning, plus the corrected design, is in `docs/architecture.md`'s
"Why `research_agent/api_app/`, not `research_agent/api/`" section.

**Revised and executed plan:**
1. Never rename or relocate `api.py` — it stays the live entrypoint for
   the entire phase, patchable exactly as before.
2. Created `research_agent/api_app/routers/` as an entirely separate,
   temporary package name, specifically so it can exist *alongside* the
   still-live `api.py` with zero import collision, ever.
3. Moved one router at a time (actual order: `health` → `library` →
   `export` → `summarize` → `chat` → `search` → curation, staged
   internally into 5 groups by shared-helper/patch-boundary cohesion —
   see `docs/architecture.md`'s router inventory table for the final,
   concrete 11-file list). Each moved handler reaches back into
   `research_agent.api` via `import research_agent.api as api` and calls
   `api.<name>(...)` for every dependency tests patch or that's otherwise
   still owned by `api.py` (models, shared helpers) — a fresh attribute
   lookup at call time, confirmed empirically (a three-way scratch test)
   to correctly see patches applied to the original module, unlike a
   direct `from research_agent.<module> import <name>` binding.
4. After each router move: `uv run pytest tests/test_api.py tests/
   test_curation_api.py -q` plus `uv run python -c "from research_agent.api
   import app; print(app.title)"`; full `uv run pytest -q` at both the
   `/search` move and the final curation-chat move; full frontend
   (`npm test` + `npm run build`) at the final move.
5. Two real, pre-existing test-coverage gaps were found and closed
   *before* moving the routes they covered, rather than moving first and
   hoping: `POST /search`'s `use_query_expansion=True` branch had zero
   API-level coverage (added before `/search` moved), and none of
   `_upstream_error_guard`'s 6 curation call sites had a test proving it
   actually converts a real upstream failure into a clean 503 (added
   before the curation-core group moved). Both closed with the smallest
   possible deterministic test, no existing test weakened.
6. One registration-order constraint (`GET /curation/reviews` must be
   registered before `GET /curation/{session_id}`, or Starlette matches
   `{session_id}="reviews"` first) was preserved by keeping both routes in
   the same file, in the same relative order, and verified directly (not
   just trusted) with a real `TestClient` request against the actual
   persisted dev database after that move.

Final validation: `uv run pytest -q` → 342 passed; `cd frontend && npm
test` → 98 passed; `cd frontend && npm run build` → clean.

**Transition debt intentionally left for Phase 3** (see `docs/
architecture.md` for the full list): every router still reaches into
`research_agent.api` for its shared dependencies (decoupled in location
only, not in fact); every request/response Pydantic model and every shared
helper (`_upstream_error_guard`, `_state`, `_curation_config`,
`_turn_result_to_response`, the `_paper_to_out`-family serializers) still
lives in `api.py`. Phase 3 resolves this by giving them real, independent
homes.

Rollback: revert the single commit for the router that broke something;
every other already-moved router is unaffected since each is its own
commit. Tag `pre-standardization-2026-07-29` remains the outer safety net.

## Phase 3 — Service layer (in progress, started 2026-07-29)

Directly resolves Phase 2's transition debt (see above / `docs/
architecture.md`): extract orchestration currently inline in route
handlers into `research_agent/services/{search,summary,chat,curation,
report}_service.py`, and give the request/response Pydantic models and
shared helpers (`_upstream_error_guard`, `_state`, `_curation_config`,
`_turn_result_to_response`, the `_paper_to_out`-family serializers) real,
independent homes (a `schemas`/`dependencies`-style module, not still
`api.py`) that routers and services import directly — at which point the
`import research_agent.api as api` / `api.<name>` indirection every
router currently relies on stops being necessary, since there's no longer
a single shared module whose patchability depends on staying put. Routers
become thin: validate request → call one service function → return. No
algorithm changes. Add focused tests only where a service introduces logic
that wasn't independently testable before (e.g., an orchestration sequence
currently only reachable through the full HTTP round trip).

Test gate: `uv run pytest -q` (full suite) after each service/schema/helper
is extracted, not just the affected router's tests — several are shared
across more than one route today.

**Progress so far** (one route group per step, same "propose → implement
→ validate → commit" cadence as Phase 2; full detail in `docs/
architecture.md`'s "Phase 3 (service layer) — progress so far" section):

| Step | Service | Backs | Commit | Full suite |
|---|---|---|---|---|
| 1 | `library_service.py` | `GET /library`, `GET /library/{search_id}` | `ee28449` | 342 passed |
| 2 | `summary_service.py` | `POST /summarize`, `GET /export/{search_id}` | `398022f` | 342 passed |
| 3 | `chat_service.py` | `POST /chat` (original pipeline only, not curation chat) | `d95e01d` | 342 passed |
| 4 | `search_service.py` | `POST /search` (both branches) | `c7bc8d3` | 342 passed |

Every step kept the pass count flat at 342 — no new coverage gaps
surfaced during Phase 3 so far (unlike parts of Phase 2, which closed two
pre-existing gaps before moving the routes they covered). Each service
function follows `library_service.py`'s established "return `None` (or,
for `search_service.py`, raise the existing `HTTPException` directly) on
a not-found/empty condition, let the router handle the HTTP status"
convention, and reaches shared/patched names via `import research_agent.
api as api` exactly as Phase 2's routers do — that indirection is not
resolved yet (see debt item 1 below), only relocated one layer further
in.

**Remaining transition debt** (unchanged in kind from Phase 2, now also
true of the new services, plus one item specific to this phase):
1. Routers and services both still reach `research_agent.api` as
   `api.<name>` for every schema/helper/patched name.
2. `api.py` still owns every Pydantic model.
3. `api.py` still owns every shared helper (`_upstream_error_guard`,
   `_state`, `_curation_config`, `_get_or_create_summary`/
   `_get_or_create_web_summary`, `_render_markdown`, the
   `_paper_to_out`-family serializers, `_turn_result_to_response`).
4. **Curation routes have no service layer yet** — deliberately deferred;
   see "Next: curation service extraction" below.
5. `search_service.py` raises `HTTPException` directly rather than
   returning a sentinel for the router to translate — acceptable
   temporary debt (`_upstream_error_guard` already re-raises
   `HTTPException` untouched, so behavior is unaffected), flagged for
   cleanup once the curation extraction settles a pattern for services
   with multiple distinct not-found/error branches.

**Next: curation service extraction.** This needs its own mini-plan
before it starts, rather than continuing the one-step-per-route-group
pattern above unchanged, because curation touches meaningfully more than
the four extractions so far: `get_curation_checkpointer` is an
`app.dependency_overrides`-keyed dependency (not `patch.object`-based,
a different risk profile from every name moved so far), plus report
generation/regeneration, history/reopen flows, and curation chat's own
escalation logic on top of `qa.py`. Scoped as a separate, explicit
go-ahead — not an automatic continuation of Steps 1–4.

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
