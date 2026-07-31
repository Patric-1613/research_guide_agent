# Remaining standardization plan

Audit date: 2026-07-31. Baseline: tag `standardized-single-user-backend`
(commit `635f493`), the end of the Phases 0–10 backend-architecture
migration recorded in `specs/migration-plan.md` and `docs/architecture.md`.

This document is the remaining work to standardize the *current* project
— config, evals, frontend, docs, and old-architecture cleanup — now that
the backend's internal structure is done. It does not implement anything;
every phase below needs its own explicit go-ahead, same cadence as
Phases 0–10.

## Explicit boundary

**Not in scope here, and not touched by anything below:** OAuth/
authentication, PostgreSQL migration, multi-user support. This audit
does not design, plan in detail, or take any step toward any of those —
they remain exactly what `specs/migration-plan.md`'s Phase 8 ("Multi-user
production readiness") already says: proposal-only, a separate document,
not implied by anything here. Nothing in this plan should be read as
preparing for them ahead of time.

## Phase 12 status update (2026-07-31)

Completed, docs/config-template only, no executable code touched:

- **Config Phase A** — `.env.example` now documents `FRONTEND_ORIGIN` and
  `OPENALEX_MAILTO`, matching the existing entries' comment style.
- **README targeted fixes** — the `ranking.py` contradiction is resolved
  (see §4 below for the confirmed-by-grep true status: citation-partitioned
  reranking is live via `agent.py`'s rerank tool; BM25/hybrid remain
  eval-only); the stale "148 tests" claim in "Run the tests" is replaced
  with the current backend/frontend counts and all four test/build/e2e
  commands.
- **README modernization** — a new intro paragraph plus expanded "Project
  structure" listing now mention the curation loop, curation chat, report
  generation/regeneration, the React frontend, and the standardized
  backend structure, with pointers to `docs/architecture.md`/`specs/
  migration-plan.md` rather than duplicating their detail inline.
- **`frontend/README.md`** — rewritten with real project content
  (what it's for, commands, backend connection, `VITE_API_BASE_URL`,
  structure), replacing the unmodified Vite scaffold template.
- **`frontend.zip` guidance** — added to `.gitignore` (not deleted; see
  §6 below, unchanged recommendation: confirm safe, then remove in a
  separate cleanup action).

Still pending as of Phase 12: eval archive reorganization (§2), Config
Phase B / typed settings (§1), frontend structure cleanup (§3), eval
standardization (§2), and the always-out-of-scope auth/Postgres/
multi-user work.

## Phase 13 status update (2026-07-31)

Completed, files-only, no executable code touched:

- **`frontend.zip` cleanup — done.** Confirmed untracked, `.gitignore`d,
  and referenced by nothing outside this plan's own recommendation, then
  deleted from the working tree. §6 below is now resolved, not just
  guarded against.
- **Eval archive reorganization — done.** The 4 manual snapshot CSVs
  named in §2 (`retrieval_history_pre_ranking_experiment.csv`,
  `retrieval_history_pre_citation_partition.csv`,
  `history_throwaway_smoketest.csv`, `history_two_metric_run1.csv`) moved
  into new `eval_results/archive/` via `git mv` (history preserved), with
  a new `eval_results/archive/README.md` explaining each one, confirming
  neither eval harness reads from `archive/`, and documenting
  `latency_history.csv`'s continued no-reproducing-script gap.
  `eval_results/retrieval_history.csv` and `eval_results/history.csv`
  (the current, actively-appended-to running logs) were **not** moved —
  they remain the canonical eval output exactly as §2 always intended.
  `latency_history.csv` was also **not** moved (still directly referenced
  by root `README.md`'s "Search-call parallelization" section by its
  current path; moving it would break that link, which this files-only
  phase didn't touch) — its status is now documented in
  `eval_results/archive/README.md` instead.

Still pending: Config Phase B / typed settings (§1), frontend structure
cleanup (§3), eval standardization (§2's remaining `latency_history.csv`
reproducibility decision), and the always-out-of-scope auth/Postgres/
multi-user work.

## Phase 14 status update (2026-07-31)

Completed, code-touching but behavior-preserving:

- **Config Phase B — done (first increment).** Added `research_agent/
  config/{__init__,settings}.py`: a frozen `Settings` dataclass plus
  uncached `get_settings()`, centralizing the 5 env vars this codebase's
  own code reads directly — `SEMANTIC_SCHOLAR_API_KEY`, `UNPAYWALL_EMAIL`,
  `TAVILY_API_KEY`, `FRONTEND_ORIGIN`, `OPENALEX_MAILTO` — across 6 call
  sites. Same names, same defaults, same falsy-becomes-`None` handling;
  `get_settings()` re-reads `os.environ` on every call (no caching) so
  existing `patch.dict(os.environ, ...)`-based tests keep working
  unchanged; `.env` loading unchanged. New `tests/test_config_settings.py`
  (4 tests) added — no existing test edited.
- **Intentionally not centralized in this increment**: `OPENAI_API_KEY`/
  `LANGFUSE_*` (SDK-managed — nothing in this codebase reads them
  directly; the OpenAI/Langfuse SDKs read them from `os.environ`
  internally); the 8 model-name constants (`EMBEDDING_MODEL`,
  `SUMMARY_MODEL`, `AGENT_MODEL`, `TITLE_SUGGESTION_MODEL`,
  `CANONICALIZE_TOPIC_MODEL`, `CONDENSE_MODEL`, `ANSWER_MODEL`,
  `REPORT_MODEL`); the 4 data/cache/Chroma path constants (`DATA_DIR`,
  `DB_PATH`, `CHROMA_PERSIST_DIR`, `QA_CHECKPOINT_DB_PATH`) — none of
  these are read from the environment today, so centralizing them would
  touch import structure across roughly 8 domain modules for zero
  behavior difference. No OAuth/auth/PostgreSQL/multi-user settings were
  introduced.

Validation: `tests/test_config_settings.py` 4 passed; `test_api.py` +
`test_curation_api.py` 77 passed; full backend suite 346 passed (342
baseline + 4 new); frontend `npm test` 98 passed; `npm run build` clean;
app confirmed booting under a completely clean shell environment
(`env -i`); `GET /health` 200; CORS preflight `access-control-allow-origin`
unchanged at the default.

**Remaining config debt**: decide later whether/when to centralize the
model-name constants and filesystem paths above; leave SDK-managed
secrets (`OPENAI_API_KEY`, `LANGFUSE_*`) alone unless/when a future
deployment-config need requires routing them through `Settings` too.

**Recommended next chunk: eval standardization** — define the canonical
eval commands (already documented informally in root `README.md`),
document current eval outputs (`eval_results/`'s current-vs-archived
split from Phase 13), add an eval README/spec section, clarify
generated-vs-tracked artifacts, and resolve `latency_history.csv`'s
reproducibility gap — with no model/data behavior changes initially,
matching `specs/migration-plan.md`'s existing Phase 6 outline.

---

## 1. Config audit

### What exists today

Environment variables are read via bare `os.getenv(...)` calls scattered
across 7 files, each independently, with no central settings module:

| Variable | Read in | Documented in `.env.example`? |
|---|---|---|
| `OPENAI_API_KEY` | implicitly via `OpenAI()` client construction | Yes |
| `SEMANTIC_SCHOLAR_API_KEY` | `search_service.py`, `curation_core_service.py` | Yes |
| `UNPAYWALL_EMAIL` | `enrichment.py` (×2) | Yes |
| `TAVILY_API_KEY` | `web_search.py` | Yes |
| `LANGFUSE_PUBLIC_KEY`/`SECRET_KEY`/`BASE_URL` | `tracing.py` (via Langfuse SDK) | Yes |
| `FRONTEND_ORIGIN` | `api_app/app.py` (CORS `allow_origins`) | **No** |
| `OPENALEX_MAILTO` | `curation_helpers.py`, `curation_core_service.py` | **No** |
| `VITE_API_BASE_URL` | `frontend/src/api/client.ts` | Yes (`frontend/.env.example`) |

`load_dotenv()` is called independently in 22 separate files (`api.py`
plus 21 `scripts/*.py` files) — one call each, no shared bootstrap.

Hardcoded model-name constants, one per module, no shared registry:
`EMBEDDING_MODEL` (`embeddings.py`), `SUMMARY_MODEL`/`REPORT_MODEL`
(`summarize.py`/`report.py`, both `"gpt-4.1"`), `AGENT_MODEL`
(`agent.py`, `"openai:gpt-4.1-mini"`), `TITLE_SUGGESTION_MODEL`/
`CANONICALIZE_TOPIC_MODEL` (`query_expansion.py`, both `"gpt-4.1-mini"`),
`CONDENSE_MODEL`/`ANSWER_MODEL` (`qa.py`, `"gpt-4.1-mini"`/`"gpt-4.1"`).
Several of these are the *same* literal string duplicated across files
rather than one shared constant.

Hardcoded data paths, each independently computed via the same pattern:
`DATA_DIR = Path(__file__).resolve().parent.parent / "data"` is
redefined separately in both `enrichment.py` and `embeddings.py`;
`storage.py`'s `DB_PATH` and `qa.py`'s `QA_CHECKPOINT_DB_PATH` each
compute their own `.../  "data" / "<file>"` path independently rather
than deriving from one shared `DATA_DIR`.

One confirmed manually-synced duplicate: `scripts/ragas_eval.py`'s
`STAGE1_TOP_K = 10` is a hand-copied magic number matching
`SearchRequest.top_k`'s default (`schemas.py`, `Field(default=10, ...)`)
— the script's own comment says this was checked via `grep`, i.e. the
author already knows it's a manual sync, not a shared value.

### Risks

- `FRONTEND_ORIGIN` and `OPENALEX_MAILTO` are real, live env vars with no
  entry in `.env.example` — a fresh clone has no discoverable way to know
  either exists without reading source.
- Model-name duplication means bumping a model version requires finding
  every literal occurrence by hand; a partial update silently leaves the
  app running two different models for what's meant to be one policy.
- `STAGE1_TOP_K`'s manual sync to `SearchRequest.top_k` will silently
  drift the next time either value changes without someone remembering
  to check the other.
- 22 independent `load_dotenv()` calls is redundant but low-risk (idempotent,
  cheap) — noted for completeness, not a correctness bug.

### Recommended target structure

```
research_agent/config/
  settings.py   one settings object/module reading every env var above
                once, with documented defaults; model-name constants
                centralized here (or re-exported from settings so
                existing module-level constants become thin aliases,
                not a breaking rename)
```

Environment variables keep working exactly as today — `settings.py`
reads them, it doesn't replace `.env`/`.env.example`. Every setting gets
a default plus an env-var override, and a test asserting both. This
mirrors `specs/migration-plan.md`'s already-existing Phase 5 ("Config
standardization") outline — this audit doesn't change that plan, it
confirms it's still the right shape and adds the concrete inventory
above.

### Proposed next implementation phase

**Config Phase A (low risk, additive only):** add `FRONTEND_ORIGIN` and
`OPENALEX_MAILTO` to `.env.example` with the same doc-comment style as
the existing entries. Pure documentation, zero code change, safe to do
immediately and independently of everything else in this plan.

**Config Phase B (medium risk):** introduce `research_agent/config/
settings.py`, migrate one setting group at a time (start with the model
constants, since they're pure string values with no behavioral branching),
verified by a test per setting asserting default + env-override, exactly
as `specs/migration-plan.md`'s Phase 5 already specifies.

---

## 2. Eval audit

### What exists today

Two real-pipeline (non-mocked, billable) evaluation harnesses in
`scripts/`, each with its own CSV history log in `eval_results/`:

| Harness | History log | Per-run artifact |
|---|---|---|
| `scripts/eval_retrieval.py` | `eval_results/retrieval_history.csv` | none (CSV row only) |
| `scripts/ragas_eval.py` | `eval_results/history.csv` | `eval_results/runs/run_<id>.json` + `raw_<timestamp>.jsonl` |

Both are documented, repeatable commands (`uv run python scripts/
eval_retrieval.py --note "..."` / `uv run python scripts/ragas_eval.py
--note "..."`), not ad hoc — README's "Retrieval ranking experiments"
and "RAGAS quality evaluation" sections describe them in detail and are
current.

`eval_results/` also contains 5 other files not produced by a documented,
repeatable command found in `scripts/`:

- `eval_results/history_throwaway_smoketest.csv`,
  `eval_results/history_two_metric_run1.csv` — look like manually-renamed
  snapshots of `ragas_eval.py`'s `history.csv` at a point in time (same
  column shape), likely kept as before/after comparison points.
- `eval_results/retrieval_history_pre_citation_partition.csv`,
  `eval_results/retrieval_history_pre_ranking_experiment.csv` — same
  pattern for `eval_retrieval.py`'s history, and are in fact referenced
  as intentional comparison snapshots by name (`pre_citation_partition`,
  `pre_ranking_experiment` match the two named experiments in README's
  "Retrieval ranking experiments" section).
- `eval_results/latency_history.csv` — referenced directly by name in
  README's "Search-call parallelization" section ("Full per-topic data
  ... is in `eval_results/latency_history.csv`"), but **no script in
  `scripts/` produces this file** — grepping the whole repo for its
  column headers (`elapsed_s`, `rate_limited_calls`) or filename finds
  only the README reference and the CSV itself. It was committed
  alongside the commit documenting that latency win, but the measurement
  script that produced it was apparently never committed.
- `eval_data/reference_topics.xlsx` is `.gitignore`d (binary, not
  diff-friendly) with `eval_data/reference_topics.json` tracked as its
  reviewable, converted output via `scripts/convert_reference_sheet.py`
  — this one's already correctly handled, no action needed.

`eval_results/retrieval_history.csv` is the file that has shown up as a
"pre-existing, known, modified" item throughout the entire Phases 0–10
migration (every phase's validation step noted it and left it alone) —
it's a tracked, append-only run log; every real eval run adds a row,
which is why `git status` shows it modified locally after any local eval
run. This is expected, working-as-designed behavior, not debris.

### Problems

1. `eval_results/latency_history.csv` is not reproducible from anything
   currently in `scripts/` — if it needs regenerating (e.g. re-measuring
   after a future change), there's no committed tool to do it with.
2. The two `_pre_*`/two `_throwaway`/`_run1`-named snapshot files are
   real historical artifacts but have no doc explaining the naming
   convention or that they're intentional archives rather than clutter —
   a future contributor has no way to tell without asking.
3. No `.gitignore` entry or convention distinguishes "the current running
   history log" from "an archived snapshot" — both live as plain `.csv`
   files in the same flat directory.
4. No single documented "run all evals" command — `README.md` documents
   each harness separately, correctly, but there's no `scripts/` entry
   point that runs both in sequence for a full pre-release check.

### Proposed standardized eval structure

```
eval_results/
  retrieval_history.csv     current running log (eval_retrieval.py)
  history.csv               current running log (ragas_eval.py)
  runs/                     ragas_eval.py's per-run JSON/JSONL artifacts
  archive/                  the 4 named snapshot CSVs, moved here with a
                            short README explaining what each snapshot
                            captures and why it was kept
```

`latency_history.csv` either gets a small, committed regeneration script
(`scripts/eval_latency.py`, mirroring the measurement described in
README's "Search-call parallelization" section) or an explicit note in
`docs/project-history.md` that it's a one-time historical measurement,
not a repeatable eval — whichever is true; this audit did not find the
original measurement code to confirm which.

### Next implementation phase

**Eval Phase A (near-zero risk):** move the 4 snapshot CSVs into
`eval_results/archive/` with a short `eval_results/archive/README.md`
explaining each one, and add a one-line doc note for `latency_history.csv`'s
provenance. This is a file move + a new small doc file — **out of this
audit's own allowed-changes list**, so it is not done here; flagged as
the smallest, safest first step of a future eval-standardization phase.

**Eval Phase B:** decide whether `latency_history.csv` gets a committed
regeneration script or a "historical, not repeatable" label, and act on
whichever is true.

---

## 3. Frontend audit

### What exists today

```
frontend/src/
  App.tsx                      central orchestrator (workspace mode, routing state)
  hooks/useCurationSession.ts   the one stateful hook every component reads from
  api/client.ts, api/types.ts   typed fetch wrapper + response shapes
  components/
    ReviewMode/, ReportMode/, ChatMode/     three workspace-mode panels
    ReviewsList/, TurnHistory/, TurnFeed/   left panel + turn scrollback/browser
    PaperPool/, WorkspaceMode/, AppHeader/, shared/
  test/                         test setup/helpers
```

API client (`frontend/src/api/client.ts`) reads `VITE_API_BASE_URL` at
call time via `import.meta.env` (not module load time — confirmed by its
own comment), with a `http://localhost:8000` fallback; `frontend/.env`/
`frontend/.env.example` both set it explicitly. This is already
standardized — one env var, one read site, tested directly
(`client.test.ts` stubs it via `vi.stubEnv`).

Build/test commands (`frontend/package.json`): `dev` (vite),
`build` (`tsc -b && vite build`), `test` (vitest run), `test:watch`,
`lint` (oxlint), `e2e` (playwright), `preview`. All are documented and
already used throughout this migration's validation steps (`npm test` /
`npm run build`, 98 passed / clean, every phase).

### Risks/gaps

1. **`frontend/README.md` is still the unmodified Vite scaffold template**
   (React Compiler / Oxlint-config boilerplate) — zero project-specific
   content. The root `README.md`'s "Project structure" section just says
   "see `frontend/README.md`" for the frontend, pointing a reader at
   generic template text with no actual guidance for this project.
2. **`frontend.zip`** (45MB, untracked, not `.gitignore`d) is a full
   zip archive of the `frontend/` directory *including `node_modules/`*
   and macOS `__MACOSX/` junk entries, dated after the frontend's own
   git history — clearly an ad hoc manual backup, not referenced by any
   build/test/deploy step or by any other file in the repo. See §6 below.
3. No documented error/loading-state convention audit was done at a
   component level in this pass (out of this audit's time-box) — a
   future frontend-standardization phase should look at whether
   `ReviewMode`/`ReportMode`/`ChatMode` handle loading/error states
   consistently with each other, not just whether they compile/test
   green.
4. `frontend/src/` otherwise already matches `docs/architecture.md`'s
   "Target architecture" frontend sketch reasonably well (per that
   doc's own note) — the remaining gap named there (`{pages,lib/api,
   types}/` reorganization) is `specs/migration-plan.md`'s existing
   Phase 7 ("Frontend structure"), not a new finding from this audit.

### Proposed frontend standardization phase (not implemented here)

1. Rewrite `frontend/README.md` with real project content: what this
   frontend is, `npm install`/`npm run dev`/`npm test`/`npm run build`/
   `npm run e2e` commands, the `VITE_API_BASE_URL` env var, and a short
   pointer to `docs/architecture.md` for the fuller picture — mirroring
   how the root `README.md` documents the backend.
2. Execute `specs/migration-plan.md`'s existing Phase 7 (introduce
   `frontend/src/{pages,lib/api,types}/`, move route-level views into
   `pages/` gradually) — already planned, not redesigned here.
3. Resolve `frontend.zip` per §6's recommendation.

---

## 4. README/docs audit

### What exists today

- `README.md` (690 lines) is extremely detailed and mostly accurate for
  the *original* single-pipeline agent (phases 1–7 of the pre-curation
  system) and its later robustness/ranking/latency/RAGAS work — but:
  - Says **"148 tests"** in "Run the tests"; the actual count is 342
    backend + 98 frontend (already flagged as a known gap back in
    Phase 1 of this migration, still true and now larger).
  - Never mentions `curation_loop.py`, `curation_chat.py`,
    `curation_session.py`, `report.py`, or the curation/report/chat
    system at all — a reader would have no idea from `README.md` alone
    that the app is anything beyond one-shot search/summarize/chat.
  - Describes `api.py` as *the* FastAPI backend holding the routes
    directly ("`api.py` (FastAPI) `/search /summarize /chat /export
    /library`" in the architecture diagram, "Project structure" listing
    `api.py` with no mention of `api_app/`/`services/`) — stale after
    Phases 2–10; `docs/architecture.md` is the accurate reference for
    this now, but `README.md` doesn't point to it.
  - **Contains an internal self-contradiction**: the "Project structure"
    section says `ranking.py` is "never used by the live app's default
    path," while the later "Retrieval ranking experiments" section (same
    file) explicitly documents citation-partitioned reranking being
    "promoted to the live agent's default path" and used by `agent.py`'s
    `rerank_by_relevance_tool` on every real run. Both sentences are in
    the current file; they disagree with each other.
- `docs/architecture.md` (924 lines) and `specs/migration-plan.md`
  (601 lines) are thorough and current for the backend structure —
  together they're the accurate source of truth `README.md`'s own
  Phase-1-era note already pointed to, and this audit confirms that's
  still correct and sufficient; no gap found in either doc for backend
  architecture coverage.
- `docs/project-history.md` and `specs/test-plan.md` exist and were not
  found stale in this pass (not deeply re-audited line-by-line here —
  flagged as in-scope for a future docs-cleanup phase's review, not a
  known problem today).
- No CI config exists anywhere in the repo (no `.github/workflows/`) —
  noted as a gap, not a defect; validation throughout this whole
  migration has been manual (`uv run pytest -q`, `npm test`, `npm run
  build`), which works but isn't enforced automatically on push/PR.

### What README should become

- **Quickstart**: setup + running the app (already present and
  accurate — keep as-is).
- **Architecture summary**: a short, current paragraph plus a link to
  `docs/architecture.md` for the full picture, replacing the stale
  inline description of `api.py` as monolithic.
- **Test commands**: updated counts (backend + frontend), plus
  `npm run e2e` (currently undocumented in the root README).
- **Eval commands**: already present and accurate (retrieval + RAGAS
  sections) — keep.
- **Development workflow**: a short section on the curation/report/chat
  system's existence (even just a paragraph + link to
  `docs/architecture.md`), since it's currently invisible from
  `README.md` entirely.

### README update plan (not implemented here)

1. Fix the `ranking.py` self-contradiction directly (smallest, safest
   fix — a factual correction, not a rewrite).
2. Update the test count and add the curation/report/chat system's
   existence with a link to `docs/architecture.md`, rather than
   duplicating that document's detail inline.
3. Replace the "Project structure" section's stale `api.py`-only sketch
   with the current layered summary (or a shorter version + link).
4. Update `frontend/README.md` per §3.

### Docs cleanup plan (not implemented here)

- Line-by-line re-audit of `docs/project-history.md` and `specs/
  test-plan.md` against current test counts/structure (not done in this
  pass — time-boxed out).
- Once README is updated, cross-check `docs/architecture.md`'s own
  opening paragraph (which currently describes README as accurate for
  "that part of the system" pre-curation) still describes the
  post-update README correctly.

---

## 5. Old architecture cleanup audit

### What was checked

Grepped the entire backend (`research_agent/`, `scripts/`, `tests/`) for
any remaining reference to `research_agent.api` to find dead/orphaned
usage of the compatibility surface:

- `research_agent/api_app/{schemas,serializers,errors,runtime}.py` —
  confirmed **zero** code dependency on `api.py` (only docstring/comment
  mentions of the module name, not imports) — exactly as designed in
  Phases 4/6/9.
- `research_agent/api_app/routers/*.py` and `research_agent/services/*.py`
  — every remaining `import research_agent.api as api` is for a
  currently-live reason (a patch target, `_state`, or
  `get_curation_checkpointer`) — none found dangling.
- `scripts/test_api.py` — a live, intentional manual smoke-test script
  that imports `research_agent.api as api_module` to exercise the real
  app; this is exactly what the compatibility entrypoint is for, not
  dead code.
- `scripts/ragas_eval.py` — only docstring/comment mentions of
  `research_agent/api.py`, no actual import.
- No file anywhere still imports a name from an old, pre-migration
  location that no longer exists — every import path found resolves to
  either `api.py`'s current re-exports or the new `api_app/`/`services/`
  modules directly.

### What can be safely deleted now

**Nothing.** No dead code, no orphaned old-location imports, and no
unused compatibility re-export were found. Every name `api.py` re-exports
is still reached by at least one router, service, or test via
`research_agent.api.<name>` or `patch.object(api, "<name>", ...)`.

### What must remain

- Every compatibility re-export in `api.py` (schemas, serializers,
  errors, runtime objects, patch-targeted domain function imports,
  `create_app`) — each still has live callers.
- `research_agent/api_app/` as the package name — still intentionally
  interim (see `docs/architecture.md`'s "Why `research_agent/api_app/`,
  not `research_agent/api/`" section), and renaming it requires retiring
  `api.py`'s compatibility surface first, which nothing in this audit
  found a reason to do yet.
- `research_agent.api:app` as the public ASGI entrypoint.

### What needs another phase (not this audit)

Determining *when* it's safe to start retiring `api.py`'s compatibility
re-exports (e.g., updating every test to import from `api_app/`/
`services/` directly, then removing the re-export) is real future work,
but it's a test-suite-wide change this audit's read-only, docs-only scope
explicitly excludes. Flagged here as a named future decision, not
scheduled.

---

## 6. Handling pre-existing local items

### `eval_results/retrieval_history.csv`

**Recommendation: keep tracked, keep as-is.** This is the working,
append-only history log `scripts/eval_retrieval.py` writes to on every
real run — its "modified" status in `git status` throughout this entire
migration has always been expected (a real local eval run adds a row),
not debris. No action needed; already correctly handled by every phase's
ground rule ("left exactly as found; not part of this migration's
scope").

### `frontend.zip`

**Recommendation: remove from the working tree, add to `.gitignore`.**
Findings: 45MB, untracked, not currently `.gitignore`d, contains a full
copy of `frontend/` *including `node_modules/`* plus macOS
`__MACOSX/` archive-utility junk entries, dated after the frontend's own
git history, and referenced by nothing in the build/test/deploy path or
any tracked file. This reads as an ad hoc manual backup someone made
locally, not a build artifact or a fixture any test depends on.

**Update (Phase 12, 2026-07-31):** `frontend.zip` added to the root
`.gitignore`, so it can no longer be committed by accident. Not yet
deleted at that point — flagged as a separate, explicit cleanup action.

**Update (Phase 13, 2026-07-31): done.** Re-confirmed untracked,
`.gitignore`d, and referenced by nothing outside this plan's own
recommendation, then deleted from the working tree. Nothing under
`frontend/` itself was touched.

---

## Recommended phase sequence

Each remaining item needs its own explicit go-ahead, in whatever order
is preferred — they're independent of each other except where noted.

1. ~~**Config Phase A** (near-zero risk) — add `FRONTEND_ORIGIN`/
   `OPENALEX_MAILTO` to `.env.example`.~~ **Done (Phase 12).**
2. ~~**README fix** — correct the `ranking.py` self-contradiction and the
   stale test count.~~ **Done (Phase 12).**
3. ~~**`frontend.zip` cleanup** — confirm safe, remove, `.gitignore`
   it.~~ **Done (Phase 13).**
4. ~~**README/docs update** — quickstart/architecture-summary/
   test-commands/eval-commands/curation-system-mention pass, plus
   `frontend/README.md` rewrite.~~ **Done (Phase 12).**
5. ~~**Eval archive reorganization** — move the 4 snapshot CSVs into
   `eval_results/archive/` with an explanatory README.~~ **Done
   (Phase 13).** `latency_history.csv`'s reproducibility gap remains
   documented but unresolved — see `eval_results/archive/README.md`.
6. ~~**Config Phase B** — `research_agent/config/settings.py`, one
   setting group at a time.~~ **Done, first increment (Phase 14):** the
   5 live env vars centralized. Remaining increments (model names,
   `top_k` defaults, Chroma/SQLite paths, rate-limit/retry settings) are
   still pending, each its own explicitly-scoped step.
7. **Frontend structure** — `specs/migration-plan.md`'s existing
   Phase 7 (`{pages,lib/api,types}/`). **Pending.**
8. **Eval standardization** — `specs/migration-plan.md`'s existing
   Phase 6 (`research_agent/evals/{datasets,runners,evaluators}/` +
   `cli.py`); could also resolve `latency_history.csv`'s missing
   reproducing script as part of this. **Pending.**

**Recommended next single step: Eval standardization** (item 8) —
define the canonical eval commands, document current eval outputs, add
an eval README/spec, clarify generated-vs-tracked artifacts, and resolve
`latency_history.csv`'s reproducibility gap, with no model/data behavior
changes initially. Frontend structure (item 7) is independent and can be
taken up in either order.
