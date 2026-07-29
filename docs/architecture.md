# Architecture

This document describes two things: the architecture as it actually exists
today, and the target layered architecture this project is migrating toward
incrementally (see `specs/migration-plan.md` for the phased plan). Nothing in
this document changes behavior — it's a map, not a refactor.

The existing top-level `README.md` and `research_agent_architecture.svg`
describe the original single-pipeline research agent (search → dedup →
rank → summarize/chat). That description is still accurate for that part of
the system, but predates the curation/report/chat system and the React UI
added afterward — this document fills that gap and is the more current
source of truth for anything under `research_agent/curation_*`, `report.py`,
`qa.py`, and `frontend/`.

## Current architecture

**Phase 2 (API split) is complete as of 2026-07-29.** `research_agent/api.py`
is now a thin, backward-compatible entrypoint — `uvicorn research_agent.api:
app` still boots the exact same app object it always has, and every response
shape is byte-for-byte unchanged — but every actual route handler has moved
out into `research_agent/api_app/routers/`. `api.py` itself now holds:
top-level imports, shared module-level state (`_state`, populated once in
`lifespan()`), every request/response Pydantic model, a handful of shared
serialization/error-guard helpers, and 12 `app.include_router(...)` calls
that wire the routers back onto the one live `app` object. There is still no
separate service layer — orchestration logic lives in the router handlers
now, not in `api.py`'s own handlers, but it hasn't moved any further than
that yet (see "Transition debt" below).

```
frontend/ (React + Vite)
  │  fetch()
  ▼
research_agent/api.py  (thin compatibility entrypoint — imports, _state,
  │                      every Pydantic model, shared helpers, and 12
  │                      app.include_router(...) calls; no route bodies)
  ▼
research_agent/api_app/routers/  (every actual route handler now lives here
  │                                — see the full inventory below)
  ├── original one-shot pipeline
  │     /search, /summarize, /chat, /export, /library, /health
  │
  └── curation pipeline (the interactive review/report/chat system)
        /curation/start, /curation/{id}/picks, /curation/reviews,
        /curation/{id} (GET), /curation/{id} (DELETE),
        /curation/{id}/select-from-history, /curation/{id}/reopen,
        /curation/{id}/report, /curation/{id}/report/regenerate,
        /curation/{id}/chat
  │
  ▼
domain modules (research_agent/, flat, no services/agents/graphs/rag/sources split yet)
  agent.py            LangChain tool-calling agent (genuinely agentic — the
                       model itself decides arXiv vs. Semantic Scholar vs.
                       both, query reformulation, when to rerank)
  qa.py                LangGraph StateGraph: classify → condense → retrieve →
                       generate, fixed routing over plain state checks. Powers
                       BOTH /chat and (via curation_chat.py) the curation
                       chat's grounded answers.
  curation_loop.py      LangGraph StateGraph: the interactive present → pick →
                       resume interrupt loop (checkpointed, survives a page
                       refresh or server restart mid-turn)
  curation_session.py   Minimal LangGraph StateGraph (one no-op node) whose
                       only job is giving the checkpointer something to
                       persist PaperPoolSession state around; also owns
                       select_paper_from_history/reopen_curation_session/
                       delete_curation_session/list_curation_sessions
  curation_chat.py      Plain functions: offer-and-decide web-search/
                       report-update escalation on top of qa.py's ask()
  query_expansion.py    PaperPoolSession (the curation session's own state),
                       candidate-pool building/ranking/refill, canonicalize_
                       topic, suggest_related_titles
  report.py             Literature-review report generation/regeneration
  ingestion.py           search_arxiv(), search_semantic_scholar()
  dedup.py               cross-source deduplication + merge
  embeddings.py          batched + cached embedding, Chroma storage, retrieval
  ranking.py             alternative FINAL-ranking strategies for evaluation
                       only (BM25, RRF, citation-partitioned rerank) — never
                       used by the live app's default path
  enrichment.py          abstract backfill for papers missing one
  web_search.py          Tavily web search wrapper
  citations.py           APA + BibTeX formatting (deterministic, no LLM)
  summarize.py           theme clustering + grounded per-paper summaries
                       (original one-shot pipeline only)
  storage.py             SQLite persistence for the original one-shot
                       pipeline's saved searches (data/history.sqlite)
  tracing.py             shared Langfuse helpers
  schema.py              Paper / WebArticle — the normalized records shared
                       by every phase
  │
  ▼
storage
  data/chroma_db/         vector store — source of truth for paper content +
                         embeddings, keyed by paper_id, shared everywhere
  data/history.sqlite     original pipeline's saved-search records only
  data/qa_checkpoints.sqlite  LangGraph checkpointer DB — shared by qa.py's
                         (currently inactive) conversation persistence AND
                         every curation session (namespaced by thread_id
                         prefix "curation-session:")
  data/cache/             embedding cache
```

### Phase 2: API split — router inventory

Every router lives in `research_agent/api_app/routers/`, one file per
logical endpoint group (finer-grained than the six-file sketch originally
proposed in "Target architecture" below — grouping followed test-patch
boundaries and shared-helper usage rather than a fixed count decided up
front):

| Router file | Endpoint(s) |
|---|---|
| `health.py` | `GET /health` |
| `library.py` | `GET /library`, `GET /library/{search_id}` |
| `export.py` | `GET /export/{search_id}` |
| `summarize.py` | `POST /summarize` |
| `chat.py` | `POST /chat` (original one-shot pipeline) |
| `search.py` | `POST /search` (both the agent path and the `use_query_expansion=True` path) |
| `curation_core.py` | `POST /curation/start`, `POST /curation/{id}/picks` |
| `curation_sessions.py` | `GET /curation/reviews`, `GET /curation/{id}`, `DELETE /curation/{id}` |
| `curation_history.py` | `POST /curation/{id}/select-from-history`, `POST /curation/{id}/reopen` |
| `curation_reports.py` | `POST /curation/{id}/report`, `POST /curation/{id}/report/regenerate` |
| `curation_chat.py` | `POST /curation/{id}/chat` |

### Why `research_agent/api_app/`, not `research_agent/api/`

The original Phase 2 design called for converting `api.py` into a package —
`research_agent/api/app.py` plus an `__init__.py` re-exporting `app` and
every test-patched name. A review caught a real, verified bug in that plan
before any code moved: `unittest.mock.patch.object(api, "name")` mutates
whichever module a function's `__globals__` actually points to — the module
it's *physically defined in* — not whatever re-exports that name elsewhere.
A wildcard re-export (`from .app import *`) only copies names into the
package's own dict once, at import time; it doesn't alias the two dicts.
Patching `research_agent.api.chat_turn` afterward would only touch the
package's copy, while a handler physically relocated into `app.py` would
still resolve its own bare-name call against `app.py`'s dict, unaffected —
silently running the real function instead of the mock in what's supposed
to be a fully deterministic, offline test suite. Confirmed directly with an
isolated scratch reproduction before accepting the fix, not assumed.

The fix: never rename or relocate `api.py` at all. `research_agent/api_app/`
is an entirely separate, temporary package name specifically so it can
exist *alongside* the still-live `api.py` with zero import collision at
any point, ever. Every moved handler that needs a dependency tests patch
(`build_candidate_pool`, `chat_turn`, `run_research_agent`, `ask`, etc.) does
`import research_agent.api as api` and calls `api.<name>(...)` — a fresh
attribute lookup against the *original, still-patchable* module, at call
time, every call — never a `from research_agent.<module> import <name>`
binding, which would capture a reference immune to patching. Confirmed
empirically before this became policy, not assumed: a three-way scratch
test showed a relocated handler calling `api.name(...)` still sees the
patch, while one importing `name` directly does not.

### Transition debt this leaves for Phase 3

Phase 2 deliberately optimized for the smallest possible reviewable diff
per route move, not for architectural purity — the following is real,
known debt, not an oversight:

1. **Every router still calls back into `research_agent.api` as `api.<name>`
   for its shared dependencies.** This was the whole point during Phase 2
   (see above), but it means routers aren't actually decoupled from
   `api.py` yet — they're decoupled in *location* only.
2. **Every request/response Pydantic model still lives in `api.py`.** None
   were moved to a `schemas/` module — routers reference them as
   `api.SearchResponse`, `api.CurationStateResponse`, etc.
3. **Shared helpers still live in `api.py`.** `_upstream_error_guard`,
   `_state`, `_curation_config`, `_turn_result_to_response`,
   `_paper_to_out`/`_web_article_to_out`/`_paper_out_from_batch_entry`/
   `_report_to_out`/`_turn_history_out` — all of it, still in one file.
4. **Phase 3's job is specifically to resolve 1–3**: move orchestration
   logic into `services/*.py`, and move the schemas/helpers above into
   their own stable modules that routers import directly — at which point
   the `api.<name>` indirection this phase relied on stops being needed,
   since the underlying functions will have real, independent homes
   patchable in their own right, and `api.py` can shrink to just app
   construction and route registration.

### Phase 3 (service layer) — progress so far

Phase 3 is in progress, moving one route group's orchestration at a time
into `research_agent/services/`. Extracted so far:

| Service file | Function(s) | Backs |
|---|---|---|
| `library_service.py` | `list_library_items`, `get_library_item` | `GET /library`, `GET /library/{search_id}` |
| `summary_service.py` | `summarize_search`, `export_search_markdown` | `POST /summarize`, `GET /export/{search_id}` |
| `chat_service.py` | `answer_search_chat` | `POST /chat` (original one-shot pipeline only — **not** curation chat) |
| `search_service.py` | `run_search` | `POST /search` (both the agent path and `use_query_expansion=True`) |

Not yet extracted: any curation route (`curation_core.py`,
`curation_sessions.py`, `curation_history.py`, `curation_reports.py`,
`curation_chat.py` all still hold their orchestration inline).

Current shape of the stack, mid-Phase-3:

```
api_app/routers/*.py   thin route adapters: get dependencies, call one
                        service function inside api._upstream_error_guard,
                        translate a None/sentinel result into the existing
                        HTTPException where applicable
        │
        ▼
services/*.py           orchestration moved out of routers; still reaches
                        back into api.py via `import research_agent.api as
                        api` for every schema, shared helper, and any name
                        tests patch — same `api.<name>` convention Phase 2
                        established for routers, now used by services too
        │
        ▼
api.py                  still holds every Pydantic model, `_state`, and
                        every shared helper (_upstream_error_guard,
                        _get_or_create_summary, _render_markdown,
                        _paper_to_out family, curation formatting/config
                        helpers) — unchanged since Phase 2
```

Every extracted service function follows the same "return `None` (or, for
`search_service.py`, raise the existing `HTTPException` directly) on a
not-found/empty condition, let the router layer handle the HTTP status"
convention established in `library_service.py`. Three of the four
extractions so far needed one small, explicitly-flagged signature
extension beyond the shorthand originally proposed, in each case because
the current behavior genuinely depends on an extra piece of request state
that a 1- or 2-argument signature would have dropped: `summarize_search`/
`export_search_markdown` take a `style` parameter (citation style changes
the output), `answer_search_chat` takes `history` (prior turns feed the
answer), and `run_search` takes `db` (needed to persist the search and
obtain `search_id`/`created_at`).

### Transition debt still remaining (updated for Phase 3 progress)

1. **Routers and services both still reach into `research_agent.api` as
   `api.<name>`** for every schema, shared helper, and patch-target name —
   this was Phase 2's compatibility mechanism for routers, and Phase 3's
   newly-extracted services follow the identical convention rather than
   introducing a second one. Nothing is decoupled from `api.py` in
   practice yet, only in file location.
2. **`api.py` still owns every request/response Pydantic model** — none
   have moved to a `schemas/` module.
3. **`api.py` still owns shared helpers**: `_upstream_error_guard`,
   `_state`, `_curation_config`, `_turn_result_to_response`,
   `_get_or_create_summary`/`_get_or_create_web_summary`,
   `_render_markdown`, the `_paper_to_out`/`_web_article_to_out`/
   `_paper_out_from_batch_entry`/`_report_to_out`/`_turn_history_out`
   family — all of it, unchanged since Phase 2.
4. **Curation routes have no service layer yet** — `curation_core.py`,
   `curation_sessions.py`, `curation_history.py`, `curation_reports.py`,
   and `curation_chat.py` all still orchestrate inline. This is
   deliberately deferred (see "Next planned area" below) since curation
   touches significantly more shared state (checkpointer dependency
   overrides, report generation, history/reopen flows) than the four
   simpler extractions done so far.
5. **`search_service.py` raises `HTTPException` directly**, rather than
   returning a `None`/sentinel value for the router to translate — noted
   as acceptable temporary debt for this transitional phase, not the
   final ideal layering (a service module raising an HTTP-layer exception
   is exactly the coupling Phase 3 is meant to unwind elsewhere). This
   was safe to leave as-is here because `_upstream_error_guard` already
   re-raises `HTTPException` untouched, so behavior is unaffected; it's
   flagged for cleanup once the curation extraction settles the pattern
   for services with multiple distinct not-found/error branches.

### Validation recorded across Phase 3 so far

```
Step 1 (library_service)  commit ee28449  uv run pytest -q → 342 passed
Step 2 (summary_service)  commit 398022f  uv run pytest -q → 342 passed
Step 3 (chat_service)     commit d95e01d  uv run pytest -q → 342 passed
Step 4 (search_service)   commit c7bc8d3  uv run pytest -q → 342 passed
```

Pass count has stayed flat at 342 across every step — expected, since
Phase 3 (unlike parts of Phase 2) hasn't needed to close any new
coverage gaps so far, only relocate existing orchestration.

### Next planned area: curation service extraction

Curation needs its own mini-plan before extraction starts, rather than
following the same one-step-per-route pattern used above, because it
touches several things the four extractions so far didn't:
`get_curation_checkpointer` (an `app.dependency_overrides`-keyed
dependency, not a `patch.object`-based name — a different risk profile),
report generation/regeneration, the history/reopen flows, and curation
chat's own escalation logic on top of `qa.py`. This will be scoped as a
separate, explicit go-ahead, not assumed as an automatic continuation of
Phase 3 Steps 1–4.

**Update: curation service extraction is done.** All five curation route
groups now have a service module (`curation_session_service.py`,
`curation_core_service.py`, `curation_history_service.py`,
`curation_report_service.py`, `curation_chat_service.py`), completing
Phase 3's route-group extraction for every endpoint in the API. See
`specs/migration-plan.md`'s Phase 3 section for the full commit-by-commit
record. `get_curation_checkpointer` was deliberately left in `api.py`,
untouched — every curation router still declares
`Depends(api.get_curation_checkpointer)` itself, so the
`app.dependency_overrides`-keyed identity tests rely on stayed intact.

### Phase 4 (schemas + serializers) — done

Phase 4 moved the two remaining category of things Phase 3 identified as
debt (see "Transition debt still remaining" above) that hadn't yet been
given a real home: request/response Pydantic models, and pure output/
serialization/rendering helpers.

| New module | Owns |
|---|---|
| `research_agent/api_app/schemas.py` | All 28 request/response Pydantic models (`SearchRequest` … `CurationSelectFromHistoryResponse`) |
| `research_agent/api_app/serializers.py` | Every pure output/serialization/rendering helper: `_paper_to_out`, `_web_article_to_out`, `_web_articles_from_saved`, `_summary_to_json`, `_web_summary_to_json`, `_paper_out_from_batch_entry`, `_turn_history_out`, `_report_to_out`, `_turn_result_to_response`, `_render_markdown` (plus `_STYLE_LABELS`) |

`research_agent/api.py` shrank from 787 to 369 lines as a direct result.
Neither new module imports `api.py` — `schemas.py` depends only on
pydantic/typing/`citations.CitationStyle`; `serializers.py` depends on
`schemas.py` plus `research_agent.schema`/`citations` directly. No
circular imports were introduced.

**`api.py` re-exports every moved name** at its top (`from
research_agent.api_app.schemas import ...` / `from
research_agent.api_app.serializers import ...`), so
`research_agent.api.<Name>` and `patch.object(api, "<name>", ...)` keep
resolving exactly as before — the patch mechanism mutates `api.py`'s own
module dict regardless of whether a name was originally defined there or
imported in, so this required no test changes.

**Current architecture after Phase 4:**

```
api_app/routers/*.py     thin route adapters — unchanged since Phase 3
        │
        ▼
services/*.py            orchestration for search, summary/export,
                        library, regular chat, and all five curation
                        groups — unchanged since Phase 3
        │
        ▼
api_app/schemas.py        every request/response Pydantic model — NEW
api_app/serializers.py    output conversion / markdown formatting — NEW
        │
        ▼
api.py                    compatibility/composition entrypoint: app +
                        lifespan + CORS + include_router wiring, _state,
                        get_curation_checkpointer, _upstream_error_guard,
                        _curation_config, _get_or_create_summary/
                        _get_or_create_web_summary, _server_side_rerank,
                        _filtered_candidate_count, _reselect_style, and
                        re-exports of everything in schemas.py/
                        serializers.py
```

**Why routers/services still call `api.<name>` rather than importing
schemas.py/serializers.py directly** (a deliberate choice, not an
oversight): every already-migrated router and service (~15 files) was
left untouched in this same commit, for three reasons — (1) it keeps
`patch.object(api, "<name>", ...)` working for every existing test
without auditing each call site's patch sensitivity individually, (2) it
keeps Phase 4's diff a strict, low-risk relocation with zero call-site
changes elsewhere, and (3) it leaves a safe, well-scoped future cleanup:
switching a call site to import directly from `schemas.py`/`serializers.py`
is safe wherever that name is never patched in a test, and unsafe
wherever it still is — that audit is future work, not assumed here.

**Remaining debt after Phase 4:**

1. `api.py` still owns app/lifespan/CORS/router composition — no `app.py`/
   app-factory module exists yet.
2. `api.py` still owns `_state`.
3. `api.py` still owns the dependency provider `get_curation_checkpointer`
   — not moved, to preserve `app.dependency_overrides` key identity.
4. `api.py` still owns `_upstream_error_guard`, `_curation_config`,
   `_get_or_create_summary`, `_get_or_create_web_summary`,
   `_server_side_rerank`, `_filtered_candidate_count`, `_reselect_style`
   — none of these are pure output serializers (they read `_state` or
   call DB/LLM functions), so Phase 4 deliberately left them in place.
5. `search_service.py` and all four of the curation-core/history/report/
   chat services still raise `HTTPException` directly rather than
   returning a uniform sentinel for the router to translate — flagged
   since Phase 3, still true.
6. `research_agent/api_app/` is still the interim package name chosen
   specifically to avoid the `api.py`/`api/` import collision (see "Why
   `research_agent/api_app/`, not `research_agent/api/`" above) — renaming
   it to `api/` is deferred until `api.py`'s compatibility constraints are
   deliberately retired, not before.

### Phase 5 (direct schema/serializer imports) — done

Phase 5 executed the "Low risk" option from Phase 4's table above: every
router and service touched by Phases 2–3 (9 routers + 9 services) now
imports Pydantic models from `schemas.py` and pure helpers from
`serializers.py` directly, instead of via `api.<name>`. Two files
(`routers/library.py`, `services/curation_session_service.py`) had no
remaining patch-target/state/guard references after the swap, so their
`import research_agent.api as api` was removed entirely; every other
touched file keeps that import for whatever's still only reachable that
way (patch targets, `_state`, `get_curation_checkpointer`).

One reference was deliberately left as `api.<name>` outside the allowed
schema/serializer lists: `api._merge_web_articles` in `search_service.py`
— it's a domain function from `agent.py`, not a schema or a serializer,
and wasn't in Phase 5's scope, so it was left rather than silently moved.
(Phase 6, below, later gave it a real home in `search_helpers.py`.)

`api.py`'s compatibility re-exports were untouched by this phase — no
schema/serializer moved location, only who imports them changed.
Validation: `test_api.py` + `test_curation_api.py` 77 passed; full
backend suite 342 passed; frontend `npm test` 98 passed; `npm run build`
clean.

### Phase 6 (behavioral helpers extraction) — done

Phase 6 moved the remaining non-endpoint behavioral helpers out of
`api.py` into four focused modules, executing the "Medium" option from
Phase 4's table (dependency providers/error-guard helpers) plus the
summary-cache/search/curation helper groups that emerged from auditing
what was left:

| New module | Owns |
|---|---|
| `research_agent/api_app/errors.py` | `_upstream_error_guard`, `_UPSTREAM_ERRORS` — pure, no `api.py` dependency |
| `research_agent/services/summary_cache.py` | `_get_or_create_summary`, `_get_or_create_web_summary`, `_reselect_style` |
| `research_agent/services/search_helpers.py` | `_server_side_rerank`, `_filtered_candidate_count`, and `_merge_web_articles` (re-exported here from `agent.py` instead of imported directly into `api.py` — never patched either way, so patchability is unaffected by the move) |
| `research_agent/services/curation_helpers.py` | `_curation_config` |

`research_agent/api.py` shrank from 369 to 232 lines. `get_curation_checkpointer`
and `_state` were **not** moved — both stay in `api.py`, unchanged, exactly
as scoped. Every new module reaches `api.<name>`/`api._state` via `import
research_agent.api as api`, accessed only inside function bodies (never at
import time) — the same safe circular-import pattern `api_app/routers/*.py`
has relied on since Phase 2: Python only needs the (possibly
still-loading) `research_agent.api` module object to exist in `sys.modules`
at the new module's import time, and none of them touch `api.<name>`
until a function is actually called, long after `api.py` has finished
loading.

`api.py` re-exports every moved name, so `research_agent.api.<name>` and
`patch.object(api, "<name>", ...)` keep working unchanged.
`_upstream_error_guard` is never patched in any test, so the 8 routers
that use it (`chat`, `curation_chat`, `curation_core`, `curation_history`,
`curation_reports`, `export`, `search`, `summarize`) now import it
directly from `errors.py`. The other three new modules' helpers were
**not** swept into direct imports at their call sites — `search_service.py`,
`summary_service.py`, and the curation services still reach
`_server_side_rerank`/`_filtered_candidate_count`/`_get_or_create_summary`/
`_get_or_create_web_summary`/`_curation_config` via `api.<name>`, relying
on the re-export. Phase 6 only asked for a direct-import sweep on
`_upstream_error_guard`; a broader sweep for the other three modules is
listed as Phase 7 option 1 below.

**Current architecture after Phase 6:**

```
api_app/routers/*.py     thin HTTP adapters — import schemas/serializers/
                        _upstream_error_guard directly; import api only
                        for patch targets, _state, get_curation_checkpointer
        │
        ▼
services/*.py            business orchestration + helper logic — search,
                        summary/export, library, regular chat, all five
                        curation groups, plus summary_cache/search_helpers/
                        curation_helpers
        │
        ▼
api_app/schemas.py        API data contracts (28 Pydantic models)
api_app/serializers.py    pure output conversion / markdown formatting
api_app/errors.py         upstream error normalization
        │
        ▼
api.py                    composition + compatibility only: app/lifespan/
                        CORS, _state, get_curation_checkpointer, 12
                        app.include_router(...) calls, and re-exports of
                        everything in schemas.py/serializers.py/errors.py/
                        summary_cache.py/search_helpers.py/curation_helpers.py
```

**Remaining debt after Phase 6:**

1. `api.py` still owns `_state` and `get_curation_checkpointer` — neither
   moved, by design (dependency-override key identity for the latter).
2. `api.py` still owns FastAPI app construction/lifespan/CORS/router
   composition — no app-factory module exists yet.
3. `search_service.py` and the curation core/history/report/chat services
   still raise `HTTPException` directly rather than returning a uniform
   sentinel for the router to translate — flagged since Phase 3, still
   true.
4. Some service call sites still use `api.<helper>` re-exports for
   compatibility even though those helpers now live in `summary_cache.py`/
   `search_helpers.py`/`curation_helpers.py` — a deliberately narrower
   sweep than Phase 5 did for schemas/serializers (see above).
5. `research_agent/api_app/` remains the interim package name until
   `api.py`'s compatibility constraints are intentionally retired.

### Phase 7 (direct helper imports) — done

Phase 7 executed the "Low/medium" option from Phase 6's table above:
every remaining safe `api.<helper>` reference (for helpers Phase 6 moved
out of `api.py`) was replaced with a direct import from its new module.

| File | Now imports directly | Still keeps `import research_agent.api as api` for |
|---|---|---|
| `search_service.py` | `_filtered_candidate_count`, `_server_side_rerank`, `_merge_web_articles` (from `search_helpers.py`) | `expanded_search`, `run_research_agent`, `search_web` (patch targets) |
| `summary_service.py` | `_get_or_create_summary`, `_get_or_create_web_summary` (from `summary_cache.py`) | nothing — the `api` import was removed entirely |
| `curation_core_service.py` | `_curation_config` (from `curation_helpers.py`) | `_state`, `build_candidate_pool`, `rank_full_pool`, `canonicalize_topic` (patch targets/state) |
| `curation_history_service.py` | `_curation_config` (from `curation_helpers.py`) | nothing — the `api` import was removed entirely |

`_upstream_error_guard` needed no changes in this phase — every router
already imported it directly from `api_app/errors.py` as of Phase 6.

**`api.<name>` references still remaining, and why** (this is now the
complete list — nothing else is left to sweep):
1. **Patch targets**, reached via `import research_agent.api as api` in
   whichever service still calls them: `run_research_agent`,
   `expanded_search`, `search_web`, `ask`, `chat_turn`,
   `get_papers_by_ids`, `build_candidate_pool`, `rank_full_pool`,
   `canonicalize_topic`, `generate_report_for_session`,
   `regenerate_report_with_new_sources`, `generate_summary`,
   `generate_web_summary`, `semantic_search`, `embed_and_index_papers`,
   `OpenAI`, `init_db`. These stay `api.<name>` permanently (or until the
   underlying function itself moves) — importing any of them directly
   from its source module would capture a reference immune to
   `patch.object(api, "<name>", ...)`.
2. **`api._state`** — the one piece of shared mutable state every request
   handler reads from; still owned by `api.py`.
3. **`api.get_curation_checkpointer`** — the dependency-override identity
   anchor; every curation router still declares
   `Depends(api.get_curation_checkpointer)` itself, unchanged.

**Current architecture after Phase 7:**

```
api_app/routers/*.py     thin HTTP adapters — import schemas/serializers/
                        _upstream_error_guard directly; import api only
                        for patch targets, _state, get_curation_checkpointer
        │
        ▼
services/*.py            orchestration — import schemas/serializers/
                        summary_cache/search_helpers/curation_helpers
                        directly; import api only for patch targets,
                        _state, get_curation_checkpointer
        │
        ▼
api_app/{schemas,serializers,errors}.py + services/{summary_cache,
search_helpers,curation_helpers}.py    every schema, pure helper, and
                                       behavioral helper now has a real,
                                       independently-importable home
        │
        ▼
api.py                    compatibility/composition + state/dependency
                        anchor only: app/lifespan/CORS, _state,
                        get_curation_checkpointer, 12
                        app.include_router(...) calls, and re-exports of
                        every name above for anything still reaching
                        them via `research_agent.api.<name>`
```

**Remaining debt after Phase 7** (down to exactly the items Phase 6's
table flagged as not "Low/medium" risk):

1. `search_service.py` and the curation core/history/report/chat services
   still raise `HTTPException` directly rather than returning a uniform
   sentinel/typed result for the router to translate.
2. `api.py` still owns `_state`.
3. `api.py` still owns `get_curation_checkpointer`.
4. `api.py` still owns app/lifespan/CORS/router composition.
5. `research_agent/api_app/` remains the interim package name until
   `api.py`'s compatibility constraints are intentionally retired.

**Recommended Phase 8**: normalize the direct `HTTPException` usage inside
services (item 1 above) into explicit service results or small domain
exceptions that routers map to the identical existing status codes and
detail payloads — behavior-preserving, not a new error taxonomy visible
to clients. Does **not** move `_state`, `get_curation_checkpointer`, or
extract an app factory in the same phase — those stay separate, later
decisions (items 2–4 above), each needing its own explicit go-ahead per
Phase 6's risk ranking ("Medium/high" and "High").

### Validation recorded at the end of Phase 2 (2026-07-29)

```
uv run pytest -q                    → 342 passed
cd frontend && npm test             → 98 passed
cd frontend && npm run build        → clean (tsc -b && vite build)
```

Frontend structure today (already reasonably layered, see `specs/migration-
plan.md` Phase 7 for the small remaining moves):

```
frontend/src/
  App.tsx                 central orchestrator (workspace mode, routing state)
  hooks/useCurationSession.ts   the one stateful hook every component reads from
  api/client.ts, api/types.ts   typed fetch wrapper + response shapes
  components/
    ReviewMode/, ReportMode/, ChatMode/    the three workspace-mode panels
    ReviewsList/, TurnHistory/, TurnFeed/  left panel + turn scrollback/browser
    PaperPool/, WorkspaceMode/, AppHeader/, shared/
```

### Why LangGraph is used where it's used (not what makes this "agentic")

`qa.py` and `curation_loop.py` use `StateGraph` for two specific, non-agentic
reasons: **checkpointing** (SQLite-backed state that survives a restart or
refresh) and **`interrupt()`/`Command(resume=...)`** (pause execution until a
real HTTP request resumes it with the user's input). Every conditional edge
in both graphs is a plain Python function reading state you already computed
(`session.remaining() == 0`, `state["is_non_substantive"]`) — never a model
deciding the next node. `agent.py` is the one place in this codebase where
the model itself picks the next action (which tool to call, with what
arguments) — that's the actual agentic component.

## Target architecture

The direction (see `specs/migration-plan.md` for the phased path there,
which does **not** happen all at once):

```
UI  →  API  →  Services  →  Agents / Graphs / RAG / Sources  →  Storage / Repositories  →  Database
```

Concretely, for this project:

```
frontend/src/{pages,components,hooks,lib/api,types}/

research_agent/
  api/               the eventual final name -- currently realized as
                     api_app/ (see "Why research_agent/api_app/" above)
                     while api.py still needs to exist for backward
                     compatibility; renaming api_app/ -> api/ is a
                     separate future decision, not implied by Phase 3
    app.py            FastAPI app construction, middleware, lifespan
    dependencies.py    shared Depends() (checkpointer, client, db conn)
    schemas/           request/response Pydantic models, split by domain
    routers/           one file per logical endpoint group -- see the
                       router inventory table above for the current,
                       concrete list (11 files, not the 6 originally
                       sketched here; grouping followed test-patch
                       boundaries and shared-helper usage in practice)
  services/            orchestration currently living inline in api.py's
                       handlers moves here — search_service.py,
                       summary_service.py, chat_service.py,
                       curation_service.py, report_service.py
  agents/
    research_agent.py  from agent.py — unchanged logic, new location
  graphs/
    qa_graph.py         graph-building parts of qa.py
    curation_graph.py    from curation_loop.py
  rag/
    embeddings.py, ranking.py, retrieval.py
  sources/
    arxiv.py, semantic_scholar.py   (split out of ingestion.py)
    web.py                          (from web_search.py)
  db/ , storage/        repositories wrapping SQLite/Chroma access
  evals/                datasets/, runners/, evaluators/, cli.py
  config/               settings.py — env vars centralized, not hardcoded
                       across modules
  observability/        tracing.py's helpers

docs/                 architecture.md (this file), evaluation.md,
                     api-contracts.md, deployment.md, project-history.md
specs/                migration-plan.md, test-plan.md
```

The API/route surface and every response shape stay identical throughout —
this is a file-organization and dependency-direction change, not a behavior
change. Business logic (ranking math, the curation state machine, the QA
graph's routing) does not change; it only moves to a more conventional home
and gets a thin service function in front of it where orchestration logic
currently lives inline in a route handler.
