# Production readiness roadmap

**Status: design/audit document only. Nothing in this document is implemented.**
No auth, no OAuth, no PostgreSQL, no multi-user behavior, and no deployment
tooling exist in this codebase as of this writing. This document is Phase 18
of the project's standardization effort — a read-only audit of the current
single-user codebase plus a concrete design roadmap for what production
readiness would require, so that if/when that work starts, it starts from an
informed plan instead of an ad hoc one.

## 1. Current baseline

This codebase is a **standardized single-user, SQLite-backed, unauthenticated
local application**. Two tags mark the end of two prior standardization arcs:

- `standardized-single-user-backend` — end of Phases 0–10 (backend router →
  service → schema/serializer → app-factory extraction, zero endpoint-
  behavior change throughout).
- `standardized-single-user-project` (**current HEAD, current baseline**) —
  end of Phases 11–17 (repo hygiene, typed config for the 5 live env vars,
  documented eval workflow, standardized frontend `{pages,lib/api,types}/`
  structure, final validation checkpoint: 346 backend tests + 98 frontend
  tests passing, clean build).

Standardized: backend architecture (`api.py` → `api_app/` → `services/`),
frontend structure, eval workflow documentation, and a first increment of
typed config. **Not started, anywhere in this codebase**: authentication of
any kind, PostgreSQL (or any database other than SQLite), multi-user data
isolation, containerization, or CI. This document does not change that —
everything below is a plan to read later, not code to run now.

**This roadmap does not disturb `standardized-single-user-project`.**
Everything from here on is scoped as a separate, later arc (Phases 19+,
see §10), gated on explicit go-aheads the same way every phase in this
project's history has been.

## 2. Product definition questions

Before any implementation work in this arc, these need real answers — not
technical questions, product ones. Implementing against an unanswered one of
these would mean guessing at scope and likely rebuilding:

1. **Who are the users?** A handful of named individuals (you + a few
   collaborators)? An open self-serve signup? Internal team members only?
2. **Individual accounts, or team/workspace-based?** Does every user get
   their own private space, or do multiple users share a workspace (a lab
   group curating the same literature review together)?
3. **Can users share research sessions?** If yes: read-only share links?
   Full collaborative editing (two people picking papers in the same
   curation session concurrently)? Fork-and-own (copy someone else's
   session into your own)?
4. **What data is private by default?** Saved searches? Curation sessions?
   Chat history? All of it?
5. **What data can be global/shared across all users?** The strongest
   candidate is the Chroma paper corpus itself (see §3) — re-fetching and
   re-embedding a paper every user has already searched for is wasteful.
   Does that hold, or does the product want per-user isolation even there?
6. **Is mentor/admin access needed?** A read-only "view any user's session"
   capability for review/demo purposes, separate from being that user?
7. **Expected deployment target?** Still a single local machine (your own),
   a small shared server for a few known users, or a real multi-tenant
   hosted product? This materially changes the auth and Postgres
   recommendations in §4/§5 — a "few known people on a server I control"
   answer points toward simpler options than "public signup."

Nothing past this point should be treated as a decided answer to any of
these — §4 and §5 below give a *recommended default* for an unanswered
question, clearly labeled as a default, not a decision made on your behalf.

## 3. Current data inventory

Audited directly from `research_agent/storage.py`, `research_agent/qa.py`,
`research_agent/query_expansion.py`'s `PaperPoolSession`, and
`research_agent/embeddings.py` — not guessed.

| Data | Current storage | Schema/shape today | Needs `user_id`/`workspace_id` later? | Migration risk |
|---|---|---|---|---|
| Saved searches (one-shot pipeline) | `data/history.sqlite`, table `searches` | `id, topic, created_at, paper_ids (JSON), scores (JSON), summary (JSON), web_articles (JSON), web_summary (JSON)` | Yes — no owner column exists today | **Low.** One flat table, `ALTER TABLE ADD COLUMN` pattern already established in `init_db()` for exactly this kind of additive schema change. |
| Library items | Same table (`GET /library` reads all rows) | Same as above | Yes — `/library` currently returns *every* saved search, unfiltered | **Low**, same table. |
| Curation sessions (state) | `data/qa_checkpoints.sqlite`, LangGraph `SqliteSaver`, keyed by `thread_id` (a `curation-session:` prefixed uuid4 hex) | `PaperPoolSession` dataclass serialized as checkpoint state: `topic, reserve, cursor, seen_paper_ids, seen_titles, stage, target_count, selected_paper_ids, selected_papers, report, chat_history, web_articles_added, pending_web_offer, pending_report_update, refinement_notes, report_covered_web_article_count, display_title, turn_history, stop_reason` | Yes — no owner field; `GET /curation/reviews` lists every session in the checkpointer | **Medium.** LangGraph's checkpointer schema is managed by the `langgraph-checkpoint-sqlite` package, not hand-rolled — adding an owner concept means either (a) encoding it into the `thread_id` itself (e.g. `curation-session:<user_id>:<uuid>`, filterable by prefix) or (b) a separate SQLite table mapping `thread_id → user_id`, read alongside the checkpointer. Option (b) is safer — it doesn't touch the checkpointer library's own internals. |
| Reports | Embedded inside `PaperPoolSession.report` (same checkpointer) | `dict` (findings/limitations/future_scope sections) | Inherits from curation session's ownership | Same as curation sessions — no separate migration. |
| Chat history (one-shot `/chat`) | **Not persisted server-side at all** — client carries it in the request body | N/A | N/A — there's nothing server-side to own | None. |
| Chat history (curation chat) | Inside `PaperPoolSession.chat_history` (same checkpointer) | `list[dict]` | Inherits from curation session's ownership | Same as curation sessions. |
| Papers / web articles (content) | `data/chroma_db/` (ChromaDB), keyed by `paper_id` | Title, abstract, authors, DOI, citation count, source URLs + embedding vector | **Likely no** — see §2 question 5. This is naturally a shared, global reference corpus (the same arXiv paper looked up by two different users is the same paper); duplicating it per-user would be wasteful and is not how it's structured today. | **None if kept global.** Real risk only appears if a future product decision requires per-user paper visibility, which would be a much larger redesign than anything else in this table. |
| Embedding cache | `data/cache/embeddings.sqlite`, keyed by content hash (not `paper_id` — survives dedup merges) | `(text_hash, model) → vector` | No — pure cost-saving cache, same reasoning as the vector store above | None. |
| Enrichment cache (abstract recovery) | `data/cache/enrichment.sqlite` | DOI → recovered abstract | No — same as above | None. |
| Eval outputs | `eval_results/*.csv`, `eval_results/archive/`, `eval_results/runs/` (gitignored) | CSV run logs + per-run JSON | No — developer-facing, not product data | None; out of scope for user-data migration entirely. |

**Summary**: exactly two data types need an ownership concept —
**saved searches/library items** (one flat SQLite table, low-risk additive
change) and **curation sessions** (LangGraph-checkpointer-backed, needs an
owner-mapping table rather than touching the checkpointer schema directly).
Everything else is either already stateless (one-shot chat) or is a natural
candidate to stay global/shared (the Chroma corpus and both caches).

## 4. Auth/OAuth design options

Compared as options only — **none implemented here**.

| Option | Pros | Cons | Internship/demo suitability | Implementation risk |
|---|---|---|---|---|
| **Keep no-auth local mode** (status quo, possibly kept as a dev/demo mode alongside real auth later) | Zero new code, zero new failure modes, current test suite unaffected | Not a product — fine for exactly what it is today, a single trusted local user | **Best possible** for a solo demo/portfolio project as-is | None — this is what exists now |
| **Simple email/password** (hand-rolled: hashed passwords, session cookies or JWT) | No external dependency, full control, easiest to explain in an interview/review | Most implementation surface of any option here: password hashing, reset flows, session/token handling, email delivery for verification/reset — all security-sensitive code written and maintained by you | Medium — demonstrates real understanding of auth mechanics, but is also the option most likely to have a subtle security bug in a first pass | **Highest** of the realistic options — this is the one place "we built it ourselves" is a real liability, not just extra work |
| **Google OAuth** | No password storage/handling at all; users almost certainly already have a Google account; well-documented; free | Requires a registered OAuth app + client secret management; still need to build the callback flow, session issuance, and user-record creation yourself | Good — a single, recognizable "Sign in with Google" button is exactly what a demo audience expects | Medium — the OAuth handshake itself is standard, but session/user-record plumbing afterward is still yours to build |
| **Microsoft OAuth** | Same shape as Google OAuth; relevant if the audience is enterprise/university-Microsoft-account users | Same plumbing cost as Google; smaller likely audience overlap for a research-paper tool than Google | Good, narrower audience fit | Medium, same reasoning as Google |
| **GitHub OAuth** | Same shape again; audience overlap is strong for a developer-facing tool like this one | Narrower fit than Google for a *research* tool specifically (not every researcher has a GitHub account) | Good if the actual audience skews technical (e.g. an internship demo to engineers) | Medium, same reasoning |
| **External provider (Auth0 / Clerk / Supabase Auth)**, listed as an option only | Handles password/OAuth/session/reset/email all at once; meaningfully less custom security-sensitive code to write and maintain; usually has a real free tier for a small user count | A new external dependency and a new account/billing relationship; another service that can be down; some vendor lock-in on the auth data model | Good — fast to stand up, and "used a managed auth provider instead of reinventing it" is itself a defensible engineering decision to explain | **Lowest** of the realistic multi-provider options — least custom code, most of the hard parts (password hashing, token rotation, email delivery) are the vendor's problem, not yours |

**Recommended default (not implemented): start with Google OAuth, with the
no-auth local mode preserved as an explicit, separately-flagged dev/demo
path (not deleted).** Reasoning: it directly answers §2's likely-common case
("a handful of named individuals") with the least custom security-sensitive
code of the DIY options, doesn't require standing up a new external
account/billing relationship the way a managed-auth vendor would, and is the
single most recognizable "this is a real login" signal for a demo audience.
If §2's answers later reveal a genuinely open/public signup product, an
external provider (Auth0/Clerk/Supabase Auth) becomes the stronger
recommendation instead, specifically to avoid maintaining password/session
security code at that scale. This is a starting recommendation for Phase 20
to formally evaluate, not a decision made here.

## 5. PostgreSQL migration design options

| Option | Migration tooling | Test DB needs | Rollback | Local dev ergonomics | Data migration from existing SQLite |
|---|---|---|---|---|---|
| **Keep SQLite for now** | None needed | None — current mocked-connection test pattern (`_make_test_db_override`, an isolated temp-file SQLite DB per test) keeps working unchanged | Trivial — nothing changed | Best possible — zero setup, `uv run uvicorn` just works, exactly today's experience | N/A |
| **Add PostgreSQL only for a real production deployment, keep SQLite for local dev** | A migration tool (Alembic is the natural fit given SQLAlchemy's common pairing, though this codebase currently uses raw `sqlite3`, not an ORM — adopting one would itself be a real, separate decision) | A real Postgres instance for CI/staging; `docker-compose` becomes the natural way to give every contributor one locally without installing Postgres by hand | Requires care: a rollback from a Postgres-only production incident back to SQLite is not a live option once a deployment has been running against Postgres for any length of time — "rollback" here really means "roll back the deploy to a previous known-good Postgres schema/data state," not "revert to SQLite" | Good — local dev stays exactly as easy as today if the dev-mode default remains SQLite; only production/staging need real Postgres | One-time export/import script needed at cutover; low risk given the current schema is small (one flat `searches` table plus checkpointer state) |
| **Dual SQLite/Postgres support** (an abstraction layer choosing the backend by config) | Requires a real data-access abstraction (likely an ORM or at minimum a query-builder layer) so the same code path works against both engines — this is nontrivial given today's code uses raw parameterized `sqlite3` SQL directly, including SQLite-specific pragmas (`WAL`, `busy_timeout`) with no Postgres equivalent needed | Both a local SQLite path and a Postgres path need real test coverage, roughly doubling storage-layer test surface | Straightforward — either backend still works independently, so "roll back" just means "point config at the other one" | Best of the multi-database options — contributors without Postgres installed still have a fully working local SQLite path | Same one-time export/import as above, but only actually exercised by whoever opts into Postgres |
| **Full PostgreSQL migration** (drop SQLite entirely) | Alembic (or equivalent) migration scripts, one-time SQLite → Postgres data export/import | Every contributor needs Postgres locally (or a `docker-compose` dev setup) — no more "just `uv run uvicorn`, zero setup" | Hardest of the four — no SQLite fallback exists once this is done; a bad migration is a real incident, not a config flip | Worst of the four for local dev — a new required local dependency for every contributor | Same one-time export/import, but now it's the *only* path, not an optional one, so it needs to be genuinely well-tested before cutover, not just "good enough for the last mile" |

**Recommended safest path: "Add PostgreSQL only for production, keep SQLite
for local dev."** Reasoning: it's the option that changes local development
ergonomics the least (matters a lot for a solo/small-team internship-style
project where fast local iteration is valuable), doesn't require adopting a
full dual-backend abstraction layer's added complexity (the "dual support"
option) unless/until there's a concrete reason two backends both need to
work long-term, and keeps a full production-vs-local rollback story simple
("both environments' schemas evolve together via one migration tool," not
"two divergent code paths to keep in sync"). A full SQLite-drop migration is
explicitly *not* recommended until there's a real multi-tenant production
deployment that needs Postgres's concurrent-write characteristics — nothing
in the current single-flat-table-plus-checkpointer data model needs it yet.

## 6. Multi-user data ownership model (first-pass proposal)

Not implemented. A first-pass classification of every persisted (or
conceptually persistable) entity, to seed Phase 19's real design:

| Entity | Ownership | Notes |
|---|---|---|
| **User** | — (the root entity) | Doesn't exist today. Minimal shape: id, auth-provider identity, display name, created_at. |
| **Workspace** (if §2 answers "yes" to team/workspace-based) | — (a container users belong to) | Not needed at all if every user just gets a private space — only design this if §2's answer requires it. Deliberately not assumed here. |
| **Saved search** (one-shot pipeline) | **User-owned** (or workspace-owned, if workspaces exist) | Today: unowned, global list. See §3. |
| **Curation session** | **User-owned** (or workspace-owned) | Today: unowned, global list via the checkpointer. See §3. |
| **Report** | Inherits its curation session's ownership | Not a separate entity today — embedded in `PaperPoolSession.report`. |
| **Chat turns** (curation chat) | Inherits its curation session's ownership | Embedded in `PaperPoolSession.chat_history`. |
| **Chat turns** (one-shot `/chat`) | N/A — not persisted server-side at all | No ownership question — there's nothing to own. |
| **Papers** (title/abstract/authors/DOI/etc.) | **Global/reference** | Shared, deduplicated knowledge base — the same real-world paper looked up by two different users should be the same stored record, not two copies. This is already how `embeddings.py` is structured (keyed by `paper_id`, not by who searched for it). |
| **Web articles** | **Global/reference**, same reasoning as papers | Currently stored inline in `saved.web_articles` (JSON on the search row) rather than as their own Chroma-style shared table — worth revisiting whether they should also become a shared, deduplicated table if multi-user search volume makes redundant Tavily calls costly. |
| **Vector entries** (Chroma embeddings) | **Global/reference** | Same reasoning as papers — re-embedding the same abstract per-user would be pure waste, both in cost and in storage. |
| **Embedding cache / enrichment cache** | **Generated/cache** | Pure cost-saving caches, keyed by content hash / DOI — no user-facing meaning at all, never need an owner. |

**The load-bearing design decision this table surfaces**: papers, web
articles, and vector entries stay global/shared even in a multi-user world
— only the *search*/*curation session* records that reference them (via
`paper_ids`) need an owner. This mirrors the existing architecture's own
separation of concerns almost exactly (`storage.py`'s own docstring: "not
the papers' own content... this table's `paper_ids` are the join key back
to Chroma") — multi-user ownership is naturally an extra column on the
*join* records, not a restructuring of the shared corpus underneath them.

## 7. API impact (design only — not implemented)

Likely changes, based on the current `api_app/routers/` + `services/`
structure (see `docs/architecture.md`):

- **New auth dependency**: a `get_current_user` FastAPI dependency,
  following the exact same pattern `get_db_connection` and
  `get_curation_checkpointer` already establish (`Depends(...)` injected
  per-request). This is the natural integration point — the codebase
  already uses this pattern pervasively, so adding one more dependency is
  architecturally consistent, not a new pattern to introduce.
- **`current_user` injection** into every router handler that reads/writes
  a user-owned entity (search/library/curation routes) — 10 of the 11
  current router files (all except `health.py`).
- **Filtering queries by owner**: `list_library_items`
  (`services/library_service.py`) and `list_reviews`
  (`services/curation_session_service.py`) both currently return *every*
  row/session unfiltered — both need a `WHERE owner = ?`-shaped filter
  once ownership exists (see §3, §6).
  `get_library_item`/`get_state`/`delete_session`/etc. (single-entity
  reads/writes) all need an ownership check before returning/mutating,
  not just after finding the entity — a 404 for "exists but isn't yours"
  is the standard pattern, not a 403 (avoids leaking existence).
- **New endpoints**: at minimum, an auth callback/session endpoint (or a
  set of them, depending on the option chosen in §4), plus possibly a
  "who am I" / current-session endpoint the frontend can call on load.
- **Changed error states**: a new `401 Unauthorized` for "no valid
  session" (distinct from every existing `404`/`400`/`503` this API
  already returns cleanly via `_upstream_error_guard`/`ServiceError` — see
  `docs/architecture.md`), and the 404-not-403 ownership pattern noted
  above.
- **Backward compatibility / local dev bypass**: per §4's recommendation,
  a config-flagged "no-auth local mode" (e.g. `AUTH_ENABLED=false`,
  defaulting to `true` in anything resembling a real deployment) so local
  development and the existing test suite aren't forced to stand up real
  auth just to run. This is the single most important compatibility
  decision in this whole section — it's what lets Phases 19–23 be
  implemented and merged incrementally without breaking the existing,
  fully-tested single-user flows at any intermediate step.

## 8. Frontend impact (design only — not implemented)

- **Login/logout**: a real entry point (currently `App.tsx` renders
  `CurationWorkspacePage` unconditionally, with zero notion of "logged
  out" — see `docs/architecture.md`'s frontend section).
- **Session persistence**: today, the *only* piece of client-side session
  state is the `?session=`/`?mode=` URL query params
  (`useCurationSession.ts`, `pages/CurationWorkspacePage.tsx`) — there is
  currently **zero** use of `localStorage`/`sessionStorage`/cookies
  anywhere in `frontend/src/`. An auth token/session needs a real storage
  decision (httpOnly cookie is the safer default over `localStorage` for
  a token, given XSS exposure) — this is new frontend surface, not an
  extension of an existing pattern.
- **Protected requests**: `lib/api/client.ts`'s `request()` helper (the
  one shared fetch wrapper every `curationApi` method already goes
  through) is the single, natural place to attach an auth
  header/credential — already centralized, so this is a low-risk,
  single-file change once the token storage decision above is made.
- **User menu**: new UI, likely living near `AppHeader/` (today just a
  static header with no user-identity concept).
- **New error states**: `lib/api/client.ts`'s `ApiError` already carries
  `status`/`body` generically — a `401` needs explicit handling (redirect
  to login) distinct from how every other status is handled today.
- **Environment config**: `VITE_API_BASE_URL` stays as-is; likely one more
  env var for an OAuth client ID (if a provider from §4 is chosen) or an
  auth-provider base URL.
- **Possible onboarding flow**: first-login UX (create/name a workspace,
  if §2 answers "workspace-based") — entirely new, no current equivalent.

## 9. Testing strategy (required before implementation starts)

- **Auth unit tests**: token/session issuance and validation, in
  isolation from the rest of the app (mirrors this project's own
  established pattern — see `research_agent/config`'s own
  `tests/test_config_settings.py` for the "small, focused, new module ⇒
  new test file" convention already in use).
- **API permission tests**: for every user-owned route (§7), at minimum —
  owner can read/write their own entity; a different authenticated user
  gets 404 (not 403, not 200) on someone else's entity; an unauthenticated
  request gets 401. This directly extends the existing pattern in
  `tests/test_api.py`/`tests/test_curation_api.py`, which already test
  the current 404/400/503 error shapes exhaustively — the same rigor,
  applied to one new dimension.
- **Storage migration tests**: for the `ALTER TABLE ADD COLUMN`-style
  ownership migration on `searches` (§3) — verify pre-migration rows
  still read correctly (mirrors the exact pattern
  `tests/test_storage.py` already exercises for the existing
  `web_articles`/`web_summary` column-addition migration).
- **Frontend auth-state tests**: logged-out redirect behavior, token
  attachment on requests, 401 handling — new test file(s) alongside the
  existing `vitest` suite, following `App.test.tsx`'s established
  mocking conventions.
- **e2e smoke tests**: a real login → search → curate → logout round trip
  via Playwright, extending `frontend/e2e/` (currently one spec file,
  `full-flow-and-refresh-persistence.spec.ts`, run manually against real
  local servers — same "not automated, run manually" status this project
  has always given its "live" tests, see `docs/architecture.md`).
- **Regression tests for current single-user flows**: the entire existing
  346-test backend suite and 98-test frontend suite must stay green
  throughout — the local-mode bypass in §7 is precisely what makes this
  possible without every intermediate phase needing to build real auth
  scaffolding just to keep testing the *existing* behavior.

## 10. Phased implementation plan

None of these phases are started. Each needs its own explicit go-ahead —
same cadence this entire project has used since Phase 0.

| Phase | Goal | Files likely touched | Risk | Validation required | Rollback |
|---|---|---|---|---|---|
| **19 — Data ownership spec** | Turn §3/§6 into a concrete schema design: exact new columns/tables, the curation-session owner-mapping approach, decided (not just proposed) | None (design doc only, like this one) | None — docs only | N/A | N/A |
| **20 — Auth design finalization** | Answer §2, pick one option from §4 for real (not just "recommended default"), document the exact token/session shape | None (design doc only) | None — docs only | N/A | N/A |
| **21 — Database migration spec** | Turn §5's recommendation into a concrete migration plan: schema diffs, the export/import script's design, exact rollback steps | None (design doc only) | None — docs only | N/A | N/A |
| **22 — Config/deployment foundation** | `research_agent/config/settings.py` gains deployment-relevant settings (`AUTH_ENABLED`, database URL, etc.); introduce the first CI config and/or a `Dockerfile`/`docker-compose.yml` (none exist today) | `research_agent/config/settings.py`, new `.github/workflows/`, new `Dockerfile`/`docker-compose.yml` | Low — additive config + new deployment scaffolding, doesn't touch existing request-handling code | Full existing test suite green; a real local `docker-compose up` boot | Delete the new files; nothing existing depends on them yet |
| **23 — Auth skeleton behind a feature flag** | Implement `get_current_user`, the chosen §4 provider's login/callback flow, and the `AUTH_ENABLED=false` bypass (default) — no route yet actually *requires* auth | `api_app/routers/` (new auth router), new `services/auth_service.py`-style module, `research_agent/config/settings.py` | **Medium** — first real security-sensitive code in this codebase | New auth unit tests (§9) green; full existing suite still green with `AUTH_ENABLED=false` (default) | Flip `AUTH_ENABLED` off; the flag itself is the rollback mechanism |
| **24 — PostgreSQL support** | Add Postgres as an optional backend per §5's recommendation, config-selected; local dev stays SQLite by default | `research_agent/storage.py` (or its successor), `research_agent/config/settings.py`, new migration tooling/scripts | **Medium/high** — first change to the storage layer since the original per-request-connection work | New storage tests against both backends; full existing suite green against SQLite (unchanged default) | Config flip back to SQLite; SQLite path is untouched throughout this phase |
| **25 — Multi-user enforcement** | Wire ownership columns/filters into every route from §7's list; auth becomes required (not just skeleton) wherever `AUTH_ENABLED=true` | `services/library_service.py`, `services/curation_session_service.py`, `services/curation_core_service.py`, and every other user-data-touching service/router | **High** — this is the phase that actually changes existing single-user behavior for the first time | Full permission-test matrix from §9 green; full existing suite green with `AUTH_ENABLED=false`; manual verification with `AUTH_ENABLED=true` | This is the point past which "roll back to single-user" stops being a config flip — needs its own explicit go/no-go checkpoint before starting, not just a rollback plan after |
| **26 — Frontend auth UX** | Login/logout, user menu, protected requests, 401 handling, onboarding if needed (§8) | `frontend/src/App.tsx`/`pages/`, `lib/api/client.ts`, new `components/Auth/`-style directory | Medium — frontend-only, but touches the shared request wrapper every existing call goes through | Full frontend suite green; new auth-state tests (§9) green; manual click-through | Feature-flag the new UI behind the same `AUTH_ENABLED` signal, surfaced to the frontend via a config/health endpoint |
| **27 — Production hardening** | Rate limiting, structured logging/observability beyond the existing Langfuse tracing, secrets management for real deployment, load testing | Cross-cutting — `api_app/`, `research_agent/config/`, new deployment docs | Medium — mostly additive, but touches request-path middleware | Full suite green; a real staging deployment smoke test | Each hardening measure should land independently and be independently revertible, same one-logical-change-per-commit discipline this whole project has used since Phase 0 |

**Phase 25 is the pivot point.** Phases 19–24 are additive and gated behind
`AUTH_ENABLED=false` by default — the existing single-user product keeps
working, untouched, at every step. Phase 25 is the first phase that changes
what an unauthenticated request to a user-data route actually does, and
should be treated with the same "pause, confirm scope, validate, then
commit" discipline this project applied to every risky step in Phases 0–17 —
if anything, more of it.

## 11. Explicit non-goals (this document, Phase 18)

- **No auth was implemented in Phase 18.** No login endpoint, no session
  handling, no `get_current_user` dependency, no OAuth client registration.
- **No PostgreSQL was implemented in Phase 18.** No new database
  connection, no migration tooling installed, no schema changes.
- **No multi-user behavior was implemented in Phase 18.** No ownership
  columns added, no query filtering by user, no new tables.
- **No current single-user behavior was changed.** Every existing
  endpoint, response shape, and test result is identical to the
  `standardized-single-user-project` baseline this document was written
  against. `git diff` for this phase touches documentation only.
