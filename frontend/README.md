# Frontend

React + Vite UI for the Research Paper Summarizer's interactive curation
workflow: review a batch of candidate papers and pick which ones matter,
read/regenerate a synthesized literature-review report, and chat about
the curated set (with optional live web-context and report-update
escalation). See the repo root [`README.md`](../README.md) and
[`docs/architecture.md`](../docs/architecture.md) for the full picture,
including the original one-shot `/search`/`/summarize`/`/chat`/`/export`
endpoints this frontend does not have its own UI for (use `/docs` on the
backend directly for those).

## Commands

```bash
npm install          # first time only
npm run dev           # start the Vite dev server (http://localhost:5173)
npm test               # vitest — unit/component tests
npm run build           # tsc -b && vite build — type-check + production build
npm run e2e              # Playwright end-to-end tests
npm run lint               # oxlint
```

## Connecting to the backend

Start the backend separately first (see the root `README.md`'s "Quick
start" section):

```bash
uv run uvicorn research_agent.api:app --reload --reload-exclude "frontend/*"
```

The frontend reads the backend's base URL from `VITE_API_BASE_URL`
(`src/lib/api/client.ts`, read at call time via `import.meta.env`, not
module load time), defaulting to `http://localhost:8000` if unset. Copy
`.env.example` to `.env` to set it explicitly:

```bash
cp .env.example .env
```

## Authentication (`VITE_AUTH_MODE`)

The frontend's auth mode must match the backend's `AUTH_MODE`
(`research_agent/config/settings.py`):

| `VITE_AUTH_MODE` | Backend `AUTH_MODE` | Frontend behaviour |
| --- | --- | --- |
| unset / `disabled` | `disabled` | No sign-in. The current local-dev default; Firebase is never loaded. |
| `basic` | `basic` (or legacy `AUTH_ENABLED=true`) | No sign-in UI; the browser replays HTTP Basic-Auth credentials. Firebase is never loaded. |
| `firebase` | `firebase` | Google sign-in required. The app calls `GET /me`, shows an **"Awaiting approval"** screen until the account is approved (an operator runs `UPDATE users SET approved = true`), then renders the app with a small account / sign-out control in the header. |

In `firebase` mode the `VITE_FIREBASE_*` values in `.env.example` are
**required** — the app throws a clear startup error naming any that are
missing. They are public client-side values (from the Firebase console's
Web app config), not secrets. `VITE_FIREBASE_PROJECT_ID` must equal the
backend's `FIREBASE_PROJECT_ID` (a mismatch = every token rejected with
`401`, and the user lands on a "can't access this app" screen). Because
`vite build` inlines these at build time, a Docker image must receive
them as build args (see `docs/deployment.md` — a Day-6 task).

User flow in `firebase` mode: **Continue with Google** (popup) → the app
calls `GET /me` → an **Awaiting approval** screen while `approved` is
false (with *Check again* / *Sign out*) → the normal app once approved,
with an account / sign-out menu in the header. An operator approves an
account with `UPDATE users SET approved = true …`.

Token handling: the Firebase ID token is attached as
`Authorization: Bearer <token>` to every API call, both SSE streams, and
the report-export download, and is refreshed automatically by the Firebase
SDK. It is held **in memory only** — never in `localStorage`,
`sessionStorage`, or IndexedDB — so a page reload returns to the sign-in
screen (Google normally re-establishes the session in one click). A `401`
surfaces an honest "session expired" state; non-idempotent requests are
never silently retried.

## Structure

```
src/
  App.tsx                        thin entrypoint — <AuthProvider><AuthGate><CurationWorkspacePage/>
  lib/auth/                      VITE_AUTH_MODE contract, the Firebase-mode
                                  state machine (lazy chunk), the token/401
                                  bridge to lib/api/ (firebase mode only)
  components/Auth/               AuthGate (renders the app only once approved)
                                  + the account / sign-out header menu
  pages/CurationWorkspacePage.tsx  the app's one page: workspace-mode state,
                                  top-level layout, URL-param mode sync
                                  (no client-side router — a single-view SPA
                                  with a ?mode= query param, not multi-page
                                  routing)
  hooks/useCurationSession.ts     the one stateful hook every component reads from
  lib/api/client.ts               typed fetch wrapper — request paths, methods,
                                  payloads, error handling
  types/index.ts                  shared response/request types, mirroring
                                  research_agent/api_app/schemas.py field-for-field
  components/
    ReviewMode/, ReportMode/, ChatMode/     the three workspace-mode panels
    ReviewsList/, TurnHistory/, TurnFeed/   left panel + turn scrollback/browser
    PaperPool/, WorkspaceMode/, AppHeader/, shared/
```
