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

## Structure

```
src/
  App.tsx                        thin entrypoint — renders CurationWorkspacePage
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
