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

Start the backend separately first (see the root `README.md`'s "Running
the app" section):

```bash
uv run uvicorn research_agent.api:app --reload --reload-exclude "frontend/*"
```

The frontend reads the backend's base URL from `VITE_API_BASE_URL`
(`src/api/client.ts`, read at call time via `import.meta.env`, not module
load time), defaulting to `http://localhost:8000` if unset. Copy
`.env.example` to `.env` to set it explicitly:

```bash
cp .env.example .env
```

## Structure

```
src/
  App.tsx                      central orchestrator (workspace mode, routing state)
  hooks/useCurationSession.ts   the one stateful hook every component reads from
  api/client.ts, api/types.ts   typed fetch wrapper + response shapes
  components/
    ReviewMode/, ReportMode/, ChatMode/     the three workspace-mode panels
    ReviewsList/, TurnHistory/, TurnFeed/   left panel + turn scrollback/browser
    PaperPool/, WorkspaceMode/, AppHeader/, shared/
```
