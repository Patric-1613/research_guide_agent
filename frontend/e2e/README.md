# End-to-end tests

These drive a real browser against a real, running FastAPI backend and a
real Vite dev server — no mocks. They make real arXiv/Semantic Scholar/
OpenAI/Tavily calls, so they cost real time and (small) real money, and
they write real rows into the local checkpointer DB
(`data/qa_checkpoints.sqlite`) the same way manually clicking through the
app would.

## Running locally

From the repo root, in one terminal:

```
uv run uvicorn research_agent.api:app --port 8000
```

In a second terminal, from `frontend/`:

```
npm run dev
```

In a third terminal, from `frontend/`:

```
npm run e2e
```

`playwright.config.ts` points at `http://localhost:5173` by default
(override with `E2E_BASE_URL`); it does not start either server for you.

## Research Lanes specs (RL5/RL6)

- `research-lanes-mocked.spec.ts` + `../playwright.mocked.config.ts` —
  **fully mocked**, zero network/provider calls, no backend needed
  (`webServer` starts its own Vite on :5174 and every `/curation/*`
  response is `page.route`-intercepted). Validates the 11-point lane UI
  journey at desktop (1280×800) and narrow mobile (375×667). Safe to run
  anytime: `npx playwright test --config=playwright.mocked.config.ts`.
  One test is `test.skip`ped on the mobile project — the active-review
  workspace needs the desktop layout because the app shell keeps a fixed
  288-px review sidebar (a pre-existing whole-app constraint); the New
  Review lane editor lives in that sidebar and *is* covered at 375 px.
- `research-lanes-live.spec.ts` + `../playwright.live.config.ts` — the
  **one approved live journey** (RL6 Part D). Requires servers started
  externally: a backend on :8001 with `RESEARCH_LANES_ENABLED=true` and
  `FRONTEND_ORIGIN=http://localhost:5174`, and a Vite on :5174 with
  `VITE_API_BASE_URL=http://localhost:8001`. Makes ~7 real OpenAI calls
  for one Suggest + one Start (two enabled lanes) on a disposable topic,
  then deletes the session. Not part of any normal suite. The
  `.rl6-live-evidence.json` it writes is gitignored.
