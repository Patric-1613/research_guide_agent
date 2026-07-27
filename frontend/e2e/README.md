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
