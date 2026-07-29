# Test plan

Baseline captured 2026-07-29, before any migration work: backend 340
passed, frontend 98 passed (10 files), frontend build clean. This document
is the checklist for what to run at each migration checkpoint in
`specs/migration-plan.md`.

## Backend — `uv run pytest -q` (deterministic, no network/API keys needed)

| File | Tests | Covers |
|---|---:|---|
| `test_agent.py` | 9 | LangChain tool-calling agent orchestration |
| `test_api.py` | 27 | original one-shot pipeline endpoints (`/search`, `/summarize`, `/chat`, `/export`, `/library`) |
| `test_citations.py` | 17 | APA/BibTeX formatting |
| `test_curation_api.py` | 48 | every `/curation/*` endpoint, HTTP-level |
| `test_curation_chat.py` | 25 | offer-and-decide web-search/report-update escalation |
| `test_curation_loop.py` | 31 | the interrupt-based present/pick/resume graph |
| `test_curation_session.py` | 31 | checkpointed save/load, select-from-history, reopen, delete |
| `test_dedup.py` | 4 | cross-source dedup/merge |
| `test_embeddings.py` | 12 | batched/cached embedding, Chroma retrieval |
| `test_enrichment.py` | 15 | abstract backfill |
| `test_ingestion.py` | 10 | arXiv/Semantic Scholar search |
| `test_qa.py` | 23 | the QA graph (classify/condense/retrieve/generate), citation renumbering |
| `test_query_expansion.py` | 27 | `PaperPoolSession`, pool building/ranking/refill, `canonicalize_topic` |
| `test_ranking.py` | 19 | BM25/RRF/citation-partitioned ranking (eval-only paths) |
| `test_report.py` | 15 | report generation/regeneration |
| `test_storage.py` | 10 | SQLite saved-search persistence |
| `test_summarize.py` | 7 | theme clustering + per-paper summaries |
| `test_web_search.py` | 7 | Tavily wrapper (degrade-to-empty on failure) |

Run targeted subsets during a migration phase, full suite at phase
boundaries — see `specs/migration-plan.md`'s per-phase test gates.

## Frontend — `cd frontend && npm test` (vitest)

| File | Covers |
|---|---|
| `App.test.tsx` | mode switching, auto-unlock/lock, turn-history visibility |
| `api/client.test.ts` | typed fetch wrapper against every endpoint shape |
| `hooks/useCurationSession.test.ts` | the one stateful hook — session lifecycle, turn events, chat search meta |
| `components/ReviewMode/ReviewModePanel.test.tsx` | batch rendering, empty/target-reached states |
| `components/ReviewMode/PoolSummaryPanel.test.tsx` | right-panel pool stats |
| `components/ChatMode/ChatModePanel.test.tsx` | optimistic send, offer buttons, search-outcome note |
| `components/ReportMode/ReportModePanel.test.tsx` | report display/regeneration |
| `components/ReviewsList/ReviewsList.test.tsx` | left-panel review grouping |
| `components/TurnHistory/TurnHistoryBrowser.test.tsx` | past-turn browsing, locked-review view-only rule |
| `components/WorkspaceMode/WorkspaceModeSwitcher.test.tsx` | tab lock/unlock affordance |

`cd frontend && npm run build` (`tsc -b && vite build`) must also stay
clean — a type error here is not caught by vitest alone.

## Eval scripts (not run by `pytest`/`npm test` — real API calls, real cost)

`scripts/eval_retrieval.py`, `scripts/ragas_eval.py`, and every
`scripts/test_*.py` file hit real arXiv/Semantic Scholar/OpenAI/Tavily
APIs deliberately, per this project's own established discipline of
pairing every LLM-dependent feature with at least one live, non-mocked
check. These are **not** part of an automated migration test gate; run
manually when a phase specifically touches eval logic (Phase 6) or an
LLM-call site whose mocked test coverage can't fully prove correctness
(e.g. the citation-renumbering fix, the web-offer query-resolution fix).

## Manual smoke tests

Run whenever Phase 2 (API split) or Phase 3 (services) changes a route's
wiring, in addition to the automated suites:

1. `uv run uvicorn research_agent.api:app --reload --reload-exclude
   "frontend/*"` boots without error; `curl localhost:8000/health` → `200`.
2. `cd frontend && npm run dev`; open `localhost:5173`; start a new review,
   confirm a batch renders.
3. Pick papers to reach target, click "I'm done," generate a report,
   switch to Chat, ask a question, confirm a grounded answer with
   citations.
4. Refresh the page mid-curation (a batch pending) and mid-chat (an offer
   pending) — confirm state survives via the checkpointer, not just React
   state.
