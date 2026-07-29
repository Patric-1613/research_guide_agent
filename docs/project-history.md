# Project history

A phase-by-phase account of how this project got to its current state. This
is a **new** document — nothing has been removed from `README.md`; the two
are complementary. `README.md` is the authoritative deep-dive for the
original one-shot research pipeline (search → dedup → rank → summarize/
chat) and its own retrieval/ranking/RAGAS experiments. This document covers
the same ground at a summary level, plus everything built afterward that
`README.md` doesn't yet describe: the interactive curation system, the
report/chat workflow, and the React frontend.

Branch names below are exact (`git branch -a` / `git log --merges`);
per-item detail depth reflects how directly each phase is documented in
this repo's own code comments and commit history, not a judgment about
importance.

## Arc 1 — the original research pipeline

Round-1 phases (see `README.md`'s own "Project structure" and "Try each
phase individually" sections for the authoritative description):
ingestion (arXiv/Semantic Scholar search) → dedup → embeddings/ranking →
`agent.py`'s tool-calling orchestration → `summarize.py`'s themed summaries
→ `qa.py`'s conversational Q&A → `api.py`'s FastAPI surface.

Notable branches/PRs building on that foundation:
- `round2-enhancements`, `round3-interactive-triage` — early enhancement
  rounds (see README "Key design decisions" for what shipped).
- `retrieval-eval`, `ranking-experiment`, `citation-partition-experiment`,
  `citation-partition-final-rule`, `citation-partition-k-generalization` —
  the retrieval-ranking experiments README documents in detail under
  "Retrieval ranking experiments" (BM25/RRF confirmed worse than
  semantic-only; citation-partitioned reranking confirmed as a real win;
  the derived proportion rule tested across k=3–30).
- `ragas-integration`, `ragas-full-metrics` — RAGAS quality evaluation
  harness (README: "RAGAS quality evaluation").
- `mentor-feedback-fixes` — the robustness/reliability hardening pass
  README documents under "Robustness & reliability pass."
- `semantic-classify-message` — replaced an exact-match non-substantive-
  message allowlist with an embedding-similarity classifier (four
  independent guards: question-mark veto, length cap, content-override
  words, similarity threshold — still live in `qa.py`'s `classify_message`
  node today).
- `qa-langgraph-conversion` — converted `qa.py`'s `ask()` from a plain
  function into a compiled LangGraph `StateGraph`, laying the checkpointing
  foundation the curation system later activated for real.
- `parallelize-search-calls`, `add-openalex-fallback` — search-call
  parallelization and an OpenAlex fallback for Semantic Scholar rate-limit
  flakiness (README: "Search-call parallelization," "Agent-path concurrency
  fixes").
- `langfuse-integration` — the Langfuse tracing layer (README:
  "Observability").
- `streamlit-removal` — removed the original Streamlit UI in favor of the
  React/Vite frontend once it reached parity.

## Arc 2 — the curation system (review → report → chat)

The larger, more recent arc: an interactive literature-review curation
assistant built on top of the same retrieval/ranking/QA machinery, with a
dedicated React frontend. Phase numbering below matches the in-code
comments (`curation-*` prefixes throughout `research_agent/`).

- **Phase 1–2 (`curation-pool-foundation`, `curation-checkpointer`)** —
  `PaperPoolSession`/candidate-pool building in `query_expansion.py`; the
  SQLite checkpointer activated for this flow specifically
  (`curation_session.py`), namespaced separately from any future chat
  persistence via a `curation-session:` thread-id prefix.
- **Phase 3 (`curation-interrupt-loop`)** — the interactive present/pick/
  resume loop (`curation_loop.py`), built on LangGraph's `interrupt()`/
  `Command(resume=...)`, verified empirically against the installed
  LangGraph version rather than assumed from documentation.
- **Phase 4–5 (`curation-api-and-ui`)** — the `/curation/*` FastAPI surface
  and the first React UI, replacing Streamlit for this flow.
- **Phase 5 (`curation-report-synthesis`)** — `report.py`: structured
  literature-review report generation grounded strictly in selected
  abstracts.
- **Phase 5c (`curation-chat-web-escalation`)** — `curation_chat.py`: the
  offer-and-decide web-search escalation on top of `qa.ask()` (an
  unanswerable question offers a web search; a small LLM call classifies
  accept/decline/an unrelated new question).
- **Phase 6 (`curation-refinement-and-auto-offer`)** — mid-curation
  refinement text forcing a fresh, steered search; the report-update offer
  reusing the same accept/decline mechanism once new web sources land.
- **Phase 7 — UI redesign** — locked-tab workspace switcher (Review/Chat/
  Report), grouped review-list sections, abstracts rendered in the center
  panel.
- **Phase 8 (`curation-review-management`)** — review delete/abandon, a
  corrected status taxonomy, a fix for a reopened-completed-review render
  gap, canonical display titles (`canonicalize_topic`) distinct from the
  raw search topic.
- **Phase 9 (`curation-turn-history`)** — turn-history browsing across
  every past batch (not just the current one), on-demand refill
  ("Search for more candidates" independent of true exhaustion), and
  `select_paper_from_history` recovery for a session that ran dry short of
  target.
- **Phase 10 (`curation-editable-until-locked`)** — redefined "locked" from
  "reached `target_count` or the pool exhausted" to "a report exists and/or
  chat has started" specifically. `target_met`/`exhausted` no longer force
  `stage="synthesize"` — both loop back with a message instead, leaving an
  explicit "I'm done" as the only real stop. Added the `/curation/{id}/
  reopen` endpoint for a stopped-but-untouched review.
- **`chat-ux-fixes`** — five reported bugs fixed in one arc: the turn-
  history browser scoped to Review/Chat only (not Report); the user's own
  chat message rendering optimistically instead of after the full round
  trip; accepting a web-search offer resolving the actual follow-up
  fragment into a real standalone query (reusing `qa.py`'s own
  `condense_question`) and recording a curated transcript label instead of
  a repeated question or a bare "yes"; `[Paper N]`/`[Web N]` citation
  markers renumbered deterministically instead of trusted from the model;
  and a real bug where the turn-history browser could still add papers to
  an already-locked review through `select_paper_from_history`, which had
  never been updated to check the new lock condition.

## Test-count trajectory

A rough proxy for how much of this arc is covered by deterministic tests
(not a precise phase-by-phase log, but directionally accurate from this
project's own running commentary across the curation arc): the backend
suite grew from roughly 148 tests (the original pipeline, per `README.md`)
through the 200s and into the 300s as the curation system was built out,
standing at 340 as of this document (2026-07-29) — see `specs/test-plan.md`
for the current per-file breakdown.
