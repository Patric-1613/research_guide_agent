# Research Paper Summarizer Agent

An agent that takes a natural-language research topic, searches arXiv and
Semantic Scholar, deduplicates and ranks results by semantic relevance,
produces a structured literature summary grounded strictly in retrieved
abstracts, generates citations, and answers conversational follow-up
questions — with every claim traceable back to a specific paper.

No Google Scholar (no official API, scraping violates ToS) — only the
arXiv and Semantic Scholar APIs are used.

## Architecture

```
Topic
  │
  ▼
┌────────────────────┐   ┌──────────────────────┐
│ ingestion.py        │   │ dedup.py              │
│ search_arxiv()       │──▶│ fuzzy title + DOI     │
│ search_semantic_     │   │ match, merge records  │
│ scholar()            │   └───────────┬──────────┘
└────────────────────┘               │
                                       ▼
                          ┌──────────────────────┐
                          │ embeddings.py         │
                          │ batch-embed abstracts │
                          │ (cached), store in    │
                          │ Chroma, cosine search │
                          └───────────┬──────────┘
                                       ▼
                          ┌──────────────────────┐
                          │ agent.py              │
                          │ LangChain tool-calling │
                          │ agent orchestrates the │
                          │ above: which source(s),│
                          │ query reformulation,   │
                          │ when to rerank         │
                          └───────────┬──────────┘
                                       ▼
                    ┌──────────────────┴──────────────────┐
                    ▼                                       ▼
        ┌──────────────────────┐                ┌──────────────────────┐
        │ summarize.py          │                │ qa.py                 │
        │ theme clustering +     │                │ conversational RAG:    │
        │ grounded per-paper     │                │ condense follow-up →   │
        │ summaries + citations  │                │ retrieve → answer with │
        │ (citations.py: APA/    │                │ inline [n] citations   │
        │ BibTeX, no LLM)        │                │                        │
        └──────────────────────┘                └──────────────────────┘
                    │                                       │
                    └──────────────────┬───────────────────┘
                                        ▼
                          ┌──────────────────────┐
                          │ storage.py (SQLite)   │
                          │ saved searches:        │
                          │ topic, paper_ids,      │
                          │ scores, summary        │
                          └───────────┬──────────┘
                                       ▼
                          ┌──────────────────────┐
                          │ api.py (FastAPI)       │
                          │ /search /summarize     │
                          │ /chat /export /library │
                          └───────────┬──────────┘
                                       ▼
                          ┌──────────────────────┐
                          │ app.py (Streamlit)     │
                          │ topic input, results,  │
                          │ summary, chat, export   │
                          └──────────────────────┘
```

Persistence has two layers with different jobs:
- **ChromaDB** (`data/chroma_db/`) is the source of truth for paper content
  (title, abstract, authors, ...) and their embeddings — it's keyed by
  `paper_id` and shared across every phase.
- **SQLite** (`data/history.sqlite`) only tracks *which* `paper_id`s belong
  to which saved search (topic, timestamp, scores, generated summary). It
  never duplicates paper content — `paper_id` is the join key back to Chroma.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
cp .env.example .env
```

`uv sync` creates a `.venv` and installs the exact pinned versions from
`uv.lock`. Run project commands with `uv run <command>`, or activate the
environment directly with `source .venv/bin/activate`.

Edit `.env` and set:
- `OPENAI_API_KEY` — required (embeddings + summarization + chat + agent).
- `SEMANTIC_SCHOLAR_API_KEY` — optional. Semantic Scholar works
  unauthenticated at a low, shared rate limit; a free key
  ([semanticscholar.org/product/api](https://www.semanticscholar.org/product/api))
  raises it. Search functions degrade gracefully (empty result, not a crash)
  if rate-limited.

## Running the app

Two processes, in separate terminals:

```bash
uv run uvicorn research_agent.api:app --reload --reload-exclude "app.py"
uv run streamlit run research_agent/app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).
Interactive API docs are at `http://localhost:8000/docs`.

`--reload-exclude "app.py"` matters: by default `--reload` watches every
file in the project, including `app.py` (the Streamlit frontend, which the
FastAPI backend never imports). Without the exclude, editing the frontend
mid-request restarts the backend and kills whatever request was in flight —
easy to mistake for a hang or timeout.

## Project structure

```
research_agent/
  schema.py       Paper — the normalized record shared by every phase
  ingestion.py     search_arxiv(), search_semantic_scholar()
  dedup.py         cross-source deduplication + merge
  embeddings.py    batched + cached embedding, Chroma storage, cosine retrieval
  agent.py         LangChain tool-calling orchestration agent
  summarize.py     theme clustering + grounded per-paper summaries
  citations.py     APA + BibTeX formatting (deterministic, no LLM)
  qa.py            conversational RAG over retrieved abstracts
  storage.py       SQLite persistence for saved searches
  api.py           FastAPI backend
  app.py           Streamlit frontend
scripts/           runnable CLI demos for each phase (see below)
tests/             deterministic unit tests (30 tests, no network/LLM calls
                   except where explicitly a live smoke test)
data/              gitignored: chroma_db/, cache/, history.sqlite
```

### Try each phase individually

```bash
uv run python scripts/test_ingestion.py "your topic"          # phase 1: raw search
uv run python scripts/test_dedup.py "your topic"               # phase 2: dedup/merge
uv run python scripts/test_ranking.py                          # phase 3: keyword vs. semantic ranking
uv run python scripts/test_agent.py "your topic"                # phase 4: agent + tool-call log
uv run python scripts/test_summarize.py "your topic"            # phase 5: themed summary + citations
uv run python scripts/test_qa.py                                # phase 6: multi-turn grounded chat
uv run python scripts/test_api.py                                # phase 7: full API flow, live
```

### Run the tests

```bash
uv run pytest tests/ -v
```

All 30 tests are deterministic (mocked LLM/API calls where relevant) except
the live smoke tests in `scripts/`, which hit real APIs and cost a small
amount of real tokens.

## Key design decisions

- **Dedup cache key is a content hash, not `paper_id`.** A merged record's
  `paper_id` changes (`arxiv_id+s2_id`) when dedup collapses a duplicate —
  keying the embedding cache on the abstract's hash instead means a paper is
  never re-billed just because it was later found to be a duplicate.
- **Relevance ranking *is* retrieval** — cosine similarity over cached
  embeddings, no separate scoring system layered on top.
- **Agent model vs. summarization model:** `gpt-4.1-mini` for the phase-4
  agent's tool-calling loop (many calls per session, cost compounds) vs.
  `gpt-4.1` for summarization/Q&A (infrequent, user-facing, faithfulness
  matters more than marginal cost).
- **Grounding is enforced structurally, not just by prompting.** Every
  citation (`paper_id`) the model can emit is constrained to a dynamic
  `Literal` type built from the exact papers retrieved for that call —
  fabricating a citation is a schema violation, not just discouraged.
  Verified directly (a test asserts an out-of-set `paper_id` is rejected).
- **Theme clustering is prompt-based, not embedding clustering (KMeans
  etc.).** At this project's scale (a handful to ~20 papers per topic), an
  embedding-clustering approach would still need an LLM call afterward just
  to name each cluster — folding grouping into the same call that writes
  the summaries is fewer LLM calls, not more.
- **APA/BibTeX citations are pure deterministic string formatting — no
  LLM.** There's nothing a model would do better here, and every LLM call
  is a chance to hallucinate a citation detail.
- **Follow-up questions are condensed into a standalone query before
  retrieval** (e.g. "what about its limitations?" → "what are RoCoFT's
  limitations?"), costing one small extra LLM call per turn — skipped
  entirely on the first turn, where there's no history to resolve against.
- **Chat history is not persisted server-side.** The client carries it
  forward per-request; only *searches* are saved to SQLite, per the brief.
- **`/summarize` and `/export` reuse a previously generated summary** for a
  given `search_id` instead of re-billing the LLM on repeat calls.
- **Every LLM-backed call in the Streamlit app is gated behind an explicit
  button/chat-input action.** Streamlit reruns the entire script on every
  widget interaction, so an unguarded call would silently re-trigger (and
  re-bill) on unrelated clicks.

## Known limitations

- Abstracts only — no PDF full-text ingestion (out of scope for v1).
- No auth/multi-user support; chat history lives in the browser session,
  not the database.
- Semantic Scholar's unauthenticated tier rate-limits under repeated use;
  get a free key if you hit this often.
- Author-name parsing for APA/BibTeX (`"First Last"` → `"Last, F."`) is a
  heuristic — it will mis-format multi-word surnames.
- Structural grounding prevents citing a paper that wasn't retrieved, but
  can't fully prevent an LLM from mis-stating a detail *within* a correctly
  cited paper's summary — an inherent limit of free-text generation, not
  specific to this project.
