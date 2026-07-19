# Research Paper Summarizer Agent

An agent that takes a natural-language research topic, searches arXiv and
Semantic Scholar, deduplicates and ranks results by semantic relevance,
produces a structured literature summary grounded strictly in retrieved
abstracts, generates citations, and answers conversational follow-up
questions — with every claim traceable back to a specific paper.

No Google Scholar (no official API, scraping violates ToS) — only the
arXiv and Semantic Scholar APIs are used.

## Architecture

![Architecture diagram](research_agent_architecture.svg)

The diagram above is the detailed, file-by-file view (updated for the
robustness/reliability changes below — brighter highlighted lines within
each box). The condensed version:

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
                          │ per-request conn. via │
                          │ FastAPI Depends + WAL │
                          │ saved searches:       │
                          │ topic, paper_ids,     │
                          │ scores, summary       │
                          └───────────┬──────────┘
                                       ▼
                          ┌──────────────────────┐
                          │ api.py (FastAPI)       │
                          │ /search /summarize     │
                          │ /chat /export /library │
                          │ upstream errors →      │
                          │ clean 503, no raw 500  │
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
  schema.py         Paper — the normalized record shared by every phase
  ingestion.py       search_arxiv(), search_semantic_scholar()
  dedup.py           cross-source deduplication + merge
  embeddings.py      batched + cached embedding, Chroma storage, cosine retrieval
  query_expansion.py LLM-suggested-title candidate-pool widening (build_candidate_pool
                     + expanded_search) — the live app's opt-in query-expansion mode
  ranking.py         opt-in alternative FINAL ranking steps for evaluation only —
                     BM25, RRF hybrid fusion, citation-partitioned reranking, and
                     the derived get_partition_n(k) rule (see "Retrieval ranking
                     experiments" below); never used by the live app's default path
  agent.py           LangChain tool-calling orchestration agent
  summarize.py       theme clustering + grounded per-paper summaries
  citations.py       APA + BibTeX formatting (deterministic, no LLM)
  qa.py              conversational RAG over retrieved abstracts
  storage.py         SQLite persistence for saved searches
  api.py             FastAPI backend
  app.py             Streamlit frontend
scripts/           runnable CLI demos for each phase (see below), plus two
                   real-pipeline evaluation harnesses: eval_retrieval.py
                   (retrieval precision/recall + ranking-mode experiments)
                   and ragas_eval.py (RAGAS Faithfulness/Answer Relevancy/
                   Context Precision/Context Recall over a curated question set)
tests/             deterministic unit tests (147 tests, zero network/LLM
                   calls required — see "Run the tests" below)
eval_data/         curated reference sets consumed by the eval harnesses
                   (17-topic retrieval reference set, 24-scenario RAGAS set)
eval_results/      CSV run history for both harnesses, plus eval_results/runs/
                   (per-run RAGAS artifacts — see "RAGAS quality evaluation")
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

All 147 tests in `tests/` are fully deterministic and need no network access
and no API keys — every LLM call (OpenAI) and every external API call
(arXiv, Semantic Scholar, Unpaywall/CrossRef, Tavily) is mocked, including
the `OpenAI()` client construction in `api.py`'s FastAPI `lifespan()`, which
otherwise runs unconditionally at `TestClient` startup regardless of which
endpoint a given test hits. Verified directly: `tests/` passes 147/147 with
`.env` entirely absent.

The live smoke tests live separately in `scripts/` (not `tests/`, and not
run by `pytest`) — those intentionally hit real APIs and cost a small amount
of real tokens; see "Try each phase individually" above.

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

## Robustness & reliability pass

A later review pass (branch `mentor-feedback-fixes`) hardened several edge
cases and failure modes that the original phases above didn't cover. No
behavior changed for valid/well-formed input anywhere in this list — every
fix below only changes what happens on an already-broken or edge-case input,
verified by keeping the full test suite green (101 → 128 tests) throughout.

- **Defensive parsing fixes:** a blank/`None` author name inside an
  otherwise normal author list no longer crashes APA/Harvard formatting
  (falls back to `"Unknown"`); a malformed/empty Semantic Scholar response
  body is caught and degrades to `[]` instead of raising;
  `Retry-After` is parsed as either plain seconds or an HTTP-date (both are
  valid per RFC 9110), falling back to the existing backoff default if
  neither parses; a paper with both an empty title and no abstract is
  skipped (logged) rather than failing the whole embedding batch.
- **Embedding order correctness:** OpenAI's embeddings API doesn't guarantee
  response order matches request order — vectors are now sorted by the
  API's own `index` field before being assigned back to their papers,
  instead of trusting list position.
- **Resilient agent tool calls:** each of the four agent tools
  (`search_arxiv_tool`, `search_semantic_scholar_tool`,
  `rerank_by_relevance_tool`, `search_web_tool`) now wraps its body in
  try/except, so one tool's failure (e.g. an OpenAI embedding call erroring
  mid-rerank) returns a description instead of killing the whole agent run
  — already-gathered papers survive, and the failed step is retryable.
- **Per-request SQLite connections:** `storage.py` moved from one SQLite
  connection shared across every request (`check_same_thread=False`) to a
  FastAPI `Depends`-based connection opened and closed per request
  (`get_db_connection`), plus WAL journal mode and a `busy_timeout` — the
  standard pairing for safe concurrent access under FastAPI's
  multi-threaded request handling. Verified with a 20-thread concurrent
  write test.
- **Clean upstream error responses:** `/search`, `/summarize`, `/chat`, and
  `/export` now catch `OpenAIError`/`ArxivError`/`RequestException` at the
  endpoint boundary and return a clean `503 {"error": "... unavailable"}`
  instead of leaking a raw 500 with an internal stack trace — while
  intentional 404s (`HTTPException`) are re-raised untouched, not swallowed
  into a 503.
- **Duplicate-citation guard:** the dynamic-`Literal` grounding in
  `summarize.py` guarantees a cited `paper_id` was actually retrieved, but
  never prevented the *same* `paper_id` from being placed in more than one
  theme — a post-generation check now keeps only its first occurrence
  (logged as a warning).
- **Capped chat history:** `qa.py` now caps chat history to the last 8
  turns (16 messages) before either LLM call it makes per turn (follow-up
  condensing and answer generation), bounding cost/latency growth as a
  conversation lengthens. The cap is prompt-only — `session.history` itself
  stays fully intact for a UI transcript.
- **Zero-API-key test suite:** the `OpenAI()` client construction in
  `api.py`'s `lifespan()` (previously unconditional and unmocked in tests)
  is now mocked in the shared test fixture — `pytest tests/` passes
  128/128 with `.env` entirely absent, not just with real credentials
  configured.

## Retrieval ranking experiments

A later, separate line of work asked whether the live app's ranking step
(cosine similarity over embeddings, nothing else) is actually the best
available option, and whether a diagnosed recurring failure — foundational
papers (e.g. LoRA) losing rerank against generic survey papers that repeat
a topic's wording densely — has a fix. All of this is **opt-in evaluation
tooling only**, wired through `scripts/eval_retrieval.py`'s `--ranking-mode`
flag; `research_agent/ranking.py` is never imported by `api.py`, `app.py`,
or `qa.py`, and the live app's default ranking behavior is unchanged.

Every number below is a real run against the same 17-topic reference set
(`eval_data/reference_topics.json`) used throughout, logged to
`eval_results/retrieval_history.csv`.

### BM25 and hybrid (RRF) — both confirmed worse than semantic-only

| Mode | Precision@10 | Recall@10 |
|---|---|---|
| semantic (baseline) | 0.029 | 0.216 |
| bm25 | 0.018 | 0.147 |
| hybrid (RRF, k=60) | 0.018 | 0.137 |

BM25-alone underperforming was the predicted outcome (term-frequency
density rewards the same generic survey papers that were the original
diagnosed problem) and was confirmed. Hybrid was expected to be genuinely
uncertain; instead it measurably underperformed BM25 alone and cratered to
0.0 recall on the `easy` difficulty tier — RRF's k=60 constant (the
standard from Cormack et al. 2009, still the Elasticsearch/OpenSearch/
Azure AI Search default) is validated at web-scale candidate-list sizes,
and appears to dilute its own top-rank-rewarding mechanism against this
project's much smaller ~20–40-paper per-topic pools. **Neither replaces
semantic-only ranking.**

### Citation-partitioned reranking — a real, large win

A different idea: sort the candidate pool by citation count (papers with
no citation count are never eligible, never treated as zero), reserve `n`
guaranteed final-result slots for the highest-cited eligible papers
(Partition A), then rank *everything* — both partitions — by semantic
similarity against the original topic, subject to that guarantee. Final
order is always by semantic score, never partition-then-partition
stacking; the guarantee only changes anything when Partition A wasn't
already going to rank well on its own merit.

At `top_k=10`, `n=2` recovers **LoRA** — the paper this whole diagnostic
arc kept circling back to — and roughly doubles recall over the semantic
baseline:

| n (top_k=10) | Precision@10 | Recall@10 |
|---|---|---|
| 0 (semantic baseline) | 0.029 | 0.216 |
| 1 | 0.043 | 0.431 |
| **2** | **0.055** | **0.549** |
| 3 | 0.047 | 0.471 |
| 4 | 0.047 | 0.471 |

**The citation-bias tradeoff is real and was deliberately probed to its
extreme, not just measured in aggregate**: at `top_k=3`, forcing every
single result to be a Partition-A member (`n=3`, 100%) collapsed the
`domain` difficulty tier from a day-long-solid 0.667 recall to 0.000 —
even though that same setting gave the LoRA/QLoRA topic its best recall of
the whole study (0.667). The ideal setting is genuinely topic-dependent,
not just a k-dependent knob.

### Does the winning proportion generalize across k? No — tested at k=3, 5, 10, 20, 25, 30

The original "reserve 20% of k" rule was **disconfirmed** once tested
outside k=10: the true peak proportion swings from ~67% at k=3 down to
~7% at k=30, while the true peak *absolute* n stays in a much narrower,
non-monotonic band:

| k | True peak n | True peak recall |
|---|---|---|
| 3 | 2 | 0.392 |
| 5 | 3 | 0.471 |
| 10 | 2 | 0.549 |
| 20 | 4 (confirmed via a complete, gap-free n=1–8 sweep) | 0.578 |
| 25 | 3 | 0.578 |
| 30 | 2 (a wide plateau, n=2 through n=8) | 0.549 |

`research_agent/ranking.py`'s `get_partition_n(k)` implements the derived
production rule, **`n = min(2, k)`** — a flat constant, not a scaling
formula. It was chosen over two candidates (a hand-fit step function; a
light `k/8`-scaled-and-clamped formula) that scored marginally better in
raw aggregate fit but only by tuning thresholds to exactly six data
points with no independent validation. `n=2` is exactly optimal at k=3,
at k=10 (the documented production default in `api.py`/`app.py`), and at
k=30 (the production maximum) — reported honestly, it is *not* optimal
everywhere: it leaves real recall on the table at k=5 (−0.049) and k=20
(−0.088, the largest gap found). `eval_retrieval.py`'s `citation_partition`
mode uses this rule automatically unless `--partition-n` or
`--partition-proportion` explicitly overrides it for further
experimentation.

**Deployment status: still opt-in only.** Even with the k-generalization
question resolved, this hasn't been validated against live user queries
(only the offline 17-topic reference set) and isn't merged into any
default path — promoting it would require that live-query validation
first.

Run it yourself:
```bash
uv run python scripts/eval_retrieval.py --note "..." --ranking-mode citation_partition --top-k 10
uv run python scripts/eval_retrieval.py --note "..." --ranking-mode bm25          # or hybrid, semantic
```

## RAGAS quality evaluation

A curated, hand-verified test set (`eval_data/stage1_ragas_questions.json`,
24 scenarios across single-paper, cross-paper-comparison, multi-turn, and
deliberately-unanswerable categories) drives `scripts/ragas_eval.py`,
which runs the real pipeline — real search, real `qa.py` answers — through
all four RAGAS metrics. Every scenario's target paper was independently
confirmed to actually survive the real two-stage retrieval (Stage 1
candidate search at `top_k=10`, Stage 2 answer-time re-rank at `qa.py`'s
own `TOP_K_DEFAULT=5`) before being included — 15 of an original 25
candidate questions were dropped or rewritten after verification showed
their target paper (e.g. LoRA, ResNet, ARES) never actually survives
either stage under the topics as originally phrased.

Real results from the latest full run:

| Metric | Overall | Notes |
|---|---|---|
| Faithfulness | 0.945–0.949 | Uniformly high across every category |
| Answer Relevancy | ~0.58–0.64 | Mechanically explained, not a defect: RAGAS's own noncommittal-answer penalty zeroes any refusal/hedge response, and 6 of 24 scenarios are deliberately unanswerable |
| Context Precision | 0.778 (20/24 scenarios have a reference) | Notably lower for comparison-category questions (0.506) — their reference only concerns 2 of the 5 papers Stage 2 always retrieves, so the other 3 get penalized as irrelevant |
| Context Recall | 1.000 | Expected/structural, not independent proof of retrieval quality — references were drafted from the same abstracts retrieval reliably surfaces |

17 of 24 scenarios have a hand-drafted reference answer (grounded strictly
in the real retrieved abstracts, never general knowledge) enabling
Context Precision/Recall; the other 7 (6 deliberately-unanswerable
scenarios, plus one comparison question flagged during review as too thin
to ground a confident reference) are excluded from those two metrics
specifically, with the reason recorded per-scenario in the data file —
Faithfulness/Answer Relevancy are still computed for all 24.

Every run also writes `eval_results/runs/run_<id>.json` (the full scored
per-turn record: question, real retrieved paper titles, real generated
answer, every metric) plus an incremental `raw_<timestamp>.jsonl`, written
turn-by-turn *during* generation rather than only at the end — so
already-paid-for generation data survives even if scoring itself later
crashes or rate-limits.

```bash
uv run python scripts/ragas_eval.py --note "..."
```

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
