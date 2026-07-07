# Project Brief: Research Paper Summarizer Agent

## Context
I'm a GenAI intern building a portfolio project. I've previously built a RAG-powered
study buddy (PDF parsing, chunking, OpenAI embeddings, ChromaDB, LangChain retriever,
stateful topic tracking) so I'm comfortable with that stack. This project extends that
experience into a multi-source retrieval agent.

Build in **phases**. Do not proceed to the next phase until the current one is working
and I've confirmed it. After each phase, show me what you built, how to run/test it,
and any assumptions you made.

## Objective
Build an agent that takes a natural-language research topic, automatically searches
arXiv and Semantic Scholar, deduplicates and ranks papers by semantic relevance,
produces a structured literature summary grouped by theme, generates proper citations,
and supports conversational follow-up questions grounded in the retrieved abstracts.

## Hard constraints
- No Google Scholar — no official API and scraping violates their ToS. Use only
  the arXiv API and Semantic Scholar API.
- All secrets (API keys) go in a `.env` file, never hardcoded, never committed.
  Create a `.env.example` with placeholder keys and add `.env` to `.gitignore`.
- Every claim in the generated summary must be traceable to a specific retrieved
  paper. No fabricated citations, no filling gaps from general knowledge.
- Keep dependencies minimal and pinned in `requirements.txt`.
- Batch embedding calls to the OpenAI API (don't embed one abstract per request)
  and cache embeddings locally so re-running a search doesn't re-embed papers
  I've already indexed — I'm paying per token for this.

## Tech stack
- Python 3.11+
- LangChain (agent orchestration + tool calling)
- ChromaDB (vector store, local persistence)
- OpenAI `text-embedding-3-small` for embeddings (batch requests, cache results)
- OpenAI GPT (e.g. `gpt-4o` or `gpt-4o-mini` — your call based on cost/quality
  tradeoff, explain which you pick and why) for summarization, clustering, and
  conversational Q&A
- FastAPI (backend API)
- Streamlit (frontend, v1 — keep it simple, functional over polished)
- SQLite (search history / saved library persistence)
- rapidfuzz (fuzzy title matching for dedup)

## Functional requirements (build in this order)

### Phase 1 — Data ingestion
1. `search_arxiv(query, max_results)` — wraps the `arxiv` package, returns normalized
   records: `{title, authors, year, venue, abstract, url, doi, citation_count, source}`
   (arXiv won't have citation_count — set to `None`)
2. `search_semantic_scholar(query, max_results)` — calls the Semantic Scholar
   `/graph/v1/paper/search` REST endpoint, same normalized schema
3. Write a small CLI test script that runs both for a sample query and prints
   the normalized results side by side, so I can sanity-check the schema before
   we build anything on top of it

**Success criteria for Phase 1:** I can run one script, give it a topic, and see
clean structured JSON from both sources with no crashes on edge cases (zero
results, rate limit, malformed abstract).

### Phase 2 — Deduplication and merge
1. Fuzzy-match on title (rapidfuzz, threshold ~90) plus exact DOI match where
   available
2. When a duplicate is found across sources, merge into one record — keep the
   richer abstract, combine citation counts, keep both source URLs

**Success criteria:** feed it a query that surfaces the same paper from both
sources, confirm it collapses to one record, not two.

### Phase 3 — Embedding and relevance ranking
1. Embed abstracts with OpenAI `text-embedding-3-small`, store in ChromaDB
   (local persist directory, not in-memory). Batch the embedding calls and
   cache results (e.g. keyed by paper ID or a hash of the abstract text) so
   we don't re-pay for papers already embedded
2. Embed the user's query, retrieve top-k by cosine similarity
3. This retrieval step **is** the relevance ranking — don't build a separate
   scoring system on top of it
4. Log estimated token usage/cost per embedding batch so I can keep an eye on
   spend while testing

**Success criteria:** for an ambiguous query, top results are genuinely more
relevant than a plain keyword search would return — show me a before/after
comparison.

### Phase 4 — Agent/orchestration layer
1. Build a LangChain agent with tools: `search_arxiv`, `search_semantic_scholar`,
   `rerank_by_relevance`
2. The agent should decide whether to search both sources or one, and whether to
   reformulate an ambiguous query (e.g. expand acronyms or add synonyms) before
   searching
3. Log the agent's tool calls and reasoning so I can see its decisions, not just
   the final output — this matters for me to actually understand and explain
   how it works

### Phase 5 — Summarization
1. Cluster retrieved papers by theme/methodology (embeddings + simple clustering,
   or prompt-based grouping — your call, but explain the tradeoff)
2. Generate a structured summary: themes as headers, 2-3 sentences per paper
   grounded strictly in its abstract, explicit callout of any gaps or
   disagreements across papers
3. Generate citations in APA format (BibTeX as a stretch goal)

### Phase 6 — Conversational Q&A (RAG)
1. Standard retrieve-then-answer: user question against the Chroma store,
   answer grounded in retrieved chunks, cite which paper(s) support each claim
2. If the retrieved papers can't answer the question, say so explicitly —
   don't hallucinate an answer

### Phase 7 — Backend + storage
1. FastAPI endpoints: `/search`, `/summarize`, `/chat`, `/export`, `/library`
2. SQLite table for saved searches (topic, timestamp, paper IDs, summary)

### Phase 8 — Frontend
1. Streamlit app: topic input, results list with relevance scores, summary view,
   chat panel, export button (Markdown output)
2. Keep it functional and clean — no need for custom CSS or heavy styling in v1

## What I want from you at each phase
- Working code, not pseudocode
- A short explanation of any design decision with more than one reasonable option
- Point out if a requirement above is a bad idea technically, and suggest the fix
- Tell me explicitly if something needs an API key I haven't provided yet
- After each phase, a one-paragraph summary I could drop into my internship
  progress report describing what was built and why

## Out of scope for v1 (don't build unless I ask)
- Authentication/multi-user support
- Deployment/hosting
- PDF full-text ingestion (abstracts only for now)
- React frontend (Streamlit is enough for v1)

Start with Phase 1. Ask me for my Semantic Scholar and OpenAI API keys before
you write any code that needs them, and confirm the project folder structure
with me before creating files.
