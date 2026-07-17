"""Phase 6: conversational Q&A grounded in the retrieved papers (RAG).

Retrieval unit stays "one paper's abstract" — the same granularity used in
phases 3-5. PDF full-text ingestion is explicitly out of scope for v1
(per the brief), so there's no sub-document chunking to do; an abstract is
already a short, mostly self-contained unit, and splitting it further would
fragment context for no benefit.

Two design decisions worth calling out:

1. Follow-up questions are "condensed" into a standalone query before
   retrieval (e.g. "what about its limitations?" -> "what are RoCoFT's
   limitations?"), using conversation history. This costs one extra small
   LLM call per turn (skipped on the first turn, where there's no history to
   resolve against). The cheaper alternative — embedding the raw follow-up
   question as-is — is one call cheaper per turn, but pronoun-heavy
   follow-ups retrieve poorly on their own (the embedding for "what about
   its limitations?" isn't close to any paper's abstract). The condensed
   query is only used for retrieval; the model still answers the user's
   original question, so nothing about the conversation's phrasing changes.

2. Grounding is enforced the same way as phase 5: `cited_paper_ids` is
   constrained to a dynamic Literal built from the exact papers retrieved
   for this turn, so the model cannot cite a paper it wasn't shown. The
   answer text uses inline [Paper 1], [Paper 2] markers (in the order of
   cited_paper_ids) so a claim in the answer can be traced to a specific
   paper without needing a full per-sentence structured breakdown — a
   reasonable middle ground for a conversational answer, versus phase 5's
   per-paper summaries where a full breakdown was already the natural unit.

Round-2 enhancement 5 extends this to a second, independent corpus: web
articles (web_search.py). `cited_web_urls` gets the identical Literal-
grounding treatment, keyed on URL instead of paper_id — a web citation is
structurally impossible unless that URL was actually retrieved this turn,
the same guarantee level as paper citations, not a weaker one. The two
corpora use separate marker namespaces in the answer text ([Paper N] vs
[Web N], not a shared [N]) specifically so a user can tell a peer-reviewed
source from a web source at a glance, per the brief. Unlike papers, the web
article pool isn't re-ranked by embedding similarity for each question — at
the scale this pool actually runs at (3-4 articles per session, the same
"small enough that ranking doesn't earn its keep" scale reasoning
summarize.py already applies to generate_web_summary()), the whole pool is
just included in context directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, create_model

from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, semantic_search
from research_agent.schema import Paper, WebArticle

logger = logging.getLogger(__name__)

# Condensing a follow-up question is a small, frequent, low-stakes rewrite —
# same cost tier as the phase-4 agent's tool orchestration. Answer synthesis
# is the user-facing, quality-sensitive step, so it gets the stronger model,
# same reasoning as phase 5's summarizer.
CONDENSE_MODEL = "gpt-4.1-mini"
ANSWER_MODEL = "gpt-4.1"

TOP_K_DEFAULT = 5

# Every turn re-sends the full history to two LLM calls (condense + answer),
# so unbounded history means unbounded per-turn cost/latency growth as a
# conversation lengthens. Capped at the last 8 turns (user+assistant pairs,
# i.e. 16 messages) — confirmed with the project owner: coherence rarely
# depends on more than a handful of recent turns here, since each answer is
# re-grounded in the retrieved paper/web context every time, not carried
# forward from distant conversation history the way a general-purpose
# chatbot needs.
MAX_HISTORY_TURNS = 8

CONDENSE_SYSTEM_PROMPT = """Given a conversation history and a follow-up question, rewrite the follow-up as a standalone question that makes sense without the history — resolve pronouns and implicit references (e.g. "it", "that method", "the second one") to what they actually refer to.

If the follow-up question is already standalone (doesn't depend on the history), return it unchanged. Return ONLY the rewritten question, nothing else.
"""

ANSWER_SYSTEM_PROMPT = """You are a research assistant answering questions using ONLY the paper abstracts and/or web article snippets provided below. Do not use outside knowledge about these sources, their authors, or the topic beyond what they explicitly state.

Two distinct kinds of sources may be provided: retrieved papers (peer-reviewed/preprint academic literature) and retrieved web articles (news, tooling, docs — current/practical context, not peer-reviewed). Always keep them clearly distinguished — never imply a web article is a paper or vice versa.

If the provided sources do not contain enough information to answer the question, set answerable to false and explain in your answer what's missing — do not guess or fill the gap from general knowledge.

If you can answer, write a clear natural-language answer. Use inline bracket markers to mark which source supports each claim: [Paper 1], [Paper 2], ... for papers, in the order you list them in cited_paper_ids; [Web 1], [Web 2], ... for web articles (if any were provided), in the order you list them in cited_web_urls. These are two separate numbering sequences, never merged into one — a bare [1] that doesn't say "Paper" or "Web" is not acceptable. Every claim should be traceable to at least one marker.
"""


@dataclass
class ChatSession:
    """Grounding set + running history for one conversation. `papers` is
    normally whatever a prior search/rerank (phases 2-4) produced for a
    topic — Q&A doesn't search on its own, only answers from what's already
    been retrieved. `web_articles` (round-2 enhancement 5) is the same idea
    for the separate web-context corpus.
    """

    papers: list[Paper] = field(default_factory=list)
    web_articles: list[WebArticle] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)  # [{"role": "user"|"assistant", "content": str}]


def _build_answer_schema(paper_ids: list[str], web_urls: list[str] | None = None) -> type[BaseModel]:
    """Both paper_ids and web_urls are optional-guarded the same way: a
    Literal can't be built from an empty tuple, and a chat turn can now
    legitimately have papers only, web articles only, or both (round-2
    enhancement 5) — so whichever corpus is empty for this turn just has no
    corresponding cited_* field in the schema at all, rather than a field
    that (incorrectly) allows any value.
    """
    fields: dict = {
        "answerable": (bool, Field(description="True if the retrieved sources contain enough information to answer")),
        "answer": (str, Field(description="Natural-language answer with inline [Paper N]/[Web N] markers matching cited_paper_ids/cited_web_urls order")),
    }
    if paper_ids:
        paper_id_literal = Literal[tuple(paper_ids)]
        fields["cited_paper_ids"] = (
            list[paper_id_literal],
            Field(description="paper_ids supporting the answer, in [Paper 1],[Paper 2]... order; empty if not answerable"),
        )
    if web_urls:
        web_url_literal = Literal[tuple(web_urls)]
        fields["cited_web_urls"] = (
            list[web_url_literal],
            Field(description="web article urls supporting the answer, in [Web 1],[Web 2]... order; empty if not answerable"),
        )
    return create_model("ChatAnswer", **fields)


def _condense_question(history: list[dict], question: str, client: OpenAI, model: str = CONDENSE_MODEL) -> str:
    if not history:
        return question

    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation history:\n{transcript}\n\nFollow-up question: {question}"},
        ],
    )
    condensed = (response.choices[0].message.content or "").strip() or question
    if condensed != question:
        logger.info("Condensed follow-up %r -> standalone query %r", question, condensed)
    return condensed


def _recent_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """Caps history to the last max_turns user+assistant pairs (history
    always grows in pairs — see _no_sources_result and the end of ask()
    below), dropping older turns rather than letting the prompt sent on
    every call grow without bound as a conversation lengthens. A no-op for
    any conversation shorter than the cap."""
    return history[-2 * max_turns:]


def _no_sources_result(session: ChatSession, question: str, answer: str) -> dict:
    session.history.append({"role": "user", "content": question})
    session.history.append({"role": "assistant", "content": answer})
    return {
        "answer": answer, "answerable": False,
        "cited_papers": [], "retrieved_papers": [],
        "cited_web_articles": [], "retrieved_web_articles": [],
    }


def ask(
    session: ChatSession,
    question: str,
    client: OpenAI | None = None,
    top_k: int = TOP_K_DEFAULT,
) -> dict:
    """Answer a question grounded in session.papers and session.web_articles,
    using session.history for follow-up context. Appends the turn to
    session.history and returns {"answer", "answerable",
    "cited_papers": [Paper...], "retrieved_papers": [Paper...],
    "cited_web_articles": [WebArticle...], "retrieved_web_articles": [WebArticle...]}.
    """
    if not session.papers and not session.web_articles:
        return _no_sources_result(
            session, question,
            "No papers or web articles have been retrieved yet for this conversation — search a topic first.",
        )

    client = client or OpenAI()

    recent_history = _recent_history(session.history)
    standalone_query = _condense_question(recent_history, question, client)

    retrieved_papers: list[Paper] = []
    if session.papers:
        collection = get_chroma_collection()
        embed_and_index_papers(session.papers, collection=collection, client=client)
        ids = [p.paper_id for p in session.papers]
        retrieved = semantic_search(
            standalone_query, collection=collection, client=client, top_k=top_k,
            where={"paper_id": {"$in": ids}},
        )
        retrieved_papers = [p for p, _ in retrieved]

    # Web articles aren't re-ranked per question — the pool is small enough
    # (3-4 per session, per web_search.py's default) that including all of
    # it is simpler and no less relevant than embedding-ranking it would be.
    retrieved_web_articles = list(session.web_articles)

    if not retrieved_papers and not retrieved_web_articles:
        return _no_sources_result(session, question, "No indexed papers or web articles are available to answer this question.")

    papers_by_id = {p.paper_id: p for p in retrieved_papers}
    web_by_url = {a.url: a for a in retrieved_web_articles}
    schema = _build_answer_schema(list(papers_by_id), list(web_by_url) or None)

    context_sections = []
    if retrieved_papers:
        paper_context = "\n\n".join(
            f"paper_id: {p.paper_id}\ntitle: {p.title}\nabstract: {p.abstract or '(no abstract available)'}"
            for p in retrieved_papers
        )
        context_sections.append(f"Retrieved papers:\n\n{paper_context}")
    if retrieved_web_articles:
        web_context = "\n\n".join(
            f"url: {a.url}\ntitle: {a.title}\nsnippet: {a.snippet or '(no snippet available)'}"
            for a in retrieved_web_articles
        )
        context_sections.append(f"Retrieved web articles:\n\n{web_context}")

    messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
    messages.extend(recent_history)
    messages.append({
        "role": "user",
        "content": "\n\n".join(context_sections) + f"\n\nQuestion: {question}",
    })

    response = client.chat.completions.parse(
        model=ANSWER_MODEL,
        messages=messages,
        response_format=schema,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(f"Model refused to answer: {response.choices[0].message.refusal}")

    usage = response.usage
    logger.info(
        "ask: %d tokens billed (prompt=%d, completion=%d)",
        usage.total_tokens, usage.prompt_tokens, usage.completion_tokens,
    )

    # Defensive: don't trust the model to honor "empty if not answerable" on
    # its own — enforce it, since a fabricated citation on an "I can't
    # answer this" response would be worse than the field being redundant.
    cited_paper_ids = list(getattr(parsed, "cited_paper_ids", [])) if parsed.answerable else []
    cited_papers = [papers_by_id[pid] for pid in cited_paper_ids]

    cited_web_urls = list(getattr(parsed, "cited_web_urls", [])) if parsed.answerable else []
    cited_web_articles = [web_by_url[url] for url in cited_web_urls]

    session.history.append({"role": "user", "content": question})
    session.history.append({"role": "assistant", "content": parsed.answer})

    return {
        "answer": parsed.answer,
        "answerable": parsed.answerable,
        "cited_papers": cited_papers,
        "retrieved_papers": retrieved_papers,
        "cited_web_articles": cited_web_articles,
        "retrieved_web_articles": retrieved_web_articles,
    }
