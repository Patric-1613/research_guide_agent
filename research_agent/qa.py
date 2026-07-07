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
   answer text uses inline [1], [2] markers (in the order of
   cited_paper_ids) so a claim in the answer can be traced to a specific
   paper without needing a full per-sentence structured breakdown — a
   reasonable middle ground for a conversational answer, versus phase 5's
   per-paper summaries where a full breakdown was already the natural unit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, create_model

from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, semantic_search
from research_agent.schema import Paper

logger = logging.getLogger(__name__)

# Condensing a follow-up question is a small, frequent, low-stakes rewrite —
# same cost tier as the phase-4 agent's tool orchestration. Answer synthesis
# is the user-facing, quality-sensitive step, so it gets the stronger model,
# same reasoning as phase 5's summarizer.
CONDENSE_MODEL = "gpt-4.1-mini"
ANSWER_MODEL = "gpt-4.1"

TOP_K_DEFAULT = 5

CONDENSE_SYSTEM_PROMPT = """Given a conversation history and a follow-up question, rewrite the follow-up as a standalone question that makes sense without the history — resolve pronouns and implicit references (e.g. "it", "that method", "the second one") to what they actually refer to.

If the follow-up question is already standalone (doesn't depend on the history), return it unchanged. Return ONLY the rewritten question, nothing else.
"""

ANSWER_SYSTEM_PROMPT = """You are a research assistant answering questions using ONLY the abstracts of the papers provided below. Do not use outside knowledge about these papers, their authors, or the topic beyond what the abstracts state.

If the provided abstracts do not contain enough information to answer the question, set answerable to false and explain in your answer what's missing — do not guess or fill the gap from general knowledge.

If you can answer, write a clear natural-language answer. Use inline bracket markers like [1], [2] to mark which paper supports each claim, in the order you list them in cited_paper_ids. Every claim should be traceable to at least one marker.
"""


@dataclass
class ChatSession:
    """Grounding set + running history for one conversation. `papers` is
    normally whatever a prior search/rerank (phases 2-4) produced for a
    topic — Q&A doesn't search on its own, only answers from what's already
    been retrieved.
    """

    papers: list[Paper] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)  # [{"role": "user"|"assistant", "content": str}]


def _build_answer_schema(paper_ids: list[str]) -> type[BaseModel]:
    paper_id_literal = Literal[tuple(paper_ids)]
    return create_model(
        "ChatAnswer",
        answerable=(bool, Field(description="True if the retrieved abstracts contain enough information to answer")),
        answer=(str, Field(description="Natural-language answer with inline [1], [2] markers matching cited_paper_ids order")),
        cited_paper_ids=(list[paper_id_literal], Field(description="paper_ids supporting the answer, in [1],[2]... order; empty if not answerable")),
    )


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


def ask(
    session: ChatSession,
    question: str,
    client: OpenAI | None = None,
    top_k: int = TOP_K_DEFAULT,
) -> dict:
    """Answer a question grounded in session.papers, using session.history
    for follow-up context. Appends the turn to session.history and returns
    {"answer", "answerable", "cited_papers": [Paper...], "retrieved_papers": [Paper...]}.
    """
    if not session.papers:
        answer = "No papers have been retrieved yet for this conversation — search a topic first."
        session.history.append({"role": "user", "content": question})
        session.history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "answerable": False, "cited_papers": [], "retrieved_papers": []}

    client = client or OpenAI()

    standalone_query = _condense_question(session.history, question, client)

    collection = get_chroma_collection()
    embed_and_index_papers(session.papers, collection=collection, client=client)
    ids = [p.paper_id for p in session.papers]
    retrieved = semantic_search(
        standalone_query, collection=collection, client=client, top_k=top_k,
        where={"paper_id": {"$in": ids}},
    )
    retrieved_papers = [p for p, _ in retrieved]

    if not retrieved_papers:
        answer = "No indexed papers are available to answer this question."
        session.history.append({"role": "user", "content": question})
        session.history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "answerable": False, "cited_papers": [], "retrieved_papers": []}

    papers_by_id = {p.paper_id: p for p in retrieved_papers}
    schema = _build_answer_schema(list(papers_by_id))

    context = "\n\n".join(
        f"paper_id: {p.paper_id}\ntitle: {p.title}\nabstract: {p.abstract or '(no abstract available)'}"
        for p in retrieved_papers
    )

    messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
    messages.extend(session.history)
    messages.append({
        "role": "user",
        "content": f"Retrieved papers:\n\n{context}\n\nQuestion: {question}",
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
    cited_paper_ids = list(parsed.cited_paper_ids) if parsed.answerable else []
    cited_papers = [papers_by_id[pid] for pid in cited_paper_ids]

    session.history.append({"role": "user", "content": question})
    session.history.append({"role": "assistant", "content": parsed.answer})

    return {
        "answer": parsed.answer,
        "answerable": parsed.answerable,
        "cited_papers": cited_papers,
        "retrieved_papers": retrieved_papers,
    }
