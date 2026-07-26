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

qa-langgraph-conversion: ask() is now a thin wrapper around a compiled
LangGraph StateGraph (_DEFAULT_GRAPH) instead of a plain function — laying
the foundation for summarization/checkpointing/human-in-the-loop work
planned on top of this later. This phase is a structural refactor only:
the graph reproduces today's exact routing (the two "no sources" early
exits and condense_question's first-turn skip are now conditional edges
instead of inline `if`s), not new behavior. A SqliteSaver checkpointer is
wired in and available (see sqlite_checkpointer() below) but NOT activated
on the default path — ask() runs exactly as statelessly as before. Its own
physical file (data/qa_checkpoints.sqlite) is deliberately separate from
storage.py's history.sqlite: different concern (conversation state vs.
saved-search identity), different connection lifecycle (a checkpointer
wants one long-lived connection; storage.py opens one per request).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, Literal

from langfuse import get_client, observe
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field, create_model

from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, semantic_search
from research_agent.schema import Paper, WebArticle
from research_agent.tracing import paper_metadata

logger = logging.getLogger(__name__)

# Separate from storage.py's DB_PATH (data/history.sqlite) by design — see
# module docstring. Not written to on the default (non-persistent) path;
# only used if a future phase activates sqlite_checkpointer() below.
QA_CHECKPOINT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "qa_checkpoints.sqlite"

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


@observe(name="condense_question", as_type="generation", capture_input=False, capture_output=False)
def _condense_question(history: list[dict], question: str, client: OpenAI, model: str = CONDENSE_MODEL) -> str:
    if not history:
        # No LLM call on the common first-turn case — nothing to condense
        # against, so the span is left with no generation details recorded,
        # same convention query_expansion.py's suggest_related_titles() uses
        # for its own early-return/skip case.
        return question

    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    messages = [
        {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Conversation history:\n{transcript}\n\nFollow-up question: {question}"},
    ]
    langfuse = get_client()
    langfuse.update_current_generation(input=messages, model=model)

    response = client.chat.completions.create(model=model, messages=messages)
    condensed = (response.choices[0].message.content or "").strip() or question
    if condensed != question:
        logger.info("Condensed follow-up %r -> standalone query %r", question, condensed)

    usage = response.usage
    if usage is not None:
        langfuse.update_current_generation(
            usage_details={
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        )
    langfuse.update_current_generation(output=condensed)
    return condensed


def _recent_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """Caps history to the last max_turns user+assistant pairs (history
    always grows in pairs — see _no_sources_result and the end of ask()
    below), dropping older turns rather than letting the prompt sent on
    every call grow without bound as a conversation lengthens. A no-op for
    any conversation shorter than the cap."""
    return history[-2 * max_turns:]


# Phase 2 (qa-langgraph-conversion): a fixed, deterministic set of short
# conversational closers/acknowledgments. Deliberately an EXACT-match
# allowlist, not a "starts with" or fuzzy check — "thanks, but can you also
# tell me about its limitations?" must NOT match, since only the bare
# closer itself is safe to skip retrieval/generation for. Anything not in
# this exact set (after normalization) falls through to the real answer
# path — a false negative here just costs one ordinary LLM call; a false
# positive means silently never answering a real question.
_NON_SUBSTANTIVE_PHRASES = {
    "thanks", "thank you",
    "thanks a lot", "thanks so much", "thank you so much", "many thanks",
    "much appreciated", "appreciate it", "appreciated",
    "ok", "okay", "k", "kk",
    "cool", "great", "nice", "perfect", "awesome", "cheers", "alright",
    "got it", "gotcha", "noted", "understood",
    "sounds good", "makes sense", "good to know",
    "that helps", "that works", "will do", "sure thing", "no worries", "all good", "fair enough",
}

# Independent of the phrase list above — a real follow-up question is
# almost always longer than this, so anything over the cap is never
# treated as non-substantive even if it happens to start with a closer
# phrase (same "thanks, but ..." trap case).
_NON_SUBSTANTIVE_MAX_WORDS = 5

_NON_SUBSTANTIVE_RESPONSE = "You're welcome! Let me know if you have any more questions about these papers."


def _is_non_substantive(question: str) -> bool:
    """Deterministic, deliberately conservative classifier for messages
    that need neither retrieval nor an LLM call at all (a bare "thanks!"
    after a real answer). Three independent signals must ALL agree to
    skip: (1) no question mark anywhere in the raw text, (2) short enough
    (word-count gate), (3) the normalized text is an EXACT match against a
    fixed closer-phrase allowlist. Any one of them failing means "don't
    skip" — biased hard against false positives: wrongly skipping a real
    question is a far worse failure than occasionally spending a real LLM
    call on a message that turns out to be a plain acknowledgment.
    """
    if "?" in question:
        return False

    normalized = " ".join(question.strip().lower().split())
    if not normalized:
        return False
    if len(normalized.split()) > _NON_SUBSTANTIVE_MAX_WORDS:
        return False

    stripped = normalized.rstrip("!.,;: ")
    return stripped in _NON_SUBSTANTIVE_PHRASES


def _no_sources_result(session: ChatSession, question: str, answer: str) -> dict:
    session.history.append({"role": "user", "content": question})
    session.history.append({"role": "assistant", "content": answer})
    # Called from both of the graph's "no sources" nodes (never had any
    # sources at all; had sources but nothing was retrieved this turn) — one
    # update here covers both without duplicating it at each call site.
    # Relies on ask()'s own @observe span still being the current span while
    # the graph's nodes run synchronously inside it, the same way any plain
    # helper called from a decorated function updates that function's span.
    langfuse = get_client()
    langfuse.update_current_span(
        input={"question": question},
        output={"answerable": False, "answer": answer},
    )
    return {
        "answer": answer, "answerable": False,
        "cited_papers": [], "retrieved_papers": [],
        "cited_web_articles": [], "retrieved_web_articles": [],
        # trace_id is NOT set here — ask() stamps it once on the final
        # result after the graph finishes, regardless of which node
        # produced it. See ask()'s own docstring/comment for why.
    }


@observe(name="generate_answer", as_type="generation", capture_input=False, capture_output=False)
def _generate_answer(messages: list[dict], schema: type[BaseModel], client: OpenAI, model: str = ANSWER_MODEL) -> BaseModel:
    langfuse = get_client()
    langfuse.update_current_generation(input=messages, model=model)

    response = client.chat.completions.parse(model=model, messages=messages, response_format=schema)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        langfuse.update_current_generation(output=None, level="WARNING", status_message="Model refused to answer")
        raise RuntimeError(f"Model refused to answer: {response.choices[0].message.refusal}")

    usage = response.usage
    logger.info(
        "ask: %d tokens billed (prompt=%d, completion=%d)",
        usage.total_tokens, usage.prompt_tokens, usage.completion_tokens,
    )
    if usage is not None:
        langfuse.update_current_generation(
            usage_details={
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        )
    langfuse.update_current_generation(output=parsed.model_dump())
    return parsed


class QAState(TypedDict):
    """State threaded through the graph for one ask() call. `session` is
    the same ChatSession object the caller passed in — held by reference,
    not copied, so a node appending to session.history (see
    _no_sources_result and _generate_node) mutates the caller's own object
    exactly as the old plain-function version did. Not yet a fully
    JSON-serializable state on purpose: persistence isn't activated on the
    default path (see module docstring), so there's no requirement yet for
    every field to survive a checkpoint round-trip. That requirement lands
    with whichever future phase actually turns persistence on.
    """

    session: ChatSession
    question: str
    recent_history: list[dict]
    client: OpenAI
    top_k: int
    is_non_substantive: bool
    standalone_query: str | None
    retrieved_papers: list[Paper]
    retrieved_web_articles: list[WebArticle]
    result: dict


def _classify_node(state: QAState) -> dict:
    """The graph's first real node (per the Phase 0 design) — runs before
    even the "do we have any sources" check, since a bare "thanks!" should
    short-circuit regardless of whether papers exist for this session."""
    is_skip = _is_non_substantive(state["question"])
    if is_skip:
        logger.info(
            "classify_message: treating %r as non-substantive — skipping retrieval/LLM calls",
            state["question"],
        )
        get_client().update_current_span(metadata={"non_substantive_skip": True})
    return {"is_non_substantive": is_skip}


def _non_substantive_node(state: QAState) -> dict:
    session = state["session"]
    question = state["question"]
    session.history.append({"role": "user", "content": question})
    session.history.append({"role": "assistant", "content": _NON_SUBSTANTIVE_RESPONSE})
    get_client().update_current_span(
        input={"question": question},
        output={"answerable": True, "answer": _NON_SUBSTANTIVE_RESPONSE},
    )
    return {
        "result": {
            "answer": _NON_SUBSTANTIVE_RESPONSE,
            "answerable": True,
            "cited_papers": [], "retrieved_papers": [],
            "cited_web_articles": [], "retrieved_web_articles": [],
        },
    }


def _route_after_classify(state: QAState) -> str:
    """Routes on classify_message's verdict first, then falls back to the
    original entry logic: today's "no sources at all" guard, plus
    condense_question's "skip on first turn" logic promoted from an inline
    `if` to a real edge — first-turn conversations go straight to retrieve
    with no LLM call, identical cost/behavior to before."""
    if state["is_non_substantive"]:
        return "non_substantive"
    session = state["session"]
    if not session.papers and not session.web_articles:
        return "no_sources"
    return "condense" if state["recent_history"] else "retrieve"


def _condense_node(state: QAState) -> dict:
    standalone = _condense_question(state["recent_history"], state["question"], state["client"])
    return {"standalone_query": standalone}


def _retrieve_node(state: QAState) -> dict:
    session = state["session"]
    query = state["standalone_query"] or state["question"]

    retrieved_papers: list[Paper] = []
    if session.papers:
        collection = get_chroma_collection()
        embed_and_index_papers(session.papers, collection=collection, client=state["client"])
        ids = [p.paper_id for p in session.papers]
        retrieved = semantic_search(
            query, collection=collection, client=state["client"], top_k=state["top_k"],
            where={"paper_id": {"$in": ids}},
        )
        retrieved_papers = [p for p, _ in retrieved]

    # Web articles aren't re-ranked per question — the pool is small enough
    # (3-4 per session, per web_search.py's default) that including all of
    # it is simpler and no less relevant than embedding-ranking it would be.
    retrieved_web_articles = list(session.web_articles)
    return {"retrieved_papers": retrieved_papers, "retrieved_web_articles": retrieved_web_articles}


def _route_retrieved(state: QAState) -> str:
    """Second routing decision: today's second "nothing retrieved this
    turn" guard, now an edge off retrieve instead of an inline `if`."""
    if not state["retrieved_papers"] and not state["retrieved_web_articles"]:
        return "no_sources"
    return "generate"


def _no_sources_initial_node(state: QAState) -> dict:
    result = _no_sources_result(
        state["session"], state["question"],
        "No papers or web articles have been retrieved yet for this conversation — search a topic first.",
    )
    return {"result": result}


def _no_sources_empty_node(state: QAState) -> dict:
    result = _no_sources_result(
        state["session"], state["question"],
        "No indexed papers or web articles are available to answer this question.",
    )
    return {"result": result}


def _generate_node(state: QAState) -> dict:
    session = state["session"]
    question = state["question"]
    top_k = state["top_k"]
    retrieved_papers = state["retrieved_papers"]
    retrieved_web_articles = state["retrieved_web_articles"]

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
    messages.extend(state["recent_history"])
    messages.append({
        "role": "user",
        "content": "\n\n".join(context_sections) + f"\n\nQuestion: {question}",
    })

    parsed = _generate_answer(messages, schema, state["client"], model=ANSWER_MODEL)

    # Defensive: don't trust the model to honor "empty if not answerable" on
    # its own — enforce it, since a fabricated citation on an "I can't
    # answer this" response would be worse than the field being redundant.
    cited_paper_ids = list(getattr(parsed, "cited_paper_ids", [])) if parsed.answerable else []
    cited_papers = [papers_by_id[pid] for pid in cited_paper_ids]

    cited_web_urls = list(getattr(parsed, "cited_web_urls", [])) if parsed.answerable else []
    cited_web_articles = [web_by_url[url] for url in cited_web_urls]

    session.history.append({"role": "user", "content": question})
    session.history.append({"role": "assistant", "content": parsed.answer})

    get_client().update_current_span(
        input={"question": question, "top_k": top_k},
        output={
            "answerable": parsed.answerable,
            "cited_papers": paper_metadata(cited_papers),
            "retrieved_papers": paper_metadata(retrieved_papers),
            "cited_web_articles": [{"url": a.url, "title": a.title} for a in cited_web_articles],
            "retrieved_web_articles": [{"url": a.url, "title": a.title} for a in retrieved_web_articles],
        },
    )

    return {
        "result": {
            "answer": parsed.answer,
            "answerable": parsed.answerable,
            "cited_papers": cited_papers,
            "retrieved_papers": retrieved_papers,
            "cited_web_articles": cited_web_articles,
            "retrieved_web_articles": retrieved_web_articles,
        },
    }


def build_qa_graph(checkpointer: BaseCheckpointSaver | None = None) -> object:
    """Builds and compiles the qa graph. checkpointer=None (the default
    used by _DEFAULT_GRAPH below) compiles a graph with no persistence at
    all — every invoke() is as stateless as the old plain-function ask()
    was. Passing a real checkpointer (see sqlite_checkpointer()) is how a
    future phase turns on cross-request conversation memory; not wired
    into ask()'s default path yet (qa-langgraph-conversion Phase 0 decision:
    lay the foundation, don't activate it in this phase).

    START ─► classify_message ─┬─"non_substantive"──────────────► non_substantive_response ─► END
                                ├─"no_sources"───────────────────► no_sources_initial ─► END
                                ├─"condense"──► condense_question ─┐
                                │                                   │
                                └─"retrieve"(first turn)────────────┤
                                                                     │
                                                     route_retrieved │
                                               ┌─"no_sources"────────┤
                                               ▼                      │
                                      no_sources_empty ─► END          └─"generate"─► generate_answer ─► END
    """
    graph = StateGraph(QAState)
    graph.add_node("classify_message", _classify_node)
    graph.add_node("non_substantive_response", _non_substantive_node)
    graph.add_node("condense_question", _condense_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("generate_answer", _generate_node)
    graph.add_node("no_sources_initial", _no_sources_initial_node)
    graph.add_node("no_sources_empty", _no_sources_empty_node)

    graph.add_edge(START, "classify_message")
    graph.add_conditional_edges("classify_message", _route_after_classify, {
        "non_substantive": "non_substantive_response",
        "no_sources": "no_sources_initial",
        "condense": "condense_question",
        "retrieve": "retrieve",
    })
    graph.add_edge("condense_question", "retrieve")
    graph.add_conditional_edges("retrieve", _route_retrieved, {
        "no_sources": "no_sources_empty",
        "generate": "generate_answer",
    })
    graph.add_edge("non_substantive_response", END)
    graph.add_edge("no_sources_initial", END)
    graph.add_edge("no_sources_empty", END)
    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=checkpointer)


# Compiled once at import time, no checkpointer — matches today's fully
# stateless-per-call behavior. ask() invokes this by default.
_DEFAULT_GRAPH = build_qa_graph()


@contextmanager
def sqlite_checkpointer(path: Path = QA_CHECKPOINT_DB_PATH):
    """Not used by ask() today — persistence is deliberately opt-in for now
    (see module docstring). Available for a future phase to activate real
    cross-request conversation memory via
    build_qa_graph(checkpointer=saver). Lives in its own physical .sqlite
    file, separate from storage.py's history.sqlite by design.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


@observe(name="ask", capture_input=False, capture_output=False)
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

    Runs on _DEFAULT_GRAPH (see build_qa_graph) internally as of
    qa-langgraph-conversion — this wrapper's signature and return shape are
    unchanged so every existing caller (api.py's /chat endpoint, RAGAS eval
    scripts) keeps working as-is.
    """
    client = client or OpenAI()
    recent_history = _recent_history(session.history)

    initial_state: QAState = {
        "session": session,
        "question": question,
        "recent_history": recent_history,
        "client": client,
        "top_k": top_k,
        "is_non_substantive": False,
        "standalone_query": None,
        "retrieved_papers": [],
        "retrieved_web_articles": [],
        "result": {},
    }
    final_state = _DEFAULT_GRAPH.invoke(initial_state)
    result = dict(final_state["result"])
    # Stamped once here regardless of which node produced the result — see
    # _no_sources_result's matching comment for why the per-node paths
    # don't set this themselves. Relies on ask()'s own @observe span still
    # being the current span once graph.invoke() returns, verified against
    # real Langfuse traces in Phase 4 rather than assumed.
    result["trace_id"] = get_client().get_current_trace_id()
    return result
