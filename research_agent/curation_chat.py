"""curation-chat-web-escalation Phase 5b/5c: baseline chat grounded in a
curation session's exact selected_papers set (5b), plus the
offer-and-decide web escalation mechanism on top (5c).

Per the approved Phase 5a design, this is deliberately plain functions,
not a graph node -- there's no pause/resume/branch decision that needs
interrupt() here (that's what Phase 3's interrupt-based loop already
resolved by the time session.stage == "synthesize"), matching
report.py's own plain-function precedent. The grounding mechanism
itself is not reimplemented: this module constructs a real
qa.ChatSession from the session's own state and calls qa.ask()
completely unmodified, so it inherits qa.py's existing
Literal-constrained citation guarantee for free instead of re-deriving
it.

Phase 5c's offer-and-decide design: qa.ask()'s existing answerable=False
signal (papers/web don't cover the question -- qa.py already computes
this for its own refusal behavior) is reused as the trigger for
offering a web search, rather than inventing a second "does this need
the web" classifier. When answerable is False, chat_turn() appends a
web-search offer to the answer and records it on
session.pending_web_offer (just the question -- enough to resume
without needing interrupt()/a graph). The NEXT chat_turn() call, if a
pending offer exists, classifies the new message as accept/decline/
"other" (a genuinely new/unrelated question -- neither yes nor no) via
a small LLM call, since keyword-matching yes/no phrasing is exactly the
kind of thing real conversational language breaks in both directions
(matching qa.py's own reasoning for using embedding similarity instead
of an exact-match allowlist for acknowledgments). "other" is also the
safe fallback if the classifier itself refuses to answer -- an
unresolved offer must never silently persist and get misread as a
yes/no on some later, unrelated turn.
"""

from __future__ import annotations

import logging
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from research_agent import qa
from research_agent.query_expansion import PaperPoolSession
from research_agent.web_search import search_web

logger = logging.getLogger(__name__)


def _build_chat_session(session: PaperPoolSession) -> qa.ChatSession:
    """session.chat_history is passed by reference, not copied -- qa.ask()
    appends turns directly onto the ChatSession's .history list, so this
    is the same list object as session.chat_history, and the explicit
    reassignment in ask_in_session() below (rather than relying on that
    aliasing) is what actually keeps the two in sync regardless of
    whether qa.py's internals ever start reassigning the list instead of
    mutating it in place."""
    return qa.ChatSession(
        papers=session.selected_papers,
        web_articles=session.web_articles_added,
        history=session.chat_history,
    )


def ask_in_session(
    session: PaperPoolSession,
    question: str,
    client: OpenAI | None = None,
    top_k: int = qa.TOP_K_DEFAULT,
) -> dict:
    """Answers `question` grounded in session.selected_papers (and any
    web_articles_added), updating session.chat_history with the new
    turn. Returns qa.ask()'s own result dict unchanged: {"answer",
    "answerable", "cited_papers", "retrieved_papers",
    "cited_web_articles", "retrieved_web_articles", "trace_id"}.

    No empty-selection guard here -- qa.ask() already handles the
    no-papers-and-no-web-articles case cleanly (routes to its own
    "no_sources" path), so re-checking it here would just duplicate
    that behavior instead of reusing it.
    """
    chat_session = _build_chat_session(session)
    result = qa.ask(chat_session, question, client=client, top_k=top_k)
    session.chat_history = chat_session.history
    return result


_WEB_OFFER_SUFFIX = " Would you like me to search the web for more on this?"


class _OfferResponseIntent(BaseModel):
    """Fixed 3-way choice, unlike qa.py's per-call dynamic Literals
    (paper_ids/web_urls) -- there's nothing data-dependent about this
    schema, so a plain static model is enough."""

    intent: Literal["accept", "decline", "other"] = Field(
        description=(
            "accept: the user wants the web search to happen (e.g. 'yes', 'sure', 'go ahead'). "
            "decline: the user does not want a web search (e.g. 'no thanks', 'not necessary'). "
            "other: the message is neither -- a new or unrelated question/statement, not a reply "
            "to the offer at all. Prefer 'other' whenever it's ambiguous."
        )
    )


_OFFER_CLASSIFIER_SYSTEM_PROMPT = """You are classifying how a user responded to an assistant's offer to search the web, since the currently selected papers didn't fully cover their previous question.

Classify the user's next message as exactly one of:
- "accept": clearly wants the web search to happen.
- "decline": clearly does not want a web search.
- "other": neither -- e.g. a new or unrelated question, a change of subject, or anything that isn't primarily a reply to the offer itself.

When in doubt, choose "other" -- only choose accept/decline when the message is unambiguously a response to the offer."""


def _classify_offer_response(offer_question: str, message: str, client: OpenAI, model: str = qa.CONDENSE_MODEL) -> str:
    messages = [
        {"role": "system", "content": _OFFER_CLASSIFIER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"The pending offer was to search the web for more on: {offer_question!r}\n\n"
                f"The user's next message: {message!r}"
            ),
        },
    ]
    response = client.chat.completions.parse(model=model, messages=messages, response_format=_OfferResponseIntent)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        # Same reasoning as qa.py's own refusal handling: a classifier that
        # can't decide must not be silently treated as an "accept" (would
        # trigger an unrequested web search) or a "decline" (would silently
        # drop a real pending offer) -- "other" clears the offer and falls
        # through to answering the message as an ordinary fresh question,
        # the same outcome the user gets when their message genuinely was
        # unrelated to begin with.
        return "other"
    return parsed.intent


def _maybe_set_web_offer(session: PaperPoolSession, question: str, result: dict) -> dict:
    if result["answerable"]:
        return result
    session.pending_web_offer = {"question": question}
    augmented_answer = result["answer"] + _WEB_OFFER_SUFFIX
    if session.chat_history and session.chat_history[-1]["role"] == "assistant":
        session.chat_history[-1] = {"role": "assistant", "content": augmented_answer}
    return {**result, "answer": augmented_answer, "web_offer_made": True}


def _accept_web_offer(session: PaperPoolSession, client: OpenAI, top_k: int) -> dict:
    question = session.pending_web_offer["question"]
    session.pending_web_offer = None

    existing_urls = {a.url for a in session.web_articles_added}
    try:
        found = search_web(question)
    except Exception:
        # search_web()'s own docstring promises it never raises -- degrades
        # to [] on missing key, zero results, or any API/network error --
        # but this project has hit real, recurring flakiness from external
        # search APIs all session (arXiv, Semantic Scholar), so trusting
        # that contract alone here would be exactly the kind of assumption
        # this project's discipline says not to make. Same degrade-to-empty
        # outcome as a documented failure, just reached defensively instead
        # of by trusting the callee never to violate its own contract.
        logger.warning("search_web raised unexpectedly for query %r -- treating as no new results", question, exc_info=True)
        found = []
    new_articles = [a for a in found if a.url not in existing_urls]
    session.web_articles_added.extend(new_articles)

    # Deliberately does NOT run back through _maybe_set_web_offer: if the
    # re-asked question is still unanswerable even with fresh web results
    # (or search_web found nothing new at all -- it degrades to [] rather
    # than raising, see web_search.py), immediately re-offering the same
    # search the user just accepted would loop pointlessly. A still-open
    # information gap after a genuine web search attempt is reported
    # plainly instead.
    result = ask_in_session(session, question, client=client, top_k=top_k)
    return {**result, "web_search_used": True, "new_web_articles_found": len(new_articles)}


def chat_turn(
    session: PaperPoolSession,
    message: str,
    client: OpenAI | None = None,
    top_k: int = qa.TOP_K_DEFAULT,
) -> dict:
    """The Phase 5c entry point: wraps ask_in_session() with the
    offer-and-decide web escalation mechanism. If a web offer is
    currently pending, the message is first classified as accept/
    decline/other before anything else happens.

    decline and other both route through ask_in_session() rather than a
    canned short-circuit for decline -- a message the classifier reads
    as "decline" is not guaranteed to be JUST a decline: "no wait, what
    about vector databases instead?" starts exactly like a decline but
    carries a real, unrelated question a canned "ok, sticking to the
    selected papers" ack would have silently swallowed. qa.ask()'s own
    classify_message gate -- already proven against this precise kind of
    trap phrasing via its question-mark veto and content-override words,
    see qa.py's "thanks, but explain" case -- is what decides whether
    there's a real question left to answer, not a second, redundant
    classifier bolted on here that could get it wrong the same way
    _classify_offer_response just did.

    decline deliberately does NOT then run through _maybe_set_web_offer,
    unlike other -- confirmed by real testing (not assumed): a decline
    phrase qa.py's non-substantive gate doesn't happen to recognize
    (e.g. "nah, papers are enough") falls through to a real answer
    attempt, which naturally comes back unanswerable since it isn't a
    real question -- and re-offering a web search built from the
    decline text itself would be a nonsensical loop. A genuinely new
    question attached to a decline still gets answered (that's the bug
    this design fixes); it just doesn't also get its own fresh web
    offer within the same turn -- asking it again on a later turn will.

    Refuses cleanly if the session hasn't finished curation yet, same
    guard pattern as report.py's generate_report_for_session(). Guards
    on stage == "synthesize", not the "chat" value query_expansion.py's
    own PaperPoolSession docstring names as this phase's stage -- checked
    directly against the actual state machine (not assumed): nothing in
    the codebase has ever transitioned a session into "chat"; curation_
    loop.py only ever sets stage to "synthesize" once curation finishes,
    the exact same point report.py already treats as "ready." Guarding
    on a value nothing ever reaches would make chat permanently
    unusable, so this mirrors report.py's real, reachable gate instead.
    """
    if session.stage != "synthesize":
        raise ValueError(
            f"Session is not ready for chat (stage={session.stage!r}, expected 'synthesize') -- "
            "curation must finish (target met, user stopped, or topic exhausted) before chatting."
        )
    client = client or OpenAI()

    if session.pending_web_offer is not None:
        intent = _classify_offer_response(session.pending_web_offer["question"], message, client)
        if intent == "accept":
            return _accept_web_offer(session, client, top_k)
        # decline or other: clear the stale offer either way -- neither
        # case may leave it lingering for a later turn to misread.
        session.pending_web_offer = None
        result = ask_in_session(session, message, client=client, top_k=top_k)
        if intent == "decline":
            return {**result, "web_offer_declined": True}
        return _maybe_set_web_offer(session, message, result)

    result = ask_in_session(session, message, client=client, top_k=top_k)
    return _maybe_set_web_offer(session, message, result)
