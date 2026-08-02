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
import uuid
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from research_agent import qa
from research_agent.query_expansion import PaperPoolSession
from research_agent.report import regenerate_report_with_new_sources
from research_agent.schema import WebArticle
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
    schema, so a plain static model is enough. Shared verbatim between
    the web-search offer (Phase 5c) and the report-update offer (Phase
    6f) -- not a second classifier, just called with a different
    offer_description each time (see _classify_offer_response below)."""

    intent: Literal["accept", "decline", "other"] = Field(
        description=(
            "accept: the user clearly wants the offer carried out (e.g. 'yes', 'sure', 'go ahead'). "
            "decline: the user clearly does not want it (e.g. 'no thanks', 'not necessary'). "
            "other: the message is neither -- a new or unrelated question/statement, not a reply "
            "to the offer at all. Prefer 'other' whenever it's ambiguous."
        )
    )


_OFFER_CLASSIFIER_SYSTEM_PROMPT = """You are classifying how a user responded to an assistant's offer.

Classify the user's next message as exactly one of:
- "accept": clearly wants the offer carried out.
- "decline": clearly does not want it carried out.
- "other": neither -- e.g. a new or unrelated question, a change of subject, or anything that isn't primarily a reply to the offer itself.

When in doubt, choose "other" -- only choose accept/decline when the message is unambiguously a response to the offer."""


def _classify_offer_response(offer_description: str, message: str, client: OpenAI, model: str = qa.CONDENSE_MODEL) -> str:
    """curation-refinement-and-auto-offer Phase 6f: generalized from
    Phase 5c's web-search-only version to take a plain-language
    description of WHATEVER is being offered, rather than assuming it's
    always a web search -- reused as-is for the report-update offer
    (chat_turn() below), not duplicated into a second classifier."""
    messages = [
        {"role": "system", "content": _OFFER_CLASSIFIER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"The pending offer was to: {offer_description}\n\n"
                f"The user's next message: {message!r}"
            ),
        },
    ]
    response = client.chat.completions.parse(model=model, messages=messages, response_format=_OfferResponseIntent)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        # Same reasoning as qa.py's own refusal handling: a classifier that
        # can't decide must not be silently treated as an "accept" (would
        # trigger an unrequested action) or a "decline" (would silently
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


def _accept_web_offer(session: PaperPoolSession, message: str, client: OpenAI, top_k: int) -> dict:
    question = session.pending_web_offer["question"]
    session.pending_web_offer = None

    # chat-ux-fixes bug 2/4 (second pass): `question` is the RAW, literal
    # per-turn message that triggered the offer -- for a follow-up like
    # "I mean very recent like in the 2026?" that's an incoherent web
    # search query on its own (no subject; it only makes sense given
    # earlier turns' context). qa.py already resolves exactly this kind
    # of fragment into a real standalone question for its own paper-
    # retrieval condensing (condense_question/capped_history, made
    # public for this reuse rather than reinventing a second resolver).
    # Computed at ACCEPT time only, never at offer time -- a declined or
    # ignored offer never spends this extra LLM call (approved design:
    # showing the resolved query in the transcript AFTER accepting, not
    # in the offer prompt itself, is worth the offer-time call it saves).
    try:
        search_query = qa.condense_question(qa.capped_history(session.chat_history), question, client)
    except Exception:
        # Same defensive posture as the search_web guard right below --
        # an external-call failure here must degrade to the raw question
        # as the search query, not blow up the whole accept flow.
        logger.warning(
            "condense_question raised unexpectedly for %r -- using the raw question as the search query",
            question, exc_info=True,
        )
        search_query = question

    existing_urls = {a.url for a in session.web_articles_added}
    try:
        found = search_web(search_query)
    except Exception:
        # search_web()'s own docstring promises it never raises -- degrades
        # to [] on missing key, zero results, or any API/network error --
        # but this project has hit real, recurring flakiness from external
        # search APIs all session (arXiv, Semantic Scholar), so trusting
        # that contract alone here would be exactly the kind of assumption
        # this project's discipline says not to make. Same degrade-to-empty
        # outcome as a documented failure, just reached defensively instead
        # of by trusting the callee never to violate its own contract.
        logger.warning("search_web raised unexpectedly for query %r -- treating as no new results", search_query, exc_info=True)
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
    # chat-ux-fixes bug 4 (first pass): ask_in_session (via qa.ask()) just
    # appended {"role": "user", "content": question} -- the ORIGINAL
    # question, needed here for real retrieval/generation grounding -- as
    # the new user turn, making the same question appear a second time
    # verbatim in the transcript. chat-ux-fixes bug 4 (second pass): a
    # bare "yes" (that first fix) turned out not to be good enough either
    # -- every accept in a conversation looked identical, with no way to
    # tell what any of them were actually for. Replaced with a label
    # naming the actual resolved search query, so this turn is
    # self-describing in the transcript on its own. Same "patch the
    # freshly-appended turn after the fact" pattern _maybe_set_web_offer
    # above already uses for the assistant side (and _accept_report_
    # update below reuses again) -- this is the user-side equivalent.
    # qa.py's ask()/_no_sources_result always append exactly one user
    # then one assistant entry per call (documented and relied on by
    # qa.py's own capped_history), so the turn ask_in_session just added
    # is guaranteed to be at [-2], never anything upstream of it.
    if len(session.chat_history) >= 2:
        session.chat_history[-2] = {"role": "user", "content": f'Search the web for: "{search_query}"'}
    result = {**result, "web_search_used": True, "new_web_articles_found": len(new_articles)}
    # curation-refinement-and-auto-offer Phase 6f-3: the "immediately, as
    # soon as the trigger condition is true" point this design settled
    # on -- right after new web articles land, not deferred/batched.
    return _maybe_set_report_update_offer(session, result)


_REPORT_UPDATE_OFFER_SUFFIX = " I also found new web source(s) since the report was last generated — want me to update it to include them?"


def _maybe_set_report_update_offer(session: PaperPoolSession, result: dict) -> dict:
    """session.report_covered_web_article_count is the persisted
    equivalent of what Phase 6c's frontend originally tracked
    client-side only (and got wrong -- the banner never cleared after
    regenerating, since "any web article ever added" stays true
    forever). Comparing it against the CURRENT web_articles_added count
    is what actually answers "does the report on hand reflect what's
    been approved so far," not just "has anything ever been approved."
    """
    if session.report is None or len(session.web_articles_added) <= session.report_covered_web_article_count:
        return result
    session.pending_report_update = {
        "new_article_count": len(session.web_articles_added) - session.report_covered_web_article_count,
    }
    augmented_answer = result["answer"] + _REPORT_UPDATE_OFFER_SUFFIX
    if session.chat_history and session.chat_history[-1]["role"] == "assistant":
        session.chat_history[-1] = {"role": "assistant", "content": augmented_answer}
    return {**result, "answer": augmented_answer, "report_update_offer_made": True}


def _accept_report_update(session: PaperPoolSession, message: str, client: OpenAI) -> dict:
    """Unlike _accept_web_offer, doesn't route through ask_in_session at
    all -- there's no user question to re-answer here, just a report to
    regenerate -- so the user/assistant turn is appended directly,
    mirroring what ask_in_session would have done for any other turn."""
    new_count = session.pending_report_update.get("new_article_count", 1)
    session.pending_report_update = None
    session.report = regenerate_report_with_new_sources(session, client=client)
    session.report_covered_web_article_count = len(session.web_articles_added)

    answer = "Done — I've updated the report to include the newly approved web source(s)."
    # chat-ux-fixes bug 4 (second pass): a curated label instead of the
    # literal message (e.g. "yes") -- same reasoning as _accept_web_
    # offer's label above: a bare "yes" here is indistinguishable from
    # every other accept in the conversation.
    label = f"Update the report to include {new_count} new source{'s' if new_count != 1 else ''}"
    session.chat_history.append({"role": "user", "content": label})
    session.chat_history.append({"role": "assistant", "content": answer})
    return {
        "answer": answer,
        "answerable": True,
        "cited_papers": [],
        "cited_web_articles": [],
        "report_updated": True,
    }


def _chat_turn_impl(
    session: PaperPoolSession,
    message: str,
    client: OpenAI | None = None,
    top_k: int = qa.TOP_K_DEFAULT,
) -> dict:
    """The Phase 5c entry point: wraps ask_in_session() with the
    offer-and-decide web escalation mechanism, plus (Phase 6f-3) the
    same offer-and-decide shape reused for the automatic report-update
    offer. At most one of pending_web_offer/pending_report_update is
    ever set at once in practice (the latter is only set from inside
    _accept_web_offer, by which point the former has already resolved),
    checked in that order below. If either is currently pending, the
    message is first classified as accept/decline/other before anything
    else happens.

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
        offer_description = f"search the web for more on: {session.pending_web_offer['question']!r}"
        intent = _classify_offer_response(offer_description, message, client)
        if intent == "accept":
            return _accept_web_offer(session, message, client, top_k)
        # decline or other: clear the stale offer either way -- neither
        # case may leave it lingering for a later turn to misread.
        session.pending_web_offer = None
        result = ask_in_session(session, message, client=client, top_k=top_k)
        if intent == "decline":
            return {**result, "web_offer_declined": True}
        return _maybe_set_web_offer(session, message, result)

    # curation-refinement-and-auto-offer Phase 6f-3: the SAME
    # accept/decline/other classifier as above, just a different
    # offer_description -- reused, not duplicated. Same "unrelated
    # message must clear the offer" discipline applies here too (the
    # exact bug class the web offer hit once already): decline and
    # other both route through ask_in_session() rather than a canned
    # reply, so a trailing real question attached to either isn't
    # silently dropped.
    if session.pending_report_update is not None:
        new_count = session.pending_report_update.get("new_article_count", 1)
        offer_description = f"regenerate the literature review report to include {new_count} newly approved web source(s)"
        intent = _classify_offer_response(offer_description, message, client)
        if intent == "accept":
            return _accept_report_update(session, message, client)
        session.pending_report_update = None
        result = ask_in_session(session, message, client=client, top_k=top_k)
        if intent == "decline":
            return {**result, "report_update_declined": True}
        return _maybe_set_web_offer(session, message, result)

    result = ask_in_session(session, message, client=client, top_k=top_k)
    return _maybe_set_web_offer(session, message, result)


def _attach_exchange_metadata(session: PaperPoolSession, result: dict) -> None:
    """curation-chat-metadata Phase 1: stamps the exchange _chat_turn_impl
    just appended with a shared exchange_id and, on the assistant entry
    only, persisted per-answer metadata (used_web_search,
    cited_web_articles, added_to_report) later phases will read and
    (eventually) write.

    Invariant this relies on: exactly ONE user entry and ONE assistant
    entry get appended per _chat_turn_impl call, in that order, ending up
    at chat_history[-2] and chat_history[-1] -- guaranteed by every
    internal branch (offer accept/decline/other, report-update accept/
    decline/other, or the plain fallback; see qa.ask()'s and this
    module's own docstrings/comments, which already rely on this same
    fact). Pre-Phase-1 entries earlier in chat_history are never touched
    -- only the pair this call just produced.
    """
    if len(session.chat_history) < 2:
        return
    exchange_id = uuid.uuid4().hex
    session.chat_history[-2]["exchange_id"] = exchange_id
    assistant_turn = session.chat_history[-1]
    assistant_turn["exchange_id"] = exchange_id
    cited_web_articles = result.get("cited_web_articles") or []
    assistant_turn["used_web_search"] = bool(cited_web_articles)
    assistant_turn["cited_web_articles"] = [{"url": a.url, "title": a.title} for a in cited_web_articles]
    assistant_turn["added_to_report"] = False


def chat_turn(
    session: PaperPoolSession,
    message: str,
    client: OpenAI | None = None,
    top_k: int = qa.TOP_K_DEFAULT,
) -> dict:
    """Public entry point, same signature/behavior as before Phase 1.
    Thin wrapper around _chat_turn_impl (which holds the real
    offer-and-decide logic, unchanged) so metadata attachment happens in
    exactly one place regardless of which of _chat_turn_impl's several
    internal branches produced the result, and so a ValueError raised by
    _chat_turn_impl's own stage guard propagates before any chat_history
    mutation is attempted here.
    """
    result = _chat_turn_impl(session, message, client=client, top_k=top_k)
    _attach_exchange_metadata(session, result)
    return result


def delete_chat_exchanges(session: PaperPoolSession, exchange_ids: list[str]) -> tuple[list[str], bool]:
    """curation-chat-delete Phase 3: removes every chat_history entry whose
    exchange_id is in exchange_ids -- both the user question and assistant
    answer of a matching exchange, since chat_turn() (Phase 1) always
    stamps them with the same id. Entries with exchange_id None
    (pre-Phase-1 history) are NEVER touched, even if exchange_ids somehow
    contained an empty string -- deletion only ever matches a real,
    non-empty, non-None id.

    Idempotent: an exchange_id not present in chat_history at all is
    silently a no-op for that id, not an error -- matches this module's
    existing precedent of degrading gracefully on a client-supplied id
    that doesn't resolve to anything real (see select_paper_from_history's
    docstring in curation_session.py for the same principle applied to
    picks).

    Mutates session.chat_history in place is NOT relied upon here --
    reassigns it to a new, filtered list instead. Safe because nothing
    holds a live alias to it at this point (unlike mid-chat_turn(), where
    qa.ChatSession.history IS the same list by reference -- see
    _build_chat_session's own docstring): delete is always its own
    separate request, after any prior chat_turn() call has already
    returned and released its ChatSession.

    Returns (deleted_exchange_ids, report_possibly_stale):
      - deleted_exchange_ids: the subset of the requested ids that
        actually matched >=1 entry and were removed (sorted, never
        includes an id that matched nothing).
      - report_possibly_stale: True if any REMOVED assistant entry had
        added_to_report=True. Phase 3 deliberately does not regenerate or
        otherwise touch session.report here -- out of scope this phase;
        this is only a signal the API response carries for the frontend.
    """
    target_ids = {eid for eid in exchange_ids if eid}
    if not target_ids:
        return [], False

    matched_ids: set[str] = set()
    report_possibly_stale = False
    kept: list[dict] = []
    for turn in session.chat_history:
        turn_exchange_id = turn.get("exchange_id")
        if turn_exchange_id is not None and turn_exchange_id in target_ids:
            matched_ids.add(turn_exchange_id)
            if turn.get("role") == "assistant" and turn.get("added_to_report"):
                report_possibly_stale = True
            continue
        kept.append(turn)

    session.chat_history = kept
    return sorted(matched_ids), report_possibly_stale


# --- curation-chat-add-to-report Phase 4 ---

def _assistant_entries_by_exchange_id(session: PaperPoolSession) -> dict[str, dict]:
    """Exactly one assistant entry per real exchange_id, by the same
    invariant _attach_exchange_metadata relies on (one user + one
    assistant entry per chat_turn() call, sharing one exchange_id).
    Entries with exchange_id None (pre-Phase-1 history) are never keys
    here -- structurally ineligible for anything in this section."""
    return {
        turn["exchange_id"]: turn
        for turn in session.chat_history
        if turn.get("role") == "assistant" and turn.get("exchange_id")
    }


def select_eligible_exchanges_for_report(session: PaperPoolSession, exchange_ids: list[str]) -> tuple[list[str], list[str]]:
    """Partitions the requested exchange_ids (deduped, request order
    preserved) into (eligible, skipped). Eligible means: a real,
    resolvable exchange_id, whose assistant entry has used_web_search=True
    AND a non-empty cited_web_articles, AND is not already
    added_to_report=True. Everything else -- unknown id, a paper-only
    answer, or an already-added one -- lands in skipped, not an error;
    the caller decides whether an empty eligible list is itself an error.
    """
    assistant_by_id = _assistant_entries_by_exchange_id(session)
    seen: set[str] = set()
    eligible: list[str] = []
    skipped: list[str] = []
    for exchange_id in exchange_ids:
        if not exchange_id or exchange_id in seen:
            continue
        seen.add(exchange_id)
        turn = assistant_by_id.get(exchange_id)
        if (
            turn is not None
            and turn.get("used_web_search")
            and turn.get("cited_web_articles")
            and not turn.get("added_to_report")
        ):
            eligible.append(exchange_id)
        else:
            skipped.append(exchange_id)
    return eligible, skipped


def cited_web_article_urls_for_exchanges(session: PaperPoolSession, exchange_ids: list[str]) -> set[str]:
    """Union of cited_web_articles urls across the given exchange_ids --
    expected to already be the ELIGIBLE subset (select_eligible_exchanges_
    for_report's first return value), but reads defensively via .get()
    either way so an unknown/ineligible id just contributes nothing rather
    than raising."""
    assistant_by_id = _assistant_entries_by_exchange_id(session)
    urls: set[str] = set()
    for exchange_id in exchange_ids:
        turn = assistant_by_id.get(exchange_id)
        if turn is None:
            continue
        for article in turn.get("cited_web_articles") or []:
            urls.add(article["url"])
    return urls


def resolve_approved_web_articles_for_regeneration(session: PaperPoolSession, newly_approved_urls: set[str]) -> list[WebArticle]:
    """The one function that actually enforces "unapproved web articles
    never reach the report": filters session.web_articles_added (the raw,
    unfiltered pool) down to just the union of session.report_approved_
    web_article_urls (previously approved) and newly_approved_urls (this
    call's newly-eligible exchanges) -- by URL membership, nothing else.
    Anything in the raw pool whose URL isn't in that union is simply
    absent from the returned list, never passed to
    regenerate_report_with_approved_web_sources()."""
    approved_urls = session.report_approved_web_article_urls | newly_approved_urls
    return [article for article in session.web_articles_added if article.url in approved_urls]


def approve_web_article_urls(session: PaperPoolSession, urls: set[str]) -> None:
    """Only ever called AFTER a successful report regeneration (see the
    service layer) -- a failed regeneration must never grow the approved
    set, or a later retry would believe sources are already approved that
    were never actually reflected in any real report."""
    session.report_approved_web_article_urls = session.report_approved_web_article_urls | urls


def mark_exchanges_added_to_report(session: PaperPoolSession, exchange_ids: list[str]) -> None:
    """Flips added_to_report=True on the assistant entry of each given
    exchange_id. Same "only after success" rule as approve_web_article_
    urls above -- called from the same post-regeneration point, never
    before."""
    target_ids = set(exchange_ids)
    for turn in session.chat_history:
        if turn.get("role") == "assistant" and turn.get("exchange_id") in target_ids:
            turn["added_to_report"] = True
