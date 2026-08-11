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
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlparse

from openai import OpenAI
from pydantic import BaseModel, Field

import research_agent.telemetry as telemetry
from research_agent import qa
from research_agent.query_expansion import PaperPoolSession
from research_agent.report import (
    GENERATION_REASON_CHAT_AUTO_UPDATE,
    append_report_version,
    build_references_and_renumber,
    regenerate_report_with_new_sources,
)
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
    mutating it in place.

    chat-web-relevance-guardrails R7A: also populates the new
    ChatSession.topic from session.topic -- purely additive metadata as
    of this phase (see ChatSession's own docstring), not yet read by any
    qa.py graph node.

    chat-web-relevance-guardrails R7E.3: also populates ChatSession.
    web_article_provenance_by_url from session.web_article_provenance_by_url
    (R7E.2's own field) -- same pass-through convention as topic above,
    now actually read by _filter_web_relevance_node for the stale-pool
    re-check."""
    return qa.ChatSession(
        papers=session.selected_papers,
        web_articles=session.web_articles_added,
        history=session.chat_history,
        topic=session.topic,
        web_article_provenance_by_url=session.web_article_provenance_by_url,
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

# chat-web-relevance-guardrails R7B
_WEB_SEARCH_NO_RELEVANT_RESULTS_SUFFIX = (
    " I searched the web, but I did not find sources clearly relevant to this review topic."
)


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
    with telemetry.timed_child_call("classify_offer_response", "openai", model=model) as call:
        response = client.chat.completions.parse(model=model, messages=messages, response_format=_OfferResponseIntent)
        call.set_usage(response.usage)
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


def _record_web_article_provenance(session: PaperPoolSession, articles: list[WebArticle], source_query: str) -> None:
    """R7E.2: records one provenance entry per URL in `articles` --
    {"source_query", "added_at"} -- on session.web_article_provenance_by_url.
    Metadata only, read by no filtering logic yet (see that field's own
    docstring in query_expansion.py). `source_exchange_id` is
    deliberately NOT recorded here -- curation_chat.py's own
    _attach_exchange_metadata mints exchange_id AFTER _accept_web_offer
    (this function's only caller) already returns, so it doesn't exist
    yet at this point in the flow; wiring it through is a small,
    deferred follow-up (_accept_web_offer would need to report which
    URLs it just added back to its caller), not implemented in this
    phase. Overwrites any existing entry for a URL already present
    (matching web_articles_added's own "URL is the identity" convention
    -- existing_urls dedup in _accept_web_offer already means this is
    only ever called with genuinely new URLs in practice, but overwrite-
    not-append keeps this function correct even if that ever changes)."""
    added_at = datetime.now(timezone.utc).isoformat()
    for article in articles:
        session.web_article_provenance_by_url[article.url] = {
            "source_query": source_query,
            "added_at": added_at,
        }


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
    candidate_articles = [a for a in found if a.url not in existing_urls]

    # chat-web-relevance-guardrails R7B: the insertion-time gate. Judged
    # against search_query (the same string that actually drove the
    # search above, not the raw per-turn `question` fragment) AND
    # session.topic -- an article must clear both to ever join the
    # persistent web_articles_added pool. fail_open=False here on
    # purpose, unlike qa.py's own answer-time filter (which stays
    # fail-open) -- see _filter_relevant_web_articles's own docstring
    # for why these two call sites need opposite failure postures: this
    # is the ONLY gate deciding whether a brand-new article ever enters
    # a pool that outlives this turn, so a failure here must reject, not
    # silently admit whatever search_web happened to return.
    #
    # chat-web-relevance-guardrails R7E.3: deliberately does NOT pass
    # provenance_by_url here -- these are brand-new candidates that have
    # no provenance recorded yet (they're being judged BEFORE
    # _record_web_article_provenance below ever runs for them), so the
    # stale-pool re-check has nothing to apply to at this point in the
    # flow anyway. The stale-pool check only ever matters for the
    # answer-time re-filter of an ALREADY-persistent pool -- see
    # qa.py::_filter_web_relevance_node, the only call site that passes
    # provenance_by_url.
    #
    # chat-web-relevance-guardrails R7E.4: the temporal-freshness
    # re-check needs no new argument here either -- search_query and
    # each candidate's published_date are already in scope inside
    # _filter_relevant_web_articles, so a freshness-sensitive search
    # query (e.g. "latest AI regulation") already gets a known-stale
    # brand-new search result rejected right here at insertion, not just
    # caught later at answer time.
    #
    # chat-web-relevance-guardrails R7E.5: enable_direct_relevance_judge=
    # True here too -- this is one of the two REAL production call sites
    # (the other is qa.py's _filter_web_relevance_node, answer-time).
    # fail_open=False already set above governs the judge's own
    # uncertain/failure/over-cap fallback the same way it already governs
    # an embedding exception -- an unresolved gray-zone judgment must not
    # silently admit a brand-new article into the persistent pool, same
    # reasoning as every other insertion-time failure mode.
    relevant_articles = qa._filter_relevant_web_articles(
        search_query, candidate_articles, client, topic=session.topic, fail_open=False,
        enable_direct_relevance_judge=True,
    )
    # True only when search_web genuinely returned deduped candidates
    # AND every one of them failed relevance -- deliberately distinct
    # from search_web finding nothing at all (candidate_articles == []),
    # which keeps its existing, unmodified plain-refusal behavior below.
    web_search_found_nothing_relevant = bool(candidate_articles) and not relevant_articles
    session.web_articles_added.extend(relevant_articles)
    _record_web_article_provenance(session, relevant_articles, search_query)

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
    # chat-web-relevance-guardrails R7B: appended, not a replacement --
    # if papers or an already-approved older web source still answered
    # the question despite THIS search finding nothing relevant, the
    # user gets both the real answer and the honest caveat, not a hard
    # stop. Patches the same freshly-appended assistant turn the label
    # fix above patches the user side of -- same "answer text is not
    # final until this function returns" pattern _maybe_set_web_offer
    # and _maybe_set_report_update_offer both already rely on. Does NOT
    # re-set pending_web_offer (unlike _maybe_set_web_offer) -- see the
    # "deliberately does NOT run back through _maybe_set_web_offer"
    # comment above for why re-offering the same search would loop.
    if web_search_found_nothing_relevant:
        augmented_answer = result["answer"] + _WEB_SEARCH_NO_RELEVANT_RESULTS_SUFFIX
        if session.chat_history and session.chat_history[-1]["role"] == "assistant":
            session.chat_history[-1] = {"role": "assistant", "content": augmented_answer}
        result = {**result, "answer": augmented_answer}
    result = {
        **result, "web_search_used": True, "new_web_articles_found": len(relevant_articles),
    }
    # report-quality Phase R2D citation-revocation fix: this call only
    # ever ADDS to chat_history, never removes -- so no live_before
    # snapshot is needed, just un-revoke whatever the fresh answer
    # above actually cited (a genuine re-discovery/re-approval of a URL
    # previously marked revoked by an earlier delete/edit).
    session.revoked_web_article_urls -= live_cited_web_article_urls(session)
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
    mirroring what ask_in_session would have done for any other turn.

    report-quality Phase R3: the regenerated report is appended as a new
    version (generation_reason=GENERATION_REASON_CHAT_AUTO_UPDATE) via
    append_report_version, not assigned to session.report directly --
    this is one of the four real report-mutation call sites report-
    quality Phase R3 had to update, and the one most easily missed since
    it lives here rather than in either report service module.
    append_report_version keeps session.report mirrored to the new
    version as a side effect, so session.report is updated_report
    afterward exactly as it always was."""
    new_count = session.pending_report_update.get("new_article_count", 1)
    session.pending_report_update = None
    updated_report = regenerate_report_with_new_sources(session, client=client)
    append_report_version(session, updated_report, GENERATION_REASON_CHAT_AUTO_UPDATE)
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
    cited_web_articles, cited_papers, added_to_report) later phases will
    read and (eventually) write.

    Invariant this relies on: exactly ONE user entry and ONE assistant
    entry get appended per _chat_turn_impl call, in that order, ending up
    at chat_history[-2] and chat_history[-1] -- guaranteed by every
    internal branch (offer accept/decline/other, report-update accept/
    decline/other, or the plain fallback; see qa.ask()'s and this
    module's own docstrings/comments, which already rely on this same
    fact). Pre-Phase-1 entries earlier in chat_history are never touched
    -- only the pair this call just produced.

    report-quality Phase R3.2 Chunk 1: cited_papers is stamped the same
    way cited_web_articles already was -- result["cited_papers"] (real
    Paper objects, already present in qa.ask()'s own result dict for
    every code path, just never persisted before this) reduced down to
    lightweight {paper_id, title} dicts, NOT full Paper objects --
    session.selected_papers already holds the full data for the whole
    session, so a paper_id is enough to resolve a real citation/hyperlink
    later without duplicating paper data onto every chat turn.
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
    cited_papers = result.get("cited_papers") or []
    assistant_turn["cited_papers"] = [{"paper_id": p.paper_id, "title": p.title} for p in cited_papers]
    assistant_turn["added_to_report"] = False
    # chat-web-relevance-guardrails R7C: only stamped when there's a real
    # web citation to judge -- a turn with no cited_web_articles has
    # nothing for report-promotion eligibility to gate on this basis, so
    # leaving the key entirely absent (same treatment a pre-R7C turn
    # gets) avoids a meaningless True/False on a citation-less turn.
    # qa.ask()'s result carries web_relevance_verified on every code path
    # that can produce a citation; .get(..., False) is a defensive
    # fallback, not the expected case.
    if cited_web_articles:
        assistant_turn["web_relevance_verified"] = result.get("web_relevance_verified", False)


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
        chat-ux-report-semantics Phase B: when this is True,
        session.report_approved_web_article_urls is also pruned (see
        prune_report_approved_web_article_urls below) so a FUTURE
        selective regeneration can't resurrect a source that no longer
        has any added_to_report exchange backing it -- the already-
        generated session.report itself is still left untouched here,
        same as before.

        report-quality Phase R2D citation-revocation fix: independent of
        report_possibly_stale/added_to_report, session.revoked_web_
        article_urls is also synced (see _sync_revoked_web_article_urls)
        -- any web article URL that had a live chat backing before this
        delete and no longer does after it is recorded as revoked, so a
        LATER whole-pool regeneration (regenerate_report_with_new_
        sources) can permanently exclude it as a candidate even though
        session.web_articles_added itself is never pruned.
    """
    target_ids = {eid for eid in exchange_ids if eid}
    if not target_ids:
        return [], False

    live_before = live_cited_web_article_urls(session)
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
    if report_possibly_stale:
        prune_report_approved_web_article_urls(session)
    _sync_revoked_web_article_urls(session, live_before)
    return sorted(matched_ids), report_possibly_stale


def approved_web_article_urls_from_added_to_report_entries(session: PaperPoolSession) -> set[str]:
    """chat-ux-report-semantics Phase B: the invariant this whole feature
    exists to maintain -- session.report_approved_web_article_urls should
    always equal exactly this: the union of cited_web_articles URLs across
    every assistant chat_history entry that CURRENTLY has added_to_report
    =True. add_curation_chat_exchanges_to_report already keeps the two in
    lockstep on the way up (approve_web_article_urls and
    mark_exchanges_added_to_report are always called together, see their
    own docstrings); delete_chat_exchanges/edit_chat_exchange are the only
    two places that can shrink the added_to_report side (by removing or
    truncating entries), so they're the only two places that need to call
    prune_report_approved_web_article_urls below to keep the invariant
    holding on the way down too. Pure -- does not mutate session."""
    urls: set[str] = set()
    for turn in session.chat_history:
        if turn.get("role") == "assistant" and turn.get("added_to_report"):
            for article in turn.get("cited_web_articles") or []:
                urls.add(article["url"])
    return urls


def live_cited_web_article_urls(session: PaperPoolSession) -> set[str]:
    """report-quality Phase R2D citation-revocation fix: union of
    cited_web_articles URLs across every assistant entry CURRENTLY
    present in session.chat_history -- deliberately NOT filtered to
    added_to_report=True entries, unlike approved_web_article_urls_
    from_added_to_report_entries just above (which is scoped
    specifically to that narrower invariant). This answers the broader
    question "does ANY live chat exchange still back this source at
    all," which is what actually gets revoked when a chat exchange is
    deleted or edited away -- regardless of whether that source was
    ever run through the separate add-to-report approval flow. Pure --
    does not mutate session."""
    urls: set[str] = set()
    for turn in session.chat_history:
        if turn.get("role") == "assistant":
            for article in turn.get("cited_web_articles") or []:
                urls.add(article["url"])
    return urls


def _sync_revoked_web_article_urls(session: PaperPoolSession, live_before: set[str]) -> None:
    """report-quality Phase R2D citation-revocation fix: updates
    session.revoked_web_article_urls given a snapshot of what was live
    in chat BEFORE a chat_history-mutating operation (delete, edit, or
    a fresh web-search accept) -- any URL that was live before but isn't
    live now gets marked revoked (added to the set); any URL that's live
    now (including one marked revoked by an EARLIER call, now
    rediscovered/re-cited) gets un-revoked (removed from the set).
    Called with the CURRENT (post-mutation) session.chat_history already
    in place, so live_cited_web_article_urls(session) here reflects
    "after." Deliberately NOT gated on report_possibly_stale/added_to_
    report -- this tracks ANY loss of live chat backing, broader than
    that narrower invariant (see live_cited_web_article_urls's own
    docstring)."""
    live_after = live_cited_web_article_urls(session)
    newly_dead = live_before - live_after
    session.revoked_web_article_urls = (session.revoked_web_article_urls | newly_dead) - live_after


def prune_report_approved_web_article_urls(session: PaperPoolSession) -> None:
    """chat-ux-report-semantics Phase B: recomputes session.report_
    approved_web_article_urls FROM SCRATCH (not an intersection with the
    prior value) as exactly approved_web_article_urls_from_added_to_
    report_entries(session) -- see that function's docstring for the
    invariant this restores. Recompute-from-scratch rather than
    diffing/intersecting the removed entries is deliberate: it's
    self-correcting and can't drift, at the same O(len(chat_history)) cost
    the caller already pays to compute report_possibly_stale.

    Deliberately does NOT touch session.web_articles_added (the raw,
    unfiltered discovery pool -- see its own docstring in
    query_expansion.py) and does NOT regenerate session.report -- pruning
    only affects what a FUTURE selective regeneration is allowed to
    include, never the report already generated."""
    session.report_approved_web_article_urls = approved_web_article_urls_from_added_to_report_entries(session)


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


def _resolve_cited_web_article(entry: dict, web_by_url: dict[str, WebArticle]) -> WebArticle:
    """report-quality Phase R3.2 Chunk 2: resolves a chat turn's stored,
    lightweight {"url", "title"} cited_web_articles entry to a full
    WebArticle -- looked up in session.web_articles_added first (the
    raw discovery pool a cited web source should always be findable in,
    since a source can only ever have been cited by having first been
    retrieved through that exact pool). Falls back to a degraded
    WebArticle built from the lightweight data alone only for the
    unexpected case where it's missing from web_articles_added --
    source_domain is derived deterministically from the url itself
    (never fabricated, never an LLM call), snippet/published_date left
    empty/None -- so format_web_citation still produces something
    readable rather than an empty-domain artifact, and this can never
    crash regardless of what web_articles_added currently holds."""
    url = entry.get("url", "")
    if url in web_by_url:
        return web_by_url[url]
    return WebArticle(
        title=entry.get("title") or url, url=url, snippet="",
        published_date=None, source_domain=urlparse(url).netloc or url,
    )


def derive_chat_references(session: PaperPoolSession) -> dict:
    """report-quality Phase R3.2 Chunk 2: derives a GLOBAL, chat-scoped
    [N] citation numbering + references list from session.chat_history
    -- the chat-side counterpart to report.py's own report-scoped
    derivation, reusing the exact same proven algorithm (report.
    build_references_and_renumber, a public wrapper around report.py's
    own _build_references_and_renumber) rather than reimplementing
    grouped-marker parsing, raw-source-id-marker handling, invalid-
    marker stripping, and whitespace cleanup a second time.

    Independence from report numbering is structural, not just by
    convention: this function never reads session.report, never
    mutates it, and every call to build_references_and_renumber builds
    its own brand-new reference registry internally (see that
    function's own docstring) -- there is no shared state a chat-scoped
    and a report-scoped call could ever collide over, regardless of how
    many times either runs.

    Derived FRESH every call, from whatever chat_history currently is
    -- never persisted, and session.chat_history itself is never
    mutated (a fresh, independent sections_out dict is built from each
    qualifying turn's own data; build_references_and_renumber's own
    mutation happens to those copies, not the original stored dicts).
    This is exactly why delete/edit "just works" for chat references
    with no extra bookkeeping: call this again after a delete and the
    shorter chat_history naturally produces a clean, re-compacted 1..N
    sequence on its own.

    Only assistant turns with a real exchange_id participate -- same
    structural-ineligibility convention _assistant_entries_by_
    exchange_id above already uses for pre-Phase-1 legacy turns. A
    qualifying turn missing cited_papers/cited_web_articles entirely
    (predates report-quality Phase R3.2 Chunk 1) degrades to "this turn
    cited nothing" rather than crashing, via the same .get(..., [])
    discipline this module already uses throughout.

    Returns {"chat_history": [...], "references": [...]}. chat_history
    is the FULL list, same length/order as session.chat_history --
    every non-qualifying turn (a user turn, or an assistant turn with
    no exchange_id) passes through as a plain, independent dict copy
    with its content untouched; each qualifying assistant turn's own
    dict copy has just its "content" key replaced with the marker-
    rewritten version. references is the flat, deduped, numbered list
    those rewritten markers point into.
    """
    papers_by_id = {p.paper_id: p for p in session.selected_papers}
    web_by_url = {a.url: a for a in session.web_articles_added}

    exchange_ids: list[str] = []
    sections_out: dict[str, dict] = {}
    for turn in session.chat_history:
        exchange_id = turn.get("exchange_id")
        if turn.get("role") != "assistant" or not exchange_id:
            continue
        cited_papers = [
            papers_by_id[p["paper_id"]]
            for p in (turn.get("cited_papers") or [])
            if p.get("paper_id") in papers_by_id
        ]
        cited_web_articles = [_resolve_cited_web_article(a, web_by_url) for a in (turn.get("cited_web_articles") or [])]
        exchange_ids.append(exchange_id)
        sections_out[exchange_id] = {
            "content": turn["content"], "cited_papers": cited_papers, "cited_web_articles": cited_web_articles,
        }

    result = build_references_and_renumber(sections_out, tuple(exchange_ids))

    chat_history_out = []
    for turn in session.chat_history:
        exchange_id = turn.get("exchange_id")
        if turn.get("role") == "assistant" and exchange_id in sections_out:
            chat_history_out.append({**turn, "content": result[exchange_id]["content"]})
        else:
            chat_history_out.append(dict(turn))

    return {"chat_history": chat_history_out, "references": result["references"]}


def select_eligible_exchanges_for_report(session: PaperPoolSession, exchange_ids: list[str]) -> tuple[list[str], list[str]]:
    """Partitions the requested exchange_ids (deduped, request order
    preserved) into (eligible, skipped). Eligible means: a real,
    resolvable exchange_id, whose assistant entry has used_web_search=True
    AND a non-empty cited_web_articles, AND is not already
    added_to_report=True. Everything else -- unknown id, a paper-only
    answer, or an already-added one -- lands in skipped, not an error;
    the caller decides whether an empty eligible list is itself an error.

    chat-web-relevance-guardrails R7C: additionally excludes a turn whose
    web citations came from a fail-open (unverified) relevance check --
    `turn.get("web_relevance_verified", True) is not False`, so a
    missing key (pre-R7C turn, or a turn that never stamped one because
    it cited no web articles -- moot either way since the mechanical
    checks above already require cited_web_articles) is treated as
    eligible for backward compatibility, and only an EXPLICIT stored
    `False` excludes. This is the one place that stored judgment is
    actually enforced -- see _attach_exchange_metadata for where it's
    recorded.
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
            and turn.get("web_relevance_verified", True) is not False
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


# --- curation-chat-edit Phase 5 ---

def edit_chat_exchange(
    session: PaperPoolSession, exchange_id: str, new_question: str,
    client: OpenAI | None = None, top_k: int = qa.TOP_K_DEFAULT,
) -> tuple[dict, bool]:
    """curation-chat-edit Phase 5: truncate-and-regenerate, not an
    in-place edit. Locates the USER entry carrying exchange_id (edit only
    ever targets a question, never an answer) and truncates chat_history
    to everything strictly BEFORE it -- this removes the old answer to
    that question AND every later exchange in one slice, since
    chat_history is strictly append-only/chronological (Phase 3's delete
    only ever removes entries, never reorders them). No separate step
    ever looks for "the assistant answer" specifically -- truncating at
    the user entry's own index already removes it and everything
    chronologically after, which is also why "what if only one side
    exists" isn't a case this needs to handle specially.

    Raises ValueError if exchange_id doesn't resolve to a user entry --
    covers an unknown id, an id that only matches an assistant entry (the
    role=="user" filter below), and the case a pre-Phase-1 entry
    (exchange_id is None) could never match at all.

    pending_web_offer/pending_report_update are unconditionally cleared
    before the fresh chat_turn() call below -- NOT a rare edge case: both
    fields only ever describe state left by the chronologically LAST
    exchange, and an edit always truncates that exchange away too (the
    edited exchange_id can never be after the list's true end), so
    whatever most recently set either field is always being erased by
    any edit whatsoever. Leaving either set would let the edited
    question get misclassified as a reply to an offer about content that
    no longer exists.

    The actual new-answer generation is delegated to the existing,
    UNMODIFIED chat_turn() -- appends a fresh user+assistant pair with
    its own NEW exchange_id (this is not an in-place mutation of the old
    exchange_id), and inherits Phase 1's capped_history() sanitization
    for free since nothing here builds an LLM prompt directly.

    chat-ux-report-semantics Phase B: report_approved_web_article_urls is
    now PRUNED (not left untouched -- superseding the old "Option A" note
    this docstring used to have) whenever report_possibly_stale is True,
    via prune_report_approved_web_article_urls -- see that function's
    docstring for the invariant it restores. This still does not fix the
    staleness of the already-generated session.report itself (pruning
    only constrains a FUTURE selective regeneration); only a real
    regeneration does that, which this function still never triggers.

    Returns (chat_turn_result, report_possibly_stale) -- the latter is
    True if ANY assistant entry in the truncated-away range (the edited
    exchange's own old answer, or any later exchange) had
    added_to_report=True, computed BEFORE truncation.

    report-quality Phase R2D citation-revocation fix: session.revoked_
    web_article_urls is synced (_sync_revoked_web_article_urls) across
    this WHOLE operation -- truncation AND the fresh chat_turn() call
    together -- using a live-urls snapshot taken before either. A URL
    the truncated-away exchange(s) alone backed is recorded revoked; a
    URL the FRESH replacement answer re-cites (even if it's the exact
    same URL) is correctly left un-revoked, since the snapshot comparison
    only sees the net before/after effect of the edit as a whole.
    """
    user_idx = next(
        (i for i, turn in enumerate(session.chat_history) if turn.get("role") == "user" and turn.get("exchange_id") == exchange_id),
        None,
    )
    if user_idx is None:
        raise ValueError(f"No editable user question found for exchange_id {exchange_id!r}")

    live_before = live_cited_web_article_urls(session)
    removed = session.chat_history[user_idx:]
    report_possibly_stale = any(turn.get("role") == "assistant" and turn.get("added_to_report") for turn in removed)

    session.chat_history = session.chat_history[:user_idx]
    if report_possibly_stale:
        prune_report_approved_web_article_urls(session)
    session.pending_web_offer = None
    session.pending_report_update = None

    result = chat_turn(session, new_question, client=client, top_k=top_k)
    _sync_revoked_web_article_urls(session, live_before)
    return result, report_possibly_stale
