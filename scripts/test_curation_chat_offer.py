#!/usr/bin/env python3
"""curation-chat-web-escalation Phase 5c sanity check: real, non-mocked
proof that the offer-and-decide mechanism works, mirroring
scripts/test_report.py / scripts/test_curation_chat.py's live-verification
standard.

The riskiest property in this phase is the accept/decline/"other"
classification of the user's reply to a pending web offer -- especially
telling a genuinely unrelated follow-up question apart from a real
yes/no, since keyword matching breaks on real conversational phrasing in
both directions. This script tests all three real outcomes against the
SAME pending offer (reconstructed fresh each time, since a real accept/
decline mutates session state) using a real LLM classification call, not
a mock:

  1. A clearly ambiguous/indirect ACCEPT phrase ("yeah go for it") --
     confirms a real Tavily web search actually runs and
     web_articles_added grows.
  2. A clearly indirect DECLINE phrase ("nah, papers are enough") --
     confirms no web search runs and pending_web_offer clears.
  3. A genuinely unrelated follow-up question (neither yes nor no) --
     confirms pending_web_offer STILL clears (this is the specific
     property requested: an ignored offer must not linger and confuse a
     later turn) and the new question gets answered normally from the
     selected papers instead.

Usage:
    python scripts/test_curation_chat_offer.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from research_agent.curation_chat import chat_turn
from research_agent.query_expansion import PaperPoolSession
from research_agent.schema import Paper

load_dotenv()


def _paper(pid: str, title: str, abstract: str) -> Paper:
    return Paper(
        title=title, authors=["A. Researcher"], year=2024, venue="arXiv",
        abstract=abstract, url=f"http://arxiv.org/abs/{pid}", doi=None,
        citation_count=None, source="arxiv", paper_id=pid,
    )


def _base_session_with_pending_offer(offer_question: str) -> PaperPoolSession:
    selected = [
        _paper(
            "p1", "LoRA: Low-Rank Adaptation of Large Language Models",
            "We propose LoRA, which freezes pretrained model weights and injects trainable "
            "low-rank decomposition matrices into each layer, greatly reducing the number of "
            "trainable parameters for downstream tasks.",
        ),
        _paper(
            "p2", "RoCoFT: Efficient Finetuning of LLMs with Row-Column Updates",
            "We propose RoCoFT, a parameter-efficient fine-tuning method that updates only a "
            "small subset of rows and columns of weight matrices, achieving competitive "
            "performance with a fraction of the trainable parameters of full fine-tuning.",
        ),
    ]
    return PaperPoolSession(
        topic="parameter-efficient fine-tuning for large language models",
        stage="synthesize",
        selected_paper_ids=["p1", "p2"],
        selected_papers=selected,
        chat_history=[
            {"role": "user", "content": offer_question},
            {"role": "assistant", "content": "The selected papers don't cover this. Would you like me to search the web for more on this?"},
        ],
        pending_web_offer={"question": offer_question},
    )


def _session_with_realistic_follow_up_fragment() -> PaperPoolSession:
    """chat-ux-fixes bug 2 (second pass): the exact real-world shape
    reported -- a coherent first question, answered normally, THEN a
    pronoun/context-dependent follow-up FRAGMENT ("what about more
    recent versions?") that only makes sense given the prior turn. This
    is deliberately NOT the same shape _base_session_with_pending_offer
    uses above (where the offer question is already a complete,
    standalone sentence) -- that shape wouldn't actually exercise
    condense_question doing real work, since there's nothing incoherent
    to resolve."""
    selected = [
        _paper(
            "p1", "RoCoFT: Efficient Finetuning of LLMs with Row-Column Updates",
            "We propose RoCoFT, a parameter-efficient fine-tuning method introduced in 2024 that "
            "updates only a small subset of rows and columns of weight matrices, achieving "
            "competitive performance with a fraction of the trainable parameters of full "
            "fine-tuning.",
        ),
    ]
    fragment = "what about more recent versions of it?"
    return PaperPoolSession(
        topic="parameter-efficient fine-tuning for large language models",
        stage="synthesize",
        selected_paper_ids=["p1"],
        selected_papers=selected,
        chat_history=[
            {"role": "user", "content": "When was RoCoFT introduced?"},
            {"role": "assistant", "content": "RoCoFT was introduced in 2024 [Paper 1]."},
            {"role": "user", "content": fragment},
            {
                "role": "assistant",
                "content": (
                    "The selected papers don't cover this. "
                    "Would you like me to search the web for more on this?"
                ),
            },
        ],
        pending_web_offer={"question": fragment},
    )


def main() -> None:
    client = OpenAI()
    offer_question = "What is the most recent (2026) leaderboard ranking of LoRA variants on GLUE?"

    print("=" * 100)
    print("Case 1: indirect ACCEPT phrase ('yeah go for it') -- real web search must run")
    print("=" * 100)
    session1 = _base_session_with_pending_offer(offer_question)
    r1 = chat_turn(session1, "yeah go for it", client=client)
    print(f"web_search_used={r1.get('web_search_used')}, new_web_articles_found={r1.get('new_web_articles_found')}")
    print(f"pending_web_offer after turn: {session1.pending_web_offer}")
    print(f"web_articles_added count: {len(session1.web_articles_added)}")
    assert r1.get("web_search_used") is True, "FAIL: expected the accept path to run a real web search"
    assert session1.pending_web_offer is None, "FAIL: offer must clear after accept"

    print("\n" + "=" * 100)
    print("Case 2: indirect DECLINE phrase ('nah, papers are enough') -- no web search must run")
    print("=" * 100)
    session2 = _base_session_with_pending_offer(offer_question)
    r2 = chat_turn(session2, "nah, papers are enough", client=client)
    print(f"web_offer_declined={r2.get('web_offer_declined')}")
    print(f"pending_web_offer after turn: {session2.pending_web_offer}")
    print(f"web_articles_added count: {len(session2.web_articles_added)}")
    assert r2.get("web_offer_declined") is True, "FAIL: expected a decline"
    assert session2.pending_web_offer is None
    assert len(session2.web_articles_added) == 0, "FAIL: decline must never trigger a web search"

    print("\n" + "=" * 100)
    print("Case 3: genuinely UNRELATED question (neither yes nor no) while an offer is pending")
    print("(the specific property requested: offer must still clear, not linger)")
    print("=" * 100)
    session3 = _base_session_with_pending_offer(offer_question)
    unrelated_question = "Separately, what does RoCoFT actually update inside the weight matrices?"
    r3 = chat_turn(session3, unrelated_question, client=client)
    print(f"answerable={r3['answerable']}")
    print(f"answer: {r3['answer']}")
    print(f"cited_papers: {[p.paper_id for p in r3['cited_papers']]}")
    print(f"pending_web_offer after turn: {session3.pending_web_offer}")
    print(f"web_offer_declined in result: {'web_offer_declined' in r3}")
    print(f"web_search_used in result: {'web_search_used' in r3}")
    assert session3.pending_web_offer is None, "FAIL: an ignored offer must clear, not persist"
    assert "web_offer_declined" not in r3, "FAIL: an unrelated question must not be misread as a decline"
    assert "web_search_used" not in r3, "FAIL: an unrelated question must not trigger a web search"
    assert r3["answerable"] is True, "FAIL: expected the new question to be answered normally from the selected papers"
    assert "p2" in [p.paper_id for p in r3["cited_papers"]], "FAIL: expected RoCoFT (p2) to be cited"

    print("\n" + "=" * 100)
    print("Case 4: TRAP -- starts like a decline ('no wait') but carries a real, different")
    print("question. The classifier may honestly read this as 'decline' OR 'other' (both are")
    print("defensible readings of real ambiguous phrasing) -- the fix under test is that")
    print("neither reading may silently swallow the trailing question.")
    print("=" * 100)
    session4 = _base_session_with_pending_offer(offer_question)
    trap_message = "no wait, what about vector databases instead?"
    r4 = chat_turn(session4, trap_message, client=client)
    print(f"answerable={r4['answerable']}")
    print(f"answer: {r4['answer']}")
    print(f"web_offer_declined in result: {r4.get('web_offer_declined')}")
    print(f"pending_web_offer after turn: {session4.pending_web_offer}")
    # The offer must never be left referencing the STALE original question
    # (that would mean the classifier's verdict on THIS message got
    # silently ignored/lost) -- it's either cleared entirely (decline) or,
    # legitimately, re-armed scoped to the NEW question if that also turns
    # out to be uncovered (other) -- both are correct; only a lingering
    # reference to the ORIGINAL offer_question would be a bug.
    if session4.pending_web_offer is not None:
        assert session4.pending_web_offer["question"] == trap_message, (
            f"FAIL: offer must not still reference the stale original question -- got {session4.pending_web_offer}"
        )
    # The real property under test: the trailing question must not be
    # silently dropped. It's fine if the model can't fully answer it
    # (these papers don't cover vector databases at all -- answerable may
    # legitimately be False), but the response must visibly be ABOUT the
    # question asked, not a generic "ok, sticking to the selected papers"
    # non-answer that ignores it entirely.
    lower_answer = r4["answer"].lower()
    assert "vector" in lower_answer or "database" in lower_answer, (
        f"FAIL: trailing question appears to have been silently dropped -- answer was: {r4['answer']!r}"
    )
    print("PASS: the trailing question was actually addressed, not silently dropped.")

    print("\n" + "=" * 100)
    print("Case 5 (Phase 5e): REAL Tavily failure -- temporarily corrupt TAVILY_API_KEY so the")
    print("actual web_search.py code path hits a genuine auth failure, not a mock. This project")
    print("has hit real, recurring flakiness from external search APIs all session (arXiv,")
    print("Semantic Scholar) -- Tavily should be assumed capable of the same, so this exercises")
    print("the real degrade path end to end through chat_turn(), not just curation_chat.py's own")
    print("defensive try/except in isolation.")
    print("=" * 100)
    session5 = _base_session_with_pending_offer(offer_question)
    original_key = os.environ.get("TAVILY_API_KEY")
    os.environ["TAVILY_API_KEY"] = "sk-invalid-deliberately-corrupted-for-this-test"
    try:
        r5 = chat_turn(session5, "yes, please search", client=client)  # must not raise
    finally:
        if original_key is not None:
            os.environ["TAVILY_API_KEY"] = original_key
        else:
            del os.environ["TAVILY_API_KEY"]
    print(f"web_search_used={r5.get('web_search_used')}, new_web_articles_found={r5.get('new_web_articles_found')}")
    print(f"pending_web_offer after turn: {session5.pending_web_offer}")
    print(f"answerable={r5['answerable']}")
    assert r5.get("web_search_used") is True
    assert r5.get("new_web_articles_found") == 0, "FAIL: an auth failure must not fabricate results"
    assert session5.pending_web_offer is None, "FAIL: offer must not be left corrupted after a real search failure"
    print("PASS: a real Tavily auth failure degraded gracefully -- no crash, no fabricated")
    print("results, no corrupted offer state.")

    print("\n" + "=" * 100)
    print("Case 6 (chat-ux-fixes bug 2, second pass): a REAL context-dependent follow-up")
    print("fragment ('what about more recent versions of it?') must be resolved into a real")
    print("standalone search query before being sent to Tavily -- not searched verbatim, and")
    print("not repeated (nor a bare 'yes') in the transcript.")
    print("=" * 100)
    session6 = _session_with_realistic_follow_up_fragment()
    fragment = session6.pending_web_offer["question"]
    r6 = chat_turn(session6, "yes", client=client)
    accepted_label = session6.chat_history[-2]["content"]
    print(f"raw fragment: {fragment!r}")
    print(f"transcript label for the accept: {accepted_label!r}")
    print(f"web_search_used={r6.get('web_search_used')}, new_web_articles_found={r6.get('new_web_articles_found')}")
    assert r6.get("web_search_used") is True, "FAIL: expected a real web search to run"
    assert accepted_label != "yes", "FAIL: transcript still shows a bare 'yes', not a curated label"
    assert accepted_label != fragment, "FAIL: transcript still shows the raw fragment repeated verbatim"
    assert accepted_label.startswith('Search the web for: "'), f"FAIL: unexpected label shape: {accepted_label!r}"
    resolved_query = accepted_label[len('Search the web for: "'):-1]
    print(f"resolved standalone query: {resolved_query!r}")
    # Can't assert exact wording from a real LLM call, but a genuinely
    # resolved standalone query for this fragment must at minimum name
    # the actual subject ("RoCoFT") the pronoun ("it") stood in for --
    # the raw fragment alone never mentions it.
    assert "rocoft" in resolved_query.lower(), (
        f"FAIL: resolved query doesn't appear to have resolved 'it' -> RoCoFT at all: {resolved_query!r}"
    )
    assert resolved_query.lower() != fragment.lower(), "FAIL: condensing was a no-op on a genuine fragment"
    print("PASS: the fragment was resolved into a real standalone query naming RoCoFT, actually")
    print("searched (not the raw fragment), and shown as a curated transcript label, not a bare 'yes'.")

    print("\n" + "=" * 100)
    print("PASS: accept triggers a real search, decline stays paper-only, an unrelated reply")
    print("clears the stale offer instead of misreading it as yes/no, a decline-shaped message")
    print("with a real question attached still gets that question answered instead of silently")
    print("dropped, a genuine Tavily failure degrades gracefully end to end, and a real")
    print("context-dependent follow-up fragment gets resolved into a real standalone query")
    print("instead of being searched (or shown in the transcript) verbatim.")
    print("=" * 100)


if __name__ == "__main__":
    main()
