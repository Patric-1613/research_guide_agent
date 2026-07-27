#!/usr/bin/env python3
"""curation-refinement-and-auto-offer Phase 6f-3 sanity check: real,
non-mocked proof that the automatic report-update offer works, mirroring
scripts/test_curation_chat_offer.py's (Phase 5c) live-verification
standard exactly -- including the specific "unrelated message must still
clear the offer" property that bit this project once already.

Sequence:
  1. Real report generation over two real papers.
  2. A real web-offer accept (real Tavily search) -- this is what
     ACTUALLY sets pending_report_update via the real mechanism, not a
     hand-constructed session state.
  3. Three separate copies of that exact post-accept session (same
     pattern as Phase 5c's own script) tested against real, deliberately
     ambiguous phrasing:
       a. An indirect ACCEPT ("yeah go ahead and update it") -- real
          regenerate_report_with_new_sources() call, report changes.
       b. An indirect DECLINE ("nah, leave it as is") -- no regeneration,
          report unchanged.
       c. A genuinely UNRELATED question -- pending_report_update must
          still clear (not linger to confuse a later turn), and the new
          question gets answered normally.

Usage:
    python scripts/test_report_update_offer_live.py
"""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from research_agent.curation_chat import chat_turn
from research_agent.query_expansion import PaperPoolSession
from research_agent.report import generate_report_for_session
from research_agent.schema import Paper

load_dotenv()


def _paper(pid: str, title: str, abstract: str) -> Paper:
    return Paper(
        title=title, authors=["A. Researcher"], year=2024, venue="arXiv",
        abstract=abstract, url=f"http://arxiv.org/abs/{pid}", doi=None,
        citation_count=None, source="arxiv", paper_id=pid,
    )


def main() -> None:
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
    session = PaperPoolSession(
        topic="parameter-efficient fine-tuning for large language models",
        stage="synthesize", selected_paper_ids=["p1", "p2"], selected_papers=selected,
    )

    client = OpenAI()

    print("=" * 100)
    print("Step 1: real initial report generation")
    print("=" * 100)
    session.report = generate_report_for_session(session, client=client)
    session.report_covered_web_article_count = len(session.web_articles_added)
    original_findings = session.report["findings"]["content"]
    print(f"Findings (first 150 chars): {original_findings[:150]}...")

    print("\n" + "=" * 100)
    print("Step 2: real web-offer accept -- this is what actually sets pending_report_update")
    print("=" * 100)
    session.pending_web_offer = {"question": "What is the latest 2026 GLUE leaderboard ranking for these exact methods?"}
    r = chat_turn(session, "yes please search", client=client)
    print(f"web_search_used={r.get('web_search_used')}, new_web_articles_found={r.get('new_web_articles_found')}")
    print(f"report_update_offer_made={r.get('report_update_offer_made')}")
    print(f"pending_report_update: {session.pending_report_update}")
    assert session.pending_report_update is not None, "FAIL: expected the real web-offer accept to set a real report-update offer"
    print("PASS: a real web source was approved and the report-update offer was set for real.")

    base_session = copy.deepcopy(session)

    print("\n" + "=" * 100)
    print("Case A: indirect ACCEPT ('yeah go ahead and update it') -- real regeneration must happen")
    print("=" * 100)
    session_a = copy.deepcopy(base_session)
    ra = chat_turn(session_a, "yeah go ahead and update it", client=client)
    print(f"report_updated={ra.get('report_updated')}")
    print(f"pending_report_update after turn: {session_a.pending_report_update}")
    new_findings = session_a.report["findings"]["content"]
    print(f"New findings (first 150 chars): {new_findings[:150]}...")
    assert ra.get("report_updated") is True, "FAIL: expected a real regeneration to happen"
    assert session_a.pending_report_update is None
    assert session_a.report_covered_web_article_count == len(session_a.web_articles_added)
    print("PASS: real regeneration happened and the offer cleared.")

    print("\n" + "=" * 100)
    print("Case B: indirect DECLINE ('nah, leave it as is') -- no regeneration, report unchanged")
    print("=" * 100)
    session_b = copy.deepcopy(base_session)
    report_before = session_b.report["findings"]["content"]
    rb = chat_turn(session_b, "nah, leave it as is", client=client)
    print(f"report_update_declined={rb.get('report_update_declined')}")
    print(f"pending_report_update after turn: {session_b.pending_report_update}")
    assert rb.get("report_update_declined") is True, "FAIL: expected a decline"
    assert session_b.pending_report_update is None
    assert session_b.report["findings"]["content"] == report_before, "FAIL: decline must never regenerate the report"
    print("PASS: declined for real, report genuinely untouched.")

    print("\n" + "=" * 100)
    print("Case C: genuinely UNRELATED question while the report-update offer is pending")
    print("(the specific property requested: offer must still clear, not linger)")
    print("=" * 100)
    session_c = copy.deepcopy(base_session)
    unrelated_question = "Separately, what does RoCoFT actually update inside the weight matrices?"
    rc = chat_turn(session_c, unrelated_question, client=client)
    print(f"answerable={rc['answerable']}")
    print(f"answer: {rc['answer']}")
    print(f"cited_papers: {[p.paper_id for p in rc['cited_papers']]}")
    print(f"pending_report_update after turn: {session_c.pending_report_update}")
    print(f"report_update_declined in result: {'report_update_declined' in rc}")
    print(f"report_updated in result: {'report_updated' in rc}")
    assert session_c.pending_report_update is None, "FAIL: an ignored offer must clear, not persist"
    assert "report_update_declined" not in rc, "FAIL: an unrelated question must not be misread as a decline"
    assert "report_updated" not in rc, "FAIL: an unrelated question must not trigger a regeneration"
    assert rc["answerable"] is True, "FAIL: expected the new question to be answered normally from the selected papers"
    assert "p2" in [p.paper_id for p in rc["cited_papers"]], "FAIL: expected RoCoFT (p2) to be cited"

    print("\n" + "=" * 100)
    print("PASS: accept triggers a real regeneration, decline leaves the report genuinely")
    print("untouched, and an unrelated reply clears the stale offer instead of misreading it")
    print("as yes/no or leaving it to confuse a later turn.")
    print("=" * 100)


if __name__ == "__main__":
    main()
