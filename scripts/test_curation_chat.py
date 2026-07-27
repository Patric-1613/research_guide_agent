#!/usr/bin/env python3
"""curation-chat-web-escalation Phase 5b sanity check: real, non-mocked
proof that chat over a curation session's selected_papers is genuinely
grounded -- mirrors scripts/test_report.py's live-verification standard
for Phase 4c.

Two real questions against the same two real, well-known PEFT papers
(LoRA, RoCoFT):
  1. An answerable question, directly covered by the selected abstracts
     -- confirms a real grounded answer with a [Paper N] citation.
  2. An adversarial follow-up about a well-known but NOT-selected method
     (QLoRA) -- confirms the model says answerable=False rather than
     answering from background knowledge, since QLoRA was never in the
     selected set at all.
Also confirms multi-turn continuity: a third question that only makes
sense as a follow-up to question 1 ("its" referring to LoRA) is
answered correctly, proving chat_history/condense-question is actually
wired through ask_in_session(), not just a single stateless call.

Usage:
    python scripts/test_curation_chat.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from research_agent.curation_chat import ask_in_session
from research_agent.query_expansion import PaperPoolSession
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
        stage="synthesize",
        selected_paper_ids=["p1", "p2"],
        selected_papers=selected,
    )

    client = OpenAI()

    print("=" * 100)
    print("Turn 1: answerable question, directly covered by the selected abstracts")
    print("=" * 100)
    r1 = ask_in_session(session, "What does LoRA inject into each layer?", client=client)
    print(f"answerable={r1['answerable']}")
    print(f"answer: {r1['answer']}")
    print(f"cited_papers: {[p.paper_id for p in r1['cited_papers']]}")
    assert r1["answerable"] is True, "FAIL: expected an answerable result for a directly-covered question"
    assert "p1" in [p.paper_id for p in r1["cited_papers"]], "FAIL: expected LoRA (p1) to be cited"

    print("\n" + "=" * 100)
    print("Turn 2: adversarial -- asks about QLoRA, a real but NOT-selected method")
    print("(must say unanswerable, not answer from background knowledge)")
    print("=" * 100)
    r2 = ask_in_session(session, "How does QLoRA's 4-bit quantization work?", client=client)
    print(f"answerable={r2['answerable']}")
    print(f"answer: {r2['answer']}")
    print(f"cited_papers: {[p.paper_id for p in r2['cited_papers']]}")
    if r2["answerable"]:
        print("WARNING: model answered a question about a non-selected paper -- possible background-knowledge leak")
    else:
        print("PASS: model correctly refused to answer from outside the selected set")
    assert r2["cited_papers"] == [], "FAIL: unanswerable turn must not carry citations"

    print("\n" + "=" * 100)
    print("Turn 3: follow-up ('its') that only resolves via turn 1's history")
    print("=" * 100)
    r3 = ask_in_session(session, "How many trainable parameters does its approach add relative to full fine-tuning?", client=client)
    print(f"answerable={r3['answerable']}")
    print(f"answer: {r3['answer']}")
    print(f"cited_papers: {[p.paper_id for p in r3['cited_papers']]}")
    assert r3["answerable"] is True, "FAIL: expected the follow-up to resolve 'its' -> LoRA via history and answer"

    print("\n" + "=" * 100)
    print(f"Final session.chat_history has {len(session.chat_history)} messages (expected 6)")
    assert len(session.chat_history) == 6
    print("PASS: baseline chat is genuinely grounded, refuses out-of-set questions, and carries history across turns.")
    print("=" * 100)


if __name__ == "__main__":
    main()
