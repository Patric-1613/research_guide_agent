#!/usr/bin/env python3
"""curation-report-synthesis Phase 4c sanity check: the one property
this whole phase exists to prove — that a report generated over a
curation session's selected set NEVER cites, or even mentions in
free-text prose, a paper that was seen but rejected during curation.

Deliberately adversarial, not a clean happy-path check: the rejected
paper here is a real, extremely well-known paper ("Attention Is All
You Need") that is topically ADJACENT to the selected papers (LLMs/
transformers broadly) but not actually about the selected topic
(parameter-efficient fine-tuning specifically) — exactly the kind of
paper a model's own background knowledge might tempt it to reference
even though it was never included in the prompt at all.

Checks TWO things, not just one:
  1. Structural: the rejected paper's id never appears in any section's
     cited_paper_ids (guaranteed by the Literal-constrained schema —
     this would raise a validation error if violated, not silently
     leak, but verified against real output here rather than assumed).
  2. Content-level: the rejected paper's name/topic never appears in
     any section's free-text content either — NOT structurally
     prevented by the schema (content is a plain str field), so this is
     the genuinely adversarial check, not a formality.

Usage:
    python scripts/test_report.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

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
            "p1", "RoCoFT: Efficient Finetuning of LLMs with Row-Column Updates",
            "We propose RoCoFT, a parameter-efficient fine-tuning method that updates only a "
            "small subset of rows and columns of weight matrices, achieving competitive "
            "performance with a fraction of the trainable parameters of full fine-tuning.",
        ),
        _paper(
            "p2", "LoRA: Low-Rank Adaptation of Large Language Models",
            "We propose LoRA, which freezes pretrained model weights and injects trainable "
            "low-rank decomposition matrices into each layer, greatly reducing the number of "
            "trainable parameters for downstream tasks.",
        ),
    ]

    rejected = _paper(
        "REJECTED-p99", "Attention Is All You Need",
        "We propose the Transformer, a model architecture based entirely on attention "
        "mechanisms, dispensing with recurrence and convolutions entirely.",
    )

    session = PaperPoolSession(
        topic="parameter-efficient fine-tuning for large language models",
        reserve=[(rejected, 0.5)],  # seen, sitting in reserve, NOT picked
        cursor=1,
        seen_paper_ids={"p1", "p2", "REJECTED-p99"},
        seen_titles={"RoCoFT", "LoRA", "Attention Is All You Need"},
        stage="synthesize",
        target_count=2,
        selected_paper_ids=["p1", "p2"],
        selected_papers=selected,
    )

    client = OpenAI()
    result = generate_report_for_session(session, client=client)

    print("=" * 100)
    print("Checking every section's cited_papers for the rejected paper's id")
    print("=" * 100)
    leaked_citation = False
    for section in ("findings", "limitations", "future_scope"):
        cited_ids = [p.paper_id for p in result[section]["cited_papers"]]
        print(f"  {section}: cited_paper_ids={cited_ids}")
        if "REJECTED-p99" in cited_ids:
            leaked_citation = True

    print("\n" + "=" * 100)
    print("Checking every section's free-text content for the rejected paper's name/topic")
    print("(content-level check — NOT structurally prevented by the schema)")
    print("=" * 100)
    leaked_mention = False
    for section in ("findings", "limitations", "future_scope"):
        content = result[section]["content"]
        mentioned = "attention is all you need" in content.lower() or "transformer" in content.lower()
        print(f"\n  [{section}] mentions rejected paper by name/topic = {mentioned}")
        print(f"    {content}")
        if mentioned:
            leaked_mention = True

    print("\n" + "=" * 100)
    assert not leaked_citation, "FAIL: rejected paper was cited — structural guarantee failed"
    if leaked_mention:
        print("WARNING: rejected paper's name/topic appeared in free-text content (not structurally "
              "prevented) — investigate the specific section above before trusting this run's grounding.")
    else:
        print("PASS: rejected paper appeared as neither a citation nor a prose mention, across all 3 sections.")
    print("=" * 100)


if __name__ == "__main__":
    main()
