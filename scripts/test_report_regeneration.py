#!/usr/bin/env python3
"""curation-chat-web-escalation Phase 5d sanity check: real, non-mocked
proof that report regeneration preserves prior citations, matching
scripts/test_report.py's (Phase 4c) live-verification standard.

Unlike the mocked isolation test in tests/test_report.py (which
deliberately defeats the prompt layer to prove the defensive re-append
works alone), this script exercises BOTH layers together against a real
model: generates a real initial report over two real papers, adds one
real web source (from a real Tavily search), regenerates for real, and
confirms every paper cited in the original report is still cited in the
same section of the regenerated one -- the property a real user actually
depends on, end to end.

Usage:
    python scripts/test_report_regeneration.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from research_agent.query_expansion import PaperPoolSession
from research_agent.report import SECTION_NAMES, generate_report_for_session, regenerate_report_with_new_sources
from research_agent.schema import Paper
from research_agent.web_search import search_web

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
    print("Step 1: real initial report generation (Phase 4 path, unmodified)")
    print("=" * 100)
    session.report = generate_report_for_session(session, client=client)
    original_citations = {
        name: sorted(p.paper_id for p in session.report[name]["cited_papers"])
        for name in SECTION_NAMES
    }
    for name, ids in original_citations.items():
        print(f"  {name}: cited_paper_ids={ids}")

    print("\n" + "=" * 100)
    print("Step 2: real web search, approved into the session (mirrors Phase 5c's accept path)")
    print("=" * 100)
    found = search_web("LoRA parameter-efficient fine-tuning recent advances 2026")
    session.web_articles_added.extend(found)
    print(f"  web_articles_added: {len(session.web_articles_added)} article(s)")
    for a in session.web_articles_added:
        print(f"    - {a.title} ({a.url})")

    print("\n" + "=" * 100)
    print("Step 3: real regeneration incorporating the new web source(s)")
    print("=" * 100)
    regenerated = regenerate_report_with_new_sources(session, client=client)
    new_citations = {
        name: sorted(p.paper_id for p in regenerated[name]["cited_papers"])
        for name in SECTION_NAMES
    }
    for name, ids in new_citations.items():
        web_ids = [a.url for a in regenerated[name].get("cited_web_articles", [])]
        print(f"  {name}: cited_paper_ids={ids}, cited_web_urls={web_ids}")
        print(f"    content: {regenerated[name]['content']}")

    print("\n" + "=" * 100)
    print("Checking every ORIGINAL citation survived regeneration, section by section")
    print("=" * 100)
    all_preserved = True
    for name in SECTION_NAMES:
        missing = set(original_citations[name]) - set(new_citations[name])
        status = "OK" if not missing else f"MISSING: {missing}"
        print(f"  {name}: original={original_citations[name]} -> regenerated={new_citations[name]}  [{status}]")
        if missing:
            all_preserved = False

    print("\n" + "=" * 100)
    assert all_preserved, "FAIL: at least one original citation did not survive regeneration"
    print("PASS: every citation present in the original report is still present after a real,")
    print("live regeneration with a new web source added.")
    print("=" * 100)


if __name__ == "__main__":
    main()
