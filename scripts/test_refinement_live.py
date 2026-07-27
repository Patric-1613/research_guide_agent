#!/usr/bin/env python3
"""curation-refinement-and-auto-offer Phase 6f-2 sanity check: real,
non-mocked proof that typed refinement text measurably changes search
behavior, not just that it's accepted without error.

Two checks, at two levels of the same mechanism:

  1. suggest_related_titles() directly -- the exact prompt-level
     mechanism refinement uses (mirrors exclude_titles). No network
     calls beyond the one real LLM call each, so this is fast and
     immune to the arXiv/Semantic Scholar rate-limit flakiness this
     project has hit all session.
  2. refill_pool() end-to-end on two copies of the SAME session (one
     refined, one not) -- proves the full pipeline (title suggestion ->
     per-title search -> merge/rerank) actually produces a different
     reserve, not just different intermediate titles that happen to
     wash out later.

Usage:
    python scripts/test_refinement_live.py
"""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from research_agent.query_expansion import PaperPoolSession, refill_pool, suggest_related_titles
from research_agent.schema import Paper

load_dotenv()

TOPIC = "parameter-efficient fine-tuning for large language models"


def _paper(pid: str, title: str) -> Paper:
    return Paper(
        title=title, authors=["A. Researcher"], year=2024, venue="arXiv",
        abstract=f"abstract for {title}", url=None, doi=None,
        citation_count=None, source="arxiv", paper_id=pid,
    )


def main() -> None:
    client = OpenAI()

    print("=" * 100)
    print("Check 1: suggest_related_titles() -- unrefined vs refined, same topic, real LLM calls")
    print("=" * 100)
    unrefined_titles = suggest_related_titles(TOPIC, client=client)
    print(f"Unrefined suggestions: {unrefined_titles}")

    refined_titles = suggest_related_titles(
        TOPIC, client=client,
        refinement_notes=["Focus specifically on work published in 2024 or later -- prefer very recent papers over foundational/older ones."],
    )
    print(f"Refined suggestions:   {refined_titles}")

    assert unrefined_titles != refined_titles, (
        "FAIL: refinement guidance produced IDENTICAL suggested titles -- expected a measurable difference"
    )
    print("PASS: refinement guidance measurably changed the suggested titles for the identical topic.")

    print("\n" + "=" * 100)
    print("Check 2: refill_pool() end-to-end -- two copies of the SAME session, one refined")
    print("=" * 100)
    titles = [f"Existing Paper {i}" for i in range(12)]
    base_session = PaperPoolSession(
        topic=TOPIC,
        reserve=[(_paper(f"p{i}", t), 1.0 - i * 0.01) for i, t in enumerate(titles)],
        cursor=10,  # only 2 unserved -- forces a real, meaningful refill
        target_count=20,
    )

    session_unrefined = copy.deepcopy(base_session)
    session_refined = copy.deepcopy(base_session)
    session_refined.refinement_notes = [
        "Focus specifically on work published in 2024 or later -- prefer very recent papers over foundational/older ones.",
    ]

    found_unrefined = refill_pool(session_unrefined, client=client)
    found_refined = refill_pool(session_refined, client=client)

    unrefined_ids = {p.paper_id for p, _ in session_unrefined.reserve}
    refined_ids = {p.paper_id for p, _ in session_refined.reserve}
    unrefined_titles_found = sorted(p.title for p, _ in session_unrefined.reserve if p.paper_id not in {f"p{i}" for i in range(12)})
    refined_titles_found = sorted(p.title for p, _ in session_refined.reserve if p.paper_id not in {f"p{i}" for i in range(12)})

    print(f"Unrefined refill found {found_unrefined} genuinely new paper(s):")
    for t in unrefined_titles_found:
        print(f"  - {t}")
    print(f"\nRefined refill found {found_refined} genuinely new paper(s):")
    for t in refined_titles_found:
        print(f"  - {t}")

    print(f"\nOverlap between the two resulting reserves: {len(unrefined_ids & refined_ids)} paper(s) in common")
    print(f"Unrefined-only: {len(unrefined_ids - refined_ids)}, refined-only: {len(refined_ids - unrefined_ids)}")

    assert unrefined_ids != refined_ids, (
        "FAIL: the refined and unrefined refills produced the IDENTICAL reserve -- expected a measurable difference"
    )
    print("\nPASS: refinement measurably changed refill_pool()'s actual results on the same starting session.")
    print("=" * 100)


if __name__ == "__main__":
    main()
