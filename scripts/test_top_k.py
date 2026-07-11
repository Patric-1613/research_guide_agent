#!/usr/bin/env python3
"""Round-2 enhancement 1 sanity check: run the agent on the same topic with
two different top_k values and confirm the returned ranking count actually
tracks what was requested, rather than silently defaulting to 5 regardless
of input (the bug this enhancement fixes).

Usage:
    python scripts/test_top_k.py ["<topic>"]
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from research_agent.agent import run_research_agent

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_TOPIC = "retrieval augmented generation for large language models"


def main() -> None:
    load_dotenv()
    topic = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None

    results = {}
    for top_k in (5, 15):
        print(f"\n{'=' * 80}\nRunning topic {topic!r} with top_k={top_k}\n{'=' * 80}")
        session = run_research_agent(topic, s2_api_key=s2_key, top_k=top_k)
        results[top_k] = len(session.ranked)
        print(f"session.papers (working pool): {len(session.papers)}")
        print(f"session.ranked (final count):  {len(session.ranked)}")
        for i, (p, score) in enumerate(session.ranked, 1):
            print(f"  [{i}] ({score:.3f}) {p.title}")

    print(f"\n{'=' * 80}\nSummary: top_k=5 -> {results[5]} result(s), top_k=15 -> {results[15]} result(s)")
    if results[5] == results[15]:
        print("FAIL: counts are identical — top_k is not affecting the result count.")
        sys.exit(1)
    else:
        print("PASS: result counts differ as requested.")


if __name__ == "__main__":
    main()
