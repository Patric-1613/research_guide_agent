#!/usr/bin/env python3
"""Round-2 enhancement 5 sanity check: run the full agent on a topic where
web context should matter (current tools/practical state) and confirm it
decides to call search_web_tool, gathers a separate web-article pool
(never merged into the paper pool/count), and that generate_web_summary
produces a grounded synthesis citing only retrieved URLs.

Usage:
    python scripts/test_web_search.py ["<topic>"]
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from research_agent.agent import run_research_agent
from research_agent.summarize import generate_web_summary

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_TOPIC = "current best open-source tools and frameworks for building RAG pipelines in 2026"


def main() -> None:
    load_dotenv()
    topic = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None

    print(f"{'=' * 80}\nTopic: {topic!r}\n{'=' * 80}")
    session = run_research_agent(topic, s2_api_key=s2_key, top_k=5, web_max_results=4)

    print(f"\nPapers: {len(session.ranked)} ranked (out of {len(session.papers)} gathered)")
    print(f"Web articles: {len(session.web_articles)} gathered (independent pool, not counted toward papers)")

    if not session.web_articles:
        print("\nFAIL: agent did not call search_web_tool for a topic that clearly calls for current context.")
        sys.exit(1)

    for i, a in enumerate(session.web_articles, 1):
        print(f"  [{i}] {a.title}  ({a.source_domain})")
        print(f"      {a.snippet[:120]}...")

    print(f"\n{'=' * 80}\ngenerate_web_summary()\n{'=' * 80}")
    client = OpenAI()
    result = generate_web_summary(topic, session.web_articles, client=client)
    print(f"Synthesis: {result['synthesis']}")
    print(f"\nCited {len(result['cited_articles'])} of {len(session.web_articles)} article(s):")
    cited_urls = {a.url for a in result["cited_articles"]}
    retrieved_urls = {a.url for a in session.web_articles}
    for a in result["cited_articles"]:
        print(f"  - {a.title} ({a.url})")

    ok = cited_urls.issubset(retrieved_urls)
    print(f"\n{'PASS' if ok else 'FAIL'}: every cited URL was actually retrieved: {ok}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
