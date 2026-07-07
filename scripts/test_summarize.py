#!/usr/bin/env python3
"""Phase 5 sanity check: fetch + dedup + rank papers for a topic, then
generate the clustered, grounded literature summary with citations.

Usage:
    python scripts/test_summarize.py "<topic>" [top_k]
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from research_agent.dedup import deduplicate
from research_agent.embeddings import embed_and_index_papers, get_chroma_collection, semantic_search
from research_agent.ingestion import search_arxiv, search_semantic_scholar
from research_agent.summarize import generate_summary

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_TOPIC = "parameter-efficient fine-tuning methods for large language models"


def main() -> None:
    load_dotenv()
    topic = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    client = OpenAI()

    print(f"Fetching papers for: {topic!r}")
    arxiv_papers = search_arxiv(topic, max_results=15)
    s2_papers = search_semantic_scholar(topic, max_results=15, api_key=s2_key)
    pool = deduplicate(arxiv_papers + s2_papers)
    pool = [p for p in pool if p.abstract]
    print(f"Candidate pool: {len(pool)} papers after dedup")

    collection = get_chroma_collection()
    embed_and_index_papers(pool, collection=collection, client=client)
    ids = [p.paper_id for p in pool]
    ranked = semantic_search(topic, collection=collection, client=client, top_k=top_k, where={"paper_id": {"$in": ids}})
    top_papers = [p for p, _ in ranked]
    print(f"Ranked and selected top {len(top_papers)} papers for summarization\n")

    result = generate_summary(topic, top_papers, client=client)

    print(f"{'=' * 80}\nLITERATURE SUMMARY: {topic}\n{'=' * 80}")
    for theme in result["themes"]:
        print(f"\n## {theme['theme_name']}\n")
        for entry in theme["papers"]:
            print(f"- **{entry['paper'].title}**")
            print(f"  {entry['summary']}")
            print(f"  Citation: {entry['apa_citation']}\n")

    print(f"{'=' * 80}\nGaps / Disagreements\n{'=' * 80}")
    print(result["gaps_and_disagreements"])

    if result["skipped_papers"]:
        print(f"\n{'=' * 80}\nRetrieved but not referenced in summary ({len(result['skipped_papers'])})")
        for p in result["skipped_papers"]:
            print(f"- {p.title}")

    print(f"\n{'=' * 80}\nBibTeX export\n{'=' * 80}")
    for theme in result["themes"]:
        for entry in theme["papers"]:
            print(entry["bibtex"])
            print()


if __name__ == "__main__":
    main()
