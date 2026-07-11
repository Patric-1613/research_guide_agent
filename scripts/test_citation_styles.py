#!/usr/bin/env python3
"""Round-2 enhancement 3 sanity check: generate a summary once, then confirm
that switching the citation style actually changes the emitted citation
strings — and that the default (apa) behavior is unchanged.

Usage:
    python scripts/test_citation_styles.py ["<topic>"]
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

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_TOPIC = "parameter-efficient fine-tuning methods for large language models"


def main() -> None:
    load_dotenv()
    topic = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    client = OpenAI()

    print(f"Fetching a small pool for: {topic!r}")
    arxiv_papers = search_arxiv(topic, max_results=8)
    s2_papers = search_semantic_scholar(topic, max_results=8, api_key=s2_key)
    pool = deduplicate(arxiv_papers + s2_papers)
    pool = [p for p in pool if p.abstract][:3]  # keep this cheap — one LLM call, few papers
    print(f"Using {len(pool)} papers for the summary\n")

    collection = get_chroma_collection()
    embed_and_index_papers(pool, collection=collection, client=client)
    ids = [p.paper_id for p in pool]
    ranked = semantic_search(topic, collection=collection, client=client, top_k=len(pool), where={"paper_id": {"$in": ids}})
    top_papers = [p for p, _ in ranked]

    result = generate_summary(topic, top_papers, client=client, style="harvard")

    print(f"{'=' * 80}\nCitations by style, per paper\n{'=' * 80}")
    all_ok = True
    for theme in result["themes"]:
        for entry in theme["papers"]:
            print(f"\n{entry['paper'].title}")
            print(f"  APA:      {entry['apa_citation']}")
            print(f"  Harvard:  {entry['harvard_citation']}")
            print(f"  BibTeX:   {entry['bibtex'].splitlines()[0]} ...")
            print(f"  citation field (style='harvard' requested): {entry['citation']}")
            if entry["citation"] != entry["harvard_citation"]:
                all_ok = False
            if entry["apa_citation"] == entry["harvard_citation"]:
                all_ok = False

    print(f"\n{'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
