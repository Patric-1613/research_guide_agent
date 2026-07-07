#!/usr/bin/env python3
"""Phase 6 sanity check: fetch + dedup + rank a paper pool for a topic, then
run a multi-turn conversation against it, including:
  - a follow-up question with a pronoun ("its limitations?") to check
    question condensing resolves it before retrieval
  - an out-of-scope question the retrieved abstracts can't answer, to check
    the agent says so explicitly instead of guessing

Usage:
    python scripts/test_qa.py
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
from research_agent.qa import ChatSession, ask

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

TOPIC = "parameter-efficient fine-tuning methods for large language models"
QUESTIONS = [
    "What is RoCoFT and how does it work?",
    "What are its limitations?",  # follow-up: "its" should resolve to RoCoFT
    "What did these papers report about the stock market?",  # out-of-scope, should be refused
]


def main() -> None:
    load_dotenv()
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    client = OpenAI()

    print(f"Fetching papers for: {TOPIC!r}")
    arxiv_papers = search_arxiv(TOPIC, max_results=15)
    s2_papers = search_semantic_scholar(TOPIC, max_results=15, api_key=s2_key)
    pool = deduplicate(arxiv_papers + s2_papers)
    pool = [p for p in pool if p.abstract]

    collection = get_chroma_collection()
    embed_and_index_papers(pool, collection=collection, client=client)
    ids = [p.paper_id for p in pool]
    ranked = semantic_search(TOPIC, collection=collection, client=client, top_k=8, where={"paper_id": {"$in": ids}})
    top_papers = [p for p, _ in ranked]
    print(f"Grounding set: {len(top_papers)} papers\n")

    session = ChatSession(papers=top_papers)

    for question in QUESTIONS:
        print(f"{'=' * 80}\nQ: {question}\n{'=' * 80}")
        result = ask(session, question, client=client)
        print(f"answerable: {result['answerable']}")
        print(f"\nA: {result['answer']}\n")
        if result["cited_papers"]:
            print("Cited papers:")
            for i, p in enumerate(result["cited_papers"], 1):
                print(f"  [{i}] {p.title}")
        print()


if __name__ == "__main__":
    main()
