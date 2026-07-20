"""Shared Langfuse tracing helpers.

Loading .env here (idempotent) guarantees Langfuse's env vars are present
before any @observe-decorated function runs, regardless of which entry
point imports it first.
"""

from __future__ import annotations

from dotenv import load_dotenv

from research_agent.schema import Paper

load_dotenv()


def paper_metadata(papers: list[Paper]) -> list[dict]:
    """Redacted, Langfuse-safe view of search results: everything except
    abstract text. Abstracts are public but withheld from the third-party
    trace payload by explicit confirmation, not by default."""
    return [
        {
            "paper_id": p.paper_id,
            "title": p.title,
            "year": p.year,
            "venue": p.venue,
            "source": p.source,
            "url": p.url,
            "doi": p.doi,
            "citation_count": p.citation_count,
        }
        for p in papers
    ]
