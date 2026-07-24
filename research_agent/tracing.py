"""Shared Langfuse tracing helpers.

Loading .env here (idempotent) guarantees Langfuse's env vars are present
before any @observe-decorated function runs, regardless of which entry
point imports it first.
"""

from __future__ import annotations

from dotenv import load_dotenv
from langfuse import get_client

from research_agent.schema import Paper

load_dotenv()


def tag_current_trace(tags: list[str]) -> None:
    """Tag the currently-active trace (root or nested) so the two retrieval
    paths (agent vs. expanded_search) can be filtered by tag in the
    Langfuse dashboard instead of only by trace name or a nested Input
    field. Uses the SDK's internal ingestion helper because langfuse-python
    4.14.1 has no public method to set trace tags outside the (unrelated)
    legacy `langfuse.trace()` API — confirmed by inspecting the client
    (`update_current_span`/`start_as_current_observation` take metadata but
    not tags; `@observe` takes no tags kwarg either). Same category of gap
    as the rate-limit-metadata propagation issue elsewhere in this project:
    worked around deliberately, not overlooked."""
    client = get_client()
    trace_id = client.get_current_trace_id()
    if trace_id:
        client._create_trace_tags_via_ingestion(trace_id=trace_id, tags=tags)


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


def ranked_paper_metadata(ranked: list[tuple[Paper, float]]) -> list[dict]:
    """Same redaction as paper_metadata(), for (Paper, similarity-score)
    pairs as returned by semantic_search()/expanded_search()."""
    return [{**meta, "score": score} for meta, (_, score) in zip(paper_metadata([p for p, _ in ranked]), ranked)]
