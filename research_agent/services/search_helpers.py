"""Shared, non-endpoint search orchestration helpers: the server-side
rerank fallback (when the agent's own rerank tool never ran), the
filter-survival count that decides whether that fallback is needed, and
the web-article merge/dedup used by both the agent's own accumulation and
/search's web-search fallback.

Moved out of api.py (Phase 6) so these have a real, independent home —
api.py re-exports all three names so `research_agent.api.<name>` keeps
working unchanged for anything still reaching them that way. Reaches
`api.embed_and_index_papers`/`api.semantic_search` and `api._state` via
`import research_agent.api as api`, accessed only inside function bodies
(never at import time), so `patch.object(api, "<name>", ...)` keeps
intercepting these calls exactly as it did when this logic lived in
api.py itself.
"""

from __future__ import annotations

import research_agent.api as api
from research_agent.agent import _merge_web_articles
from research_agent.enrichment import enrich_missing_abstracts
from research_agent.schema import Paper


def _server_side_rerank(
    session, topic: str, top_k: int, doi_required: bool = False, min_citation_count: int = 0,
):
    collection = api._state["collection"]
    client = api._state["client"]
    # If the agent's own rerank tool never ran (why we're in this fallback
    # at all), session.papers may not have gone through abstract recovery
    # yet either — try it here too. Cached by DOI, so if it already ran
    # this is just a cheap SQLite lookup, not a repeat network round trip.
    enrich_missing_abstracts(session.papers)
    api.embed_and_index_papers(session.papers, collection=collection, client=client)
    ids = [p.paper_id for p in session.papers]
    return api.semantic_search(
        topic, collection=collection, client=client,
        top_k=top_k, where={"paper_id": {"$in": ids}},
        require_doi=doi_required, min_citation_count=min_citation_count or None,
    )


def _filtered_candidate_count(papers: list[Paper], doi_required: bool, min_citation_count: int) -> int:
    """How many of the agent's gathered papers would survive the requested
    filters — used only to decide whether the agent's own ranking already
    honored top_k/filters, or whether a server-side re-rank is needed. Pure
    Python over already-in-memory Paper objects, no extra API/LLM cost."""
    count = 0
    for p in papers:
        if doi_required and not p.doi:
            continue
        if min_citation_count and (p.citation_count or 0) < min_citation_count:
            continue
        count += 1
    return count
