"""Generate-or-get caching for /summarize, /export, and /library's paper
and web-article summaries.

Moved out of api.py (Phase 6) so this has a real, independent home —
api.py re-exports these three names so `research_agent.api.<name>` keeps
working unchanged for anything still reaching them that way. Reaches
`api.get_papers_by_ids`/`api.generate_summary`/`api.generate_web_summary`
and `api._state` via `import research_agent.api as api`, accessed only
inside function bodies (never at import time), so
`patch.object(api, "<name>", ...)` keeps intercepting these calls exactly
as it did when this logic lived in api.py itself.
"""

from __future__ import annotations

import sqlite3

import research_agent.api as api
from research_agent.api_app.serializers import _summary_to_json, _web_articles_from_saved, _web_summary_to_json
from research_agent.citations import CitationStyle, select_citation
from research_agent.storage import update_summary, update_web_summary


def _reselect_style(summary_json: dict, style: CitationStyle) -> dict:
    """Re-picks the `citation` field for a cached summary against a
    possibly different style than the one it was first generated with.
    Citation formatting is pure/cheap string logic, not an LLM call — a
    cache hit still needs to honor whatever style THIS request asked for,
    and doing that costs nothing beyond a dict lookup (see
    citations.select_citation)."""
    return {
        "themes": [
            {
                "theme_name": theme["theme_name"],
                "papers": [
                    {
                        **p,
                        "citation": select_citation(
                            p.get("apa_citation", ""),
                            p.get("harvard_citation") or p.get("apa_citation", ""),
                            p.get("bibtex", ""),
                            style,
                        ),
                    }
                    for p in theme["papers"]
                ],
            }
            for theme in summary_json["themes"]
        ],
        "gaps_and_disagreements": summary_json["gaps_and_disagreements"],
        "skipped_paper_ids": summary_json["skipped_paper_ids"],
    }


def _get_or_create_summary(db: sqlite3.Connection, search_id: int, saved, style: CitationStyle = "apa") -> dict:
    """Reuse a previously-generated summary if one exists for this
    search_id, rather than re-billing the LLM every time /summarize or
    /export is called for the same search — mirrors the embedding cache's
    cost-consciousness from phase 3. A different `style` than the one the
    summary was originally generated with is still honored on a cache hit
    (via _reselect_style) since picking a citation format costs nothing."""
    if saved.summary is not None:
        return _reselect_style(saved.summary, style)
    papers = api.get_papers_by_ids(saved.paper_ids, collection=api._state["collection"])
    result = api.generate_summary(saved.topic, papers, client=api._state["client"], style=style)
    summary_json = _summary_to_json(result, style=style)
    update_summary(db, search_id, summary_json)
    return summary_json


def _get_or_create_web_summary(db: sqlite3.Connection, search_id: int, saved) -> dict | None:
    """Mirrors _get_or_create_summary's cost-consciousness for the separate
    web-article corpus — its own cache column, never merged into the paper
    summary's cache. Returns None if this search found no web articles at
    all, so callers render the paper summary alone rather than an empty
    web-context block."""
    if not saved.web_articles:
        return None
    if saved.web_summary is not None:
        return saved.web_summary
    articles = _web_articles_from_saved(saved)
    result = api.generate_web_summary(saved.topic, articles, client=api._state["client"])
    web_summary_json = _web_summary_to_json(result)
    update_web_summary(db, search_id, web_summary_json)
    return web_summary_json
