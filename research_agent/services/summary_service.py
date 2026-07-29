from __future__ import annotations

import sqlite3

import research_agent.api as api
from research_agent.citations import CitationStyle
from research_agent.storage import get_search


def summarize_search(db: sqlite3.Connection, search_id: int, style: CitationStyle) -> api.SummarizeResponse | None:
    saved = get_search(db, search_id)
    if saved is None:
        return None

    summary_json = api._get_or_create_summary(db, search_id, saved, style=style)
    web_summary_json = api._get_or_create_web_summary(db, search_id, saved)
    web_summary_out = api.WebSummaryOut(**web_summary_json) if web_summary_json is not None else None
    return api.SummarizeResponse(
        search_id=search_id, topic=saved.topic, style=style, web_summary=web_summary_out, **summary_json,
    )


def export_search_markdown(db: sqlite3.Connection, search_id: int, style: CitationStyle) -> str | None:
    saved = get_search(db, search_id)
    if saved is None:
        return None

    summary_json = api._get_or_create_summary(db, search_id, saved, style=style)
    web_summary_json = api._get_or_create_web_summary(db, search_id, saved)
    return api._render_markdown(saved.topic, summary_json, style=style, web_summary_json=web_summary_json)
