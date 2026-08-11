from __future__ import annotations

import sqlite3

from research_agent.api_app.schemas import SummarizeResponse, WebSummaryOut
from research_agent.api_app.serializers import _render_markdown
from research_agent.citations import CitationStyle
from research_agent.services.summary_cache import _get_or_create_summary, _get_or_create_web_summary
from research_agent.storage import SavedSearch, get_search
from research_agent.usage_guard import guard_paid_action


def _needs_generation(saved: SavedSearch) -> bool:
    return saved.summary is None or bool(saved.web_articles) and saved.web_summary is None


def _generate_or_get_summaries(db: sqlite3.Connection, search_id: int, saved: SavedSearch, style: CitationStyle) -> tuple[dict, dict | None]:
    """Usage Protection M2.2B: the ONE guarded generation boundary both
    /summarize and /export delegate to, so neither can open its own
    independent top-level "summarize" action for what is conceptually a
    single cached artifact -- matches M1's own original design, which
    already wrapped both _get_or_create_* calls in a single outer
    paid_action, not two.

    A pure cache hit (both pieces already generated) never opens the
    guard at all -- no admission check, no lease. Only entered when at
    least one piece needs generation; `saved.summary is None`/
    `saved.web_summary is None` are the exact same conditions
    _get_or_create_summary/_get_or_create_web_summary already check
    internally (see summary_cache.py), read here off the same `saved`
    row already loaded by the caller -- not a duplicated routing
    decision, a direct field check.

    Re-checks the cache (a fresh get_search()) AFTER acquiring the
    lease: a concurrent request may have already generated and saved
    while this one was waiting/racing for the lease.
    _get_or_create_summary/_get_or_create_web_summary run against that
    fresh copy and skip regeneration on their own if it's now cached --
    no changes needed to summary_cache.py itself for that.
    """
    if not _needs_generation(saved):
        summary_json = _get_or_create_summary(db, search_id, saved, style=style)
        web_summary_json = _get_or_create_web_summary(db, search_id, saved)
        return summary_json, web_summary_json

    with guard_paid_action("summarize", subject=("search", str(search_id)), use_lease=True):
        saved = get_search(db, search_id) or saved
        summary_json = _get_or_create_summary(db, search_id, saved, style=style)
        web_summary_json = _get_or_create_web_summary(db, search_id, saved)
    return summary_json, web_summary_json


def summarize_search(db: sqlite3.Connection, search_id: int, style: CitationStyle) -> SummarizeResponse | None:
    saved = get_search(db, search_id)
    if saved is None:
        return None

    summary_json, web_summary_json = _generate_or_get_summaries(db, search_id, saved, style)
    web_summary_out = WebSummaryOut(**web_summary_json) if web_summary_json is not None else None
    return SummarizeResponse(
        search_id=search_id, topic=saved.topic, style=style, web_summary=web_summary_out, **summary_json,
    )


def export_search_markdown(db: sqlite3.Connection, search_id: int, style: CitationStyle) -> str | None:
    saved = get_search(db, search_id)
    if saved is None:
        return None

    # /export/{search_id} shares the exact same generate-or-cache summary
    # mechanism /summarize uses (same underlying operation, different final
    # rendering) -- instrumented as the same "summarize" action type rather
    # than inventing a separate one for what is, underneath, the identical
    # billable work.
    summary_json, web_summary_json = _generate_or_get_summaries(db, search_id, saved, style)
    return _render_markdown(saved.topic, summary_json, style=style, web_summary_json=web_summary_json)
