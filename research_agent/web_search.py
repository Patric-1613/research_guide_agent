"""Round-2 enhancement 5: web search via Tavily, producing a genuinely
separate corpus from papers (schema.py's WebArticle — see its docstring for
why it isn't a repurposed Paper).

Uses the official langchain-tavily integration (`langchain_tavily.TavilySearch`,
package `langchain-tavily`) rather than hand-rolling the Tavily HTTP call —
it's the LangChain-maintained package (the older `langchain_community.tools.
tavily_search` is deprecated), so retries, the correct endpoint, and the
response shape are already handled correctly upstream.

Defensive handling matches ingestion.py's philosophy: no API key configured,
zero results, and any API/network error all degrade to an empty list rather
than raising. A missing/invalid TAVILY_API_KEY must make the whole app fall
back to the existing paper-only behavior, not crash the search.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from research_agent.schema import WebArticle

logger = logging.getLogger(__name__)


def _source_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
    except ValueError:
        return ""
    return netloc[4:] if netloc.startswith("www.") else netloc


def search_web(query: str, max_results: int = 4) -> list[WebArticle]:
    """Search the web via Tavily and return normalized WebArticle records.

    Returns [] — never raises — if: no TAVILY_API_KEY is configured, the
    langchain-tavily package isn't installed, the API call fails for any
    reason (auth, rate limit, network), or there are simply zero results.
    Callers should treat an empty list as "no web context available this
    run," the same way ingestion.py's search functions treat a zero-result
    search as a normal outcome, not a failure worth surfacing to the user.
    """
    if not query.strip():
        logger.warning("search_web called with empty query")
        return []

    if not os.getenv("TAVILY_API_KEY"):
        logger.info("TAVILY_API_KEY not set — skipping web search, paper-only results apply")
        return []

    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        logger.warning("langchain-tavily is not installed — skipping web search")
        return []

    try:
        tool = TavilySearch(max_results=max_results, topic="general")
        response = tool.invoke({"query": query})
    except Exception as exc:
        # Tavily/httpx raise a mix of exception types across failure modes
        # (bad key, rate limit, network error) rather than one common base
        # class worth enumerating — this is a best-effort supplementary
        # corpus, so any failure here degrades to "no web results" rather
        # than breaking the whole search.
        logger.warning("Tavily search failed for query %r: %s", query, exc)
        return []

    raw_results = (response or {}).get("results") or []
    articles: list[WebArticle] = []
    for item in raw_results:
        url = item.get("url")
        title = (item.get("title") or "").strip()
        if not url or not title:
            continue
        articles.append(
            WebArticle(
                title=title,
                url=url,
                snippet=(item.get("content") or "").strip(),
                published_date=item.get("published_date"),
                source_domain=_source_domain(url),
            )
        )

    if not articles:
        logger.info("search_web: no usable results for query %r", query)
    return articles
