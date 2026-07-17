"""Phase 1: search arXiv and Semantic Scholar, normalize results to Paper records.

Both functions are defensive about the three edge cases called out in the
brief: zero results, rate limiting, and malformed/missing abstracts. Neither
raises on those cases — they log a warning and return the best list they can
(possibly empty), so a caller never has to wrap them in try/except just to
run a search.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import arxiv
import requests

from research_agent.schema import Paper

logger = logging.getLogger(__name__)

_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = "title,authors,year,venue,abstract,externalIds,citationCount,url"
_S2_MAX_LIMIT = 100  # hard cap enforced by the Semantic Scholar API per request

# Some arXiv authors self-populate journal_ref with a full citation dump
# (authors + year + title again + venue) instead of a plain venue name —
# harmless in isolation, but it produces garbled, duplicated-looking APA
# citations downstream (phase 5). Genuine venue names, even long conference
# titles, are reliably well under this length; observed citation-dumps run
# 200+ characters.
_MAX_PLAUSIBLE_VENUE_LEN = 150


def _clean_abstract(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = " ".join(text.split())
    return cleaned or None


def _clean_venue(journal_ref: str | None) -> str | None:
    if journal_ref and len(journal_ref) <= _MAX_PLAUSIBLE_VENUE_LEN:
        return journal_ref
    return None


def _parse_retry_after(value: str | None, default: float) -> float:
    """Parse a `Retry-After` header value, which per RFC 9110 is either a
    plain number of delay-seconds or an HTTP-date. Falls back to `default`
    (the existing exponential backoff value) if it's neither, rather than
    letting `float()` raise on a date string and crash the retry loop.
    """
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(value)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        delta = (target - datetime.now(timezone.utc)).total_seconds()
        return max(delta, 0.0)
    except (TypeError, ValueError):
        logger.warning("Unparseable Retry-After header %r, using default backoff", value)
        return default


def search_arxiv(query: str, max_results: int = 20) -> list[Paper]:
    """Search arXiv and return normalized Paper records.

    Uses the `arxiv` package's Client, which handles arXiv's own rate
    limiting (3s between requests) internally. On any unexpected failure
    (network error, malformed feed page) we log and return whatever we
    already collected rather than raising.
    """
    if not query.strip():
        logger.warning("search_arxiv called with empty query")
        return []

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[Paper] = []
    try:
        for result in client.results(search):
            papers.append(
                Paper(
                    title=result.title.strip(),
                    authors=[a.name for a in result.authors],
                    year=result.published.year if result.published else None,
                    venue=_clean_venue(result.journal_ref) or "arXiv preprint",
                    abstract=_clean_abstract(result.summary),
                    url=result.entry_id,
                    doi=result.doi,
                    citation_count=None,  # arXiv doesn't track citations
                    source="arxiv",
                    paper_id=result.get_short_id(),
                )
            )
    except arxiv.ArxivError as exc:
        logger.warning("arXiv search failed for query %r: %s", query, exc)
    except requests.RequestException as exc:
        logger.warning("Network error during arXiv search for query %r: %s", query, exc)

    if not papers:
        logger.info("search_arxiv: no results for query %r", query)
    return papers


def search_semantic_scholar(
    query: str,
    max_results: int = 20,
    api_key: str | None = None,
    max_retries: int = 3,
) -> list[Paper]:
    """Search Semantic Scholar's /graph/v1/paper/search endpoint.

    Retries on 429 with exponential backoff (honoring Retry-After if the API
    sends one). Unauthenticated calls share a low, strict rate limit, so
    backoff matters more here than for arXiv.
    """
    if not query.strip():
        logger.warning("search_semantic_scholar called with empty query")
        return []

    limit = min(max_results, _S2_MAX_LIMIT)
    headers = {"x-api-key": api_key} if api_key else {}
    params = {"query": query, "limit": limit, "fields": _S2_FIELDS}

    backoff = 1.0
    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                _S2_SEARCH_URL, params=params, headers=headers, timeout=15
            )
        except requests.RequestException as exc:
            logger.warning(
                "Network error during Semantic Scholar search for query %r (attempt %d/%d): %s",
                query, attempt, max_retries, exc,
            )
            if attempt == max_retries:
                return []
            time.sleep(backoff)
            backoff *= 2
            continue

        if response.status_code == 429:
            wait = _parse_retry_after(response.headers.get("Retry-After"), backoff)
            logger.warning(
                "Semantic Scholar rate limited us (attempt %d/%d), waiting %.1fs",
                attempt, max_retries, wait,
            )
            if attempt == max_retries:
                logger.warning("Giving up on Semantic Scholar search for query %r", query)
                return []
            time.sleep(wait)
            backoff *= 2
            continue

        break

    if response is None or response.status_code != 200:
        status = response.status_code if response is not None else "no response"
        logger.warning(
            "Semantic Scholar search failed for query %r: status=%s", query, status
        )
        return []

    try:
        payload = response.json()
    except ValueError as exc:
        logger.warning(
            "Semantic Scholar returned a malformed/empty response body for query %r: %s",
            query, exc,
        )
        return []
    raw_results = payload.get("data", [])

    papers: list[Paper] = []
    for item in raw_results:
        authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
        external_ids = item.get("externalIds") or {}
        paper_id = item.get("paperId") or ""
        papers.append(
            Paper(
                title=(item.get("title") or "").strip(),
                authors=authors,
                year=item.get("year"),
                venue=item.get("venue") or None,
                abstract=_clean_abstract(item.get("abstract")),
                url=item.get("url") or (f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None),
                doi=external_ids.get("DOI"),
                citation_count=item.get("citationCount"),
                source="semantic_scholar",
                paper_id=paper_id or "",
            )
        )

    if not papers:
        logger.info("search_semantic_scholar: no results for query %r", query)
    return papers
