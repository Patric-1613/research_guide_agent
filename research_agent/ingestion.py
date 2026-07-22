"""Phase 1: search arXiv and Semantic Scholar, normalize results to Paper records.

Both functions are defensive about the three edge cases called out in the
brief: zero results, rate limiting, and malformed/missing abstracts. Neither
raises on those cases — they log a warning and return the best list they can
(possibly empty), so a caller never has to wrap them in try/except just to
run a search.
"""

from __future__ import annotations

import contextvars
import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import arxiv
import requests
from langfuse import get_client, observe

from research_agent.schema import Paper
from research_agent.tracing import paper_metadata

logger = logging.getLogger(__name__)

_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = "title,authors,year,venue,abstract,externalIds,citationCount,url"
_S2_MAX_LIMIT = 100  # hard cap enforced by the Semantic Scholar API per request

# Lets a caller that makes several search_semantic_scholar calls for one
# logical operation (query_expansion.py's build_candidate_pool(), which
# calls it once for the original query plus once per suggested title) roll
# up "how many of those calls needed a retry" into its OWN span's metadata
# afterward — a plain Python contextvars.ContextVar, not a Langfuse
# mechanism, because none exists for this: verified against the installed
# langfuse==4.14.1 SDK that there is no current, public method for updating
# an ANCESTOR span's metadata from a descendant after the ancestor's span
# has already started. update_current_span()/update_current_generation()
# only ever target whichever span is CURRENTLY active (i.e. the innermost
# one — search_semantic_scholar's own, not build_candidate_pool's or
# expanded_search's), and propagate_attributes() explicitly only flows
# attributes forward to future child spans, never retroactively to an
# already-created parent (its own docs: "Pre-existing spans will NOT be
# retroactively updated"). The only mechanism that does this at all
# (Langfuse._create_trace_tags_via_ingestion, tags only, not general
# metadata) is a private, underscore-prefixed method not meant for
# application code to call directly.
#
# So instead: the caller that actually owns the "how many child calls
# happened" context (build_candidate_pool) calls reset_rate_limit_tracking()
# once before making its batch of calls, search_semantic_scholar increments
# the counter here at most once per call (not once per retry ATTEMPT within
# a call), and the caller reads get_rate_limited_call_count() afterward to
# fold into its own already-existing update_current_span(metadata=...) call
# — the same public API this file already uses everywhere else. Whichever
# @observe-decorated function turns out to be the actual root of a given
# trace (build_candidate_pool itself, when called directly by
# scripts/eval_retrieval.py's ranking-mode experiments; expanded_search,
# when it wraps build_candidate_pool in the live app's default path) ends
# up with this in its own metadata simply because each level already
# updates its own span the same way, not because of anything Langfuse-
# specific to "root" at all.
_rate_limit_tracker: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar(
    "_rate_limit_tracker", default=None,
)


def reset_rate_limit_tracking() -> None:
    """Starts (or restarts) counting how many search_semantic_scholar calls
    need at least one retry, for the rest of the current logical operation
    (call this once, before making a batch of search calls)."""
    _rate_limit_tracker.set([0])


def get_rate_limited_call_count() -> int:
    """How many search_semantic_scholar calls needed at least one retry
    since the last reset_rate_limit_tracking() call. 0 if tracking was
    never started (e.g. a caller that doesn't use this mechanism at all)."""
    tracker = _rate_limit_tracker.get()
    return tracker[0] if tracker is not None else 0

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


@observe(name="search_arxiv", capture_input=False, capture_output=False)
def search_arxiv(query: str, max_results: int = 20) -> list[Paper]:
    """Search arXiv and return normalized Paper records.

    Uses the `arxiv` package's Client, which handles arXiv's own rate
    limiting (3s between requests) internally. On any unexpected failure
    (network error, malformed feed page) we log and return whatever we
    already collected rather than raising.
    """
    if not query.strip():
        logger.warning("search_arxiv called with empty query")
        get_client().update_current_span(
            input={"query": query, "max_results": max_results},
            output={"count": 0, "papers": []},
        )
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

    get_client().update_current_span(
        input={"query": query, "max_results": max_results},
        output={"count": len(papers), "papers": paper_metadata(papers)},
    )
    return papers


@observe(name="search_semantic_scholar", capture_input=False, capture_output=False)
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
        get_client().update_current_span(
            input={"query": query, "max_results": max_results},
            output={"count": 0, "papers": []},
        )
        return []

    get_client().update_current_span(input={"query": query, "max_results": max_results})

    limit = min(max_results, _S2_MAX_LIMIT)
    headers = {"x-api-key": api_key} if api_key else {}
    params = {"query": query, "limit": limit, "fields": _S2_FIELDS}

    backoff = 1.0
    response = None
    already_counted_this_call = False
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
                get_client().update_current_span(output={"count": 0, "papers": []})
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
            # Explicit span metadata (not just a long duration to infer it
            # from) so rate-limit events are directly searchable/filterable
            # in Langfuse. update_current_span merges rather than replaces,
            # so a later successful attempt's output={"count": ...} update
            # doesn't erase this — a call that got rate-limited but
            # eventually succeeded still shows rate_limited=True with the
            # retry count it took, not just a silent success.
            get_client().update_current_span(metadata={"rate_limited": True, "retry_count": attempt})
            # Once per CALL, not once per retry attempt — a call that
            # retried 3 times before succeeding is still one affected call,
            # not three, for the caller's "how many child calls were
            # affected" rollup (see reset_rate_limit_tracking()/
            # get_rate_limited_call_count() above).
            if not already_counted_this_call:
                tracker = _rate_limit_tracker.get()
                if tracker is not None:
                    tracker[0] += 1
                already_counted_this_call = True
            if attempt == max_retries:
                logger.warning("Giving up on Semantic Scholar search for query %r", query)
                get_client().update_current_span(output={"count": 0, "papers": []})
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
        get_client().update_current_span(output={"count": 0, "papers": []})
        return []

    try:
        payload = response.json()
    except ValueError as exc:
        logger.warning(
            "Semantic Scholar returned a malformed/empty response body for query %r: %s",
            query, exc,
        )
        get_client().update_current_span(output={"count": 0, "papers": []})
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

    get_client().update_current_span(output={"count": len(papers), "papers": paper_metadata(papers)})
    return papers
