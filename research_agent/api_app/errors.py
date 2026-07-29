"""Shared upstream-error handling for research_agent/api.py's endpoints.

Moved out of api.py (Phase 6) so this has a real, independent home —
api.py re-exports `_upstream_error_guard` so `research_agent.api.
_upstream_error_guard` and `patch.object(api, "_upstream_error_guard",
...)` keep working unchanged for anything still reaching it that way.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import arxiv
import requests
from fastapi import HTTPException
from openai import OpenAIError

logger = logging.getLogger(__name__)

# Exception types raised by the upstream services this API depends on
# (OpenAI, arXiv, Semantic Scholar/requests) that should surface as a clean
# error response, not a raw 500 with a stack trace leaked to the caller.
_UPSTREAM_ERRORS = (OpenAIError, arxiv.ArxivError, requests.RequestException)


@contextmanager
def _upstream_error_guard(service: str):
    """Wraps an endpoint body that calls out to arXiv, Semantic Scholar, or
    OpenAI. Those calls already retry/degrade internally where they can
    (ingestion.py, embeddings.py) — this is the last line of defense for
    what still gets through: a raw 500 with an internal stack trace leaking
    to the caller instead of a clean, actionable error response.

    HTTPException is re-raised untouched — those are this API's own
    intentional 404s (e.g. "search_id not found"), not upstream failures,
    and must not be swallowed into a 503.
    """
    try:
        yield
    except HTTPException:
        raise
    except _UPSTREAM_ERRORS as exc:
        logger.exception("Upstream service failure during %s", service)
        raise HTTPException(status_code=503, detail={"error": f"{service} service unavailable"}) from exc
