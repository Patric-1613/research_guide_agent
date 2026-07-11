"""Round-2 enhancement 4: best-effort abstract recovery for papers Semantic
Scholar returned without one — confirmed root cause is publisher licensing
restrictions on certain hosted papers (e.g. some ScienceDirect-hosted
articles), not a normalization bug. Tries two free, no-key APIs, in order:
Unpaywall (open-access location lookup) then CrossRef's /works endpoint,
which sometimes carries a JATS-XML abstract even when Semantic Scholar has
none.

This is a separate module from ingestion.py rather than more functions
added there: ingestion.py's job is fetching NEW records from a source and
normalizing them; this module's job is enriching a Paper record that
already exists, keyed on a field it already has (doi), with its own cache
table. Keeping them apart keeps ingestion.py's one job — search a source,
return Paper records — undiluted.

Deliberately NOT full-text PDF extraction — matches the project's existing
"PDF full-text ingestion is out of scope for v1" boundary (see qa.py's
module docstring). This only ever recovers the same kind of short abstract
text embeddings.py already expects, never a full paper body.

Only attempted when abstract is None AND doi is present (per the brief) —
papers with a real abstract already, or no DOI to look up by, incur zero
extra latency; they fall straight through to embeddings.py's existing
title-fallback behavior untouched.

Caching design mirrors embeddings.py's philosophy (cache by the stable key
this specific lookup depends on) but the key itself is different on
purpose: embeddings.py hashes *content* because content is what determines
whether re-embedding is needed and paper_id can change under dedup merges.
Here, DOI is already the stable, unchanging lookup key Unpaywall/CrossRef
are queried by — hashing it would add nothing, so the cache is keyed on the
normalized DOI string directly, in its own table (not embedding_cache).
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from pathlib import Path

import requests

from research_agent.schema import Paper

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DB_PATH = DATA_DIR / "cache" / "enrichment.sqlite"

_UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"
_CROSSREF_URL = "https://api.crossref.org/works/{doi}"
# Unpaywall's terms require a real, working contact email as a query param
# on every request and actively rejects placeholder addresses (confirmed:
# it 422s "Please use your own email address" against an @example.com
# address) — so unlike everything else in this module, there's no safe
# generic default. UNPAYWALL_EMAIL must be set in .env for the Unpaywall
# step specifically; if it isn't, that one source is skipped (falls straight
# through to CrossRef, which only *recommends* a contact email in its
# User-Agent and doesn't reject a generic one).
_CROSSREF_CONTACT_FALLBACK = "research-agent-enrichment@example.com"
_REQUEST_TIMEOUT = 10
_JATS_TAG_RE = re.compile(r"<[^>]+>")


def _unpaywall_email() -> str | None:
    return os.getenv("UNPAYWALL_EMAIL") or None


def _crossref_contact() -> str:
    return os.getenv("UNPAYWALL_EMAIL") or _CROSSREF_CONTACT_FALLBACK


def _init_cache_db(path: Path = CACHE_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS abstract_cache (
            doi TEXT PRIMARY KEY,
            abstract TEXT,
            source TEXT,
            created_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _normalize_doi(doi: str) -> str:
    return doi.strip().lower()


def _get_cached(conn: sqlite3.Connection, doi: str) -> tuple[str | None, str | None] | None:
    row = conn.execute(
        "SELECT abstract, source FROM abstract_cache WHERE doi = ?", (_normalize_doi(doi),)
    ).fetchone()
    return (row[0], row[1]) if row else None


def _set_cached(conn: sqlite3.Connection, doi: str, abstract: str | None, source: str | None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO abstract_cache (doi, abstract, source, created_at) VALUES (?, ?, ?, ?)",
        (_normalize_doi(doi), abstract, source, time.time()),
    )
    conn.commit()


def _strip_jats_tags(text: str) -> str | None:
    cleaned = " ".join(_JATS_TAG_RE.sub(" ", text).split())
    return cleaned or None


def _fetch_unpaywall_abstract(doi: str) -> str | None:
    """Unpaywall's primary purpose is locating an open-access copy, not
    serving abstracts — its documented response schema has no guaranteed
    abstract field. This checks defensively for one anyway (occasionally
    present as incidental metadata) since it's a free, zero-cost check
    before falling through to CrossRef, which is the more reliable of the
    two for this specific purpose.
    """
    email = _unpaywall_email()
    if not email:
        logger.debug(
            "UNPAYWALL_EMAIL not set — skipping Unpaywall lookup for DOI %r, falling through to CrossRef", doi
        )
        return None

    try:
        response = requests.get(
            _UNPAYWALL_URL.format(doi=doi),
            params={"email": email},
            timeout=_REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        logger.info("Unpaywall lookup failed for DOI %r: %s", doi, exc)
        return None

    if response.status_code == 429:
        logger.info("Unpaywall rate-limited enrichment lookup for DOI %r — skipping, best-effort only", doi)
        return None
    if response.status_code != 200:
        return None

    try:
        data = response.json()
    except ValueError:
        logger.info("Unpaywall returned malformed JSON for DOI %r", doi)
        return None

    abstract = data.get("abstract") if isinstance(data, dict) else None
    return _strip_jats_tags(abstract) if abstract else None


def _fetch_crossref_abstract(doi: str) -> str | None:
    """CrossRef's /works/{doi} sometimes carries a JATS-XML-tagged abstract
    in its metadata even when Semantic Scholar has none — this is the more
    likely of the two sources to actually recover something.
    """
    try:
        response = requests.get(
            _CROSSREF_URL.format(doi=doi),
            headers={"User-Agent": f"research-agent-enrichment (mailto:{_crossref_contact()})"},
            timeout=_REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        logger.info("CrossRef lookup failed for DOI %r: %s", doi, exc)
        return None

    if response.status_code == 429:
        logger.info("CrossRef rate-limited enrichment lookup for DOI %r — skipping, best-effort only", doi)
        return None
    if response.status_code != 200:
        return None

    try:
        data = response.json()
    except ValueError:
        logger.info("CrossRef returned malformed JSON for DOI %r", doi)
        return None

    message = data.get("message") if isinstance(data, dict) else None
    abstract = message.get("abstract") if isinstance(message, dict) else None
    return _strip_jats_tags(abstract) if abstract else None


def recover_abstract(doi: str, conn: sqlite3.Connection | None = None) -> str | None:
    """Try Unpaywall then CrossRef for a missing abstract, given a DOI.
    Cached by DOI so a paper found unrecoverable isn't retried on every
    future search that happens to surface it again.

    Never raises: any failure here is best-effort enrichment, not a
    required step. embeddings.py already has a graceful fallback (embed
    title instead) for whatever this can't recover, so a failure here just
    means that fallback still applies — never a crash or a hang.
    """
    owns_conn = conn is None
    conn = conn or _init_cache_db()
    try:
        cached = _get_cached(conn, doi)
        if cached is not None:
            return cached[0]

        try:
            abstract = _fetch_unpaywall_abstract(doi)
            source = "unpaywall" if abstract else None
            if not abstract:
                abstract = _fetch_crossref_abstract(doi)
                source = "crossref" if abstract else None
        except Exception as exc:  # best-effort: never let enrichment break a search
            logger.warning("Unexpected error recovering abstract for DOI %r: %s", doi, exc)
            abstract, source = None, None

        _set_cached(conn, doi, abstract, source)
        if abstract:
            logger.info("Recovered abstract for DOI %r from %s", doi, source)
        return abstract
    finally:
        if owns_conn:
            conn.close()


def enrich_missing_abstracts(papers: list[Paper]) -> int:
    """In-place: for every paper with a doi but no abstract, try to recover
    one. Returns the count actually recovered. Papers that already have an
    abstract, or have no DOI to look up by, are skipped entirely — this
    must not add latency to the common case where Semantic Scholar/arXiv
    already returned a usable abstract.
    """
    candidates = [p for p in papers if p.doi and not p.abstract]
    if not candidates:
        return 0

    conn = _init_cache_db()
    recovered = 0
    try:
        for paper in candidates:
            abstract = recover_abstract(paper.doi, conn=conn)
            if abstract:
                paper.abstract = abstract
                recovered += 1
    finally:
        conn.close()

    if recovered:
        logger.info("enrich_missing_abstracts: recovered %d/%d missing abstract(s)", recovered, len(candidates))
    return recovered
