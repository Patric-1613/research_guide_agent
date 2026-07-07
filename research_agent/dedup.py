"""Phase 2: cross-source deduplication and merge.

A paper found on both arXiv and Semantic Scholar shows up as two Paper
records with different paper_ids, possibly different venues (preprint vs.
published), and only one of them carrying a citation count. deduplicate()
clusters records that are almost certainly the same paper and collapses
each cluster into one merged record.
"""

from __future__ import annotations

import logging

from rapidfuzz import fuzz

from research_agent.schema import Paper

logger = logging.getLogger(__name__)

# rapidfuzz.fuzz.ratio score out of 100. 90 tolerates minor differences
# (trailing punctuation, "&" vs "and", LaTeX-escaped characters) while still
# rejecting distinct-but-similar titles (e.g. a paper and its "-v2" survey).
TITLE_FUZZY_THRESHOLD = 90


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def _same_paper(a: Paper, b: Paper) -> bool:
    """True if a and b are almost certainly the same paper.

    DOI is checked first: it's an unambiguous exact identifier when both
    records have one. Title fuzzy-matching is the fallback, since arXiv and
    Semantic Scholar frequently disagree on whether a DOI is populated at
    all.
    """
    if a.doi and b.doi and a.doi.strip().lower() == b.doi.strip().lower():
        return True
    return fuzz.ratio(_normalize_title(a.title), _normalize_title(b.title)) >= TITLE_FUZZY_THRESHOLD


def _merge_authors(author_lists: list[list[str]]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for authors in author_lists:
        for name in authors:
            key = name.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(name.strip())
    return merged


def _merge_cluster(cluster: list[Paper]) -> Paper:
    if len(cluster) == 1:
        return cluster[0]

    # Richer abstract = the longer non-empty one. Semantic Scholar
    # abstracts are sometimes truncated or missing; arXiv's is the author's
    # own full abstract, so it usually wins, but we don't hardcode that.
    abstract = max((p.abstract for p in cluster if p.abstract), key=len, default=None)

    # Prefer a real venue name over arXiv's generic "arXiv preprint" placeholder.
    named_venues = [p.venue for p in cluster if p.venue and p.venue != "arXiv preprint"]
    venue = named_venues[0] if named_venues else next((p.venue for p in cluster if p.venue), None)

    years = [p.year for p in cluster if p.year is not None]
    # Earliest known year, usually the preprint date, which predates any
    # later peer-reviewed publication of the same work.
    year = min(years) if years else None

    doi = next((p.doi for p in cluster if p.doi), None)

    # Citation counts from different sources are different (possibly stale)
    # attempts to count citations of the *same* real paper, not additive
    # quantities, so we take the largest known count rather than summing.
    counts = [p.citation_count for p in cluster if p.citation_count is not None]
    citation_count = max(counts) if counts else None

    source_urls: dict[str, str] = {}
    for p in cluster:
        source_urls.update(p.source_urls)
    sources = sorted(source_urls) or sorted({p.source for p in cluster})

    merged = Paper(
        title=cluster[0].title,
        authors=_merge_authors([p.authors for p in cluster]),
        year=year,
        venue=venue,
        abstract=abstract,
        url=cluster[0].url,
        doi=doi,
        citation_count=citation_count,
        source="+".join(sources),
        paper_id="+".join(sorted({p.paper_id for p in cluster})),
    )
    merged.source_urls = source_urls
    return merged


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Cluster duplicate papers (DOI or fuzzy title match) and merge each cluster.

    O(n^2) comparisons — fine at the scale this agent runs at (tens of
    results per query), not intended for bulk corpus dedup.
    """
    clusters: list[list[Paper]] = []
    for paper in papers:
        for cluster in clusters:
            if any(_same_paper(existing, paper) for existing in cluster):
                cluster.append(paper)
                break
        else:
            clusters.append([paper])

    merged = [_merge_cluster(cluster) for cluster in clusters]
    n_dupes = len(papers) - len(merged)
    if n_dupes:
        logger.info("deduplicate: merged %d duplicate record(s) across sources", n_dupes)
    return merged
