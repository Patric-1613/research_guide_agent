"""Normalized paper record shared by every data source."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Paper:
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str | None
    url: str | None
    doi: str | None
    citation_count: int | None
    source: str
    # Not in the phase-1 spec verbatim, but needed as a stable key for
    # dedup (phase 2) and the embedding cache (phase 3). Falls back to the
    # source name plus title if the API gave us nothing better.
    paper_id: str = field(default="")
    # source -> url. Populated automatically for single-source records;
    # dedup (phase 2) merges these so a collapsed record keeps every
    # source's link instead of dropping all but one.
    source_urls: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.paper_id:
            self.paper_id = f"{self.source}:{self.title.strip().lower()}"
        if not self.source_urls and self.url:
            self.source_urls = {self.source: self.url}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WebArticle:
    """A live web search result (round-2 enhancement 5) — deliberately NOT
    a repurposed Paper. Web articles have no authors/DOI/citation_count and
    are never peer-reviewed, so folding them into Paper would either force
    those fields to None everywhere (misleading — implies "we checked and
    there's no DOI" rather than "this concept doesn't apply here") or
    require a discriminator field bolted onto Paper. A separate type keeps
    the two corpora structurally distinct wherever they're handled, which
    matters most in qa.py: citations must be tagged by type ([Paper N] vs
    [Web N]) so a user can tell a peer-reviewed source from a web source at
    a glance, and that's much harder to guarantee if both corpora share one
    type.
    """

    title: str
    url: str
    snippet: str
    published_date: str | None
    source_domain: str

    def to_dict(self) -> dict:
        return asdict(self)
