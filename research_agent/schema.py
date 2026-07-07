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
