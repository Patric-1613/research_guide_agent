"""Round 3, phase 1: the interactive, multi-round triage session data model.

A TriageSession is the whole in-memory browsing experience from first
keyword to Summarize: an accumulated, cross-round-deduplicated pool of
papers and web articles, plus the user's basket picks. It is never touched
by SQLite or Chroma directly (see storage.py/embeddings.py) — the caller
(a future stateless API endpoint, per round 3's design) is responsible for
persistence, which only happens on an explicit save (phase 4) or embed
(phase 3).

Cross-round dedup reuses dedup.deduplicate() as-is (not reimplemented here),
applied to the *entire* accumulated pool plus each round's new results —
the same pattern agent.py's search tools already use within a single
search (`session.papers = deduplicate(session.papers + papers)`), just
carried across rounds instead of within one.

The one subtlety that pattern doesn't handle for free: dedup.py gives a
merged cluster of 2+ papers a brand new composite paper_id
("arxiv_id+s2_id", sorted). If a paper already in the pool (and possibly
already in the user's basket, or already recorded in an earlier round's
paper_ids_found) turns out to be a duplicate of something a later round
just found, its identity changes out from under it. _apply_round reconciles
this: it detects which merged records descend from which pre-round ids
(their paper_id's "+"-joined components are a superset of the old id's own
components, since a composite id can only ever grow via repeated merges of
this same pool) and rewrites every stored reference to the old id -
basket_paper_ids and every earlier round's paper_ids_found/new_paper_ids -
to the new canonical id. That's what makes "scrolling back to an earlier
round still correctly shows basket status" (phase 1 success criterion) true
regardless of which round a paper is displayed under.

Web articles need none of this: WebArticle identity is the URL (see
web_search.py / agent.py's _merge_web_articles), which never changes across
rounds, so there's no renaming concern for them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal

from research_agent.dedup import deduplicate
from research_agent.schema import Paper, WebArticle

PaperState = Literal["new", "seen", "basket"]


@dataclass
class Round:
    """One search action within a session: a keyword (or set of keywords)
    searched, and which papers/web articles it surfaced.

    paper_ids_found / web_urls_found are ALL ids surfaced by this round's
    search (new and resurfaced alike); new_paper_ids / new_web_urls are the
    subset that had never appeared in any earlier round of this session.
    Ids here are always kept current — if a later round's dedup renames an
    id this round originally recorded, _apply_round rewrites it in place so
    this round's own display always resolves against session.all_papers.
    """

    round_number: int
    keywords_used: list[str]
    timestamp: str
    paper_ids_found: list[str] = field(default_factory=list)
    new_paper_ids: list[str] = field(default_factory=list)
    web_urls_found: list[str] = field(default_factory=list)
    new_web_urls: list[str] = field(default_factory=list)


@dataclass
class TriageSession:
    """Ephemeral, in-memory-only state for one multi-round browsing session.

    Nothing here is persisted until the caller explicitly saves a bag
    (phase 4) or embeds the basket (phase 3) — this dataclass has no
    knowledge of SQLite or Chroma at all.
    """

    topic: str = ""
    rounds: list[Round] = field(default_factory=list)
    all_papers: dict[str, Paper] = field(default_factory=dict)
    all_web_articles: dict[str, WebArticle] = field(default_factory=dict)
    basket_paper_ids: set[str] = field(default_factory=set)
    basket_web_urls: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        """Plain-JSON representation of the whole session — the shape a
        stateless API endpoint (round 3 phase 2) can hand back to the
        caller and receive again verbatim on the next round-search call,
        since nothing about this session lives on the server between
        requests."""
        return {
            "topic": self.topic,
            "rounds": [asdict(r) for r in self.rounds],
            "all_papers": {pid: p.to_dict() for pid, p in self.all_papers.items()},
            "all_web_articles": {url: a.to_dict() for url, a in self.all_web_articles.items()},
            "basket_paper_ids": sorted(self.basket_paper_ids),
            "basket_web_urls": sorted(self.basket_web_urls),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TriageSession":
        return cls(
            topic=data.get("topic", ""),
            rounds=[Round(**r) for r in data.get("rounds", [])],
            all_papers={pid: Paper(**p) for pid, p in data.get("all_papers", {}).items()},
            all_web_articles={url: WebArticle(**a) for url, a in data.get("all_web_articles", {}).items()},
            basket_paper_ids=set(data.get("basket_paper_ids", [])),
            basket_web_urls=set(data.get("basket_web_urls", [])),
        )


def _id_components(paper_id: str) -> set[str]:
    return set(paper_id.split("+"))


def _merge_papers_into_round(
    session: TriageSession, found_papers: list[Paper]
) -> tuple[list[str], list[str]]:
    """Dedup `found_papers` against the session's accumulated pool (reusing
    dedup.deduplicate over the whole pool, same as agent.py's within-search
    pattern), update session.all_papers/basket_paper_ids/earlier rounds in
    place for any id that a merge renamed, and return this round's own
    (paper_ids_found, new_paper_ids) — scoped to just the records this
    round's search actually touched, not the whole accumulated pool.
    """
    if not found_papers:
        return [], []

    old_papers = list(session.all_papers.values())
    old_components = {p.paper_id: _id_components(p.paper_id) for p in old_papers}
    found_component_ids = {p.paper_id for p in found_papers}

    merged_pool = deduplicate(old_papers + found_papers)

    new_all_papers: dict[str, Paper] = {}
    id_rename_map: dict[str, str] = {}
    paper_ids_found: list[str] = []
    new_paper_ids: list[str] = []

    for merged in merged_pool:
        merged_components = _id_components(merged.paper_id)
        new_all_papers[merged.paper_id] = merged

        matched_old_ids = [
            old_id for old_id, comps in old_components.items() if comps <= merged_components
        ]
        for old_id in matched_old_ids:
            if old_id != merged.paper_id:
                id_rename_map[old_id] = merged.paper_id

        if merged_components & found_component_ids:
            paper_ids_found.append(merged.paper_id)
            if not matched_old_ids:
                new_paper_ids.append(merged.paper_id)

    session.all_papers = new_all_papers

    if id_rename_map:
        session.basket_paper_ids = {
            id_rename_map.get(pid, pid) for pid in session.basket_paper_ids
        }
        for past_round in session.rounds:
            past_round.paper_ids_found = [
                id_rename_map.get(pid, pid) for pid in past_round.paper_ids_found
            ]
            past_round.new_paper_ids = [
                id_rename_map.get(pid, pid) for pid in past_round.new_paper_ids
            ]

    return paper_ids_found, new_paper_ids


def _merge_web_articles_into_round(
    session: TriageSession, found_articles: list[WebArticle]
) -> tuple[list[str], list[str]]:
    """URL-keyed equivalent of _merge_papers_into_round. No renaming needed
    since a WebArticle's identity (its URL) never changes across rounds."""
    existing_before = set(session.all_web_articles)
    web_urls_found: list[str] = []
    new_web_urls: list[str] = []

    for article in found_articles:
        if article.url in web_urls_found:
            continue
        session.all_web_articles.setdefault(article.url, article)
        web_urls_found.append(article.url)
        if article.url not in existing_before:
            new_web_urls.append(article.url)

    return web_urls_found, new_web_urls


def add_round(
    session: TriageSession,
    keywords: list[str],
    found_papers: list[Paper],
    found_web_articles: list[WebArticle] | None = None,
) -> Round:
    """Run one search round: merge newly-found papers/web articles into the
    session's accumulated pool (deduping across all rounds so far, not just
    within this call), and record a Round describing what this round
    surfaced. Does not touch SQLite or Chroma."""
    paper_ids_found, new_paper_ids = _merge_papers_into_round(session, found_papers)
    web_urls_found, new_web_urls = _merge_web_articles_into_round(
        session, found_web_articles or []
    )

    round_ = Round(
        round_number=len(session.rounds) + 1,
        keywords_used=list(keywords),
        timestamp=datetime.now(timezone.utc).isoformat(),
        paper_ids_found=paper_ids_found,
        new_paper_ids=new_paper_ids,
        web_urls_found=web_urls_found,
        new_web_urls=new_web_urls,
    )
    session.rounds.append(round_)
    return round_


def paper_state(session: TriageSession, paper_id: str, round_: Round) -> PaperState:
    """Which of the three display states a paper is in when shown as part
    of `round_`. Basket status is a property of the paper (checked against
    session.basket_paper_ids directly), never of the round it's displayed
    under — so this gives the same answer for a paper regardless of which
    round's group the caller is currently rendering."""
    if paper_id in session.basket_paper_ids:
        return "basket"
    if paper_id in round_.new_paper_ids:
        return "new"
    return "seen"


def web_article_state(session: TriageSession, url: str, round_: Round) -> PaperState:
    """Web-article equivalent of paper_state — same three-state logic,
    kept as a separate function (not a shared helper keyed on a generic id)
    to match schema.py's deliberate separation of Paper and WebArticle."""
    if url in session.basket_web_urls:
        return "basket"
    if url in round_.new_web_urls:
        return "new"
    return "seen"
