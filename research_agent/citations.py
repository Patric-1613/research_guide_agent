"""APA and BibTeX citation formatting. Pure string formatting from a Paper's
already-verified fields — no LLM involved, since there's nothing here an LLM
would do better than deterministic code, and every LLM call is a chance to
hallucinate a citation detail.

Caveat: name parsing ("First Middle Last" -> "Last, F. M.") is a best-effort
heuristic (last whitespace-separated token = surname). It will mis-format
multi-word surnames (e.g. "van der Berg"), which APA itself has no fully
mechanical rule for either — full correctness would need a curated name
database, out of scope here.
"""

from __future__ import annotations

import re

from research_agent.schema import Paper


def _format_author_apa(name: str) -> str:
    parts = name.strip().split()
    if len(parts) == 1:
        return parts[0]
    *first_names, last = parts
    initials = " ".join(f"{p[0]}." for p in first_names if p)
    return f"{last}, {initials}"


def format_authors_apa(authors: list[str]) -> str:
    """APA 7 author list: 1 author 'Last, F.'; 2 'Last, F., & Last, F.';
    3-20 'Last, F., Last, F., ... & Last, F.'; 21+ first 19, ellipsis, last.
    """
    if not authors:
        return "Unknown Author"

    formatted = [_format_author_apa(a) for a in authors]

    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) <= 20:
        if len(formatted) == 2:
            return f"{formatted[0]}, & {formatted[1]}"
        return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"

    # 21+ authors: APA 7 lists the first 19, an ellipsis, then the final author.
    return ", ".join(formatted[:19]) + ", ... " + formatted[-1]


def format_apa_citation(paper: Paper) -> str:
    """Best-effort APA 7 reference-list entry.

    Uses whatever venue string we have (a real journal/conference name, or
    the literal "arXiv preprint") since the schema doesn't track
    volume/issue/page numbers (out of scope per the project brief).
    """
    authors = format_authors_apa(paper.authors)
    year = paper.year if paper.year is not None else "n.d."
    title = paper.title.rstrip(".")
    venue = paper.venue or "arXiv preprint"
    link = f"https://doi.org/{paper.doi}" if paper.doi else paper.url

    citation = f"{authors} ({year}). {title}. {venue}."
    if link:
        citation += f" {link}"
    return citation


_BIBTEX_KEY_STOPWORDS = {"a", "an", "the", "of", "for", "on", "in", "and", "to"}


def generate_bibtex_key(paper: Paper) -> str:
    first_author_last = paper.authors[0].strip().split()[-1] if paper.authors else "unknown"
    first_author_last = re.sub(r"[^A-Za-z]", "", first_author_last).lower() or "unknown"
    year = str(paper.year) if paper.year is not None else "nd"
    title_words = re.findall(r"[A-Za-z]+", paper.title.lower())
    first_word = next((w for w in title_words if w not in _BIBTEX_KEY_STOPWORDS), title_words[0] if title_words else "paper")
    return f"{first_author_last}{year}{first_word}"


def format_bibtex_citation(paper: Paper, key: str | None = None) -> str:
    key = key or generate_bibtex_key(paper)
    authors_bibtex = " and ".join(paper.authors) if paper.authors else "Unknown Author"
    entry_type = "article" if paper.venue and paper.venue != "arXiv preprint" else "misc"

    fields = [
        f'  title     = {{{paper.title}}}',
        f'  author    = {{{authors_bibtex}}}',
        f'  year      = {{{paper.year if paper.year is not None else "n.d."}}}',
    ]
    if paper.venue:
        fields.append(f'  journal   = {{{paper.venue}}}')
    if paper.doi:
        fields.append(f'  doi       = {{{paper.doi}}}')
    if paper.url:
        fields.append(f'  url       = {{{paper.url}}}')

    return f"@{entry_type}{{{key},\n" + ",\n".join(fields) + "\n}"
