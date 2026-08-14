"""Paper Keywords and Filtering, K1: deterministic, offline, model-free
keyword extraction from a paper's title + abstract.

**No LLM call, no embeddings API, no model download, no network call --
ever.** `yake.KeywordExtractor` is a pure statistical/rule-based algorithm
(term frequency, position, casing, sentence spread, co-occurrence) with no
training step and no external weights to fetch; the same input always
produces the same output on any machine, confirmed directly (see this
module's own tests).

**Computation boundary**: this module is a pure function library. It is
called from exactly one place, `query_expansion.py::build_candidate_pool`,
immediately after `deduplicate(combined_raw)` returns -- never from
ingestion (per-source, pre-dedup: a paper later merged by `dedup.py` would
have its keywords silently dropped, since `dedup.py`'s own merge rebuilds
a fresh `Paper(...)` and does not carry an arbitrary extra field forward),
never from a serializer, an API route, or session loading, and never from
the frontend. Both `expanded_search()` (the one-shot search/summarize
path) and `refill_candidate_pool()` (curation refill) call
`build_candidate_pool()`, so this one boundary covers every paper the app
ever surfaces, computed exactly once per deduplicated paper.

`KEYWORD_EXTRACTOR_VERSION` is informational documentation only -- a
human-readable marker of which extractor/parameters produced a given
`Paper.keywords` value, for future debugging or a future deliberate
re-extraction decision. It is NOT wired into any cache-invalidation or
persistence-migration mechanism; changing it does not, by itself, cause
any stored keywords to be recomputed or treated as stale.
"""

from __future__ import annotations

import re

import yake

KEYWORD_EXTRACTOR_VERSION = "yake-v1"

MAX_KEYWORDS = 6

# Below this, an abstract is treated as "no real evidence" rather than
# genuinely analyzed -- YAKE can technically run on a handful of words,
# but the resulting "keywords" are typically just the input words back,
# not a meaningful extraction. Two independent floors (word count AND
# character count) so neither a handful of very long words nor a longer
# run of very short ones alone can slip past the other check.
_MIN_ABSTRACT_WORDS = 8
_MIN_ABSTRACT_CHARS = 30

_URL_RE = re.compile(r"https?://\S+")
_DOI_RE = re.compile(r"\bdoi\s*:\s*\S+", re.IGNORECASE)
# "[1]", "[1,2,3]", "[1-3]" -- numeric citation markers. Never a bracketed
# non-numeric aside (e.g. "[sic]"), which is real prose, not a citation.
_CITATION_MARKER_RE = re.compile(r"\[\d+(?:\s*[,\-]\s*\d+)*\]")
# "(Smith et al., 2020)", "(Smith, 2020)", "(Smith and Jones, 2020)" --
# author-year citations. Requires a capitalized leading word and a
# 4-digit year so an ordinary parenthetical remark ("(see below)") is
# never stripped.
_CITATION_PAREN_RE = re.compile(r"\([A-Z][\w.\-]*(?:\s+(?:and|&|et al\.?)\s+[\w.\-]*)?,?\s*\d{4}[a-z]?\)")

_extractor = yake.KeywordExtractor(lan="en", n=2, top=12, dedupLim=0.85)


def _normalize(text: str) -> str:
    """Strips URLs/DOIs/citation markers, collapses whitespace. Applied to
    both title and abstract before either reaches YAKE -- a citation
    fragment or bare URL is never real evidence of what a paper is about,
    and left in place it routinely gets scored as a spurious "keyword"."""
    text = _URL_RE.sub(" ", text)
    text = _DOI_RE.sub(" ", text)
    text = _CITATION_MARKER_RE.sub(" ", text)
    text = _CITATION_PAREN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_noise_phrase(phrase: str) -> bool:
    """A phrase YAKE returned that is never useful as a displayed keyword,
    checked independently of whatever normalization already ran on the
    source text -- a defensive second layer, not redundant: YAKE's own
    tokenizer can still emit a bare number or single character as its own
    unigram candidate even from already-normalized text."""
    stripped = phrase.strip()
    if len(stripped) <= 1:
        return True
    # A pure number (with optional . , % - separators, e.g. "99.2%", "3",
    # "1,234") carries no topical meaning on its own.
    if re.fullmatch(r"[\d.,%\-]+", stripped):
        return True
    if _URL_RE.search(stripped) or "@" in stripped:
        return True
    return False


def extract_keywords(title: str | None, abstract: str | None) -> list[str]:
    """Up to `MAX_KEYWORDS` keywords for one paper, in YAKE's own
    ascending-score (most-relevant-first) order -- deterministic: the same
    (title, abstract) pair always produces the same list, on any machine,
    with no randomness and no external state.

    Returns `[]` whenever `abstract` is `None`, blank, or -- after
    stripping URLs/DOIs/citations and collapsing whitespace -- still
    shorter than a small evidence floor (see `_MIN_ABSTRACT_WORDS`/
    `_MIN_ABSTRACT_CHARS`). `title` alone is never enough evidence and is
    never evaluated against that floor; it only ever supplements a
    genuinely present abstract, exactly mirroring how a reader would judge
    "is there enough here to say what this paper is about" -- a title
    alone answers that; a title with no abstract does not.
    """
    if not abstract:
        return []
    normalized_abstract = _normalize(abstract)
    if len(normalized_abstract) < _MIN_ABSTRACT_CHARS or len(normalized_abstract.split()) < _MIN_ABSTRACT_WORDS:
        return []

    normalized_title = _normalize(title) if title else ""
    combined = f"{normalized_title}. {normalized_abstract}" if normalized_title else normalized_abstract

    raw_candidates = _extractor.extract_keywords(combined)

    seen_lower: set[str] = set()
    keywords: list[str] = []
    for phrase, _score in raw_candidates:
        cleaned = phrase.strip()
        if _is_noise_phrase(cleaned):
            continue
        lower = cleaned.lower()
        # Case-insensitive dedup -- YAKE's own dedupLim is a Levenshtein-
        # similarity threshold between DIFFERENT candidate phrases (e.g.
        # collapsing "neural networks" and "neural network"), confirmed
        # directly NOT to collapse the same phrase in two different
        # casings (a mid-sentence "neural networks" and a sentence-initial
        # "Neural Networks" surfaced as two separate candidates in
        # testing) -- this exact-modulo-case check is what actually closes
        # that gap, deliberately kept a plain string comparison rather
        # than a fuzzy one (see this module's own docstring for why
        # RapidFuzz is not used here).
        if lower in seen_lower:
            continue
        seen_lower.add(lower)
        keywords.append(cleaned)
        if len(keywords) >= MAX_KEYWORDS:
            break
    return keywords
