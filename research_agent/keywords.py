"""Paper Keywords and Filtering, K4.1: deterministic, offline, model-free
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
re-extraction decision (see `scripts/re_extract_keywords.py`). It is NOT
wired into any cache-invalidation or persistence-migration mechanism;
changing it does not, by itself, cause any stored keywords to be
recomputed or treated as stale.

**K4.1 changes from yake-v1 (title-first, n=2, single-pass exact-case
dedup) to yake-v2:**

- The abstract, not `title + abstract`, is the primary extraction text --
  YAKE scores earlier text as more relevant, so title-first concatenation
  let title fragments dominate the output even when a substantial abstract
  was present. Title and abstract are now extracted *separately*, and at
  most ONE title-only candidate is ever allowed into the final result
  (appended last, so it only ever fills a slot the abstract itself didn't
  earn) -- confirmed by direct evidence (see this module's tests and the
  K4 investigation) that simple abstract-first reordering alone is not
  sufficient for a self-referential abstract that repeats its own title's
  terms.
- `n=3` (was `n=2`) so a genuine three-word compound (e.g. "natural
  language processing") can survive extraction as one complete candidate
  instead of two adjacent, incomplete two-word fragments.
- Candidate cleanup now NFKC-normalizes text and rejects any candidate
  containing an embedded comma/semicolon (YAKE, run over prose with a
  missing space after a comma, can produce a malformed candidate spanning
  a clause boundary, e.g. "Agentic AI,this" -- confirmed directly against
  real paper text).
- Redundancy resolution is now a dedicated, explicit, bidirectional pass
  over the *entire* candidate set (`_resolve_redundancy`): a candidate is
  dropped when its normalized token sequence is a contiguous subsequence
  of another candidate's, regardless of which one YAKE ranked first --
  n=3 alone does not remove shorter overlapping fragments (both
  "natural language processing" and "natural language" can be emitted as
  separate candidates for the same source text), and a naive
  keep-first/drop-later-duplicates pass misses the case where the shorter
  fragment happens to rank before the longer, complete phrase. Standalone
  uppercase acronyms (2-6 characters, e.g. "RAG") are exempt from this
  drop, so "RAG" survives even when "Agentic RAG" is also present.

**K4.1b -- excluding organization/affiliation candidates.** A paper's
title/abstract routinely names the authors' own institution (e.g. "...to
support Information Technology students at Hai Phong University",
"Scientists and operators at SLAC National Accelerator Laboratory...")
-- confirmed directly against real production data that this can rank
highly enough to become a paper's own top keyword (`_resolve_redundancy`
correctly assembles the COMPLETE institution name, which then survives
as one of the most relevant candidates precisely because K4.1's own
completeness fixes work correctly; being complete does not make it a
topic). `_is_organization_candidate()` rejects any candidate whose
canonical tokens include an organization/affiliation designator
(university, college, department, faculty, school, institute,
laboratory, lab, corporation, corp, company, consortium) as a COMPLETE
token, applied inside `_filter_candidates()` -- the same per-candidate
filtering stage `_is_noise_phrase()`/the clause-join check already use,
never a whole-paper or whole-sentence exclusion. Whole-token matching
(via the same `_canonical_tokens()` normalization redundancy resolution
already uses) is deliberate and load-bearing: a naive substring check
would wrongly reject real candidates like "annotated scientific
corpora"/"large textual corpora" (contain "corp" as a substring of
"corpora", never as its own token) and "conversation remains
labor-intensive" ("lab" as a substring of "labor"), both confirmed
directly against real production data. The designator list is
deliberately small, generic, and topic-agnostic -- no institution names,
no topic-specific terms, no NER model/LLM/embeddings/KeyBERT/new
dependency of any kind.
"""

from __future__ import annotations

import re
import unicodedata

import yake

KEYWORD_EXTRACTOR_VERSION = "yake-v2"

MAX_KEYWORDS = 6

# How many raw candidates to ask YAKE for before this module's own
# cleanup/comma-rejection/redundancy passes narrow that down to at most
# MAX_KEYWORDS. Generous headroom is needed because redundancy resolution
# now deliberately drops shorter fragments once a longer, complete phrase
# covering them is present.
_ABSTRACT_TOP = 25
# The title only ever contributes at most one final keyword (see module
# docstring), so it needs far fewer raw candidates.
_TITLE_TOP = 8

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

# A candidate containing a comma or semicolon is never a real phrase --
# it is YAKE having spanned a clause boundary, almost always because the
# source prose is missing a space after punctuation (e.g. "Agentic
# AI,this paper proposes..."). Confirmed directly against real paper
# text; a legitimate keyword phrase never legitimately contains one.
_CLAUSE_JOIN_RE = re.compile(r"[,;]")

# Unicode dash/hyphen variants that must compare equal to a plain space
# when building a canonical comparison key, so "Retrieval-Augmented
# Generation" and "Retrieval Augmented Generation" are recognized as the
# same phrase. Covers hyphen-minus plus the common Unicode dash block;
# deliberately does NOT touch the *display* surface form.
_DASH_VARIANTS_RE = re.compile("[-‐‑‒–—―−]")

# A standalone technical acronym (RAG, LLM, NLP, GPT4, ...): an uppercase
# token, 2-6 characters, that must be preserved even when a longer
# candidate containing it (e.g. "Agentic RAG") also survives -- YAKE
# emitting both is not redundancy, it is two different useful keywords.
_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9]{1,5}$")

# K4.1b: organization/affiliation designator tokens -- deliberately small,
# generic, and topic-agnostic (no institution names, no topic-specific
# terms). Matched as COMPLETE canonical tokens only (see
# `_is_organization_candidate` below), never a substring: "corp" must
# never match inside "corpora"/"corporate"/"incorporating", and "lab"
# must never match inside "labor"/"labor-intensive" -- both confirmed
# directly against real production candidate text, not hypothetical.
# "laboratory" is a separate, whole designator token of its own.
_ORGANIZATION_DESIGNATOR_TOKENS = frozenset({
    "university", "college", "department", "faculty", "school",
    "institute", "laboratory", "lab", "corporation", "corp",
    "company", "consortium",
})

_abstract_extractor = yake.KeywordExtractor(lan="en", n=3, top=_ABSTRACT_TOP, dedupLim=0.85)
_title_extractor = yake.KeywordExtractor(lan="en", n=3, top=_TITLE_TOP, dedupLim=0.85)


def _normalize(text: str) -> str:
    """NFKC-normalizes Unicode, strips URLs/DOIs/citation markers, collapses
    whitespace. Applied to both title and abstract before either reaches
    YAKE -- a citation fragment or bare URL is never real evidence of what
    a paper is about, and left in place it routinely gets scored as a
    spurious "keyword"."""
    text = unicodedata.normalize("NFKC", text)
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


def _canonical_tokens(phrase: str) -> list[str]:
    """The comparison key for redundancy/duplicate resolution: casefolded,
    Unicode dash variants treated as spaces, whitespace collapsed, split
    into tokens. Never used for display -- callers keep the original
    cleaned surface form for that."""
    key = phrase.casefold()
    key = _DASH_VARIANTS_RE.sub(" ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key.split(" ") if key else []


def _is_acronym(phrase: str) -> bool:
    return bool(_ACRONYM_RE.match(phrase.strip()))


def _is_organization_candidate(phrase: str) -> bool:
    """K4.1b: True if `phrase` names an organization/affiliation (a
    university, lab, company, ...) rather than the paper's own topic --
    its canonical tokens include an organization designator (see
    `_ORGANIZATION_DESIGNATOR_TOKENS`) as a COMPLETE token. Reuses
    `_canonical_tokens()` (casefold + Unicode dash-to-space + whitespace
    collapse + split) so matching is whole-token, never substring --
    "corpora"/"corpus"/"incorporating" never match "corp", "labor"/
    "labor-intensive" never match "lab". Applies to exactly the one
    candidate phrase being checked -- a different candidate from the
    same abstract that doesn't itself contain a designator token (a
    paper's own genuine research topic, however close in the source text
    to a mention of its authors' university) is entirely unaffected."""
    return any(token in _ORGANIZATION_DESIGNATOR_TOKENS for token in _canonical_tokens(phrase))


def _is_contiguous_subsequence(short_tokens: list[str], long_tokens: list[str]) -> bool:
    """True when `short_tokens` appears as a contiguous run inside
    `long_tokens` and is strictly shorter -- the structural test behind
    redundancy resolution. Deliberately NOT a "last word equals first
    word" adjacency check: two candidates that merely share a boundary
    word (e.g. "natural language" and "language processing") are NOT
    treated as redundant by this check alone, only when one is fully
    contained in the other."""
    n, m = len(short_tokens), len(long_tokens)
    if n == 0 or n >= m:
        return False
    for start in range(m - n + 1):
        if long_tokens[start : start + n] == short_tokens:
            return True
    return False


def _filter_candidates(raw_candidates: list[tuple[str, float]]) -> list[str]:
    """NFKC-normalizes, strips noise phrases, clause-spanning malformed
    candidates, and organization/affiliation candidates (K4.1b),
    preserving YAKE's own relevance order (ascending score)."""
    cleaned_candidates: list[str] = []
    for phrase, _score in raw_candidates:
        cleaned = unicodedata.normalize("NFKC", phrase).strip()
        if _is_noise_phrase(cleaned):
            continue
        if _CLAUSE_JOIN_RE.search(cleaned):
            continue
        if _is_organization_candidate(cleaned):
            continue
        cleaned_candidates.append(cleaned)
    return cleaned_candidates


def _dedup_canonical(candidates: list[str]) -> list[str]:
    """Drops exact duplicates under the canonical comparison key (e.g. a
    hyphenated and a spaced surface form of the same phrase), keeping the
    first (most relevant) occurrence's surface form for display."""
    seen: set[str] = set()
    deduped: list[str] = []
    for candidate in candidates:
        key = " ".join(_canonical_tokens(candidate))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _resolve_redundancy(candidates: list[str]) -> list[str]:
    """Bidirectional containment resolution over the *complete* candidate
    set: a candidate is dropped if its normalized tokens are a contiguous
    subsequence of ANY other candidate's tokens in the set, regardless of
    which one YAKE ranked first -- a one-pass "drop only if already kept"
    algorithm would miss the (confirmed, see this module's tests) case
    where the shorter, less-informative fragment ranks ahead of the
    longer, complete phrase. Acronyms are exempt (see `_is_acronym`)."""
    token_lists = [_canonical_tokens(candidate) for candidate in candidates]
    keep = [True] * len(candidates)
    for i, candidate in enumerate(candidates):
        if _is_acronym(candidate):
            continue
        for j in range(len(candidates)):
            if i == j:
                continue
            if _is_contiguous_subsequence(token_lists[i], token_lists[j]):
                keep[i] = False
                break
    return [candidate for candidate, kept in zip(candidates, keep) if kept]


def extract_keywords(title: str | None, abstract: str | None) -> list[str]:
    """Up to `MAX_KEYWORDS` keywords for one paper, in stable relevance
    order -- deterministic: the same (title, abstract) pair always
    produces the same list, on any machine, with no randomness and no
    external state.

    Returns `[]` whenever `abstract` is `None`, blank, or -- after
    stripping URLs/DOIs/citations and collapsing whitespace -- still
    shorter than a small evidence floor (see `_MIN_ABSTRACT_WORDS`/
    `_MIN_ABSTRACT_CHARS`). `title` alone is never enough evidence and is
    never evaluated against that floor; it only ever supplements a
    genuinely present abstract, and contributes at most one of the final
    keywords (see module docstring), exactly mirroring how a reader would
    judge "is there enough here to say what this paper is about" -- a
    title alone answers that; a title with no abstract does not.
    """
    if not abstract:
        return []
    normalized_abstract = _normalize(abstract)
    if len(normalized_abstract) < _MIN_ABSTRACT_CHARS or len(normalized_abstract.split()) < _MIN_ABSTRACT_WORDS:
        return []

    abstract_raw = _abstract_extractor.extract_keywords(normalized_abstract)
    abstract_candidates = _filter_candidates(abstract_raw)

    normalized_title = _normalize(title) if title else ""
    title_candidate: str | None = None
    if normalized_title:
        title_raw = _title_extractor.extract_keywords(normalized_title)
        title_candidates = _filter_candidates(title_raw)
        if title_candidates:
            title_candidate = title_candidates[0]

    # Title is appended LAST, never interleaved -- if the abstract alone
    # already fills MAX_KEYWORDS with candidates that survive redundancy
    # resolution, the title candidate is naturally truncated away below,
    # which is exactly "title may still contribute but must not crowd out
    # the abstract."
    merged_candidates = list(abstract_candidates)
    if title_candidate is not None:
        merged_candidates.append(title_candidate)

    deduped = _dedup_canonical(merged_candidates)
    resolved = _resolve_redundancy(deduped)

    return resolved[:MAX_KEYWORDS]
