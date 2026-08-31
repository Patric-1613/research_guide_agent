"""Shared, deterministic prompt-injection phrase detection.

One canonical registry of multi-word, high-confidence directive phrases,
used by every boundary that faces untrusted retrieved or echoed text.
Two call sites use it today, with different ACTIONS but identical pattern
definitions and normalization policy:

- `qa.py`'s web-relevance filter DETECTS and rejects a retrieved
  candidate before the relevance judge is ever consulted.
- `chat_summarization.py` REDACTS matched spans from a model-generated
  summary before it is rendered or persisted.

Deliberately narrow, and NOT a complete prompt-injection defense:

- multi-word phrases only -- a lone keyword ("system", "prompt", "model",
  "instructions", "override") never matches, because all of them appear
  routinely in genuine academic and technical writing
- no LLM classifier, no dependency, no broad adversarial-obfuscation
  coverage (leetspeak, zero-width characters, other languages, encoded
  payloads); that is future security-evaluation work
- a quoted academic discussion OF an attack phrase can still be flagged;
  a known, accepted limitation

Pattern IDs are stable and safe to surface in debug output; the matched
text itself is never returned.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

DEFAULT_REDACTION_PLACEHOLDER = "[redacted]"

# The justified union of the phrase registries that used to live
# separately in qa.py and chat_summarization.py. IDs are stable:
# `system_override`, `ignore_prior_instructions`,
# `disregard_prior_instructions`, `mark_candidate_as_relevant` and
# `forced_verdict_output` carry over from qa.py unchanged; `new_instructions`
# carries over from chat_summarization.py (previously the one pattern that
# existed on only one side). Each body uses `\s+` between words, so
# whitespace and newline variance within a phrase is handled by the
# regex; `detect()` additionally NFKC/case/whitespace-normalizes its
# input first.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("system_override", re.compile(r"\bsystem\s+override\b")),
    (
        "ignore_prior_instructions",
        re.compile(r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:prior|previous|above)\s+instructions\b"),
    ),
    (
        "disregard_prior_instructions",
        re.compile(r"\bdisregard\s+(?:all\s+)?(?:the\s+)?(?:prior|previous|above)\s+instructions\b"),
    ),
    (
        "mark_candidate_as_relevant",
        re.compile(r"\bmark\s+(?:this\s+)?(?:candidate|source|article)\s+as\s+(?:directly\s+)?relevant\b"),
    ),
    (
        "forced_verdict_output",
        re.compile(r"\b(?:return|output)\s+(?:a\s+|the\s+)?(?:required\s+)?relevance\s+(?:verdict|result|answer)\b"),
    ),
    (
        "directive_addressed_to_model",
        re.compile(r"\byou\s+(?:must|should)\s+(?:mark|classify|treat|report|answer|respond|say)\b"),
    ),
    ("new_instructions", re.compile(r"\bnew\s+instructions?\s*:")),
)

PATTERN_IDS: tuple[str, ...] = tuple(pattern_id for pattern_id, _ in _PATTERNS)

# For redaction only: the same phrase bodies, compiled case-insensitively
# so a matched span can be substituted in the ORIGINAL text without first
# lower-casing (which would rewrite every non-matched character too).
_REDACTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern.pattern, re.IGNORECASE) for _, pattern in _PATTERNS
)


@dataclass(frozen=True)
class InjectionMatch:
    detected: bool
    pattern_ids: list[str]


def normalize(text: str) -> str:
    """Unicode NFKC (folds full-width / compatibility character variants
    to a canonical form), lower-cased, then every run of whitespace --
    newlines included -- collapsed to a single space. Makes the phrase
    patterns robust to trivial formatting variance without attempting
    anything more elaborate (see the module docstring for what is
    explicitly out of scope)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower())


def detect(text: str) -> InjectionMatch:
    """Every canonical pattern id whose phrase appears in `text` after
    `normalize()`. Usually zero or one, occasionally more. Never returns
    the matched substring -- callers surface the stable pattern ids, not
    a copy of the malicious string."""
    normalized = normalize(text)
    matched = [pattern_id for pattern_id, pattern in _PATTERNS if pattern.search(normalized)]
    return InjectionMatch(detected=bool(matched), pattern_ids=matched)


def redact(text: str, placeholder: str = DEFAULT_REDACTION_PLACEHOLDER) -> str:
    """Replace each matched phrase span with `placeholder`, leaving the
    rest of `text` intact -- never whole-field deletion. The
    false-positive cost of a high-precision detector (losing one clause)
    is far below the availability cost of discarding an entire
    otherwise-good field over one matched span.

    `text` is NFKC-normalized first, so a phrase built from full-width or
    compatibility character variants is caught the same way `detect()`
    catches it. For ordinary text NFKC is a no-op, so non-matched content
    is returned unchanged; case and whitespace outside a matched span are
    always preserved."""
    text = unicodedata.normalize("NFKC", text)
    for pattern in _REDACTION_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text
