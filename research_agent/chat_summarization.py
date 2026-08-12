"""Usage Protection M3.1: deterministic foundation for bounding curation
chat's model-bound context without deleting/truncating the user-visible,
persisted `PaperPoolSession.chat_history`.

**Nothing in this module is wired into the live request path yet.** No
function here is called by `qa.py`, `curation_chat.py`, or any service
-- that wiring (plus real telemetry, real invalidation on delete/edit,
and the actual OpenAI summarization call) is M3.2. This module contains
only pure, deterministic helpers: no `OpenAI()` construction, no network
call, no mutation of its own inputs.

**Two things this module deliberately does NOT do, both by design, not
oversight:**

1. It never shrinks or deletes anything in `session.chat_history` --
   only decides what SUBSET of that (always-intact) history gets sent to
   the model on a given turn, and optionally what a persisted summary
   should replace that subset with in the MODEL-BOUND prompt only.
2. It never counts or bounds retrieved paper/web evidence context (the
   `context_sections` block `qa.py::_generate_node` builds from
   `retrieved_papers`/`retrieved_web_articles`). That's a separate,
   already-bounded concern (top_k retrieval, M2.2C's provider-fan-out
   limit) -- this module's own token/trigger accounting only ever
   considers the system prompt plus the conversation-history component,
   confirmed by every function signature below never accepting a
   paper/web evidence argument at all.

**Coverage is a raw `chat_history` entry count, never a turn count.**
`chat_summary_covers_history_count == 12` means the persisted summary
covers exactly `chat_history[:12]` -- never multiplied or divided by
two anywhere in this module. `UsagePolicy.chat_summary_keep_recent_turns`/
`chat_summary_min_new_turns`, by contrast, are counted in USER
TURNS/EXCHANGES (one user+assistant pair) -- the two units are
deliberately different and never silently converted into each other
except via `group_into_exchanges` below, which is the one place that
translates between them.

**Coupling posture**: this module does not import anything private from
`research_agent/qa.py` (no `_CITATION_MARKER_RE`, no `_PROMPT_INJECTION_PATTERNS`,
no `_detect_retrieved_prompt_injection`) -- those are `qa.py`'s own
private internals, coupling to them would tie two otherwise-independent
modules together for a small amount of shared logic. This module
re-implements its own small, local, high-precision bracket-marker strip
and instruction-override redaction instead, documented and tested
independently (see `_strip_bracket_citation_markers`/
`_redact_injection_phrases` below) -- deliberate, bounded duplication,
not a missed reuse opportunity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any

from langchain_core.messages.utils import count_tokens_approximately
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from research_agent.config import UsagePolicy
from research_agent.schema import Paper, WebArticle

# --- Part B: structured summary schema -------------------------------------

# Centralized, documented per-field/per-item limits -- a valid
# ChatHistorySummary cannot become arbitrarily large even though every
# individual constraint looks small in isolation (10 conclusions x 300
# chars each, etc.), matching the task's own "apply reasonable per-item
# limits ... cannot still become arbitrarily large" requirement. The
# list-length limits (10/10/30/30) and the two explicitly-specified
# scalar limits (500/300) are exactly what the task specified; the
# per-list-ITEM string limits (300 chars for a conclusion/question, 200
# for a paper_id, 2000 for a URL) were not numerically specified by the
# task and are this module's own provisional, centralized choice --
# revisit here, in one place, if they prove wrong in practice.
MAX_RESEARCH_INTENT_LENGTH = 500
MAX_RESOLVED_TERMINOLOGY_LENGTH = 300
MAX_KEY_CONCLUSIONS_ITEMS = 10
MAX_KEY_CONCLUSION_ITEM_LENGTH = 300
MAX_OPEN_QUESTIONS_ITEMS = 10
MAX_OPEN_QUESTION_ITEM_LENGTH = 300
MAX_PAPERS_DISCUSSED_ITEMS = 30
MAX_PAPER_ID_LENGTH = 200
MAX_WEB_ARTICLES_DISCUSSED_ITEMS = 30
MAX_WEB_URL_LENGTH = 2_000
MAX_USER_PREFERENCES_LENGTH = 300
MAX_UNRESOLVED_DISAGREEMENTS_LENGTH = 300

_BoundedConclusion = Annotated[str, Field(max_length=MAX_KEY_CONCLUSION_ITEM_LENGTH)]
_BoundedQuestion = Annotated[str, Field(max_length=MAX_OPEN_QUESTION_ITEM_LENGTH)]
_BoundedPaperId = Annotated[str, Field(max_length=MAX_PAPER_ID_LENGTH)]
_BoundedUrl = Annotated[str, Field(max_length=MAX_WEB_URL_LENGTH)]


class ChatHistorySummary(BaseModel):
    """A structured, JSON-compatible replacement for free-text
    conversation-history summarization -- chosen (over a free-text blob,
    e.g. `SummarizationMiddleware`'s own default) specifically so
    invalidation/citation-safety checks can inspect fields directly
    instead of parsing prose. `model_config`'s `extra="forbid"` is the
    concrete mechanism behind "validation never accepts extra control/
    application metadata" -- a dict carrying a stray `exchange_id`/
    `cited_papers`/etc. key fails validation outright rather than being
    silently dropped (Pydantic v2's own default `extra="ignore")` would
    have silently dropped it, which is NOT the same guarantee).

    No citation markers ([Paper N]/[Web N]/[N]) belong inside any field
    here -- this object is never a citable source (see this module's own
    `render_summary_message` and `validate_replacement_summary`, neither
    of which trust bracket-marker-shaped text). `papers_discussed`/
    `web_articles_discussed` are real `paper_id`/URL values ONLY, never
    model-authored titles -- titles are always resolved locally against
    the real session source pools at render time (see
    `render_summary_message`), never trusted from model output.
    """

    model_config = ConfigDict(extra="forbid")

    research_intent: str = Field(max_length=MAX_RESEARCH_INTENT_LENGTH)
    resolved_terminology: str = Field(default="", max_length=MAX_RESOLVED_TERMINOLOGY_LENGTH)
    key_conclusions: list[_BoundedConclusion] = Field(default_factory=list, max_length=MAX_KEY_CONCLUSIONS_ITEMS)
    open_questions: list[_BoundedQuestion] = Field(default_factory=list, max_length=MAX_OPEN_QUESTIONS_ITEMS)
    papers_discussed: list[_BoundedPaperId] = Field(default_factory=list, max_length=MAX_PAPERS_DISCUSSED_ITEMS)
    web_articles_discussed: list[_BoundedUrl] = Field(default_factory=list, max_length=MAX_WEB_ARTICLES_DISCUSSED_ITEMS)
    user_preferences: str = Field(default="", max_length=MAX_USER_PREFERENCES_LENGTH)
    unresolved_disagreements: str = Field(default="", max_length=MAX_UNRESOLVED_DISAGREEMENTS_LENGTH)


# --- Shared internals -------------------------------------------------------

def _strip_to_role_content(entries: list[dict]) -> list[dict]:
    """Same discipline as qa.py's own `capped_history`: brand-new dicts
    containing only {role, content}, regardless of what extra persisted
    metadata keys (exchange_id, cited_papers, used_web_search, ...) the
    input entries carry -- this is what's actually eligible to reach an
    LLM prompt (either as the verbatim recent tail, or as the raw
    material handed to a future summarization call), so no persisted
    metadata may leak into either. Never mutates the input."""
    return [{"role": e["role"], "content": e["content"]} for e in entries]


_BRACKET_CITATION_MARKER_RE = re.compile(r"\[(?:Paper|Web)?\s*\d+\]")


def _strip_bracket_citation_markers(text: str) -> str:
    """Removes anything shaped like `[Paper N]`, `[Web N]`, or a bare
    `[N]` -- a summary is never a citable source (see this module's own
    docstring), so a marker-shaped string echoed from a summarized chat
    turn's content must never survive into rendered model context or a
    persisted replacement summary looking like a real, resolvable
    citation."""
    return _BRACKET_CITATION_MARKER_RE.sub("", text)


# High-precision, low-recall, deliberately narrow phrase patterns --
# every pattern is a multi-word phrase describing an instruction
# DIRECTED AT the model, never a single keyword ("system", "prompt",
# "instructions" alone all appear routinely in genuine academic/
# technical writing and must never trigger this on their own). This is
# a SEPARATE, LOCAL, independent implementation from qa.py's own
# comparable `_PROMPT_INJECTION_PATTERNS` guard -- see this module's own
# docstring for why it is not imported/reused. Known, accepted, and
# explicitly NOT solved here: obfuscation (leetspeak, zero-width
# characters, other languages, encoded payloads), and a quoted academic
# discussion OF prompt injection could still be redacted incorrectly --
# the same limitations qa.py's own guard documents for itself.
_INJECTION_REDACTION_PLACEHOLDER = "[redacted]"
_INJECTION_PHRASE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bsystem\s+override\b", re.IGNORECASE),
    re.compile(r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:prior|previous|above)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:the\s+)?(?:prior|previous|above)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\byou\s+(?:must|should)\s+(?:mark|classify|treat|report|answer|respond|say)\b", re.IGNORECASE),
    re.compile(r"\bnew\s+instructions?\s*:", re.IGNORECASE),
]


def _redact_injection_phrases(text: str) -> str:
    """Replaces only the matched span with a neutral placeholder --
    deliberately NOT whole-field rejection for one flagged phrase. A
    high-precision detector's false-positive cost (losing one clause) is
    much lower than the availability cost of discarding an entire
    otherwise-good summary field over one matched span. See this
    module's own docstring for the documented limitations of this
    detector."""
    for pattern in _INJECTION_PHRASE_PATTERNS:
        text = pattern.sub(_INJECTION_REDACTION_PLACEHOLDER, text)
    return text


def _sanitize_free_text(text: str) -> str:
    return _redact_injection_phrases(_strip_bracket_citation_markers(text))


def _dedupe_stable(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# --- Part D.1: validating/loading the persisted structured summary --------

def load_persisted_summary_state(
    chat_summary: dict[str, Any] | None,
    chat_summary_covers_history_count: int,
    history_length: int,
) -> tuple[ChatHistorySummary | None, int]:
    """Returns `(valid_summary_or_None, safe_covers_count)` -- always a
    self-consistent pair: `valid_summary is None` implies
    `safe_covers_count == 0`, never a stale/inconsistent leftover count.

    Conservative by design (per the task's own instruction): ANY of the
    "impossible" states below is treated as "no valid summary at all",
    not partially trusted or clamped-and-kept -- a summary whose claimed
    coverage boundary can't be trusted can't safely be used for anything
    (we would not know where "already covered" actually ends, so we
    could not safely build a non-duplicating/non-gapped verbatim tail
    around it either):
      - `chat_summary` is `None` (nothing persisted yet, or `covers_count`
        is a stale leftover from before a full invalidation -- either
        way, `covers_count` is IGNORED here, never trusted with no
        summary object behind it).
      - `chat_summary_covers_history_count` is negative.
      - `chat_summary_covers_history_count` exceeds the CURRENT
        `history_length` (the summary claims to cover more than exists
        right now -- e.g. a delete/edit ran without invalidating it, or
        corrupt data).
      - `chat_summary` fails `ChatHistorySummary` schema validation
        (malformed dict, extra keys, out-of-bound field).

    Never raises -- a `pydantic.ValidationError` on a malformed
    persisted dict is caught here, not propagated, so model-context
    construction (via `build_chat_context` below) can never crash
    because an old/corrupt session contains bad summary state. This
    function is read-only: it never mutates or re-persists anything,
    consistent with "do not silently rewrite persisted state during a
    read-only load."
    """
    if chat_summary is None:
        return None, 0
    if chat_summary_covers_history_count < 0:
        return None, 0
    if chat_summary_covers_history_count > history_length:
        return None, 0
    try:
        validated = ChatHistorySummary.model_validate(chat_summary)
    except ValidationError:
        return None, 0
    return validated, chat_summary_covers_history_count


# --- Part D.2: exchange boundaries from raw history -------------------------

def group_into_exchanges(history: list[dict]) -> list[list[int]]:
    """Groups raw `chat_history` entry INDICES into exchange-sized
    chunks, oldest to newest, preserving original order. A normal
    exchange is one `role == "user"` entry immediately followed by one
    `role == "assistant"` entry -- `curation_chat.py::chat_turn`'s own
    documented invariant ("exactly ONE user entry and ONE assistant
    entry get appended per call, in that order"). Any entry that does
    not fit that exact shape -- two same-role entries in a row, an
    unpaired trailing entry, an unrecognized role -- becomes its own
    SINGLE-entry group instead of being merged incorrectly, silently
    dropped, or raised on. This is intentionally conservative: it is
    always safe to treat "not clearly part of a pair" as its own
    boundary, since every boundary this module ever selects (see
    `select_summarizable_slice` below) is chosen to land on a GROUP
    start -- which this grouping guarantees can never fall inside a
    genuine, correctly-shaped pair, regardless of what malformed shapes
    appear elsewhere in the same history.

    Legacy entries (no `exchange_id`) are grouped purely by role
    alternation, exactly like current entries -- `exchange_id` is never
    read or required by this function at all, so a legacy entry
    participates identically to a current one.

    Never mutates `history`. `O(n)`, one pass.
    """
    groups: list[list[int]] = []
    i = 0
    n = len(history)
    while i < n:
        role = history[i].get("role")
        if role == "user" and i + 1 < n and history[i + 1].get("role") == "assistant":
            groups.append([i, i + 1])
            i += 2
        else:
            groups.append([i])
            i += 1
    return groups


# --- Part D.3: selecting the summarizable slice -----------------------------

def select_summarizable_slice(
    history: list[dict],
    covers_count: int,
    keep_recent_turns: int,
) -> tuple[list[dict], int]:
    """Returns `(history_to_summarize, retain_boundary_index)`.

    `retain_boundary_index` is the raw `chat_history` index where the
    always-verbatim "recent" window begins -- the start index of the
    `keep_recent_turns`-th-from-the-end exchange GROUP (see
    `group_into_exchanges`), or `0` if there are `<= keep_recent_turns`
    groups in total (everything is retained, nothing is old enough to be
    a summarization candidate at all -- "fewer than the retention amount
    selects nothing"). Because this is always a real group's own start
    index, it can never fall inside a valid pair.

    `history_to_summarize` is `history[start:retain_boundary_index]`,
    stripped to `{role, content}` (see `_strip_to_role_content`) --
    EXCLUDING anything already covered (`start = max(covers_count, 0)`,
    clamped to never exceed `retain_boundary_index`) and excluding the
    retained recent window. `covers_count` is used directly as a raw
    entry index here -- never multiplied or divided by two. Returns
    `([], retain_boundary_index)` when there is nothing newly eligible
    (either nothing old enough exists, or `covers_count` already reaches
    the retained boundary -- "incremental selection never resummarizes
    covered entries").

    Never mutates `history` or its dicts.
    """
    groups = group_into_exchanges(history)
    if len(groups) <= keep_recent_turns:
        retain_boundary_index = 0
    else:
        retain_boundary_index = groups[-keep_recent_turns][0]

    start_index = min(max(covers_count, 0), retain_boundary_index)
    if start_index >= retain_boundary_index:
        return [], retain_boundary_index
    return _strip_to_role_content(history[start_index:retain_boundary_index]), retain_boundary_index


# --- Part D.4: trigger decision ---------------------------------------------

def should_trigger_summarization(
    *,
    system_prompt: str,
    history: list[dict],
    covers_count: int,
    eligible_group_count: int,
    has_valid_previous_summary: bool,
    policy: UsagePolicy,
) -> bool:
    """Whether a (re)summarization pass should run this turn.

    Token estimate uses `langchain_core.messages.utils.
    count_tokens_approximately` (installed, already a transitive import
    of this project's own pinned `langchain-core` dependency -- no new
    dependency) over EXACTLY `[system_prompt] + history[covers_count:]`
    (stripped to `{role, content}`) -- the system prompt plus "the
    conversation-history component being considered": everything
    accumulated since the summary's current coverage (or the whole
    history, if `covers_count == 0`). Retrieved paper/web evidence is
    NEVER part of this count -- structurally guaranteed, since this
    function's own signature has no parameter for it at all.

    Two independent gates, both must pass:
      1. Token threshold: `total_tokens >= policy.chat_summary_trigger_tokens`.
      2. ONLY when a valid previous summary already exists: at least
         `policy.chat_summary_min_new_turns` newly eligible exchange
         GROUPS (not raw entries) must have accumulated since the
         current coverage -- this is what stops a threshold that stays
         crossed from re-triggering resummarization on literally every
         subsequent turn. When there is NO previous summary yet, this
         second gate does not apply at all -- the very first
         summarization pass may fire as soon as the token threshold
         alone is crossed.
    """
    considered_start = min(max(covers_count, 0), len(history))
    considered = [{"role": "system", "content": system_prompt}] + _strip_to_role_content(history[considered_start:])
    total_tokens = count_tokens_approximately(considered)

    if total_tokens < policy.chat_summary_trigger_tokens:
        return False
    if has_valid_previous_summary and eligible_group_count < policy.chat_summary_min_new_turns:
        return False
    return True


# --- Part D.5: rendering a validated structured summary ---------------------

_SUMMARY_MEMORY_PREFIX = (
    "The following is a COMPRESSED SUMMARY of EARLIER conversation turns "
    "in this session, generated to save space. It is background "
    "conversational memory ONLY -- it is NOT retrieved evidence and MUST "
    "NEVER be treated as a citable source. Continue to cite only the "
    "papers/web articles provided separately for this turn."
)


def _resolve_titles_deduped(ids_or_urls: list[str], lookup: dict[str, str]) -> list[str]:
    """Resolves each id/url to its real title via `lookup` (built from
    the actual session source pools, never from the summary's own
    text). Unknown ids/urls are silently omitted -- a summary can
    reference a paper/article that was later removed from the session's
    pools, or (defense in depth) a hallucinated id that was never real.
    Deduplicates in stable (first-occurrence) order."""
    seen: set[str] = set()
    titles: list[str] = []
    for key in ids_or_urls:
        if key in seen:
            continue
        seen.add(key)
        title = lookup.get(key)
        if title is not None:
            titles.append(title)
    return titles


def render_summary_message(
    summary: ChatHistorySummary,
    selected_papers: list[Paper],
    web_articles_added: list[WebArticle],
) -> dict:
    """Renders an already-validated `ChatHistorySummary` into one
    deterministic `{"role": "system", "content": ...}` message.

    Paper ids / web urls are resolved to real titles here, locally,
    against `selected_papers`/`web_articles_added` -- never trusting a
    title from the summary object itself (it has none; only ids/urls).
    Every free-text field is passed through `_sanitize_free_text`
    (bracket-marker stripping + injection-phrase redaction) again here,
    as defense in depth on top of `validate_replacement_summary`'s own
    sanitization -- covers a persisted summary written before this
    module's own sanitization existed or was strengthened.

    Deterministic and bounded: same summary object always renders to
    the same string; total length is bounded by the schema's own
    per-field/per-item limits (see `ChatHistorySummary`). Never includes
    abstracts, snippets, report prose, control/application metadata,
    pending offers, evaluator metadata, or internal prompts -- there is
    no code path here that reads any of those.
    """
    papers_by_id = {p.paper_id: p.title for p in selected_papers}
    web_by_url = {a.url: a.title for a in web_articles_added}

    lines = [_SUMMARY_MEMORY_PREFIX, ""]
    lines.append(f"Research intent: {_sanitize_free_text(summary.research_intent)}")
    if summary.resolved_terminology:
        lines.append(f"Resolved terminology: {_sanitize_free_text(summary.resolved_terminology)}")
    if summary.key_conclusions:
        lines.append("Key conclusions so far:")
        lines.extend(f"- {_sanitize_free_text(c)}" for c in summary.key_conclusions)
    if summary.open_questions:
        lines.append("Open questions:")
        lines.extend(f"- {_sanitize_free_text(q)}" for q in summary.open_questions)

    resolved_paper_titles = _resolve_titles_deduped(summary.papers_discussed, papers_by_id)
    if resolved_paper_titles:
        lines.append("Papers discussed earlier: " + "; ".join(resolved_paper_titles))
    resolved_web_titles = _resolve_titles_deduped(summary.web_articles_discussed, web_by_url)
    if resolved_web_titles:
        lines.append("Web articles discussed earlier: " + "; ".join(resolved_web_titles))

    if summary.user_preferences:
        lines.append(f"User preferences noted: {_sanitize_free_text(summary.user_preferences)}")
    if summary.unresolved_disagreements:
        lines.append(f"Unresolved/uncertain points: {_sanitize_free_text(summary.unresolved_disagreements)}")

    return {"role": "system", "content": "\n".join(lines)}


# --- Part D.6: bounded conversation-history component -----------------------

@dataclass
class ChatContextResult:
    """The structured result `build_chat_context` returns -- everything
    a future M3.2 orchestration needs, in one place, so it never has to
    re-derive coverage/trigger state itself."""

    # The bounded conversation-history component: ready to splice
    # directly where `qa.py::capped_history`'s return value is spliced
    # today. NEVER includes the system answer-prompt itself (same scope
    # `capped_history` already has) and NEVER includes raw, unbounded
    # full history, regardless of any field below.
    model_messages: list[dict]
    # Whether a (re)summarization pass is warranted right now (M3.1 never
    # acts on this -- it's exposed for M3.2's real orchestration).
    should_summarize: bool
    # The exact new, currently-uncovered old-history slice (stripped to
    # {role, content}) a future summarizer would consume, PLUS the
    # existing valid_previous_summary below -- both empty/None when
    # should_summarize is False.
    history_to_summarize: list[dict]
    # The raw chat_history index a NEW summary would cover if
    # history_to_summarize were summarized right now. Equal to the
    # current safe coverage count when should_summarize is False (no
    # change proposed).
    prospective_coverage_count: int
    # The validated, currently-usable previous summary (None if there
    # isn't one, or if the persisted one was malformed -- see
    # used_emergency_trim).
    valid_previous_summary: ChatHistorySummary | None
    # True only when a persisted chat_summary EXISTED but failed
    # validation (load_persisted_summary_state discarded it) -- NOT true
    # for the ordinary "never summarized yet" case, which is not a
    # failure/degradation of anything.
    used_emergency_trim: bool


def build_chat_context(
    history: list[dict],
    chat_summary: dict[str, Any] | None,
    chat_summary_covers_history_count: int,
    policy: UsagePolicy,
    system_prompt: str,
    selected_papers: list[Paper] | None = None,
    web_articles_added: list[WebArticle] | None = None,
) -> ChatContextResult:
    """The one function that composes every helper above into the
    bounded conversation-history component for one curation-chat turn.
    Never called from the live request path yet (see this module's own
    docstring) -- M3.2 is what actually wires this in.

    - If no valid previous summary exists and nothing is old enough to
      be summarization-eligible: `model_messages` is exactly what
      `qa.py::capped_history(history, max_turns=policy.
      chat_summary_keep_recent_turns)` already returns today -- byte-
      identical no-op-below-the-cap behavior preserved.
    - If a valid previous summary exists: `model_messages` is
      `[render_summary_message(...)] + <bounded recent verbatim tail>`
      -- the summary stands in for everything it covers; recent turns
      are still sent verbatim, in full.
    - If the persisted summary is malformed/inconsistent
      (`used_emergency_trim=True`): it is ignored entirely, falling back
      to the same bounded, capped-history-only behavior as "no valid
      summary" above -- never crashes, never sends raw unbounded
      history.
    - Regardless of `should_summarize`'s value, `model_messages` is
      ALWAYS already a safe, currently-usable, bounded context -- M3.1
      never withholds a usable answer path while "waiting" for a future
      summarization pass that doesn't exist yet in this phase.

    Never mutates `history`, `session.chat_history`, or any of their
    dicts.
    """
    selected_papers = selected_papers or []
    web_articles_added = web_articles_added or []

    valid_summary, safe_covers_count = load_persisted_summary_state(
        chat_summary, chat_summary_covers_history_count, len(history),
    )
    used_emergency_trim = chat_summary is not None and valid_summary is None

    history_to_summarize, retain_boundary_index = select_summarizable_slice(
        history, safe_covers_count, policy.chat_summary_keep_recent_turns,
    )

    should_summarize = False
    if history_to_summarize:
        groups = group_into_exchanges(history)
        eligible_group_count = sum(
            1 for g in groups if g[0] >= safe_covers_count and g[-1] < retain_boundary_index
        )
        should_summarize = should_trigger_summarization(
            system_prompt=system_prompt,
            history=history,
            covers_count=safe_covers_count,
            eligible_group_count=eligible_group_count,
            has_valid_previous_summary=valid_summary is not None,
            policy=policy,
        )

    if should_summarize:
        prospective_coverage_count = retain_boundary_index
    else:
        history_to_summarize = []
        prospective_coverage_count = safe_covers_count

    # ALWAYS the bounded recent window, regardless of should_summarize --
    # M3.1 never has a real summarizer to call, so "summarization is
    # needed" only ever changes the METADATA exposed here, never
    # model_messages' own boundedness.
    tail_start = max(retain_boundary_index, safe_covers_count)
    verbatim_tail = _strip_to_role_content(history[tail_start:])

    model_messages: list[dict] = []
    if valid_summary is not None:
        model_messages.append(render_summary_message(valid_summary, selected_papers, web_articles_added))
    model_messages.extend(verbatim_tail)

    return ChatContextResult(
        model_messages=model_messages,
        should_summarize=should_summarize,
        history_to_summarize=history_to_summarize,
        prospective_coverage_count=prospective_coverage_count,
        valid_previous_summary=valid_summary,
        used_emergency_trim=used_emergency_trim,
    )


# --- Part E: summary replacement validation ---------------------------------

def validate_replacement_summary(
    proposed: dict[str, Any],
    selected_papers: list[Paper],
    web_articles_added: list[WebArticle],
) -> ChatHistorySummary:
    """Validates a PROPOSED replacement summary (e.g. a future M3.2
    summarization call's raw structured-output payload) and returns a
    normalized, JSON-compatible `ChatHistorySummary`.

    - Schema validation first (`ChatHistorySummary.model_validate`) --
      raises `pydantic.ValidationError` on anything schema-invalid
      (wrong type, over a length/count bound, an extra/unknown key).
      This is a genuine failure for the CALLER (M3.2) to handle -- no
      automatic fallback model call happens here, and this function
      never silently truncates arbitrary prose mid-word to force
      something to fit; an over-length field is a rejection, not a
      truncation.
    - `papers_discussed`/`web_articles_discussed` are filtered down to
      ids/urls that actually exist in `selected_papers`/
      `web_articles_added` right now, then deduplicated in stable order.
    - Every free-text field is sanitized (`_sanitize_free_text`: bracket-
      marker stripping + injection-phrase redaction -- see this
      module's own docstring for the documented, high-precision-only
      posture of the redaction).
    - Re-validated (not `model_copy`, which would skip validation) after
      sanitization/filtering -- if redaction ever pushed a field over
      its own schema bound, that surfaces as a clean `ValidationError`
      here rather than silently producing an over-length object.
    - Rejects an empty/meaningless result: if `research_intent` is blank
      after sanitization, raises `ValueError` -- a replacement summary
      that says nothing is not a useful replacement.
    - Does NOT set/touch any `updated_at` timestamp -- that's the live
      orchestration's job (M3.2), set only once persistence of a
      genuinely successful summarization actually happens.
    """
    validated = ChatHistorySummary.model_validate(proposed)

    known_paper_ids = {p.paper_id for p in selected_papers}
    known_urls = {a.url for a in web_articles_added}

    cleaned = ChatHistorySummary.model_validate({
        **validated.model_dump(),
        "papers_discussed": _dedupe_stable(
            [pid for pid in validated.papers_discussed if pid in known_paper_ids]
        ),
        "web_articles_discussed": _dedupe_stable(
            [url for url in validated.web_articles_discussed if url in known_urls]
        ),
        "research_intent": _sanitize_free_text(validated.research_intent),
        "resolved_terminology": _sanitize_free_text(validated.resolved_terminology),
        "key_conclusions": [_sanitize_free_text(c) for c in validated.key_conclusions],
        "open_questions": [_sanitize_free_text(q) for q in validated.open_questions],
        "user_preferences": _sanitize_free_text(validated.user_preferences),
        "unresolved_disagreements": _sanitize_free_text(validated.unresolved_disagreements),
    })

    if not cleaned.research_intent.strip():
        raise ValueError("replacement summary is empty/meaningless after normalization (blank research_intent)")

    return cleaned


# --- Part F: invalidation ----------------------------------------------------

def determine_invalidation(
    covers_count: int,
    earliest_affected_index: int,
    has_valid_summary: bool,
) -> bool:
    """Whether a history mutation (delete/edit/truncation) whose
    earliest affected raw `chat_history` index is `earliest_affected_index`
    invalidates the current summary coverage.

    Simple, conservative rule (full invalidation, not semantic
    subtraction -- see this module's own docstring):
      - No valid summary at all -> nothing to invalidate -> `False`.
      - The mutation's earliest affected index is `< covers_count` (it
        touches something the summary claims to cover) -> invalidate.
      - The mutation's earliest affected index is `>= covers_count` (it
        only touches recent, never-summarized entries) -> retain, the
        summary's own claim is untouched.
    """
    if not has_valid_summary:
        return False
    return earliest_affected_index < covers_count


def cleared_chat_summary_fields() -> dict[str, Any]:
    """The reset values for the three persisted summary fields on
    `PaperPoolSession`, as a plain dict -- NOT a mutation itself (this
    module has no `PaperPoolSession` dependency and performs no side
    effects). A future caller (M3.2) applies these directly, e.g.
    `for k, v in cleared_chat_summary_fields().items(): setattr(session, k, v)`."""
    return {
        "chat_summary": None,
        "chat_summary_covers_history_count": 0,
        "chat_summary_updated_at": None,
    }
