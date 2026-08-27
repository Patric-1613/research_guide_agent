"""Research Lanes (RL1): backward-compatible domain contracts + persistence
normalization for optional multi-query curation.

RL1 is STRUCTURE ONLY. Nothing here -- or anywhere else in RL1 -- generates
lanes, runs a search, ranks, interleaves, wires an API route, or touches the
frontend; those are RL2-RL5. This module exists so the session data model,
its serialization, and the per-turn provenance shape are frozen and tested
before any behavior is built on them.

Frozen v1 decisions encoded here:
  - single search stays the default; Research Lanes is opt-in (RL2+ gates
    lane creation/suggestion on config.Settings.research_lanes_enabled --
    that flag exists as of RL1 but is NOT wired to any behavior yet);
  - 3 suggested lanes, HARD MAXIMUM 4 (``MAX_LANES_PER_REVIEW``);
  - exactly ONE query per lane (a lane has one ``query`` string, not a list);
  - lane definitions FREEZE at curation Start -- no mid-curation editing,
    no per-lane refill;
  - old sessions load unchanged: a session with no lane keys deserializes
    to ``lanes=[]`` / ``paper_lane_ids={}`` / ``lane_result_counts={}``.

Two validation tiers, deliberately different -- the same split as
``config.get_auth_config``'s strict construction check vs.
``curation_session._dict_to_session``'s lenient ``.get()`` loads:

  1. CONSTRUCTION (RL2 lane generation, RL4 user edits) -- STRICT.
     ``ResearchLane.__post_init__`` enforces the structural invariants
     every real lane always satisfies; ``validate_lane_for_construction`` /
     ``validate_lane_list_for_construction`` add the input-shaped limits
     (bounded lengths, opaque non-label lane_id, hard lane count). All
     raise ``ValueError`` on violation.

  2. DESERIALIZATION (``curation_session._dict_to_session``) -- LENIENT,
     never raises. ``research_lanes_from_persisted`` /
     ``normalize_paper_lane_ids`` / ``normalize_lane_result_counts`` coerce,
     drop dangling/duplicate references deterministically, preserve order,
     and log -- so a malformed or stale persisted lane blob can never make
     an already-persisted session unloadable (frozen v1 requirement).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass
from typing import Iterable

from research_agent.config import get_usage_policy

logger = logging.getLogger(__name__)

# Frozen v1 ceiling -- 3 suggested, hard max 4 (see module docstring and the
# RL planning report's cost envelope: 4 lanes x 2 providers + capped shared
# title widening fits inside UsagePolicy.provider_fan_out_limit with margin).
MAX_LANES_PER_REVIEW = 4
DEFAULT_SUGGESTED_LANE_COUNT = 3  # informational; RL2 owns generation

# Bounded lengths -- generous caps a real lane never approaches. They exist
# to reject obviously-wrong CONSTRUCTION input (RL2 LLM output, RL4 user
# edits), never to re-constrain already-persisted data: the load path
# (research_lanes_from_persisted) deliberately does not check these, so a
# historical lane can always be rebuilt without a length policy in scope.
# label/question are short display strings; `query` reuses the same
# user-text bound as CurationPicksRequest.refinement
# (UsagePolicy.max_text_length), read at call time (uncached, same
# convention as every other config read in this codebase).
LANE_LABEL_MAX_LENGTH = 80
LANE_QUESTION_MAX_LENGTH = 300

LANE_ORIGINS = ("suggested", "user", "implicit")

# The optional key RL3/RL4 will add to each NEW turn_history entry of a
# lane-enabled session (see build_turn_paper_lane_ids). Frozen here so the
# reader/writer helpers and the serialization tests agree on the spelling.
TURN_PAPER_LANE_IDS_KEY = "paper_lane_ids"


def new_lane_id() -> str:
    """A stable, opaque lane identifier -- uuid4 hex, the same scheme
    report.py already uses for report version_ids. Deliberately NOT derived
    from the label (a user-controlled string): a label can be renamed, is
    not unique, and must never be an identity key."""
    return uuid.uuid4().hex


@dataclass
class ResearchLane:
    """One retrieval facet of a review topic: one label, one purpose
    question, exactly one search query. See the module docstring for the
    frozen v1 shape.

    ``__post_init__`` normalizes (strips surrounding whitespace, coerces
    ``enabled`` to bool) and enforces the invariants EVERY real lane
    satisfies regardless of origin: non-empty ``lane_id``/``label``/
    ``query``, a known ``origin``, ``generation_version >= 1``. It does
    NOT check bounded lengths or lane count -- those are construction-input
    concerns (``validate_lane_for_construction``), not model invariants,
    so the lenient load path can rebuild any historical lane without a
    length policy in scope.
    """

    lane_id: str
    label: str
    question: str
    query: str
    enabled: bool = True
    origin: str = "suggested"
    generation_version: int = 1

    def __post_init__(self) -> None:
        self.lane_id = str(self.lane_id).strip()
        self.label = str(self.label).strip()
        self.question = str(self.question).strip()
        self.query = str(self.query).strip()
        self.enabled = bool(self.enabled)
        if not self.lane_id:
            raise ValueError("ResearchLane.lane_id must be a non-empty string")
        if not self.label:
            raise ValueError("ResearchLane.label must be a non-empty string")
        if not self.query:
            raise ValueError("ResearchLane.query must be a non-empty string")
        if self.origin not in LANE_ORIGINS:
            raise ValueError(f"ResearchLane.origin must be one of {LANE_ORIGINS}, got {self.origin!r}")
        # isinstance(True, int) is True in Python -- guard bool explicitly so
        # generation_version=True never silently reads as 1.
        if isinstance(self.generation_version, bool) or not isinstance(self.generation_version, int):
            raise ValueError("ResearchLane.generation_version must be an int")
        if self.generation_version < 1:
            raise ValueError("ResearchLane.generation_version must be a positive integer (>= 1)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchLane":
        """Reconstruct from a persisted dict. Ignores unknown keys (so a
        FUTURE added field never breaks an OLD reader) and fills missing
        optional keys with the dataclass defaults. Propagates
        ValueError/TypeError from ``__post_init__`` when the structural
        invariants aren't met -- ``research_lanes_from_persisted`` is what
        catches that on the load path."""
        if not isinstance(d, dict):
            raise TypeError(f"ResearchLane.from_dict expects a dict, got {type(d).__name__}")
        return cls(
            lane_id=d.get("lane_id", ""),
            label=d.get("label", ""),
            question=d.get("question", ""),
            query=d.get("query", ""),
            enabled=d.get("enabled", True),
            origin=d.get("origin", "suggested"),
            generation_version=d.get("generation_version", 1),
        )


# --- Construction-time validation (RL2 suggestion output, RL4 user edits) ---

def validate_lane_for_construction(lane: ResearchLane) -> None:
    """Strict input validation for a lane being CREATED -- raises
    ``ValueError`` on any violation. On top of ``ResearchLane``'s own
    structural invariants: bounded lengths and an opaque (not
    label-derived) ``lane_id``. NOT called on the deserialization path."""
    if len(lane.label) > LANE_LABEL_MAX_LENGTH:
        raise ValueError(f"lane label exceeds {LANE_LABEL_MAX_LENGTH} characters")
    if len(lane.question) > LANE_QUESTION_MAX_LENGTH:
        raise ValueError(f"lane question exceeds {LANE_QUESTION_MAX_LENGTH} characters")
    max_query = get_usage_policy().max_text_length
    if len(lane.query) > max_query:
        raise ValueError(f"lane query exceeds {max_query} characters")
    if lane.lane_id.strip().lower() == lane.label.strip().lower():
        raise ValueError("lane_id must not be derived from the user-controlled label")
    if any(ch.isspace() for ch in lane.lane_id):
        raise ValueError("lane_id must be an opaque token (no whitespace)")


def validate_lane_list_for_construction(lanes: list[ResearchLane]) -> None:
    """Strict validation for a full lane set at curation Start (RL4 will
    call this). Enforces the hard maximum, at least one ENABLED lane,
    unique ``lane_id``s, and each lane individually. Raises ``ValueError``."""
    if len(lanes) > MAX_LANES_PER_REVIEW:
        raise ValueError(f"a review may have at most {MAX_LANES_PER_REVIEW} research lanes, got {len(lanes)}")
    if not any(lane.enabled for lane in lanes):
        raise ValueError("at least one research lane must be enabled")
    seen: set[str] = set()
    for lane in lanes:
        if lane.lane_id in seen:
            raise ValueError(f"duplicate lane_id: {lane.lane_id!r}")
        seen.add(lane.lane_id)
        validate_lane_for_construction(lane)


# --- Lenient deserialization (curation_session._dict_to_session) ------------

def research_lanes_from_persisted(raw: object) -> list[ResearchLane]:
    """Lenient deserialization for ``session.lanes``. NEVER raises.

    - a non-list (the caller passes ``[]`` for a missing key) -> ``[]``;
    - each entry that can't be structurally rebuilt into a ``ResearchLane``
      (not a dict, empty lane_id/label/query, unknown origin, bad
      generation_version) is DROPPED and logged -- a stale/corrupt lane
      entry must never make the whole session unloadable;
    - duplicate ``lane_id``s: FIRST occurrence wins, later ones dropped
      (deterministic);
    - input order is preserved;
    - lane COUNT is not enforced here (Start already caps it at
      ``MAX_LANES_PER_REVIEW``); a persisted set somehow larger is logged,
      never truncated (truncating would silently drop a real facet).
    """
    if not isinstance(raw, list):
        if raw:
            logger.warning("research_lanes_from_persisted: expected a list, got %s -- ignoring", type(raw).__name__)
        return []
    lanes: list[ResearchLane] = []
    seen: set[str] = set()
    for entry in raw:
        try:
            lane = ResearchLane.from_dict(entry)
        except (ValueError, TypeError) as exc:
            logger.warning("research_lanes_from_persisted: dropping malformed lane entry (%s)", exc)
            continue
        if lane.lane_id in seen:
            logger.warning("research_lanes_from_persisted: dropping duplicate lane_id %r", lane.lane_id)
            continue
        seen.add(lane.lane_id)
        lanes.append(lane)
    if len(lanes) > MAX_LANES_PER_REVIEW:
        logger.warning(
            "research_lanes_from_persisted: session has %d lanes (> hard max %d) -- keeping all, not truncating",
            len(lanes), MAX_LANES_PER_REVIEW,
        )
    return lanes


def normalize_paper_lane_ids(raw: object, *, allowed_lane_ids: Iterable[str] | None) -> dict[str, list[str]]:
    """Lenient deserialization for ``session.paper_lane_ids`` -- cumulative
    discovery provenance, ``paper_id -> [lane_id, ...]``. NEVER raises.

    - a non-dict -> ``{}``;
    - a non-str key or non-list value -> that entry dropped;
    - within a paper's list: non-str entries dropped; DUPLICATES collapsed
      keeping first-seen order (deterministic);
    - DANGLING refs (a ``lane_id`` not in ``allowed_lane_ids``) dropped --
      pass ``allowed_lane_ids=None`` to skip that filter (structural
      cleanup only);
    - a paper whose list becomes empty after cleanup is dropped entirely
      (same "missing key == no provenance recorded" soft-link posture as
      ``PaperPoolSession.web_article_provenance_by_url``);
    - key order is preserved.
    """
    if not isinstance(raw, dict):
        if raw:
            logger.warning("normalize_paper_lane_ids: expected a dict, got %s -- ignoring", type(raw).__name__)
        return {}
    allowed = set(allowed_lane_ids) if allowed_lane_ids is not None else None
    out: dict[str, list[str]] = {}
    for paper_id, lane_ids in raw.items():
        if not isinstance(paper_id, str) or not isinstance(lane_ids, list):
            logger.warning("normalize_paper_lane_ids: dropping malformed entry for key %r", paper_id)
            continue
        cleaned: list[str] = []
        seen: set[str] = set()
        for lid in lane_ids:
            if not isinstance(lid, str) or lid in seen:
                continue
            if allowed is not None and lid not in allowed:
                logger.warning("normalize_paper_lane_ids: dropping dangling lane_id %r for paper %r", lid, paper_id)
                continue
            seen.add(lid)
            cleaned.append(lid)
        if cleaned:
            out[paper_id] = cleaned
    return out


def normalize_lane_result_counts(raw: object, *, allowed_lane_ids: Iterable[str] | None) -> dict[str, int]:
    """Lenient deserialization for ``session.lane_result_counts`` --
    ``lane_id -> distinct-papers-surfaced count`` (diagnostic / future-UI
    only). NEVER raises.

    - a non-dict -> ``{}``;
    - a non-str key, or a non-int / bool / negative value -> entry dropped;
    - a DANGLING ``lane_id`` (not in ``allowed_lane_ids``) dropped --
      ``None`` skips that filter;
    - key order is preserved.
    """
    if not isinstance(raw, dict):
        if raw:
            logger.warning("normalize_lane_result_counts: expected a dict, got %s -- ignoring", type(raw).__name__)
        return {}
    allowed = set(allowed_lane_ids) if allowed_lane_ids is not None else None
    out: dict[str, int] = {}
    for lane_id, count in raw.items():
        if not isinstance(lane_id, str):
            continue
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            logger.warning("normalize_lane_result_counts: dropping non-count value for %r", lane_id)
            continue
        if allowed is not None and lane_id not in allowed:
            logger.warning("normalize_lane_result_counts: dropping dangling lane_id %r", lane_id)
            continue
        out[lane_id] = count
    return out


# --- Per-turn discovery-provenance snapshot (frozen for RL3/RL4) ------------

def build_turn_paper_lane_ids(
    batch_paper_ids: Iterable[str],
    session_paper_lane_ids: dict[str, list[str]],
) -> dict[str, list[str]]:
    """The per-turn provenance snapshot RL3/RL4 will attach -- as an
    OPTIONAL ``"paper_lane_ids"`` key -- to each NEW ``turn_history`` entry
    of a lane-enabled session. It is a projection of the cumulative
    ``session.paper_lane_ids`` restricted to exactly this turn's batch.

    RL1 freezes the shape and this helper; it does NOT change turn creation
    (RL3/RL4). ``turn_history`` entries are already opaquely round-tripped
    by ``curation_session._session_to_dict`` and a ``dict[str, list[str]]``
    is JSON-native, so NO serialization code change is needed for the new
    key -- proven by ``tests/test_curation_session.py``.

    Deterministic ordering: keys in ``batch_paper_ids`` order (deduped);
    each value copied in ``session.paper_lane_ids``'s own stored order.
    Returns ``{}`` when no batch paper has recorded provenance -- so the
    caller may omit the key entirely for that turn, and an old entry
    without it stays valid.
    """
    out: dict[str, list[str]] = {}
    seen: set[str] = set()
    for pid in batch_paper_ids:
        if pid in seen:
            continue
        seen.add(pid)
        lane_ids = session_paper_lane_ids.get(pid)
        if lane_ids:
            out[pid] = list(lane_ids)
    return out


def read_turn_paper_lane_ids(
    turn_entry: dict, *, allowed_lane_ids: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """Read the optional per-turn provenance snapshot from a
    ``turn_history`` entry. An entry that predates this field (no
    ``"paper_lane_ids"`` key) yields ``{}`` -- the same ``.get()``
    backward-compat convention as every other optional session/turn field.
    Structurally normalized (dedup, order preserved) via
    ``normalize_paper_lane_ids``; pass ``allowed_lane_ids`` to also drop
    dangling refs."""
    if not isinstance(turn_entry, dict):
        return {}
    return normalize_paper_lane_ids(
        turn_entry.get(TURN_PAPER_LANE_IDS_KEY, {}), allowed_lane_ids=allowed_lane_ids,
    )
