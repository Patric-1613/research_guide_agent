"""Research Lanes (RL1): unit tests for the domain contracts and the
lenient persistence-normalization helpers in
research_agent/research_lanes.py.

Pure Python -- no HTTP, no network, no provider calls, no telemetry DB,
no checkpoint DB. The real-SQLite round-trip and DB-fingerprint proofs
live in tests/test_curation_session.py alongside the existing session
serialization tests.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from research_agent.config import get_usage_policy
from research_agent.research_lanes import (
    DEFAULT_SUGGESTED_LANE_COUNT,
    LANE_LABEL_MAX_LENGTH,
    LANE_ORIGINS,
    LANE_QUESTION_MAX_LENGTH,
    MAX_LANES_PER_REVIEW,
    ResearchLane,
    build_turn_paper_lane_ids,
    new_lane_id,
    normalize_lane_result_counts,
    normalize_paper_lane_ids,
    read_turn_paper_lane_ids,
    research_lanes_from_persisted,
    validate_lane_for_construction,
    validate_lane_list_for_construction,
)


def _lane(lane_id: str = "l1", label: str = "Evaluation methods", query: str = "rag evaluation methods", **kw) -> ResearchLane:
    return ResearchLane(
        lane_id=lane_id, label=label, question=kw.pop("question", "how are these systems evaluated?"),
        query=query, **kw,
    )


# --- frozen v1 constants -------------------------------------------------

def test_frozen_v1_lane_ceilings():
    assert DEFAULT_SUGGESTED_LANE_COUNT == 3
    assert MAX_LANES_PER_REVIEW == 4
    assert LANE_ORIGINS == ("suggested", "user", "implicit")


# --- ResearchLane model + structural invariants ------------------------

def test_research_lane_happy_path_and_to_from_dict_round_trip():
    lane = _lane(enabled=False, origin="user", generation_version=3)
    d = lane.to_dict()
    assert d == {
        "lane_id": "l1", "label": "Evaluation methods", "question": "how are these systems evaluated?",
        "query": "rag evaluation methods", "enabled": False, "origin": "user", "generation_version": 3,
    }
    assert ResearchLane.from_dict(d) == lane


def test_research_lane_strips_whitespace_and_coerces_enabled():
    lane = ResearchLane(lane_id="  l1  ", label="  Datasets  ", question="  q  ", query="  ds  ", enabled=1)
    assert lane.lane_id == "l1"
    assert lane.label == "Datasets"
    assert lane.question == "q"
    assert lane.query == "ds"
    assert lane.enabled is True


def test_research_lane_from_dict_fills_optional_defaults_and_ignores_unknown_keys():
    lane = ResearchLane.from_dict({
        "lane_id": "l1", "label": "L", "query": "q", "some_future_field": "ignored",
    })
    assert lane.question == ""
    assert lane.enabled is True
    assert lane.origin == "suggested"
    assert lane.generation_version == 1


@pytest.mark.parametrize("kwargs", [
    {"lane_id": ""},
    {"lane_id": "   "},
    {"label": ""},
    {"query": ""},
    {"query": "   "},
    {"origin": "viewpoint"},
    {"generation_version": 0},
    {"generation_version": -1},
    {"generation_version": True},
    {"generation_version": "2"},
    {"generation_version": 1.0},
])
def test_research_lane_rejects_structural_violations(kwargs):
    base = {"lane_id": "l1", "label": "L", "question": "q", "query": "qq"}
    base.update(kwargs)
    with pytest.raises(ValueError):
        ResearchLane(**base)


def test_from_dict_rejects_non_dict():
    with pytest.raises(TypeError):
        ResearchLane.from_dict(["not", "a", "dict"])


# --- new_lane_id ------------------------------------------------------

def test_new_lane_id_is_opaque_and_unique():
    ids = {new_lane_id() for _ in range(50)}
    assert len(ids) == 50
    for lid in ids:
        assert lid and not any(ch.isspace() for ch in lid)


# --- construction-time validation (RL2/RL4 entry points) ---------------

def test_validate_lane_for_construction_accepts_a_normal_lane():
    validate_lane_for_construction(_lane(lane_id=new_lane_id()))


def test_validate_lane_for_construction_rejects_over_length_label():
    with pytest.raises(ValueError, match="label"):
        validate_lane_for_construction(_lane(label="x" * (LANE_LABEL_MAX_LENGTH + 1)))


def test_validate_lane_for_construction_rejects_over_length_question():
    lane = ResearchLane(lane_id="l1", label="L", question="x" * (LANE_QUESTION_MAX_LENGTH + 1), query="q")
    with pytest.raises(ValueError, match="question"):
        validate_lane_for_construction(lane)


def test_validate_lane_for_construction_rejects_over_length_query():
    over = "x" * (get_usage_policy().max_text_length + 1)
    with pytest.raises(ValueError, match="query"):
        validate_lane_for_construction(_lane(query=over))


def test_validate_lane_for_construction_rejects_label_used_as_lane_id():
    with pytest.raises(ValueError, match="lane_id"):
        validate_lane_for_construction(_lane(lane_id="Evaluation methods", label="Evaluation methods"))
    # case-insensitive / whitespace-insensitive too
    with pytest.raises(ValueError, match="lane_id"):
        validate_lane_for_construction(_lane(lane_id="evaluation methods", label="Evaluation Methods"))


def test_validate_lane_list_rejects_more_than_the_hard_maximum():
    lanes = [_lane(lane_id=new_lane_id(), label=f"L{i}") for i in range(MAX_LANES_PER_REVIEW + 1)]
    with pytest.raises(ValueError, match="at most"):
        validate_lane_list_for_construction(lanes)


def test_validate_lane_list_accepts_exactly_the_hard_maximum():
    lanes = [_lane(lane_id=new_lane_id(), label=f"L{i}") for i in range(MAX_LANES_PER_REVIEW)]
    validate_lane_list_for_construction(lanes)


def test_validate_lane_list_requires_at_least_one_enabled_lane():
    lanes = [_lane(lane_id=new_lane_id(), label=f"L{i}", enabled=False) for i in range(2)]
    with pytest.raises(ValueError, match="enabled"):
        validate_lane_list_for_construction(lanes)


def test_validate_lane_list_rejects_duplicate_lane_ids():
    lanes = [_lane(lane_id="dup", label="A"), _lane(lane_id="dup", label="B")]
    with pytest.raises(ValueError, match="duplicate"):
        validate_lane_list_for_construction(lanes)


# --- research_lanes_from_persisted (lenient, never raises) -------------

def test_from_persisted_missing_or_empty_yields_empty_list():
    assert research_lanes_from_persisted([]) == []
    assert research_lanes_from_persisted(None) == []
    assert research_lanes_from_persisted("nonsense") == []
    assert research_lanes_from_persisted({"not": "a list"}) == []


def test_from_persisted_round_trips_valid_lanes_in_order():
    raw = [
        _lane(lane_id="a", label="Alpha").to_dict(),
        _lane(lane_id="b", label="Beta", enabled=False, origin="user").to_dict(),
    ]
    lanes = research_lanes_from_persisted(raw)
    assert [l.lane_id for l in lanes] == ["a", "b"]
    assert lanes[1].enabled is False
    assert lanes[1].origin == "user"


def test_from_persisted_drops_malformed_entries_without_raising():
    raw = [
        {"lane_id": "ok", "label": "Good", "query": "q"},
        {"lane_id": "", "label": "no id", "query": "q"},       # empty lane_id
        {"label": "missing id/query"},                          # missing keys
        "not-a-dict",
        {"lane_id": "bad-origin", "label": "L", "query": "q", "origin": "date_range"},
        {"lane_id": "bad-ver", "label": "L", "query": "q", "generation_version": 0},
    ]
    lanes = research_lanes_from_persisted(raw)
    assert [l.lane_id for l in lanes] == ["ok"]


def test_from_persisted_collapses_duplicate_lane_ids_first_wins_deterministically():
    raw = [
        _lane(lane_id="dup", label="First", query="q1").to_dict(),
        _lane(lane_id="dup", label="Second", query="q2").to_dict(),
        _lane(lane_id="other", label="Other").to_dict(),
    ]
    lanes = research_lanes_from_persisted(raw)
    assert [l.lane_id for l in lanes] == ["dup", "other"]
    assert lanes[0].label == "First"  # first occurrence kept


def test_from_persisted_keeps_all_lanes_even_above_hard_max_without_truncating():
    raw = [_lane(lane_id=f"l{i}", label=f"L{i}").to_dict() for i in range(MAX_LANES_PER_REVIEW + 2)]
    lanes = research_lanes_from_persisted(raw)
    assert len(lanes) == MAX_LANES_PER_REVIEW + 2  # logged, not silently dropped


# --- normalize_paper_lane_ids ----------------------------------------

def test_normalize_paper_lane_ids_non_dict_yields_empty():
    assert normalize_paper_lane_ids(None, allowed_lane_ids=None) == {}
    assert normalize_paper_lane_ids(["x"], allowed_lane_ids=None) == {}


def test_normalize_paper_lane_ids_preserves_key_and_value_order():
    raw = {"p2": ["b", "a"], "p1": ["a"]}
    assert list(normalize_paper_lane_ids(raw, allowed_lane_ids=None).items()) == [("p2", ["b", "a"]), ("p1", ["a"])]


def test_normalize_paper_lane_ids_dedupes_within_a_paper_first_seen_order():
    out = normalize_paper_lane_ids({"p1": ["a", "b", "a", "b", "c"]}, allowed_lane_ids=None)
    assert out == {"p1": ["a", "b", "c"]}


def test_normalize_paper_lane_ids_drops_dangling_refs_when_allowed_set_given():
    out = normalize_paper_lane_ids({"p1": ["a", "ghost", "b"], "p2": ["ghost"]}, allowed_lane_ids={"a", "b"})
    assert out == {"p1": ["a", "b"]}  # p2 dropped entirely -- became empty


def test_normalize_paper_lane_ids_drops_malformed_entries():
    raw = {"p1": ["a"], 5: ["a"], "p2": "not-a-list", "p3": [1, 2, "b"]}
    out = normalize_paper_lane_ids(raw, allowed_lane_ids=None)
    assert out == {"p1": ["a"], "p3": ["b"]}


# --- normalize_lane_result_counts ----------------------------------

def test_normalize_lane_result_counts_coerces_and_filters():
    raw = {"a": 5, "b": -1, "c": "3", "d": True, "e": 0, 9: 1, "ghost": 2}
    out = normalize_lane_result_counts(raw, allowed_lane_ids={"a", "b", "c", "d", "e"})
    assert out == {"a": 5, "e": 0}  # b negative, c str, d bool, 9 non-str key, ghost dangling


def test_normalize_lane_result_counts_non_dict_yields_empty():
    assert normalize_lane_result_counts("x", allowed_lane_ids=None) == {}


# --- per-turn provenance snapshot helpers (frozen for RL3/RL4) -------

def test_build_turn_paper_lane_ids_projects_cumulative_map_onto_the_batch():
    session_map = {"p1": ["a", "b"], "p2": ["b"], "p9": ["c"]}
    snap = build_turn_paper_lane_ids(["p2", "p1", "p3"], session_map)
    # p3 has no provenance -> omitted; p9 not in this batch -> omitted;
    # key order follows batch order.
    assert list(snap.items()) == [("p2", ["b"]), ("p1", ["a", "b"])]


def test_build_turn_paper_lane_ids_dedupes_batch_ids_and_copies_values():
    session_map = {"p1": ["a"]}
    snap = build_turn_paper_lane_ids(["p1", "p1"], session_map)
    assert snap == {"p1": ["a"]}
    snap["p1"].append("mutated")
    assert session_map["p1"] == ["a"]  # value was copied, not aliased


def test_build_turn_paper_lane_ids_returns_empty_when_no_batch_paper_has_provenance():
    assert build_turn_paper_lane_ids(["p1", "p2"], {"p9": ["a"]}) == {}


def test_read_turn_paper_lane_ids_old_entry_without_the_key_yields_empty():
    assert read_turn_paper_lane_ids({"turn_number": 1, "batch": [], "refilled": False}) == {}
    assert read_turn_paper_lane_ids("not-a-dict") == {}


def test_read_turn_paper_lane_ids_normalizes_and_can_drop_dangling():
    entry = {"turn_number": 3, "paper_lane_ids": {"p1": ["a", "a", "ghost"]}}
    assert read_turn_paper_lane_ids(entry) == {"p1": ["a", "ghost"]}
    assert read_turn_paper_lane_ids(entry, allowed_lane_ids={"a"}) == {"p1": ["a"]}


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call(["pytest", "-q", __file__]))
