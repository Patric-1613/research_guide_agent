"""Tests for curation_loop.py (curation-interrupt-loop Phase 3): the
interactive present/interrupt/resume curation loop. Baseline
single-process correctness here; the harder cross-process resume proof
lives in scripts run for Phase 3d (see that phase's own driver scripts),
since pytest itself runs everything in one process by construction.

Usage Protection M2.2B: curation_loop.py's own _refill_node now opens
research_agent.usage_guard.guard_paid_action on a real refill turn, so
every test in this file -- not just the ones added for M2.2B -- gets an
autouse fixture redirecting telemetry/admission/leases USAGE_DB_PATH to
a fresh tmp_path file. Without it, any pre-existing test that happens
to trigger a real refill would read and write the real
data/usage_telemetry.sqlite.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import research_agent.admission as admission
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.curation_loop import (
    get_curation_state,
    resume_curation_turn,
    start_curation_turn,
)
from research_agent.curation_session import _session_to_dict
from research_agent.qa import sqlite_checkpointer
from research_agent.query_expansion import PaperPoolSession
from research_agent.schema import Paper
from research_agent.telemetry import init_usage_db
from research_agent.usage_guard import UsageGuardRejection
from tests._usage_db_fingerprint import fingerprint_usage_db

_REAL_USAGE_DB_PATH = telemetry.USAGE_DB_PATH
_REAL_USAGE_DB_FINGERPRINT_BEFORE = fingerprint_usage_db(_REAL_USAGE_DB_PATH)


@pytest.fixture(autouse=True)
def usage_db_path(tmp_path, monkeypatch):
    db_path = tmp_path / "usage.sqlite"
    monkeypatch.setattr(telemetry, "USAGE_DB_PATH", db_path)
    monkeypatch.setattr(admission, "USAGE_DB_PATH", db_path)
    monkeypatch.setattr(leases, "USAGE_DB_PATH", db_path)
    init_usage_db(path=db_path).close()
    return db_path


def _paper(pid: str) -> Paper:
    return Paper(
        title=f"Paper {pid}", authors=["A"], year=2024, venue="X",
        abstract=f"abstract {pid}", url=None, doi=None, citation_count=None,
        source="arxiv", paper_id=pid,
    )


def _session(n: int, target_count: int) -> PaperPoolSession:
    return PaperPoolSession(
        topic="q",
        reserve=[(_paper(f"p{i}"), 1.0 - i * 0.01) for i in range(n)],
        target_count=target_count,
    )


def test_reaching_target_count_no_longer_stops_curation():
    """curation-editable-until-locked Phase 10b: hitting target_count is
    no longer a hard stop -- only an explicit stop=True ends the loop.
    The user stays free to keep picking/searching past their original
    target."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(15, target_count=5)))
            assert "__interrupt__" in result
            batch = result["__interrupt__"][0].value["batch"]
            assert len(batch) == 10

            picks = [p[0]["paper_id"] for p in batch[:5]]
            result2 = resume_curation_turn("s1", cp, picked_paper_ids=picks)

        assert result2["stop_reason"] is None
        assert result2["session"]["selected_paper_ids"] == picks
        assert result2["session"]["stage"] == "curate"
        assert "__interrupt__" in result2, "must keep curating past target, not auto-stop"
        assert len(result2["__interrupt__"][0].value["batch"]) == 5  # 15 - 10 served turn 1


def test_multi_turn_loop_continues_past_target_met_since_it_no_longer_stops():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(25, target_count=12)))
            batch1 = result["__interrupt__"][0].value["batch"]
            picks1 = [p[0]["paper_id"] for p in batch1[:6]]  # 6 of 10 -> 6/12, not met

            result = resume_curation_turn("s1", cp, picked_paper_ids=picks1)
            assert "__interrupt__" in result, "must loop back for turn 2, not stop"
            batch2 = result["__interrupt__"][0].value["batch"]
            # turn 2's batch must be disjoint from turn 1's (cursor advanced, not re-served)
            assert set(p[0]["paper_id"] for p in batch2).isdisjoint(picks1)
            picks2 = [p[0]["paper_id"] for p in batch2[:6]]  # 6 more -> 12/12, met exactly on a batch boundary

            result = resume_curation_turn("s1", cp, picked_paper_ids=picks2)

        assert result["stop_reason"] is None
        assert result["session"]["stage"] == "curate"
        assert set(result["session"]["selected_paper_ids"]) == set(picks1) | set(picks2)
        assert len(result["session"]["selected_paper_ids"]) == 12
        assert "__interrupt__" in result, "target_met must not stop curation"


def test_picks_not_in_presented_batch_are_silently_rejected():
    """The user's explicit addition: validate against the ACTUAL batch,
    ignore anything else, rather than trusting the resume payload."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(15, target_count=1)))
            batch = result["__interrupt__"][0].value["batch"]
            real_id = batch[0][0]["paper_id"]

            # A mix of one real pick, one paper that exists elsewhere in the
            # reserve but was NOT in this batch, and one that doesn't exist at all.
            not_in_batch_but_in_reserve = "p12"  # reserve has p0..p14, batch is only the first 10
            fabricated = "does-not-exist-anywhere"
            result2 = resume_curation_turn(
                "s1", cp, picked_paper_ids=[real_id, not_in_batch_but_in_reserve, fabricated], stop=True,
            )

        assert result2["stop_reason"] == "user_stopped"
        assert result2["session"]["selected_paper_ids"] == [real_id]


def test_double_mutation_across_interrupt_resume_boundary():
    """Explicit, instrumented, empirical proof (not just the structural
    "nothing runs before interrupt()" argument) that serve_next_batch's
    state mutation (cursor advance, seen-set additions) happens exactly
    ONCE per turn, not once on the halting call and again on resume."""
    from research_agent import curation_loop as loop_module

    call_count = {"n": 0}
    real_serve_next_batch = loop_module.serve_next_batch

    def _counting_serve_next_batch(*args, **kwargs):
        call_count["n"] += 1
        return real_serve_next_batch(*args, **kwargs)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch.object(loop_module, "serve_next_batch", side_effect=_counting_serve_next_batch), \
             sqlite_checkpointer(db_path) as cp:
            # curation-editable-until-locked Phase 10b: target_count no
            # longer stops the graph on its own, so an explicit stop=True
            # below is what keeps this test isolated to exactly the
            # interrupt/resume boundary it's meant to check -- otherwise
            # the resume would loop back and genuinely serve a second
            # batch (a real, separate call to serve_next_batch), which is
            # not what this test is about.
            result = start_curation_turn("s1", cp, _session_to_dict(_session(15, target_count=3)))
            assert call_count["n"] == 1, "serve_next_batch must run exactly once on the halting call"

            batch = result["__interrupt__"][0].value["batch"]
            cursor_after_halt = result["session"]["cursor"]

            picks = [p[0]["paper_id"] for p in batch[:3]]
            result2 = resume_curation_turn("s1", cp, picked_paper_ids=picks, stop=True)

        # The real assertion: resuming present_and_apply must NOT have
        # re-invoked serve_next_batch (that node isn't the one re-executed
        # on resume) — still exactly 1 call across the whole turn.
        assert call_count["n"] == 1, "serve_next_batch must NOT be called again on resume"
        assert result2["session"]["cursor"] == cursor_after_halt, "cursor must not have advanced a second time"
        assert result2["session"]["cursor"] == 10  # one batch of 10 served, exactly once


# --- Phase 3c: adversarial loop cases ---

def test_user_picks_zero_still_progresses_to_a_new_batch_not_a_stall():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(25, target_count=5)))
            batch1 = result["__interrupt__"][0].value["batch"]
            batch1_ids = {p[0]["paper_id"] for p in batch1}

            result = resume_curation_turn("s1", cp, picked_paper_ids=[])  # picks nothing

        # The resume call keeps running through serve_batch for the NEXT
        # turn until it hits another interrupt, all within this one call
        # (same behavior already confirmed in the multi-turn test above) —
        # so by the time this returns, turn 2's batch has already been
        # served too (cursor at 20, not 10).
        assert "__interrupt__" in result, "must loop to a new batch, not stall"
        assert result["session"]["selected_paper_ids"] == []
        assert result["session"]["cursor"] == 20  # progressed past BOTH the unpicked and the new batch
        batch2_ids = {p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]}
        assert batch2_ids.isdisjoint(batch1_ids), "turn 2's batch must be genuinely new, not a repeat"


def test_user_picks_all_ten_from_a_batch():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(25, target_count=10)))
            batch = result["__interrupt__"][0].value["batch"]
            all_ids = [p[0]["paper_id"] for p in batch]

            result = resume_curation_turn("s1", cp, picked_paper_ids=all_ids)

        # target_count=10 hit exactly, but that no longer stops curation
        # (curation-editable-until-locked Phase 10b) -- 15 papers remain.
        assert result["stop_reason"] is None
        assert result["session"]["stage"] == "curate"
        assert result["session"]["selected_paper_ids"] == all_ids
        assert "__interrupt__" in result


def test_user_stops_before_hitting_target():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(25, target_count=10)))
            batch = result["__interrupt__"][0].value["batch"]
            picks = [batch[0][0]["paper_id"], batch[1][0]["paper_id"]]  # only 2 of the target 10

            result = resume_curation_turn("s1", cp, picked_paper_ids=picks, stop=True)

        assert result["stop_reason"] == "user_stopped"
        assert result["session"]["selected_paper_ids"] == picks
        assert len(result["session"]["selected_paper_ids"]) < 10
        assert result["session"]["stage"] == "synthesize"


def test_target_hit_exactly_on_a_batch_boundary():
    """target_count=20: turn 1 picks a full batch of 10 (10/20), turn 2
    picks the ENTIRE second batch of 10 (20/20). curation-editable-
    until-locked Phase 10b: hitting target exactly on a batch boundary
    must still not stop curation -- 5 papers remain in the reserve."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(25, target_count=20)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]

            result = resume_curation_turn("s1", cp, picked_paper_ids=batch1_ids)
            assert "__interrupt__" in result  # 10/20, must continue

            batch2_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            result = resume_curation_turn("s1", cp, picked_paper_ids=batch2_ids)

        assert result["stop_reason"] is None
        assert result["session"]["stage"] == "curate"
        assert len(result["session"]["selected_paper_ids"]) == 20
        assert "__interrupt__" in result, "target_met must not stop curation"


def test_target_hit_mid_batch():
    """target_count=13: turn 1 picks a full batch of 10 (10/13), turn 2
    picks only 3 of the 10 presented -- target is met mid-batch, with 7
    of turn 2's batch never picked. curation-editable-until-locked
    Phase 10b: this must not stop curation either."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(25, target_count=13)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]

            result = resume_curation_turn("s1", cp, picked_paper_ids=batch1_ids)
            assert "__interrupt__" in result  # 10/13, must continue

            batch2 = result["__interrupt__"][0].value["batch"]
            picks2 = [p[0]["paper_id"] for p in batch2[:3]]  # only 3 of 10
            result = resume_curation_turn("s1", cp, picked_paper_ids=picks2)

        assert result["stop_reason"] is None
        assert result["session"]["stage"] == "curate"
        assert len(result["session"]["selected_paper_ids"]) == 13
        assert result["session"]["selected_paper_ids"] == batch1_ids + picks2
        assert "__interrupt__" in result, "target_met must not stop curation"


# --- curation-report-synthesis Phase 4: selected_papers/selected_paper_ids sync invariant ---

def _assert_selected_lists_in_sync(session_dict: dict) -> None:
    ids_from_selected_paper_ids = session_dict["selected_paper_ids"]
    ids_from_selected_papers = [p["paper_id"] for p in session_dict["selected_papers"]]
    assert ids_from_selected_papers == ids_from_selected_paper_ids, (
        f"selected_papers {ids_from_selected_papers} out of sync with "
        f"selected_paper_ids {ids_from_selected_paper_ids}"
    )


def test_selected_papers_and_selected_paper_ids_stay_in_sync_across_a_refill():
    """Not just trusting that populating both in the same node guarantees
    this — explicitly checked (same members, same order) at every step of
    a session that genuinely triggers a refill mid-way."""
    from research_agent import query_expansion as qe_module

    def _identity_rank(topic, papers, client=None, **kwargs):
        return [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}

    fresh_papers = [_paper(f"new{i}") for i in range(8)]

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch.object(qe_module, "build_candidate_pool", return_value=fresh_papers), \
             patch.object(qe_module, "rank_full_pool", side_effect=_identity_rank), \
             sqlite_checkpointer(db_path) as cp:

            # reserve=10 (exactly one batch): serving turn 1's entire
            # batch drains the reserve to remaining=0 regardless of how
            # many of it get picked -- curation-turn-history Phase 9d
            # narrowed the auto-refill trigger to true exhaustion
            # (remaining()==0), not just "< BATCH_SIZE" -- so this is now
            # what it takes to force a real refill without an explicit
            # request_refill. target_count=100 (unreachable) lets natural
            # exhaustion/continuation end the test rather than an
            # artificial target.
            result = start_curation_turn("s1", cp, _session_to_dict(_session(10, target_count=100)))
            _assert_selected_lists_in_sync(result["session"])
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            picks1 = batch1_ids[:5]

            # This resume triggers the refill internally (remaining=0)
            # before serving turn 2's batch -- confirmed by the mocked
            # build_candidate_pool having been called. refill_pool() itself
            # runs for real (only its own internals are mocked above), so
            # it still needs a client passed through config.
            result = resume_curation_turn("s1", cp, picked_paper_ids=picks1, config={"client": MagicMock()})
            qe_module.build_candidate_pool.assert_called()
            _assert_selected_lists_in_sync(result["session"])
            assert result["session"]["selected_paper_ids"] == picks1

            assert "__interrupt__" in result, "expected turn 2 to still be presenting, not stopped"
            batch2_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            picks2 = batch2_ids[:3]

            result = resume_curation_turn("s1", cp, picked_paper_ids=picks2, stop=True)
            _assert_selected_lists_in_sync(result["session"])

        assert result["stop_reason"] == "user_stopped"
        assert result["session"]["selected_paper_ids"] == picks1 + picks2


def test_get_curation_state_returns_none_for_a_never_started_session():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            assert get_curation_state("never-started", cp) is None


def test_get_curation_state_exposes_pending_batch_mid_interrupt():
    """curation-api-and-ui Phase 6a: the property GET /curation/{id} needs
    for a page refresh mid-curation -- the presented-but-not-yet-picked
    batch must be recoverable from the checkpointer alone, not from
    anything the caller still had in memory."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            start_curation_turn("s1", cp, _session_to_dict(_session(15, target_count=5)))
            state = get_curation_state("s1", cp)

        assert state is not None
        assert state["session"].stage == "curate"
        assert state["pending_batch"] is not None
        assert len(state["pending_batch"]) == 10


def test_get_curation_state_pending_batch_is_none_once_curation_finishes():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(15, target_count=5)))
            batch = result["__interrupt__"][0].value["batch"]
            picks = [p[0]["paper_id"] for p in batch[:5]]
            # explicit stop=True: target_count alone no longer finishes
            # curation (curation-editable-until-locked Phase 10b).
            resume_curation_turn("s1", cp, picked_paper_ids=picks, stop=True)
            state = get_curation_state("s1", cp)

        assert state["pending_batch"] is None
        assert state["session"].stage == "synthesize"
        assert state["session"].selected_paper_ids == picks


def test_refilled_is_false_on_the_first_turn_when_the_pool_is_already_large_enough():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(15, target_count=5)))
        assert result["refilled"] is False


def test_refilled_is_true_exactly_on_the_turn_that_triggers_a_real_refill_and_resets_after():
    """curation-api-and-ui Phase 6c: the frontend's turn-feed divider
    needs to say "from existing pool" vs "triggered a new search"
    correctly per turn -- confirmed here against a session that
    genuinely refills partway through, reusing the exact same
    refill-triggering setup as the sync-invariant test above, not a
    contrived one."""
    from research_agent import query_expansion as qe_module

    def _identity_rank(topic, papers, client=None, **kwargs):
        return [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}

    fresh_papers = [_paper(f"new{i}") for i in range(8)]

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch.object(qe_module, "build_candidate_pool", return_value=fresh_papers), \
             patch.object(qe_module, "rank_full_pool", side_effect=_identity_rank), \
             sqlite_checkpointer(db_path) as cp:

            # reserve=10 (exactly one batch): serving it all drains
            # remaining to 0 -- curation-turn-history Phase 9d narrowed
            # the auto-refill trigger to true exhaustion, not "< BATCH_SIZE".
            result = start_curation_turn("s1", cp, _session_to_dict(_session(10, target_count=100)))
            assert result["refilled"] is False  # turn 1: reserve exactly covers one batch, no refill needed yet
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            picks1 = batch1_ids[:5]

            # remaining=0 -> this resume triggers refill_pool for real.
            result = resume_curation_turn("s1", cp, picked_paper_ids=picks1, config={"client": MagicMock()})
            assert result["refilled"] is True
            assert "__interrupt__" in result
            batch2_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            picks2 = batch2_ids[:2]

            # Next turn: freshly-refilled reserve should be well above
            # BATCH_SIZE again -- refilled must reset to False, not stay
            # stuck True from the prior turn.
            result = resume_curation_turn("s1", cp, picked_paper_ids=picks2, stop=True)
            assert result["refilled"] is False


def test_get_curation_state_surfaces_refilled_for_the_currently_pending_batch():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            start_curation_turn("s1", cp, _session_to_dict(_session(15, target_count=5)))
            state = get_curation_state("s1", cp)
        assert state["refilled"] is False


def test_refinement_forces_a_refill_even_when_the_pool_does_not_need_one():
    """curation-refinement-and-auto-offer Phase 6f: refinement is a
    "change the search now" request -- it must force a refill on the
    SAME turn it's submitted, not wait for the reserve to naturally run
    low. Reserve here is large enough (25 papers, 10 served) that
    needs_refill() alone would say False."""
    from research_agent import query_expansion as qe_module

    fresh_papers = [_paper(f"new{i}") for i in range(8)]

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch.object(qe_module, "build_candidate_pool", return_value=fresh_papers) as mock_build, \
             patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                 [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
             )), \
             sqlite_checkpointer(db_path) as cp:

            result = start_curation_turn("s1", cp, _session_to_dict(_session(25, target_count=100)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            picks1 = batch1_ids[:2]  # remaining after: 25-10=15, well above BATCH_SIZE=10

            mock_build.assert_not_called()
            result = resume_curation_turn(
                "s1", cp, picked_paper_ids=picks1, refinement="focus on more recent work",
                config={"client": MagicMock()},
            )

        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs["refinement_notes"] == ["focus on more recent work"]
        assert result["session"]["refinement_notes"] == ["focus on more recent work"]


def test_refinement_notes_accumulate_across_multiple_refinements():
    from research_agent import query_expansion as qe_module

    fresh_papers = [_paper(f"new{i}") for i in range(8)]

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch.object(qe_module, "build_candidate_pool", return_value=fresh_papers) as mock_build, \
             patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                 [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
             )), \
             sqlite_checkpointer(db_path) as cp:

            result = start_curation_turn("s1", cp, _session_to_dict(_session(25, target_count=100)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]

            result = resume_curation_turn(
                "s1", cp, picked_paper_ids=batch1_ids[:1], refinement="focus on more recent work",
                config={"client": MagicMock()},
            )
            assert result["session"]["refinement_notes"] == ["focus on more recent work"]

            batch2_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            result = resume_curation_turn(
                "s1", cp, picked_paper_ids=batch2_ids[:1], refinement="prefer applied over theoretical work",
                config={"client": MagicMock()},
            )

        assert result["session"]["refinement_notes"] == [
            "focus on more recent work", "prefer applied over theoretical work",
        ]
        # both accumulated notes reach the SECOND refill's build_candidate_pool call
        assert mock_build.call_args.kwargs["refinement_notes"] == [
            "focus on more recent work", "prefer applied over theoretical work",
        ]


def test_no_refinement_does_not_force_a_refill_or_touch_refinement_notes():
    """Regression proof: a resume with no refinement text (the default,
    every existing caller) behaves byte-identically to before this
    feature existed."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(25, target_count=100)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]

            result = resume_curation_turn("s1", cp, picked_paper_ids=batch1_ids[:2])

        assert result["refilled"] is False
        assert result["session"]["refinement_notes"] == []


# --- curation-turn-history Phase 9b: turn_history + stop_reason ---

def test_turn_history_records_each_served_batch_with_turn_number_and_refilled_flag():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(25, target_count=100)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]

            result = resume_curation_turn("s1", cp, picked_paper_ids=batch1_ids[:2], stop=True)

        history = result["session"]["turn_history"]
        assert len(history) == 1
        assert history[0]["turn_number"] == 1
        assert history[0]["refilled"] is False
        assert [p["paper_id"] for p, _ in history[0]["batch"]] == batch1_ids


def test_turn_history_accumulates_across_multiple_turns_including_a_real_refill():
    from research_agent import query_expansion as qe_module

    fresh_papers = [_paper(f"new{i}") for i in range(8)]

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch.object(qe_module, "build_candidate_pool", return_value=fresh_papers), \
             patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                 [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
             )), \
             sqlite_checkpointer(db_path) as cp:

            # reserve=10 (exactly one batch) -- remaining hits 0 after
            # turn 1's serve regardless of pick count, the only auto-
            # refill trigger as of Phase 9d.
            result = start_curation_turn("s1", cp, _session_to_dict(_session(10, target_count=100)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            picks1 = batch1_ids[:5]

            # remaining=0 -> real refill on this resume.
            result = resume_curation_turn("s1", cp, picked_paper_ids=picks1, config={"client": MagicMock()})
            batch2_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]

            result = resume_curation_turn("s1", cp, picked_paper_ids=batch2_ids[:1], stop=True)

        history = result["session"]["turn_history"]
        assert len(history) == 2
        assert [entry["turn_number"] for entry in history] == [1, 2]
        assert history[0]["refilled"] is False
        assert history[1]["refilled"] is True  # the turn that actually triggered the refill
        assert [p["paper_id"] for p, _ in history[0]["batch"]] == batch1_ids
        assert [p["paper_id"] for p, _ in history[1]["batch"]] == batch2_ids


def test_turn_history_does_not_log_an_empty_batch():
    """curation-editable-until-locked Phase 10b: an exhausted search no
    longer stops the graph -- it presents an empty batch instead, still
    editable. Either way, an empty batch isn't a real turn a user could
    browse back to, so it must not appear as a hollow turn_history entry."""
    from research_agent import query_expansion as qe_module

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch.object(qe_module, "build_candidate_pool", return_value=[]), \
             patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                 [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
             )), \
             sqlite_checkpointer(db_path) as cp:

            result = start_curation_turn(
                "s1", cp, _session_to_dict(_session(5, target_count=100)), config={"client": MagicMock()},
            )
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            assert len(batch1_ids) == 5  # entire tiny reserve served in one go

            # Nothing new to find -> reserve empties out -> presented as an
            # empty batch, curation stays open (no stop, no exhausted).
            result = resume_curation_turn("s1", cp, picked_paper_ids=[], config={"client": MagicMock()})

        assert "__interrupt__" in result
        assert result["__interrupt__"][0].value["batch"] == []
        assert result["stop_reason"] is None
        assert result["session"]["stage"] == "curate"
        history = result["session"]["turn_history"]
        assert len(history) == 1  # only the real turn 1, no hollow "turn 2" entry
        assert history[0]["turn_number"] == 1


def test_after_an_empty_batch_a_refinement_search_can_find_new_candidates():
    """curation-editable-until-locked Phase 10b: the core point of not
    locking on exhaustion -- the SAME refinement mechanism (Phase 6f)
    already used to steer searches can pull the user out of an empty
    batch by finding genuinely new candidates, no separate "unstick"
    mechanism needed."""
    from research_agent import query_expansion as qe_module

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch.object(qe_module, "build_candidate_pool", return_value=[]), \
             patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                 [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
             )), \
             sqlite_checkpointer(db_path) as cp:

            result = start_curation_turn(
                "s1", cp, _session_to_dict(_session(5, target_count=100)), config={"client": MagicMock()},
            )
            # First search comes back dry -> empty batch, still open.
            result = resume_curation_turn("s1", cp, picked_paper_ids=[], config={"client": MagicMock()})
            assert result["__interrupt__"][0].value["batch"] == []
            assert result["session"]["stage"] == "curate"

            new_papers = [_paper(f"new{i}") for i in range(6)]
            with patch.object(qe_module, "build_candidate_pool", return_value=new_papers), \
                 patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                     [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
                 )):
                result = resume_curation_turn(
                    "s1", cp, picked_paper_ids=[], refinement="try a different angle",
                    config={"client": MagicMock()},
                )

        assert "__interrupt__" in result
        new_batch = result["__interrupt__"][0].value["batch"]
        assert len(new_batch) == 6
        assert {p[0]["paper_id"] for p in new_batch} == {p.paper_id for p in new_papers}
        assert result["session"]["refinement_notes"] == ["try a different angle"]


def test_stop_reason_persists_on_the_session_readable_via_get_curation_state():
    """Previously only ever existed in the one HTTP response of the turn
    that caused it -- now must survive independently, readable from the
    session even in a genuinely separate call."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(15, target_count=5)))
            batch = result["__interrupt__"][0].value["batch"]
            resume_curation_turn("s1", cp, picked_paper_ids=[batch[0][0]["paper_id"]], stop=True)

            state = get_curation_state("s1", cp)

        assert state["session"].stop_reason == "user_stopped"


def test_stop_reason_is_none_while_curation_is_still_in_progress():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            start_curation_turn("s1", cp, _session_to_dict(_session(15, target_count=5)))
            state = get_curation_state("s1", cp)

        assert state["session"].stop_reason is None


# --- curation-turn-history Phase 9d: explicit refill, narrowed auto-refill, curate-stage history picks ---

def test_partial_batch_serves_as_is_without_an_automatic_refill():
    """The actual reported bug's fix: remaining()==6 (nonzero, but well
    below the old BATCH_SIZE threshold) must NOT force a search -- the
    partial batch is presented as-is."""
    from research_agent import query_expansion as qe_module

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch.object(qe_module, "build_candidate_pool") as mock_build, \
             sqlite_checkpointer(db_path) as cp:

            # reserve=16: turn 1 serves 10, remaining=6 -- nonzero, so no
            # auto-refill under the new remaining()==0 rule.
            result = start_curation_turn("s1", cp, _session_to_dict(_session(16, target_count=100)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]

            result = resume_curation_turn("s1", cp, picked_paper_ids=batch1_ids[:2])

        mock_build.assert_not_called()
        assert result["refilled"] is False
        assert "__interrupt__" in result
        batch2 = result["__interrupt__"][0].value["batch"]
        assert len(batch2) == 6  # the entire remaining partial batch, served as-is


def test_true_exhaustion_still_triggers_one_automatic_refill_attempt():
    """Design decision 2 (approved): remaining()==0 still gets one
    automatic refill attempt -- least surprising default, not silence."""
    from research_agent import query_expansion as qe_module

    fresh_papers = [_paper(f"new{i}") for i in range(8)]

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch.object(qe_module, "build_candidate_pool", return_value=fresh_papers) as mock_build, \
             patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                 [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
             )), \
             sqlite_checkpointer(db_path) as cp:

            # reserve=10: turn 1 serves exactly all of it, remaining=0.
            result = start_curation_turn("s1", cp, _session_to_dict(_session(10, target_count=100)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]

            result = resume_curation_turn("s1", cp, picked_paper_ids=batch1_ids[:2], config={"client": MagicMock()})

        mock_build.assert_called_once()
        assert result["refilled"] is True
        assert "__interrupt__" in result


def test_request_refill_forces_a_search_even_with_a_comfortable_reserve():
    """The new explicit, user-controlled action -- reuses force_refill,
    same mechanism refinement text already triggers."""
    from research_agent import query_expansion as qe_module

    fresh_papers = [_paper(f"new{i}") for i in range(8)]

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch.object(qe_module, "build_candidate_pool", return_value=fresh_papers) as mock_build, \
             patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                 [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
             )), \
             sqlite_checkpointer(db_path) as cp:

            # reserve=25: remaining after turn 1 is 15, comfortably above
            # zero -- request_refill must force a search anyway.
            result = start_curation_turn("s1", cp, _session_to_dict(_session(25, target_count=100)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]

            mock_build.assert_not_called()
            result = resume_curation_turn(
                "s1", cp, picked_paper_ids=batch1_ids[:2], request_refill=True, config={"client": MagicMock()},
            )

        mock_build.assert_called_once()
        assert result["refilled"] is True


def test_request_refill_false_is_the_default_and_does_not_force_anything():
    """Regression proof: every existing caller (request_refill omitted)
    behaves byte-identically to before this feature existed."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(25, target_count=100)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]

            result = resume_curation_turn("s1", cp, picked_paper_ids=batch1_ids[:2])

        assert result["refilled"] is False


def test_picks_can_reference_a_paper_from_an_earlier_turn_while_still_curating():
    """The curate-stage-safe channel for picking from history: bundled
    into the SAME /picks resume call, validated against the whole
    turn_history, not just current_batch -- the only channel safe to use
    while a real interrupt is pending (see _present_and_apply_node's own
    docstring for why select_paper_from_history's out-of-band write
    can't be used here instead)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(25, target_count=100)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            turn1_unpicked = batch1_ids[3]  # seen in turn 1, deliberately never picked there

            result = resume_curation_turn("s1", cp, picked_paper_ids=batch1_ids[:2])  # turn 1 picks
            assert "__interrupt__" in result
            batch2_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            assert turn1_unpicked not in batch2_ids  # confirms it's genuinely from an EARLIER turn

            # Turn 2's picks include one paper from turn 2's own batch AND
            # one from turn 1's -- both in the SAME resume call.
            result = resume_curation_turn(
                "s1", cp, picked_paper_ids=[batch2_ids[0], turn1_unpicked], stop=True,
            )

        assert set(result["session"]["selected_paper_ids"]) == {batch1_ids[0], batch1_ids[1], batch2_ids[0], turn1_unpicked}


def test_picks_referencing_a_paper_never_served_at_all_are_still_silently_rejected():
    """Regression proof for the existing validation guarantee, now that
    it checks turn_history instead of just current_batch: a fabricated
    id must still be dropped, not corrupt selected_paper_ids."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(15, target_count=1)))
            batch = result["__interrupt__"][0].value["batch"]
            real_id = batch[0][0]["paper_id"]

            result = resume_curation_turn(
                "s1", cp, picked_paper_ids=[real_id, "does-not-exist-anywhere"],
            )

        assert result["session"]["selected_paper_ids"] == [real_id]


# =====================================================================
# Usage Protection M2.2B: the conditional curation-refill guard, opened
# inside curation_loop.py's own _refill_node -- the exact, single point
# where a refill turn's paid work becomes certain.
# =====================================================================

def _rows(db_path, table):
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}").fetchall()]
    finally:
        conn.close()


def test_no_refill_turn_bypasses_admission_and_lease(usage_db_path):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            # reserve=15, batch_size=10 -> remaining=5 after turn 1, no refill.
            start_curation_turn("s1", cp, _session_to_dict(_session(15, target_count=100)))
    assert _rows(usage_db_path, "paid_actions") == []
    assert _rows(usage_db_path, "action_leases") == []


def test_real_refill_turn_is_admitted_and_leased(usage_db_path):
    from research_agent import query_expansion as qe_module

    fresh_papers = [_paper(f"new{i}") for i in range(8)]
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch.object(qe_module, "build_candidate_pool", return_value=fresh_papers), \
             patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                 [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
             )), \
             sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(10, target_count=100)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            result = resume_curation_turn("s1", cp, picked_paper_ids=batch1_ids[:2], config={"client": MagicMock()})
    assert result["refilled"] is True

    rows = _rows(usage_db_path, "paid_actions")
    assert len(rows) == 1  # exactly one top-level record, no duplicate/child-guard rows
    assert rows[0]["action_type"] == "curation_refill"
    assert rows[0]["subject_type"] == "session"
    assert rows[0]["subject_id"] == "s1"
    assert _rows(usage_db_path, "action_leases") == []  # released


def test_exhausted_budget_rejects_refill_before_provider_work(usage_db_path):
    from datetime import datetime, timezone

    from research_agent import query_expansion as qe_module
    from research_agent.config import get_usage_policy

    policy = get_usage_policy()
    now = datetime.now(timezone.utc).isoformat()
    for i in range(policy.max_paid_actions_per_session_per_hour):
        telemetry._write_paid_action(
            action_id=f"seed-{i}", action_type="curation_chat", request_id=None,
            subject_type="session", subject_id="s1", outcome="success", started_at=now, ended_at=now,
            latency_ms=1.0, input_tokens=None, output_tokens=None, total_tokens=None, total_call_count=1,
            child_calls_json="[]", path=usage_db_path,
        )

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch.object(qe_module, "build_candidate_pool") as mock_build, \
             sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(10, target_count=100)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            with pytest.raises(UsageGuardRejection) as exc_info:
                resume_curation_turn("s1", cp, picked_paper_ids=batch1_ids[:2], config={"client": MagicMock()})

    assert exc_info.value.reason_code == "session_hourly_limit_reached"
    mock_build.assert_not_called()  # rejected before any provider work
    # Still exactly the seeded rows -- the rejected attempt added none.
    assert len(_rows(usage_db_path, "paid_actions")) == policy.max_paid_actions_per_session_per_hour


def test_lease_releases_after_refill_failure():
    from research_agent import query_expansion as qe_module

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with patch.object(qe_module, "build_candidate_pool", side_effect=RuntimeError("boom")), \
             sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(10, target_count=100)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]

            with pytest.raises(RuntimeError):
                resume_curation_turn("s1", cp, picked_paper_ids=batch1_ids[:2], config={"client": MagicMock()})

            # Lease released despite the failure -- a second refill
            # attempt for the same session must not be blocked by it.
            with patch.object(qe_module, "build_candidate_pool", return_value=[_paper("new1")]), \
                 patch.object(qe_module, "rank_full_pool", return_value=([(_paper("new1"), 1.0)], {})):
                result2 = resume_curation_turn("s1", cp, picked_paper_ids=[], stop=False, request_refill=True, config={"client": MagicMock()})
            assert result2["refilled"] is True


def test_concurrent_same_session_refill_allows_exactly_one(usage_db_path):
    """Real OS threads racing on _refill_node itself -- the exact node
    this phase modified -- rather than through two concurrent
    graph.invoke() calls on the SAME LangGraph thread_id. LangGraph's own
    checkpointer does not guarantee well-defined behavior for two
    truly-concurrent resumes of the same pending interrupt (confirmed
    empirically: it introduces its own nondeterministic retry/ordering
    noise unrelated to the guard), so that would not be a clean test of
    the guard's own exclusion -- which is the thing this test, and
    tests/test_usage_guard.py's own thorough multi-thread lease
    coverage, actually need to prove. Calling the node function directly
    with two threads exercises the identical guard_paid_action call
    _refill_node makes inside the real graph, deterministically."""
    from research_agent.curation_loop import CurationLoopState, _refill_node

    release_event = threading.Event()
    entered_event = threading.Event()
    session = _session(10, target_count=100)
    state: CurationLoopState = {
        "session": _session_to_dict(session), "current_batch": [], "stop_reason": None,
        "should_stop": False, "refilled": False, "force_refill": False,
    }
    config = {"configurable": {"thread_id": "curation-session:s1", "client": MagicMock()}}

    def _blocking_refill_pool(session, **kwargs):
        entered_event.set()
        release_event.wait(timeout=5)
        session.reserve = session.reserve + [(_paper("new1"), 1.0)]
        return 1

    with patch("research_agent.curation_loop.refill_pool", side_effect=_blocking_refill_pool):
        outcomes = {}

        def _attempt(key):
            try:
                _refill_node(state, config)
                outcomes[key] = "admitted"
            except UsageGuardRejection as exc:
                outcomes[key] = exc.reason_code

        t1 = threading.Thread(target=_attempt, args=("t1",))
        t1.start()
        assert entered_event.wait(timeout=5)  # t1 now holds the lease, blocked in refill_pool

        _attempt("t2")  # runs on the main thread while t1 still holds the lease

        release_event.set()
        t1.join(timeout=5)

    assert outcomes["t1"] == "admitted"
    assert outcomes["t2"] == "action_in_progress"


def test_real_usage_db_path_untouched():
    """Does NOT assert nonexistence -- a legitimate local
    usage_telemetry.sqlite from real dev-server use is normal, valid
    state. Proves nothing in this file's test run created, deleted, or
    modified it (or its -wal/-shm sidecars)."""
    assert fingerprint_usage_db(_REAL_USAGE_DB_PATH) == _REAL_USAGE_DB_FINGERPRINT_BEFORE


if __name__ == "__main__":
    test_reaching_target_count_no_longer_stops_curation()
    test_multi_turn_loop_continues_past_target_met_since_it_no_longer_stops()
    test_picks_not_in_presented_batch_are_silently_rejected()
    test_double_mutation_across_interrupt_resume_boundary()
    test_user_picks_zero_still_progresses_to_a_new_batch_not_a_stall()
    test_user_picks_all_ten_from_a_batch()
    test_user_stops_before_hitting_target()
    test_target_hit_exactly_on_a_batch_boundary()
    test_target_hit_mid_batch()
    test_selected_papers_and_selected_paper_ids_stay_in_sync_across_a_refill()
    test_get_curation_state_returns_none_for_a_never_started_session()
    test_get_curation_state_exposes_pending_batch_mid_interrupt()
    test_get_curation_state_pending_batch_is_none_once_curation_finishes()
    test_refilled_is_false_on_the_first_turn_when_the_pool_is_already_large_enough()
    test_refilled_is_true_exactly_on_the_turn_that_triggers_a_real_refill_and_resets_after()
    test_get_curation_state_surfaces_refilled_for_the_currently_pending_batch()
    test_refinement_forces_a_refill_even_when_the_pool_does_not_need_one()
    test_refinement_notes_accumulate_across_multiple_refinements()
    test_no_refinement_does_not_force_a_refill_or_touch_refinement_notes()
    test_turn_history_records_each_served_batch_with_turn_number_and_refilled_flag()
    test_turn_history_accumulates_across_multiple_turns_including_a_real_refill()
    test_turn_history_does_not_log_an_empty_batch()
    test_after_an_empty_batch_a_refinement_search_can_find_new_candidates()
    test_stop_reason_persists_on_the_session_readable_via_get_curation_state()
    test_stop_reason_is_none_while_curation_is_still_in_progress()
    test_partial_batch_serves_as_is_without_an_automatic_refill()
    test_true_exhaustion_still_triggers_one_automatic_refill_attempt()
    test_request_refill_forces_a_search_even_with_a_comfortable_reserve()
    test_request_refill_false_is_the_default_and_does_not_force_anything()
    test_picks_can_reference_a_paper_from_an_earlier_turn_while_still_curating()
    test_picks_referencing_a_paper_never_served_at_all_are_still_silently_rejected()
    print("All curation_loop tests passed.")
