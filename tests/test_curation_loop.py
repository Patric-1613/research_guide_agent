"""Tests for curation_loop.py (curation-interrupt-loop Phase 3): the
interactive present/interrupt/resume curation loop. Baseline
single-process correctness here; the harder cross-process resume proof
lives in scripts run for Phase 3d (see that phase's own driver scripts),
since pytest itself runs everything in one process by construction.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.curation_loop import (
    resume_curation_turn,
    start_curation_turn,
)
from research_agent.curation_session import _session_to_dict
from research_agent.qa import sqlite_checkpointer
from research_agent.query_expansion import PaperPoolSession
from research_agent.schema import Paper


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


def test_single_turn_interrupt_then_resume_reaches_target():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(15, target_count=5)))
            assert "__interrupt__" in result
            batch = result["__interrupt__"][0].value["batch"]
            assert len(batch) == 10

            picks = [p[0]["paper_id"] for p in batch[:5]]
            result2 = resume_curation_turn("s1", cp, picked_paper_ids=picks)

        assert result2["stop_reason"] == "target_met"
        assert result2["session"]["selected_paper_ids"] == picks
        assert result2["session"]["stage"] == "synthesize"


def test_multi_turn_loop_continues_across_batches_until_target_met():
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

        assert result["stop_reason"] == "target_met"
        assert set(result["session"]["selected_paper_ids"]) == set(picks1) | set(picks2)
        assert len(result["session"]["selected_paper_ids"]) == 12


def test_picks_not_in_presented_batch_are_silently_rejected():
    """The user's explicit addition: validate against the ACTUAL batch,
    ignore anything else, rather than trusting the resume payload."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            # target_count=1 so the single valid pick below meets it
            # immediately -- this test is about pick validation, not
            # multi-turn looping/refill, so the loop must stop here.
            result = start_curation_turn("s1", cp, _session_to_dict(_session(15, target_count=1)))
            batch = result["__interrupt__"][0].value["batch"]
            real_id = batch[0][0]["paper_id"]

            # A mix of one real pick, one paper that exists elsewhere in the
            # reserve but was NOT in this batch, and one that doesn't exist at all.
            not_in_batch_but_in_reserve = "p12"  # reserve has p0..p14, batch is only the first 10
            fabricated = "does-not-exist-anywhere"
            result2 = resume_curation_turn(
                "s1", cp, picked_paper_ids=[real_id, not_in_batch_but_in_reserve, fabricated],
            )

        assert result2["stop_reason"] == "target_met"
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
            # target_count=3, picking 3 below meets it in one batch -- no
            # second batch/refill needed, keeping this test isolated to
            # exactly the interrupt/resume boundary it's meant to check.
            result = start_curation_turn("s1", cp, _session_to_dict(_session(15, target_count=3)))
            assert call_count["n"] == 1, "serve_next_batch must run exactly once on the halting call"

            batch = result["__interrupt__"][0].value["batch"]
            cursor_after_halt = result["session"]["cursor"]

            picks = [p[0]["paper_id"] for p in batch[:3]]
            result2 = resume_curation_turn("s1", cp, picked_paper_ids=picks)

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

        assert result["stop_reason"] == "target_met"
        assert result["session"]["selected_paper_ids"] == all_ids


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
    picks the ENTIRE second batch of 10 (20/20) -- target is met exactly
    when the second batch is fully consumed, not mid-way through it."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.sqlite"
        with sqlite_checkpointer(db_path) as cp:
            result = start_curation_turn("s1", cp, _session_to_dict(_session(25, target_count=20)))
            batch1_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]

            result = resume_curation_turn("s1", cp, picked_paper_ids=batch1_ids)
            assert "__interrupt__" in result  # 10/20, must continue

            batch2_ids = [p[0]["paper_id"] for p in result["__interrupt__"][0].value["batch"]]
            result = resume_curation_turn("s1", cp, picked_paper_ids=batch2_ids)

        assert result["stop_reason"] == "target_met"
        assert len(result["session"]["selected_paper_ids"]) == 20


def test_target_hit_mid_batch():
    """target_count=13: turn 1 picks a full batch of 10 (10/13), turn 2
    picks only 3 of the 10 presented -- target is met mid-batch, with 7
    of turn 2's batch never picked."""
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

        assert result["stop_reason"] == "target_met"
        assert len(result["session"]["selected_paper_ids"]) == 13
        assert result["session"]["selected_paper_ids"] == batch1_ids + picks2


if __name__ == "__main__":
    test_single_turn_interrupt_then_resume_reaches_target()
    test_multi_turn_loop_continues_across_batches_until_target_met()
    test_picks_not_in_presented_batch_are_silently_rejected()
    test_double_mutation_across_interrupt_resume_boundary()
    test_user_picks_zero_still_progresses_to_a_new_batch_not_a_stall()
    test_user_picks_all_ten_from_a_batch()
    test_user_stops_before_hitting_target()
    test_target_hit_exactly_on_a_batch_boundary()
    test_target_hit_mid_batch()
    print("All curation_loop tests passed.")
