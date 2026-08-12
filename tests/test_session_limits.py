"""Usage Protection M2.2C Parts C/D: unit tests for
research_agent/session_limits.py's pure functions -- the shared
capacity-check logic both curation_loop.py's _present_and_apply_node/
curation_session.py's select_paper_from_history (selected-paper cap)
and curation_chat_service.py (chat-turn cap) call. No DB, no HTTP, no
network -- these are pure Python functions over plain lists/dicts.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from research_agent.config import get_usage_policy
from research_agent.session_limits import (
    SessionCapacityError,
    check_chat_turn_capacity,
    check_selected_paper_capacity,
    count_chat_turns,
)

POLICY = get_usage_policy()


class TestSelectedPaperCapacity:
    def test_prospective_unique_count_under_cap_allowed(self):
        check_selected_paper_capacity(["a", "b"], ["c", "d"], POLICY)  # must not raise

    def test_duplicates_already_selected_do_not_consume_capacity(self):
        existing = [f"p{i}" for i in range(POLICY.max_selected_papers_per_session)]
        # Re-picking IDs already selected must not push it over the cap.
        check_selected_paper_capacity(existing, existing[:5], POLICY)  # must not raise

    def test_exactly_60_unique_is_allowed(self):
        existing = [f"p{i}" for i in range(POLICY.max_selected_papers_per_session - 1)]
        check_selected_paper_capacity(existing, ["new-one"], POLICY)  # -> exactly the cap, must not raise

    def test_61_unique_is_rejected(self):
        existing = [f"p{i}" for i in range(POLICY.max_selected_papers_per_session)]
        with pytest.raises(SessionCapacityError) as exc_info:
            check_selected_paper_capacity(existing, ["one-more"], POLICY)
        assert exc_info.value.reason_code == "selected_paper_limit_reached"

    def test_mixed_duplicate_and_new_ids_counts_only_genuinely_new_ones(self):
        existing = [f"p{i}" for i in range(POLICY.max_selected_papers_per_session - 2)]
        # 2 slots left: 1 duplicate (free) + 2 new ones would be 2 new -> exactly fills it.
        check_selected_paper_capacity(existing, [existing[0], "new-1", "new-2"], POLICY)  # must not raise
        with pytest.raises(SessionCapacityError):
            check_selected_paper_capacity(existing, [existing[0], "new-1", "new-2", "new-3"], POLICY)


class TestChatTurnCounting:
    def test_counts_user_entries_only(self):
        history = [
            {"role": "user", "content": "q1", "exchange_id": "e1"},
            {"role": "assistant", "content": "a1", "exchange_id": "e1"},
            {"role": "user", "content": "q2", "exchange_id": "e2"},
            {"role": "assistant", "content": "a2", "exchange_id": "e2"},
        ]
        assert count_chat_turns(history) == 2

    def test_old_entries_without_exchange_metadata_still_count(self):
        # Pre-Phase-1 entries: no exchange_id key at all.
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        assert count_chat_turns(history) == 1

    def test_empty_history_counts_zero(self):
        assert count_chat_turns([]) == 0


class TestChatTurnCapacity:
    def _history_with_n_turns(self, n):
        history = []
        for i in range(n):
            history.append({"role": "user", "content": f"q{i}", "exchange_id": f"e{i}"})
            history.append({"role": "assistant", "content": f"a{i}", "exchange_id": f"e{i}"})
        return history

    def test_first_99_turns_allowed_before_the_100th_begins(self):
        history = self._history_with_n_turns(POLICY.max_chat_turns_per_session - 1)
        check_chat_turn_capacity(history, POLICY)  # the 100th turn may still begin

    def test_100_turns_already_present_rejects_the_101st(self):
        history = self._history_with_n_turns(POLICY.max_chat_turns_per_session)
        with pytest.raises(SessionCapacityError) as exc_info:
            check_chat_turn_capacity(history, POLICY)
        assert exc_info.value.reason_code == "chat_turn_limit_reached"

    def test_old_entries_without_metadata_count_toward_the_cap(self):
        history = [{"role": "user", "content": f"q{i}"} for i in range(POLICY.max_chat_turns_per_session)]
        # Interleave fake assistant replies so counting stays representative.
        interleaved = []
        for turn in history:
            interleaved.append(turn)
            interleaved.append({"role": "assistant", "content": "a"})
        with pytest.raises(SessionCapacityError):
            check_chat_turn_capacity(interleaved, POLICY)

    def test_deleting_an_exchange_restores_capacity(self):
        history = self._history_with_n_turns(POLICY.max_chat_turns_per_session)
        with pytest.raises(SessionCapacityError):
            check_chat_turn_capacity(history, POLICY)

        # Simulate delete_chat_exchanges removing exchange "e0" (both its entries).
        after_delete = [t for t in history if t.get("exchange_id") != "e0"]
        check_chat_turn_capacity(after_delete, POLICY)  # must not raise now
