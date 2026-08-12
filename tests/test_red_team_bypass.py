"""Usage Protection M2.3 Part E: a compact adversarial regression matrix
proving the M2.1-M2.2C protections cannot be bypassed through obvious
alternate paths -- false Content-Length, alternate endpoints sharing the
same constrained type, malformed subject identifiers, race conditions
around the session capacity/admission/lease boundaries, etc.

These are deterministic backend/API tests, not LLM evaluations -- no
network/paid calls, no new eval CLI suite. Reuses the existing
test_curation_api.py fixtures (_client, _paper, _ranked, _finish_curation)
rather than re-deriving them, and test_api.py's own usage-DB-aware
_client_with_usage_db-style pattern where a raw usage-DB fingerprint is
needed. Real data/usage_telemetry.sqlite is never touched -- every test
here goes through _client()'s own USAGE_DB_PATH redirection (all three
of telemetry/admission/leases, see that fixture's own comment).
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import research_agent.admission as admission
import research_agent.api as api
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.config import get_usage_policy
from research_agent.usage_guard import UsageGuardRejection
from tests.test_curation_api import _client, _finish_curation, _paper, _ranked

POLICY = get_usage_policy()


def _rows(db_path, table):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}").fetchall()]
    finally:
        conn.close()


# =====================================================================
# Request/body bypass attempts
# =====================================================================

class TestBodyBypass:
    def test_oversized_body_with_false_small_content_length_still_rejected(self):
        with _client() as client:
            oversized = b'{"topic": "' + b"x" * (POLICY.max_request_body_bytes + 500) + b'"}'
            # httpx/TestClient computes its own real Content-Length from the
            # actual body -- to simulate a client LYING with a small
            # declared Content-Length, send the request through the raw
            # ASGI transport instead, forging the header directly.
            resp = client.post(
                "/search",
                content=oversized,
                headers={"content-type": "application/json", "content-length": "10"},
            )
        assert resp.status_code == 413
        assert resp.json()["detail"]["reason_code"] == "request_body_too_large"

    def test_oversized_chunked_body_rejected_with_no_content_length_header(self):
        with _client() as client:
            oversized = b'{"topic": "' + b"x" * (POLICY.max_request_body_bytes + 500) + b'"}'
            resp = client.post("/search", content=oversized, headers={"content-type": "application/json"})
        assert resp.status_code == 413

    def test_multibyte_unicode_body_exceeding_byte_limit_is_rejected(self):
        """A 4-byte-per-character emoji body can exceed the byte limit
        with far fewer "characters" than a naive character-count check
        would allow -- the limit must be enforced in bytes."""
        with _client() as client:
            # Each emoji is 4 UTF-8 bytes; well under max_text_length
            # (2000) characters but over max_request_body_bytes in bytes.
            emoji_count = (POLICY.max_request_body_bytes // 4) + 100
            payload = ("\U0001F600" * emoji_count).encode("utf-8")
            body = b'{"topic": "' + payload + b'"}'
            assert len(body) > POLICY.max_request_body_bytes
            resp = client.post("/search", content=body, headers={"content-type": "application/json"})
        assert resp.status_code == 413

    def test_oversized_body_rejected_before_telemetry_lease_route_or_mutation(self):
        with _client() as client:
            oversized = b'{"topic": "' + b"x" * (POLICY.max_request_body_bytes + 500) + b'"}'
            with patch.object(api, "run_research_agent") as mock_agent:
                resp = client.post("/search", content=oversized, headers={"content-type": "application/json"})
        assert resp.status_code == 413
        mock_agent.assert_not_called()  # never reached the route/provider


# =====================================================================
# Text/schema bypass attempts -- alternate endpoints sharing one type
# =====================================================================

class TestSchemaBypass:
    def test_2001_char_topic_rejected_via_search(self):
        with _client() as client:
            resp = client.post("/search", json={"topic": "x" * 2001})
        assert resp.status_code == 422

    def test_2001_char_topic_rejected_via_curation_start_too(self):
        """Same UserText type, a DIFFERENT endpoint -- confirms the
        constraint isn't accidentally endpoint-specific."""
        with _client() as client:
            resp = client.post("/curation/start", json={"topic": "x" * 2001, "target_count": 5})
        assert resp.status_code == 422

    def test_2001_char_chat_message_rejected(self):
        with _client() as client:
            session_id, _ = _finish_curation(client)
            resp = client.post(f"/curation/{session_id}/chat", json={"message": "x" * 2001})
        assert resp.status_code == 422

    def test_oversized_edit_question_rejected(self):
        with _client() as client:
            session_id, _ = _finish_curation(client)
            resp = client.post(
                f"/curation/{session_id}/chat/exchanges/edit",
                json={"exchange_id": "e1", "question": "x" * 2001},
            )
        assert resp.status_code == 422

    def test_31_picked_ids_rejected(self):
        with _client() as client:
            session_id, _ = _finish_curation(client)
            resp = client.post(
                f"/curation/{session_id}/picks",
                json={"picked_paper_ids": [f"p{i}" for i in range(31)]},
            )
        assert resp.status_code == 422

    def test_31_exchange_ids_rejected_via_delete(self):
        with _client() as client:
            session_id, _ = _finish_curation(client)
            resp = client.post(
                f"/curation/{session_id}/chat/exchanges/delete",
                json={"exchange_ids": [f"e{i}" for i in range(31)]},
            )
        assert resp.status_code == 422

    def test_31_exchange_ids_rejected_via_add_to_report_too(self):
        """Same IdList type as delete, a DIFFERENT endpoint -- confirms
        add-to-report can't be used as a bypass route around delete's
        own 30-id cap."""
        with _client() as client:
            session_id, _ = _finish_curation(client)
            resp = client.post(
                f"/curation/{session_id}/chat/exchanges/add-to-report",
                json={"exchange_ids": [f"e{i}" for i in range(31)]},
            )
        assert resp.status_code == 422


# =====================================================================
# Session capacity bypass attempts
# =====================================================================

class TestSessionCapacityBypass:
    def test_duplicate_paper_ids_do_not_inflate_unique_count(self):
        with _client() as client:
            session_id, pick_ids = _finish_curation(client, target_count=2, n_papers=12)
            state = client.get(f"/curation/{session_id}").json()
            before = len(state["selected_paper_ids"])
            # Re-submitting already-selected ids via select-from-history
            # (a no-op per its own documented duplicate tolerance) must
            # not be counted as consuming capacity.
            resp = client.post(f"/curation/{session_id}/select-from-history", json={"paper_id": pick_ids[0]})
        assert resp.status_code == 200
        assert len(resp.json()["selected_paper_ids"]) == before

    def test_history_selection_path_cannot_bypass_the_60_cap(self):
        """select_paper_from_history is a SEPARATE code path from /picks
        -- confirms it shares check_selected_paper_capacity, not a
        second, weaker check."""
        from tests.test_curation_api import _select_up_to_n_papers

        with _client() as client:
            session_id, selected = _select_up_to_n_papers(client, 60)
            client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [], "stop": True})
            state = client.get(f"/curation/{session_id}").json()
            all_served = {p["paper_id"] for entry in state["turn_history"] for p in entry["batch"]}
            unselected = next(iter(all_served - set(selected)))
            resp = client.post(f"/curation/{session_id}/select-from-history", json={"paper_id": unselected})
        assert resp.status_code == 409
        assert resp.json()["detail"]["reason_code"] == "selected_paper_limit_reached"

    def test_normal_picks_path_cannot_bypass_the_60_cap(self):
        from tests.test_curation_api import _select_up_to_n_papers

        with _client() as client:
            session_id, selected = _select_up_to_n_papers(client, 60)
            state = client.get(f"/curation/{session_id}").json()
            over_pick = state["pending_batch"][0]["paper_id"]
            resp = client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [over_pick], "stop": False})
        assert resp.status_code == 409
        assert resp.json()["detail"]["reason_code"] == "selected_paper_limit_reached"

    def test_old_chat_entries_without_exchange_id_still_count_by_role(self, monkeypatch):
        monkeypatch.setenv("USAGE_MAX_PAID_ACTIONS_PER_SESSION_PER_HOUR", "1000")
        monkeypatch.setenv("USAGE_GLOBAL_PAID_ACTION_LIMIT", "1000")
        limit = POLICY.max_chat_turns_per_session
        with _client() as client:
            session_id, _ = _finish_curation(client)

            def _bare_chat_turn(session, message, client=None, **kwargs):
                # Old-shape entries: no exchange_id/cited_papers/etc at all.
                session.chat_history.append({"role": "user", "content": message})
                session.chat_history.append({"role": "assistant", "content": "ok"})
                return {"answer": "ok", "answerable": True, "cited_papers": [], "cited_web_articles": []}

            with patch.object(api, "chat_turn", side_effect=_bare_chat_turn):
                for i in range(limit):
                    assert client.post(f"/curation/{session_id}/chat", json={"message": f"q{i}"}).status_code == 200
                resp = client.post(f"/curation/{session_id}/chat", json={"message": "over"})
        assert resp.status_code == 409
        assert resp.json()["detail"]["reason_code"] == "chat_turn_limit_reached"

    def test_edit_at_cap_follows_truncate_then_append_not_a_hard_block(self, monkeypatch):
        """Editing the OLDEST exchange while at the 100-turn cap must
        still succeed -- truncation removes 99, the fresh replacement
        adds 1, landing well under the cap, not treated as "the session
        is full" by a naive pre-edit count check."""
        monkeypatch.setenv("USAGE_MAX_PAID_ACTIONS_PER_SESSION_PER_HOUR", "1000")
        monkeypatch.setenv("USAGE_GLOBAL_PAID_ACTION_LIMIT", "1000")
        limit = POLICY.max_chat_turns_per_session
        with _client() as client:
            session_id, _ = _finish_curation(client)

            def _fake_chat_turn(session, message, client=None, **kwargs):
                eid = f"e{len(session.chat_history)}"
                session.chat_history.append({"role": "user", "content": message, "exchange_id": eid})
                session.chat_history.append({"role": "assistant", "content": "ok", "exchange_id": eid})
                return {"answer": "ok", "answerable": True, "cited_papers": [], "cited_web_articles": []}

            with patch.object(api, "chat_turn", side_effect=_fake_chat_turn):
                for i in range(limit):
                    client.post(f"/curation/{session_id}/chat", json={"message": f"q{i}"})

            def _fake_edit(session, exchange_id, new_question, client=None, **kwargs):
                _fake_chat_turn(session, new_question, client=client)
                return ({"answer": "edited", "answerable": True, "cited_papers": [], "cited_web_articles": []}, False)

            with patch.object(api, "edit_chat_exchange", side_effect=_fake_edit):
                resp = client.post(f"/curation/{session_id}/chat/exchanges/edit", json={"exchange_id": "e0", "question": "edited"})
        assert resp.status_code == 200

    def test_deletion_restores_room_for_another_chat_turn(self, monkeypatch):
        monkeypatch.setenv("USAGE_MAX_PAID_ACTIONS_PER_SESSION_PER_HOUR", "1000")
        monkeypatch.setenv("USAGE_GLOBAL_PAID_ACTION_LIMIT", "1000")
        limit = POLICY.max_chat_turns_per_session
        with _client() as client:
            session_id, _ = _finish_curation(client)

            def _fake_chat_turn(session, message, client=None, **kwargs):
                eid = f"e{len(session.chat_history)}"
                session.chat_history.append({"role": "user", "content": message, "exchange_id": eid})
                session.chat_history.append({"role": "assistant", "content": "ok", "exchange_id": eid})
                return {"answer": "ok", "answerable": True, "cited_papers": [], "cited_web_articles": []}

            with patch.object(api, "chat_turn", side_effect=_fake_chat_turn):
                for i in range(limit):
                    client.post(f"/curation/{session_id}/chat", json={"message": f"q{i}"})
                blocked = client.post(f"/curation/{session_id}/chat", json={"message": "blocked"})
                assert blocked.status_code == 409

                client.post(f"/curation/{session_id}/chat/exchanges/delete", json={"exchange_ids": ["e0"]})
                resp = client.post(f"/curation/{session_id}/chat", json={"message": "now fits"})
        assert resp.status_code == 200


# =====================================================================
# Admission/concurrency bypass attempts
# =====================================================================

class TestAdmissionConcurrencyBypass:
    def test_same_session_concurrent_report_and_chat_conflict_through_the_shared_lease(self):
        with _client() as client:
            session_id, _ = _finish_curation(client)
            release_event = threading.Event()
            entered_event = threading.Event()

            def _blocking_generate(session, client=None, **kwargs):
                entered_event.set()
                release_event.wait(timeout=5)
                return {
                    "findings": {"content": "f", "cited_papers": []}, "limitations": {"content": "l", "cited_papers": []},
                    "future_scope": {"content": "fs", "cited_papers": []}, "skipped_papers": [],
                }

            results = {}

            def _generate_report():
                with patch.object(api, "generate_report_for_session", side_effect=_blocking_generate):
                    results["report"] = client.post(f"/curation/{session_id}/report")

            t = threading.Thread(target=_generate_report)
            t.start()
            assert entered_event.wait(timeout=5)

            # A concurrent CHAT call for the SAME session must conflict --
            # both report_generate and curation_chat share the one
            # EXPENSIVE_ACTION_GROUP lease per session (research_agent/
            # leases.py), a deliberate design choice, not an accident.
            results["chat"] = client.post(f"/curation/{session_id}/chat", json={"message": "hi"})

            release_event.set()
            t.join(timeout=5)

        assert results["chat"].status_code == 409
        assert results["chat"].json()["detail"]["reason_code"] == "action_in_progress"
        assert results["report"].status_code == 200

    def test_different_sessions_remain_independent_under_the_lease(self):
        with _client() as client:
            session_id_1, _ = _finish_curation(client)
            session_id_2, _ = _finish_curation(client)
            release_event = threading.Event()
            entered_event = threading.Event()

            def _blocking_chat_turn(session, message, client=None, **kwargs):
                entered_event.set()
                release_event.wait(timeout=5)
                session.chat_history.append({"role": "user", "content": message})
                session.chat_history.append({"role": "assistant", "content": "ok"})
                return {"answer": "ok", "answerable": True, "cited_papers": [], "cited_web_articles": []}

            results = {}

            def _call_session_1():
                with patch.object(api, "chat_turn", side_effect=_blocking_chat_turn):
                    results["one"] = client.post(f"/curation/{session_id_1}/chat", json={"message": "hi"})

            t = threading.Thread(target=_call_session_1)
            t.start()
            assert entered_event.wait(timeout=5)

            with patch.object(api, "chat_turn", return_value={"answer": "ok", "answerable": True, "cited_papers": [], "cited_web_articles": []}):
                results["two"] = client.post(f"/curation/{session_id_2}/chat", json={"message": "hi"})

            release_event.set()
            t.join(timeout=5)

        assert results["one"].status_code == 200
        assert results["two"].status_code == 200  # not blocked by session 1's lease

    def test_rejected_work_performs_zero_provider_calls(self):
        with _client() as client:
            session_id, _ = _finish_curation(client)
            now = datetime.now(timezone.utc).isoformat()
            for i in range(POLICY.max_paid_actions_per_session_per_hour):
                telemetry._write_paid_action(
                    action_id=f"seed-{i}", action_type="curation_chat", request_id=None,
                    subject_type="session", subject_id=session_id, outcome="success",
                    started_at=now, ended_at=now, latency_ms=1.0, input_tokens=None,
                    output_tokens=None, total_tokens=None, total_call_count=1,
                    child_calls_json="[]", path=telemetry.USAGE_DB_PATH,
                )
            with patch.object(api, "chat_turn") as mock_chat_turn:
                resp = client.post(f"/curation/{session_id}/chat", json={"message": "hi"})
        assert resp.status_code == 429
        mock_chat_turn.assert_not_called()

    def test_budget_rejection_creates_no_paid_action_row(self):
        with _client() as client:
            session_id, _ = _finish_curation(client)
            now = datetime.now(timezone.utc).isoformat()
            for i in range(POLICY.max_paid_actions_per_session_per_hour):
                telemetry._write_paid_action(
                    action_id=f"seed2-{i}", action_type="curation_chat", request_id=None,
                    subject_type="session", subject_id=session_id, outcome="success",
                    started_at=now, ended_at=now, latency_ms=1.0, input_tokens=None,
                    output_tokens=None, total_tokens=None, total_call_count=1,
                    child_calls_json="[]", path=telemetry.USAGE_DB_PATH,
                )
            before = len(_rows(telemetry.USAGE_DB_PATH, "paid_actions"))
            with patch.object(api, "chat_turn"):
                resp = client.post(f"/curation/{session_id}/chat", json={"message": "hi"})
            after = len(_rows(telemetry.USAGE_DB_PATH, "paid_actions"))
        assert resp.status_code == 429
        assert after == before

    def test_stale_lease_replacement_works(self, tmp_path):
        db_path = tmp_path / "usage.sqlite"
        telemetry.init_usage_db(path=db_path).close()
        with patch.object(telemetry, "USAGE_DB_PATH", db_path), \
             patch.object(admission, "USAGE_DB_PATH", db_path), \
             patch.object(leases, "USAGE_DB_PATH", db_path):
            first = leases.acquire_lease("session", "s1", ttl_seconds=0)
            assert first.acquired is True
            import time
            time.sleep(0.02)
            second = leases.acquire_lease("session", "s1", ttl_seconds=60)
            assert second.acquired is True
            assert second.token != first.token

    def test_malformed_missing_subject_identifiers_cannot_create_a_useful_bypass(self):
        """guard_paid_action(subject=None) -- the code path used by
        /search and /curation/start, which genuinely have no subject yet
        -- only ever grants the coarse GLOBAL admission, never skips
        admission altogether. Confirmed directly against the guard, not
        just its two real call sites."""
        import tempfile

        from research_agent.usage_guard import guard_paid_action

        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            db_path = Path(tmp) / "usage.sqlite"
            telemetry.init_usage_db(path=db_path).close()
            with patch.object(telemetry, "USAGE_DB_PATH", db_path), \
                 patch.object(admission, "USAGE_DB_PATH", db_path), \
                 patch.object(leases, "USAGE_DB_PATH", db_path):
                now = datetime.now(timezone.utc).isoformat()
                for i in range(POLICY.global_paid_action_limit):
                    telemetry._write_paid_action(
                        action_id=f"g{i}", action_type="search", request_id=None,
                        subject_type=None, subject_id=None, outcome="success",
                        started_at=now, ended_at=now, latency_ms=1.0, input_tokens=None,
                        output_tokens=None, total_tokens=None, total_call_count=1,
                        child_calls_json="[]", path=db_path,
                    )
                with pytest.raises(UsageGuardRejection) as exc_info:
                    with guard_paid_action("search", subject=None):
                        pass
                assert exc_info.value.reason_code == "global_window_limit_reached"

    def test_cache_hit_no_op_paths_remain_unguarded(self):
        """Report export and version activation (no paid work) must
        remain usable even when a session's budget is fully exhausted --
        confirms they were never wired to the guard, not that the guard
        happens to allow them."""
        with _client() as client:
            session_id, _ = _finish_curation(client)
            with patch.object(api, "generate_report_for_session", return_value={
                "findings": {"content": "f", "cited_papers": []}, "limitations": {"content": "l", "cited_papers": []},
                "future_scope": {"content": "fs", "cited_papers": []}, "skipped_papers": [],
            }):
                client.post(f"/curation/{session_id}/report")

            now = datetime.now(timezone.utc).isoformat()
            for i in range(POLICY.max_paid_actions_per_session_per_hour):
                telemetry._write_paid_action(
                    action_id=f"seed3-{i}", action_type="report_generate", request_id=None,
                    subject_type="session", subject_id=session_id, outcome="success",
                    started_at=now, ended_at=now, latency_ms=1.0, input_tokens=None,
                    output_tokens=None, total_tokens=None, total_call_count=1,
                    child_calls_json="[]", path=telemetry.USAGE_DB_PATH,
                )

            export_resp = client.get(f"/curation/{session_id}/report/export?format=markdown")
            state_resp = client.get(f"/curation/{session_id}")
        assert export_resp.status_code == 200
        assert state_resp.status_code == 200


# =====================================================================
# Boundary ordering
# =====================================================================

class TestBoundaryOrdering:
    def test_static_validation_rejects_before_admission(self):
        """A 422-triggering oversized topic must never even reach
        admission -- confirmed by exhausting the global budget AND
        sending an oversized topic in the same request: a 422, not a
        429, proves Pydantic validation ran first (FastAPI's own
        request lifecycle already guarantees this; asserted here for
        the explicit ordering claim)."""
        with _client() as client:
            now = datetime.now(timezone.utc).isoformat()
            for i in range(POLICY.global_paid_action_limit):
                telemetry._write_paid_action(
                    action_id=f"ord1-{i}", action_type="search", request_id=None,
                    subject_type=None, subject_id=None, outcome="success",
                    started_at=now, ended_at=now, latency_ms=1.0, input_tokens=None,
                    output_tokens=None, total_tokens=None, total_call_count=1,
                    child_calls_json="[]", path=telemetry.USAGE_DB_PATH,
                )
            resp = client.post("/search", json={"topic": "x" * 2001})
        assert resp.status_code == 422  # not 429 -- schema validation wins

    def test_capacity_rejection_occurs_before_admission(self):
        """The selected-paper-cap check runs INSIDE _present_and_apply_
        node before the graph would ever route to a guarded refill --
        exhausting the (unrelated) admission budget alongside an
        at-capacity session still surfaces the CAPACITY reason, proving
        capacity is checked first, structurally (not a race)."""
        from tests.test_curation_api import _select_up_to_n_papers

        with _client() as client:
            session_id, selected = _select_up_to_n_papers(client, 60)
            now = datetime.now(timezone.utc).isoformat()
            for i in range(POLICY.max_paid_actions_per_session_per_hour):
                telemetry._write_paid_action(
                    action_id=f"ord2-{i}", action_type="curation_refill", request_id=None,
                    subject_type="session", subject_id=session_id, outcome="success",
                    started_at=now, ended_at=now, latency_ms=1.0, input_tokens=None,
                    output_tokens=None, total_tokens=None, total_call_count=1,
                    child_calls_json="[]", path=telemetry.USAGE_DB_PATH,
                )
            state = client.get(f"/curation/{session_id}").json()
            over_pick = state["pending_batch"][0]["paper_id"]
            resp = client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [over_pick], "stop": False})
        assert resp.status_code == 409
        assert resp.json()["detail"]["reason_code"] == "selected_paper_limit_reached"

    def test_admission_rejection_occurs_before_lease_or_provider_work(self):
        with _client() as client:
            session_id, _ = _finish_curation(client)
            now = datetime.now(timezone.utc).isoformat()
            for i in range(POLICY.max_paid_actions_per_session_per_hour):
                telemetry._write_paid_action(
                    action_id=f"ord3-{i}", action_type="curation_chat", request_id=None,
                    subject_type="session", subject_id=session_id, outcome="success",
                    started_at=now, ended_at=now, latency_ms=1.0, input_tokens=None,
                    output_tokens=None, total_tokens=None, total_call_count=1,
                    child_calls_json="[]", path=telemetry.USAGE_DB_PATH,
                )
            with patch.object(api, "chat_turn") as mock_chat:
                resp = client.post(f"/curation/{session_id}/chat", json={"message": "hi"})
            leases_after = _rows(telemetry.USAGE_DB_PATH, "action_leases")
        assert resp.status_code == 429
        mock_chat.assert_not_called()
        assert leases_after == []  # no lease was ever acquired

    def test_lease_conflict_occurs_before_paid_action_or_provider_work(self):
        with _client() as client:
            session_id, _ = _finish_curation(client)
            release_event = threading.Event()
            entered_event = threading.Event()

            def _blocking_chat_turn(session, message, client=None, **kwargs):
                entered_event.set()
                release_event.wait(timeout=5)
                session.chat_history.append({"role": "user", "content": message})
                session.chat_history.append({"role": "assistant", "content": "ok"})
                return {"answer": "ok", "answerable": True, "cited_papers": [], "cited_web_articles": []}

            results = {}

            def _first_call():
                with patch.object(api, "chat_turn", side_effect=_blocking_chat_turn):
                    results["first"] = client.post(f"/curation/{session_id}/chat", json={"message": "one"})

            t = threading.Thread(target=_first_call)
            t.start()
            assert entered_event.wait(timeout=5)

            with patch.object(api, "chat_turn") as mock_chat_second:
                second = client.post(f"/curation/{session_id}/chat", json={"message": "two"})

            release_event.set()
            t.join(timeout=5)

        assert second.status_code == 409
        mock_chat_second.assert_not_called()  # rejected before the provider call

    def test_admitted_work_retains_one_top_level_row_with_child_calls_attached(self):
        with _client() as client:
            session_id, _ = _finish_curation(client)

            def _fake_chat_turn_with_child_call(session, message, client=None, **kwargs):
                telemetry.record_child_call("ask", "openai", latency_ms=1.0, outcome="success")
                session.chat_history.append({"role": "user", "content": message})
                session.chat_history.append({"role": "assistant", "content": "ok"})
                return {"answer": "ok", "answerable": True, "cited_papers": [], "cited_web_articles": []}

            with patch.object(api, "chat_turn", side_effect=_fake_chat_turn_with_child_call):
                resp = client.post(f"/curation/{session_id}/chat", json={"message": "hi"})

            rows = [
                r for r in _rows(telemetry.USAGE_DB_PATH, "paid_actions")
                if r["action_type"] == "curation_chat" and r["subject_id"] == session_id
            ]
        assert resp.status_code == 200
        assert len(rows) == 1
        assert rows[0]["total_call_count"] == 1
