"""Usage Protection M4.2A Part F: end-to-end tests for
`POST /curation/{session_id}/chat/stream`, driven through the real
FastAPI app via TestClient (same isolation/fixture conventions as
tests/test_curation_api.py -- fresh temp SQLite files per test, real
admission/lease/telemetry queries against them, no real network call).

Covers: preflight/HTTP contracts (content type, 422/404/429/409/503,
Retry-After, rejected-preflight does no work), event lifecycle over
the wire, persistence parity with the existing sync endpoint, and a
regression check that the existing sync endpoint is untouched.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research_agent.admission as admission
import research_agent.api as api
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.chat_streaming import AnswerCompleted, StreamedChatAnswer
from research_agent.config import get_usage_policy
from research_agent.usage_guard import UsageGuardRejection
from tests.test_curation_api import _client, _finish_curation, _paper


async def _fake_stream_chat_answer(*args, **kwargs):
    yield AnswerCompleted(result=StreamedChatAnswer(
        answer="Per [Web 1], streamed answer.", answerable=True, cited_papers=[], cited_web_articles=[],
    ))


def _parse_sse_events(text: str) -> list[tuple[str, str]]:
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = block.split("\n")
        event_type = next((l[len("event:"):].strip() for l in lines if l.startswith("event:")), None)
        data = next((l[len("data:"):].strip() for l in lines if l.startswith("data:")), None)
        if event_type is not None:
            events.append((event_type, data))
    return events


def _patch_no_real_retrieval():
    """`_finish_curation` always picks >=1 real paper (curation cannot
    reach stage="synthesize" otherwise) -- `_client()`'s own MagicMock
    OpenAI client has no real embeddings behind it, so letting the true
    streaming path's real `qa.prepare_qa_turn` reach the real Chroma
    collection with that mock client would hit a genuine embedding-shape
    error, not a representative test failure. Patched at qa.py's own
    call-site names (its `from ... import` binding), matching this
    project's own established convention for patching a module's
    imported names rather than the defining module's."""
    return patch.multiple(
        "research_agent.qa",
        embed_and_index_papers=lambda *a, **k: None,
        semantic_search=lambda *a, **k: [],
    )


def test_stream_endpoint_valid_request_returns_sse_content_type_and_full_lifecycle():
    with _client() as client:
        session_id, _pick_ids = _finish_curation(client)

        with patch("research_agent.curation_chat_streaming.stream_chat_answer", _fake_stream_chat_answer), \
             patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
             _patch_no_real_retrieval():
            resp = client.post(f"/curation/{session_id}/chat/stream", json={"message": "what does the web say?"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers.get("cache-control", "").startswith("no-cache")

    events = _parse_sse_events(resp.text)
    types = [t for t, _ in events]
    assert types[0] == "started"
    assert types[-2:] == ["completed", "done"]
    assert "error" not in types
    # Retrieval is forced empty above (no real papers/web articles
    # survive it), so this exercises the zero-delta, no-sources-empty
    # early-return branch's real HTTP/SSE plumbing -- the true
    # incremental-delta branch's own event ordering is already proven
    # directly in tests/test_curation_chat_streaming.py, which does not
    # depend on a real Chroma-backed embedding call.
    assert "phase" in types
    completed_data = next(data for t, data in events if t == "completed")
    assert '"answerable":false' in completed_data


def test_malformed_request_returns_422_before_any_session_work():
    with _client() as client:
        resp = client.post("/curation/some-session/chat/stream", json={})
    assert resp.status_code == 422


def test_unknown_session_returns_404():
    with _client() as client:
        resp = client.post("/curation/does-not-exist/chat/stream", json={"message": "hi"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "session_id not found"


def test_rate_limit_returns_429_before_headers_or_provider_work():
    with _client() as client:
        session_id, _pick_ids = _finish_curation(client)

        policy = get_usage_policy()
        now = datetime.now(timezone.utc).isoformat()
        for i in range(policy.max_paid_actions_per_session_per_hour):
            telemetry._write_paid_action(
                action_id=f"seed-{i}", action_type="curation_chat", request_id=None,
                subject_type="session", subject_id=session_id, outcome="success",
                started_at=now, ended_at=now, latency_ms=1.0,
                input_tokens=None, output_tokens=None, total_tokens=None,
                total_call_count=1, child_calls_json="[]",
            )

        with patch("research_agent.curation_chat_streaming.stream_chat_answer") as mock_stream:
            resp = client.post(f"/curation/{session_id}/chat/stream", json={"message": "hi"})
            mock_stream.assert_not_called()

    assert resp.status_code == 429
    assert resp.headers.get("retry-after") is not None
    assert resp.json()["detail"]["reason_code"] == "session_hourly_limit_reached"


def test_concurrency_conflict_returns_409_before_headers_or_provider_work():
    with _client() as client:
        session_id, _pick_ids = _finish_curation(client)

        # A held lease for the SAME subject/group the guard itself uses.
        decision = leases.acquire_lease("session", session_id)
        assert decision.acquired

        try:
            with patch("research_agent.curation_chat_streaming.stream_chat_answer") as mock_stream:
                resp = client.post(f"/curation/{session_id}/chat/stream", json={"message": "hi"})
                mock_stream.assert_not_called()
        finally:
            leases.release_lease("session", session_id, leases.EXPENSIVE_ACTION_GROUP, decision.token)

    assert resp.status_code == 409
    assert resp.json()["detail"]["reason_code"] == "action_in_progress"


def test_storage_unavailable_returns_503_before_headers_or_provider_work():
    with _client() as client:
        session_id, _pick_ids = _finish_curation(client)

        with patch("research_agent.usage_guard.check_admission", side_effect=UsageGuardRejection("usage_protection_unavailable")), \
             patch("research_agent.curation_chat_streaming.stream_chat_answer") as mock_stream:
            resp = client.post(f"/curation/{session_id}/chat/stream", json={"message": "hi"})
            mock_stream.assert_not_called()

    assert resp.status_code == 503
    assert resp.json()["detail"]["reason_code"] == "usage_protection_unavailable"


def test_rejected_preflight_creates_no_paid_action_row():
    with _client() as client:
        session_id, _pick_ids = _finish_curation(client)

        decision = leases.acquire_lease("session", session_id)
        assert decision.acquired
        try:
            client.post(f"/curation/{session_id}/chat/stream", json={"message": "hi"})
        finally:
            leases.release_lease("session", session_id, leases.EXPENSIVE_ACTION_GROUP, decision.token)

        import sqlite3

        conn = sqlite3.connect(telemetry.USAGE_DB_PATH)
        try:
            rows = conn.execute(
                "SELECT * FROM paid_actions WHERE subject_id = ? AND outcome != 'success'", (session_id,),
            ).fetchall()
        finally:
            conn.close()
        # The rejected attempt itself never opens paid_action at all --
        # zero rows for this subject from the rejected call (the earlier
        # successful lease acquisition above isn't a paid_action row).
        assert rows == []


def test_stream_persists_exactly_one_exchange_matching_completed_payload():
    with _client() as client:
        session_id, _pick_ids = _finish_curation(client)

        with patch("research_agent.curation_chat_streaming.stream_chat_answer", _fake_stream_chat_answer), \
             patch("research_agent.qa._classify_non_substantive", return_value=(False, None, 0.0)), \
             _patch_no_real_retrieval():
            resp = client.post(f"/curation/{session_id}/chat/stream", json={"message": "what does the web say?"})

        state_resp = client.get(f"/curation/{session_id}")

    events = _parse_sse_events(resp.text)
    completed_data = next(data for t, data in events if t == "completed")

    chat_history = state_resp.json()["chat_history"]
    assert len(chat_history) == 2
    assert chat_history[0]["role"] == "user"
    assert chat_history[1]["role"] == "assistant"
    # The persisted assistant answer must exactly equal the completed
    # event's own answer text -- the core persistence-parity guarantee.
    import json as _json

    assert chat_history[1]["content"] == _json.loads(completed_data)["answer"]


def test_existing_sync_chat_endpoint_still_works_unchanged():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        fake_result = {
            "answer": "Per [Paper 1], X is true.", "answerable": True,
            "cited_papers": [_paper(pick_ids[0], "Paper 0")], "cited_web_articles": [],
        }

        def _fake_chat_turn(session, message, client=None, **kwargs):
            session.chat_history.append({"role": "user", "content": message})
            session.chat_history.append({"role": "assistant", "content": fake_result["answer"]})
            return fake_result

        with patch.object(api, "chat_turn", side_effect=_fake_chat_turn):
            resp = client.post(f"/curation/{session_id}/chat", json={"message": "what does paper 0 say?"})

    assert resp.status_code == 200
    assert resp.json()["answer"] == "Per [Paper 1], X is true."
    assert resp.headers["content-type"].startswith("application/json")
