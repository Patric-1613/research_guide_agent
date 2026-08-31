"""Usage Protection M4.3A Part E: full HTTP-level tests for POST
/curation/{session_id}/report/stream and POST /curation/{session_id}/
report/regenerate/stream, via TestClient -- complements tests/test_
report_streaming.py (domain generator, no HTTP/guard) and tests/
test_curation_report_stream_service.py (service-level lease/telemetry,
no HTTP). Reuses tests/test_curation_api.py's own `_client`/`_finish_
curation`/`_paper` fixtures (same cross-test-file import precedent
tests/test_red_team_bypass.py already established for chat), and
tests/test_curation_chat_stream_api.py's own established convention of
a plain `client.post(...)` (TestClient drives the ASGI app to
completion synchronously) followed by reading `resp.text` -- never
`client.stream(...)`.
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research_agent.api as api
import research_agent.telemetry as telemetry
from research_agent import report
from research_agent.config import get_usage_policy
from research_agent.usage_guard import UsageGuardRejection
from tests.test_curation_api import _client, _finish_curation, _paper, _ranked


def _fake_report(pick_ids: list[str]) -> dict:
    """Same OLD-SHAPE fixture test_curation_api.py's own sync report
    test uses -- generate_report_for_session/regenerate_report_with_new_
    sources are patched directly to return this, so these HTTP-level
    tests exercise the real router -> service -> streaming domain ->
    persistence chain without needing a configured mock OpenAI client at
    all (refinement_mode is never requested in these tests, so refine_
    report_if_requested's own zero-extra-calls passthrough means nothing
    else ever touches the client either)."""
    return {
        "findings": {"content": "f", "cited_papers": [_paper(pick_ids[0], "Paper 0")]},
        "limitations": {"content": "l", "cited_papers": []},
        "future_scope": {"content": "fs", "cited_papers": []},
        "skipped_papers": [],
    }


def test_stream_generate_returns_sse_content_type_and_full_lifecycle():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        with patch.object(report, "generate_report_for_session", return_value=_fake_report(pick_ids)):
            resp = client.post(f"/curation/{session_id}/report/stream", json={})

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert resp.headers.get("cache-control", "").startswith("no-cache")

        body = resp.text
        assert "event: started" in body
        assert '"phase":"generating"' in body
        assert '"phase":"saving"' in body
        assert "event: completed" in body
        assert "event: done" in body
        assert "event: error" not in body

        # canonical state genuinely persisted -- a separate GET sees it.
        state = client.get(f"/curation/{session_id}").json()
        assert state["report"] is not None
        assert len(state["report_versions"]) == 1
        assert state["report_versions"][0]["generation_reason"] == "initial"
        assert state["active_report_version_id"] == state["report_versions"][0]["version_id"]


def test_stream_generate_cache_hit_returns_started_completed_done_only():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        with patch.object(report, "generate_report_for_session", return_value=_fake_report(pick_ids)):
            client.post(f"/curation/{session_id}/report/stream", json={})

        # Second call: session.report already exists -- cache hit, no
        # phase events, and generate_report_for_session must not be
        # patched/called at all this time -- a real, unpatched call
        # would explode against the test client's bare MagicMock OpenAI
        # client if it were ever actually reached, proving the cache
        # branch skipped it entirely.
        resp2 = client.post(f"/curation/{session_id}/report/stream", json={})

        assert resp2.status_code == 200
        body2 = resp2.text
        assert "event: started" in body2
        assert '"phase"' not in body2
        assert "event: completed" in body2
        assert "event: done" in body2


def test_stream_regenerate_returns_sse_and_appends_a_new_version():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        with patch.object(report, "generate_report_for_session", return_value=_fake_report(pick_ids)):
            client.post(f"/curation/{session_id}/report/stream", json={})

        with patch.object(report, "regenerate_report_with_new_sources", return_value=_fake_report(pick_ids)):
            resp2 = client.post(f"/curation/{session_id}/report/regenerate/stream", json={})

        assert resp2.status_code == 200
        body2 = resp2.text
        assert "event: completed" in body2
        assert "event: done" in body2
        state = client.get(f"/curation/{session_id}").json()
        assert len(state["report_versions"]) == 2
        assert state["report_versions"][1]["generation_reason"] == "regenerate"
        assert state["active_report_version_id"] == state["report_versions"][1]["version_id"]


def test_stream_generate_unknown_session_returns_404():
    with _client() as client:
        resp = client.post("/curation/does-not-exist/report/stream", json={})
        assert resp.status_code == 404
        # A synchronous ServiceError raised before the stream starts is
        # mapped by the centralized handler -- a plain JSON body, not an
        # SSE error event.
        assert resp.json() == {"detail": "session_id not found"}
        assert resp.headers["content-type"] == "application/json"


def test_stream_regenerate_unknown_session_returns_404():
    with _client() as client:
        resp = client.post("/curation/does-not-exist/report/regenerate/stream", json={})
        assert resp.status_code == 404
        assert resp.json() == {"detail": "session_id not found"}
        assert resp.headers["content-type"] == "application/json"


def test_stream_generate_malformed_request_returns_422():
    with _client() as client:
        session_id, _pick_ids = _finish_curation(client)
        resp = client.post(f"/curation/{session_id}/report/stream", json={"report_template": "not-a-real-template"})
        assert resp.status_code == 422


def test_stream_regenerate_before_any_report_exists_returns_400():
    with _client() as client:
        session_id, _pick_ids = _finish_curation(client)
        resp = client.post(f"/curation/{session_id}/report/regenerate/stream", json={})
        assert resp.status_code == 400
        assert "no existing report" in resp.json()["detail"].lower()


def test_stream_generate_before_stage_ready_returns_400():
    with _client() as client:
        # curate stage, never stopped -- report generation is not ready yet.
        papers = [_paper(f"p{i}", f"Paper {i}") for i in range(4)]
        with patch.object(api, "build_candidate_pool", return_value=papers), \
             patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
            start_body = client.post("/curation/start", json={"topic": "t", "target_count": 2}).json()
        session_id = start_body["session_id"]

        resp = client.post(f"/curation/{session_id}/report/stream", json={})
        assert resp.status_code == 400
        assert "not ready for report synthesis" in resp.json()["detail"].lower()


def test_stream_generate_rate_limit_returns_429_before_headers_or_provider_work():
    with _client() as client:
        session_id, _pick_ids = _finish_curation(client)
        policy = get_usage_policy()
        for i in range(policy.max_paid_actions_per_session_per_hour):
            telemetry._write_paid_action(
                action_id=f"seed-{i}-{os.urandom(4).hex()}", action_type="search", request_id=None,
                subject_type="session", subject_id=session_id, outcome="success",
                started_at=datetime.now(timezone.utc).isoformat(), ended_at=datetime.now(timezone.utc).isoformat(),
                latency_ms=1.0, input_tokens=None, output_tokens=None, total_tokens=None,
                total_call_count=1, child_calls_json="[]", path=telemetry.USAGE_DB_PATH,
            )

        with patch.object(report, "generate_report_for_session") as mock_gen:
            resp = client.post(f"/curation/{session_id}/report/stream", json={})
            mock_gen.assert_not_called()

        assert resp.status_code == 429
        assert resp.headers.get("content-type", "").startswith("application/json")
        assert "retry-after" in {k.lower() for k in resp.headers.keys()}


def test_stream_generate_concurrency_conflict_returns_409():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)

        # Hold the lease open across the second request by patching
        # generate_report_for_session with a function that blocks until
        # released -- simplest deterministic way to prove a genuinely
        # held lease rejects a second concurrent stream start.
        started = threading.Event()
        release = threading.Event()

        def _slow_generate(*args, **kwargs):
            started.set()
            assert release.wait(timeout=5)
            return _fake_report(pick_ids)

        results = {}

        def _first_request():
            with patch.object(report, "generate_report_for_session", _slow_generate):
                resp = client.post(f"/curation/{session_id}/report/stream", json={})
                results["first_status"] = resp.status_code

        t = threading.Thread(target=_first_request)
        t.start()
        assert started.wait(timeout=5)

        resp2 = client.post(f"/curation/{session_id}/report/stream", json={})
        assert resp2.status_code == 409

        release.set()
        t.join(timeout=5)
        assert results.get("first_status") == 200


def test_stream_generate_storage_unavailable_returns_503():
    with _client() as client:
        session_id, _pick_ids = _finish_curation(client)
        with patch("research_agent.usage_guard.check_admission") as mock_check:
            mock_check.side_effect = UsageGuardRejection("usage_protection_unavailable")
            resp = client.post(f"/curation/{session_id}/report/stream", json={})

        assert resp.status_code == 503


def test_existing_sync_report_endpoints_still_work_unchanged():
    with _client() as client:
        session_id, pick_ids = _finish_curation(client)
        # The SYNC service functions (services/curation_report_service.py's
        # get_or_create_report/regenerate_report) call api.generate_report_
        # for_session/api.regenerate_report_with_new_sources -- a SEPARATE
        # name binding from research_agent.report's own module attribute
        # (api.py did `from research_agent.report import ...` at its own
        # import time), so these must be patched on `api`, matching test_
        # curation_api.py's own established convention -- unlike this
        # file's streaming tests, which patch `report.X` directly since
        # the streaming domain module references it via `from research_
        # agent import report; report.X(...)`.
        with patch.object(api, "generate_report_for_session", return_value=_fake_report(pick_ids)):
            resp = client.post(f"/curation/{session_id}/report")
        assert resp.status_code == 200
        assert resp.json()["findings"]["content"] == "f"

        with patch.object(api, "regenerate_report_with_new_sources", return_value=_fake_report(pick_ids)):
            resp2 = client.post(f"/curation/{session_id}/report/regenerate")
        assert resp2.status_code == 200
