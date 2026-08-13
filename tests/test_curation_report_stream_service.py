"""Usage Protection M4.3A Part F: integration tests for services/
curation_report_service.py's `stream_generate_report`/`stream_
regenerate_report` -- calling the SERVICE functions directly (not
through HTTP/TestClient), consuming the returned `StreamingResponse`'s
own `body_iterator` manually. Complements tests/test_curation_report_
stream_api.py (full HTTP/preflight contracts) and tests/test_report_
streaming.py (the domain generator in isolation, no guard/telemetry at
all) by proving the real lease + `telemetry.paid_action` lifecycle end
to end through the actual service/router-adjacent code path, including
cancellation and the cache-hit-bypasses-the-guard-entirely contract.

Same isolation/fingerprint conventions as tests/test_curation_chat_
stream_service.py.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import research_agent.admission as admission
import research_agent.api as api
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.api_app.schemas import CurationGenerateReportRequest, CurationRegenerateReportRequest
from research_agent.curation_report_streaming import HandledReportStreamFailure
from research_agent.curation_session import save_curation_session
from research_agent.qa import sqlite_checkpointer
from research_agent.query_expansion import PaperPoolSession
from research_agent.report import ANALYTICAL_SECTION_NAMES, REPORT_SECTION_DEFINITIONS, _build_references_and_renumber
from research_agent.report_streaming import build_started_event
from research_agent.schema import Paper
from research_agent.services.curation_report_service import stream_generate_report, stream_regenerate_report
from research_agent.telemetry import init_usage_db
from tests._usage_db_fingerprint import fingerprint_usage_db
from tests.test_report import _analytical_parsed, _build_report_schema, _mock_parsed_response, _project_legacy_fields, _sections_list

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


def _rows(db_path, table):
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}").fetchall()]
    finally:
        conn.close()


def _paper(paper_id: str, title: str) -> Paper:
    return Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract=f"Abstract for {title}.", url=f"http://arxiv.org/abs/{paper_id}",
        doi=None, citation_count=None, source="arxiv", paper_id=paper_id,
    )


@pytest.fixture
def session_setup(tmp_path):
    cp_db_path = Path(tmp_path) / "checkpoints.sqlite"
    session_id = "svc-report-test-session"
    p1 = _paper("1111", "Paper One")
    session = PaperPoolSession(topic="AI safety", stage="synthesize", selected_papers=[p1])
    with sqlite_checkpointer(cp_db_path) as cp:
        save_curation_session(session, session_id, cp)
        yield session_id, cp, p1


@pytest.fixture
def cached_session_setup(tmp_path):
    """A session that already has a (fully-projected) report -- the
    cache-hit path for stream_generate_report."""
    cp_db_path = Path(tmp_path) / "checkpoints.sqlite"
    session_id = "svc-report-cached-session"
    p1 = _paper("1111", "Paper One")
    report = _project_legacy_fields(_analytical_parsed_draft(p1))
    report["sections"] = _sections_list(report)
    report["report_template"] = "analytical"
    session = PaperPoolSession(
        topic="AI safety", stage="synthesize", selected_papers=[p1], report=report,
        report_versions=[{
            "version_id": "v1", "version_number": 1, "created_at": None,
            "report_template": "analytical", "generation_reason": "initial", "report": report,
        }],
        active_report_version_id="v1",
    )
    with sqlite_checkpointer(cp_db_path) as cp:
        save_curation_session(session, session_id, cp)
        yield session_id, cp


def _analytical_parsed_draft(p1: Paper) -> dict:
    sections_out = {
        key: {"content": f"{key} text [Paper 1].", "cited_papers": [p1]}
        for key in ANALYTICAL_SECTION_NAMES
    }
    return _build_references_and_renumber({**sections_out, "skipped_papers": []}, ANALYTICAL_SECTION_NAMES)


async def _consume(response) -> list[str]:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return chunks


def _mock_client():
    from unittest.mock import MagicMock

    client = MagicMock()
    client.chat.completions.parse.return_value = _mock_parsed_response(
        _analytical_parsed(_build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)),
    )
    return client


def test_generate_success_records_paid_action_and_releases_lease(usage_db_path, session_setup):
    session_id, cp, _p1 = session_setup
    req = CurationGenerateReportRequest()

    async def scenario():
        response = stream_generate_report(session_id, req, cp)
        return await _consume(response)

    with patch.object(api, "_state", {"client": _mock_client()}):
        chunks = asyncio.run(scenario())

    assert any("event: started" in c for c in chunks)
    assert any("event: completed" in c for c in chunks)
    assert any("event: done" in c for c in chunks)

    rows = _rows(usage_db_path, "paid_actions")
    assert len(rows) == 1
    assert rows[0]["action_type"] == "report_generate"
    assert rows[0]["outcome"] == "success"
    assert len(_rows(usage_db_path, "action_leases")) == 0


def test_cache_hit_creates_no_paid_action_row_and_no_lease(usage_db_path, cached_session_setup):
    session_id, cp = cached_session_setup
    req = CurationGenerateReportRequest()

    async def scenario():
        response = stream_generate_report(session_id, req, cp)
        return await _consume(response)

    # A client that would explode if ever actually called -- proves the
    # cache-hit path never touches it.
    with patch.object(api, "_state", {"client": None}):
        chunks = asyncio.run(scenario())

    assert any("event: started" in c for c in chunks)
    assert any("event: completed" in c for c in chunks)
    assert any("event: done" in c for c in chunks)
    assert len(_rows(usage_db_path, "paid_actions")) == 0
    assert len(_rows(usage_db_path, "action_leases")) == 0


def test_cache_hit_bypasses_admission_even_when_budget_exhausted(usage_db_path, cached_session_setup):
    """A cache hit must succeed even if the session's own budget is
    already exhausted -- it never reaches check_admission at all."""
    from datetime import datetime, timezone

    from research_agent.config import get_usage_policy

    session_id, cp = cached_session_setup
    policy = get_usage_policy()
    for i in range(policy.max_paid_actions_per_session_per_hour):
        telemetry._write_paid_action(
            action_id=f"seed-{i}-{os.urandom(4).hex()}", action_type="search", request_id=None,
            subject_type="session", subject_id=session_id, outcome="success",
            started_at=datetime.now(timezone.utc).isoformat(), ended_at=datetime.now(timezone.utc).isoformat(),
            latency_ms=1.0, input_tokens=None, output_tokens=None, total_tokens=None,
            total_call_count=1, child_calls_json="[]", path=usage_db_path,
        )
    req = CurationGenerateReportRequest()

    async def scenario():
        response = stream_generate_report(session_id, req, cp)
        return await _consume(response)

    with patch.object(api, "_state", {"client": None}):
        chunks = asyncio.run(scenario())

    assert any("event: completed" in c for c in chunks)


def test_regenerate_success_records_paid_action_and_releases_lease(usage_db_path, session_setup):
    session_id, cp, p1 = session_setup
    # regenerate requires an existing report -- seed one via a real generate first.
    with patch.object(api, "_state", {"client": _mock_client()}):
        asyncio.run(_consume(stream_generate_report(session_id, CurationGenerateReportRequest(), cp)))

    req = CurationRegenerateReportRequest()

    async def scenario():
        response = stream_regenerate_report(session_id, req, cp)
        return await _consume(response)

    with patch.object(api, "_state", {"client": _mock_client()}):
        chunks = asyncio.run(scenario())

    assert any("event: completed" in c for c in chunks)
    rows = _rows(usage_db_path, "paid_actions")
    # 2 rows total: the seeding generate call above, plus this regenerate.
    assert len(rows) == 2
    regenerate_rows = [r for r in rows if r["action_type"] == "report_regenerate"]
    assert len(regenerate_rows) == 1
    assert regenerate_rows[0]["outcome"] == "success"
    assert len(_rows(usage_db_path, "action_leases")) == 0


def test_cancellation_records_cancelled_outcome_and_releases_lease(usage_db_path, session_setup):
    session_id, cp, _p1 = session_setup
    req = CurationGenerateReportRequest()

    async def scenario():
        response = stream_generate_report(session_id, req, cp)
        with pytest.raises(asyncio.CancelledError):
            await _consume(response)

    async def _raise_cancelled(*args, **kwargs):
        yield build_started_event()
        raise asyncio.CancelledError()

    with patch.object(api, "_state", {"client": _mock_client()}), \
         patch("research_agent.services.curation_report_service.stream_generate_report_turn", _raise_cancelled):
        asyncio.run(scenario())

    rows = _rows(usage_db_path, "paid_actions")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "cancelled"
    assert len(_rows(usage_db_path, "action_leases")) == 0


def _drive_with_handled_failure(session_id, cp, req, reason_code, message):
    from research_agent.curation_report_streaming import HandledReportStreamFailure

    async def _raise_handled_failure(*args, **kwargs):
        yield build_started_event()
        raise HandledReportStreamFailure(reason_code, message)

    async def scenario():
        with patch.object(api, "_state", {"client": _mock_client()}), \
             patch("research_agent.services.curation_report_service.stream_generate_report_turn", _raise_handled_failure):
            response = stream_generate_report(session_id, req, cp)
            return await _consume(response)

    return asyncio.run(scenario())


def test_handled_failure_records_top_level_outcome_error(usage_db_path, session_setup):
    session_id, cp, _p1 = session_setup
    chunks = _drive_with_handled_failure(
        session_id, cp, CurationGenerateReportRequest(), "provider_error", "The model provider returned an error.",
    )

    rows = _rows(usage_db_path, "paid_actions")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "error"
    assert len(_rows(usage_db_path, "action_leases")) == 0


def test_handled_failure_yields_safe_error_then_done_never_completed(usage_db_path, session_setup):
    session_id, cp, _p1 = session_setup
    chunks = _drive_with_handled_failure(
        session_id, cp, CurationGenerateReportRequest(), "provider_error", "The model provider returned an error.",
    )
    joined = "".join(chunks)

    assert "event: started" in joined
    assert "event: error" in joined
    assert '"reason_code":"provider_error"' in joined
    assert "event: done" in joined
    assert "event: completed" not in joined
    assert "Traceback" not in joined


def test_lease_remains_held_through_persistence_and_releases_only_after_cancellation_settles(usage_db_path, session_setup):
    """Service-layer mirror of test_report_streaming.py's own domain-level
    proof, plus the LEASE specifically: a real, controllable, slow save
    runs in its own OS thread; the consuming Task is cancelled WHILE
    that save is genuinely still in flight; the lease must still be held
    at that moment, the save must still run to full completion, `completed`
    must never be emitted, and only once the save has settled may the
    lease actually be released."""
    session_id, cp, _p1 = session_setup
    req = CurationGenerateReportRequest()

    started = threading.Event()
    proceed = threading.Event()
    finished = threading.Event()

    def _slow_save(session, sid, checkpointer):
        started.set()
        assert proceed.wait(timeout=5), "test driver never released the save -- deadlock"
        save_curation_session(session, sid, checkpointer)
        finished.set()

    collected: list[str] = []

    async def consume():
        response = stream_generate_report(session_id, req, cp)
        async for chunk in response.body_iterator:
            collected.append(chunk)

    async def scenario():
        with patch.object(api, "_state", {"client": _mock_client()}), \
             patch("research_agent.curation_report_streaming.save_curation_session", _slow_save):
            task = asyncio.create_task(consume())

            for _ in range(2000):
                if started.is_set():
                    break
                await asyncio.sleep(0.005)
            else:
                raise AssertionError("save never started -- test setup bug")

            task.cancel()
            await asyncio.sleep(0.05)
            assert not finished.is_set(), "the save settled before the assertion window -- test is not exercising the race"
            assert len(_rows(usage_db_path, "action_leases")) == 1, "lease released while persistence was still in flight"

            proceed.set()

            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())

    assert finished.is_set()
    assert len(_rows(usage_db_path, "action_leases")) == 0
    rows = _rows(usage_db_path, "paid_actions")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "cancelled"
    assert not any("event: completed" in c for c in collected)


def test_real_usage_db_path_untouched():
    assert fingerprint_usage_db(_REAL_USAGE_DB_PATH) == _REAL_USAGE_DB_FINGERPRINT_BEFORE
