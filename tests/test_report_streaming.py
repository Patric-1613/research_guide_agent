"""Usage Protection M4.3A Part D: tests for research_agent/curation_
report_streaming.py -- the production report-generation/regeneration
streaming domain service. No admission/lease/HTTP concerns here (that's
tests/test_curation_report_stream_api.py) -- this file drives `stream_
generate_report_turn`/`stream_regenerate_report_turn`/`stream_cached_
report` directly, with a MagicMock OpenAI client (no real network call),
reusing test_report.py's own established fixtures (`_paper`, `_mock_
parsed_response`, `_analytical_parsed`, `_evaluation`, `_clean_
analytical_draft`) so the REAL, unmodified generate_report_for_session/
regenerate_report_with_new_sources/refine_report_if_requested/evaluate_
report/revise_report pipeline runs end to end -- only the OpenAI call
boundary (`client.chat.completions.parse`) is faked, exactly like
test_report.py's own tests, so call-count assertions here prove genuine
integration, not a re-description of a mock's own configuration.

Same `asyncio.run()`-per-test convention as tests/test_chat_streaming.py/
tests/test_curation_chat_streaming.py (no pytest-asyncio dependency).
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from research_agent.curation_report_streaming import (
    HandledReportStreamFailure,
    stream_cached_report,
    stream_generate_report_turn,
    stream_regenerate_report_turn,
)
from research_agent.query_expansion import PaperPoolSession
from research_agent.report import ANALYTICAL_SECTION_NAMES, GENERATION_REASON_INITIAL, GENERATION_REASON_REGENERATE
from research_agent.schema import WebArticle
from tests.test_report import (
    REPORT_SECTION_DEFINITIONS,
    _analytical_parsed,
    _build_report_schema,
    _clean_analytical_draft,
    _evaluation,
    _mock_parsed_response,
    _paper,
    _project_legacy_fields,
    _sections_list,
)


def _full_analytical_parsed(schema, paper_id: str, **overrides):
    """Every section gets non-empty, cited content -- unlike test_report.
    py's own `_analytical_parsed` (which defaults every un-overridden
    section to empty content, deliberately, for tests that only care
    about ONE section), a generation response used as the DRAFT input to
    a real `evaluate_report` call must not itself trip `_deterministic_
    report_checks`' own "missing or empty required section" hard-issue
    gate -- that gate unconditionally forces `needs_revision=True`
    regardless of what the mocked evaluator LLM response says, which
    would silently turn a "no revision needed" test into a "revision
    needed" one and desync it from its own configured mock side_effect
    list (observed directly: an exhausted `side_effect` list raises
    `StopIteration` from inside a worker thread, which corrupts asyncio's
    own Future machinery rather than surfacing as an ordinary test
    failure -- this helper exists specifically to avoid that trap)."""
    section_cls = schema.model_fields["executive_summary"].annotation
    kwargs = {}
    for key in ANALYTICAL_SECTION_NAMES:
        if key in overrides:
            kwargs[key] = overrides[key]
        else:
            kwargs[key] = section_cls(content=f"{key} text [Paper 1].", cited_paper_ids=[paper_id])
    return schema(**kwargs)


def _session(**kwargs) -> PaperPoolSession:
    defaults = dict(topic="AI safety", stage="synthesize")
    defaults.update(kwargs)
    return PaperPoolSession(**defaults)


def _run(coro):
    return asyncio.run(coro)


async def _collect(gen) -> list[tuple[str, dict]]:
    events = []
    async for event in gen:
        events.append((event.type, event.data))
    return events


async def _collect_until_raise(gen):
    events = []
    try:
        async for event in gen:
            events.append((event.type, event.data))
    except (HandledReportStreamFailure, asyncio.CancelledError) as exc:
        return events, exc
    raise AssertionError("expected the generator to raise, but it completed normally")


def _types(events):
    return [t for t, _ in events]


# --- event sequences ------------------------------------------------

@patch("research_agent.curation_report_streaming.save_curation_session")
def test_generate_refinement_off_sequence(mock_save):
    p1 = _paper("1111", "Paper One")
    session = _session(selected_papers=[p1])
    client = MagicMock()
    client.chat.completions.parse.return_value = _mock_parsed_response(
        _analytical_parsed(_build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)),
    )

    events = _run(_collect(stream_generate_report_turn(session, "s1", MagicMock(), client, None, None)))

    assert _types(events) == ["started", "phase", "phase", "completed", "done"]
    phases = [data["phase"] for t, data in events if t == "phase"]
    assert phases == ["generating", "saving"]
    client.chat.completions.parse.assert_called_once()
    mock_save.assert_called_once()
    completed = next(data for t, data in events if t == "completed")
    assert completed["report_template"] == "analytical"


@patch("research_agent.curation_report_streaming.save_curation_session")
def test_generate_refinement_evaluation_no_revision_sequence(mock_save):
    p1 = _paper("1111", "Paper One")
    session = _session(selected_papers=[p1])
    client = MagicMock()
    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    client.chat.completions.parse.side_effect = [
        _mock_parsed_response(_full_analytical_parsed(schema, "1111")),
        _mock_parsed_response(_evaluation(overall_score=88, needs_revision=False)),
    ]

    events = _run(_collect(stream_generate_report_turn(session, "s1", MagicMock(), client, None, "single")))

    assert _types(events) == ["started", "phase", "phase", "phase", "completed", "done"]
    phases = [data["phase"] for t, data in events if t == "phase"]
    assert phases == ["generating", "evaluating", "saving"]
    assert client.chat.completions.parse.call_count == 2


@patch("research_agent.curation_report_streaming.save_curation_session")
def test_generate_refinement_evaluation_plus_revision_sequence(mock_save):
    p1 = _paper("1111", "Paper One")
    session = _session(selected_papers=[p1])
    client = MagicMock()
    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    client.chat.completions.parse.side_effect = [
        _mock_parsed_response(_analytical_parsed(schema)),
        _mock_parsed_response(_evaluation(overall_score=40, needs_revision=True, revision_instructions="improve")),
        _mock_parsed_response(_analytical_parsed(
            schema, thematic_findings=section_cls(content="Better [Paper 1].", cited_paper_ids=["1111"]),
        )),
    ]

    events = _run(_collect(stream_generate_report_turn(session, "s1", MagicMock(), client, None, "single")))

    assert _types(events) == ["started", "phase", "phase", "phase", "phase", "completed", "done"]
    phases = [data["phase"] for t, data in events if t == "phase"]
    assert phases == ["generating", "evaluating", "revising", "saving"]
    assert client.chat.completions.parse.call_count == 3
    completed = next(data for t, data in events if t == "completed")
    thematic_findings_section = next(s for s in completed["sections"] if s["key"] == "thematic_findings")
    assert thematic_findings_section["content"] == "Better [1]."


def test_cache_hit_sequence_no_phases_no_provider_call():
    p1 = _paper("1111", "Paper One")
    # _report_to_out (called by stream_cached_report) requires a FULLY
    # projected report dict (either a real "sections" list or the legacy
    # findings/limitations/future_scope keys) -- _clean_analytical_draft
    # alone (test_report.py's own R4-refinement fixture, fed directly to
    # refine_report_if_requested/evaluate_report/revise_report, none of
    # which need either) stops one step short of that; apply the same
    # _project_legacy_fields -> _sections_list chain generate_report's
    # own top-level flow always applies before returning.
    existing_report = _project_legacy_fields(_clean_analytical_draft([p1]))
    existing_report["sections"] = _sections_list(existing_report)
    session = _session(
        selected_papers=[p1], report=existing_report,
        report_versions=[{
            "version_id": "v1", "version_number": 1, "created_at": None,
            "report_template": "analytical", "generation_reason": GENERATION_REASON_INITIAL, "report": existing_report,
        }],
        active_report_version_id="v1",
    )

    events = _run(_collect(stream_cached_report(session)))

    assert _types(events) == ["started", "completed", "done"]
    completed = next(data for t, data in events if t == "completed")
    assert completed["report_template"] == "analytical"


# --- handled failures -------------------------------------------------

def test_handled_generation_failure_raises_handled_report_stream_failure_never_completed():
    session = _session(selected_papers=[_paper("1111", "Paper One")])
    client = MagicMock()
    client.chat.completions.parse.side_effect = RuntimeError("provider exploded")

    events, exc = _run(_collect_until_raise(stream_generate_report_turn(session, "s1", MagicMock(), client, None, None)))

    assert isinstance(exc, HandledReportStreamFailure)
    assert exc.reason_code == "provider_error"
    assert "provider exploded" not in exc.message
    assert "completed" not in _types(events)


def test_handled_evaluation_failure_raises_handled_report_stream_failure():
    p1 = _paper("1111", "Paper One")
    session = _session(selected_papers=[p1])
    client = MagicMock()
    client.chat.completions.parse.side_effect = [
        _mock_parsed_response(_analytical_parsed(_build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS))),
        RuntimeError("evaluator exploded"),
    ]

    events, exc = _run(_collect_until_raise(stream_generate_report_turn(session, "s1", MagicMock(), client, None, "single")))

    assert isinstance(exc, HandledReportStreamFailure)
    assert exc.reason_code == "provider_error"
    assert "completed" not in _types(events)


@patch("research_agent.curation_report_streaming.save_curation_session")
def test_handled_persistence_failure_raises_handled_report_stream_failure_never_completed(mock_save):
    mock_save.side_effect = RuntimeError("disk full")
    session = _session(selected_papers=[_paper("1111", "Paper One")])
    client = MagicMock()
    client.chat.completions.parse.return_value = _mock_parsed_response(
        _analytical_parsed(_build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)),
    )

    events, exc = _run(_collect_until_raise(stream_generate_report_turn(session, "s1", MagicMock(), client, None, None)))

    assert isinstance(exc, HandledReportStreamFailure)
    assert exc.reason_code == "persistence_failed"
    assert "disk full" not in exc.message
    assert "completed" not in _types(events)


def test_unexpected_exception_becomes_safe_handled_report_stream_failure():
    session = _session(selected_papers=[_paper("1111", "Paper One")])
    client = MagicMock()
    client.chat.completions.parse.side_effect = ValueError("boom")

    events, exc = _run(_collect_until_raise(stream_generate_report_turn(session, "s1", MagicMock(), client, None, None)))

    assert isinstance(exc, HandledReportStreamFailure)
    assert "boom" not in exc.message


# --- regeneration -------------------------------------------------------

@patch("research_agent.curation_report_streaming.save_curation_session")
def test_regenerate_sequence_and_generation_reason(mock_save):
    p1 = _paper("1111", "Paper One")
    existing = _clean_analytical_draft([p1])
    session = _session(
        selected_papers=[p1], report=existing,
        web_articles_added=[WebArticle(title="W", url="https://x.com", snippet="s", published_date=None, source_domain="x.com")],
    )
    client = MagicMock()
    client.chat.completions.parse.return_value = _mock_parsed_response(
        _analytical_parsed(_build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)),
    )

    events = _run(_collect(stream_regenerate_report_turn(session, "s1", MagicMock(), client, None, None)))

    assert _types(events) == ["started", "phase", "phase", "completed", "done"]
    mock_save.assert_called_once()


@patch("research_agent.curation_report_streaming.save_curation_session")
def test_regenerate_excludes_revoked_web_articles_from_refinement_context(mock_save):
    p1 = _paper("1111", "Paper One")
    existing = _clean_analytical_draft([p1])
    kept = WebArticle(title="Kept", url="https://kept.com", snippet="s", published_date=None, source_domain="kept.com")
    revoked = WebArticle(title="Revoked", url="https://revoked.com", snippet="s", published_date=None, source_domain="revoked.com")
    session = _session(
        selected_papers=[p1], report=existing, web_articles_added=[kept, revoked],
        revoked_web_article_urls={"https://revoked.com"},
    )
    client = MagicMock()
    schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
    client.chat.completions.parse.side_effect = [
        _mock_parsed_response(_full_analytical_parsed(schema, "1111")),
        _mock_parsed_response(_evaluation(overall_score=90, needs_revision=False)),
    ]

    _run(_collect(stream_regenerate_report_turn(session, "s1", MagicMock(), client, None, "single")))

    # The evaluation prompt is built from web_articles passed to evaluate_report --
    # verified indirectly via call count (2: regenerate + evaluate) and success,
    # matching report.py's own already-tested revocation filter (test_report.py).
    assert client.chat.completions.parse.call_count == 2


# --- cancellation: waits for the currently running task to settle -----

def _blocking_client(started: threading.Event, proceed: threading.Event, response) -> MagicMock:
    def _slow_parse(*args, **kwargs):
        started.set()
        assert proceed.wait(timeout=5), "test driver never released the call -- deadlock"
        return response

    client = MagicMock()
    client.chat.completions.parse.side_effect = _slow_parse
    return client


def test_cancellation_during_generating_waits_for_settlement_then_reraises():
    p1 = _paper("1111", "Paper One")
    session = _session(selected_papers=[p1])
    started = threading.Event()
    proceed = threading.Event()
    finished = threading.Event()
    response = _mock_parsed_response(_analytical_parsed(_build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)))

    def _slow_parse(*args, **kwargs):
        started.set()
        assert proceed.wait(timeout=5)
        finished.set()
        return response

    client = MagicMock()
    client.chat.completions.parse.side_effect = _slow_parse

    async def scenario():
        gen = stream_generate_report_turn(session, "s1", MagicMock(), client, None, None)
        task = asyncio.create_task(_collect(gen))
        for _ in range(2000):  # bounded wait (~10s) -- fail loudly, never hang CI
            if started.is_set():
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("worker never started -- test setup bug")
        task.cancel()
        await asyncio.sleep(0.05)
        assert not finished.is_set(), "the worker settled before the assertion window -- test is not exercising the race"
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()  # the real call ran to completion, never aborted

    asyncio.run(scenario())


@patch("research_agent.curation_report_streaming.save_curation_session")
def test_cancellation_during_evaluating_waits_for_settlement_never_starts_saving(mock_save):
    p1 = _paper("1111", "Paper One")
    session = _session(selected_papers=[p1])
    started = threading.Event()
    proceed = threading.Event()
    finished = threading.Event()
    generate_response = _mock_parsed_response(_analytical_parsed(_build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)))
    eval_response = _mock_parsed_response(_evaluation(overall_score=90, needs_revision=False))
    call_count = {"n": 0}

    def _side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return generate_response
        started.set()
        assert proceed.wait(timeout=5)
        finished.set()
        return eval_response

    client = MagicMock()
    client.chat.completions.parse.side_effect = _side_effect

    async def scenario():
        gen = stream_generate_report_turn(session, "s1", MagicMock(), client, None, "single")
        task = asyncio.create_task(_collect(gen))
        for _ in range(2000):  # bounded wait (~10s) -- fail loudly, never hang CI
            if started.is_set():
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("worker never started -- test setup bug")
        task.cancel()
        await asyncio.sleep(0.05)
        assert not finished.is_set()
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(scenario())
    mock_save.assert_not_called()  # saving never began after the cancellation


def test_cancellation_during_saving_waits_for_save_settlement_then_reraises_no_completed():
    """Mirrors M4.2A's own test_lease_remains_held_through_persistence_
    and_releases_only_after_cancellation_settles proof, at the domain
    layer: the real save_curation_session is patched with a slow,
    controllable stand-in; cancellation must wait for it to genuinely
    finish before the CancelledError propagates, and `completed` must
    never be emitted."""
    p1 = _paper("1111", "Paper One")
    session = _session(selected_papers=[p1])
    started = threading.Event()
    proceed = threading.Event()
    finished = threading.Event()

    def _slow_save(*args, **kwargs):
        started.set()
        assert proceed.wait(timeout=5)
        finished.set()

    client = MagicMock()
    client.chat.completions.parse.return_value = _mock_parsed_response(
        _analytical_parsed(_build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)),
    )
    collected: list[tuple[str, dict]] = []

    async def scenario():
        with patch("research_agent.curation_report_streaming.save_curation_session", _slow_save):
            gen = stream_generate_report_turn(session, "s1", MagicMock(), client, None, None)

            async def drive():
                async for event in gen:
                    collected.append((event.type, event.data))

            task = asyncio.create_task(drive())
            for _ in range(2000):  # bounded wait (~10s) -- fail loudly, never hang CI
                if started.is_set():
                    break
                await asyncio.sleep(0.005)
            else:
                raise AssertionError("worker never started -- test setup bug")
            task.cancel()
            await asyncio.sleep(0.05)
            assert not finished.is_set()
            proceed.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()

    asyncio.run(scenario())
    assert "completed" not in _types(collected)
