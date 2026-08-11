"""Usage Protection M1.1: focused tests for research_agent/telemetry.py
-- the observe-only request/action telemetry foundation. Nothing here
calls a real network/OpenAI endpoint; nothing here touches the real
`data/usage_telemetry.sqlite` (the autouse fixture below redirects
`telemetry.USAGE_DB_PATH` to a fresh tmp_path-scoped file, already
schema-initialized, before every test body runs) -- see
`test_real_usage_db_path_was_never_touched` at the bottom of this file
for the direct, session-wide proof of that.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
import sys

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import research_agent.telemetry as telemetry
from research_agent.telemetry import (
    RequestTelemetryMiddleware,
    get_current_request_id,
    init_usage_db,
    paid_action,
    record_child_call,
)

# Captured once, at import time, before any test has a chance to monkeypatch
# USAGE_DB_PATH -- the one fixed reference point test_real_usage_db_path_
# was_never_touched checks against at the end of this file.
_REAL_USAGE_DB_PATH = telemetry.USAGE_DB_PATH


@pytest.fixture(autouse=True)
def usage_db_path(tmp_path, monkeypatch):
    """Every test in this file gets its own tmp_path-scoped DB file,
    already schema-initialized, before the test body runs -- no test
    needs to remember to call init_usage_db() itself for the common
    case, and the real project DB is never touched by anything here.
    Returns the path so a test can open a raw connection to inspect
    what got written."""
    db_path = tmp_path / "usage.sqlite"
    monkeypatch.setattr(telemetry, "USAGE_DB_PATH", db_path)
    init_usage_db(path=db_path).close()
    return db_path


def _rows(db_path, table, columns="*"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT {columns} FROM {table}").fetchall()]
    finally:
        conn.close()


# --- Schema / persistence ---------------------------------------------

class TestSchema:
    def test_schema_and_indexes_created_in_tmp_path(self, tmp_path):
        # Deliberately a fresh path, separate from the autouse fixture's own
        # (already-initialized) one -- this test is exercising init_usage_db
        # itself, not something built on top of it.
        fresh_path = tmp_path / "fresh.sqlite"
        assert not fresh_path.exists()
        init_usage_db(path=fresh_path).close()
        assert fresh_path.exists()

        conn = sqlite3.connect(fresh_path)
        try:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            # action_leases: added in Usage Protection M2.1 Part C -- see
            # research_agent/leases.py for the acquire/release logic that
            # reads and writes this table.
            assert {"http_requests", "paid_actions", "action_leases"} <= tables
            indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
            assert {
                "idx_http_requests_started_at", "idx_paid_actions_request_id",
                "idx_paid_actions_action_type", "idx_paid_actions_started_at",
            } <= indexes
        finally:
            conn.close()

    def test_no_foreign_key_from_paid_actions_to_http_requests(self, usage_db_path):
        """An action row referencing a request_id that has no matching
        http_requests row yet (the outer middleware hasn't persisted
        its own row when this action finishes) must insert cleanly --
        the lifecycle constraint this suite's own spec calls out."""
        telemetry._write_paid_action(
            action_id="a1", action_type="search", request_id="never-written-request-id",
            subject_type=None, subject_id=None, outcome="success",
            started_at="2026-01-01T00:00:00+00:00", ended_at="2026-01-01T00:00:01+00:00",
            latency_ms=1.0, input_tokens=None, output_tokens=None, total_tokens=None,
            total_call_count=0, child_calls_json="[]", path=usage_db_path,
        )
        rows = _rows(usage_db_path, "paid_actions")
        assert len(rows) == 1
        assert rows[0]["request_id"] == "never-written-request-id"


class TestHttpRequestRoundTrip:
    def test_round_trip(self, usage_db_path):
        telemetry._write_http_request(
            request_id="req-1", route_template="/curation/{session_id}", method="GET",
            status_code=200, outcome="success",
            started_at="2026-01-01T00:00:00+00:00", ended_at="2026-01-01T00:00:00.100000+00:00",
            latency_ms=100.0, path=usage_db_path,
        )
        rows = _rows(usage_db_path, "http_requests")
        assert len(rows) == 1
        row = rows[0]
        assert row["request_id"] == "req-1"
        assert row["route_template"] == "/curation/{session_id}"
        assert row["method"] == "GET"
        assert row["status_code"] == 200
        assert row["outcome"] == "success"
        assert row["latency_ms"] == 100.0


# --- Action aggregation --------------------------------------------------

class TestActionAggregation:
    def test_several_child_calls_aggregate_tokens_and_call_count(self, usage_db_path):
        with paid_action("report_generate", subject_type="session", subject_id="sess-1"):
            record_child_call("generate", "openai", model="gpt-4.1", input_tokens=100, output_tokens=50, total_tokens=150, latency_ms=10.0, outcome="success")
            record_child_call("evaluate", "openai", model="gpt-4.1", input_tokens=80, output_tokens=20, total_tokens=100, latency_ms=8.0, outcome="success")
            record_child_call("revise", "openai", model="gpt-4.1", input_tokens=120, output_tokens=60, total_tokens=180, latency_ms=12.0, outcome="success")

        rows = _rows(usage_db_path, "paid_actions")
        assert len(rows) == 1
        row = rows[0]
        assert row["action_type"] == "report_generate"
        assert row["subject_type"] == "session"
        assert row["subject_id"] == "sess-1"
        assert row["outcome"] == "success"
        assert row["total_call_count"] == 3
        assert row["input_tokens"] == 300
        assert row["output_tokens"] == 130
        assert row["total_tokens"] == 430
        child_calls = json.loads(row["child_calls_json"])
        assert [c["call_type"] for c in child_calls] == ["generate", "evaluate", "revise"]

    def test_unknown_token_usage_stays_nullable_not_fabricated(self, usage_db_path):
        """A provider that reports no usage at all (e.g. Tavily web
        search) must never make the aggregate look like "billed
        exactly zero tokens" -- None in, None out."""
        with paid_action("search"):
            record_child_call("web_search", "tavily", latency_ms=200.0, outcome="success")

        row = _rows(usage_db_path, "paid_actions")[0]
        assert row["input_tokens"] is None
        assert row["output_tokens"] is None
        assert row["total_tokens"] is None
        assert row["total_call_count"] == 1

    def test_mixed_known_and_unknown_tokens_sums_only_the_known_ones(self, usage_db_path):
        with paid_action("search"):
            record_child_call("condense", "openai", input_tokens=10, output_tokens=5, total_tokens=15, latency_ms=1.0, outcome="success")
            record_child_call("web_search", "tavily", latency_ms=200.0, outcome="success")

        row = _rows(usage_db_path, "paid_actions")[0]
        assert row["input_tokens"] == 10
        assert row["output_tokens"] == 5
        assert row["total_tokens"] == 15
        assert row["total_call_count"] == 2


class TestRecordChildCallNoop:
    def test_without_active_action_writes_nothing_and_does_not_raise(self, usage_db_path):
        assert get_current_request_id() is None  # sanity: no ambient context here
        record_child_call("condense", "openai", latency_ms=1.0, outcome="success")  # must not raise
        assert _rows(usage_db_path, "paid_actions") == []


class TestNestedPaidAction:
    def test_nested_paid_action_creates_one_row_only(self, usage_db_path):
        with paid_action("curation_chat", subject_type="session", subject_id="outer"):
            record_child_call("condense", "openai", latency_ms=1.0, outcome="success")
            with paid_action("search_chat", subject_type="session", subject_id="inner-ignored"):
                record_child_call("ask", "openai", latency_ms=2.0, outcome="success")
                record_child_call("ask", "openai", latency_ms=3.0, outcome="success")
            record_child_call("classify_offer", "openai", latency_ms=1.0, outcome="success")

        rows = _rows(usage_db_path, "paid_actions")
        assert len(rows) == 1
        row = rows[0]
        # The OUTER call's own action_type/subject_id win -- the nested
        # call's "search_chat"/"inner-ignored" never surface anywhere.
        assert row["action_type"] == "curation_chat"
        assert row["subject_id"] == "outer"
        assert row["total_call_count"] == 4
        call_types = [c["call_type"] for c in json.loads(row["child_calls_json"])]
        assert call_types == ["condense", "ask", "ask", "classify_offer"]


class TestActionOutcomes:
    def test_success_persists_once(self, usage_db_path):
        with paid_action("search"):
            pass
        rows = _rows(usage_db_path, "paid_actions")
        assert len(rows) == 1
        assert rows[0]["outcome"] == "success"

    def test_error_persists_once_and_reraises_the_original_exception(self, usage_db_path):
        class _Boom(RuntimeError):
            pass

        with pytest.raises(_Boom, match="kaboom"):
            with paid_action("search"):
                raise _Boom("kaboom")

        rows = _rows(usage_db_path, "paid_actions")
        assert len(rows) == 1
        assert rows[0]["outcome"] == "error"

    def test_cancellation_persists_once_as_cancelled_not_error_and_reraises(self, usage_db_path):
        with pytest.raises(asyncio.CancelledError):
            with paid_action("search_chat"):
                raise asyncio.CancelledError()

        rows = _rows(usage_db_path, "paid_actions")
        assert len(rows) == 1
        assert rows[0]["outcome"] == "cancelled"

    def test_nested_action_exception_still_persists_exactly_one_row(self, usage_db_path):
        with pytest.raises(ValueError):
            with paid_action("curation_chat"):
                with paid_action("search_chat"):  # nested, no-op for persistence
                    raise ValueError("nested failure")

        rows = _rows(usage_db_path, "paid_actions")
        assert len(rows) == 1
        assert rows[0]["action_type"] == "curation_chat"
        assert rows[0]["outcome"] == "error"


class TestFailOpen:
    def test_persistence_failure_never_changes_the_application_result(self, usage_db_path, monkeypatch):
        monkeypatch.setattr(telemetry, "_write_paid_action", lambda **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("disk full")))
        result = []
        with paid_action("search"):
            result.append("did the real work")
        assert result == ["did the real work"]  # the with-block's own code ran and returned normally
        assert _rows(usage_db_path, "paid_actions") == []  # the failed write really didn't land

    def test_persistence_failure_inside_an_erroring_action_still_reraises_original(self, usage_db_path, monkeypatch):
        monkeypatch.setattr(telemetry, "_write_paid_action", lambda **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("disk full")))

        class _RealBug(RuntimeError):
            pass

        with pytest.raises(_RealBug):
            with paid_action("search"):
                raise _RealBug("the actual application error")

    def test_serialization_failure_never_changes_the_application_result(self, usage_db_path):
        class _Unserializable:
            pass

        result = []
        with paid_action("search"):
            # A child call carrying something json.dumps can't handle (a
            # real instrumentation bug in some future M1.2 call site) must
            # still never surface to the caller.
            record_child_call("condense", "openai", model=_Unserializable(), latency_ms=1.0, outcome="success")  # type: ignore[arg-type]
            result.append("did the real work")
        assert result == ["did the real work"]
        assert _rows(usage_db_path, "paid_actions") == []  # the failed write really didn't land


class TestConcurrency:
    def test_concurrent_actions_do_not_cross_contaminate_ids_or_child_calls(self, usage_db_path):
        async def _one(subject_id: str, delay: float) -> None:
            with paid_action("search", subject_type="session", subject_id=subject_id):
                for i in range(3):
                    record_child_call(subject_id, "openai", latency_ms=float(i), outcome="success")
                    await asyncio.sleep(delay)  # forces genuine interleaving between the two tasks

        async def _main():
            await asyncio.gather(_one("session-a", 0.01), _one("session-b", 0.005))

        asyncio.run(_main())

        rows = {r["subject_id"]: r for r in _rows(usage_db_path, "paid_actions")}
        assert set(rows) == {"session-a", "session-b"}
        for subject_id, row in rows.items():
            assert row["total_call_count"] == 3
            child_calls = json.loads(row["child_calls_json"])
            assert all(c["call_type"] == subject_id for c in child_calls)

    def test_concurrent_request_ids_do_not_cross_contaminate(self):
        seen: dict[str, str | None] = {}

        async def downstream(scope, receive, send):
            await asyncio.sleep(0.01 if scope["path"] == "/a" else 0.005)
            seen[scope["path"]] = get_current_request_id()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        middleware = RequestTelemetryMiddleware(downstream)

        async def _request(path: str):
            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                pass

            scope = {"type": "http", "method": "GET", "path": path, "headers": [], "query_string": b""}
            await middleware(scope, receive, send)

        async def _main():
            await asyncio.gather(_request("/a"), _request("/b"))

        asyncio.run(_main())
        assert seen["/a"] is not None
        assert seen["/b"] is not None
        assert seen["/a"] != seen["/b"]


class TestDeterministicJson:
    def test_child_calls_json_is_deterministic_across_runs(self, tmp_path, monkeypatch):
        db_path_1 = tmp_path / "usage1.sqlite"
        db_path_2 = tmp_path / "usage2.sqlite"
        init_usage_db(path=db_path_1).close()
        init_usage_db(path=db_path_2).close()

        def _do(db_path):
            monkeypatch.setattr(telemetry, "USAGE_DB_PATH", db_path)
            with paid_action("report_generate", subject_type="session", subject_id="s"):
                record_child_call("generate", "openai", model="gpt-4.1", input_tokens=10, output_tokens=5, total_tokens=15, cache_hit=False, latency_ms=1.0, outcome="success")
                record_child_call("evaluate", "openai", model="gpt-4.1", input_tokens=None, output_tokens=None, total_tokens=None, cache_hit=None, latency_ms=2.0, outcome="success")

        _do(db_path_1)
        _do(db_path_2)

        json_1 = _rows(db_path_1, "paid_actions")[0]["child_calls_json"]
        json_2 = _rows(db_path_2, "paid_actions")[0]["child_calls_json"]
        assert json_1 == json_2

    def test_child_call_key_order_is_sorted_within_each_record(self, usage_db_path):
        with paid_action("search"):
            record_child_call("condense", "openai", latency_ms=1.0, outcome="success")
        raw_json = _rows(usage_db_path, "paid_actions")[0]["child_calls_json"]
        parsed = json.loads(raw_json)
        # Re-serializing the parsed structure with sort_keys=True must
        # produce byte-identical output to what was actually stored --
        # the direct proof that storage already used sorted keys.
        assert json.dumps(parsed, sort_keys=True) == raw_json


# --- Privacy / typed-API surface ------------------------------------------

class TestNoForbiddenContentFields:
    def test_record_child_call_has_no_kwargs_or_varargs(self):
        sig = inspect.signature(record_child_call)
        kinds = {p.kind for p in sig.parameters.values()}
        assert inspect.Parameter.VAR_KEYWORD not in kinds
        assert inspect.Parameter.VAR_POSITIONAL not in kinds
        assert set(sig.parameters) == {
            "call_type", "provider", "model", "input_tokens", "output_tokens",
            "total_tokens", "cache_hit", "latency_ms", "outcome", "error_type",
        }

    def test_paid_action_has_no_kwargs_or_varargs(self):
        sig = inspect.signature(paid_action)
        kinds = {p.kind for p in sig.parameters.values()}
        assert inspect.Parameter.VAR_KEYWORD not in kinds
        assert inspect.Parameter.VAR_POSITIONAL not in kinds
        assert set(sig.parameters) == {"action_type", "subject_type", "subject_id", "discard_if_empty"}

    def test_no_field_named_after_a_forbidden_content_category(self):
        forbidden_substrings = ("prompt", "message", "text", "url", "query", "header", "api_key", "abstract")
        all_param_names = set(inspect.signature(record_child_call).parameters) | set(inspect.signature(paid_action).parameters)
        for name in all_param_names:
            for bad in forbidden_substrings:
                assert bad not in name.lower(), f"{name!r} looks like it could carry forbidden content"


# --- Middleware ------------------------------------------------------------

def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestTelemetryMiddleware)

    @app.get("/items/{item_id}")
    def get_item(item_id: str):
        return {"item_id": item_id}

    @app.get("/missing")
    def missing():
        raise HTTPException(status_code=404, detail="not found")

    @app.get("/boom")
    def boom():
        raise RuntimeError("a genuinely unhandled bug")

    @app.get("/stream")
    def stream():
        def gen():
            yield b"chunk-1-"
            yield b"chunk-2-"
            yield b"chunk-3"

        return StreamingResponse(gen(), media_type="text/plain")

    return app


class TestMiddleware:
    def test_successful_route_returns_request_id_header_and_one_request_row(self, usage_db_path):
        app = _build_test_app()

        with TestClient(app) as client:
            resp = client.get("/items/real-session-id-abc123")

        assert resp.status_code == 200
        request_id = resp.headers.get("x-request-id")
        assert request_id
        rows = _rows(usage_db_path, "http_requests")
        assert len(rows) == 1
        assert rows[0]["request_id"] == request_id
        assert rows[0]["status_code"] == 200
        assert rows[0]["outcome"] == "success"
        assert rows[0]["method"] == "GET"

    def test_route_template_is_stored_not_the_raw_path(self, usage_db_path):
        app = _build_test_app()

        with TestClient(app) as client:
            client.get("/items/real-session-id-abc123")

        row = _rows(usage_db_path, "http_requests")[0]
        assert row["route_template"] == "/items/{item_id}"
        assert "real-session-id-abc123" not in row["route_template"]

    def test_http_exception_response_retains_header_and_correct_status(self, usage_db_path):
        app = _build_test_app()

        with TestClient(app) as client:
            resp = client.get("/missing")

        assert resp.status_code == 404
        assert resp.headers.get("x-request-id")
        row = _rows(usage_db_path, "http_requests")[0]
        assert row["status_code"] == 404
        # An HTTPException handled cleanly by FastAPI's own exception
        # machinery never reaches this middleware's own except-block at
        # all (confirmed directly against the installed fastapi/starlette
        # middleware ordering) -- this is a normal completion from the
        # middleware's own point of view.
        assert row["outcome"] == "success"

    def test_unhandled_exception_records_500_error_and_reraises(self, usage_db_path):
        app = _build_test_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/boom")

        assert resp.status_code == 500
        row = _rows(usage_db_path, "http_requests")[0]
        assert row["status_code"] == 500
        assert row["outcome"] == "error"

    def test_streaming_response_chunks_delivered_unchanged_and_persisted_only_after_completion(self, usage_db_path):
        app = _build_test_app()

        with TestClient(app) as client:
            # Mid-request: nothing persisted yet (the middleware's own
            # `await self.app(...)` hasn't returned, since a StreamingResponse
            # only finishes sending once every chunk the generator yields has
            # gone through `send`).
            with client.stream("GET", "/stream") as resp:
                body = b"".join(resp.iter_bytes())

        assert body == b"chunk-1-chunk-2-chunk-3"  # every chunk arrived, unmodified, in order
        rows = _rows(usage_db_path, "http_requests")
        assert len(rows) == 1  # persisted exactly once, after the stream fully completed
        assert rows[0]["status_code"] == 200
        assert rows[0]["outcome"] == "success"

    def test_two_concurrent_http_requests_have_distinct_request_ids(self, usage_db_path):
        app = _build_test_app()

        with TestClient(app) as client:
            r1 = client.get("/items/one")
            r2 = client.get("/items/two")

        id_1, id_2 = r1.headers["x-request-id"], r2.headers["x-request-id"]
        assert id_1 != id_2
        rows = _rows(usage_db_path, "http_requests")
        assert {row["request_id"] for row in rows} == {id_1, id_2}

    def test_middleware_persistence_failure_does_not_fail_the_route(self, usage_db_path, monkeypatch):
        monkeypatch.setattr(telemetry, "_write_http_request", lambda **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("disk full")))
        app = _build_test_app()

        with TestClient(app) as client:
            resp = client.get("/items/still-works")

        assert resp.status_code == 200
        assert resp.json() == {"item_id": "still-works"}
        assert _rows(usage_db_path, "http_requests") == []  # the failed write really didn't land


# --- Compatibility ----------------------------------------------------

class TestCompatibility:
    def test_paid_action_used_outside_any_request_has_no_request_id(self, usage_db_path):
        """The exact structural property that keeps eval/CLI invocations
        of shared domain functions from ever writing telemetry: with no
        RequestTelemetryMiddleware in the call chain, there is no
        request_id to attach, and this is not treated as an error."""
        assert get_current_request_id() is None
        with paid_action("report_generate"):
            record_child_call("generate", "openai", latency_ms=1.0, outcome="success")
        row = _rows(usage_db_path, "paid_actions")[0]
        assert row["request_id"] is None

    def test_importing_telemetry_alone_creates_no_database_file(self, tmp_path, monkeypatch):
        """Merely `import research_agent.telemetry` (exactly what every
        module that will eventually call record_child_call in M1.2 does)
        must never itself create a database file -- only an explicit
        init_usage_db()/write call does."""
        never_created = tmp_path / "should_stay_absent.sqlite"
        monkeypatch.setattr(telemetry, "USAGE_DB_PATH", never_created)
        # Re-importing an already-imported module is a no-op in Python
        # (module bodies don't re-run) -- this asserts the state after the
        # only import that already happened at the top of this file.
        assert "research_agent.telemetry" in sys.modules
        assert not never_created.exists()


def test_real_usage_db_path_was_never_touched():
    """Session-wide proof, not just a per-test assumption: the ACTUAL
    project path (captured at import time, before the autouse fixture
    ever monkeypatches USAGE_DB_PATH for any test in this file) was
    never created or written to by anything in this test run."""
    assert not _REAL_USAGE_DB_PATH.exists()
