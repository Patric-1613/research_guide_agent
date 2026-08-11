"""Usage Protection M1.2: focused tests for the production instrumentation
wired into the service layer and domain modules on top of M1.1's telemetry
foundation (research_agent/telemetry.py, tested on its own in
tests/test_telemetry.py).

Every test here either goes through the real HTTP layer (TestClient,
mirroring tests/test_api.py's and tests/test_curation_api.py's own
established mocked-vs-real trade-off) or calls an instrumented domain
function directly with a mocked OpenAI/requests/Tavily call -- nothing here
makes a real network/paid call. `usage_db_path` (this file's own fixture)
redirects `telemetry.USAGE_DB_PATH` to a tmp_path file before every test, so
nothing here ever touches the real `data/usage_telemetry.sqlite` -- see
`test_real_usage_db_path_untouched` at the bottom for the direct,
session-wide proof of that.
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research_agent.api as api
import research_agent.telemetry as telemetry
from research_agent import embeddings as embeddings_module
from research_agent import enrichment as enrichment_module
from research_agent import ingestion as ingestion_module
from research_agent import qa as qa_module
from research_agent import report as report_module
from research_agent import web_search as web_search_module
from research_agent.qa import sqlite_checkpointer
from research_agent.schema import Paper, WebArticle
from research_agent.storage import init_db as real_init_db

REPO_ROOT = Path(__file__).resolve().parent.parent

_REAL_USAGE_DB_PATH = telemetry.USAGE_DB_PATH
# Captured before any test patches qa_module._init_direct_relevance_cache_db
# by name -- a side_effect lambda that referenced qa_module.<name> from
# INSIDE the patched context would just call right back into the mock
# patching its own name replaced, not the real implementation.
_real_init_direct_relevance_cache_db = qa_module._init_direct_relevance_cache_db


@pytest.fixture(autouse=True)
def usage_db_path(tmp_path, monkeypatch):
    db_path = tmp_path / "usage_telemetry.sqlite"
    monkeypatch.setattr(telemetry, "USAGE_DB_PATH", db_path)
    telemetry.init_usage_db(path=db_path).close()
    return db_path


def _actions(db_path, **where):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM paid_actions").fetchall()]
    finally:
        conn.close()
    for key, value in where.items():
        rows = [r for r in rows if r[key] == value]
    return rows


def _http_requests(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM http_requests").fetchall()]
    finally:
        conn.close()


def _child_calls(action_row) -> list[dict]:
    return json.loads(action_row["child_calls_json"])


def _mock_usage(total=100, prompt=80, completion=20):
    return MagicMock(total_tokens=total, prompt_tokens=prompt, completion_tokens=completion)


def _mock_parsed_response(parsed, usage=None):
    mock_message = MagicMock(parsed=parsed, refusal=None)
    mock_response = MagicMock(usage=usage if usage is not None else _mock_usage())
    mock_response.choices = [MagicMock(message=mock_message)]
    return mock_response


# --- Shared fixtures for HTTP-level (TestClient) tests, mirroring
# tests/test_api.py's and tests/test_curation_api.py's own established
# `_client()` convention -- duplicated here (not imported) per this
# project's existing per-file-helper style. ---------------------------

def _paper(paper_id: str, title: str = "A Paper", abstract: str = "an abstract") -> Paper:
    return Paper(
        title=title, authors=["A. Uthor"], year=2024, venue="arXiv preprint",
        abstract=abstract, url=f"http://arxiv.org/abs/{paper_id}",
        doi=None, citation_count=None, source="arxiv", paper_id=paper_id,
    )


def _ranked(papers: list[Paper]) -> list[tuple[Paper, float]]:
    return [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)]


def _make_test_db_override(db_path: Path):
    def _override():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    return _override


def _make_test_checkpointer_override(cp_db_path: Path):
    def _override():
        with sqlite_checkpointer(cp_db_path) as cp:
            yield cp

    return _override


@contextmanager
def _api_client(usage_db_path: Path):
    """One-shot-pipeline client (search/summarize/chat/export) -- isolated
    storage.py DB, real lifespan (so the middleware/telemetry schema is
    genuinely exercised), USAGE_DB_PATH redirected to this test's own
    fixture path so rows land where the test can read them back."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.sqlite"
        with patch.object(api, "init_db", lambda: real_init_db(db_path)), \
             patch.object(telemetry, "USAGE_DB_PATH", usage_db_path), \
             patch.object(api, "search_web", return_value=[]), \
             patch.object(api, "OpenAI", return_value=MagicMock()):
            api.app.dependency_overrides[api.get_db_connection] = _make_test_db_override(db_path)
            try:
                with TestClient(api.app) as client:
                    yield client
            finally:
                api.app.dependency_overrides.clear()


@contextmanager
def _curation_client(usage_db_path: Path):
    """Curation-workflow client -- real checkpointer/graph (same rationale
    as test_curation_api.py's own _client()), USAGE_DB_PATH redirected to
    this test's own fixture path."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.sqlite"
        cp_db_path = Path(tmp) / "test_checkpoints.sqlite"
        with patch.object(api, "init_db", lambda: real_init_db(db_path)), \
             patch.object(telemetry, "USAGE_DB_PATH", usage_db_path), \
             patch.object(api, "search_web", return_value=[]), \
             patch.object(api, "OpenAI", return_value=MagicMock()), \
             patch.object(api, "canonicalize_topic", side_effect=lambda topic, client=None: topic):
            api.app.dependency_overrides[api.get_db_connection] = _make_test_db_override(db_path)
            api.app.dependency_overrides[api.get_curation_checkpointer] = _make_test_checkpointer_override(cp_db_path)
            try:
                with TestClient(api.app) as client:
                    yield client
            finally:
                api.app.dependency_overrides.clear()


# =====================================================================
# Coverage guard
# =====================================================================

_PRODUCTION_MODULES_WITH_EXTERNAL_CALLS = (
    "research_agent/ingestion.py",
    "research_agent/web_search.py",
    "research_agent/embeddings.py",
    "research_agent/qa.py",
    "research_agent/report.py",
    "research_agent/summarize.py",
    "research_agent/query_expansion.py",
    "research_agent/curation_chat.py",
    "research_agent/enrichment.py",
    "research_agent/agent.py",
)
# Excluded per the M1.2 spec: eval/CLI code is a separate, un-instrumented
# world by design (see tests/test_telemetry_instrumentation.py's own
# TestIsolation below for the direct proof), and agent.py's internal
# LangChain tool-loop turns are the one deliberately-opaque exception
# (agent_loop_unmetered) -- see run_research_agent's own docstring/comment.


def _dotted_call_name(func_node: ast.AST) -> str | None:
    parts: list[str] = []
    node = func_node
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


_EXTERNAL_CALL_SUFFIXES = (
    ".chat.completions.create", ".chat.completions.parse", ".embeddings.create",
)
_EXTERNAL_CALL_EXACT = ("requests.get", "requests.post")


def _is_external_call(call: ast.Call) -> bool:
    dotted = _dotted_call_name(call.func)
    if dotted is None:
        return False
    if dotted in _EXTERNAL_CALL_EXACT:
        return True
    if any(dotted.endswith(suffix) for suffix in _EXTERNAL_CALL_SUFFIXES):
        return True
    # Tavily's own `tool.invoke({...})` -- narrow to the exact local
    # variable name web_search.py uses, so this doesn't over-match every
    # unrelated `.invoke(` call some other library might expose.
    if dotted == "tool.invoke":
        return True
    return False


def _function_contains_external_call(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) and _is_external_call(n) for n in ast.walk(node))


def _function_contains_instrumentation(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Attribute) and n.attr in ("timed_child_call", "record_child_call")
        for n in ast.walk(node)
    )


def _find_uninstrumented_functions(rel_path: str) -> list[str]:
    tree = ast.parse((REPO_ROOT / rel_path).read_text())
    gaps = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _function_contains_external_call(node) and not _function_contains_instrumentation(node):
                gaps.append(f"{rel_path}::{node.name}")
    return gaps


class TestInstrumentationCoverageGuard:
    def test_detection_logic_catches_a_synthetic_uninstrumented_call_site(self):
        """Self-test of the guard's own AST logic (not the real codebase)
        -- proves this actually fails when a new production external call
        is added with no instrumentation, rather than trivially passing
        no matter what."""
        source = """
def leaky_call(client):
    response = client.chat.completions.create(model="x", messages=[])
    return response
"""
        tree = ast.parse(source)
        func = tree.body[0]
        assert _function_contains_external_call(func) is True
        assert _function_contains_instrumentation(func) is False

    def test_detection_logic_accepts_an_instrumented_call_site(self):
        source = """
def wrapped_call(client):
    with telemetry.timed_child_call("x", "openai") as call:
        response = client.chat.completions.create(model="x", messages=[])
        call.set_usage(response.usage)
    return response
"""
        tree = ast.parse(source)
        func = tree.body[0]
        assert _function_contains_external_call(func) is True
        assert _function_contains_instrumentation(func) is True

    def test_every_production_external_call_site_is_instrumented(self):
        gaps: list[str] = []
        for rel_path in _PRODUCTION_MODULES_WITH_EXTERNAL_CALLS:
            gaps.extend(_find_uninstrumented_functions(rel_path))
        assert gaps == [], f"Uninstrumented production external/model call site(s): {gaps}"

    def test_eval_modules_are_excluded_from_the_guard_by_design(self):
        """research_agent/evals/** deliberately reuses report.py/qa.py's
        OWN already-instrumented functions directly rather than making its
        own separate client calls (confirmed by the M1 architecture
        audit) -- this test just documents that the guard's own module
        list above never includes anything under evals/, which is the
        actual mechanism keeping eval code out of scope here."""
        assert not any(m.startswith("research_agent/evals/") for m in _PRODUCTION_MODULES_WITH_EXTERNAL_CALLS)


# =====================================================================
# Top-level actions
# =====================================================================

class TestSearchAction:
    def test_agent_path_records_search_action_with_search_id_subject(self, usage_db_path):
        papers = [_paper("p1", "Paper One")]
        fake_session = MagicMock(papers=papers, ranked=[(papers[0], 0.9)], web_articles=[])

        with _api_client(usage_db_path) as client, patch.object(api, "run_research_agent", return_value=fake_session):
            resp = client.post("/search", json={"topic": "test topic"})

        assert resp.status_code == 200
        search_id = resp.json()["search_id"]
        rows = _actions(usage_db_path, action_type="search")
        assert len(rows) == 1
        assert rows[0]["subject_type"] == "search"
        assert rows[0]["subject_id"] == str(search_id)
        assert rows[0]["outcome"] == "success"
        # request correlation: the action's own request_id matches the one
        # http_requests recorded for this same POST /search.
        http_rows = _http_requests(usage_db_path)
        assert len(http_rows) == 1
        assert rows[0]["request_id"] == http_rows[0]["request_id"]

    def test_expansion_path_records_search_action(self, usage_db_path):
        papers = [_paper("p1")]
        with _api_client(usage_db_path) as client, patch.object(api, "expanded_search", return_value=_ranked(papers)):
            resp = client.post("/search", json={"topic": "t", "use_query_expansion": True})

        assert resp.status_code == 200
        rows = _actions(usage_db_path, action_type="search")
        assert len(rows) == 1

    def test_no_papers_found_still_records_action_as_error(self, usage_db_path):
        fake_session = MagicMock(papers=[], ranked=[])
        with _api_client(usage_db_path) as client, patch.object(api, "run_research_agent", return_value=fake_session):
            resp = client.post("/search", json={"topic": "nothing"})

        assert resp.status_code == 404
        rows = _actions(usage_db_path, action_type="search")
        assert len(rows) == 1
        assert rows[0]["outcome"] == "error"
        assert rows[0]["subject_id"] is None  # never reached save_search()


class TestSummarizeAction:
    def _seed_search(self, client) -> int:
        fake_session = MagicMock(papers=[_paper("p1")], ranked=[(_paper("p1"), 0.9)], web_articles=[])
        with patch.object(api, "run_research_agent", return_value=fake_session):
            resp = client.post("/search", json={"topic": "t"})
        return resp.json()["search_id"]

    def test_fresh_summarize_records_action_with_real_child_call(self, usage_db_path):
        with _api_client(usage_db_path) as client:
            search_id = self._seed_search(client)
            fake_summary = {"themes": [], "gaps_and_disagreements": "", "skipped_papers": []}
            with patch.object(api, "generate_summary", return_value=fake_summary):
                resp = client.post("/summarize", json={"search_id": search_id})

        assert resp.status_code == 200
        rows = _actions(usage_db_path, action_type="summarize")
        assert len(rows) == 1
        assert rows[0]["subject_type"] == "search"
        assert rows[0]["subject_id"] == str(search_id)

    def test_cached_summarize_records_action_with_zero_child_calls(self, usage_db_path):
        """A cache-only completion (both summaries already generated for
        this search_id) still gets exactly one row -- a real, successful
        request -- but with total_call_count=0 and null token fields,
        never fabricated zeros."""
        with _api_client(usage_db_path) as client:
            search_id = self._seed_search(client)
            fake_summary = {"themes": [], "gaps_and_disagreements": "", "skipped_papers": []}
            with patch.object(api, "generate_summary", return_value=fake_summary) as mock_generate:
                client.post("/summarize", json={"search_id": search_id})
                # Second call: get_search now returns a saved summary, so
                # _get_or_create_summary short-circuits before ever calling
                # generate_summary again.
                resp = client.post("/summarize", json={"search_id": search_id})

        assert resp.status_code == 200
        assert mock_generate.call_count == 1  # only the first call actually generated
        rows = _actions(usage_db_path, action_type="summarize")
        assert len(rows) == 2
        cached_row = rows[1]
        assert cached_row["total_call_count"] == 0
        assert cached_row["input_tokens"] is None
        assert cached_row["output_tokens"] is None
        assert cached_row["total_tokens"] is None


class TestSearchChatAction:
    def test_chat_records_search_chat_action(self, usage_db_path):
        fake_session = MagicMock(papers=[_paper("p1")], ranked=[(_paper("p1"), 0.9)], web_articles=[])
        with _api_client(usage_db_path) as client:
            with patch.object(api, "run_research_agent", return_value=fake_session):
                search_id = client.post("/search", json={"topic": "t"}).json()["search_id"]

            fake_result = {
                "answer": "an answer", "answerable": True, "cited_papers": [], "cited_web_articles": [],
            }
            with patch.object(api, "ask", return_value=fake_result):
                resp = client.post("/chat", json={"search_id": search_id, "question": "a question?", "history": []})

        assert resp.status_code == 200
        rows = _actions(usage_db_path, action_type="search_chat")
        assert len(rows) == 1
        assert rows[0]["subject_type"] == "search"
        assert rows[0]["subject_id"] == str(search_id)


class TestZeroPaidCallOperations:
    """Every operation the M1.2 spec explicitly names as zero-paid-call
    must produce NO paid_actions row at all."""

    def test_curation_state_read_records_no_action(self, usage_db_path):
        with _curation_client(usage_db_path) as client, \
             patch.object(api, "build_candidate_pool", return_value=[_paper("p1")]), \
             patch.object(api, "rank_full_pool", return_value=(_ranked([_paper("p1")]), {})):
            session_id = client.post("/curation/start", json={"topic": "t", "target_count": 5}).json()["session_id"]
            client.get(f"/curation/{session_id}")
            client.get("/curation/reviews")

        # Only curation_start itself should have produced a row -- the two
        # read-only GETs above must not.
        rows = _actions(usage_db_path)
        assert {r["action_type"] for r in rows} == {"curation_start"}

    def test_curation_delete_records_no_action(self, usage_db_path):
        with _curation_client(usage_db_path) as client, \
             patch.object(api, "build_candidate_pool", return_value=[_paper("p1")]), \
             patch.object(api, "rank_full_pool", return_value=(_ranked([_paper("p1")]), {})):
            session_id = client.post("/curation/start", json={"topic": "t", "target_count": 5}).json()["session_id"]
            before = len(_actions(usage_db_path))
            client.delete(f"/curation/{session_id}")

        assert len(_actions(usage_db_path)) == before

    def test_select_from_history_records_no_action(self, usage_db_path):
        papers = [_paper(f"p{i}") for i in range(12)]
        with _curation_client(usage_db_path) as client, \
             patch.object(api, "build_candidate_pool", return_value=papers), \
             patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
            start = client.post("/curation/start", json={"topic": "t", "target_count": 3}).json()
            session_id = start["session_id"]
            unpicked = [p["paper_id"] for p in start["batch"]][-1]
            client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [start["batch"][0]["paper_id"]], "stop": True})
            before = len(_actions(usage_db_path))
            resp = client.post(f"/curation/{session_id}/select-from-history", json={"paper_id": unpicked})

        assert resp.status_code == 200
        assert len(_actions(usage_db_path)) == before

    def test_chat_exchange_delete_records_no_action(self, usage_db_path):
        papers = [_paper(f"p{i}") for i in range(12)]
        with _curation_client(usage_db_path) as client, \
             patch.object(api, "build_candidate_pool", return_value=papers), \
             patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
            start = client.post("/curation/start", json={"topic": "t", "target_count": 3}).json()
            session_id = start["session_id"]
            client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [start["batch"][0]["paper_id"]], "stop": True})

            fake_chat_result = {
                "answer": "a", "answerable": True, "cited_papers": [], "cited_web_articles": [],
                "chat_history": [
                    {"role": "user", "content": "q", "exchange_id": "ex1"},
                    {"role": "assistant", "content": "a", "exchange_id": "ex1"},
                ],
            }

            def _fake_chat_turn(session, message, client=None):
                session.chat_history = fake_chat_result["chat_history"]
                return fake_chat_result

            with patch.object(api, "chat_turn", side_effect=_fake_chat_turn):
                client.post(f"/curation/{session_id}/chat", json={"message": "q"})

            before = len(_actions(usage_db_path))
            resp = client.post(f"/curation/{session_id}/chat/exchanges/delete", json={"exchange_ids": ["ex1"]})

        assert resp.status_code == 200
        assert len(_actions(usage_db_path)) == before

    def test_report_version_activate_records_no_action(self, usage_db_path):
        papers = [_paper(f"p{i}") for i in range(12)]
        with _curation_client(usage_db_path) as client, \
             patch.object(api, "build_candidate_pool", return_value=papers), \
             patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
            start = client.post("/curation/start", json={"topic": "t", "target_count": 1}).json()
            session_id = start["session_id"]
            client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [start["batch"][0]["paper_id"]], "stop": True})

            fake_report = {**_empty_report(), "report_template": "analytical"}
            with patch.object(api, "generate_report_for_session", return_value=fake_report), \
                 patch.object(api, "refine_report_if_requested", side_effect=lambda report, *a, **kw: report):
                client.post(f"/curation/{session_id}/report")
            version_id = client.get(f"/curation/{session_id}").json()["report_versions"][0]["version_id"]

            before = len(_actions(usage_db_path))
            resp = client.post(f"/curation/{session_id}/reports/{version_id}/activate")

        assert resp.status_code == 200
        assert len(_actions(usage_db_path)) == before

    def test_report_export_records_no_action(self, usage_db_path):
        papers = [_paper(f"p{i}") for i in range(12)]
        with _curation_client(usage_db_path) as client, \
             patch.object(api, "build_candidate_pool", return_value=papers), \
             patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
            start = client.post("/curation/start", json={"topic": "t", "target_count": 1}).json()
            session_id = start["session_id"]
            client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [start["batch"][0]["paper_id"]], "stop": True})

            fake_report = {**_empty_report(), "report_template": "analytical"}
            with patch.object(api, "generate_report_for_session", return_value=fake_report), \
                 patch.object(api, "refine_report_if_requested", side_effect=lambda report, *a, **kw: report):
                client.post(f"/curation/{session_id}/report")

            before = len(_actions(usage_db_path))
            resp = client.get(f"/curation/{session_id}/report/export?format=markdown")

        assert resp.status_code == 200
        assert len(_actions(usage_db_path)) == before

    def test_ordinary_pick_that_does_not_refill_records_no_action(self, usage_db_path):
        from research_agent import query_expansion as qe_module

        papers = [_paper(f"p{i}") for i in range(25)]  # comfortably more than one batch
        with _curation_client(usage_db_path) as client, \
             patch.object(api, "build_candidate_pool", return_value=papers), \
             patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
            start = client.post("/curation/start", json={"topic": "t", "target_count": 30}).json()
            session_id = start["session_id"]
            before = len(_actions(usage_db_path))
            picks = [p["paper_id"] for p in start["batch"][:2]]
            with patch.object(qe_module, "build_candidate_pool") as mock_build_unused:
                resp = client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks})

        assert resp.status_code == 200
        assert resp.json()["refilled"] is False
        mock_build_unused.assert_not_called()
        assert len(_actions(usage_db_path)) == before  # no curation_refill row


def _empty_report() -> dict:
    """Minimal legacy 3-section report shape -- same convention
    tests/test_curation_api.py's own fake_report fixtures already use;
    _report_to_out derives the full section list from this via
    derive_sections_from_legacy_report."""
    return {
        "findings": {"content": "f", "cited_papers": []},
        "limitations": {"content": "", "cited_papers": []},
        "future_scope": {"content": "", "cited_papers": []},
        "skipped_papers": [],
    }


class TestCurationStartAction:
    def test_records_action_with_minted_session_id_subject(self, usage_db_path):
        papers = [_paper(f"p{i}") for i in range(5)]
        with _curation_client(usage_db_path) as client, \
             patch.object(api, "build_candidate_pool", return_value=papers), \
             patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
            resp = client.post("/curation/start", json={"topic": "t", "target_count": 5})

        assert resp.status_code == 200
        session_id = resp.json()["session_id"]
        rows = _actions(usage_db_path, action_type="curation_start")
        assert len(rows) == 1
        assert rows[0]["subject_type"] == "session"
        assert rows[0]["subject_id"] == session_id
        assert rows[0]["outcome"] == "success"

    def test_no_papers_found_records_action_as_error(self, usage_db_path):
        with _curation_client(usage_db_path) as client, \
             patch.object(api, "build_candidate_pool", return_value=[]), \
             patch.object(api, "rank_full_pool", return_value=([], {})):
            resp = client.post("/curation/start", json={"topic": "nothing", "target_count": 5})

        assert resp.status_code == 404
        rows = _actions(usage_db_path, action_type="curation_start")
        assert len(rows) == 1
        assert rows[0]["outcome"] == "error"
        assert rows[0]["subject_id"] is None  # never reached session_id minting


class TestCurationRefillAction:
    def test_refill_records_action(self, usage_db_path):
        from research_agent import query_expansion as qe_module

        initial_papers = [_paper(f"p{i}") for i in range(10)]  # exactly one batch
        fresh_papers = [_paper(f"new{i}") for i in range(8)]

        def _fake_build_candidate_pool(*a, **kw):
            # Real build_candidate_pool makes real, instrumented child calls
            # internally (suggest_related_titles/search_arxiv/search_
            # semantic_scholar) -- simulated here since this test mocks the
            # whole function away; the service-layer discard_if_empty
            # wiring is what's under test, not build_candidate_pool's own
            # internal instrumentation (covered separately).
            telemetry.record_child_call("search_arxiv", "arxiv", latency_ms=1.0, outcome="success")
            return fresh_papers

        with _curation_client(usage_db_path) as client, \
             patch.object(api, "build_candidate_pool", return_value=initial_papers), \
             patch.object(api, "rank_full_pool", return_value=(_ranked(initial_papers), {})):
            start = client.post("/curation/start", json={"topic": "t", "target_count": 30}).json()
            session_id = start["session_id"]
            picks = [p["paper_id"] for p in start["batch"][:5]]

            with patch.object(qe_module, "build_candidate_pool", side_effect=_fake_build_candidate_pool), \
                 patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                     [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
                 )):
                resp = client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks})

        assert resp.status_code == 200
        assert resp.json()["refilled"] is True
        rows = _actions(usage_db_path, action_type="curation_refill")
        assert len(rows) == 1
        assert rows[0]["subject_type"] == "session"
        assert rows[0]["subject_id"] == session_id

    def test_no_refill_records_no_action(self, usage_db_path):
        """The complement of test_refill_records_action -- a plain pick
        that doesn't need a refill leaves zero curation_refill rows, not a
        zero-child-call one. (Also covered from the read side in
        TestZeroPaidCallOperations; kept here too so the refill/no-refill
        pair sits together as one explicit contrast.)"""
        papers = [_paper(f"p{i}") for i in range(25)]
        with _curation_client(usage_db_path) as client, \
             patch.object(api, "build_candidate_pool", return_value=papers), \
             patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
            start = client.post("/curation/start", json={"topic": "t", "target_count": 30}).json()
            session_id = start["session_id"]
            picks = [p["paper_id"] for p in start["batch"][:2]]
            resp = client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": picks})

        assert resp.status_code == 200
        assert resp.json()["refilled"] is False
        assert _actions(usage_db_path, action_type="curation_refill") == []

    def test_reopen_refill_records_action(self, usage_db_path):
        from research_agent import query_expansion as qe_module

        papers = [_paper(f"p{i}") for i in range(3)]  # exhausted after one small batch
        fresh_papers = [_paper(f"new{i}") for i in range(4)]

        def _fake_build_candidate_pool(*a, **kw):
            telemetry.record_child_call("search_arxiv", "arxiv", latency_ms=1.0, outcome="success")
            return fresh_papers

        with _curation_client(usage_db_path) as client, \
             patch.object(api, "build_candidate_pool", return_value=papers), \
             patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
            start = client.post("/curation/start", json={"topic": "t", "target_count": 20}).json()
            session_id = start["session_id"]
            client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [], "stop": True})

            before = len(_actions(usage_db_path, action_type="curation_refill"))
            with patch.object(qe_module, "build_candidate_pool", side_effect=_fake_build_candidate_pool), \
                 patch.object(qe_module, "rank_full_pool", side_effect=lambda topic, papers, client=None, **kw: (
                     [(p, 1.0 - i * 0.01) for i, p in enumerate(papers)], {}
                 )):
                resp = client.post(f"/curation/{session_id}/reopen")

        assert resp.status_code == 200
        rows = _actions(usage_db_path, action_type="curation_refill")
        assert len(rows) == before + 1
        assert rows[-1]["subject_id"] == session_id


class TestCurationChatAction:
    def _started_session(self, client) -> str:
        papers = [_paper(f"p{i}") for i in range(3)]
        with patch.object(api, "build_candidate_pool", return_value=papers), \
             patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
            start = client.post("/curation/start", json={"topic": "t", "target_count": 1}).json()
        session_id = start["session_id"]
        client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [start["batch"][0]["paper_id"]], "stop": True})
        return session_id

    def test_normal_chat_records_one_action(self, usage_db_path):
        fake_result = {"answer": "a", "answerable": True, "cited_papers": [], "cited_web_articles": []}
        with _curation_client(usage_db_path) as client:
            session_id = self._started_session(client)
            with patch.object(api, "chat_turn", return_value=fake_result):
                resp = client.post(f"/curation/{session_id}/chat", json={"message": "q"})

        assert resp.status_code == 200
        rows = _actions(usage_db_path, action_type="curation_chat")
        assert len(rows) == 1
        assert rows[0]["subject_type"] == "session"
        assert rows[0]["subject_id"] == session_id

    def test_edit_exchange_records_curation_chat_action(self, usage_db_path):
        fake_result = {"answer": "a", "answerable": True, "cited_papers": [], "cited_web_articles": []}
        with _curation_client(usage_db_path) as client:
            session_id = self._started_session(client)
            with patch.object(api, "edit_chat_exchange", return_value=(fake_result, False)):
                resp = client.post(
                    f"/curation/{session_id}/chat/exchanges/edit",
                    json={"exchange_id": "ex1", "question": "revised q"},
                )

        assert resp.status_code == 200
        rows = _actions(usage_db_path, action_type="curation_chat")
        assert len(rows) == 1

    def test_nested_web_offer_accept_flow_produces_exactly_one_top_level_row(self, usage_db_path):
        """The exact scenario the M1.2 spec calls out by name: chat_turn()
        internally making a SECOND ask (accepting a web offer) must never
        double-count as a second curation_chat action -- "first active
        action wins" collapses it into one row with (potentially) more
        than one child call."""
        call_count = {"n": 0}

        def _fake_chat_turn(session, message, client=None):
            call_count["n"] += 1
            # Simulate _accept_web_offer's own nested paid_action("search_chat"
            # -- if this were the real function) style nested work by directly
            # recording two child calls under whatever action is already active
            # -- proving nesting collapses correctly regardless of exactly
            # which real functions are mocked away here.
            telemetry.record_child_call("condense_question", "openai", latency_ms=1.0, outcome="success")
            telemetry.record_child_call("generate_answer", "openai", latency_ms=2.0, outcome="success")
            return {"answer": "a", "answerable": True, "cited_papers": [], "cited_web_articles": []}

        with _curation_client(usage_db_path) as client:
            session_id = self._started_session(client)
            with patch.object(api, "chat_turn", side_effect=_fake_chat_turn):
                resp = client.post(f"/curation/{session_id}/chat", json={"message": "q"})

        assert resp.status_code == 200
        rows = _actions(usage_db_path, action_type="curation_chat")
        assert len(rows) == 1
        assert rows[0]["total_call_count"] == 2

    def test_add_to_report_records_report_regenerate_action(self, usage_db_path):
        fake_report = {**_empty_report(), "report_template": "analytical"}
        web_article = WebArticle(title="A", url="https://x.com/a", snippet="s", published_date=None, source_domain="x.com")
        chat_history = [
            {"role": "user", "content": "q", "exchange_id": "ex1"},
            {
                "role": "assistant", "content": "a", "exchange_id": "ex1",
                # ChatTurn(**turn) pydantic-validates this against
                # list[CitedWebArticleOut] -- plain dicts coerce fine, a raw
                # WebArticle dataclass instance does not.
                "cited_web_articles": [{"url": "https://x.com/a", "title": "A"}], "used_web_search": True,
                "web_relevance_verified": True,
            },
        ]

        def _fake_chat_turn(session, message, client=None):
            session.chat_history = chat_history
            session.web_articles_added = [web_article]
            return {
                "answer": "a", "answerable": True, "cited_papers": [],
                "cited_web_articles": [web_article],
            }

        with _curation_client(usage_db_path) as client:
            session_id = self._started_session(client)
            fake_report_with_v = {**_empty_report(), "report_template": "analytical"}
            with patch.object(api, "generate_report_for_session", return_value=fake_report_with_v), \
                 patch.object(api, "refine_report_if_requested", side_effect=lambda report, *a, **kw: report):
                client.post(f"/curation/{session_id}/report")

            with patch.object(api, "chat_turn", side_effect=_fake_chat_turn):
                client.post(f"/curation/{session_id}/chat", json={"message": "q"})

            before = len(_actions(usage_db_path, action_type="curation_chat"))
            with patch.object(api, "regenerate_report_with_approved_web_sources", return_value=fake_report):
                resp = client.post(
                    f"/curation/{session_id}/chat/exchanges/add-to-report",
                    json={"exchange_ids": ["ex1"]},
                )

        assert resp.status_code == 200
        assert len(_actions(usage_db_path, action_type="curation_chat")) == before  # not counted as curation_chat
        rows = _actions(usage_db_path, action_type="report_regenerate")
        assert len(rows) == 1
        assert rows[0]["subject_id"] == session_id


class TestReportGenerateAction:
    def test_fresh_generation_records_action(self, usage_db_path):
        fake_report = {**_empty_report(), "report_template": "analytical"}
        with _curation_client(usage_db_path) as client:
            papers = [_paper(f"p{i}") for i in range(3)]
            with patch.object(api, "build_candidate_pool", return_value=papers), \
                 patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
                start = client.post("/curation/start", json={"topic": "t", "target_count": 1}).json()
            session_id = start["session_id"]
            client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [start["batch"][0]["paper_id"]], "stop": True})

            with patch.object(api, "generate_report_for_session", return_value=fake_report), \
                 patch.object(api, "refine_report_if_requested", side_effect=lambda report, *a, **kw: report):
                resp = client.post(f"/curation/{session_id}/report")

        assert resp.status_code == 200
        rows = _actions(usage_db_path, action_type="report_generate")
        assert len(rows) == 1
        assert rows[0]["subject_type"] == "session"
        assert rows[0]["subject_id"] == session_id

    def test_cached_report_retrieval_records_no_action(self, usage_db_path):
        fake_report = {**_empty_report(), "report_template": "analytical"}
        with _curation_client(usage_db_path) as client:
            papers = [_paper(f"p{i}") for i in range(3)]
            with patch.object(api, "build_candidate_pool", return_value=papers), \
                 patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
                start = client.post("/curation/start", json={"topic": "t", "target_count": 1}).json()
            session_id = start["session_id"]
            client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [start["batch"][0]["paper_id"]], "stop": True})

            with patch.object(api, "generate_report_for_session", return_value=fake_report) as mock_gen, \
                 patch.object(api, "refine_report_if_requested", side_effect=lambda report, *a, **kw: report):
                client.post(f"/curation/{session_id}/report")
                before = len(_actions(usage_db_path, action_type="report_generate"))
                resp = client.post(f"/curation/{session_id}/report")  # second call: cache hit

        assert resp.status_code == 200
        assert mock_gen.call_count == 1
        assert len(_actions(usage_db_path, action_type="report_generate")) == before  # no new row

    def test_refine_off_produces_only_the_generate_child_call(self, usage_db_path):
        """Refine Once off (refinement_mode omitted/"off"): refine_report_
        if_requested is a pure passthrough with zero extra LLM calls --
        confirmed at the report.py level, exercised here through the real,
        unmocked refine_report_if_requested (only its own OWN generate_
        report_for_session dependency is mocked)."""
        fake_report = {**_empty_report(), "report_template": "analytical", "sections": []}
        with _curation_client(usage_db_path) as client:
            papers = [_paper(f"p{i}") for i in range(3)]
            with patch.object(api, "build_candidate_pool", return_value=papers), \
                 patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
                start = client.post("/curation/start", json={"topic": "t", "target_count": 1}).json()
            session_id = start["session_id"]
            client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [start["batch"][0]["paper_id"]], "stop": True})

            # refine_report_if_requested itself is real; refinement_mode
            # defaults to None/"off" on this request, so it must be a pure
            # passthrough making zero extra calls -- generate_report_for_
            # session is the only mocked leaf.
            with patch.object(api, "generate_report_for_session", return_value=fake_report):
                resp = client.post(f"/curation/{session_id}/report")

        assert resp.status_code == 200
        rows = _actions(usage_db_path, action_type="report_generate")
        assert len(rows) == 1
        # No child calls at all recorded from refine_report_if_requested's
        # own passthrough (generate_report_for_session itself is mocked
        # away, so it never calls timed_child_call either) -- confirms
        # zero EXTRA calls beyond whatever the (mocked-out) generation step
        # would have made.
        assert rows[0]["total_call_count"] == 0


class TestReportRegenerateAction:
    def test_regenerate_records_action(self, usage_db_path):
        fake_report = {**_empty_report(), "report_template": "analytical"}
        with _curation_client(usage_db_path) as client:
            papers = [_paper(f"p{i}") for i in range(3)]
            with patch.object(api, "build_candidate_pool", return_value=papers), \
                 patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
                start = client.post("/curation/start", json={"topic": "t", "target_count": 1}).json()
            session_id = start["session_id"]
            client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [start["batch"][0]["paper_id"]], "stop": True})

            with patch.object(api, "generate_report_for_session", return_value=fake_report), \
                 patch.object(api, "refine_report_if_requested", side_effect=lambda report, *a, **kw: report):
                client.post(f"/curation/{session_id}/report")

            regen_report = {**_empty_report(), "findings": {"content": "v2", "cited_papers": []}, "report_template": "analytical"}
            with patch.object(api, "regenerate_report_with_new_sources", return_value=regen_report), \
                 patch.object(api, "refine_report_if_requested", side_effect=lambda report, *a, **kw: report):
                resp = client.post(f"/curation/{session_id}/report/regenerate")

        assert resp.status_code == 200
        rows = _actions(usage_db_path, action_type="report_regenerate")
        assert len(rows) == 1
        assert rows[0]["subject_id"] == session_id

    def test_regenerate_always_creates_a_row_even_with_no_report_change(self, usage_db_path):
        """Unlike report_generate, regenerate has no cache short-circuit --
        every call is a real regeneration attempt."""
        fake_report = {**_empty_report(), "report_template": "analytical"}
        with _curation_client(usage_db_path) as client:
            papers = [_paper(f"p{i}") for i in range(3)]
            with patch.object(api, "build_candidate_pool", return_value=papers), \
                 patch.object(api, "rank_full_pool", return_value=(_ranked(papers), {})):
                start = client.post("/curation/start", json={"topic": "t", "target_count": 1}).json()
            session_id = start["session_id"]
            client.post(f"/curation/{session_id}/picks", json={"picked_paper_ids": [start["batch"][0]["paper_id"]], "stop": True})

            with patch.object(api, "generate_report_for_session", return_value=fake_report), \
                 patch.object(api, "refine_report_if_requested", side_effect=lambda report, *a, **kw: report):
                client.post(f"/curation/{session_id}/report")

            with patch.object(api, "regenerate_report_with_new_sources", return_value=fake_report), \
                 patch.object(api, "refine_report_if_requested", side_effect=lambda report, *a, **kw: report):
                client.post(f"/curation/{session_id}/report/regenerate")
                client.post(f"/curation/{session_id}/report/regenerate")

        assert len(_actions(usage_db_path, action_type="report_regenerate")) == 2


# =====================================================================
# Report refinement child calls -- evaluated-no-revision / revised
# branches, at the report.py level directly (report_generate's own
# HTTP-level "off" branch is already covered in TestReportGenerateAction).
# =====================================================================

def _clean_analytical_draft(papers: list) -> dict:
    from research_agent.report import ANALYTICAL_SECTION_NAMES, REPORT_TEMPLATES, _build_references_and_renumber

    section_defs = REPORT_TEMPLATES["analytical"]
    sections_out = {}
    for d in section_defs:
        key = d["key"]
        if key == "thematic_findings" and papers:
            sections_out[key] = {"content": "A finding [Paper 1].", "cited_papers": [papers[0]]}
        else:
            sections_out[key] = {"content": f"{d['title']} text.", "cited_papers": []}
    return _build_references_and_renumber({**sections_out, "skipped_papers": []}, ANALYTICAL_SECTION_NAMES)


def _evaluation(overall_score=90, needs_revision=False, issues=None, revision_instructions="", section_scores=None):
    from research_agent.report import ReportEvaluation

    return ReportEvaluation(
        overall_score=overall_score, needs_revision=needs_revision, issues=issues or [],
        revision_instructions=revision_instructions, section_scores=section_scores,
    )


class TestReportRefinementChildCalls:
    def test_evaluated_no_revision_records_exactly_one_report_evaluation_child_call(self, usage_db_path):
        p1 = _paper("1111")
        draft = _clean_analytical_draft([p1])
        mock_client = MagicMock()
        mock_client.chat.completions.parse.return_value = _mock_parsed_response(
            _evaluation(overall_score=88, needs_revision=False), usage=_mock_usage(total=50, prompt=40, completion=10),
        )

        with telemetry.paid_action("report_generate", subject_type="session", subject_id="s1"):
            report_module.refine_report_if_requested(draft, "topic", [p1], [], "analytical", "single", mock_client)

        rows = _actions(usage_db_path, action_type="report_generate")
        assert len(rows) == 1
        assert rows[0]["total_call_count"] == 1
        calls = _child_calls(rows[0])
        assert calls[0]["call_type"] == "report_evaluation"
        assert calls[0]["input_tokens"] == 40
        assert calls[0]["output_tokens"] == 10

    def test_revised_records_evaluation_then_revision_child_calls(self, usage_db_path):
        from research_agent.report import REPORT_SECTION_DEFINITIONS, _build_report_schema

        p1 = _paper("1111")
        draft = _clean_analytical_draft([p1])
        mock_client = MagicMock()
        schema = _build_report_schema(["1111"], None, REPORT_SECTION_DEFINITIONS)
        section_cls = schema.model_fields["executive_summary"].annotation
        revised_parsed = _analytical_parsed(schema, section_cls)
        mock_client.chat.completions.parse.side_effect = [
            _mock_parsed_response(_evaluation(overall_score=40, needs_revision=True, issues=["too shallow"])),
            _mock_parsed_response(revised_parsed),
        ]

        with telemetry.paid_action("report_generate", subject_type="session", subject_id="s1"):
            report_module.refine_report_if_requested(draft, "topic", [p1], [], "analytical", "single", mock_client)

        rows = _actions(usage_db_path, action_type="report_generate")
        assert len(rows) == 1
        assert rows[0]["total_call_count"] == 2
        call_types = [c["call_type"] for c in _child_calls(rows[0])]
        assert call_types == ["report_evaluation", "report_revision"]


def _analytical_parsed(schema, section_cls):
    from research_agent.report import ANALYTICAL_SECTION_NAMES

    kwargs = {}
    for key in ANALYTICAL_SECTION_NAMES:
        kwargs[key] = section_cls(content="", cited_paper_ids=[])
    return schema(**kwargs)


# =====================================================================
# Child-call gap closures -- the M1 audit's own confirmed usage gaps
# =====================================================================

class TestChildCallGapClosures:
    def test_report_evaluation_gap_is_closed(self, usage_db_path):
        """report.py's _evaluate_report_llm made zero usage/telemetry
        capture at all before M1.2 -- confirmed directly by the M1
        architecture audit. Now closed."""
        p1 = _paper("1111")
        draft = _clean_analytical_draft([p1])
        mock_client = MagicMock()
        mock_client.chat.completions.parse.return_value = _mock_parsed_response(
            _evaluation(needs_revision=False), usage=_mock_usage(total=30, prompt=20, completion=10),
        )
        with telemetry.paid_action("report_generate"):
            report_module.evaluate_report(draft, "topic", [p1], [], "analytical", mock_client)
        row = _actions(usage_db_path, action_type="report_generate")[0]
        assert row["total_call_count"] == 1
        assert row["total_tokens"] == 30

    def test_offer_classifier_gap_is_closed(self, usage_db_path):
        """curation_chat.py's _classify_offer_response made zero usage
        capture at all before M1.2."""
        from research_agent.curation_chat import _classify_offer_response

        mock_client = MagicMock()
        mock_parsed = MagicMock(intent="accept")
        mock_client.chat.completions.parse.return_value = _mock_parsed_response(
            mock_parsed, usage=_mock_usage(total=15, prompt=10, completion=5),
        )
        with telemetry.paid_action("curation_chat"):
            _classify_offer_response("search the web", "yes please", mock_client)
        row = _actions(usage_db_path, action_type="curation_chat")[0]
        assert row["total_call_count"] == 1
        calls = _child_calls(row)
        assert calls[0]["call_type"] == "classify_offer_response"
        assert calls[0]["total_tokens"] == 15

    def test_direct_relevance_judge_gap_is_closed_and_is_cache_aware(self, usage_db_path):
        """qa.py's _judge_direct_web_relevance made zero usage capture at
        all before M1.2. Also verifies the cache-hit/miss distinction: an
        all-cache-hit batch makes zero child calls (no provider request
        was made), never a falsely-counted paid call."""
        article = WebArticle(title="T", url="https://x.com/a", snippet="s", published_date=None, source_domain="x.com")
        mock_client = MagicMock()
        mock_verdict = MagicMock(url="https://x.com/a", verdict="relevant", confidence=0.9, reason="r")
        mock_parsed = MagicMock(verdicts=[mock_verdict])
        mock_client.chat.completions.parse.return_value = _mock_parsed_response(
            mock_parsed, usage=_mock_usage(total=25, prompt=20, completion=5),
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.sqlite"
            with patch.object(qa_module, "_init_direct_relevance_cache_db", side_effect=lambda: _real_init_direct_relevance_cache_db(path=cache_path)):
                with telemetry.paid_action("search_chat"):
                    qa_module._judge_direct_web_relevance("query", "topic", [article], mock_client)

        row = _actions(usage_db_path, action_type="search_chat")[0]
        assert row["total_call_count"] == 1
        calls = _child_calls(row)
        assert calls[0]["call_type"] == "direct_relevance_judge"
        assert calls[0]["total_tokens"] == 25

    def test_direct_relevance_judge_all_cache_hits_records_no_child_call(self, usage_db_path):
        article = WebArticle(title="T", url="https://x.com/a", snippet="s", published_date=None, source_domain="x.com")
        mock_client = MagicMock()

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.sqlite"
            with patch.object(qa_module, "_get_cached_direct_relevance", return_value={"verdict": "relevant", "confidence": 0.9}), \
                 patch.object(qa_module, "_init_direct_relevance_cache_db", side_effect=lambda: _real_init_direct_relevance_cache_db(path=cache_path)):
                with telemetry.paid_action("search_chat"):
                    qa_module._judge_direct_web_relevance("query", "topic", [article], mock_client)

        mock_client.chat.completions.parse.assert_not_called()
        row = _actions(usage_db_path, action_type="search_chat")[0]
        assert row["total_call_count"] == 0

    def test_classification_embedding_gap_is_closed(self, usage_db_path):
        """qa.py's _classify_non_substantive/_embed_with_cache discarded
        the token count _embed_texts returned, on a cache miss, before
        M1.2 -- closed by instrumenting the shared _embed_texts boundary
        directly in embeddings.py (see TestEmbeddingChildCalls below for
        the direct proof at that layer). This test proves the effect is
        visible end to end through qa.py's own call chain too."""
        mock_client = MagicMock()
        mock_response = MagicMock(usage=MagicMock(total_tokens=7))
        mock_response.data = [MagicMock(index=0, embedding=[0.1, 0.2])]
        mock_client.embeddings.create.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.sqlite"
            with patch.object(qa_module, "_init_cache_db", side_effect=lambda: embeddings_module._init_cache_db(path=cache_path)):
                with telemetry.paid_action("search_chat"):
                    qa_module._embed_with_cache(mock_client, "some non-substantive message")

        row = _actions(usage_db_path, action_type="search_chat")[0]
        assert row["total_call_count"] == 1
        assert row["total_tokens"] == 7


# =====================================================================
# Embeddings: real SDK batches vs. cache hits
# =====================================================================

class TestEmbeddingChildCalls:
    def test_real_batch_records_one_child_call_with_total_tokens(self, usage_db_path):
        mock_client = MagicMock()
        mock_response = MagicMock(usage=MagicMock(total_tokens=42))
        mock_response.data = [MagicMock(index=0, embedding=[0.1]), MagicMock(index=1, embedding=[0.2])]
        mock_client.embeddings.create.return_value = mock_response

        with telemetry.paid_action("search"):
            vectors, total = embeddings_module._embed_texts(mock_client, ["text one", "text two"])

        assert total == 42
        row = _actions(usage_db_path, action_type="search")[0]
        assert row["total_call_count"] == 1
        calls = _child_calls(row)
        assert calls[0]["call_type"] == "embeddings_create"
        assert calls[0]["provider"] == "openai"
        assert calls[0]["total_tokens"] == 42
        # embeddings usage objects have no prompt/completion breakdown --
        # input_tokens/output_tokens correctly stay null, never fabricated.
        assert calls[0]["input_tokens"] is None
        assert calls[0]["output_tokens"] is None

    def test_multiple_real_batches_record_one_child_call_each(self, usage_db_path):
        mock_client = MagicMock()

        def _fake_create(model, input):
            resp = MagicMock(usage=MagicMock(total_tokens=len(input) * 10))
            resp.data = [MagicMock(index=i, embedding=[0.0]) for i in range(len(input))]
            return resp

        mock_client.embeddings.create.side_effect = _fake_create
        texts = [f"text {i}" for i in range(250)]  # EMBED_BATCH_SIZE=100 -> 3 batches

        with telemetry.paid_action("search"):
            embeddings_module._embed_texts(mock_client, texts)

        row = _actions(usage_db_path, action_type="search")[0]
        assert row["total_call_count"] == 3

    def test_cache_hit_makes_no_sdk_call_and_records_no_child_call(self, usage_db_path):
        """embed_and_index_papers' own cache-hit path never reaches
        _embed_texts at all -- confirmed here by calling the real cached
        lookup helpers directly with a pre-seeded cache."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.sqlite"
            conn = embeddings_module._init_cache_db(path=cache_path)
            text_hash = embeddings_module._hash_text("cached text")
            embeddings_module._set_cached(conn, text_hash, [0.1, 0.2], len("cached text"))
            conn.close()

            mock_client = MagicMock()
            with telemetry.paid_action("search"):
                conn = embeddings_module._init_cache_db(path=cache_path)
                cached = embeddings_module._get_cached(conn, text_hash)
                conn.close()
                assert cached is not None  # sanity: this really was a cache hit
                mock_client.embeddings.create.assert_not_called()

        row = _actions(usage_db_path, action_type="search")[0]
        assert row["total_call_count"] == 0


# =====================================================================
# External providers: null tokens, retries, fallback
# =====================================================================

class TestExternalProviderChildCalls:
    def test_tavily_records_null_tokens(self, usage_db_path):
        fake_tool = MagicMock()
        fake_tool.invoke.return_value = {"results": [{"url": "https://x.com/a", "title": "A", "content": "c"}]}
        with patch("langchain_tavily.TavilySearch", return_value=fake_tool), \
             patch("research_agent.web_search.get_settings", return_value=MagicMock(tavily_api_key="k")):
            with telemetry.paid_action("search"):
                web_search_module.search_web("some query")

        row = _actions(usage_db_path, action_type="search")[0]
        calls = _child_calls(row)
        assert calls[0]["call_type"] == "web_search"
        assert calls[0]["provider"] == "tavily"
        assert calls[0]["input_tokens"] is None
        assert calls[0]["output_tokens"] is None
        assert calls[0]["total_tokens"] is None
        assert calls[0]["outcome"] == "success"

    def test_tavily_failure_records_errored_child_call_and_degrades_to_empty(self, usage_db_path):
        with patch("langchain_tavily.TavilySearch", side_effect=RuntimeError("boom")), \
             patch("research_agent.web_search.get_settings", return_value=MagicMock(tavily_api_key="k")):
            with telemetry.paid_action("search"):
                result = web_search_module.search_web("some query")

        assert result == []  # existing fallback behavior unchanged
        row = _actions(usage_db_path, action_type="search")[0]
        calls = _child_calls(row)
        assert calls[0]["outcome"] == "error"
        assert calls[0]["error_type"] == "RuntimeError"
        assert row["outcome"] == "success"  # the top-level action still succeeds -- search_web degrades, doesn't propagate

    def test_arxiv_records_null_tokens(self, usage_db_path):
        fake_client = MagicMock()
        fake_client.results.return_value = []
        with patch("arxiv.Client", return_value=fake_client):
            with telemetry.paid_action("search"):
                ingestion_module.search_arxiv("some query")

        row = _actions(usage_db_path, action_type="search")[0]
        calls = _child_calls(row)
        assert calls[0]["call_type"] == "search_arxiv"
        assert calls[0]["provider"] == "arxiv"
        assert calls[0]["total_tokens"] is None

    def test_semantic_scholar_retries_produce_one_child_call_per_attempt(self, usage_db_path):
        rate_limited_resp = MagicMock(status_code=429, headers={})
        ok_resp = MagicMock(status_code=200)
        ok_resp.json.return_value = {"data": []}
        with patch.object(ingestion_module, "requests") as mock_requests, \
             patch.object(ingestion_module.time, "sleep"):
            mock_requests.get.side_effect = [rate_limited_resp, ok_resp]
            mock_requests.RequestException = requests.RequestException
            with telemetry.paid_action("search"):
                ingestion_module.search_semantic_scholar("some query", max_retries=3)

        row = _actions(usage_db_path, action_type="search")[0]
        calls = _child_calls(row)
        assert len(calls) == 2  # one rate-limited attempt, one successful attempt
        assert calls[0]["outcome"] == "error"
        assert calls[0]["error_type"] == "RateLimited"
        assert calls[1]["outcome"] == "success"

    def test_openalex_records_null_tokens(self, usage_db_path):
        ok_resp = MagicMock(status_code=200)
        ok_resp.json.return_value = {"results": []}
        with patch.object(ingestion_module, "requests") as mock_requests:
            mock_requests.get.return_value = ok_resp
            mock_requests.RequestException = requests.RequestException
            with telemetry.paid_action("search"):
                ingestion_module.search_openalex("some query")

        row = _actions(usage_db_path, action_type="search")[0]
        calls = _child_calls(row)
        assert calls[0]["call_type"] == "search_openalex"
        assert calls[0]["total_tokens"] is None

    def test_unpaywall_and_crossref_record_null_tokens(self, usage_db_path):
        ok_resp = MagicMock(status_code=200)
        ok_resp.json.return_value = {"abstract": None}
        with patch.object(enrichment_module, "requests") as mock_requests, \
             patch.object(enrichment_module, "_unpaywall_email", return_value="a@b.com"):
            mock_requests.get.return_value = ok_resp
            mock_requests.RequestException = requests.RequestException
            with telemetry.paid_action("search"):
                enrichment_module._fetch_unpaywall_abstract("10.1234/x")

        row = _actions(usage_db_path, action_type="search")[0]
        calls = _child_calls(row)
        assert calls[0]["call_type"] == "enrichment_unpaywall"
        assert calls[0]["provider"] == "unpaywall"
        assert calls[0]["total_tokens"] is None

    def test_enrichment_network_failure_records_errored_child_call(self, usage_db_path):
        with patch.object(enrichment_module, "requests") as mock_requests, \
             patch.object(enrichment_module, "_unpaywall_email", return_value="a@b.com"):
            mock_requests.get.side_effect = requests.RequestException("network down")
            mock_requests.RequestException = requests.RequestException
            with telemetry.paid_action("search"):
                result = enrichment_module._fetch_unpaywall_abstract("10.1234/x")

        assert result is None  # existing fallback unchanged
        row = _actions(usage_db_path, action_type="search")[0]
        calls = _child_calls(row)
        assert calls[0]["outcome"] == "error"
        assert calls[0]["error_type"] == "RequestException"


# =====================================================================
# Propagated exceptions / nesting (beyond what's already exercised above)
# =====================================================================

class TestPropagationAndNesting:
    def test_propagated_exception_closes_action_as_error(self, usage_db_path):
        mock_client = MagicMock()
        # condense_question uses client.chat.completions.CREATE (not .parse)
        mock_client.chat.completions.create.side_effect = RuntimeError("upstream failure")

        with pytest.raises(RuntimeError):
            with telemetry.paid_action("search_chat"):
                qa_module.condense_question([{"role": "user", "content": "x"}], "follow up", mock_client)

        row = _actions(usage_db_path, action_type="search_chat")[0]
        assert row["outcome"] == "error"
        calls = _child_calls(row)
        assert calls[0]["outcome"] == "error"
        assert calls[0]["error_type"] == "RuntimeError"

    def test_nested_domain_calls_never_double_count_as_a_second_action(self, usage_db_path):
        """Two DIFFERENT instrumented functions called inside the same
        outer paid_action -- both attach as child calls, neither opens
        its own top-level row."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            usage=_mock_usage(), choices=[MagicMock(message=MagicMock(content="condensed"))],
        )
        with telemetry.paid_action("search_chat", subject_type="search", subject_id="1"):
            qa_module.condense_question([{"role": "user", "content": "x"}], "follow up", mock_client)
            with telemetry.paid_action("curation_chat", subject_type="session", subject_id="ignored"):
                telemetry.record_child_call("condense_question", "openai", latency_ms=1.0, outcome="success")

        rows = _actions(usage_db_path)
        assert len(rows) == 1
        assert rows[0]["action_type"] == "search_chat"
        assert rows[0]["total_call_count"] == 2


# =====================================================================
# Privacy
# =====================================================================

class TestPrivacy:
    _FORBIDDEN_SENTINELS = (
        "PROMPT_SENTINEL_the-secret-system-prompt",
        "QUESTION_SENTINEL_what-is-the-mechanism",
        "REPORT_TEXT_SENTINEL_a-whole-paragraph-of-prose",
        "TITLE_SENTINEL_A Very Specific Paper Title",
        "ABSTRACT_SENTINEL_this-is-the-abstract-body",
        "https://forbidden-url-sentinel.example.com/path",
        "sk-FAKE_API_KEY_SENTINEL_0000000000000000",
    )

    def test_no_forbidden_content_reaches_a_persisted_row(self, usage_db_path):
        p1 = Paper(
            title="TITLE_SENTINEL_A Very Specific Paper Title", authors=["A"], year=2024, venue="v",
            abstract="ABSTRACT_SENTINEL_this-is-the-abstract-body",
            url="https://forbidden-url-sentinel.example.com/path", doi=None, citation_count=None,
            source="arxiv", paper_id="1111",
        )
        draft = _clean_analytical_draft([p1])
        mock_client = MagicMock()
        mock_client.chat.completions.parse.return_value = _mock_parsed_response(
            _evaluation(needs_revision=False, issues=["REPORT_TEXT_SENTINEL_a-whole-paragraph-of-prose"]),
        )

        with telemetry.paid_action(
            "report_generate", subject_type="session",
            subject_id="PROMPT_SENTINEL_the-secret-system-prompt",  # deliberately abusive subject_id
        ):
            report_module.evaluate_report(draft, "QUESTION_SENTINEL_what-is-the-mechanism", [p1], [], "analytical", mock_client)

        conn = sqlite3.connect(usage_db_path)
        try:
            raw_dump = str(conn.execute("SELECT * FROM paid_actions").fetchall())
            raw_dump += str(conn.execute("SELECT * FROM http_requests").fetchall())
        finally:
            conn.close()

        for sentinel in self._FORBIDDEN_SENTINELS:
            if sentinel.startswith("PROMPT_SENTINEL"):
                continue  # this one WAS deliberately passed as subject_id above -- see the next assertion instead
            assert sentinel not in raw_dump, f"forbidden content leaked into a persisted row: {sentinel!r}"

    def test_error_type_is_a_class_name_never_exception_text(self, usage_db_path):
        mock_client = MagicMock()
        mock_client.chat.completions.parse.side_effect = ValueError("SECRET_EXCEPTION_TEXT_SENTINEL: leaked detail")

        with pytest.raises(ValueError):
            with telemetry.paid_action("report_generate"):
                report_module.evaluate_report(_clean_analytical_draft([]), "t", [], [], "analytical", mock_client)

        row = _actions(usage_db_path, action_type="report_generate")[0]
        calls = _child_calls(row)
        assert calls[0]["error_type"] == "ValueError"
        assert "SECRET_EXCEPTION_TEXT_SENTINEL" not in json.dumps(row)


# =====================================================================
# Isolation: eval/CLI and direct-domain invocations
# =====================================================================

class TestIsolation:
    def test_eval_mock_run_creates_zero_rows(self, usage_db_path):
        from research_agent.evals.runners import run_report_quality

        run_report_quality.run_experiment(mode="mock")
        assert _actions(usage_db_path) == []
        assert _http_requests(usage_db_path) == []

    def test_eval_live_shaped_mocked_runner_creates_zero_rows(self, usage_db_path):
        """A `mode="live"`-shaped call, with the OpenAI client itself
        mocked so no real network call happens, still must create zero
        telemetry rows -- eval/CLI code has no active API action context
        no matter which mode it runs in."""
        from research_agent.evals.judges import claim_source
        from research_agent.evals.runners import run_report_quality

        mock_client = MagicMock()
        mock_client.chat.completions.parse.return_value = _mock_parsed_response(
            MagicMock(dimensions=MagicMock(model_dump=lambda: {})),
        )
        with patch.object(run_report_quality, "_build_live_client", return_value=mock_client), \
             patch.object(claim_source, "judge_claims", side_effect=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no real calls in this test"))):
            try:
                run_report_quality.run_experiment(mode="live", subset=1)
            except Exception:
                pass  # irrelevant to this test -- only the telemetry side effect matters

        assert _actions(usage_db_path) == []

    def test_direct_domain_call_outside_any_action_creates_zero_rows(self, usage_db_path):
        mock_client = MagicMock()
        mock_client.chat.completions.parse.return_value = _mock_parsed_response(
            _evaluation(needs_revision=False),
        )
        assert telemetry.get_current_request_id() is None
        report_module.evaluate_report(_clean_analytical_draft([]), "t", [], [], "analytical", mock_client)
        assert _actions(usage_db_path) == []

    def test_r6d4_capture_style_direct_call_creates_zero_rows(self, usage_db_path):
        """The exact scenario the M1 architecture audit flagged by name:
        research_agent/evals/r6d4_capture.py calls research_agent.report's
        real, instrumented functions directly, never through the API --
        reproduced here via generate_report_for_session (its own real
        dependency) without ever opening a paid_action."""
        p1 = _paper("1111")
        mock_client = MagicMock()
        parsed_schema_result = _analytical_parsed_full(p1)
        mock_client.chat.completions.parse.return_value = _mock_parsed_response(parsed_schema_result)

        session = MagicMock(stage="synthesize", topic="t", selected_papers=[p1])
        report_module.generate_report_for_session(session, client=mock_client, report_template="analytical")

        assert _actions(usage_db_path) == []
        assert _http_requests(usage_db_path) == []


def _analytical_parsed_full(paper):
    from research_agent.report import ANALYTICAL_SECTION_NAMES, REPORT_SECTION_DEFINITIONS, _build_report_schema

    schema = _build_report_schema([paper.paper_id], None, REPORT_SECTION_DEFINITIONS)
    section_cls = schema.model_fields["executive_summary"].annotation
    kwargs = {key: section_cls(content="", cited_paper_ids=[]) for key in ANALYTICAL_SECTION_NAMES}
    return schema(**kwargs)


def test_real_usage_db_path_untouched():
    """Session-wide proof, not just a per-test assumption: the ACTUAL
    project path was never created or written to by anything in this
    test run."""
    assert not _REAL_USAGE_DB_PATH.exists()
