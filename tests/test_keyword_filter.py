"""K5D.2 tests for research_agent/keyword_filter.py -- the production-
owned, off-by-default Policy C filter. No test here makes a real
network/provider call; every provider-facing test uses a fake client.

Contract equivalence against scripts/k5_heldout_llm_prep.py is a TEST-
ONLY import -- production code (research_agent/keyword_filter.py
itself) never imports scripts/, proven by test_production_module_never_
imports_from_scripts below (a real source-level check, not just true by
inspection today).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research_agent.keyword_filter as kf
from scripts import k5_heldout_llm_prep as prep
from tests._usage_db_fingerprint import fingerprint_usage_db

_REAL_CACHE_DB_PATH = kf.CACHE_DB_PATH
_REAL_CACHE_DB_FINGERPRINT_BEFORE = fingerprint_usage_db(_REAL_CACHE_DB_PATH)


@pytest.fixture(autouse=True)
def default_cache_path_redirect(tmp_path, monkeypatch):
    """Every test in this file that calls plan_paper()/resolve_batch()
    WITHOUT an explicit cache_path= must still never touch the real,
    production data/cache/keyword_filter_cache.sqlite -- redirect the
    module's own default resolution target instead of trusting every
    call site to remember to pass cache_path=tmp_path/... by hand."""
    monkeypatch.setattr(kf, "CACHE_DB_PATH", tmp_path / "keyword_filter_cache.sqlite")


# ---------------------------------------------------------------------------
# Production ownership boundary
# ---------------------------------------------------------------------------

def test_production_module_never_imports_from_scripts():
    source_path = Path(kf.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.split(".")[0] == "scripts", f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module.split(".")[0] != "scripts", f"forbidden import from: {module}"


def test_no_research_agent_production_module_imports_scripts():
    """Broader regression guard: nothing under research_agent/ (not just
    keyword_filter.py) may import the gitignored, non-shipped scripts/
    evaluation harness."""
    research_agent_dir = Path(kf.__file__).resolve().parent
    for path in research_agent_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] != "scripts", f"{path}: forbidden import {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module.split(".")[0] != "scripts", f"{path}: forbidden import from {module}"


# ---------------------------------------------------------------------------
# Contract equivalence against the frozen, validated K5D experiment
# ---------------------------------------------------------------------------

def test_system_prompt_matches_frozen_contract_exactly():
    assert kf.SYSTEM_PROMPT == prep.SYSTEM_PROMPT


def test_decisions_and_model_settings_match_frozen_contract():
    assert kf.DECISIONS == prep.DECISIONS
    assert kf.REMOVE_DECISIONS == prep.REMOVE_DECISIONS
    assert set(kf.DECISIONS) - kf.REMOVE_DECISIONS == {"keep", "uncertain"}
    assert kf.MODEL == prep.MODEL == "gpt-4.1-mini"
    assert kf.TEMPERATURE == prep.TEMPERATURE == 0
    assert kf.PROMPT_VERSION == prep.PROMPT_VERSION


def test_message_shape_matches_frozen_contract():
    frozen_messages = prep.build_messages({"opaque_paper_id": "X", "candidates": [{"candidate_id": "C0", "phrase": "p"}]})
    production_messages = kf.build_messages("X", [{"candidate_id": "C0", "phrase": "p"}])
    assert frozen_messages == production_messages
    assert [m["role"] for m in production_messages] == ["system", "user"]
    assert json.loads(production_messages[1]["content"]) == {"opaque_paper_id": "X", "candidates": [{"candidate_id": "C0", "phrase": "p"}]}


def test_response_schema_accepts_and_rejects_the_same_shapes_as_frozen_contract():
    ids = ["C0", "C1"]
    frozen_schema = prep.build_response_schema(ids)
    production_schema = kf.build_response_schema(ids)
    valid_rows = [{"candidate_id": "C0", "decision": "keep"}, {"candidate_id": "C1", "decision": "uncertain"}]
    assert len(frozen_schema(results=valid_rows).results) == len(production_schema(results=valid_rows).results) == 2
    invented = [{"candidate_id": "C0", "decision": "keep"}, {"candidate_id": "INVENTED", "decision": "keep"}]
    with pytest.raises(ValidationError):
        frozen_schema(results=invented)
    with pytest.raises(ValidationError):
        production_schema(results=invented)


def test_policy_application_behaves_identically_to_frozen_contract():
    ids = ["C0", "C1", "C2", "C3"]
    decisions_by_id = {"C0": "keep", "C1": "malformed_fragment", "C2": "sentence_fragment", "C3": "uncertain"}
    frozen_call = {"status": "success", "results": [{"candidate_id": cid, "decision": d} for cid, d in decisions_by_id.items()]}
    frozen_removed = prep.apply_policy_c(frozen_call, ids)
    production_removed = kf.apply_policy_c(decisions_by_id, ids)
    assert frozen_removed == production_removed == {"C1", "C2"}


def test_no_forbidden_provider_fields_in_either_contract():
    with pytest.raises(ValueError, match="forbidden candidate field"):
        kf.build_messages("X", [{"candidate_id": "C0", "phrase": "p", "title": "leak"}])
    with pytest.raises(ValueError, match="forbidden candidate field"):
        kf.build_messages("X", [{"candidate_id": "C0", "phrase": "p", "session_id": "leak"}])


# ---------------------------------------------------------------------------
# Cache: exact ordered key, no raw content stored, fail-open
# ---------------------------------------------------------------------------

def test_cache_key_is_order_sensitive_not_sorted():
    key_ab = kf.cache_key_for(["alpha", "beta"])
    key_ba = kf.cache_key_for(["beta", "alpha"])
    assert key_ab != key_ba


def test_cache_key_changes_with_model_prompt_or_policy_version(monkeypatch):
    base = kf.cache_key_for(["alpha", "beta"])
    monkeypatch.setattr(kf, "MODEL", "gpt-different")
    assert kf.cache_key_for(["alpha", "beta"]) != base
    monkeypatch.setattr(kf, "MODEL", kf.MODEL)  # restore before next patch to isolate variables


def test_cache_key_changes_with_prompt_version(monkeypatch):
    base = kf.cache_key_for(["alpha", "beta"])
    monkeypatch.setattr(kf, "PROMPT_VERSION", "some-other-version")
    assert kf.cache_key_for(["alpha", "beta"]) != base


def test_cache_key_changes_with_policy_version(monkeypatch):
    base = kf.cache_key_for(["alpha", "beta"])
    monkeypatch.setattr(kf, "POLICY_VERSION", "policy-c-v2")
    assert kf.cache_key_for(["alpha", "beta"]) != base


def test_cache_key_identical_phrases_same_order_produce_same_key():
    assert kf.cache_key_for(["alpha", "beta"]) == kf.cache_key_for(["alpha", "beta"])


def test_cache_hit_causes_zero_provider_calls(tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    plan = kf.plan_paper(["alpha beta", "gamma delta"], cache_path=cache_path)
    kf._cache_set(plan.cache_key, {"C0": "keep", "C1": "sentence_fragment"}, cache_path)

    calls = []

    class _ExplodingClient:
        class chat:
            class completions:
                @staticmethod
                def parse(**kwargs):
                    calls.append(kwargs)
                    raise AssertionError("provider must not be called on a cache hit")

    out = kf.resolve_batch(_ExplodingClient(), [kf.plan_paper(["alpha beta", "gamma delta"], cache_path=cache_path)], max_concurrent=3)
    assert calls == []
    assert out == [["alpha beta"]]


def test_cache_stores_only_hashed_key_versions_decisions_and_timestamp_no_raw_content(tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    plan = kf.plan_paper(["a real phrase that must never be stored"], cache_path=cache_path)
    kf._cache_set(plan.cache_key, {"C0": "keep"}, cache_path)

    conn = sqlite3.connect(cache_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(keyword_filter_cache)").fetchall()}
    assert columns == {"cache_key", "model", "prompt_version", "policy_version", "decisions_json", "created_at"}
    row = conn.execute("SELECT * FROM keyword_filter_cache").fetchone()
    conn.close()
    blob = json.dumps(row)
    assert "a real phrase" not in blob
    assert "session" not in blob.lower()


def test_corrupted_cache_db_fails_open_to_a_miss(tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    cache_path.write_bytes(b"not a sqlite database")
    plan = kf.plan_paper(["alpha beta"], cache_path=cache_path)
    assert plan.cached_decisions is None
    assert kf.needs_provider_work([plan]) is True


def test_cache_write_failure_does_not_undo_a_successful_filter(tmp_path, monkeypatch):
    unwritable_dir = tmp_path / "readonly"
    unwritable_dir.mkdir()
    cache_path = unwritable_dir / "sub" / "cache.sqlite"  # parent doesn't exist and mkdir will be blocked

    def _boom(*_a, **_k):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(kf.Path, "mkdir", _boom)
    plan = kf.plan_paper(["alpha beta"], cache_path=cache_path)
    assert plan.cached_decisions is None  # read failed open too

    class _FakeCompletions:
        def parse(self, **kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            rows = [{"candidate_id": c["candidate_id"], "decision": "keep"} for c in payload["candidates"]]
            parsed = kwargs["response_format"](results=rows)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    out = kf.resolve_batch(client, [plan], max_concurrent=1)
    assert out == [["alpha beta"]]  # "keep" -> retained; the cache write failure never surfaces


# ---------------------------------------------------------------------------
# Batch behavior
# ---------------------------------------------------------------------------

def test_empty_paper_keyword_list_is_a_pure_noop():
    plan = kf.plan_paper([])
    assert plan.candidate_ids == []
    assert kf.needs_provider_work([plan]) is False
    assert kf.resolve_batch(None, [plan], max_concurrent=3) == [[]]


def test_maximum_ten_calls_for_a_ten_paper_fully_uncached_batch(tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    plans = [kf.plan_paper([f"p{i} one", f"p{i} two"], cache_path=cache_path) for i in range(10)]
    assert kf.needs_provider_work(plans) is True

    call_count = {"n": 0}

    class _FakeCompletions:
        def parse(self, **kwargs):
            call_count["n"] += 1
            payload = json.loads(kwargs["messages"][1]["content"])
            rows = [{"candidate_id": c["candidate_id"], "decision": "keep"} for c in payload["candidates"]]
            parsed = kwargs["response_format"](results=rows)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    out = kf.resolve_batch(client, plans, max_concurrent=3)
    assert call_count["n"] == 10
    assert len(out) == 10
    for i, row in enumerate(out):
        assert row == [f"p{i} one", f"p{i} two"]  # all "keep"


def test_bounded_concurrency_preserves_paper_and_keyword_order(tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    plans = [kf.plan_paper([f"p{i} keep", f"p{i} drop"], cache_path=cache_path) for i in range(6)]
    active = {"current": 0, "max_seen": 0}
    lock_free_max_concurrent = 2

    class _FakeCompletions:
        def parse(self, **kwargs):
            active["current"] += 1
            active["max_seen"] = max(active["max_seen"], active["current"])
            time.sleep(0.01)
            payload = json.loads(kwargs["messages"][1]["content"])
            rows = [
                {"candidate_id": c["candidate_id"], "decision": "keep" if c["phrase"].endswith("keep") else "sentence_fragment"}
                for c in payload["candidates"]
            ]
            active["current"] -= 1
            parsed = kwargs["response_format"](results=rows)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    out = kf.resolve_batch(client, plans, max_concurrent=lock_free_max_concurrent)
    assert active["max_seen"] <= lock_free_max_concurrent
    for i, row in enumerate(out):
        assert row == [f"p{i} keep"]


def test_never_more_workers_than_uncached_papers(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    plans = [kf.plan_paper([f"p{i} x"], cache_path=cache_path) for i in range(2)]
    seen_bound = {}

    async def _spy_bounded(client, indexed_plans, max_concurrent):
        seen_bound["value"] = max_concurrent
        return {i: p.original_keywords for i, p in indexed_plans}

    monkeypatch.setattr(kf, "_bounded_provider_calls", _spy_bounded)
    kf.resolve_batch(None, plans, max_concurrent=10)
    assert seen_bound["value"] == 2  # clamped down to the 2 actually-uncached papers, not the configured 10


def test_partial_failure_is_isolated_per_paper(tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    plans = [kf.plan_paper([f"p{i} phrase"], cache_path=cache_path) for i in range(3)]

    class _FlakyCompletions:
        def __init__(self):
            self.calls = 0

        def parse(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated outage on the second paper only")
            payload = json.loads(kwargs["messages"][1]["content"])
            rows = [{"candidate_id": c["candidate_id"], "decision": "sentence_fragment"} for c in payload["candidates"]]
            parsed = kwargs["response_format"](results=rows)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=_FlakyCompletions()))
    out = kf.resolve_batch(client, plans, max_concurrent=1)  # concurrency=1 makes call order deterministic
    assert out[0] == []       # succeeded, sentence_fragment removed
    assert out[1] == ["p1 phrase"]  # the one that errored -> original retained
    assert out[2] == []       # succeeded, unaffected by paper 1's failure


@pytest.mark.parametrize("break_response", [
    lambda kwargs: (_ for _ in ()).throw(RuntimeError("provider exception")),
    lambda kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=None, refusal="policy"))], usage=None),
])
def test_provider_error_and_refusal_fail_open(break_response, tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    plan = kf.plan_paper(["only phrase"], cache_path=cache_path)

    class _BrokenCompletions:
        def parse(self, **kwargs):
            return break_response(kwargs)

    client = SimpleNamespace(chat=SimpleNamespace(completions=_BrokenCompletions()))
    out = kf.resolve_batch(client, [plan], max_concurrent=1)
    assert out == [["only phrase"]]


def test_malformed_response_missing_duplicate_invented_id_fail_open(tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    plan = kf.plan_paper(["one", "two"], cache_path=cache_path)

    for bad_rows in (
        [{"candidate_id": "C0", "decision": "keep"}],  # missing C1
        [{"candidate_id": "C0", "decision": "keep"}, {"candidate_id": "C0", "decision": "keep"}],  # duplicate
        [{"candidate_id": "C0", "decision": "keep"}, {"candidate_id": "INVENTED", "decision": "keep"}],  # invented
        [{"candidate_id": "C0", "decision": "keep"}, {"candidate_id": "C1"}],  # missing decision
        [{"candidate_id": "C0", "decision": "keep"}, {"candidate_id": "C1", "decision": "not_a_real_decision"}],  # unsupported decision
    ):
        class _BadCompletions:
            def parse(self, **kwargs):
                # Bypass the real schema's own validation to simulate a
                # provider that returns something structurally odd, the
                # way K5C/K5D's own tests do with a malformed dict result.
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(parsed={"results": bad_rows}, refusal=None))],
                    usage=None,
                )

        client = SimpleNamespace(chat=SimpleNamespace(completions=_BadCompletions()))
        out = kf.resolve_batch(client, [plan], max_concurrent=1)
        assert out == [["one", "two"]], f"failed to fail-open for: {bad_rows}"


def test_unexpected_internal_exception_fails_open(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.sqlite"
    plan = kf.plan_paper(["one"], cache_path=cache_path)
    monkeypatch.setattr(kf, "build_response_schema", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bug")))
    out = kf.resolve_batch(SimpleNamespace(), [plan], max_concurrent=1)
    assert out == [["one"]]


# ---------------------------------------------------------------------------
# Rollback limitation, proven in code
# ---------------------------------------------------------------------------

def test_disabling_the_flag_stops_future_filtering_but_does_not_touch_a_prior_result():
    """Not itself an end-to-end curation test (see
    tests/test_curation_loop_keyword_filter.py for that) -- proves the
    narrower, module-level fact: once a batch's keywords have been
    filtered and returned, keyword_filter.py has no further handle on
    that data. Nothing in this module re-reads or rewrites a caller's
    already-returned result, so there is no code path here that could
    "roll back" a previously filtered batch even if it wanted to."""
    plan = kf.plan_paper(["kept phrase"])
    filtered_once = kf._apply_decisions_to_plan(plan, {"C0": "keep"})
    assert filtered_once == ["kept phrase"]
    # Disabling is simulated by simply never calling keyword_filter again
    # for this data -- there is no "undo" function in this module's
    # public surface to call even if a caller wanted automatic backfill.
    assert not hasattr(kf, "rollback_filtered_keywords")
    assert not hasattr(kf, "restore_original_keywords")


def test_real_keyword_filter_cache_db_untouched_by_this_file():
    assert fingerprint_usage_db(_REAL_CACHE_DB_PATH) == _REAL_CACHE_DB_FINGERPRINT_BEFORE
