"""Research Lanes (RL5): HTTP-level tests for GET /curation/capabilities.

The endpoint is a zero-provider, zero-telemetry, zero-DB read of the
RESEARCH_LANES_ENABLED flag (uncached get_settings()). These tests prove:
the flag on/off/unset/invalid behavior, that ONLY the one boolean key is
returned, that the outermost Basic Auth middleware protects it, and that a
call mutates neither the real checkpoint DB nor the real usage DB.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

import research_agent.admission as admission
import research_agent.api as api
import research_agent.leases as leases
import research_agent.telemetry as telemetry
from research_agent.qa import QA_CHECKPOINT_DB_PATH, sqlite_checkpointer
from research_agent.storage import init_db as real_init_db
from tests._usage_db_fingerprint import fingerprint_usage_db

_REAL_CHECKPOINT_DB = QA_CHECKPOINT_DB_PATH
_REAL_USAGE_DB = telemetry.USAGE_DB_PATH
_REAL_CHECKPOINT_FINGERPRINT_BEFORE = fingerprint_usage_db(_REAL_CHECKPOINT_DB)
_REAL_USAGE_FINGERPRINT_BEFORE = fingerprint_usage_db(_REAL_USAGE_DB)


def _make_db_override(db_path: Path):
    def _override():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    return _override


def _make_cp_override(cp_db_path: Path):
    def _override():
        with sqlite_checkpointer(cp_db_path) as cp:
            yield cp

    return _override


@contextmanager
def _client(flag: str | None = "true"):
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.sqlite"
        cp_db_path = Path(tmp) / "test_checkpoints.sqlite"
        usage_db_path = Path(tmp) / "usage_telemetry.sqlite"
        env = {} if flag is None else {"RESEARCH_LANES_ENABLED": flag}
        with patch.dict(os.environ, env), \
             patch.object(api, "init_db", lambda: real_init_db(db_path)), \
             patch.object(telemetry, "USAGE_DB_PATH", usage_db_path), \
             patch.object(admission, "USAGE_DB_PATH", usage_db_path), \
             patch.object(leases, "USAGE_DB_PATH", usage_db_path), \
             patch.object(api, "OpenAI", return_value=MagicMock(name="fake_openai_client")), \
             patch.object(api, "get_chroma_collection", return_value=MagicMock(name="fake_chroma_collection")):
            api.app.dependency_overrides[api.get_db_connection] = _make_db_override(db_path)
            api.app.dependency_overrides[api.get_curation_checkpointer] = _make_cp_override(cp_db_path)
            try:
                with TestClient(api.app) as client:
                    yield client, usage_db_path, cp_db_path
            finally:
                api.app.dependency_overrides.clear()


def _paid_rows(usage_db_path: Path) -> list:
    conn = sqlite3.connect(usage_db_path)
    try:
        return list(conn.execute("SELECT * FROM paid_actions"))
    finally:
        conn.close()


def test_capability_true_when_flag_enabled():
    with _client(flag="true") as (client, _usage, _cp):
        resp = client.get("/curation/capabilities")
        assert resp.status_code == 200
        assert resp.json() == {"research_lanes_enabled": True}


def test_capability_false_when_flag_disabled():
    with _client(flag="false") as (client, _usage, _cp):
        resp = client.get("/curation/capabilities")
        assert resp.status_code == 200
        assert resp.json() == {"research_lanes_enabled": False}


def test_capability_false_when_flag_unset():
    with _client(flag=None) as (client, _usage, _cp):
        resp = client.get("/curation/capabilities")
        assert resp.status_code == 200
        assert resp.json() == {"research_lanes_enabled": False}


def test_response_carries_only_the_single_boolean_key():
    with _client(flag="true") as (client, _usage, _cp):
        body = client.get("/curation/capabilities").json()
        assert list(body.keys()) == ["research_lanes_enabled"]
        assert isinstance(body["research_lanes_enabled"], bool)
        text = client.get("/curation/capabilities").text.lower()
        for leak in ("openai", "gpt-", "api_key", "password", "policy", "limit", "token"):
            assert leak not in text


def test_invalid_flag_configuration_is_fail_loud():
    with _client(flag="maybe-on") as (client, _usage, _cp):
        with pytest.raises(ValueError, match="RESEARCH_LANES_ENABLED"):
            client.get("/curation/capabilities")


def test_capability_read_does_zero_provider_admission_or_telemetry_work():
    import chromadb

    with patch.object(
        chromadb, "PersistentClient",
        side_effect=AssertionError("real chromadb.PersistentClient must never be constructed"),
    ) as persistent_client_spy:
        with _client(flag="true") as (client, usage_db_path, cp_db_path):
            cp_before = fingerprint_usage_db(cp_db_path)
            assert client.get("/curation/capabilities").status_code == 200
            assert fingerprint_usage_db(cp_db_path) == cp_before
            assert _paid_rows(usage_db_path) == []
        persistent_client_spy.assert_not_called()


@contextmanager
def _auth_client(flag: str = "true"):
    import base64

    from research_agent.api_app.app import create_app

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.sqlite"
        cp_db_path = Path(tmp) / "test_checkpoints.sqlite"
        usage_db_path = Path(tmp) / "usage_telemetry.sqlite"
        env = {
            "APP_ENV": "local", "AUTH_ENABLED": "true",
            "AUTH_USERNAME": "alice", "AUTH_PASSWORD": "s3curePlatformSecret!",
            "RESEARCH_LANES_ENABLED": flag,
        }
        with patch.dict(os.environ, env), \
             patch.object(api, "init_db", lambda: real_init_db(db_path)), \
             patch.object(telemetry, "USAGE_DB_PATH", usage_db_path), \
             patch.object(admission, "USAGE_DB_PATH", usage_db_path), \
             patch.object(leases, "USAGE_DB_PATH", usage_db_path), \
             patch.object(api, "OpenAI", return_value=MagicMock(name="fake_openai_client")), \
             patch.object(api, "get_chroma_collection", return_value=MagicMock()):
            fresh_app = create_app()
            fresh_app.dependency_overrides[api.get_db_connection] = _make_db_override(db_path)
            fresh_app.dependency_overrides[api.get_curation_checkpointer] = _make_cp_override(cp_db_path)
            try:
                with TestClient(fresh_app) as client:
                    yield client, base64.b64encode(b"alice:s3curePlatformSecret!").decode()
            finally:
                fresh_app.dependency_overrides.clear()


def test_unauthorized_request_is_rejected_by_the_auth_gate():
    with _auth_client() as (client, token):
        assert client.get("/curation/capabilities").status_code == 401
        authorized = client.get("/curation/capabilities", headers={"Authorization": f"Basic {token}"})
        assert authorized.status_code == 200
        assert authorized.json() == {"research_lanes_enabled": True}


def test_real_databases_are_byte_identical():
    assert fingerprint_usage_db(_REAL_CHECKPOINT_DB) == _REAL_CHECKPOINT_FINGERPRINT_BEFORE
    assert fingerprint_usage_db(_REAL_USAGE_DB) == _REAL_USAGE_FINGERPRINT_BEFORE
