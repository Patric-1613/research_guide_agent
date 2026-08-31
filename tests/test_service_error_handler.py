"""The centralized ServiceError -> HTTP response handler.

Proves the one handler registered in research_agent/api_app/app.py
produces byte-for-byte the response the per-router
`raise HTTPException(status_code=exc.status_code, detail=exc.detail)`
used to produce, and that no router still does that mapping by hand.
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import research_agent.api as api
from research_agent.api_app.app import _handle_service_error
from research_agent.services.errors import ServiceError

_REPO = pathlib.Path(__file__).resolve().parent.parent
_ROUTERS_DIR = _REPO / "research_agent" / "api_app" / "routers"
_SERVICES_DIR = _REPO / "research_agent" / "services"

# (status, detail) pairs covering a string detail and a structured detail
# at each status code a service actually raises.
_CASES = [
    (400, "exchange_ids must not be empty"),
    (403, "Research lanes are not enabled on this deployment."),
    (404, "session_id not found"),
    (503, {"error": "curation_lane_suggest service unavailable"}),
    (400, {"reason_code": "bad_input", "message": "structured 400 body"}),
]


def _twin_apps() -> tuple[TestClient, TestClient]:
    """Two apps that should be indistinguishable: one raises HTTPException
    the old way, the other raises ServiceError and relies on the handler."""
    old = FastAPI()
    new = FastAPI()
    new.add_exception_handler(ServiceError, _handle_service_error)

    for status, detail in _CASES:
        key = f"{status}-{'str' if isinstance(detail, str) else 'obj'}"

        def _old_route(status=status, detail=detail):
            raise HTTPException(status_code=status, detail=detail)

        def _new_route(status=status, detail=detail):
            raise ServiceError(status, detail)

        old.get(f"/{key}")(_old_route)
        new.get(f"/{key}")(_new_route)

    return TestClient(old, raise_server_exceptions=False), TestClient(new, raise_server_exceptions=False)


@pytest.mark.parametrize("status,detail", _CASES)
def test_string_and_structured_details_map_identically_to_the_old_mapping(status, detail):
    old_client, new_client = _twin_apps()
    key = f"{status}-{'str' if isinstance(detail, str) else 'obj'}"

    old_resp = old_client.get(f"/{key}")
    new_resp = new_client.get(f"/{key}")

    assert new_resp.status_code == old_resp.status_code == status
    assert new_resp.json() == old_resp.json() == {"detail": detail}


def test_representative_status_codes_are_unchanged():
    _, new_client = _twin_apps()
    assert new_client.get("/400-str").status_code == 400
    assert new_client.get("/403-str").status_code == 403
    assert new_client.get("/404-str").status_code == 404
    assert new_client.get("/503-obj").status_code == 503


def test_centralized_handler_is_registered_on_the_real_app():
    assert api.app.exception_handlers.get(ServiceError) is _handle_service_error


def test_no_router_maps_service_error_by_hand_anymore():
    offenders = [
        p.name
        for p in _ROUTERS_DIR.glob("*.py")
        if "except ServiceError" in p.read_text()
    ]
    assert offenders == [], f"routers still catching ServiceError: {offenders}"


def test_service_modules_do_not_import_the_fastapi_http_layer():
    import ast

    for p in _SERVICES_DIR.glob("*.py"):
        tree = ast.parse(p.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("fastapi"):
                imported.add(node.module)
                for alias in node.names:
                    assert alias.name != "HTTPException", f"{p.name} imports HTTPException"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "fastapi", f"{p.name} imports fastapi"
        # `fastapi.responses` (a response type) is the only fastapi module a
        # service may import; never `fastapi` itself (the app/error layer).
        assert imported <= {"fastapi.responses"}, f"{p.name} imports {imported}"
