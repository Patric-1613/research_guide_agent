"""Small service-layer exception so services can signal an HTTP-mapped
failure without depending on the web framework themselves.

One centralized handler (research_agent/api_app/app.py) maps every
ServiceError to its response: status = status_code, body = {"detail":
detail}, with detail passed through unchanged whether it is a plain
string or a structured object. Services only ever raise it.
"""

from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    def __init__(self, status_code: int, detail: Any):
        super().__init__(detail if isinstance(detail, str) else str(detail))
        self.status_code = status_code
        self.detail = detail
