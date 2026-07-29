"""Small service-layer exception so services can signal an HTTP-mapped
failure without reaching into the HTTP layer (raising FastAPI's
HTTPException) themselves.

Routers catch ServiceError and convert it to the identical
HTTPException(status_code=..., detail=...) the service used to raise
directly — status_code/detail are preserved exactly, only the raise site
moves from the service to the router.
"""

from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    def __init__(self, status_code: int, detail: Any):
        super().__init__(detail if isinstance(detail, str) else str(detail))
        self.status_code = status_code
        self.detail = detail
