"""Usage Protection M2.2C Part A: the request-body size limit.

A pure ASGI middleware -- same convention as
`research_agent/telemetry.py::RequestTelemetryMiddleware` and for the
same reason: `BaseHTTPMiddleware` fully buffers a streamed response
before this app ever sees it and runs the downstream app in a separate
task, both wrong for something meant to sit underneath a future
streaming endpoint without buffering/breaking it. This middleware only
wraps `receive`, never `send` -- response streaming is completely
untouched by it.

**Why the whole body is buffered here, not enforced reactively.** "A
rejected request never reaches the route/service" requires knowing the
body is oversized *before* `self.app(...)` is ever called -- the route
itself decides when to call `receive()` to pull body chunks, so a
middleware that only inspects chunks as they pass through cannot
guarantee rejection happens strictly before routing. Instead this
middleware drains every `http.request` message itself first, counting
real bytes as they arrive (independent of whatever Content-Length
claimed, so a missing/wrong header or a body split across multiple ASGI
chunks is still caught), and only calls `self.app(...)` -- replaying the
buffered messages through a substitute `receive` -- once the whole body
is known to fit. The configured limit (64 KiB provisional) keeps this
buffering itself bounded and cheap; this is not a general upload
mechanism.

**Never logs or persists body content** -- only byte counts are ever
inspected; the bytes themselves are held in memory only long enough to
be replayed to the real app, never written anywhere.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

_BODY_TOO_LARGE_BODY = json.dumps({
    "detail": {
        "reason_code": "request_body_too_large",
        "message": "Request body exceeds the allowed size.",
    },
}).encode("utf-8")


async def _send_413(send: Callable[[dict], Awaitable[None]]) -> None:
    await send({
        "type": "http.response.start",
        "status": 413,
        "headers": [(b"content-type", b"application/json")],
    })
    await send({"type": "http.response.body", "body": _BODY_TOO_LARGE_BODY})


class RequestBodyLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Fast path: a truthful, oversized Content-Length rejects before
        # reading a single body byte. A missing or malformed header just
        # falls through to the real-byte-count enforcement below --
        # never trusted on its own to ALLOW a request, only ever used to
        # reject early when it can be trusted.
        for name, value in scope.get("headers") or []:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break
                if declared > self.max_bytes:
                    await _send_413(send)
                    return
                break

        buffered: list[dict[str, Any]] = []
        total = 0
        more_body = True
        while more_body:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                # e.g. an early http.disconnect -- nothing more to drain.
                break
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                await _send_413(send)
                return
            more_body = bool(message.get("more_body", False))

        queue = buffered

        async def replay_receive() -> dict[str, Any]:
            if queue:
                return queue.pop(0)
            return await receive()

        await self.app(scope, replay_receive, send)
