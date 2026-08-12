"""Usage Protection M2.2C Part A: tests for research_agent/request_limits.py
-- the pure-ASGI request-body size limit. Exercises the ASGI protocol
directly (raw scope/receive/send) via asyncio.run(), not through
TestClient, so exact Content-Length/chunking scenarios (including a
deliberately WRONG Content-Length) are fully controllable. Same
sync-test-wraps-asyncio.run() convention already used elsewhere in this
project (e.g. tests/test_telemetry.py) -- no pytest-asyncio dependency.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.request_limits import RequestBodyLimitMiddleware

MAX_BYTES = 100


def _http_scope(headers: list[tuple[bytes, bytes]] | None = None) -> dict:
    return {"type": "http", "method": "POST", "path": "/x", "headers": headers or []}


def _receive_from_chunks(chunks: list[bytes]):
    """Simulates the body arriving as one or more ASGI http.request
    messages, more_body=True on every chunk but the last."""
    messages = [
        {"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1}
        for i, chunk in enumerate(chunks)
    ]
    if not messages:
        messages = [{"type": "http.request", "body": b"", "more_body": False}]
    it = iter(messages)

    async def receive():
        return next(it)

    return receive


class _RecordingApp:
    """A minimal downstream ASGI app that records whether it was ever
    invoked, and (if so) drains the request body it was handed and
    echoes a 200."""

    def __init__(self):
        self.called = False
        self.received_body = b""

    async def __call__(self, scope, receive, send):
        self.called = True
        more_body = True
        while more_body:
            message = await receive()
            self.received_body += message.get("body", b"")
            more_body = message.get("more_body", False)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"ok": true}'})


class _StreamingApp:
    """Downstream app that streams multiple response body chunks --
    proves the middleware never buffers/alters the response side."""

    def __init__(self):
        self.chunks = [b"chunk-1", b"chunk-2", b"chunk-3"]

    async def __call__(self, scope, receive, send):
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        for i, chunk in enumerate(self.chunks):
            await send({"type": "http.response.body", "body": chunk, "more_body": i < len(self.chunks) - 1})


def _run(middleware, scope, receive):
    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


def _status_of(sent):
    for m in sent:
        if m["type"] == "http.response.start":
            return m["status"]
    return None


def _body_of(sent):
    return b"".join(m["body"] for m in sent if m["type"] == "http.response.body")


def test_content_length_over_limit_rejects_413():
    app = _RecordingApp()
    mw = RequestBodyLimitMiddleware(app, max_bytes=MAX_BYTES)
    scope = _http_scope([(b"content-length", str(MAX_BYTES + 1).encode())])
    receive = _receive_from_chunks([b"x" * (MAX_BYTES + 1)])

    sent = _run(mw, scope, receive)

    assert _status_of(sent) == 413
    body = json.loads(_body_of(sent))
    assert body == {"detail": {"reason_code": "request_body_too_large", "message": "Request body exceeds the allowed size."}}
    assert app.called is False  # rejected before the app ever ran


def test_incorrect_content_length_still_enforced_by_real_bytes():
    """A Content-Length that UNDER-claims the real body size must not
    let an oversized body slip through -- the real received-byte count
    is what's actually enforced."""
    app = _RecordingApp()
    mw = RequestBodyLimitMiddleware(app, max_bytes=MAX_BYTES)
    scope = _http_scope([(b"content-length", b"5")])  # lies: claims only 5 bytes
    receive = _receive_from_chunks([b"x" * (MAX_BYTES + 50)])

    sent = _run(mw, scope, receive)

    assert _status_of(sent) == 413
    assert app.called is False


def test_chunked_body_over_limit_rejects_413():
    """No Content-Length at all, body split across several ASGI
    messages -- still enforced by real accumulated byte count."""
    app = _RecordingApp()
    mw = RequestBodyLimitMiddleware(app, max_bytes=MAX_BYTES)
    scope = _http_scope()
    chunks = [b"x" * 40, b"x" * 40, b"x" * 40]  # 120 total, over MAX_BYTES=100
    receive = _receive_from_chunks(chunks)

    sent = _run(mw, scope, receive)

    assert _status_of(sent) == 413
    assert app.called is False


def test_chunked_body_under_limit_passes():
    app = _RecordingApp()
    mw = RequestBodyLimitMiddleware(app, max_bytes=MAX_BYTES)
    scope = _http_scope()
    chunks = [b"x" * 20, b"x" * 20, b"x" * 20]  # 60 total
    receive = _receive_from_chunks(chunks)

    sent = _run(mw, scope, receive)

    assert _status_of(sent) == 200
    assert app.called is True
    assert app.received_body == b"x" * 60


def test_exactly_at_limit_passes():
    app = _RecordingApp()
    mw = RequestBodyLimitMiddleware(app, max_bytes=MAX_BYTES)
    scope = _http_scope([(b"content-length", str(MAX_BYTES).encode())])
    receive = _receive_from_chunks([b"x" * MAX_BYTES])

    sent = _run(mw, scope, receive)

    assert _status_of(sent) == 200
    assert app.called is True
    assert len(app.received_body) == MAX_BYTES


def test_one_byte_over_limit_rejects():
    app = _RecordingApp()
    mw = RequestBodyLimitMiddleware(app, max_bytes=MAX_BYTES)
    scope = _http_scope([(b"content-length", str(MAX_BYTES + 1).encode())])
    receive = _receive_from_chunks([b"x" * (MAX_BYTES + 1)])

    sent = _run(mw, scope, receive)

    assert _status_of(sent) == 413
    assert app.called is False


def test_no_body_request_unaffected():
    app = _RecordingApp()
    mw = RequestBodyLimitMiddleware(app, max_bytes=MAX_BYTES)
    scope = _http_scope()
    receive = _receive_from_chunks([])

    sent = _run(mw, scope, receive)

    assert _status_of(sent) == 200
    assert app.called is True
    assert app.received_body == b""


def test_streaming_response_unaffected():
    app = _StreamingApp()
    mw = RequestBodyLimitMiddleware(app, max_bytes=MAX_BYTES)
    scope = _http_scope()
    receive = _receive_from_chunks([b"small body"])

    sent = _run(mw, scope, receive)

    body_messages = [m for m in sent if m["type"] == "http.response.body"]
    assert len(body_messages) == 3
    assert [m["body"] for m in body_messages] == [b"chunk-1", b"chunk-2", b"chunk-3"]


def test_non_http_scope_passes_through_unchanged():
    app = _RecordingApp()
    mw = RequestBodyLimitMiddleware(app, max_bytes=MAX_BYTES)
    scope = {"type": "lifespan"}

    async def receive():
        return {"type": "lifespan.startup"}

    async def go():
        sent = []

        async def send(message):
            sent.append(message)

        await mw(scope, receive, send)
        return sent

    asyncio.run(go())
    # The middleware must not have consumed/buffered anything itself --
    # the app saw the raw receive/send directly.
    assert app.called is True
