"""PR2B: the single-user HTTP Basic Auth access gate.

A pure ASGI middleware -- same convention as `research_agent/
request_limits.py`'s `RequestBodyLimitMiddleware` and `research_agent/
telemetry.py`'s `RequestTelemetryMiddleware`, and for the same reason:
this must sit outside every route -- including the chat/report SSE
streams -- without ever buffering or altering a streamed response.
Unlike `RequestBodyLimitMiddleware`, this middleware never reads the
request body at all: every decision is made from `scope["method"]`/
`scope["path"]`/`scope["headers"]` alone, before `receive()` is ever
called, so there is nothing to buffer or replay.

**Default-deny allowlist, not a denylist.** Only `GET /health` is public;
every other path and method -- every API route, every curation/session
route, both SSE streams, the report export route, `/docs`, `/openapi.json`
-- requires valid credentials whenever the gate is enabled. A future
router added to `api_app/app.py` is protected automatically, not by
someone remembering to add it to a list of "sensitive" paths here.

**Registration.** Must be added as the LAST `app.add_middleware(...)`
call in `api_app/app.py`'s `create_app()` -- Starlette's own convention
(most-recently-added middleware runs first) makes this the OUTERMOST
layer, ahead of CORS, request telemetry, and the request-body-size
limit. An unauthorized request is rejected here, before any of those
ever see it: no `http_requests`/paid-action telemetry row is written, no
body is read, no route or service code runs.

**Enabled/disabled.** `auth_config.enabled=False` (the default local-dev/
test configuration -- see `research_agent/config/settings.py`'s
`get_auth_config()`) makes this middleware a complete no-op passthrough,
byte-for-byte the current, unauthenticated behavior every existing test
already exercises. There is no OTHER runtime toggle: once a process has
booted with `enabled=True`, there is deliberately no request-time way to
reach a disabled state (see `get_auth_config()`'s own docstring --
disabling auth is never a valid production rollback; restoring a
previous known-good image/commit is).

**Never logs or persists a credential.** This module never calls
`logging`, never includes the `Authorization` header value or the
configured username/password in a response body, an exception, or any
telemetry -- the 401 body below is a fixed, generic constant, never
built from the request.

**PR2B.1 correction: the OPTIONS exemption is narrow, not blanket.** A
bare `OPTIONS` request -- one missing `Origin` or `Access-Control-
Request-Method` -- is NOT a real CORS preflight (a browser never sends
either without both) and must still require credentials like any other
request; only a request carrying BOTH headers is exempted. The original
PR2B version exempted every `OPTIONS` request outright, which meant an
`OPTIONS` request to any route -- not just a genuine preflight -- bypassed
the gate entirely.

**The 401 carries credentialed-CORS headers for an allowed origin.** This
middleware is outermost, so its `401` is emitted before `CORSMiddleware`
(registered inner) ever runs -- that response would otherwise carry no
`Access-Control-Allow-Origin`, and a permitted cross-origin browser
frontend (which sends `credentials: "include"`) would see an opaque
network failure instead of a readable `401`. `create_app()` hands this
middleware the same validated origin list it gives `CORSMiddleware`
(`get_cors_config().allowed_origins`); when the request's `Origin` header
exactly matches one of those, the `401` gets `Access-Control-Allow-
Origin: <that origin>` (echoed exactly -- never `*`, never a reflected
un-allowed value), `Access-Control-Allow-Credentials: true`, and `Vary:
Origin`. A same-origin request, a non-browser client, or a request from
an origin not on the list gets a plain `401` with none of these.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
from typing import Any, Awaitable, Callable, Iterable

from research_agent.config.settings import AuthConfig

_BASIC_PREFIX = b"basic "

_WWW_AUTHENTICATE_VALUE = b'Basic realm="research-agent", charset="UTF-8"'

_UNAUTHORIZED_BODY = json.dumps({
    "detail": {
        "reason_code": "unauthorized",
        "message": "Authentication required.",
    },
}).encode("utf-8")

# The one public route. A (method, path) pair, not a bare path -- only
# GET is exempt; a hypothetical POST /health (none exists today) would
# still require credentials.
_PUBLIC_GET_PATHS = frozenset({"/health"})


def _get_header(headers: list[tuple[bytes, bytes]], name: bytes) -> bytes | None:
    for key, value in headers:
        if key.lower() == name:
            return value
    return None


def _cors_headers_for_401(
    headers: list[tuple[bytes, bytes]], allowed_origins: frozenset[str]
) -> list[tuple[bytes, bytes]]:
    """The Access-Control-* headers CORSMiddleware would have added to a
    cross-origin response for an allowed origin -- recreated here because
    this 401 is emitted before CORSMiddleware runs. Empty list whenever
    there is no `Origin` header, or its value is not an EXACT match for
    one of the configured origins (never reflect an un-allowed origin,
    never emit `*`)."""
    if not allowed_origins:
        return []
    origin = _get_header(headers, b"origin")
    if origin is None:
        return []
    # An Origin header is always ASCII; latin-1 decodes any byte without
    # raising, so a comparison against the (str) allow-list is safe.
    if origin.decode("latin-1") not in allowed_origins:
        return []
    return [
        (b"access-control-allow-origin", origin),
        (b"access-control-allow-credentials", b"true"),
        (b"vary", b"Origin"),
    ]


async def _send_401(
    send: Callable[[dict[str, Any]], Awaitable[None]],
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    await send({
        "type": "http.response.start",
        "status": 401,
        "headers": [
            (b"content-type", b"application/json"),
            (b"www-authenticate", _WWW_AUTHENTICATE_VALUE),
            # Never let a browser/proxy cache this -- an operator rotating
            # credentials must not have a stale 401 (or a stale prior
            # success) served back from a shared cache.
            (b"cache-control", b"no-store"),
            *(extra_headers or []),
        ],
    })
    await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})


def _is_cors_preflight(headers: list[tuple[bytes, bytes]]) -> bool:
    """True only for a genuine CORS preflight: a browser sends `Origin`
    on every cross-origin request, and adds `Access-Control-Request-
    Method` ONLY on the uncredentialed OPTIONS preflight that precedes a
    non-simple cross-origin request -- neither header is ever present on
    an OPTIONS request a browser sends for any other reason, and a bare/
    forged OPTIONS missing either one is not a preflight this app needs
    to let through."""
    names = {name.lower() for name, _ in headers}
    return b"origin" in names and b"access-control-request-method" in names


def _get_single_authorization_header(headers: list[tuple[bytes, bytes]]) -> bytes | None:
    """None whenever the header is absent OR duplicated. A duplicate
    Authorization header is ambiguous -- which one is authoritative? --
    and is rejected outright rather than guessed at by taking the first
    or last."""
    matches = [value for name, value in headers if name.lower() == b"authorization"]
    if len(matches) != 1:
        return None
    return matches[0]


def _parse_basic_credentials(header_value: bytes) -> tuple[str, str] | None:
    """None on ANY malformed input -- wrong scheme, invalid Base64,
    non-UTF-8 decoded bytes, or no ':' separator. Never raises; every
    failure mode collapses to the same generic 401 the caller sends, so
    a malformed header is indistinguishable from a merely wrong one."""
    if len(header_value) < len(_BASIC_PREFIX) or header_value[: len(_BASIC_PREFIX)].lower() != _BASIC_PREFIX:
        return None
    encoded = header_value[len(_BASIC_PREFIX):].strip()
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if ":" not in text:
        return None
    username, _, password = text.partition(":")
    return username, password


def _credentials_match(username: str, password: str, expected_username: str, expected_password: str) -> bool:
    """ONE constant-time comparison over the full 'username:password'
    representation -- never two sequential compare_digest calls (one per
    field). Two sequential calls, even though each is individually
    constant-time, would reintroduce a timing signal through the branch
    between them (skip the password compare entirely on a bad username).
    Concatenating first removes that branch, so a wrong username and a
    wrong password are indistinguishable from the outside, both in the
    response body and in timing. This is a mitigation, not a guarantee --
    network jitter dominates any residual signal in practice, but the
    comparison itself carries no additional leak."""
    presented = f"{username}:{password}".encode("utf-8")
    expected = f"{expected_username}:{expected_password}".encode("utf-8")
    return hmac.compare_digest(presented, expected)


class BasicAuthMiddleware:
    def __init__(
        self, app: Any, auth_config: AuthConfig, allowed_origins: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.enabled = auth_config.enabled
        self.username = auth_config.username
        self.password = auth_config.password
        # The exact origin set CORSMiddleware (registered INNER of this
        # gate) is configured for -- see _cors_headers_for_401 and this
        # module's docstring. Exact-match only: never `*`, never a
        # reflected un-allowed value.
        self.allowed_origins: frozenset[str] = frozenset(allowed_origins)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not self.enabled:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        # CORS preflight never carries credentials by spec: Authorization
        # is a non-simple header, so a cross-origin browser asks
        # permission via an uncredentialed OPTIONS request first, before
        # ever sending the real, credentialed request. Challenging a
        # GENUINE preflight here would break CORS entirely, not just
        # leave it unauthenticated -- but the exemption must be narrow:
        # only a request carrying BOTH `Origin` and `Access-Control-
        # Request-Method` is a real preflight (see _is_cors_preflight's
        # own docstring). A bare OPTIONS -- neither header present -- is
        # not a preflight and still requires credentials, same as any
        # other method/path.
        if method == "OPTIONS" and _is_cors_preflight(scope.get("headers") or []):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if method == "GET" and path in _PUBLIC_GET_PATHS:
            await self.app(scope, receive, send)
            return

        request_headers = scope.get("headers") or []
        header_value = _get_single_authorization_header(request_headers)
        if header_value is not None:
            parsed = _parse_basic_credentials(header_value)
            if parsed is not None:
                username, password = parsed
                # self.username/self.password are guaranteed non-None
                # strings whenever self.enabled is True -- get_auth_config()
                # never returns enabled=True with either unset.
                if _credentials_match(username, password, self.username, self.password):  # type: ignore[arg-type]
                    await self.app(scope, receive, send)
                    return

        await _send_401(send, _cors_headers_for_401(request_headers, self.allowed_origins))
