"""Day 3: the Firebase identity ASGI middleware.

Same shape and same guarantees as `auth_middleware.BasicAuthMiddleware`,
for `AUTH_MODE=firebase`:

- **Pure ASGI**, so it sits OUTSIDE CORS / request-telemetry / the
  body-size limit and never buffers or alters a streamed response.
- **Default-deny allowlist.** Only `GET /health` is public; a genuine
  CORS preflight is passed through uncredentialed (a browser never sends
  `Authorization` on a preflight). Everything else -- every API route,
  both SSE streams, `/docs`, `/openapi.json`, the static frontend --
  requires a verified Firebase identity.
- **No-op unless active.** `auth_config.mode != "firebase"` makes this a
  complete passthrough, so a `basic`/`disabled` deployment is
  byte-identical to before Day 3. Only one of {BasicAuth, FirebaseAuth}
  is ever active.
- **Never logs a token, an email, a uid, or an Authorization header.**
  Every failure -- a missing/duplicate/malformed header, a bad
  signature, a wrong audience/issuer, an expired token, a disabled user
  -- collapses to one identical generic 401. An infrastructure failure
  that makes verification impossible -- a database outage resolving the
  internal user, or a failed fetch of Google's signing certificates --
  is a 503 (still denied, route never runs), never a 401.
  No response body or log line is ever built from the request.
- **CORS-readable failure responses.** Because this middleware is
  outermost, its 401/503 is emitted before `CORSMiddleware` runs -- so
  it recreates the same `Access-Control-Allow-Origin` /
  `-Allow-Credentials` / `Vary: Origin` headers `BasicAuthMiddleware`
  does, for an exact-match allowed origin, reusing that module's own
  helper so the two behave identically.

On success the verified `identity.RequestIdentity` is placed in the ASGI
`scope["state"]` (request-scoped, never a module global); `identity.
get_current_user` reads it back.

The token verification + user upsert (`authenticate`, injected) is
synchronous and may block on a one-time cert fetch and a database query,
so it runs on the AnyIO worker threadpool (bounded by `ANYIO_THREAD_LIMIT`
from Day 1) -- never on the event loop.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Iterable

import anyio

from research_agent.auth_middleware import _cors_headers_for_401, _is_cors_preflight
from research_agent.config.settings import AuthConfig
from research_agent.firebase_auth import FirebaseTokenError, FirebaseVerifierUnavailable
from research_agent.identity import IdentityDenied, IdentityUnavailable, RequestIdentity

_PUBLIC_GET_PATHS = frozenset({"/health"})

_WWW_AUTHENTICATE_VALUE = b"Bearer"

_UNAUTHORIZED_BODY = json.dumps({
    "detail": {"reason_code": "unauthorized", "message": "Authentication required."},
}).encode("utf-8")

_UNAVAILABLE_BODY = json.dumps({
    "detail": {"reason_code": "identity_store_unavailable", "message": "Authentication is temporarily unavailable."},
}).encode("utf-8")


async def _send_json_error(
    send: Callable[[dict[str, Any]], Awaitable[None]],
    *, status: int, body: bytes, extra_headers: list[tuple[bytes, bytes]],
) -> None:
    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"cache-control", b"no-store"),
        *extra_headers,
    ]
    if status == 401:
        headers.insert(1, (b"www-authenticate", _WWW_AUTHENTICATE_VALUE))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class FirebaseAuthMiddleware:
    def __init__(
        self, app: Any, auth_config: AuthConfig,
        allowed_origins: Iterable[str] = (),
        *, authenticate: Callable[[str], RequestIdentity],
    ) -> None:
        self.app = app
        self.enabled = auth_config.mode == "firebase"
        self.allowed_origins: frozenset[str] = frozenset(allowed_origins)
        self._authenticate = authenticate

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not self.enabled:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        headers: list[tuple[bytes, bytes]] = scope.get("headers") or []

        if method == "OPTIONS" and _is_cors_preflight(headers):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if method == "GET" and path in _PUBLIC_GET_PATHS:
            await self.app(scope, receive, send)
            return

        cors_headers = _cors_headers_for_401(headers, self.allowed_origins)
        authorization_values = [
            value.decode("latin-1") for key, value in headers if key.lower() == b"authorization"
        ]

        try:
            # extract_bearer_token is invoked inside `authenticate` (the
            # closure from identity.build_firebase_authenticator wraps
            # firebase_auth.verify_firebase_id_token, which does not parse
            # the header) -- so parse it here and hand the bare token in.
            from research_agent.firebase_auth import extract_bearer_token

            token = extract_bearer_token(authorization_values)
            identity = await anyio.to_thread.run_sync(self._authenticate, token)
        except (FirebaseTokenError, IdentityDenied):
            await _send_json_error(send, status=401, body=_UNAUTHORIZED_BODY, extra_headers=cors_headers)
            return
        except (IdentityUnavailable, FirebaseVerifierUnavailable):
            # Database outage OR signing-certificate fetch failure: in
            # both cases we cannot verify, so we deny with 503 (not 401)
            # -- infrastructure failure is never reported as a bad
            # credential.
            await _send_json_error(send, status=503, body=_UNAVAILABLE_BODY, extra_headers=cors_headers)
            return

        scope.setdefault("state", {})["identity"] = identity
        await self.app(scope, receive, send)
