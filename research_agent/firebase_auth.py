"""Day 3 (public multi-user deployment foundation, see
docs/plans/public-multi-user-deployment-review.md): the Firebase ID
token verification boundary.

This module does exactly ONE thing: turn an
``Authorization: Bearer <Firebase ID token>`` header value into a
verified ``(firebase_uid, email, display_name)`` triple, or raise the
single generic :class:`FirebaseTokenError`. It never touches the
database, never mints an internal user, and never decides HTTP status
codes -- that is :mod:`research_agent.identity` and the middleware.

**Library.** Verification uses ``google.oauth2.id_token.verify_firebase_
token`` from ``google-auth`` -- Google's own officially supported auth
library (the Firebase docs at
https://firebase.google.com/docs/auth/admin/verify-id-tokens recommend
either the Firebase Admin SDK or "Google's public keys" via a supported
library; this is the latter). ``verify_firebase_token`` checks the
RS256 signature against Google's ``securetoken@system`` x509
certificates, the ``aud`` claim, and ``exp``/``iat`` with clock skew.
On top of that this module additionally enforces the ``iss`` claim and a
non-empty ``sub`` -- which is what rejects Firebase *custom* tokens,
session cookies, and tokens minted for a different project (all of which
carry a different ``iss`` and/or are signed by a different key).

**Signing-certificate caching** is delegated to the standard
``cachecontrol`` library (the same one the Firebase Admin SDK uses
internally) by handing ``verify_firebase_token`` a
``google.auth.transport.Request`` backed by a CacheControl-wrapped
``requests.Session``. Google's certificate endpoint sets a multi-hour
``Cache-Control: max-age``; after the first verification the certs are a
process-local cache hit with no network call. Nothing here hand-rolls a
cache, a TTL, or a cert store.

**No service-account key.** ``verify_firebase_token`` needs only the
project ID (as the audience) and Google's *public* certificates -- no
credential of any kind. Nowhere in this module (or anywhere Day 3 adds)
is ``GOOGLE_APPLICATION_CREDENTIALS`` or a key-file path read.

**Emulator.** When ``config.firebase_emulator_host`` is set (local
development only -- ``get_auth_config`` forbids it in production), tokens
come from a locally-run Firebase Auth emulator and are **unsigned**
(``alg: none``). In that mode only, this module decodes the claims
without signature verification but still enforces ``iss``/``aud``/
``exp``/``iat``/``sub`` exactly as the real path does. This is the one
place a JWT payload is read without a signature check, it is gated on an
explicit non-production-only setting, and it mirrors what the Firebase
Admin SDK itself does when ``FIREBASE_AUTH_EMULATOR_HOST`` is set.
"""

from __future__ import annotations

import base64
import binascii
import json
import time
from dataclasses import dataclass

from research_agent.config.settings import AuthConfig

_BEARER_PREFIX = "bearer "

# Firebase ID tokens always carry this exact issuer.
_ISS_TEMPLATE = "https://securetoken.google.com/{project_id}"

# Firebase uids are at most 128 characters.
_MAX_UID_LENGTH = 128

# Same clock-skew tolerance the Firebase Admin SDK uses for exp/iat.
_CLOCK_SKEW_SECONDS = 300


class FirebaseTokenError(Exception):
    """The ONE error every verification failure maps to. Deliberately
    carries no structured detail: a caller (the middleware) turns any
    instance of this into one identical generic 401, and its message is
    never surfaced to a client or written to a log. Construct it with a
    short internal-only reason string for test assertions; nothing in
    production reads that string."""


@dataclass(frozen=True)
class VerifiedFirebaseToken:
    """The verified, trusted claims. `firebase_uid` is the immutable
    provider identity (the token's `sub`). `email`/`display_name` are
    mutable metadata and may be None (a Firebase account need not have an
    email, and `name` is optional). No other claim is retained -- the
    raw token itself is never kept past this function."""

    firebase_uid: str
    email: str | None
    display_name: str | None


def extract_bearer_token(authorization_header_values: list[str]) -> str:
    """Given every value of the `Authorization` header on a request
    (0, 1, or more), return the bare token from a single well-formed
    `Bearer <token>` header. Raises `FirebaseTokenError` for: no header,
    more than one header (ambiguous -- never guessed at), a non-`Bearer`
    scheme, or an empty token after the scheme."""
    if len(authorization_header_values) == 0:
        raise FirebaseTokenError("missing Authorization header")
    if len(authorization_header_values) > 1:
        raise FirebaseTokenError("duplicate Authorization header")
    raw = authorization_header_values[0]
    if len(raw) < len(_BEARER_PREFIX) or raw[: len(_BEARER_PREFIX)].lower() != _BEARER_PREFIX:
        raise FirebaseTokenError("Authorization scheme is not Bearer")
    token = raw[len(_BEARER_PREFIX):].strip()
    if not token:
        raise FirebaseTokenError("empty bearer token")
    return token


def _decode_unverified_claims(token: str) -> dict:
    """Base64url-decode and JSON-parse a JWT's payload segment WITHOUT
    any signature check. Used ONLY by the emulator path (emulator tokens
    are `alg: none`). A structurally malformed token raises
    FirebaseTokenError like any other failure."""
    parts = token.split(".")
    if len(parts) != 3:
        raise FirebaseTokenError("token is not a well-formed JWT")
    payload_segment = parts[1]
    padding = "=" * (-len(payload_segment) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload_segment + padding)
        claims = json.loads(decoded)
    except (binascii.Error, ValueError) as exc:
        raise FirebaseTokenError("token payload is not decodable") from exc
    if not isinstance(claims, dict):
        raise FirebaseTokenError("token payload is not a JSON object")
    return claims


def _claims_to_verified_token(claims: dict, *, project_id: str, check_time: bool) -> VerifiedFirebaseToken:
    """Shared post-signature validation for both the real and emulator
    paths: iss, aud, sub, and (real path only, since the emulator path's
    verifier already did exp/iat) the time claims."""
    if claims.get("iss") != _ISS_TEMPLATE.format(project_id=project_id):
        raise FirebaseTokenError("issuer does not match this Firebase project")
    if claims.get("aud") != project_id:
        raise FirebaseTokenError("audience does not match this Firebase project")

    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub or len(sub) > _MAX_UID_LENGTH:
        raise FirebaseTokenError("token has no valid subject (uid)")

    if check_time:
        now = time.time()
        exp = claims.get("exp")
        iat = claims.get("iat")
        if not isinstance(exp, (int, float)) or now - _CLOCK_SKEW_SECONDS > exp:
            raise FirebaseTokenError("token is expired")
        if not isinstance(iat, (int, float)) or iat - _CLOCK_SKEW_SECONDS > now:
            raise FirebaseTokenError("token was issued in the future")

    email = claims.get("email")
    name = claims.get("name")
    return VerifiedFirebaseToken(
        firebase_uid=sub,
        email=email if isinstance(email, str) and email else None,
        display_name=name if isinstance(name, str) and name else None,
    )


_cached_transport_request = None


def _get_cached_transport_request():
    """One process-wide `google.auth.transport.requests.Request` backed
    by a CacheControl-wrapped session, so `verify_firebase_token`'s cert
    fetch is cached per Google's own `Cache-Control` header. Imported
    lazily so a `disabled`/`basic` deployment never imports google-auth's
    transport stack at all."""
    global _cached_transport_request
    if _cached_transport_request is None:
        import google.auth.transport.requests as ga_requests
        import requests
        from cachecontrol import CacheControl

        _cached_transport_request = ga_requests.Request(session=CacheControl(requests.Session()))
    return _cached_transport_request


def verify_firebase_id_token(token: str, config: AuthConfig) -> VerifiedFirebaseToken:
    """Verify a Firebase ID token string and return its trusted claims,
    or raise `FirebaseTokenError` (the one generic failure). `config`
    must be a `mode == "firebase"` AuthConfig (its `firebase_project_id`
    is the required audience/issuer; `firebase_emulator_host`, if set,
    switches to the unsigned-emulator path).

    Every internal exception -- a google-auth `ValueError`, a `requests`
    network error fetching certs, a `KeyError` on a missing claim -- is
    caught and re-raised as `FirebaseTokenError` with an internal-only
    reason. The original exception is never chained into a message a
    caller could surface.
    """
    project_id = config.firebase_project_id
    assert project_id is not None  # guaranteed by get_auth_config for mode == "firebase"

    if config.firebase_emulator_host:
        # Emulator path: unsigned tokens. Decode claims, then apply the
        # exact same iss/aud/sub/exp/iat checks the real path applies.
        claims = _decode_unverified_claims(token)
        return _claims_to_verified_token(claims, project_id=project_id, check_time=True)

    # Real path: google-auth verifies signature (RS256 against Google's
    # securetoken certs), aud, exp and iat; this module then adds iss +
    # sub.
    try:
        import google.oauth2.id_token as google_id_token

        claims = google_id_token.verify_firebase_token(
            token,
            request=_get_cached_transport_request(),
            audience=project_id,
            clock_skew_in_seconds=_CLOCK_SKEW_SECONDS,
        )
    except FirebaseTokenError:
        raise
    except Exception as exc:  # noqa: BLE001 -- every verifier failure collapses to one generic error
        raise FirebaseTokenError(f"token verification failed: {type(exc).__name__}") from None

    if not isinstance(claims, dict):
        raise FirebaseTokenError("verifier returned no claims")
    # google-auth already checked exp/iat; re-checking would double-count
    # clock skew, so check_time=False here.
    return _claims_to_verified_token(claims, project_id=project_id, check_time=False)
