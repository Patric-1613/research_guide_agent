"""Day 3: the request identity model and the verified-token -> internal-user
resolution.

`RequestIdentity` is the immutable, request-scoped principal a Day-4
route/ownership check will read. It is populated by
`firebase_auth_middleware.FirebaseAuthMiddleware` into the ASGI
`scope["state"]` for each successfully-verified request -- never a module
global. `get_current_user` / `require_approved_user` are the FastAPI
dependencies that read it back.

**The ownership key is `RequestIdentity.user_id`** -- an internal UUID
(`users.id`), assigned once on a user's first sign-in and never changed.
`firebase_uid` maps one-to-one to it (unique in the schema). `email` and
`display_name` are mutable metadata and are refreshed from the token on
every sign-in; changing them never changes `user_id`. No email or
Firebase uid is ever put into a LangGraph thread ID (that stays the bare
`curation-session:<uuid4hex>` shape from Day 2).

**Fail closed.** A disabled user is denied (`IdentityDenied`). If
PostgreSQL is unreachable, a firebase-mode request is denied
(`IdentityUnavailable`) -- it must never fall through to the route with
no identity, and it must never create an unowned identity.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable

from fastapi import HTTPException, Request

from research_agent.config.settings import AuthConfig
from research_agent.firebase_auth import VerifiedFirebaseToken, verify_firebase_id_token


@dataclass(frozen=True)
class RequestIdentity:
    """The verified principal for one request. Request-scoped: one
    instance is built per verified request and lives only in that
    request's ASGI `scope["state"]`."""

    user_id: uuid.UUID
    firebase_uid: str
    email: str | None
    display_name: str | None
    approved: bool
    disabled: bool


class IdentityDenied(Exception):
    """The token verified, but this identity is not allowed through
    (today: the user row is `disabled`). Maps to a generic 401, same as
    a bad token -- a client is never told which."""


class IdentityUnavailable(Exception):
    """The token verified, but the internal user could not be resolved
    because the identity store (PostgreSQL) is unreachable. Maps to 503:
    the request is denied and the route never runs, but this is an
    infrastructure fault, not an auth failure -- retrying later may
    succeed. Never carries the underlying database exception outward."""


def resolve_request_identity(
    verified: VerifiedFirebaseToken, ownership_repo,
) -> RequestIdentity:
    """Given verified token claims and a `PostgresOwnershipRepository`,
    return the `RequestIdentity`. Creates the internal user on a first
    sign-in (`approved=False`) or refreshes its email/display-name
    metadata otherwise -- see
    `PostgresOwnershipRepository.sync_user_from_identity` for the
    race-safe upsert. Raises `IdentityDenied` for a disabled user.

    Does NOT catch database errors -- the caller
    (`build_firebase_authenticator` below) owns the connection/pool and
    converts a `psycopg`/pool failure into `IdentityUnavailable`.
    """
    user = ownership_repo.sync_user_from_identity(
        firebase_uid=verified.firebase_uid,
        email=verified.email,
        display_name=verified.display_name,
    )
    if user.disabled:
        raise IdentityDenied("user is disabled")
    return RequestIdentity(
        user_id=user.id,
        firebase_uid=user.firebase_uid,
        email=user.email,
        display_name=user.display_name,
        approved=user.approved,
        disabled=user.disabled,
    )


def build_firebase_authenticator(auth_config: AuthConfig) -> Callable[[str], RequestIdentity]:
    """Returns the one callable `FirebaseAuthMiddleware` invokes per
    request: `authenticate(token: str) -> RequestIdentity`, raising
    `firebase_auth.FirebaseTokenError` / `IdentityDenied` /
    `IdentityUnavailable`.

    The returned closure reads the connection pool from
    `research_agent.api_app.runtime._state["db_pool"]` at call time (the
    pool is created by `lifespan()`, after this closure is built in
    `create_app()`) -- the same lazy-`_state` pattern
    `get_curation_checkpointer` already uses. A missing pool, a pool
    acquisition timeout, or any `psycopg` error becomes
    `IdentityUnavailable` -- the database exception itself is never
    chained into a message.

    This function is called unconditionally in `create_app()`; in
    `disabled`/`basic` mode the middleware never invokes the result, so
    building it here is free and imports nothing heavy.
    """

    def authenticate(token: str) -> RequestIdentity:
        verified = verify_firebase_id_token(token, auth_config)

        import psycopg
        import psycopg_pool

        from research_agent.api_app.runtime import _state
        from research_agent.db.ownership_repository import PostgresOwnershipRepository

        pool = _state.get("db_pool")
        if pool is None:
            raise IdentityUnavailable("identity database pool is not initialized")
        try:
            with pool.connection() as conn:
                return resolve_request_identity(verified, PostgresOwnershipRepository(conn))
        except IdentityDenied:
            raise
        except (psycopg.Error, psycopg_pool.PoolTimeout) as exc:
            raise IdentityUnavailable(f"identity store unavailable: {type(exc).__name__}") from None

    return authenticate


def get_current_user(request: Request) -> RequestIdentity:
    """FastAPI dependency: the verified `RequestIdentity` for this
    request, or a generic 401. In `firebase` mode
    `FirebaseAuthMiddleware` guarantees this is set for every non-public
    route it lets through; reaching the 401 here means either the
    deployment is not in firebase mode (this dependency was used on a
    route in a `basic`/`disabled` deployment) or the middleware was
    bypassed -- both are treated as "not authenticated"."""
    identity = request.scope.get("state", {}).get("identity")
    if not isinstance(identity, RequestIdentity):
        raise HTTPException(
            status_code=401,
            detail={"reason_code": "unauthorized", "message": "Authentication required."},
            headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
        )
    return identity


def require_approved_user(request: Request) -> RequestIdentity:
    """FastAPI dependency: like `get_current_user`, but also 403s an
    authenticated-but-not-yet-approved account. NOT wired onto any
    product route in Day 3 -- see
    `docs/plans/public-multi-user-deployment-review.md`; global approval
    enforcement lands with Day 4's ownership wiring, applied to the same
    routes. Provided and tested now so Day 4 has one ready dependency."""
    user = get_current_user(request)
    if not user.approved:
        raise HTTPException(
            status_code=403,
            detail={
                "reason_code": "account_not_approved",
                "message": "This account is awaiting approval.",
            },
            headers={"Cache-Control": "no-store"},
        )
    return user
