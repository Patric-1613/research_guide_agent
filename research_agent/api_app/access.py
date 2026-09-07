"""Mode-aware product-access boundary: approval and per-user ownership.

One place decides, per request, whether the caller may run a *product*
operation (start or read a curation, generate a report, list reviews)
and -- for a session-scoped operation -- whether they own the resource.
Routers depend on the callables here instead of parsing ``AUTH_MODE``,
reading ``RequestIdentity`` out of the ASGI scope, or touching the
ownership tables themselves.

Behaviour by mode (``request.app.state.auth_config.mode``):

* ``firebase`` -- the request already carries a verified
  ``RequestIdentity`` (``FirebaseAuthMiddleware`` placed it in the ASGI
  scope; a request without one never reached a product route). A product
  operation additionally requires ``approved``; an unapproved account
  gets a generic ``403`` -- ``/me`` is the one route that does not use
  this dependency, so an unapproved user can still see their own status.
  A session-scoped operation additionally requires a ``curation_owners``
  row owned by this user; a row owned by someone else and a row that
  does not exist are the *same* generic ``404``, so ownership is never
  disclosed.
* ``basic`` -- the shared-credential single-user deployment. Every
  callable here is a pass-through: the request is already authenticated
  by ``BasicAuthMiddleware`` and there is exactly one user, so there is
  nothing to approve and nothing to scope.
* ``disabled`` -- local development. Same pass-through.

Fail closed: if the auth configuration is not readable or the ownership
database cannot be reached, a product operation gets a generic ``503``,
never a fall-through to the route. No credential, token, uid, email, or
database detail is ever placed in a response.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from fastapi import Depends, HTTPException, Request

from research_agent.api_app.runtime import _state
from research_agent.identity import RequestIdentity

_NO_STORE = {"Cache-Control": "no-store"}
_BEARER = {"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"}

_UNAUTHORIZED = {"reason_code": "unauthorized", "message": "Authentication required."}
_NOT_APPROVED = {"reason_code": "account_not_approved", "message": "This account is awaiting approval."}
_UNAVAILABLE = {"reason_code": "identity_store_unavailable", "message": "Authorization is temporarily unavailable."}

# The exact string the curation routes already 404 with for an unknown
# session_id -- a cross-owner session must be indistinguishable from one
# that was never created.
_SESSION_NOT_FOUND = "session_id not found"


@dataclass(frozen=True)
class AccessContext:
    """The resolved principal for one request. `identity` is populated
    only in firebase mode; `owner_id` is the internal `users.id` that
    scopes every owned resource."""

    mode: str
    identity: RequestIdentity | None

    @property
    def is_firebase(self) -> bool:
        return self.mode == "firebase"

    @property
    def owner_id(self) -> uuid.UUID | None:
        return self.identity.user_id if self.identity is not None else None


def _auth_mode(request: Request) -> str:
    auth_config = getattr(request.app.state, "auth_config", None)
    if auth_config is None:
        # create_app() always sets this; its absence means the app was
        # built by something other than create_app(). Fail closed.
        raise HTTPException(status_code=503, detail=_UNAVAILABLE, headers=_NO_STORE)
    return auth_config.mode


def get_access_context(request: Request) -> AccessContext:
    """The verified principal plus the deployment's auth mode. Does NOT
    check approval -- a product route wants `require_approved_access`."""
    mode = _auth_mode(request)
    if mode != "firebase":
        return AccessContext(mode=mode, identity=None)
    identity = request.scope.get("state", {}).get("identity")
    if not isinstance(identity, RequestIdentity):
        # FirebaseAuthMiddleware guarantees an identity for every route it
        # lets through; reaching here means it was bypassed.
        raise HTTPException(status_code=401, detail=_UNAUTHORIZED, headers=_BEARER)
    return AccessContext(mode="firebase", identity=identity)


def require_approved_access(access: AccessContext = Depends(get_access_context)) -> AccessContext:
    """Gate a product operation. firebase: the identity must be approved
    and not disabled. basic/disabled: pass-through."""
    if access.is_firebase:
        identity = access.identity
        assert identity is not None  # is_firebase implies a resolved identity
        if identity.disabled:
            raise HTTPException(status_code=401, detail=_UNAUTHORIZED, headers=_BEARER)
        if not identity.approved:
            raise HTTPException(status_code=403, detail=_NOT_APPROVED, headers=_NO_STORE)
    return access


def deny_when_multiuser(request: Request) -> None:
    """Attached to the legacy single-user search/library/summarize/chat/
    export routes: those read and mutate the shared SQLite `searches`
    table by integer id and have no per-user scoping, so in firebase
    mode they are refused outright rather than exposing one user's search
    history to everyone. See docs/architecture.md, "Authorization
    (Firebase multi-user mode)"."""
    if _auth_mode(request) == "firebase":
        raise HTTPException(
            status_code=403,
            detail={
                "reason_code": "unavailable_in_multiuser_mode",
                "message": "This endpoint is not available on this deployment.",
            },
            headers=_NO_STORE,
        )


# --- ownership store access ---------------------------------------------


@contextmanager
def ownership_repo() -> Iterator[object]:
    """A `PostgresOwnershipRepository` over one pooled connection, or a
    generic 503 if the pool is missing or unreachable. The pool is the
    one `lifespan()` opens for firebase mode (`_state["db_pool"]`), the
    same handle `identity.build_firebase_authenticator` reads."""
    import psycopg
    import psycopg_pool

    from research_agent.db.ownership_repository import PostgresOwnershipRepository

    pool = _state.get("db_pool")
    if pool is None:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE, headers=_NO_STORE)
    try:
        with pool.connection() as conn:
            yield PostgresOwnershipRepository(conn)
    except HTTPException:
        raise
    except (psycopg.Error, psycopg_pool.PoolTimeout):
        raise HTTPException(status_code=503, detail=_UNAVAILABLE, headers=_NO_STORE) from None


def require_curation_session_access(
    session_id: str, access: AccessContext = Depends(require_approved_access),
) -> AccessContext:
    """Gate a session-scoped curation operation. firebase: the caller
    must own `session_id`; a missing owner row and a cross-owner row both
    raise the same generic `404`, before any checkpoint content is read
    or any paid-action lease is opened. basic/disabled: pass-through."""
    if not access.is_firebase:
        return access

    from research_agent.curation_ownership import get_owner_id_for_session

    with ownership_repo() as repo:
        owner_id = get_owner_id_for_session(session_id, repo)
    if owner_id is None or owner_id != access.owner_id:
        raise HTTPException(status_code=404, detail=_SESSION_NOT_FOUND)
    return access


def record_curation_ownership(session_id: str, owner_id: uuid.UUID, *, topic: str, stage: str) -> None:
    """Insert the `curation_owners` row for a freshly created firebase
    session. A failure here is a generic `503`; the just-written
    checkpoint is then an unowned orphan, unreachable through
    `require_curation_session_access` and reclaimed by
    `scripts/reconcile_curation_ownership.py` -- fail-closed, never a
    session visible without an owner."""
    with ownership_repo() as repo:
        repo.create_owner_row(
            session_id=session_id, owner_id=owner_id, topic=topic,
            display_title=None, stage=stage,
        )
