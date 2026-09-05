"""Day 2 (public multi-user deployment foundation, see
docs/plans/public-multi-user-deployment-review.md, Correction 4):
the service-layer ownership coordinator.

**PostgreSQL ownership is the reachability source of truth; the SQLite/
LangGraph checkpoint is secondary.** This module is the ONE place that
composes a `PostgresOwnershipRepository` write with a
`research_agent.curation_session` checkpoint write/delete/read, in the
exact order and with the exact failure handling Correction 4 specifies.
Nothing here talks to psycopg or the checkpointer's own internals
directly beyond calling the existing, unmodified functions in
`research_agent.db.ownership_repository` and
`research_agent.curation_session` -- this module is pure composition and
ordering, not a third persistence mechanism of its own.

**Not wired into any route, service, or FastAPI dependency yet.**
`owner_id` is an explicit, caller-supplied parameter on every function
below -- there is no Firebase, no `IdentityMiddleware`, and no
`get_current_user` dependency in this checkpoint (see
`docs/plans/public-multi-user-deployment-review.md`'s own Day 3/Day 4
split). This module exists so that wiring, when it happens, has a single
correct, already-tested place to call into rather than reimplementing
this ordering inline in a router or service function.

Creation order (session_id minted here, never by the repository):
    1. `curation_owners` row inserted first.
    2. The LangGraph checkpoint created second (via
       `curation_session.save_curation_session`, unmodified).
    - Owner-insert failure -> the checkpoint step is never reached; the
      caller sees the exception and must retry with a brand-new call
      (which mints a brand-new session_id) -- never a retry against the
      same, failed session_id.
    - Checkpoint-write failure -> the owner row is left in place
      (a "fail-closed incomplete" session, see `get_owned_curation_state`
      below for what that means on read) and the exception propagates;
      this module never deletes the just-inserted owner row itself
      (that would be a second failure-handling path to get wrong) --
      `scripts/reconcile_curation_ownership.py` is the only thing that
      ever cleans up a stale incomplete owner row, and only after its own
      documented safety window has passed.

Deletion order:
    1. `curation_owners` row deleted first.
    2. The LangGraph checkpoint deleted second (via
       `curation_session.delete_curation_session`, unmodified).
    - The instant step 1 succeeds, the session is unreachable through
      this module's own read path (`get_owned_curation_state` below),
      regardless of whether step 2 later succeeds or fails.
    - Checkpoint-delete failure -> the checkpoint's bytes may remain on
      disk, orphaned and unreachable; this module never recreates or
      guesses an owner for them. `scripts/reconcile_curation_ownership.py`
      is what eventually reclaims that disk space.

Thread IDs stay opaque: `session_id` here is the same bare `uuid4().hex`
`research_agent.curation_session.curation_thread_id()` already prefixes
into `"curation-session:<session_id>"` -- no user id or email is ever
encoded into it. The mapping from session_id to owner lives ONLY in the
`curation_owners` table.
"""

from __future__ import annotations

import uuid

from langgraph.checkpoint.base import BaseCheckpointSaver

from research_agent.curation_session import (
    delete_curation_session,
    load_curation_session,
    save_curation_session,
)
from research_agent.db.ownership_repository import CurationOwnerRecord, PostgresOwnershipRepository
from research_agent.query_expansion import PaperPoolSession


def mint_session_id() -> str:
    """The one place a new curation session_id is generated for an
    owned session -- a bare uuid4 hex, identical in shape to every
    existing (unowned) session_id this codebase has ever minted (see
    `research_agent.services.curation_core_service.start_curation`),
    so nothing about a session_id's own format reveals whether it is
    owned."""
    return uuid.uuid4().hex


def create_owned_curation_session(
    *, owner_id: uuid.UUID, session: PaperPoolSession, topic: str,
    display_title: str | None, stage: str,
    ownership_repo: PostgresOwnershipRepository, checkpointer: BaseCheckpointSaver,
) -> str:
    """Mints a session_id, inserts its `curation_owners` row, then
    creates its checkpoint -- in that order, per this module's own
    docstring. Returns the new session_id on full success.

    Raises whatever the failing step raised, unchanged (no exception
    substitution) -- a caller inspects/handles the specific failure; this
    function's only contract is the ORDERING and the "never repair, only
    a fresh mint on retry" policy, not swallowing or reclassifying
    errors.
    """
    session_id = mint_session_id()
    ownership_repo.create_owner_row(
        session_id=session_id, owner_id=owner_id, topic=topic,
        display_title=display_title, stage=stage,
    )
    # If this raises, the owner row above is intentionally left in place
    # -- see this module's own docstring ("fail-closed incomplete") for
    # why that is correct, not a bug, and why this function must never
    # try to delete it itself.
    save_curation_session(session, session_id, checkpointer)
    return session_id


def delete_owned_curation_session(
    *, session_id: str, ownership_repo: PostgresOwnershipRepository, checkpointer: BaseCheckpointSaver,
) -> bool:
    """Deletes the `curation_owners` row first, then the checkpoint --
    in that order, per this module's own docstring. Returns True if an
    owner row was actually found and deleted, False if the session_id
    had no owner row to begin with (idempotent, matching
    `PostgresOwnershipRepository.delete_owner_row`'s own contract) --
    either way, the checkpoint-delete step still runs (deleting a
    checkpoint with no owner row is a safe no-op, same reasoning
    `curation_session.delete_curation_session` already documents for a
    session_id that was never saved).

    If the checkpoint-delete step raises, that exception propagates
    unchanged; the owner row is already gone by that point regardless."""
    owner_existed = ownership_repo.delete_owner_row(session_id)
    delete_curation_session(session_id, checkpointer)
    return owner_existed


def get_owner_id_for_session(
    session_id: str, ownership_repo: PostgresOwnershipRepository,
) -> uuid.UUID | None:
    """None if no `curation_owners` row exists for this session_id --
    the caller's 404-equivalent. Does not touch the checkpointer at all;
    this is the cheap "does anyone own this" check a route/service layer
    can use before deciding whether to even attempt a checkpoint read."""
    row = ownership_repo.get_owner_row(session_id)
    return row.owner_id if row is not None else None


def list_owned_curation_sessions(
    owner_id: uuid.UUID, ownership_repo: PostgresOwnershipRepository, checkpointer: BaseCheckpointSaver,
    *, limit: int = 100,
) -> list[CurationOwnerRecord]:
    """The fail-closed LISTING path, the plural counterpart to
    `get_owned_curation_state` above -- returns only this owner's
    `curation_owners` rows that ALSO have a real checkpoint behind them.

    `PostgresOwnershipRepository.list_owner_sessions` on its own returns
    every `curation_owners` row for an owner, **incomplete ones
    included** -- a row whose checkpoint write failed (the "fail-closed
    incomplete" state from this module's own docstring) is a real row
    until `scripts/reconcile_curation_ownership.py` sweeps it (default:
    up to 1 hour later). A route/service layer that lists a user's
    sessions must never surface those: they would render as "phantom"
    entries that 404 the moment the user opens them (because
    `get_owned_curation_state` correctly returns None for them). This
    function is the one place that cross-check lives, so a Day-4
    `GET /curation/reviews` wiring calls this rather than the raw
    repository method and never has to re-derive the filter.

    The cross-check is one `load_curation_session` per candidate row --
    acceptable at a controlled beta's per-user session counts, and the
    same per-row cost `research_agent.curation_session.list_curation_
    sessions` already pays for its own listing.
    """
    candidate_rows = ownership_repo.list_owner_sessions(owner_id, limit=limit)
    return [
        row for row in candidate_rows
        if load_curation_session(row.session_id, checkpointer) is not None
    ]


def get_owned_curation_state(
    session_id: str, ownership_repo: PostgresOwnershipRepository, checkpointer: BaseCheckpointSaver,
) -> tuple[CurationOwnerRecord, PaperPoolSession] | None:
    """The fail-closed read path: returns None (never a partially-usable
    result) whenever EITHER of these is true:
      - no `curation_owners` row exists for session_id at all, or
      - a `curation_owners` row exists but `load_curation_session`
        returns None (the "owner row exists, checkpoint doesn't yet /
        anymore" incomplete-or-orphaned case from this module's own
        docstring).

    Only returns a real `(owner_record, session)` pair when BOTH stores
    agree the session genuinely exists -- this is what makes "missing
    checkpoint state is never returned as a usable session" true by
    construction, not by a caller remembering to check twice.
    """
    owner_record = ownership_repo.get_owner_row(session_id)
    if owner_record is None:
        return None
    session = load_curation_session(session_id, checkpointer)
    if session is None:
        return None
    return owner_record, session
