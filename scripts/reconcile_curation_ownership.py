"""Day 2 (public multi-user deployment foundation, see
docs/plans/public-multi-user-deployment-review.md, Correction 4):
find and (optionally) remove stale cross-store ownership orphans.

Two orphan classes, exactly as documented in
`research_agent/curation_ownership.py`'s own module docstring:

1. **owner_without_checkpoint** -- a `curation_owners` row with no
   matching LangGraph checkpoint (a failed `/curation/start` between the
   owner-insert and the checkpoint-write, or a checkpoint deleted out of
   band). Eligible for removal once older than
   `--owner-orphan-safety-window-seconds` (default 3600 = 1 hour) --
   short, because a genuinely failed create attempt is already a dead
   end for its caller (who is expected to retry with a brand-new
   session_id, never this one).
2. **checkpoint_without_owner** -- a LangGraph checkpoint thread with no
   matching `curation_owners` row (a failed `DELETE` between the
   owner-delete and the checkpoint-delete, or a bug). Eligible for
   removal once its most recent checkpoint's own timestamp is older than
   `--checkpoint-orphan-safety-window-seconds` (default 86400 = 24
   hours) -- longer, to avoid racing a create that is still legitimately
   in flight (see this module's own `_checkpoint_thread_activity`).

**Never invents ownership.** This script only ever deletes an orphan
record; it never assigns a `curation_owners` row to an orphaned
checkpoint, however confident a heuristic might seem -- assigning
abandoned data to whoever happens to run this script is a security bug,
not a repair (see `research_agent/curation_ownership.py`'s own
docstring for the same point).

**Dry-run by default.** Without `--apply`, this script only reports what
it WOULD remove; nothing is deleted. `--apply` is required, explicitly,
to actually mutate either store.

**Report contains only opaque IDs and counts.** Never a topic,
`display_title`, or email -- this script never even SELECTs those
columns from `curation_owners`/`users` in the first place, so there is
no field to accidentally print.

**Explicit paths only.** `--database-url` and `--checkpoint-db-path` are
both required, with no default pointing at any real path -- an operator
must name the target database and checkpoint file explicitly every time,
the same discipline `scripts/data_backup.py` already establishes for
`--data-dir`/`--snapshots-dir`.

Usage:

    # Dry run (the default) -- reports only, mutates nothing.
    uv run python scripts/reconcile_curation_ownership.py \\
        --database-url postgresql://user:pass@host:port/db \\
        --checkpoint-db-path /path/to/qa_checkpoints.sqlite

    # Apply -- actually deletes eligible stale orphans.
    uv run python scripts/reconcile_curation_ownership.py \\
        --database-url postgresql://user:pass@host:port/db \\
        --checkpoint-db-path /path/to/qa_checkpoints.sqlite \\
        --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Same convention every other research_agent-importing script under
# scripts/ already uses (e.g. scripts/eval_retrieval.py) -- running this
# file directly (`python scripts/reconcile_curation_ownership.py`) sets
# sys.path[0] to this file's own directory, not the repo root, so
# `research_agent` is not importable without this.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg

from research_agent.curation_session import THREAD_ID_PREFIX
from research_agent.qa import sqlite_checkpointer

DEFAULT_OWNER_ORPHAN_SAFETY_WINDOW_SECONDS = 3600
DEFAULT_CHECKPOINT_ORPHAN_SAFETY_WINDOW_SECONDS = 86400


@dataclass(frozen=True)
class _OwnerRow:
    session_id: str
    created_at: datetime


@dataclass(frozen=True)
class _CheckpointThread:
    session_id: str
    last_activity: datetime


def _fetch_owner_rows(conn: psycopg.Connection) -> list[_OwnerRow]:
    """Only session_id and created_at -- never topic/display_title, and
    never a join against users (no email is ever read by this script)."""
    with conn.cursor() as cur:
        cur.execute("SELECT session_id, created_at FROM curation_owners")
        rows = cur.fetchall()
    return [_OwnerRow(session_id=r[0], created_at=r[1]) for r in rows]


def _checkpoint_threads(checkpointer) -> list[_CheckpointThread]:
    """One entry per curation-session thread_id, keyed to its MOST
    RECENT checkpoint's own timestamp -- `checkpointer.list(None)` is
    verified elsewhere in this codebase (research_agent/curation_session.
    py's own list_curation_sessions()) to iterate newest-first globally,
    so first-seen per thread_id is that thread's latest activity; reused
    here rather than re-derived."""
    seen: dict[str, datetime] = {}
    for tup in checkpointer.list(None):
        thread_id = tup.config["configurable"]["thread_id"]
        if not thread_id.startswith(THREAD_ID_PREFIX):
            continue
        session_id = thread_id[len(THREAD_ID_PREFIX):]
        if session_id in seen:
            continue
        ts_raw = tup.checkpoint.get("ts")
        ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.now(timezone.utc)
        seen[session_id] = ts
    return [_CheckpointThread(session_id=sid, last_activity=ts) for sid, ts in seen.items()]


@dataclass(frozen=True)
class ReconciliationReport:
    mode: str
    owner_without_checkpoint_total: int
    owner_without_checkpoint_eligible: list[str]
    checkpoint_without_owner_total: int
    checkpoint_without_owner_eligible: list[str]
    removed_owner_rows: int = 0
    removed_checkpoints: int = 0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "owner_without_checkpoint": {
                "total": self.owner_without_checkpoint_total,
                "eligible_count": len(self.owner_without_checkpoint_eligible),
                "eligible_session_ids": self.owner_without_checkpoint_eligible,
            },
            "checkpoint_without_owner": {
                "total": self.checkpoint_without_owner_total,
                "eligible_count": len(self.checkpoint_without_owner_eligible),
                "eligible_session_ids": self.checkpoint_without_owner_eligible,
            },
            "removed": {
                "owner_rows": self.removed_owner_rows,
                "checkpoints": self.removed_checkpoints,
            },
        }


def run_reconciliation(
    conn: psycopg.Connection, checkpointer, *, apply: bool,
    owner_orphan_safety_window: timedelta, checkpoint_orphan_safety_window: timedelta,
    now: datetime | None = None,
) -> ReconciliationReport:
    """The core, database-agnostic-to-its-CALLER logic -- takes an
    already-open connection/checkpointer so tests can exercise this
    directly against an isolated database, exactly like every other
    function in this codebase that accepts a connection rather than a
    path/URL (see storage.py's own convention)."""
    now = now if now is not None else datetime.now(timezone.utc)

    owner_rows = _fetch_owner_rows(conn)
    owner_session_ids = {row.session_id for row in owner_rows}
    checkpoint_threads = _checkpoint_threads(checkpointer)
    checkpoint_session_ids = {t.session_id for t in checkpoint_threads}

    owner_without_checkpoint = [row for row in owner_rows if row.session_id not in checkpoint_session_ids]
    owner_without_checkpoint_eligible = [
        row.session_id for row in owner_without_checkpoint
        if now - row.created_at >= owner_orphan_safety_window
    ]

    checkpoint_without_owner = [t for t in checkpoint_threads if t.session_id not in owner_session_ids]
    checkpoint_without_owner_eligible = [
        t.session_id for t in checkpoint_without_owner
        if now - t.last_activity >= checkpoint_orphan_safety_window
    ]

    removed_owner_rows = 0
    removed_checkpoints = 0
    if apply:
        # One bounded transaction for every eligible owner-row delete --
        # all-or-nothing for this store's own side of the cleanup, never
        # a partial set of rows removed if one delete somehow fails
        # partway through.
        if owner_without_checkpoint_eligible:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM curation_owners WHERE session_id = ANY(%s)",
                    (owner_without_checkpoint_eligible,),
                )
                removed_owner_rows = cur.rowcount
            conn.commit()
        # The checkpoint store is a different engine entirely -- each
        # delete_thread() call is its own operation; LangGraph's own
        # SqliteSaver.delete_thread() is a plain DELETE ... WHERE
        # thread_id = ? (see curation_session.delete_curation_session's
        # own docstring), already safe to call per-session_id.
        for session_id in checkpoint_without_owner_eligible:
            checkpointer.delete_thread(f"{THREAD_ID_PREFIX}{session_id}")
            removed_checkpoints += 1

    return ReconciliationReport(
        mode="apply" if apply else "dry-run",
        owner_without_checkpoint_total=len(owner_without_checkpoint),
        owner_without_checkpoint_eligible=owner_without_checkpoint_eligible,
        checkpoint_without_owner_total=len(checkpoint_without_owner),
        checkpoint_without_owner_eligible=checkpoint_without_owner_eligible,
        removed_owner_rows=removed_owner_rows,
        removed_checkpoints=removed_checkpoints,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-url", required=True, help="Full postgresql:// connection string. No default -- must be explicit.")
    parser.add_argument("--checkpoint-db-path", required=True, type=Path, help="Path to the LangGraph qa_checkpoints.sqlite file. No default -- must be explicit.")
    parser.add_argument("--apply", action="store_true", help="Actually delete eligible stale orphans. Without this flag, reports only.")
    parser.add_argument(
        "--owner-orphan-safety-window-seconds", type=int, default=DEFAULT_OWNER_ORPHAN_SAFETY_WINDOW_SECONDS,
        help=f"Default {DEFAULT_OWNER_ORPHAN_SAFETY_WINDOW_SECONDS}s (1 hour).",
    )
    parser.add_argument(
        "--checkpoint-orphan-safety-window-seconds", type=int, default=DEFAULT_CHECKPOINT_ORPHAN_SAFETY_WINDOW_SECONDS,
        help=f"Default {DEFAULT_CHECKPOINT_ORPHAN_SAFETY_WINDOW_SECONDS}s (24 hours).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.owner_orphan_safety_window_seconds <= 0 or args.checkpoint_orphan_safety_window_seconds <= 0:
        print("error: safety window values must be positive integers.", file=sys.stderr)
        return 1
    # Deliberately does NOT require --checkpoint-db-path to already
    # exist: a genuinely fresh deployment (no curation session has ever
    # been created yet) has no checkpoint file at all, and that is a
    # valid, non-error starting state -- sqlite_checkpointer() below
    # creates it (and its parent directory) on first use, exactly as the
    # real application does. Requiring explicitness is satisfied by
    # requiring the ARGUMENT itself (no default), not by requiring the
    # file it names to already have content.

    try:
        conn = psycopg.connect(args.database_url)
    except Exception as exc:
        print(f"error: could not connect to --database-url: {type(exc).__name__}", file=sys.stderr)
        return 1

    try:
        with conn, sqlite_checkpointer(args.checkpoint_db_path) as checkpointer:
            report = run_reconciliation(
                conn, checkpointer, apply=args.apply,
                owner_orphan_safety_window=timedelta(seconds=args.owner_orphan_safety_window_seconds),
                checkpoint_orphan_safety_window=timedelta(seconds=args.checkpoint_orphan_safety_window_seconds),
            )
    except Exception as exc:
        # Class name only, never the exception's own message -- a
        # psycopg/database error's message can echo back connection
        # details; see this module's own "never appears in logs/errors"
        # requirement, same posture research_agent/telemetry.py's
        # error_type field already establishes for provider errors.
        print(f"error: reconciliation failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
