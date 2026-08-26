#!/usr/bin/env python3
"""PR3.3: a platform-neutral backup/verify/restore workflow for this
project's data directory (SQLite databases + WAL/SHM sidecars, the
Chroma vector store, and the on-disk caches) — a plain local-filesystem
tool, deliberately. **No cloud storage client, no PostgreSQL, no managed
vector database, no new third-party dependency** — everything here is
Python's own standard library (`hashlib`, `json`, `os`, `shutil`,
`tempfile`), so this works identically whatever the eventual hosting
platform turns out to be.

Three subcommands, one file:

    data_backup.py create   --data-dir DIR --snapshots-dir DIR --confirm-quiescent [--name NAME]
    data_backup.py verify   SNAPSHOT_DIR
    data_backup.py restore  SNAPSHOT_DIR DEST_DIR [--force]

**What this tool guarantees:**

- Every REGULAR file beneath `--data-dir` is included, addressed by its
  path relative to `--data-dir` (POSIX-style, forward slashes, so a
  manifest is portable across OSes).
- A snapshot's `manifest.json` records, for every file, its relative
  path, exact byte size, and SHA-256 — the one source of truth `verify`/
  `restore` both check everything else against.
- A snapshot is built entirely inside a `.staging-*` directory (created
  with `tempfile.mkdtemp` INSIDE `--snapshots-dir`, so the final publish
  step is a same-filesystem, single `os.rename` — as close to atomic as
  a plain local filesystem gets) and is **only renamed into its final,
  published name after a full self-verification pass of the staged copy
  succeeds**. Any failure during copy or self-verification deletes the
  staging directory and leaves NO trace of a partial/corrupt snapshot —
  nothing is ever half-published.
- `restore` re-verifies the SOURCE snapshot against its own manifest
  BEFORE copying anything, and re-verifies the DESTINATION after
  copying, before declaring success.
- `restore` refuses a destination that already exists and is non-empty
  unless `--force` is passed explicitly — this tool never overwrites a
  real data directory silently. Even with `--force`, it only ever
  copies/overwrites the files the manifest lists; it never deletes
  anything already present at the destination.
- **No symlink anywhere beneath `--data-dir` (or beneath a snapshot
  being verified/restored) is ever followed or backed up** — `create`
  refuses outright the moment one is found, whether it would resolve
  inside or outside the tree. This is deliberately blunt: the one
  concrete risk being closed is a symlink planted inside the data
  directory pointing OUTSIDE it (a path-traversal vector that would
  otherwise silently pull an unrelated file's content into a backup, or
  let a restore write outside its destination) — refusing every symlink
  unconditionally is simpler and strictly safer than trying to
  distinguish "this one happens to resolve back inside the tree" case by
  case.
- `--snapshots-dir` is checked against `--data-dir` at the start of
  `create` — one can never be nested inside the other (this would
  otherwise make a snapshot try to back up its own staging area, or a
  future snapshot's files bleed into what counts as "the data
  directory").

**What this tool explicitly does NOT guarantee** (see this file's own
`create` docstring and `--confirm-quiescent` flag for the load-bearing
one):

- **It does not make copying several live SQLite/Chroma stores
  cross-consistent or atomic.** SQLite databases in this project may be
  in WAL mode with a `-wal`/`-shm` sidecar holding not-yet-checkpointed
  writes; Chroma's own on-disk layout is multiple files per collection.
  A plain file copy taken WHILE the application is running can capture
  one store mid-write and another store at a different logical moment —
  a real, silent correctness risk this tool refuses to paper over. This
  is why `create` REQUIRES `--confirm-quiescent`: the operator must have
  actually stopped the application (or otherwise proven nothing is
  writing to `--data-dir` for the whole duration of the snapshot) before
  this tool will run at all. Passing the flag is an assertion the
  operator is making, not something this tool verifies for you.
- It does not preserve empty directories (only files with content are
  ever recorded — a directory containing zero files is not represented
  in the manifest and will not be recreated by `restore`).
- It does not preserve file permissions/ownership beyond whatever
  `shutil.copy2` does by default (best-effort mtime/permission bits on
  the same OS family; no claim across OSes).
- It does not upload anywhere — every path is a local filesystem path.
  Getting a snapshot directory onto remote/cloud storage afterward
  (`rsync`, `aws s3 sync`, a platform's own volume-backup feature, ...)
  is a separate, deliberately out-of-scope step for whichever hosting
  platform is eventually chosen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MANIFEST_FILENAME = "manifest.json"
MANIFEST_FORMAT_VERSION = 1
_HASH_CHUNK_BYTES = 1024 * 1024  # 1 MiB streamed reads -- correct regardless of file size.


class BackupError(Exception):
    """Raised for any condition create/verify/restore must abort on
    before doing something unsafe. Caught only at the CLI boundary
    (main()), converted to a clear stderr message + non-zero exit --
    never swallowed silently anywhere else in this module."""


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _iter_regular_files(root: Path, ignore_top_level_names: frozenset[str] = frozenset()) -> list[Path]:
    """Every REGULAR file beneath `root`, sorted by relative path for a
    deterministic manifest. `followlinks=False` stops os.walk from
    recursing through a symlinked subdirectory at all; any symlink
    encountered as either a directory or file entry raises BackupError
    immediately -- see this module's own docstring for why symlinks are
    refused outright rather than selectively allowed. `ignore_top_level_names`
    exists only so callers scanning a SNAPSHOT directory can exclude
    `manifest.json` itself (a real, legitimate file that is never one of
    the backed-up data files)."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        if current == root and ignore_top_level_names:
            filenames = [n for n in filenames if n not in ignore_top_level_names]

        safe_dirnames = []
        for name in dirnames:
            candidate = current / name
            if candidate.is_symlink():
                raise BackupError(
                    f"refusing to proceed: {candidate} is a symlinked directory. Symlinks "
                    "anywhere under the data directory are not supported -- remove or "
                    "replace it with a real directory first."
                )
            safe_dirnames.append(name)
        dirnames[:] = safe_dirnames

        for name in filenames:
            candidate = current / name
            if candidate.is_symlink():
                raise BackupError(
                    f"refusing to proceed: {candidate} is a symlink. Symlinks anywhere "
                    "under the data directory are not supported (this specifically closes "
                    "a path-traversal vector where a symlink inside the data directory "
                    "points outside it) -- remove or replace it with a real file first."
                )
            files.append(candidate)

    files.sort(key=lambda p: _relative_posix(p, root))
    return files


def _check_no_nesting(data_dir: Path, snapshots_dir: Path) -> None:
    data_dir = data_dir.resolve()
    snapshots_dir = snapshots_dir.resolve()
    if snapshots_dir == data_dir or snapshots_dir.is_relative_to(data_dir):
        raise BackupError(f"--snapshots-dir ({snapshots_dir}) must not be inside --data-dir ({data_dir})")
    if data_dir.is_relative_to(snapshots_dir):
        raise BackupError(f"--data-dir ({data_dir}) must not be inside --snapshots-dir ({snapshots_dir})")


def build_manifest(data_dir: Path) -> dict[str, Any]:
    """Walks data_dir and returns the manifest dict `create_snapshot`
    writes into the staged snapshot -- exposed separately so tests can
    exercise the walk/hash/manifest-shape logic without touching the
    filesystem-staging machinery."""
    files = _iter_regular_files(data_dir)
    entries = []
    for f in files:
        entries.append({
            "path": _relative_posix(f, data_dir),
            "size": f.stat().st_size,
            "sha256": _sha256_of_file(f),
        })
    entries.sort(key=lambda e: e["path"])
    return {
        "manifest_format_version": MANIFEST_FORMAT_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_data_dir": str(data_dir.resolve()),
        "file_count": len(entries),
        "total_bytes": sum(e["size"] for e in entries),
        "files": entries,
    }


def _diff_against_manifest(directory: Path, manifest: dict[str, Any], check_extra: bool) -> dict[str, Any]:
    """The one comparison routine `verify_snapshot` (check_extra=True --
    a published snapshot must exactly match its own manifest, nothing
    more, nothing less) and the post-restore destination check
    (check_extra=False -- `restore --force` into a non-empty destination
    may legitimately leave OTHER, unrelated pre-existing files in place;
    that is not a corruption signal) both build on."""
    manifest_entries = {e["path"]: e for e in manifest.get("files", [])}
    on_disk = _iter_regular_files(directory, ignore_top_level_names=frozenset({MANIFEST_FILENAME}))
    on_disk_by_rel = {_relative_posix(p, directory): p for p in on_disk}

    missing = sorted(set(manifest_entries) - set(on_disk_by_rel))
    extra = sorted(set(on_disk_by_rel) - set(manifest_entries)) if check_extra else []

    corrupted = []
    for rel in sorted(set(manifest_entries) & set(on_disk_by_rel)):
        entry = manifest_entries[rel]
        actual_path = on_disk_by_rel[rel]
        actual_size = actual_path.stat().st_size
        if actual_size != entry["size"]:
            corrupted.append(f"{rel}: size mismatch (expected {entry['size']}, found {actual_size})")
            continue
        if _sha256_of_file(actual_path) != entry["sha256"]:
            corrupted.append(f"{rel}: sha256 mismatch")

    return {
        "ok": not missing and not extra and not corrupted,
        "missing": missing,
        "extra": extra,
        "corrupted": corrupted,
        "file_count": len(manifest_entries),
    }


def verify_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    """Verifies a PUBLISHED snapshot directory against its own
    manifest.json. Never raises for an ordinary missing/extra/corrupted
    finding -- those are reported in the returned dict. Raises
    BackupError only when the manifest itself can't be read at all (not
    a valid snapshot in the first place) or a symlink is found inside
    the snapshot (itself a tamper/corruption signal worth a hard stop)."""
    manifest_path = snapshot_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise BackupError(f"no {MANIFEST_FILENAME} found in {snapshot_dir} -- not a valid snapshot")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise BackupError(f"manifest at {manifest_path} is not valid JSON: {exc}") from exc

    return _diff_against_manifest(snapshot_dir, manifest, check_extra=True)


def create_snapshot(data_dir: Path, snapshots_dir: Path, name: str | None = None) -> Path:
    """Builds, self-verifies, and publishes one new snapshot. Returns the
    final, published snapshot directory. Raises BackupError (staging
    directory always cleaned up first) on any failure -- a caller never
    needs to distinguish "failed to build" from "failed to verify"; both
    leave `snapshots_dir` exactly as it was before this call.

    Caller (main()) is responsible for enforcing --confirm-quiescent --
    this function does not re-check it, so it stays trivially testable
    without a CLI harness."""
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise BackupError(f"--data-dir does not exist or is not a directory: {data_dir}")

    _check_no_nesting(data_dir, snapshots_dir)
    snapshots_dir = snapshots_dir.resolve()
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    snapshot_name = name or time.strftime("snapshot-%Y%m%dT%H%M%SZ", time.gmtime())
    final_dir = snapshots_dir / snapshot_name
    if final_dir.exists():
        raise BackupError(f"a snapshot named {snapshot_name!r} already exists at {final_dir} -- choose a different --name")

    staging_dir = Path(tempfile.mkdtemp(prefix=f".staging-{snapshot_name}-", dir=snapshots_dir))
    try:
        manifest = build_manifest(data_dir)
        for entry in manifest["files"]:
            src = data_dir / entry["path"]
            dst = staging_dir / entry["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        (staging_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        # Publish only after a full self-verification of the STAGED
        # copy -- reads every staged file back and re-hashes it,
        # independent of whatever the copy loop above believed it wrote.
        result = _diff_against_manifest(staging_dir, manifest, check_extra=True)
        if not result["ok"]:
            raise BackupError(
                "staged snapshot failed self-verification before publish -- nothing was "
                f"published: missing={result['missing']} extra={result['extra']} "
                f"corrupted={result['corrupted']}"
            )

        # Same-filesystem rename (staging_dir was created INSIDE
        # snapshots_dir specifically for this) -- as close to a single
        # atomic publish step as a plain local filesystem provides.
        os.rename(staging_dir, final_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return final_dir


def restore_snapshot(snapshot_dir: Path, dest_dir: Path, force: bool = False) -> None:
    snapshot_dir = snapshot_dir.resolve()
    dest_dir = dest_dir.resolve()

    pre_result = verify_snapshot(snapshot_dir)
    if not pre_result["ok"]:
        raise BackupError(
            "refusing to restore from a snapshot that fails its own verification: "
            f"missing={pre_result['missing']} extra={pre_result['extra']} corrupted={pre_result['corrupted']}"
        )

    if dest_dir.exists():
        if not dest_dir.is_dir():
            raise BackupError(f"destination exists and is not a directory: {dest_dir}")
        if any(dest_dir.iterdir()) and not force:
            raise BackupError(
                f"destination {dest_dir} already exists and is not empty -- refusing to "
                "restore into it without --force. This tool never overwrites a real data "
                "directory silently."
            )
    else:
        dest_dir.mkdir(parents=True)

    manifest = json.loads((snapshot_dir / MANIFEST_FILENAME).read_text())
    for entry in manifest["files"]:
        src = snapshot_dir / entry["path"]
        dst = dest_dir / entry["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # check_extra=False: a --force restore into a non-empty destination
    # may legitimately leave OTHER, unrelated files in place -- that is
    # not a sign the restore itself went wrong.
    post_result = _diff_against_manifest(dest_dir, manifest, check_extra=False)
    if not post_result["ok"]:
        raise BackupError(
            "restore completed but the destination failed post-restore verification "
            f"(this indicates the copy itself was corrupted): missing={post_result['missing']} "
            f"corrupted={post_result['corrupted']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="data_backup.py",
        description=(
            "Platform-neutral backup/verify/restore for research_agent's data directory "
            "(SQLite stores + WAL/SHM sidecars, Chroma, caches). Plain local filesystem "
            "only -- no cloud storage, no new dependency."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Build, self-verify, and publish a new snapshot.")
    create_parser.add_argument("--data-dir", required=True, type=Path, help="Directory to back up (e.g. data/).")
    create_parser.add_argument("--snapshots-dir", required=True, type=Path, help="Directory snapshots are published into.")
    create_parser.add_argument("--name", default=None, help="Snapshot directory name (default: a UTC timestamp).")
    create_parser.add_argument(
        "--confirm-quiescent", action="store_true",
        help=(
            "Required. Asserts the application (and anything else that could write to "
            "--data-dir) is stopped, or --data-dir is otherwise PROVEN quiescent, for the "
            "entire duration of this command. This tool does not verify that for you and "
            "does not claim copying several live SQLite/Chroma stores is atomic."
        ),
    )

    verify_parser = subparsers.add_parser("verify", help="Verify a published snapshot against its own manifest.")
    verify_parser.add_argument("snapshot_dir", type=Path)

    restore_parser = subparsers.add_parser("restore", help="Restore a snapshot into a new or empty destination.")
    restore_parser.add_argument("snapshot_dir", type=Path)
    restore_parser.add_argument("dest_dir", type=Path)
    restore_parser.add_argument(
        "--force", action="store_true",
        help="Allow restoring into an existing, non-empty destination. Never overwrites without this.",
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "create":
            if not args.confirm_quiescent:
                print(
                    "ERROR: --confirm-quiescent is required. Stop the application (or "
                    "otherwise prove nothing is writing to --data-dir) before taking a "
                    "cross-store snapshot, then re-run with --confirm-quiescent.",
                    file=sys.stderr,
                )
                return 2
            final_dir = create_snapshot(args.data_dir, args.snapshots_dir, name=args.name)
            print(f"Snapshot created and verified: {final_dir}")
            return 0

        if args.command == "verify":
            result = verify_snapshot(args.snapshot_dir)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["ok"] else 1

        if args.command == "restore":
            restore_snapshot(args.snapshot_dir, args.dest_dir, force=args.force)
            print(f"Restored into {args.dest_dir.resolve()}")
            return 0

        parser.error(f"unknown command {args.command!r}")
        return 2
    except BackupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
