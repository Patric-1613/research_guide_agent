#!/usr/bin/env python3
"""PR3.3/PR3.3a: a platform-neutral backup/verify/restore workflow for
this project's data directory (SQLite databases + WAL/SHM sidecars, the
Chroma vector store, and the on-disk caches) — a plain local-filesystem
tool, deliberately. **No cloud storage client, no PostgreSQL, no managed
vector database, no new third-party dependency** — everything here is
Python's own standard library (`hashlib`, `json`, `os`, `shutil`,
`tempfile`), so this works identically whatever the eventual hosting
platform turns out to be.

Three subcommands, one file:

    data_backup.py create   --data-dir DIR --snapshots-dir DIR --confirm-quiescent [--name NAME]
    data_backup.py verify   SNAPSHOT_DIR
    data_backup.py restore  SNAPSHOT_DIR DEST_DIR

**PR3.3a hardening pass** (an independent review of the original PR3.3
cut found several real safety gaps in the naive first version; every
item below reflects the corrected design, not the original):

- A snapshot `--name` must be exactly one safe path component: no
  separators (`/` or `\\`, checked regardless of host OS), no empty
  name, no `.`/`..`, no absolute path — and the constructed publish
  path is confirmed to still land as a direct child of the resolved
  `--snapshots-dir` before anything is written.
- `manifest.json` is read and schema-validated EXACTLY ONCE per
  `verify`/`restore` call (`load_and_verify_snapshot`) — `restore`
  reuses that same in-memory, already-validated manifest for the actual
  copy; it never reopens or reloads the file, so nothing on disk can be
  swapped between the verification read and the copy step. `verify`
  itself never returns the manifest, only a bounded diff report.
- The manifest schema is validated strictly: the top-level object's
  exact field set and types, the supported `manifest_format_version`,
  `file_count`/`total_bytes` cross-checked against the actual entries,
  every entry's exact field set and types, every path required to be a
  normalized relative POSIX path (no absolute path, no empty path, no
  `.`/`..`/empty component, no backslash, no leading/trailing
  whitespace, never literally `manifest.json`), a 64-character lowercase
  hex SHA-256, a non-negative integer size, and no duplicate path.
  Anything else is a single, clean `BackupError` — never a raw
  `KeyError`/`TypeError`/`IndexError` surfacing from code that assumed
  well-formed input.
- Manifest.json itself is refused outright if it is a symlink (checked
  independently of the general per-file symlink walk below, since a
  snapshot-verification walk always excludes `manifest.json` by name —
  it would otherwise never be checked at all).
- Every file this tool copies (both directions) is re-checked
  immediately before the copy (`is_symlink()`) and then opened with
  `os.O_NOFOLLOW` where the platform provides it — narrowing, not fully
  closing, the TOCTOU window between an earlier directory walk/manifest
  read and the actual copy.
- `restore` now REQUIRES a destination that does not exist at all — no
  `--force`, no non-empty-destination override. It also refuses a
  symlinked destination (or a symlinked ancestor of it), the real
  repository root, the real project `data/` directory, the snapshot
  directory itself, a destination inside the snapshot, or a destination
  that itself contains the snapshot.
- `restore` is transactional: it stages the entire restored tree under a
  temp directory created inside the destination's own resolved PARENT,
  fully re-verifies that staged copy against the SAME in-memory manifest
  used to verify the source, and only then publishes with one same-
  filesystem `os.rename`. A caught failure removes the staging directory
  and leaves the requested destination absent.

**What this tool guarantees:**

- Every REGULAR file beneath `--data-dir` is included, addressed by its
  path relative to `--data-dir` (POSIX-style, forward slashes, so a
  manifest is portable across OSes).
- A snapshot's `manifest.json` records, for every file, its relative
  path, exact byte size, and SHA-256 — the one source of truth `verify`/
  `restore` both check everything else against.
- A snapshot is built entirely inside a staging directory created with
  `tempfile.mkdtemp` INSIDE `--snapshots-dir` (so the final publish step
  is a same-filesystem, single `os.rename`) and is only renamed into its
  final, published name after a full self-verification pass of the
  staged copy succeeds. Likewise, `restore` stages under the
  destination's own resolved parent and only renames into the real
  destination path after its own full self-verification pass succeeds.
- `restore` re-verifies the SOURCE snapshot against its own manifest
  BEFORE copying anything, and re-verifies the STAGED restore before
  publishing it.
- `restore` never overwrites, merges into, or deletes anything at an
  existing destination — it requires a destination that does not exist
  and creates it atomically or not at all.
- No symlink anywhere beneath `--data-dir`, beneath a snapshot being
  read, or at/above a restore destination is ever followed, backed up,
  or restored into.
- `--snapshots-dir` is checked against `--data-dir` at the start of
  `create` — one can never be nested inside the other.

**What this tool explicitly does NOT guarantee:**

- **It does not make copying several live SQLite/Chroma stores
  cross-consistent or atomic.** SQLite databases in this project may be
  in WAL mode with a `-wal`/`-shm` sidecar holding not-yet-checkpointed
  writes; Chroma's own on-disk layout is multiple files per collection.
  A plain file copy taken WHILE the application is running can capture
  one store mid-write and another store at a different logical moment.
  This is why `create` REQUIRES `--confirm-quiescent`: the operator must
  have actually stopped the application (or otherwise proven nothing is
  writing to `--data-dir` for the whole duration of the snapshot) before
  this tool will run at all. Passing the flag is an assertion the
  operator is making, not something this tool verifies for you.
- **It does not claim "no trace after ANY failure."** A normal, caught
  exception during `create` or `restore` does clean up its staging
  directory and leaves no publish. But a `SIGKILL`, a hard power loss,
  or a failure inside the cleanup call itself (`shutil.rmtree` can also
  raise) can all leave an identifiable `.staging-*`/`.restore-staging-*`
  directory behind next to the snapshots directory or the restore
  destination's parent. Such a directory is always clearly named and
  safe to delete by hand; it is never mistaken for a published snapshot
  or a completed restore, since publishing is always the LAST step and
  only ever a same-filesystem rename of an already-fully-verified tree.
- It does not preserve empty directories (only files with content are
  ever recorded).
- It does not preserve file permissions/ownership beyond whatever
  `shutil.copystat` does by default (best-effort, same OS family only).
- It does not upload anywhere — every path is a local filesystem path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_TOP_LEVEL_KEYS = frozenset({
    "manifest_format_version", "created_at_utc", "source_data_dir", "file_count", "total_bytes", "files",
})
_ALLOWED_ENTRY_KEYS = frozenset({"path", "size", "sha256"})


class BackupError(Exception):
    """Raised for any condition create/verify/restore must abort on
    before doing something unsafe. Caught only at the CLI boundary
    (main()), converted to a clear stderr message + non-zero exit --
    never swallowed silently anywhere else in this module."""


def _schema_error(message: str) -> BackupError:
    return BackupError(f"malformed manifest: {message}")


# --- hashing / filesystem walk ---------------------------------------

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
    immediately. `ignore_top_level_names` lets a caller scanning a
    SNAPSHOT directory exclude `manifest.json` by name from this walk --
    which is exactly why manifest.json's OWN symlink-ness must be
    checked separately (see `_load_manifest_dict`): a name excluded here
    is never inspected by this function at all."""
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


def _copy_regular_file_no_follow(src: Path, dst: Path) -> None:
    """Copies src -> dst, refusing to follow a symlink at the final path
    component. Uses `os.O_NOFOLLOW` (POSIX) to narrow -- not fully close
    -- the TOCTOU window between an earlier `is_symlink()` check at the
    call site and this actual read; there is no bundled equivalent for
    every platform Python runs on, so this degrades to a plain open
    (still preceded by the caller's explicit `is_symlink()` check) where
    `O_NOFOLLOW` isn't available."""
    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is not None:
        flags |= nofollow
    try:
        fd = os.open(src, flags)
    except OSError as exc:
        raise BackupError(f"refusing to read {src} (symlink or unreadable): {exc}") from exc
    dst.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(fd, "rb") as f_src, open(dst, "wb") as f_dst:
        shutil.copyfileobj(f_src, f_dst)
    shutil.copystat(src, dst)


def _copy_manifest_entries(entries: list[dict[str, Any]], src_root: Path, dst_root: Path) -> None:
    """Shared by create_snapshot (data_dir -> staging) and
    restore_snapshot (snapshot_dir -> staging): re-checks each source
    file is still a regular, non-symlink file immediately before
    copying it (see this module's own docstring on the TOCTOU window
    this narrows), then copies via `_copy_regular_file_no_follow`."""
    for entry in entries:
        src = src_root / entry["path"]
        if src.is_symlink():
            raise BackupError(f"refusing to copy {src}: is a symlink (checked immediately before copy)")
        dst = dst_root / entry["path"]
        _copy_regular_file_no_follow(src, dst)


# --- snapshot-name validation -----------------------------------------

def _validate_snapshot_name(name: str, snapshots_dir: Path) -> Path:
    """Validates `name` as exactly one safe path component and returns
    the resolved, confirmed-contained final publication path. Rejects
    an empty name, '.', '..', any path separator (forward OR back
    slash, regardless of host OS), and an absolute path -- then, as a
    final defense-in-depth check, confirms the constructed path still
    resolves to a direct child of the resolved snapshots directory."""
    if not name:
        raise BackupError("snapshot name must not be empty")
    if name in (".", ".."):
        raise BackupError(f"snapshot name must not be {name!r}")
    if os.path.isabs(name):
        raise BackupError(f"snapshot name must not be an absolute path: {name!r}")
    if "/" in name or "\\" in name:
        raise BackupError(f"snapshot name must be a single path component with no separators: {name!r}")

    resolved_snapshots_dir = snapshots_dir.resolve()
    candidate = (resolved_snapshots_dir / name).resolve()
    if candidate.parent != resolved_snapshots_dir:
        raise BackupError(
            f"snapshot name {name!r} does not resolve to a direct child of {resolved_snapshots_dir}"
        )
    return candidate


def _check_no_nesting(data_dir: Path, snapshots_dir: Path) -> None:
    data_dir = data_dir.resolve()
    snapshots_dir = snapshots_dir.resolve()
    if snapshots_dir == data_dir or snapshots_dir.is_relative_to(data_dir):
        raise BackupError(f"--snapshots-dir ({snapshots_dir}) must not be inside --data-dir ({data_dir})")
    if data_dir.is_relative_to(snapshots_dir):
        raise BackupError(f"--data-dir ({data_dir}) must not be inside --snapshots-dir ({snapshots_dir})")


# --- manifest: build, strict-validate, load -----------------------------

def build_manifest(data_dir: Path) -> dict[str, Any]:
    """Walks data_dir and returns the manifest dict `create_snapshot`
    schema-validates and writes into the staged snapshot."""
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


def _validate_manifest_entry_path(path: str, index: int) -> None:
    if path.startswith("/"):
        raise _schema_error(f"files[{index}].path must be relative, not absolute: {path!r}")
    if "\\" in path:
        raise _schema_error(f"files[{index}].path must not contain a backslash: {path!r}")
    if path != path.strip():
        raise _schema_error(f"files[{index}].path must not have leading/trailing whitespace: {path!r}")
    parts = path.split("/")
    if any(part == "" for part in parts):
        raise _schema_error(f"files[{index}].path must not contain an empty component (e.g. '//'): {path!r}")
    if any(part in (".", "..") for part in parts):
        raise _schema_error(f"files[{index}].path must not contain '.' or '..': {path!r}")
    if path == MANIFEST_FILENAME:
        raise _schema_error(
            f"files[{index}].path must not be {MANIFEST_FILENAME!r} -- that name is reserved for the manifest itself"
        )


def validate_manifest_schema(raw: Any) -> dict[str, Any]:
    """Strictly validates a manifest's shape before anything else in
    this module ever indexes into it. Every malformed-input case (wrong
    top-level type, missing/extra field, wrong field type, a path
    escaping the snapshot, a duplicate path, an out-of-range size, a
    malformed hash, a file_count/total_bytes mismatch) is converted into
    one clean BackupError here -- never left to surface later as a raw
    KeyError/TypeError/IndexError from code that assumed well-formed
    input. Returns a freshly-built, normalized manifest dict (never the
    caller's own `raw` object) containing only the validated data."""
    if not isinstance(raw, dict):
        raise _schema_error(f"top-level manifest must be a JSON object, got {type(raw).__name__}")

    extra_keys = set(raw) - _ALLOWED_TOP_LEVEL_KEYS
    if extra_keys:
        raise _schema_error(f"unexpected top-level field(s): {sorted(extra_keys)}")
    missing_keys = _ALLOWED_TOP_LEVEL_KEYS - set(raw)
    if missing_keys:
        raise _schema_error(f"missing top-level field(s): {sorted(missing_keys)}")

    version = raw["manifest_format_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise _schema_error("manifest_format_version must be an integer")
    if version != MANIFEST_FORMAT_VERSION:
        raise _schema_error(
            f"unsupported manifest_format_version {version!r} (this tool supports {MANIFEST_FORMAT_VERSION})"
        )

    if not isinstance(raw["created_at_utc"], str):
        raise _schema_error("created_at_utc must be a string")
    if not isinstance(raw["source_data_dir"], str):
        raise _schema_error("source_data_dir must be a string")

    files = raw["files"]
    if not isinstance(files, list):
        raise _schema_error("files must be a list")

    seen_paths: set[str] = set()
    validated_files: list[dict[str, Any]] = []
    total_bytes = 0
    for i, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise _schema_error(f"files[{i}] must be an object, got {type(entry).__name__}")
        extra_entry_keys = set(entry) - _ALLOWED_ENTRY_KEYS
        if extra_entry_keys:
            raise _schema_error(f"files[{i}] has unexpected field(s): {sorted(extra_entry_keys)}")
        missing_entry_keys = _ALLOWED_ENTRY_KEYS - set(entry)
        if missing_entry_keys:
            raise _schema_error(f"files[{i}] is missing field(s): {sorted(missing_entry_keys)}")

        path = entry["path"]
        if not isinstance(path, str) or not path:
            raise _schema_error(f"files[{i}].path must be a non-empty string")
        _validate_manifest_entry_path(path, index=i)
        if path in seen_paths:
            raise _schema_error(f"duplicate path in manifest: {path!r}")
        seen_paths.add(path)

        size = entry["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise _schema_error(f"files[{i}].size must be a non-negative integer")

        sha256 = entry["sha256"]
        if not isinstance(sha256, str) or not _SHA256_RE.match(sha256):
            raise _schema_error(f"files[{i}].sha256 must be a 64-character lowercase hex string")

        validated_files.append({"path": path, "size": size, "sha256": sha256})
        total_bytes += size

    file_count = raw["file_count"]
    if not isinstance(file_count, int) or isinstance(file_count, bool):
        raise _schema_error("file_count must be an integer")
    if file_count != len(validated_files):
        raise _schema_error(
            f"file_count {file_count} does not match the actual number of entries {len(validated_files)}"
        )

    manifest_total_bytes = raw["total_bytes"]
    if not isinstance(manifest_total_bytes, int) or isinstance(manifest_total_bytes, bool):
        raise _schema_error("total_bytes must be an integer")
    if manifest_total_bytes != total_bytes:
        raise _schema_error(
            f"total_bytes {manifest_total_bytes} does not match the sum of entry sizes {total_bytes}"
        )

    return {
        "manifest_format_version": version,
        "created_at_utc": raw["created_at_utc"],
        "source_data_dir": raw["source_data_dir"],
        "file_count": file_count,
        "total_bytes": manifest_total_bytes,
        "files": validated_files,
    }


def _load_manifest_dict(snapshot_dir: Path) -> dict[str, Any]:
    """Reads manifest.json from snapshot_dir EXACTLY once, refuses if it
    is anything other than a regular, non-symlink file, and strictly
    validates its schema before ever returning it. Every caller in this
    module that needs the manifest's content goes through this function
    (or a value it already produced) -- never a second, independent read
    of the file."""
    manifest_path = snapshot_dir / MANIFEST_FILENAME
    if manifest_path.is_symlink():
        raise BackupError(f"{manifest_path} is a symlink -- {MANIFEST_FILENAME} must be a regular file")
    if not manifest_path.is_file():
        raise BackupError(f"no {MANIFEST_FILENAME} found in {snapshot_dir} -- not a valid snapshot")
    try:
        raw = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise BackupError(f"manifest at {manifest_path} is not valid JSON: {exc}") from exc
    return validate_manifest_schema(raw)


# --- diff / verify ------------------------------------------------------

def _diff_against_manifest(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """The one comparison routine every verification path in this module
    builds on: `directory` must exactly match `manifest`, nothing more
    (extra), nothing less (missing), nothing different (corrupted). Every
    caller in this module now always wants this exact check -- PR3.3a's
    conservative "destination must not already exist" restore contract
    removed the one caller that previously wanted a laxer, extras-
    tolerant comparison (a --force restore into a non-empty, partially
    unrelated destination), so that parameter was removed rather than
    kept unused."""
    manifest_entries = {e["path"]: e for e in manifest.get("files", [])}
    on_disk = _iter_regular_files(directory, ignore_top_level_names=frozenset({MANIFEST_FILENAME}))
    on_disk_by_rel = {_relative_posix(p, directory): p for p in on_disk}

    missing = sorted(set(manifest_entries) - set(on_disk_by_rel))
    extra = sorted(set(on_disk_by_rel) - set(manifest_entries))

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


def load_and_verify_snapshot(snapshot_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Loads and schema-validates manifest.json EXACTLY ONCE, diffs the
    snapshot directory against it, and returns `(manifest, report)`.
    `restore_snapshot` uses this (never `verify_snapshot`, and never its
    own separate read of manifest.json) so the exact manifest object it
    verified is the same object it later copies from -- nothing on disk
    can be swapped in between."""
    manifest = _load_manifest_dict(snapshot_dir)
    report = _diff_against_manifest(snapshot_dir, manifest)
    return manifest, report


def verify_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    """CLI/external entry point -- verifies a snapshot and returns only
    the bounded diff-style report (never the manifest's own content, to
    keep this safe to print directly regardless of snapshot size)."""
    _manifest, report = load_and_verify_snapshot(snapshot_dir)
    return report


# --- create ---------------------------------------------------------------

def create_snapshot(data_dir: Path, snapshots_dir: Path, name: str | None = None) -> Path:
    """Builds, self-verifies, and publishes one new snapshot. Returns the
    final, published snapshot directory. Raises BackupError (staging
    directory always cleaned up first) on any failure -- see this
    module's own docstring for what "cleaned up" does and does not
    guarantee under an uncatchable failure.

    Caller (main()) is responsible for enforcing --confirm-quiescent --
    this function does not re-check it, so it stays trivially testable
    without a CLI harness."""
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise BackupError(f"--data-dir does not exist or is not a directory: {data_dir}")

    _check_no_nesting(data_dir, snapshots_dir)
    snapshots_dir = snapshots_dir.resolve()
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # `name if name is not None else default`, NOT `name or default` --
    # the latter would silently substitute the default for an explicitly
    # passed empty string (Python truthiness), masking exactly the
    # "empty name" case _validate_snapshot_name below exists to reject.
    snapshot_name = name if name is not None else time.strftime("snapshot-%Y%m%dT%H%M%SZ", time.gmtime())
    final_dir = _validate_snapshot_name(snapshot_name, snapshots_dir)
    if final_dir.exists():
        raise BackupError(f"a snapshot named {snapshot_name!r} already exists at {final_dir} -- choose a different --name")

    staging_dir = Path(tempfile.mkdtemp(prefix=f".staging-{snapshot_name}-", dir=snapshots_dir))
    try:
        manifest = validate_manifest_schema(build_manifest(data_dir))
        _copy_manifest_entries(manifest["files"], data_dir, staging_dir)

        (staging_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        # Publish only after a full self-verification of the STAGED
        # copy -- reads every staged file back and re-hashes it,
        # independent of whatever the copy loop above believed it wrote.
        result = _diff_against_manifest(staging_dir, manifest)
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


# --- restore --------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _walk_up(path: Path):
    current = path
    while True:
        yield current
        if current == current.parent:
            return
        current = current.parent


def _validate_restore_destination(snapshot_dir: Path, dest_dir_raw: Path) -> Path:
    """`snapshot_dir` must already be resolved. `dest_dir_raw` must be
    ABSOLUTE but deliberately NOT resolved yet -- every symlink check
    below runs against the literal, unresolved path components FIRST,
    specifically because `Path.resolve()` silently follows every
    symlink in a path; resolving before checking would make it
    impossible to ever detect the exact thing being checked for. Only
    once no symlink is found anywhere at or above the destination is it
    safe to resolve `dest_dir_raw` and use the result for every
    subsequent identity/existence check (returned, for the caller to use
    for the rest of the restore).

    Named checks (repository root, real project data/ directory, the
    snapshot itself, a destination inside or containing the snapshot)
    run BEFORE the generic "already exists" check specifically so a
    known-dangerous target gets its own clear, specific error message
    rather than being lumped in with an ordinary already-occupied path."""
    if dest_dir_raw.is_symlink():
        raise BackupError(f"refusing to restore: destination {dest_dir_raw} is a symlink")

    for ancestor in _walk_up(dest_dir_raw.parent):
        if ancestor.is_symlink():
            raise BackupError(f"refusing to restore: an ancestor directory {ancestor} of the destination is a symlink")

    dest_dir = dest_dir_raw.resolve()

    project_root = _project_root()
    real_data_dir = project_root / "data"
    if dest_dir == project_root:
        raise BackupError(f"refusing to restore into the repository root ({dest_dir})")
    if dest_dir == real_data_dir:
        raise BackupError(f"refusing to restore into the real project data directory ({dest_dir})")
    if dest_dir == snapshot_dir:
        raise BackupError(f"refusing to restore into the snapshot directory itself ({dest_dir})")
    if dest_dir.is_relative_to(snapshot_dir):
        raise BackupError(f"refusing to restore: destination {dest_dir} is inside the snapshot directory {snapshot_dir}")
    if snapshot_dir.is_relative_to(dest_dir):
        raise BackupError(
            f"refusing to restore: the snapshot directory {snapshot_dir} is inside the destination {dest_dir}"
        )

    if dest_dir.exists():
        raise BackupError(
            f"refusing to restore: destination {dest_dir} already exists (as a file or "
            "directory, empty or not) -- restore only ever creates a brand-new destination."
        )

    return dest_dir


def restore_snapshot(snapshot_dir: Path, dest_dir: Path) -> None:
    """Transactional restore into a brand-new destination only. Stages
    the entire restored tree under a temp directory created inside
    dest_dir's own resolved PARENT, fully verifies that staged copy
    against the SAME in-memory manifest already used to verify the
    source snapshot (via load_and_verify_snapshot -- never a second,
    independent read of manifest.json), and only then publishes with one
    same-filesystem os.rename. Any caught failure removes the staging
    directory and leaves the requested dest_dir absent -- see this
    module's own docstring for the honest limit on that guarantee under
    an uncatchable failure (SIGKILL, power loss, a failing cleanup
    itself)."""
    snapshot_dir = snapshot_dir.resolve()
    # Deliberately Path.absolute(), not Path.resolve() -- see
    # _validate_restore_destination's own docstring for why resolving
    # before the symlink checks would defeat them.
    dest_dir = _validate_restore_destination(snapshot_dir, Path(dest_dir).absolute())

    manifest, pre_result = load_and_verify_snapshot(snapshot_dir)
    if not pre_result["ok"]:
        raise BackupError(
            "refusing to restore from a snapshot that fails its own verification: "
            f"missing={pre_result['missing']} extra={pre_result['extra']} corrupted={pre_result['corrupted']}"
        )

    dest_parent = dest_dir.parent
    dest_parent.mkdir(parents=True, exist_ok=True)

    staging_dir = Path(tempfile.mkdtemp(prefix=f".restore-staging-{dest_dir.name}-", dir=dest_parent))
    try:
        _copy_manifest_entries(manifest["files"], snapshot_dir, staging_dir)

        result = _diff_against_manifest(staging_dir, manifest)
        if not result["ok"]:
            raise BackupError(
                "staged restore failed verification before publish -- destination left "
                f"absent: missing={result['missing']} extra={result['extra']} corrupted={result['corrupted']}"
            )

        os.rename(staging_dir, dest_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


# --- CLI --------------------------------------------------------------

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
    create_parser.add_argument(
        "--name", default=None,
        help="Snapshot directory name -- must be a single path component (default: a UTC timestamp).",
    )
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

    restore_parser = subparsers.add_parser(
        "restore", help="Restore a snapshot into a brand-new destination (must not already exist).",
    )
    restore_parser.add_argument("snapshot_dir", type=Path)
    restore_parser.add_argument("dest_dir", type=Path)

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
            restore_snapshot(args.snapshot_dir, args.dest_dir)
            print(f"Restored into {args.dest_dir.resolve()}")
            return 0

        parser.error(f"unknown command {args.command!r}")
        return 2
    except BackupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
