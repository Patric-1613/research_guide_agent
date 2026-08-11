"""M2.1b: shared helper for proving a test run never mutated the real
data/usage_telemetry.sqlite (or its -wal/-shm sidecars).

Deliberately NOT a fixture and NOT registered in conftest.py -- every
telemetry/admission/lease test file already captures its own "before"
snapshot at import time (before pytest runs any test), the same
narrowly-scoped, per-file convention this project already used for the
plain `_REAL_USAGE_DB_PATH = telemetry.USAGE_DB_PATH` capture. This
module only supplies the comparison primitive; each file still owns its
own before/after test.

A legitimate local `usage_telemetry.sqlite` (created by real dev-server
use) is normal, valid state -- it must not cause a failure just for
existing. Nonexistence is one possible fingerprint, not the only
correct one: `fingerprint_usage_db()` is called once before any test in
the file runs and once again after, and the two are compared for
equality. Whether the file existed, didn't exist, or already contained
real rows, an unmutated file compares equal to itself either way.

Pure filesystem reads only -- never opens a sqlite3 connection. Even a
read-only connection attempt can, depending on journal mode and
whether anything else touches the file mid-test, leave a stray
`-journal`/`-wal` file behind; the whole point here is to prove nothing
touched the file, so the proof itself must not touch it either.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _FileFingerprint:
    exists: bool
    size: int | None
    mtime_ns: int | None
    sha256: str | None


@dataclass(frozen=True)
class UsageDbFingerprint:
    """Snapshot of the main DB file plus its WAL/SHM sidecars. SQLite's
    WAL mode can leave uncommitted state in either sidecar depending on
    checkpoint timing, so a real "was this touched" proof must cover
    all three paths, not just the main .sqlite file."""

    main: _FileFingerprint
    wal: _FileFingerprint
    shm: _FileFingerprint


def _fingerprint_one(path: Path) -> _FileFingerprint:
    if not path.exists():
        return _FileFingerprint(exists=False, size=None, mtime_ns=None, sha256=None)
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return _FileFingerprint(exists=True, size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest)


def fingerprint_usage_db(db_path: Path) -> UsageDbFingerprint:
    return UsageDbFingerprint(
        main=_fingerprint_one(db_path),
        wal=_fingerprint_one(db_path.with_name(db_path.name + "-wal")),
        shm=_fingerprint_one(db_path.with_name(db_path.name + "-shm")),
    )
