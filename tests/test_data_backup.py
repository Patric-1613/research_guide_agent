"""PR3.3/PR3.3a: tests for scripts/data_backup.py -- the platform-neutral
create/verify/restore backup workflow. Every fixture here builds a small,
synthetic temp data directory; nothing in this file ever mutates the
real, gitignored data/ directory. A few destination-safety tests below
DO pass the real repository root / real data/ path as a `dest_dir`
argument -- this is safe because `_validate_restore_destination` raises
BEFORE any filesystem mutation, and the only filesystem operations
involved (`.exists()`/`.is_symlink()`/`==` on already-resolved paths) are
plain path-metadata checks, never a read of any file's content.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import scripts.data_backup as data_backup
from scripts.data_backup import (
    BackupError,
    build_manifest,
    create_snapshot,
    restore_snapshot,
    validate_manifest_schema,
    verify_snapshot,
)


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _fingerprint(directory: Path) -> dict[str, bytes]:
    """A simple {relative_path: content} map -- used to prove a
    directory's content is byte-identical before/after an operation,
    independent of this module's own hashing code."""
    result = {}
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            p = Path(dirpath) / name
            result[str(p.relative_to(directory))] = p.read_bytes()
    return result


def _make_realistic_data_dir(root: Path) -> Path:
    """A synthetic data/ look-alike: a SQLite-shaped file with WAL/SHM
    sidecars, a nested Chroma-collection-shaped directory tree, a cache
    file, and a genuinely empty file -- covering every file SHAPE the
    real data/ directory has, with fake bytes."""
    data_dir = root / "data"
    _write(data_dir / "history.sqlite", b"fake-sqlite-history-bytes")
    _write(data_dir / "usage_telemetry.sqlite", b"fake-sqlite-telemetry-bytes")
    _write(data_dir / "qa_checkpoints.sqlite", b"fake-sqlite-checkpoints-bytes")
    _write(data_dir / "qa_checkpoints.sqlite-wal", b"fake-wal-bytes")
    _write(data_dir / "qa_checkpoints.sqlite-shm", b"fake-shm-bytes")
    _write(data_dir / "chroma_db" / "chroma.sqlite3", b"fake-chroma-sqlite-bytes")
    _write(data_dir / "chroma_db" / "abcd-uuid-collection" / "data_level0.bin", b"\x00\x01\x02binary-vector-data")
    _write(data_dir / "chroma_db" / "abcd-uuid-collection" / "header.bin", b"h")
    _write(data_dir / "chroma_db" / "abcd-uuid-collection" / "empty.bin", b"")  # genuinely empty file
    _write(data_dir / "cache" / "embeddings.sqlite", b"fake-embedding-cache-bytes")
    return data_dir


def _minimal_valid_manifest(files: list[dict] | None = None) -> dict:
    if files is None:
        files = [{"path": "a.txt", "size": 1, "sha256": "a" * 64}]
    return {
        "manifest_format_version": data_backup.MANIFEST_FORMAT_VERSION,
        "created_at_utc": "2026-01-01T00:00:00Z",
        "source_data_dir": "/tmp/whatever",
        "file_count": len(files),
        "total_bytes": sum(f["size"] for f in files if isinstance(f, dict) and isinstance(f.get("size"), int)),
        "files": files,
    }


# --- build_manifest / create_snapshot: coverage of every file shape ---

def test_manifest_covers_sqlite_wal_shm_nested_chroma_and_empty_files(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path)

    manifest = build_manifest(data_dir)

    paths = {e["path"] for e in manifest["files"]}
    assert paths == {
        "history.sqlite",
        "usage_telemetry.sqlite",
        "qa_checkpoints.sqlite",
        "qa_checkpoints.sqlite-wal",
        "qa_checkpoints.sqlite-shm",
        "chroma_db/chroma.sqlite3",
        "chroma_db/abcd-uuid-collection/data_level0.bin",
        "chroma_db/abcd-uuid-collection/header.bin",
        "chroma_db/abcd-uuid-collection/empty.bin",
        "cache/embeddings.sqlite",
    }
    empty_entry = next(e for e in manifest["files"] if e["path"] == "chroma_db/abcd-uuid-collection/empty.bin")
    assert empty_entry["size"] == 0
    assert empty_entry["sha256"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_create_snapshot_round_trips_every_byte_exactly(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshots_dir = tmp_path / "snapshots"
    before = _fingerprint(data_dir)

    snapshot_dir = create_snapshot(data_dir, snapshots_dir, name="s1")

    result = verify_snapshot(snapshot_dir)
    assert result["ok"] is True
    assert result["missing"] == []
    assert result["extra"] == []
    assert result["corrupted"] == []
    assert result["file_count"] == len(before)
    assert (snapshot_dir / data_backup.MANIFEST_FILENAME).is_file()
    assert _fingerprint(data_dir) == before


def test_source_data_is_byte_identical_after_create(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    before = _fingerprint(data_dir)

    create_snapshot(data_dir, tmp_path / "snapshots")

    assert _fingerprint(data_dir) == before


# --- verify: corruption / missing / extra detection ---

def test_verify_detects_corrupted_file(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")

    (snapshot_dir / "history.sqlite").write_bytes(b"tampered-content")

    result = verify_snapshot(snapshot_dir)

    assert result["ok"] is False
    assert result["corrupted"] == ["history.sqlite: size mismatch (expected 25, found 16)"]
    assert result["missing"] == []
    assert result["extra"] == []


def test_verify_detects_missing_file(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")

    (snapshot_dir / "cache" / "embeddings.sqlite").unlink()

    result = verify_snapshot(snapshot_dir)

    assert result["ok"] is False
    assert result["missing"] == ["cache/embeddings.sqlite"]
    assert result["corrupted"] == []


def test_verify_detects_extra_file(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")

    (snapshot_dir / "chroma_db" / "not-in-manifest.bin").write_bytes(b"surprise")

    result = verify_snapshot(snapshot_dir)

    assert result["ok"] is False
    assert result["extra"] == ["chroma_db/not-in-manifest.bin"]
    assert result["missing"] == []
    assert result["corrupted"] == []


def test_verify_raises_for_a_directory_with_no_manifest(tmp_path):
    not_a_snapshot = tmp_path / "just-a-directory"
    not_a_snapshot.mkdir()
    (not_a_snapshot / "file.txt").write_text("hello")

    with pytest.raises(BackupError, match="not a valid snapshot"):
        verify_snapshot(not_a_snapshot)


# --- symlink escape in the DATA directory (create) ---

def test_create_refuses_a_symlink_that_escapes_the_data_dir(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    outside_secret = tmp_path / "outside-secret.txt"
    outside_secret.write_text("SECRET-OUTSIDE-DATA-DIR")
    (data_dir / "escape-link.sqlite").symlink_to(outside_secret)

    with pytest.raises(BackupError, match="symlink"):
        create_snapshot(data_dir, tmp_path / "snapshots")

    assert not (tmp_path / "snapshots").exists() or list((tmp_path / "snapshots").iterdir()) == []


def test_create_refuses_a_symlinked_directory_too(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "secret.bin").write_bytes(b"SECRET")
    (data_dir / "linked-subdir").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(BackupError, match="symlink"):
        create_snapshot(data_dir, tmp_path / "snapshots")


def test_create_refuses_a_symlink_that_resolves_inside_the_data_dir_too(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    (data_dir / "alias.sqlite").symlink_to(data_dir / "history.sqlite")

    with pytest.raises(BackupError, match="symlink"):
        create_snapshot(data_dir, tmp_path / "snapshots")


# --- manifest.json itself must never be a symlink ---

def _replace_manifest_with_symlink(snapshot_dir: Path, tmp_path: Path) -> None:
    real_manifest = snapshot_dir / data_backup.MANIFEST_FILENAME
    decoy = tmp_path / "decoy-manifest.json"
    decoy.write_bytes(real_manifest.read_bytes())
    real_manifest.unlink()
    real_manifest.symlink_to(decoy)


def test_verify_refuses_a_symlinked_manifest(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    _replace_manifest_with_symlink(snapshot_dir, tmp_path)

    with pytest.raises(BackupError, match="symlink"):
        verify_snapshot(snapshot_dir)


def test_restore_refuses_a_symlinked_manifest(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    _replace_manifest_with_symlink(snapshot_dir, tmp_path)
    dest = tmp_path / "restored"

    with pytest.raises(BackupError, match="symlink"):
        restore_snapshot(snapshot_dir, dest)

    assert not dest.exists()


# --- manifest is read exactly once for restore (no swap possible) ---

def test_restore_reads_manifest_exactly_once(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    dest = tmp_path / "restored"

    real_load = data_backup._load_manifest_dict
    with patch.object(data_backup, "_load_manifest_dict", side_effect=real_load) as spy:
        restore_snapshot(snapshot_dir, dest)

    assert spy.call_count == 1


# --- manifest schema: strict validation, unit-level ---

def test_manifest_schema_accepts_a_well_formed_manifest():
    validated = validate_manifest_schema(_minimal_valid_manifest())
    assert validated["files"][0]["path"] == "a.txt"


def test_manifest_schema_rejects_non_dict_top_level():
    with pytest.raises(BackupError, match="JSON object"):
        validate_manifest_schema(["not", "a", "dict"])


def test_manifest_schema_rejects_missing_top_level_field():
    m = _minimal_valid_manifest()
    del m["created_at_utc"]
    with pytest.raises(BackupError, match="missing top-level field"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_unexpected_top_level_field():
    m = _minimal_valid_manifest()
    m["surprise"] = 1
    with pytest.raises(BackupError, match="unexpected top-level field"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_wrong_type_for_version():
    m = _minimal_valid_manifest()
    m["manifest_format_version"] = "1"
    with pytest.raises(BackupError, match="must be an integer"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_unsupported_format_version():
    m = _minimal_valid_manifest()
    m["manifest_format_version"] = 999
    with pytest.raises(BackupError, match="unsupported manifest_format_version"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_files_not_a_list():
    m = _minimal_valid_manifest()
    m["files"] = "not-a-list"
    with pytest.raises(BackupError, match="files must be a list"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_entry_missing_field():
    m = _minimal_valid_manifest(files=[{"path": "a.txt", "size": 1}])
    with pytest.raises(BackupError, match="missing field"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_entry_extra_field():
    m = _minimal_valid_manifest(files=[{"path": "a.txt", "size": 1, "sha256": "a" * 64, "extra": 1}])
    with pytest.raises(BackupError, match="unexpected field"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_absolute_path():
    m = _minimal_valid_manifest(files=[{"path": "/etc/passwd", "size": 1, "sha256": "a" * 64}])
    with pytest.raises(BackupError, match="absolute"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_traversal_path():
    m = _minimal_valid_manifest(files=[{"path": "../outside.txt", "size": 1, "sha256": "a" * 64}])
    with pytest.raises(BackupError, match=r"'\.' or '\.\.'"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_empty_path_component():
    m = _minimal_valid_manifest(files=[{"path": "a//b", "size": 1, "sha256": "a" * 64}])
    with pytest.raises(BackupError, match="empty component"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_backslash_path():
    m = _minimal_valid_manifest(files=[{"path": "chroma_db\\evil.bin", "size": 1, "sha256": "a" * 64}])
    with pytest.raises(BackupError, match="backslash"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_manifest_json_as_payload_entry():
    m = _minimal_valid_manifest(files=[{"path": "manifest.json", "size": 1, "sha256": "a" * 64}])
    with pytest.raises(BackupError, match="reserved"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_duplicate_paths():
    entry = {"path": "a.txt", "size": 1, "sha256": "a" * 64}
    m = _minimal_valid_manifest(files=[entry, dict(entry)])
    with pytest.raises(BackupError, match="duplicate path"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_negative_size():
    m = _minimal_valid_manifest(files=[{"path": "a.txt", "size": -1, "sha256": "a" * 64}])
    with pytest.raises(BackupError, match="non-negative integer"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_non_integer_size():
    m = _minimal_valid_manifest(files=[{"path": "a.txt", "size": "1", "sha256": "a" * 64}])
    with pytest.raises(BackupError, match="non-negative integer"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_malformed_sha256_too_short():
    m = _minimal_valid_manifest(files=[{"path": "a.txt", "size": 1, "sha256": "abc"}])
    with pytest.raises(BackupError, match="64 lowercase hexadecimal"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_sha256_too_long():
    m = _minimal_valid_manifest(files=[{"path": "a.txt", "size": 1, "sha256": "a" * 65}])
    with pytest.raises(BackupError, match="64 lowercase hexadecimal"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_uppercase_sha256():
    m = _minimal_valid_manifest(files=[{"path": "a.txt", "size": 1, "sha256": "A" * 64}])
    with pytest.raises(BackupError, match="64 lowercase hexadecimal"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_non_hexadecimal_sha256():
    m = _minimal_valid_manifest(files=[{"path": "a.txt", "size": 1, "sha256": "g" * 64}])
    with pytest.raises(BackupError, match="64 lowercase hexadecimal"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_sha256_with_a_trailing_newline():
    """PR3.3b: the original `^[0-9a-f]{64}$` pattern used with
    re.match() would WRONGLY accept this -- Python's '$' matches just
    before a single trailing newline. fullmatch() must not."""
    m = _minimal_valid_manifest(files=[{"path": "a.txt", "size": 1, "sha256": "a" * 64 + "\n"}])
    with pytest.raises(BackupError, match="64 lowercase hexadecimal"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_sha256_with_leading_or_trailing_whitespace():
    m = _minimal_valid_manifest(files=[{"path": "a.txt", "size": 1, "sha256": " " + "a" * 63}])
    with pytest.raises(BackupError, match="64 lowercase hexadecimal"):
        validate_manifest_schema(m)
    m2 = _minimal_valid_manifest(files=[{"path": "a.txt", "size": 1, "sha256": "a" * 63 + " "}])
    with pytest.raises(BackupError, match="64 lowercase hexadecimal"):
        validate_manifest_schema(m2)


def test_manifest_schema_rejects_file_count_mismatch():
    m = _minimal_valid_manifest()
    m["file_count"] = 99
    with pytest.raises(BackupError, match="file_count"):
        validate_manifest_schema(m)


def test_manifest_schema_rejects_total_bytes_mismatch():
    m = _minimal_valid_manifest()
    m["total_bytes"] = 99999
    with pytest.raises(BackupError, match="total_bytes"):
        validate_manifest_schema(m)


def test_verify_converts_malformed_on_disk_manifest_into_backup_error_not_a_raw_exception(tmp_path):
    snapshot_dir = tmp_path / "bad-snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / data_backup.MANIFEST_FILENAME).write_text(json.dumps({"files": "not-a-list"}))

    with pytest.raises(BackupError):
        verify_snapshot(snapshot_dir)


# --- restore: conservative destination contract ---

def test_restore_refuses_an_existing_empty_destination(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    dest = tmp_path / "restored"
    dest.mkdir()

    with pytest.raises(BackupError, match="already exists"):
        restore_snapshot(snapshot_dir, dest)

    assert list(dest.iterdir()) == []


def test_restore_refuses_an_existing_non_empty_destination(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    dest = tmp_path / "restored"
    dest.mkdir()
    (dest / "pre-existing.txt").write_text("do not touch")

    with pytest.raises(BackupError, match="already exists"):
        restore_snapshot(snapshot_dir, dest)

    assert (dest / "pre-existing.txt").read_text() == "do not touch"


def test_restore_refuses_a_symlinked_destination_root(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    real_target = tmp_path / "real-target-dir"
    real_target.mkdir()
    dest = tmp_path / "restored-symlink"
    dest.symlink_to(real_target, target_is_directory=True)

    with pytest.raises(BackupError, match="symlink"):
        restore_snapshot(snapshot_dir, dest)


def test_restore_refuses_a_destination_with_a_symlinked_ancestor(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    dest = linked_parent / "nested" / "restored"

    with pytest.raises(BackupError, match="symlink"):
        restore_snapshot(snapshot_dir, dest)


def test_restore_refuses_the_real_repository_root(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    repo_root = data_backup._project_root()

    with pytest.raises(BackupError, match="repository root"):
        restore_snapshot(snapshot_dir, repo_root)


def test_restore_refuses_the_real_project_data_directory(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    real_data_dir = data_backup._project_root() / "data"

    with pytest.raises(BackupError, match="real project data directory"):
        restore_snapshot(snapshot_dir, real_data_dir)


# --- PR3.3b Finding 1: the data/ protection must cover every descendant
# (resolved-path containment), not just an exact match -- and must NOT
# reject a merely similarly-named sibling like data-copy. These use a
# monkeypatched fake project root so they never touch the real repo,
# not even a read of an existing path, per this checkpoint's own "do
# not inspect or operate on real data/" boundary. ---

def test_restore_refuses_a_shallow_descendant_of_the_real_data_directory(tmp_path, monkeypatch):
    fake_project_root = tmp_path / "fake-project"
    fake_project_root.mkdir()
    monkeypatch.setattr(data_backup, "_project_root", lambda: fake_project_root)
    data_dir = _make_realistic_data_dir(tmp_path / "source-data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    descendant = fake_project_root / "data" / "restore-drill"

    with pytest.raises(BackupError, match="real project data directory"):
        restore_snapshot(snapshot_dir, descendant)

    assert not descendant.exists()


def test_restore_refuses_a_deep_descendant_of_the_real_data_directory(tmp_path, monkeypatch):
    fake_project_root = tmp_path / "fake-project"
    fake_project_root.mkdir()
    monkeypatch.setattr(data_backup, "_project_root", lambda: fake_project_root)
    data_dir = _make_realistic_data_dir(tmp_path / "source-data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    deep_descendant = fake_project_root / "data" / "a" / "b" / "c" / "restore-drill"

    with pytest.raises(BackupError, match="real project data directory"):
        restore_snapshot(snapshot_dir, deep_descendant)

    assert not deep_descendant.exists()


def test_restore_allows_a_similarly_named_sibling_of_the_data_directory(tmp_path, monkeypatch):
    """`data-copy` is NOT inside `data/` -- its string representation
    happens to start with "data", so a naive string-prefix check would
    wrongly reject it, but Path.is_relative_to() (resolved-path
    containment) correctly allows it. This is the one test in this
    group that actually completes a restore -- into a sibling of the
    FAKE (monkeypatched) project root's data/ directory, never the real
    one."""
    fake_project_root = tmp_path / "fake-project"
    fake_project_root.mkdir()
    monkeypatch.setattr(data_backup, "_project_root", lambda: fake_project_root)
    data_dir = _make_realistic_data_dir(tmp_path / "source-data")
    before = _fingerprint(data_dir)
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    sibling_dest = fake_project_root / "data-copy"

    restore_snapshot(snapshot_dir, sibling_dest)

    assert _fingerprint(sibling_dest) == before


def test_restore_refuses_the_snapshot_directory_itself(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")

    with pytest.raises(BackupError, match="snapshot directory itself"):
        restore_snapshot(snapshot_dir, snapshot_dir)


def test_restore_refuses_a_destination_inside_the_snapshot(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    nested_dest = snapshot_dir / "nested" / "new-path"

    with pytest.raises(BackupError, match="inside the snapshot directory"):
        restore_snapshot(snapshot_dir, nested_dest)


def test_restore_refuses_a_destination_that_already_contains_the_snapshot(tmp_path):
    """dest_dir here (tmp_path) already contains snapshot_dir, and --
    necessarily, since snapshot_dir is a real directory on disk --
    dest_dir must itself already exist too, so this is caught by the
    general "destination must not exist" rule. Confirms the overall
    safety property holds regardless of which specific check fires
    first."""
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")

    with pytest.raises(BackupError):
        restore_snapshot(snapshot_dir, tmp_path)


# --- restore: happy path, interrupted copy, failed post-copy verification ---

def test_restore_into_a_new_destination_reproduces_every_file_exactly(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    before = _fingerprint(data_dir)
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    before_snapshot_fp = _fingerprint(snapshot_dir)
    dest = tmp_path / "restored"

    restore_snapshot(snapshot_dir, dest)

    assert _fingerprint(dest) == before
    # The restore never mutates its own source snapshot.
    assert _fingerprint(snapshot_dir) == before_snapshot_fp


def test_restore_refuses_a_snapshot_that_fails_its_own_verification(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    (snapshot_dir / "history.sqlite").write_bytes(b"corrupted-after-publish")

    with pytest.raises(BackupError, match="fails its own verification"):
        restore_snapshot(snapshot_dir, tmp_path / "restored")

    assert not (tmp_path / "restored").exists()


# --- PR3.3b Finding 3: the immediately-before-publish re-check ---

def test_restore_fails_and_preserves_a_destination_created_immediately_before_publish(tmp_path):
    """Simulates another process winning the race in the exact window
    between this restore's own staged self-verification succeeding and
    its immediately-before-publish re-check running -- that re-check
    (PR3.3b) must catch the now-existing destination, refuse to
    publish, leave the externally-created destination completely
    untouched, and still clean up this restore's own staging
    directory."""
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    dest = tmp_path / "restored"

    real_diff = data_backup._diff_against_manifest

    def racing_diff(directory, manifest):
        result = real_diff(directory, manifest)
        if ".restore-staging-" in directory.name:
            # This restore's own staged copy just passed verification --
            # simulate something else creating dest_dir right now, before
            # the immediately-before-publish re-check gets to run.
            dest.mkdir(parents=True)
            (dest / "created-by-someone-else.txt").write_text("not this restore's doing")
        return result

    with patch.object(data_backup, "_diff_against_manifest", side_effect=racing_diff):
        with pytest.raises(BackupError, match="created by something else"):
            restore_snapshot(snapshot_dir, dest)

    # The externally-created destination is left completely untouched --
    # not replaced, not merged into.
    contents = {p.name for p in dest.iterdir()}
    assert contents == {"created-by-someone-else.txt"}
    assert (dest / "created-by-someone-else.txt").read_text() == "not this restore's doing"
    assert not (dest / "history.sqlite").exists()

    # This restore's own staging directory was still cleaned up.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".restore-staging-")]
    assert leftovers == []


def test_interrupted_restore_leaves_destination_absent(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    dest = tmp_path / "restored"

    call_count = {"n": 0}
    real_copy = data_backup._copy_regular_file_no_follow

    def flaky_copy(src, dst):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise OSError("simulated disk failure mid-restore-copy")
        return real_copy(src, dst)

    with patch.object(data_backup, "_copy_regular_file_no_follow", side_effect=flaky_copy):
        with pytest.raises(OSError, match="simulated disk failure"):
            restore_snapshot(snapshot_dir, dest)

    assert not dest.exists()
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".restore-staging-")]
    assert leftovers == []


def test_restore_with_silently_corrupted_copy_leaves_destination_absent(tmp_path):
    """The copy loop itself reports success (no exception), but the
    bytes on disk are wrong -- restore's own post-copy self-verification
    against the validated in-memory manifest must still catch this."""
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    before_snapshot_fp = _fingerprint(snapshot_dir)
    dest = tmp_path / "restored"

    real_copy = data_backup._copy_regular_file_no_follow

    def corrupting_copy(src, dst):
        result = real_copy(src, dst)
        if Path(src).name == "history.sqlite":
            Path(dst).write_bytes(b"silently-wrong-bytes")
        return result

    with patch.object(data_backup, "_copy_regular_file_no_follow", side_effect=corrupting_copy):
        with pytest.raises(BackupError, match="failed verification"):
            restore_snapshot(snapshot_dir, dest)

    assert not dest.exists()
    assert _fingerprint(snapshot_dir) == before_snapshot_fp


# --- interrupted / failed-self-verification snapshot CREATE (staging cleanup) ---

def test_interrupted_create_leaves_no_staging_or_published_directory(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshots_dir = tmp_path / "snapshots"

    call_count = {"n": 0}
    real_copy = data_backup._copy_regular_file_no_follow

    def flaky_copy(src, dst):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise OSError("simulated disk failure mid-copy")
        return real_copy(src, dst)

    with patch.object(data_backup, "_copy_regular_file_no_follow", side_effect=flaky_copy):
        with pytest.raises(OSError, match="simulated disk failure"):
            create_snapshot(data_dir, snapshots_dir, name="interrupted")

    if snapshots_dir.exists():
        assert list(snapshots_dir.iterdir()) == []
    assert not (snapshots_dir / "interrupted").exists()


def test_failed_self_verification_before_publish_leaves_no_published_snapshot(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshots_dir = tmp_path / "snapshots"
    real_copy = data_backup._copy_regular_file_no_follow

    def corrupting_copy(src, dst):
        result = real_copy(src, dst)
        if Path(src).name == "history.sqlite":
            Path(dst).write_bytes(b"silently-wrong-bytes")
        return result

    with patch.object(data_backup, "_copy_regular_file_no_follow", side_effect=corrupting_copy):
        with pytest.raises(BackupError, match="failed self-verification"):
            create_snapshot(data_dir, snapshots_dir, name="bad")

    assert not (snapshots_dir / "bad").exists()
    assert list(snapshots_dir.iterdir()) == []


# --- --data-dir / --snapshots-dir nesting guard ---

def test_create_refuses_when_snapshots_dir_is_inside_data_dir(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")

    with pytest.raises(BackupError, match="must not be inside"):
        create_snapshot(data_dir, data_dir / "snapshots")


def test_create_refuses_when_data_dir_is_inside_snapshots_dir(tmp_path):
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    data_dir = _make_realistic_data_dir(snapshots_dir / "data")

    with pytest.raises(BackupError, match="must not be inside"):
        create_snapshot(data_dir, snapshots_dir)


# --- snapshot name validation ---

def test_create_rejects_an_absolute_snapshot_name(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")

    with pytest.raises(BackupError, match="absolute"):
        create_snapshot(data_dir, tmp_path / "snapshots", name="/etc/passwd")


def test_create_rejects_a_snapshot_name_with_a_separator(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")

    with pytest.raises(BackupError, match="single path component"):
        create_snapshot(data_dir, tmp_path / "snapshots", name="nested/escape")


def test_create_rejects_a_traversal_snapshot_name(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")

    with pytest.raises(BackupError, match="must not be"):
        create_snapshot(data_dir, tmp_path / "snapshots", name="..")


def test_create_rejects_an_empty_snapshot_name(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")

    with pytest.raises(BackupError, match="must not be empty"):
        create_snapshot(data_dir, tmp_path / "snapshots", name="")


# --- CLI: --confirm-quiescent is required; end-to-end round trip ---

def test_cli_create_requires_confirm_quiescent_flag(tmp_path, capsys):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshots_dir = tmp_path / "snapshots"

    exit_code = data_backup.main([
        "create", "--data-dir", str(data_dir), "--snapshots-dir", str(snapshots_dir),
    ])

    assert exit_code == 2
    assert "confirm-quiescent" in capsys.readouterr().err
    assert not snapshots_dir.exists() or list(snapshots_dir.iterdir()) == []


def test_cli_create_with_confirm_quiescent_publishes_a_verifiable_snapshot(tmp_path, capsys):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshots_dir = tmp_path / "snapshots"

    exit_code = data_backup.main([
        "create", "--data-dir", str(data_dir), "--snapshots-dir", str(snapshots_dir),
        "--name", "cli-snap", "--confirm-quiescent",
    ])
    assert exit_code == 0
    assert "cli-snap" in capsys.readouterr().out

    capsys.readouterr()
    verify_exit = data_backup.main(["verify", str(snapshots_dir / "cli-snap")])
    report = json.loads(capsys.readouterr().out)

    assert verify_exit == 0
    assert report["ok"] is True
    assert report["missing"] == []
    assert report["extra"] == []
    assert report["corrupted"] == []


def test_cli_verify_reports_non_zero_exit_on_a_bad_snapshot(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshots_dir = tmp_path / "snapshots"
    data_backup.main([
        "create", "--data-dir", str(data_dir), "--snapshots-dir", str(snapshots_dir),
        "--name", "s1", "--confirm-quiescent",
    ])
    (snapshots_dir / "s1" / "history.sqlite").unlink()

    exit_code = data_backup.main(["verify", str(snapshots_dir / "s1")])

    assert exit_code == 1


def test_cli_restore_end_to_end(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    before = _fingerprint(data_dir)
    snapshots_dir = tmp_path / "snapshots"
    dest = tmp_path / "restored"
    data_backup.main([
        "create", "--data-dir", str(data_dir), "--snapshots-dir", str(snapshots_dir),
        "--name", "s1", "--confirm-quiescent",
    ])

    exit_code = data_backup.main(["restore", str(snapshots_dir / "s1"), str(dest)])

    assert exit_code == 0
    assert _fingerprint(dest) == before


def test_cli_restore_has_no_force_flag(tmp_path, capsys):
    """--force was removed entirely (PR3.3a) -- confirms the CLI parser
    rejects it rather than silently accepting a no-op flag."""
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshots_dir = tmp_path / "snapshots"
    data_backup.main([
        "create", "--data-dir", str(data_dir), "--snapshots-dir", str(snapshots_dir),
        "--name", "s1", "--confirm-quiescent",
    ])

    with pytest.raises(SystemExit):
        data_backup.main(["restore", str(snapshots_dir / "s1"), str(tmp_path / "restored"), "--force"])
