"""PR3.3: tests for scripts/data_backup.py -- the platform-neutral
create/verify/restore backup workflow. Every fixture here builds a small,
synthetic temp data directory; nothing in this file ever touches the
real, gitignored data/ directory or makes a network/provider call.
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
    verify_snapshot,
)


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _fingerprint(directory: Path) -> dict[str, bytes]:
    """A simple {relative_path: content} map -- used to prove a
    directory's content is byte-identical before/after an operation,
    independent of this module's own hashing code (so a bug in
    data_backup.py's own SHA-256 use couldn't mask a real content
    change)."""
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
    real data/ directory has, with fake bytes (never touching real
    SQLite/Chroma content)."""
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
    # SHA-256 of zero bytes -- a well-known constant, confirms empty files aren't skipped or mishandled.
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

    # The manifest itself lives alongside the data files but is never
    # counted as one of them.
    assert (snapshot_dir / data_backup.MANIFEST_FILENAME).is_file()

    # Source data is completely untouched by taking a backup of it.
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


# --- symlink escape ---

def test_create_refuses_a_symlink_that_escapes_the_data_dir(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    outside_secret = tmp_path / "outside-secret.txt"
    outside_secret.write_text("SECRET-OUTSIDE-DATA-DIR")
    (data_dir / "escape-link.sqlite").symlink_to(outside_secret)

    with pytest.raises(BackupError, match="symlink"):
        create_snapshot(data_dir, tmp_path / "snapshots")

    # No snapshot was ever published from a refused attempt.
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
    """Symlinks are refused unconditionally -- even a same-tree symlink
    that would resolve back inside data_dir -- see the module's own
    docstring for why this is deliberately blunt rather than case-by-case."""
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    (data_dir / "alias.sqlite").symlink_to(data_dir / "history.sqlite")

    with pytest.raises(BackupError, match="symlink"):
        create_snapshot(data_dir, tmp_path / "snapshots")


# --- restore: non-empty destination, force, full content round-trip ---

def test_restore_refuses_a_non_empty_destination_without_force(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    dest = tmp_path / "restored"
    dest.mkdir()
    (dest / "pre-existing.txt").write_text("do not clobber me")

    with pytest.raises(BackupError, match="not empty"):
        restore_snapshot(snapshot_dir, dest, force=False)

    # Refused restore must not have written anything into dest.
    assert [p.name for p in dest.iterdir()] == ["pre-existing.txt"]


def test_restore_into_a_new_destination_reproduces_every_file_exactly(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    before = _fingerprint(data_dir)
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    dest = tmp_path / "restored"

    restore_snapshot(snapshot_dir, dest, force=False)

    assert _fingerprint(dest) == before


def test_restore_into_an_empty_existing_destination_succeeds_without_force(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    before = _fingerprint(data_dir)
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    dest = tmp_path / "restored"
    dest.mkdir()  # exists, but empty

    restore_snapshot(snapshot_dir, dest, force=False)

    assert _fingerprint(dest) == before


def test_restore_with_force_into_non_empty_destination_does_not_delete_unrelated_files(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    before = _fingerprint(data_dir)
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    dest = tmp_path / "restored"
    dest.mkdir()
    (dest / "unrelated.txt").write_text("kept")

    restore_snapshot(snapshot_dir, dest, force=True)

    assert (dest / "unrelated.txt").read_text() == "kept"
    for rel, content in before.items():
        assert (dest / rel).read_bytes() == content


def test_restore_refuses_a_snapshot_that_fails_its_own_verification(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshot_dir = create_snapshot(data_dir, tmp_path / "snapshots")
    (snapshot_dir / "history.sqlite").write_bytes(b"corrupted-after-publish")

    with pytest.raises(BackupError, match="fails its own verification"):
        restore_snapshot(snapshot_dir, tmp_path / "restored", force=False)

    assert not (tmp_path / "restored").exists()


# --- interrupted snapshot cleanup ---

def test_interrupted_create_leaves_no_staging_or_published_directory(tmp_path):
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshots_dir = tmp_path / "snapshots"

    call_count = {"n": 0}
    real_copy2 = data_backup.shutil.copy2

    def flaky_copy2(src, dst, *a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise OSError("simulated disk failure mid-copy")
        return real_copy2(src, dst, *a, **kw)

    with patch.object(data_backup.shutil, "copy2", side_effect=flaky_copy2):
        with pytest.raises(OSError, match="simulated disk failure"):
            create_snapshot(data_dir, snapshots_dir, name="interrupted")

    # snapshots_dir may exist (created before staging began) but must
    # contain no staging leftovers and no published "interrupted" snapshot.
    if snapshots_dir.exists():
        assert list(snapshots_dir.iterdir()) == []
    assert not (snapshots_dir / "interrupted").exists()


def test_failed_self_verification_before_publish_leaves_no_published_snapshot(tmp_path):
    """Simulates a copy that silently produces wrong bytes (not an
    exception) -- create_snapshot's own self-verification pass must
    catch it and refuse to publish, distinct from the interruption case
    above."""
    data_dir = _make_realistic_data_dir(tmp_path / "data")
    snapshots_dir = tmp_path / "snapshots"
    real_copy2 = data_backup.shutil.copy2

    def corrupting_copy2(src, dst, *a, **kw):
        result = real_copy2(src, dst, *a, **kw)
        if Path(src).name == "history.sqlite":
            Path(dst).write_bytes(b"silently-wrong-bytes")
        return result

    with patch.object(data_backup.shutil, "copy2", side_effect=corrupting_copy2):
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


# --- CLI: --confirm-quiescent is required ---

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

    capsys.readouterr()  # drain before the verify call's own output
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
