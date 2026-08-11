"""CLI entry point for this project's own eval package (E0/R7D.1).

Usage:
    uv run python -m research_agent.evals.cli list-suites
    uv run python -m research_agent.evals.cli run --suite chat_relevance
    uv run python -m research_agent.evals.cli run --suite chat_relevance --mode mock
    uv run python -m research_agent.evals.cli run --suite chat_relevance --mode mock --subset 3
    uv run python -m research_agent.evals.cli run --suite chat_relevance --mode mock --tags redteam

`--mode` defaults to `mock` -- E0's own decision (docs/evaluation.md's
"Planned evaluation architecture" section): every suite is mocked/
offline by default, live is a separate, explicitly opt-in mode (R7D.2)
that calls the real OpenAI embeddings API -- and, as of R7E.5, also a
real chat-completion judge call for any embedding gray-zone candidate
-- and can incur cost on both. A warning is printed whenever `--mode
live` is used, and live mode is never selected implicitly by anything
in this module.

R6D.4b (below `run`): two developer-only commands built on top of
R6D.4a's `research_agent.evals.r6d4_capture` module --

    uv run --env-file .env python -m research_agent.evals.cli capture-refinement \\
        --session-id <local-session-id> --pair-id real-foundational-01 \\
        --template foundational --allow-paid-calls
    uv run python -m research_agent.evals.cli validate-refinement-capture \\
        --path eval_results/captures/real-foundational-01.json

`capture-refinement` makes REAL, billable R4 calls (report generation
+ R4's own evaluation, plus one more if R4 decides to revise) -- it is
gated behind an explicit `--allow-paid-calls` flag with zero side
effects of any kind when that flag is absent (no session load, no
OpenAI client, no directory/file). It never invokes any R6D judge --
capture and evaluation are two separate steps by design (see `run
--suite report_refinement --mode live`, above, for the evaluation
half), so re-running evaluation on an already-captured artifact never
re-triggers generation. `validate-refinement-capture` is the read-only
counterpart: zero network calls, loads no session, writes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI, OpenAIError

from research_agent.evals.r6d4_capture import R6D4CaptureError, capture_real_refinement_pair, validate_r6d4_capture
from research_agent.evals.runners._base import (
    EVAL_RESULTS_DIR,
    LiveModeSetupError,
    SuiteResult,
    _git_sha,
    append_result_csv,
    write_run_detail_json,
)

SUITES: dict[str, dict[str, Any]] = {
    "chat_relevance": {
        "description": "Topic-aware web relevance red-team fixtures (R7A-R7C).",
        "module": "research_agent.evals.runners.run_chat_relevance",
        "results_csv": "chat_relevance_history.csv",
        "live_warning": (
            "mode=live calls the real OpenAI embeddings API, and may also call a real chat-completion "
            "judge model for embedding gray-zone candidates -- both can incur cost."
        ),
    },
    "report_quality": {
        "description": (
            "Deterministic report structural/citation checks against synthetic fixtures (R6B), plus "
            "R6C.2's opt-in live claim/source and holistic judges."
        ),
        "module": "research_agent.evals.runners.run_report_quality",
        "results_csv": "report_quality_history.csv",
        "live_warning": (
            "mode=live makes real OpenAI judge calls (one claim/source call plus one holistic call per "
            "eligible report) and can incur cost."
        ),
    },
    "report_refinement": {
        "description": (
            "Deterministic pair evaluation (R6D.2) -- runs R6B's own structural/citation checks against "
            "both the draft and refined report in each R6D.1 fixture and derives a hard-failure direction. "
            "No semantic (R6C) dimension is measured yet; live paired judging is R6D.3."
        ),
        "module": "research_agent.evals.runners.run_report_refinement",
        "results_csv": "report_refinement_history.csv",
        "live_warning": (
            "mode=live makes real OpenAI judge calls -- up to 3 per pair (one claim/source call for EACH "
            "of draft and refined, plus one pairwise holistic call comparing both reports together), "
            "fewer when the identical-input optimization applies for a revision_applied=false pair -- and "
            "can incur cost."
        ),
    },
    "report_refinement_real": {
        "description": (
            "Evaluates exactly 3 frozen real R4-generated draft/refined pairs (R6D.4) against their "
            "already-frozen human/deterministic adjudications -- never the 7 synthetic report_refinement "
            "fixtures, and a separate result history. Score is exact agreement with frozen labels, not "
            "report quality or refinement benefit."
        ),
        "module": "research_agent.evals.runners.run_report_refinement_real",
        "results_csv": "report_refinement_real_history.csv",
        "live_warning": (
            "mode=live makes real OpenAI judge calls -- at most 5 total across the 3 frozen pairs under "
            "current pair content (1 claim/source call for each of the 2 byte-identical pairs, reused for "
            "the refined side; 2 claim/source calls + 1 pairwise holistic call for the 1 changed pair) -- "
            "never another overall/pairwise winner call -- and can incur cost."
        ),
    },
}

_DEFAULT_LIVE_WARNING = "mode=live makes real OpenAI API calls and can incur cost."


def _load_run_experiment(suite: str) -> Callable[..., SuiteResult]:
    import importlib

    module = importlib.import_module(SUITES[suite]["module"])
    return module.run_experiment


def cmd_list_suites(_args: argparse.Namespace) -> int:
    print(f"{'suite':<16} description")
    print(f"{'-' * 16} {'-' * 50}")
    for name, meta in SUITES.items():
        print(f"{name:<16} {meta['description']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.suite not in SUITES:
        print(f"[eval] unknown suite {args.suite!r}. Known suites: {sorted(SUITES)}", file=sys.stderr)
        return 2

    if args.mode == "live":
        warning = SUITES[args.suite].get("live_warning", _DEFAULT_LIVE_WARNING)
        print(f"[eval] WARNING: {warning}", file=sys.stderr)

    run_experiment = _load_run_experiment(args.suite)
    try:
        result = run_experiment(mode=args.mode, subset=args.subset, tags=args.tags)
    except LiveModeSetupError as exc:
        print(f"[eval] {exc}", file=sys.stderr)
        return 2
    print(result.summary_line())

    csv_path = EVAL_RESULTS_DIR / SUITES[args.suite]["results_csv"]
    run_id = append_result_csv(result, csv_path, tags=args.tags, note=args.note or "")
    detail_path = write_run_detail_json(
        result, run_id, EVAL_RESULTS_DIR / "runs", subset=args.subset, tags=args.tags, note=args.note or "",
    )
    print(f"[eval] run detail written to {detail_path}")

    return 0 if result.failed == 0 else 1


# --- R6D.4b: developer-only real-pair capture/validate commands -----------

CAPTURES_DIR = EVAL_RESULTS_DIR / "captures"

# Conservative filename-safe id: letters/digits/hyphen/underscore only,
# 2-99 characters, must start and end with a letter or digit -- never a
# path separator, never ".."/leading dot, so `--pair-id` can never be
# used to escape CAPTURES_DIR or collide with a dotfile.
_PAIR_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,97}[A-Za-z0-9]$")

_CAPTURE_PAID_CALL_WARNING = (
    "capture-refinement invokes REAL R4 generation and evaluation (2 calls) -- and, if R4's own "
    "evaluator decides the draft needs revision, one more real revision call (3 total). It does NOT "
    "invoke any R6D judge (no claim/source call, no pairwise holistic call) -- capture and evaluation "
    "are separate steps. This can incur cost. It writes exactly one frozen, unlabelled capture "
    "artifact."
)


def _validate_pair_id(pair_id: str) -> str | None:
    if not _PAIR_ID_RE.match(pair_id):
        return (
            f"--pair-id {pair_id!r} is not filename-safe -- use only letters, digits, '-', '_' "
            "(2-99 characters, must start and end with a letter or digit)"
        )
    return None


def _atomic_write_json(data: dict[str, Any], destination: Path) -> None:
    """Serializes `data` to JSON BEFORE ever touching the filesystem
    (so a serialization failure leaves no directory/temp file at all),
    then writes it to a temp file in `destination`'s own parent
    directory, flushes+fsyncs it, and `os.replace`s it into place --
    atomic on the same filesystem, so a reader of `destination` only
    ever sees either the complete previous state (nothing written yet)
    or the complete new artifact, never a partial write. The temp file
    is removed on ANY failure (serialization already happened above,
    so the only remaining failure modes are the write itself or the
    final replace)."""
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def cmd_capture_refinement(args: argparse.Namespace) -> int:
    """R6D.4b: developer-only capture of one real R4 draft/refined
    report pair. Every guard below runs in the exact order the chunk's
    own safety requirements demand: the --allow-paid-calls gate first
    (zero side effects at all when absent -- not even argument
    validation beyond argparse's own parsing), then the cheap local
    checks (pair_id shape, destination-already-exists), THEN OpenAI
    client construction, THEN session loading -- so a missing
    credential or an already-captured pair is caught before any real
    work (session load included) ever starts.
    """
    if not args.allow_paid_calls:
        print(
            "[eval] capture-refinement requires --allow-paid-calls -- it makes real, billable R4 "
            "generation/evaluation/revision calls. Re-run with --allow-paid-calls once you intend to "
            "spend those calls. No session was loaded, no OpenAI client was constructed, and nothing "
            "was written.",
            file=sys.stderr,
        )
        return 2

    pair_id_error = _validate_pair_id(args.pair_id)
    if pair_id_error:
        print(f"[eval] {pair_id_error}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    destination = output_dir / f"{args.pair_id}.json"
    if destination.exists():
        print(
            f"[eval] {destination} already exists -- refusing to overwrite it (no --force in this "
            "chunk). Choose a different --pair-id, or remove the existing file yourself first. "
            "Nothing was done.",
            file=sys.stderr,
        )
        return 2

    print(f"[eval] WARNING: {_CAPTURE_PAID_CALL_WARNING}", file=sys.stderr)

    try:
        client = OpenAI()
    except OpenAIError as exc:
        print(
            "[eval] capture-refinement requires OpenAI credentials (OPENAI_API_KEY or another "
            f"OpenAI-SDK-recognized credential) -- client construction failed: {exc}",
            file=sys.stderr,
        )
        return 2

    # Deferred imports (same convention `_load_run_experiment` above already
    # uses) -- capture-refinement is the only command in this module that
    # needs the curation-session/checkpointer stack at all.
    from research_agent.curation_session import load_curation_session
    from research_agent.qa import sqlite_checkpointer

    with sqlite_checkpointer() as cp:
        session = load_curation_session(args.session_id, cp)

    if session is None:
        # Deliberately never echoes the raw --session-id value back to the user.
        print("[eval] the requested session could not be loaded -- check --session-id and try again.", file=sys.stderr)
        return 2
    if session.stage != "synthesize":
        print(
            f"[eval] session is not ready for report synthesis (stage={session.stage!r}, expected "
            "'synthesize') -- curation must finish before a report can be generated. A different "
            "--session-id must be chosen manually; this command never substitutes one automatically.",
            file=sys.stderr,
        )
        return 2
    if not session.selected_papers:
        print(
            "[eval] session has no selected papers -- nothing to generate a report from. A different "
            "--session-id must be chosen manually; this command never substitutes one automatically.",
            file=sys.stderr,
        )
        return 2

    source_session_ref = args.source_session_ref or args.pair_id
    source_commit_sha = args.source_commit_sha or _git_sha()

    try:
        artifact = capture_real_refinement_pair(
            session, client, report_template=args.template, pair_id=args.pair_id,
            source_session_ref=source_session_ref, source_commit_sha=source_commit_sha,
        )
        validate_r6d4_capture(artifact)
    except (R6D4CaptureError, ValueError) as exc:
        print(f"[eval] capture failed: {exc}", file=sys.stderr)
        return 1

    _atomic_write_json(artifact, destination)

    refinement_context = artifact["refinement_context"]
    r4_meta = refinement_context["r4_refinement_metadata"]
    bodies_equal = artifact["draft_report"] == artifact["refined_report"]
    print(f"[eval] artifact written to {destination}")
    print(f"[eval] pair_id={artifact['id']} template={artifact['template']}")
    print(f"[eval] revision_applied={refinement_context['revision_applied']} r4_rounds={r4_meta.get('rounds')}")
    print(f"[eval] draft_report == refined_report: {bodies_equal}")
    print("[eval] validation passed")
    return 0


def cmd_validate_refinement_capture(args: argparse.Namespace) -> int:
    """R6D.4b: pure, read-only validation of an already-captured
    artifact file -- zero OpenAI calls, no session ever loaded, the
    file is only ever read, never written. An `r6d1-v1` synthetic
    fixture is rejected as the wrong schema (validate_r6d4_capture's
    own first check), never silently accepted as a real capture."""
    path = Path(args.path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[eval] could not read {path}: {exc}", file=sys.stderr)
        return 2

    try:
        artifact = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[eval] {path} is not valid JSON: {exc}", file=sys.stderr)
        return 2

    try:
        validate_r6d4_capture(artifact)
    except R6D4CaptureError as exc:
        print(f"[eval] INVALID: {exc}", file=sys.stderr)
        return 1

    print(f"[eval] VALID: {path} (id={artifact.get('id')!r}, template={artifact.get('template')!r})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research_agent.evals.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-suites", help="Print available suites.").set_defaults(func=cmd_list_suites)

    run_parser = sub.add_parser("run", help="Run one suite.")
    run_parser.add_argument("--suite", required=True, help="Suite name, e.g. chat_relevance.")
    run_parser.add_argument("--mode", default="mock", choices=["mock", "live"], help="Default: mock.")
    run_parser.add_argument(
        "--subset", type=int, default=None, help="Run on the first N examples (after tag filtering).",
    )
    run_parser.add_argument(
        "--tags", nargs="*", default=None, help="Only run examples carrying at least one of these tags.",
    )
    run_parser.add_argument("--note", default=None, help="Freeform note stored in the results CSV.")
    run_parser.set_defaults(func=cmd_run)

    capture_parser = sub.add_parser(
        "capture-refinement",
        help="R6D.4b: capture one real R4 draft/refined report pair (developer-only, makes real paid calls).",
    )
    capture_parser.add_argument("--session-id", required=True, help="Local curation session id to generate a report for (read-only, never persisted in the artifact).")
    capture_parser.add_argument("--pair-id", required=True, help="Filename-safe id for this capture, e.g. real-foundational-01.")
    capture_parser.add_argument("--template", required=True, choices=["foundational", "analytical", "expert"])
    capture_parser.add_argument("--output-dir", default=str(CAPTURES_DIR), help=f"Default: {CAPTURES_DIR}.")
    capture_parser.add_argument(
        "--source-session-ref", default=None,
        help="Opaque provenance label stored in the artifact; defaults to --pair-id. Never the raw session id.",
    )
    capture_parser.add_argument("--source-commit-sha", default=None, help="Defaults to the current git HEAD short SHA, if available.")
    capture_parser.add_argument(
        "--allow-paid-calls", action="store_true",
        help="Required. Acknowledges this makes real, billable R4 generation/evaluation/revision calls.",
    )
    capture_parser.set_defaults(func=cmd_capture_refinement)

    validate_parser = sub.add_parser(
        "validate-refinement-capture",
        help="R6D.4b: validate an existing r6d4-capture-v1 artifact file (read-only, no network, no session).",
    )
    validate_parser.add_argument("--path", required=True, help="Path to the captured JSON artifact.")
    validate_parser.set_defaults(func=cmd_validate_refinement_capture)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
