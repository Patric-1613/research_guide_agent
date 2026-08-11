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
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable

from research_agent.evals.runners._base import (
    EVAL_RESULTS_DIR,
    LiveModeSetupError,
    SuiteResult,
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
            "mode=live makes real OpenAI judge calls -- up to 4 per pair (one claim/source call plus one "
            "holistic call for EACH of draft and refined), fewer when the identical-input optimization "
            "applies for a revision_applied=false pair -- and can incur cost."
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
