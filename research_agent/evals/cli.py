"""CLI entry point for this project's own eval package (E0/R7D.1).

Usage:
    uv run python -m research_agent.evals.cli list-suites
    uv run python -m research_agent.evals.cli run --suite chat_relevance
    uv run python -m research_agent.evals.cli run --suite chat_relevance --mode mock
    uv run python -m research_agent.evals.cli run --suite chat_relevance --mode mock --subset 3
    uv run python -m research_agent.evals.cli run --suite chat_relevance --mode mock --tags redteam

`--mode` defaults to `mock` -- E0's own decision (docs/evaluation.md's
"Planned evaluation architecture" section): every suite is mocked/
offline by default, live is a separate, later, explicitly opt-in mode.
`--mode live` is a recognized value (not an argparse-level error) but is
rejected here at runtime with a clear message and a non-zero exit code
-- live mode does not exist yet in this package at all, and this
function never reaches any code path that could make a network/model
call for it.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable

from research_agent.evals.runners._base import EVAL_RESULTS_DIR, SuiteResult, append_result_csv

SUITES: dict[str, dict[str, Any]] = {
    "chat_relevance": {
        "description": "Topic-aware web relevance red-team fixtures (R7A-R7C).",
        "module": "research_agent.evals.runners.run_chat_relevance",
        "results_csv": "chat_relevance_history.csv",
    },
}


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
    if args.mode == "live":
        print(
            "[eval] mode=live is not implemented yet (research_agent.evals is "
            "mock-only as of R7D.1) -- no live model/web call was made. Use --mode mock.",
            file=sys.stderr,
        )
        return 2

    if args.suite not in SUITES:
        print(f"[eval] unknown suite {args.suite!r}. Known suites: {sorted(SUITES)}", file=sys.stderr)
        return 2

    run_experiment = _load_run_experiment(args.suite)
    result = run_experiment(mode=args.mode, subset=args.subset, tags=args.tags)
    print(result.summary_line())

    csv_path = EVAL_RESULTS_DIR / SUITES[args.suite]["results_csv"]
    append_result_csv(result, csv_path, tags=args.tags, note=args.note or "")

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
