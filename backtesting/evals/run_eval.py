"""Targeted eval harness — `python -m backtesting.evals.run_eval --category C`.

Runs every scenario in one category through the pipeline N times each
(default 3). Use this to focus on a single failure mode or to validate
a prompt change against the relevant subset of scenarios before
committing to a full eval.

See ARCHITECTURE.md §31.4.7 (tiered run specs).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from backtesting.evals._cli import (
    build_adapter,
    configure_logging,
    exit_code_for,
    print_report_paths,
    print_summary,
    write_report,
)
from backtesting.evals.framework import EvalRunner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backtesting.evals.run_eval",
        description="Targeted eval (one category, N runs per scenario).",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Scenario category to run (e.g. clear_setups, trap_setups).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Runs per scenario (default 3).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the deterministic mock pipeline (no LLM calls, no API key).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the Claude model id when running in live mode.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    if args.runs < 1:
        print(f"ERROR: --runs must be >= 1, got {args.runs}", file=sys.stderr)
        return 2

    runner = EvalRunner()
    pipeline = build_adapter(mock=args.mock, model=args.model)

    available = runner.categories()
    if args.category not in available:
        print(
            f"ERROR: unknown category {args.category!r}. "
            f"Known categories: {', '.join(sorted(available))}",
            file=sys.stderr,
        )
        return 2

    report = await runner.run_category(
        category=args.category,
        pipeline=pipeline,
        runs_per_scenario=args.runs,
    )

    json_path, html_path = write_report(report)
    print_summary(report, header=f"EVAL — category={args.category} runs={args.runs}")
    print_report_paths(json_path, html_path)
    return exit_code_for(report)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
