"""Golden-master eval harness — `python -m backtesting.evals.run_eval_full`.

Runs every scenario in the manifest through the pipeline N times each
(default 3) and writes a full JSON + HTML report. This is the most
expensive tier from ARCHITECTURE.md §31.4.7 and the one we gate
prompt deployments on. Don't run it from CI on every push — schedule
it nightly or run it manually before shipping a prompt change.
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
        prog="python -m backtesting.evals.run_eval_full",
        description="Golden-master eval (all scenarios, N runs each).",
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

    report = await runner.run_full(pipeline, runs_per_scenario=args.runs)

    json_path, html_path = write_report(report)
    print_summary(report, header=f"EVAL FULL — runs={args.runs}")
    print_report_paths(json_path, html_path)
    return exit_code_for(report)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
