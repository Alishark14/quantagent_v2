"""Smoke test harness — `python -m backtesting.evals.run_smoke [--mock]`.

Runs the EvalRunner's ``run_smoke`` mode (2 scenarios per category,
1 run each), generates the JSON + HTML report, prints a summary, and
returns CI exit code 0 if the overall pass-rate exceeds 50 % (else 1).

This is the cheapest tier from ARCHITECTURE.md §31.4.7. With ``--mock``
it makes zero LLM calls and is safe to run in CI on every push.
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
        prog="python -m backtesting.evals.run_smoke",
        description="Eval smoke test (2 scenarios per category, 1 run each).",
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
    runner = EvalRunner()
    pipeline = build_adapter(mock=args.mock, model=args.model)
    report = await runner.run_smoke(pipeline)

    json_path, html_path = write_report(report)
    print_summary(report, header="EVAL SMOKE TEST")
    print_report_paths(json_path, html_path)
    return exit_code_for(report)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
