"""Cron entry point for the Quant Data Scientist.

Usage::

    python -m mcp.quant_scientist.runner [--dry-run] [--db-url URL]
                                         [--output PATH] [--bot-id ID]

Suitable for cron::

    0 2 * * * cd /path/to/quantagent && python -m mcp.quant_scientist.runner

The runner builds a :class:`QuantDataScientist`, calls ``run()``, and
prints a summary line that mirrors the spec example:

    Analyzed 156 trades across 5 symbols. Found 8 alpha factors
    (3 new, 5 confirmed, 2 pruned). Written to alpha_factors.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from mcp.quant_scientist.agent import QuantDataScientist
from mcp.quant_scientist.factor import AlphaFactorsReport

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mcp.quant_scientist.runner",
        description="Mine alpha factors from historical trades.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run analysis but DO NOT overwrite alpha_factors.json.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help=(
            "PostgreSQL DSN. Falls back to DATABASE_URL env var. "
            "Used only when --bot-id is supplied (or trade repo is "
            "otherwise unspecified)."
        ),
    )
    parser.add_argument(
        "--output",
        default="alpha_factors.json",
        help="Output JSON path (default alpha_factors.json).",
    )
    parser.add_argument(
        "--bot-id",
        action="append",
        default=None,
        help="Restrict scan to these bot ids. Repeatable.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Trade history window in days (default 30).",
    )
    parser.add_argument(
        "--ohlcv-lookback-days",
        type=int,
        default=180,
        help="OHLCV history window in days (default 180 = 6 months).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/parquet",
        help="Parquet root for ParquetDataLoader (default data/parquet).",
    )
    parser.add_argument(
        "--exchange",
        default="hyperliquid",
        help="Exchange name for ParquetDataLoader (default hyperliquid).",
    )
    parser.add_argument(
        "--timeframe",
        action="append",
        default=None,
        help="Timeframe(s) to load OHLCV for. Repeatable. Default 1h+4h.",
    )
    parser.add_argument(
        "--no-ohlcv",
        action="store_true",
        help="Skip OHLCV loading entirely (LLM gets empty ohlcv dict).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    if args.lookback_days < 1:
        print(
            f"ERROR: --lookback-days must be >= 1, got {args.lookback_days}",
            file=sys.stderr,
        )
        return 2

    llm_provider = _build_llm_provider()
    if llm_provider is None:
        return 2

    data_loader = None
    if not args.no_ohlcv:
        data_loader = _build_data_loader(args.data_dir, args.exchange)

    timeframes = tuple(args.timeframe) if args.timeframe else ("1h", "4h")

    agent = QuantDataScientist(
        llm_provider=llm_provider,
        data_loader=data_loader,
        output_path=Path(args.output),
        db_url=args.db_url or os.environ.get("DATABASE_URL"),
        lookback_days=args.lookback_days,
        ohlcv_lookback_days=args.ohlcv_lookback_days,
        timeframes=timeframes,
        bot_ids=args.bot_id,
    )

    report = await agent.run(dry_run=args.dry_run)
    _print_summary(report)
    return 0 if report.error is None else 1


def _build_llm_provider():
    """Construct the production ClaudeProvider lazily.

    Lazy because (a) we don't want to import anthropic on `--help` and
    (b) test_runner.py monkeypatches this function to inject a fake.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY is not set. Required for the "
            "Quant Data Scientist runner.",
            file=sys.stderr,
        )
        return None
    try:
        from llm.claude import ClaudeProvider
    except ImportError as e:
        print(f"ERROR: failed to import ClaudeProvider: {e}", file=sys.stderr)
        return None
    return ClaudeProvider(api_key=api_key)


def _build_data_loader(data_dir: str, exchange: str):
    """Construct ParquetDataLoader lazily for the same reasons."""
    try:
        from backtesting.data_loader import ParquetDataLoader
    except ImportError as e:
        logger.warning(f"runner: ParquetDataLoader unavailable: {e}")
        return None
    return ParquetDataLoader(data_dir=data_dir, exchange=exchange)


def _print_summary(report: AlphaFactorsReport) -> None:
    print()
    print("=" * 72)
    print("QUANT DATA SCIENTIST")
    print("=" * 72)
    print(
        f"Analyzed {report.trades_analyzed} trades across "
        f"{report.symbols_analyzed} symbols."
    )
    if report.error:
        print(f"  ⚠️  RUN ERROR: {report.error}")
    print(
        f"Found {report.factor_count} alpha factor(s) "
        f"({report.new_count} new, "
        f"{report.confirmed_count} confirmed, "
        f"{report.pruned_count} pruned)."
    )
    if report.dry_run:
        print(f"DRY RUN — would have written to {report.output_path}")
    elif report.error is None:
        print(f"Written to {report.output_path}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
