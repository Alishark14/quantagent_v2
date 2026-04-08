"""CLI: run a Tier 1 mechanical backtest from local Parquet data.

Usage:
    python scripts/run_backtest.py \\
        --mode mechanical \\
        --symbols BTC-USDC \\
        --timeframes 1h \\
        --start 2025-10-01 \\
        --end 2026-04-01 \\
        --balance 10000 \\
        --signal-mode random_seed:42 \\
        --exchange hyperliquid

Reads from ``data/parquet/{exchange}/...`` (run scripts/download_history.py
first if you don't have it). Writes results to
``backtesting/results/{timestamp}_backtest.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtesting.data_loader import ParquetDataLoader  # noqa: E402
from backtesting.engine import BacktestConfig, BacktestEngine  # noqa: E402
from backtesting.mock_signals import MockSignalProducer  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a Tier 1 mechanical backtest")
    p.add_argument("--mode", default="mechanical", choices=["mechanical", "full"])
    p.add_argument("--symbols", required=True, help="Comma-separated symbols")
    p.add_argument("--timeframes", required=True, help="Comma-separated timeframes")
    p.add_argument("--start", required=True, help="ISO date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="ISO date (YYYY-MM-DD)")
    p.add_argument("--balance", type=float, default=10_000.0)
    p.add_argument("--exchange", default="hyperliquid")
    p.add_argument(
        "--signal-mode",
        default="random_seed:42",
        help="MockSignalProducer mode (always_long / always_short / "
             "always_skip / random_seed:N / from_file:PATH)",
    )
    p.add_argument(
        "--data-dir",
        default="data/parquet",
        help="Root Parquet directory (default: data/parquet)",
    )
    p.add_argument(
        "--results-dir",
        default="backtesting/results",
        help="Where to write the run JSON",
    )
    return p.parse_args()


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _format_summary(result) -> str:
    m = result.metrics
    pnl_pct = (
        100.0 * (result.final_balance - result.initial_balance) / result.initial_balance
        if result.initial_balance else 0.0
    )
    return (
        f"\n=== Backtest Summary ===\n"
        f"  Mode:           {result.config.mode}\n"
        f"  Symbols:        {','.join(result.config.symbols)}\n"
        f"  Timeframes:     {','.join(result.config.timeframes)}\n"
        f"  Period:         {result.config.start_date.date()} → {result.config.end_date.date()}\n"
        f"  Candles:        {result.candles_processed}\n"
        f"  Duration:       {result.duration_seconds:.2f}s\n"
        f"  Initial:        ${result.initial_balance:,.2f}\n"
        f"  Final:          ${result.final_balance:,.2f}  ({pnl_pct:+.2f}%)\n"
        f"  Trades:         {m['total_trades']}  (W {m['winning_trades']} / L {m['losing_trades']})\n"
        f"  Win rate:       {m['win_rate'] * 100:.1f}%\n"
        f"  Total PnL:      ${m['total_pnl']:,.2f}\n"
        f"  Total fees:     ${m['total_fees']:,.2f}\n"
        f"  Max drawdown:   {m['max_drawdown_pct']:.2f}%\n"
    )


async def _run(args: argparse.Namespace) -> int:
    config = BacktestConfig(
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        timeframes=[t.strip() for t in args.timeframes.split(",") if t.strip()],
        start_date=_parse_date(args.start),
        end_date=_parse_date(args.end),
        initial_balance=args.balance,
        mode=args.mode,
        exchange=args.exchange,
    )

    loader = ParquetDataLoader(data_dir=args.data_dir, exchange=args.exchange)
    producer = MockSignalProducer(args.signal_mode)
    engine = BacktestEngine(
        config=config,
        data_loader=loader,
        signal_producer=producer,
    )

    print(
        f"Running {config.mode} backtest on "
        f"{','.join(config.symbols)} {','.join(config.timeframes)} "
        f"{config.start_date.date()}→{config.end_date.date()}..."
    )
    result = await engine.run()
    print(_format_summary(result))

    # Persist
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = results_dir / f"{ts}_backtest.json"
    with out.open("w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Wrote results to {out}")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    try:
        return asyncio.run(_run(args))
    except NotImplementedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
