"""CLI: download historical OHLCV from any registered exchange into Parquet.

Usage:
    python scripts/download_history.py --exchange hyperliquid \
        --symbols BTC-USDC,ETH-USDC --timeframes 1h,4h --months 6 [--force]

Files land in ``data/parquet/{exchange}/{SYMBOL}/{TIMEFRAME}_{YYYY-MM}.parquet``.
Existing files are skipped unless --force is passed.

The exchange is resolved through ``ExchangeFactory`` so any registered
adapter (Hyperliquid today; Binance, IBKR, Alpaca later) works without code
changes here.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtesting.data_downloader import HistoricalDataDownloader  # noqa: E402
from exchanges.factory import ExchangeFactory  # noqa: E402
import exchanges.hyperliquid  # noqa: E402,F401  # ensure HL adapter registers


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download historical OHLCV → Parquet")
    p.add_argument(
        "--exchange",
        default="hyperliquid",
        help="Exchange adapter name (default: hyperliquid). Must be registered "
             "with ExchangeFactory.",
    )
    p.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated internal symbols (e.g. BTC-USDC,ETH-USDC)",
    )
    p.add_argument(
        "--timeframes",
        required=True,
        help="Comma-separated timeframes (e.g. 15m,1h,4h)",
    )
    p.add_argument(
        "--months",
        type=int,
        default=6,
        help="Number of months to download (default: 6)",
    )
    p.add_argument(
        "--data-dir",
        default="data/parquet",
        help="Output directory (default: data/parquet)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files instead of skipping",
    )
    p.add_argument(
        "--testnet",
        action="store_true",
        help="Pass testnet=True when constructing the adapter "
             "(only meaningful for adapters that support it, e.g. hyperliquid)",
    )
    return p.parse_args()


async def _run(args: argparse.Namespace) -> int:
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    if not symbols or not timeframes:
        print("error: --symbols and --timeframes must be non-empty", file=sys.stderr)
        return 2

    adapter_kwargs: dict = {}
    if args.testnet:
        adapter_kwargs["testnet"] = True
    try:
        adapter = ExchangeFactory.get_adapter(args.exchange, **adapter_kwargs)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    downloader = HistoricalDataDownloader(
        adapter=adapter,
        exchange_name=args.exchange,
        data_dir=args.data_dir,
    )

    stats = await downloader.download(
        symbols=symbols,
        timeframes=timeframes,
        months_back=args.months,
        force=args.force,
    )

    print(stats.summary())
    if stats.errors:
        print("\nErrors:", file=sys.stderr)
        for err in stats.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
