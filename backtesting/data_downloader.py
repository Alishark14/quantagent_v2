"""Historical OHLCV downloader → Parquet (offline backtest data pipeline).

Adapter-agnostic. Works with any concrete ``ExchangeAdapter`` (Hyperliquid,
Binance, IBKR, Alpaca, ...) via the abstract base class. The downloader has
zero exchange-specific knowledge — it only calls ``fetch_ohlcv`` on the ABC.

Files are partitioned by exchange + symbol + month::

    data/parquet/{exchange}/{SYMBOL}/{TIMEFRAME}_{YYYY-MM}.parquet

The exchange dimension prevents collisions when the same internal symbol
(e.g. ``BTC-USDC``) is downloaded from multiple venues with different
liquidity, fee structures, and price action.

This is a Tier-1 backtest dependency (see ARCHITECTURE.md §31.3.1): the
backtest engine only ever reads from local Parquet, never the exchange API
at runtime.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from exchanges.base import ExchangeAdapter
from storage.cache.ttl import TIMEFRAME_SECONDS

logger = logging.getLogger(__name__)


# Per-request fetch chunk. Hyperliquid CCXT typically allows up to ~5000
# candles per call; 1000 is a conservative chunk that works on most exchanges
# and keeps memory bounded.
_FETCH_CHUNK = 1000

# Pause between API calls to respect exchange rate limits.
_RATE_LIMIT_SLEEP = 0.5

# Parquet schema columns (order matters for stability).
_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class DownloadStats:
    """Aggregate stats from a download run."""

    files_written: int = 0
    files_skipped: int = 0
    candles_total: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Downloaded {self.files_written} files, "
            f"{self.candles_total} candles total "
            f"(skipped {self.files_skipped} existing, errors: {len(self.errors)})"
        )


class HistoricalDataDownloader:
    """Download historical OHLCV from an exchange and persist as Parquet.

    Files are partitioned by month so individual months can be
    re-downloaded without invalidating the rest of the dataset.
    """

    def __init__(
        self,
        adapter: ExchangeAdapter,
        exchange_name: str | None = None,
        data_dir: str | Path = "data/parquet",
        rate_limit_sleep: float = _RATE_LIMIT_SLEEP,
    ) -> None:
        """
        Args:
            adapter: Any concrete ``ExchangeAdapter`` implementation.
            exchange_name: Folder name under ``data_dir`` for this venue.
                Defaults to ``adapter.name()`` so callers usually omit it.
                Override only when you need to write to a non-standard
                location (e.g. ``"hyperliquid_testnet"``).
            data_dir: Root Parquet directory.
            rate_limit_sleep: Pause between paginated API calls.
        """
        self._adapter = adapter
        self._exchange_name = exchange_name or adapter.name()
        self._data_dir = Path(data_dir)
        self._rate_limit_sleep = rate_limit_sleep

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def download(
        self,
        symbols: list[str],
        timeframes: list[str],
        months_back: int,
        force: bool = False,
        now: datetime | None = None,
    ) -> DownloadStats:
        """Download `months_back` months of history for every (symbol, tf) pair.

        Args:
            symbols: Internal symbol format (e.g. "BTC-USDC").
            timeframes: Timeframe strings (e.g. ["1h", "4h"]).
            months_back: Number of calendar months to fetch, ending with the
                current month inclusive.
            force: If True, overwrite existing Parquet files. Otherwise skip
                them (resume mode).
            now: Reference "now" for month enumeration. Defaults to UTC now;
                tests inject a fixed value.

        Returns:
            DownloadStats summarising files written, skipped, and totals.
        """
        if months_back <= 0:
            raise ValueError(f"months_back must be > 0, got {months_back}")
        for tf in timeframes:
            if tf not in TIMEFRAME_SECONDS:
                raise ValueError(f"Unknown timeframe: {tf}")

        ref_now = now or datetime.now(tz=timezone.utc)
        months = _enumerate_months(ref_now, months_back)
        stats = DownloadStats()

        for symbol in symbols:
            for timeframe in timeframes:
                for month_start in months:
                    try:
                        await self._download_month(
                            symbol=symbol,
                            timeframe=timeframe,
                            month_start=month_start,
                            force=force,
                            stats=stats,
                        )
                    except Exception as e:  # never crash mid-batch
                        msg = f"{symbol} {timeframe} {month_start:%Y-%m}: {e}"
                        logger.error(f"Download failed for {msg}")
                        stats.errors.append(msg)

        logger.info(stats.summary())
        return stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parquet_path(self, symbol: str, timeframe: str, month_start: date) -> Path:
        # Timeframe is part of the filename so multiple TFs per symbol coexist.
        # Exchange is a top-level dir so the same symbol on different venues
        # never collides.
        return (
            self._data_dir
            / self._exchange_name
            / symbol
            / f"{timeframe}_{month_start:%Y-%m}.parquet"
        )

    async def _download_month(
        self,
        symbol: str,
        timeframe: str,
        month_start: date,
        force: bool,
        stats: DownloadStats,
    ) -> None:
        path = self._parquet_path(symbol, timeframe, month_start)
        if path.exists() and not force:
            logger.info(
                f"Skipping {symbol} {timeframe} {month_start:%Y-%m} "
                f"(file exists, use force=True to re-download)"
            )
            stats.files_skipped += 1
            return

        month_start_ms = _to_ms(
            datetime(month_start.year, month_start.month, 1, tzinfo=timezone.utc)
        )
        month_end_ms = _next_month_start_ms(month_start)

        logger.info(f"Downloading {symbol} {timeframe} {month_start:%Y-%m}...")
        candles = await self._fetch_month(
            symbol=symbol,
            timeframe=timeframe,
            start_ms=month_start_ms,
            end_ms=month_end_ms,
        )

        if not candles:
            logger.warning(
                f"  no candles returned for {symbol} {timeframe} {month_start:%Y-%m}"
            )
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        df = _candles_to_df(candles)
        df.write_parquet(path, compression="zstd")

        logger.info(
            f"Downloading {symbol} {timeframe} {month_start:%Y-%m}... "
            f"done ({len(candles)} candles)"
        )
        stats.files_written += 1
        stats.candles_total += len(candles)

    async def _fetch_month(
        self,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
    ) -> list[dict]:
        """Page through candles for a single month."""
        period_ms = TIMEFRAME_SECONDS[timeframe] * 1000
        seen: dict[int, dict] = {}  # dedup by timestamp
        cursor = start_ms

        while cursor < end_ms:
            batch = await self._adapter.fetch_ohlcv(
                symbol,
                timeframe,
                limit=_FETCH_CHUNK,
                since=cursor,
            )
            await asyncio.sleep(self._rate_limit_sleep)

            if not batch:
                break

            advanced = False
            for candle in batch:
                ts = int(candle.get("timestamp", 0))
                if ts < start_ms or ts >= end_ms:
                    continue
                if ts in seen:
                    continue
                seen[ts] = candle
                advanced = True

            last_ts = int(batch[-1].get("timestamp", 0))
            # Stop conditions: past the month, or no forward progress.
            if last_ts >= end_ms - period_ms:
                break
            if not advanced:
                break
            next_cursor = last_ts + period_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor

        return [seen[k] for k in sorted(seen)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _next_month_start_ms(month_start: date) -> int:
    if month_start.month == 12:
        nxt = datetime(month_start.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        nxt = datetime(month_start.year, month_start.month + 1, 1, tzinfo=timezone.utc)
    return _to_ms(nxt)


def _enumerate_months(now: datetime, months_back: int) -> list[date]:
    """Return the last `months_back` calendar months ending with `now`'s month."""
    months: list[date] = []
    y, m = now.year, now.month
    for _ in range(months_back):
        months.append(date(y, m, 1))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return list(reversed(months))


def _candles_to_df(candles: list[dict]) -> pl.DataFrame:
    """Convert candle dicts to a Polars DataFrame with the canonical schema."""
    return pl.DataFrame(
        {
            "timestamp": [int(c["timestamp"]) for c in candles],
            "open": [float(c["open"]) for c in candles],
            "high": [float(c["high"]) for c in candles],
            "low": [float(c["low"]) for c in candles],
            "close": [float(c["close"]) for c in candles],
            "volume": [float(c["volume"]) for c in candles],
        },
        schema={
            "timestamp": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    ).sort("timestamp")
