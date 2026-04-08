"""Polars-based loader for the historical Parquet dataset.

Reads files written by ``HistoricalDataDownloader`` and stitches them across
month boundaries. Used by Tier-1/Tier-2 backtest replay (see
ARCHITECTURE.md §31.3).

Path layout::

    {data_dir}/{exchange}/{SYMBOL}/{TIMEFRAME}_{YYYY-MM}.parquet

The loader is bound to one exchange per instance — the same symbol on
different venues is treated as different data.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from storage.cache.ttl import TIMEFRAME_SECONDS

logger = logging.getLogger(__name__)


class ParquetDataLoader:
    """Read historical OHLCV from local Parquet files.

    Files are expected at::

        {data_dir}/{exchange}/{SYMBOL}/{TIMEFRAME}_{YYYY-MM}.parquet

    Loads can span multiple months — the loader stitches them in order.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/parquet",
        exchange: str = "hyperliquid",
    ) -> None:
        self._data_dir = Path(data_dir)
        self._exchange = exchange

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame:
        """Return all candles in [start_date, end_date) as a Polars DataFrame.

        Raises:
            ValueError: invalid timeframe or end_date <= start_date.
            FileNotFoundError: any required monthly Parquet file is missing.
        """
        if timeframe not in TIMEFRAME_SECONDS:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        start_utc = _ensure_utc(start_date)
        end_utc = _ensure_utc(end_date)
        if end_utc <= start_utc:
            raise ValueError(
                f"end_date ({end_utc}) must be after start_date ({start_utc})"
            )

        months = _months_in_range(start_utc, end_utc)
        frames: list[pl.DataFrame] = []
        for month_start in months:
            path = self._parquet_path(symbol, timeframe, month_start)
            if not path.exists():
                raise FileNotFoundError(
                    f"No data for {self._exchange}/{symbol} {month_start:%Y-%m}. "
                    f"Run scripts/download_history.py first "
                    f"(missing file: {path})."
                )
            frames.append(pl.read_parquet(path))

        if not frames:
            return _empty_frame()

        df = pl.concat(frames).unique(subset=["timestamp"], keep="first").sort("timestamp")

        start_ms = int(start_utc.timestamp() * 1000)
        end_ms = int(end_utc.timestamp() * 1000)
        df = df.filter(
            (pl.col("timestamp") >= start_ms) & (pl.col("timestamp") < end_ms)
        )
        return df

    def load_as_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict]:
        """Return candles in the dict format the live pipeline consumes.

        Matches the output shape of ``ExchangeAdapter.fetch_ohlcv`` /
        ``OHLCVFetcher`` so the same downstream code (indicators, swing
        detection, charting) can run unchanged on historical data.
        """
        df = self.load(symbol, timeframe, start_date, end_date)
        return df.to_dicts()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parquet_path(self, symbol: str, timeframe: str, month_start: date) -> Path:
        return (
            self._data_dir
            / self._exchange
            / symbol
            / f"{timeframe}_{month_start:%Y-%m}.parquet"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _months_in_range(start: datetime, end: datetime) -> list[date]:
    """Every calendar month touched by ``[start, end)`` (inclusive of both)."""
    months: list[date] = []
    cursor = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cursor <= last:
        months.append(cursor)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return months


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "timestamp": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    )
