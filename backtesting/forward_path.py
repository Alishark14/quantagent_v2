"""ForwardPathLoader — load high-resolution OHLCV after a trade entry.

Tier 2 replay (ARCHITECTURE.md §31.3.2) needs to know the *exact* price
action that followed each historical entry, at higher resolution than the
trade's own timeframe, so it can simulate counterfactual SL/TP/trailing
mechanics accurately. Replaying a 1h trade against 1h candles loses
intra-bar information — a wider stop and a tighter stop both hit the same
candle's high/low. Replaying against 1-minute candles preserves the order
in which extremes occurred.

This loader is a thin wrapper over ``ParquetDataLoader``: it delegates the
actual file reads but adds:

- A duration-based windowing API (give me ``N`` candles starting at
  ``entry_timestamp``) instead of a calendar date range.
- A 1m → 5m fallback when the high-res file is missing.
- A ``recommended_resolution`` helper that picks 1m for short-TF trades
  (≤ 1h) and 5m for higher-TF trades (4h, 1d).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import polars as pl

from backtesting.data_loader import ParquetDataLoader
from storage.cache.ttl import TIMEFRAME_SECONDS

logger = logging.getLogger(__name__)


_VALID_RESOLUTIONS = ("1m", "5m")
_DEFAULT_DURATION = 60


class ForwardPathLoader:
    """Load the forward OHLCV path that followed a trade entry."""

    def __init__(self, data_loader: ParquetDataLoader) -> None:
        self._data_loader = data_loader

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        symbol: str,
        entry_timestamp: int,
        duration_candles: int = _DEFAULT_DURATION,
        resolution: str = "1m",
    ) -> pl.DataFrame:
        """Return ``duration_candles`` candles starting at ``entry_timestamp``.

        Args:
            symbol: Internal symbol (e.g. ``"BTC-USDC"``).
            entry_timestamp: Unix milliseconds at the trade entry bar.
            duration_candles: How many forward bars to return (default 60).
            resolution: ``"1m"`` or ``"5m"``. If ``"1m"`` is requested but
                the underlying Parquet file is missing, automatically falls
                back to ``"5m"`` and logs a warning.

        Returns:
            Polars DataFrame with columns ``timestamp, open, high, low,
            close, volume``, sorted ascending. May be shorter than
            ``duration_candles`` if the dataset doesn't extend that far.

        Raises:
            ValueError: invalid resolution or non-positive duration.
            FileNotFoundError: no data at any supported resolution.
        """
        if resolution not in _VALID_RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {_VALID_RESOLUTIONS}, got {resolution!r}"
            )
        if duration_candles <= 0:
            raise ValueError(
                f"duration_candles must be > 0, got {duration_candles}"
            )

        try:
            return self._load_at(symbol, entry_timestamp, duration_candles, resolution)
        except FileNotFoundError as exc:
            if resolution == "1m":
                logger.warning(
                    f"1m forward path missing for {symbol} at {entry_timestamp}, "
                    f"falling back to 5m: {exc}"
                )
                return self._load_at(symbol, entry_timestamp, duration_candles, "5m")
            raise

    @staticmethod
    def recommended_resolution(trade_timeframe: str) -> str:
        """Pick the right forward-path resolution for a given trade timeframe.

        - 15m / 30m / 1h trades → ``"1m"`` (preserves intra-bar structure)
        - 4h / 1d trades → ``"5m"`` (5x storage savings, still high enough
          fidelity for daily trailing-stop sims)

        Unknown timeframes default to ``"1m"`` (high fidelity > guessing).
        """
        if trade_timeframe in ("4h", "1d"):
            return "5m"
        return "1m"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_at(
        self,
        symbol: str,
        entry_timestamp: int,
        duration_candles: int,
        resolution: str,
    ) -> pl.DataFrame:
        """Load the forward path at a specific resolution (no fallback)."""
        period_ms = TIMEFRAME_SECONDS[resolution] * 1000
        # Pad the load window so a duration_candles slice always fits even
        # when entry_timestamp doesn't align to a bar boundary.
        end_ms = entry_timestamp + (duration_candles + 1) * period_ms
        start_dt = datetime.fromtimestamp(entry_timestamp / 1000, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

        df = self._data_loader.load(
            symbol=symbol,
            timeframe=resolution,
            start_date=start_dt,
            end_date=end_dt,
        )
        # Trim from the first candle whose timestamp ≥ entry_timestamp.
        df = df.filter(pl.col("timestamp") >= entry_timestamp).sort("timestamp")
        return df.head(duration_candles)
