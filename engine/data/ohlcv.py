"""OHLCV fetcher — assembles complete MarketData from exchange + indicators."""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from engine.config import (
    DEFAULT_PROFILES,
    TradingConfig,
    get_forecast_description,
    get_lookback_description,
)
from engine.data.indicators import compute_all_indicators
from engine.data.parent_tf import compute_parent_tf_context, get_parent_timeframe
from engine.data.swing_detection import find_swing_highs, find_swing_lows
from engine.types import MarketData
from exchanges.base import ExchangeAdapter

logger = logging.getLogger(__name__)

_MAX_STALE_RETRIES = 3
_STALE_RETRY_DELAY = 1.0  # seconds


class OHLCVFetcher:
    """Fetches OHLCV data and assembles a complete MarketData package."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        config: TradingConfig,
        cache=None,
    ) -> None:
        self._adapter = adapter
        self._config = config
        self._cache = cache  # Optional CacheManager

    async def _fetch_candles_with_validation(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[dict]:
        """Fetch candles with stale-candle rejection and retry.

        If the last candle's close timestamp doesn't match the expected
        candle boundary (within 1s tolerance), reject and retry up to 3 times.
        """
        from storage.cache.ttl import expected_candle_close

        expected_close = expected_candle_close(timeframe)

        for attempt in range(1, _MAX_STALE_RETRIES + 1):
            candles = await self._adapter.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not candles:
                return candles

            last_ts = candles[-1].get("timestamp", 0)
            # Timestamps may be in ms or seconds — normalize
            if last_ts > 1e12:
                last_ts = last_ts / 1000

            # Check if last candle's close aligns with expected boundary
            # The last candle's timestamp is its open time; its close = open + period
            from storage.cache.ttl import TIMEFRAME_SECONDS
            period = TIMEFRAME_SECONDS.get(timeframe, 3600)
            candle_close = last_ts + period

            if abs(candle_close - expected_close) <= 1.0:
                return candles  # fresh

            if attempt < _MAX_STALE_RETRIES:
                logger.debug(
                    f"Stale candle for {symbol}/{timeframe} "
                    f"(close={candle_close:.0f}, expected={expected_close:.0f}), "
                    f"retry {attempt}/{_MAX_STALE_RETRIES}"
                )
                await asyncio.sleep(_STALE_RETRY_DELAY)

        logger.warning(
            f"All {_MAX_STALE_RETRIES} retries returned stale candle for "
            f"{symbol}/{timeframe}, using data anyway"
        )
        return candles

    async def fetch(self, symbol: str, timeframe: str) -> MarketData:
        """Fetch candles, compute indicators/swings/parent TF, return MarketData."""
        profile = DEFAULT_PROFILES.get(timeframe)
        num_candles = profile.candles if profile else 150

        # 1. Fetch trading TF candles
        if self._cache is not None:
            # Epoch-aligned TTL + stale candle rejection (prevents caching stale data)
            from storage.cache import ohlcv_key
            from storage.cache.ttl import compute_ttl
            candles = await self._cache.get_or_fetch(
                ohlcv_key(symbol, timeframe),
                lambda: self._fetch_candles_with_validation(symbol, timeframe, num_candles),
                ttl=compute_ttl(timeframe),
            )
        else:
            # No cache — skip validation, fetch directly
            candles = await self._adapter.fetch_ohlcv(symbol, timeframe, limit=num_candles)
        if not candles:
            logger.warning(f"No candles returned for {symbol} {timeframe}")
            return MarketData(
                symbol=symbol,
                timeframe=timeframe,
                candles=[],
                num_candles=0,
                lookback_description=get_lookback_description(timeframe, 0),
                forecast_candles=self._config.forecast_candles,
                forecast_description=get_forecast_description(timeframe, self._config.forecast_candles),
                indicators={},
                swing_highs=[],
                swing_lows=[],
            )

        # 2. Compute indicators
        indicators = compute_all_indicators(candles)

        # 3. Detect swings
        high = np.array([c["high"] for c in candles], dtype=float)
        low = np.array([c["low"] for c in candles], dtype=float)
        swing_highs = find_swing_highs(high)
        swing_lows = find_swing_lows(low)

        # 4. Parent TF context (also epoch-aligned)
        parent_tf = None
        try:
            parent_timeframe = get_parent_timeframe(timeframe)
            if self._cache is not None:
                from storage.cache import ohlcv_key
                from storage.cache.ttl import compute_ttl
                parent_candles = await self._cache.get_or_fetch(
                    ohlcv_key(symbol, parent_timeframe),
                    lambda: self._adapter.fetch_ohlcv(symbol, parent_timeframe, limit=50),
                    ttl=compute_ttl(parent_timeframe),
                )
            else:
                parent_candles = await self._adapter.fetch_ohlcv(symbol, parent_timeframe, limit=50)
            if parent_candles:
                parent_tf = compute_parent_tf_context(parent_candles, parent_timeframe)
        except (ValueError, Exception) as e:
            logger.warning(f"Parent TF fetch failed for {symbol} {timeframe}: {e}")

        return MarketData(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            num_candles=len(candles),
            lookback_description=get_lookback_description(timeframe, len(candles)),
            forecast_candles=self._config.forecast_candles,
            forecast_description=get_forecast_description(timeframe, self._config.forecast_candles),
            indicators=indicators,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            parent_tf=parent_tf,
        )
