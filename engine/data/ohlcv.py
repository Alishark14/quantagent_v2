"""OHLCV fetcher — assembles complete MarketData from exchange + indicators."""

from __future__ import annotations

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


class OHLCVFetcher:
    """Fetches OHLCV data and assembles a complete MarketData package."""

    def __init__(self, adapter: ExchangeAdapter, config: TradingConfig) -> None:
        self._adapter = adapter
        self._config = config

    async def fetch(self, symbol: str, timeframe: str) -> MarketData:
        """Fetch candles, compute indicators/swings/parent TF, return MarketData."""
        profile = DEFAULT_PROFILES.get(timeframe)
        num_candles = profile.candles if profile else 150

        # 1. Fetch trading TF candles
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

        # 4. Parent TF context
        parent_tf = None
        try:
            parent_timeframe = get_parent_timeframe(timeframe)
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
