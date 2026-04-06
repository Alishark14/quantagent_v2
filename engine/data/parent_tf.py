"""Parent timeframe trend computation."""

from __future__ import annotations

import numpy as np

from engine.data.indicators import compute_adx, compute_bollinger_bands
from engine.types import ParentTFContext

_PARENT_TF_MAP: dict[str, str] = {
    "15m": "1h",
    "30m": "4h",
    "1h": "4h",
    "4h": "1d",
    "1d": "1w",
}


def get_parent_timeframe(trading_tf: str) -> str:
    """Map a trading timeframe to its parent timeframe."""
    if trading_tf not in _PARENT_TF_MAP:
        raise ValueError(f"No parent timeframe defined for: {trading_tf}")
    return _PARENT_TF_MAP[trading_tf]


def compute_parent_tf_context(
    candles: list[dict],
    timeframe: str,
) -> ParentTFContext:
    """Compute parent timeframe context from candle data.

    Expects ~50 parent TF candles. Computes:
    - MA direction: is price above or below 50-period SMA?
    - ADX value and classification
    - Bollinger Band width percentile
    """
    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)

    # SMA 50 (or all available if fewer)
    sma_period = min(50, len(close))
    sma = float(np.mean(close[-sma_period:]))
    current_price = float(close[-1])
    ma_position = "ABOVE_50MA" if current_price >= sma else "BELOW_50MA"

    # Trend direction from SMA slope
    if len(close) >= 5:
        sma_recent = float(np.mean(close[-5:]))
        sma_prev = float(np.mean(close[-10:-5])) if len(close) >= 10 else sma
        if sma_recent > sma_prev * 1.001:
            trend_direction = "BULLISH"
        elif sma_recent < sma_prev * 0.999:
            trend_direction = "BEARISH"
        else:
            trend_direction = "NEUTRAL"
    else:
        trend_direction = "NEUTRAL"

    # ADX
    adx_result = compute_adx(high, low, close)
    adx_value = adx_result["adx"]
    adx_classification = adx_result["classification"]

    # Bollinger Band width percentile
    bb = compute_bollinger_bands(close)
    bb_width_percentile = bb["width_percentile"]

    return ParentTFContext(
        timeframe=timeframe,
        trend_direction=trend_direction,
        ma_position=ma_position,
        adx_value=adx_value,
        adx_classification=adx_classification,
        bb_width_percentile=bb_width_percentile,
    )
