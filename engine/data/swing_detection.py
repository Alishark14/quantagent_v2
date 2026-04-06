"""Swing high/low detection from OHLCV data."""

from __future__ import annotations

import numpy as np


def find_swing_highs(
    high: np.ndarray,
    lookback: int = 50,
    num_swings: int = 3,
) -> list[float]:
    """Find the N most significant swing highs in the last `lookback` candles.

    Uses 2-bar pivot detection: high[i] > high[i-1] AND high[i] > high[i+1].
    Returns prices sorted by proximity to current price (nearest first).
    """
    segment = high[-lookback:] if len(high) > lookback else high
    current_price = float(high[-1])
    swings: list[float] = []

    # Need at least 3 bars for pivot detection
    for i in range(1, len(segment) - 1):
        if segment[i] > segment[i - 1] and segment[i] > segment[i + 1]:
            swings.append(float(segment[i]))

    # Sort by proximity to current price, take top N
    swings.sort(key=lambda s: abs(s - current_price))
    return swings[:num_swings]


def find_swing_lows(
    low: np.ndarray,
    lookback: int = 50,
    num_swings: int = 3,
) -> list[float]:
    """Find the N most significant swing lows in the last `lookback` candles.

    Uses 2-bar pivot detection: low[i] < low[i-1] AND low[i] < low[i+1].
    Returns prices sorted by proximity to current price (nearest first).
    """
    segment = low[-lookback:] if len(low) > lookback else low
    current_price = float(low[-1])
    swings: list[float] = []

    for i in range(1, len(segment) - 1):
        if segment[i] < segment[i - 1] and segment[i] < segment[i + 1]:
            swings.append(float(segment[i]))

    swings.sort(key=lambda s: abs(s - current_price))
    return swings[:num_swings]


def adjust_sl_to_structure(
    sl_price: float,
    direction: str,
    swing_highs: list[float],
    swing_lows: list[float],
    atr: float,
    buffer_pct: float = 0.002,
) -> float:
    """Snap SL to nearby swing structure if within 15% of ATR distance.

    For LONG: look at swing lows below SL, snap just below.
    For SHORT: look at swing highs above SL, snap just above.
    """
    threshold = atr * 0.15

    if direction.upper() == "LONG":
        # Find swing lows near the SL price
        candidates = [s for s in swing_lows if abs(s - sl_price) <= threshold]
        if candidates:
            nearest = min(candidates, key=lambda s: abs(s - sl_price))
            return nearest * (1 - buffer_pct)
    elif direction.upper() == "SHORT":
        candidates = [s for s in swing_highs if abs(s - sl_price) <= threshold]
        if candidates:
            nearest = min(candidates, key=lambda s: abs(s - sl_price))
            return nearest * (1 + buffer_pct)

    return sl_price
