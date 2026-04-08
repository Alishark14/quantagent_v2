"""Epoch-aligned TTL computation and per-provider TTL constants.

Fixed TTLs drift over time: a 3600s TTL set at 12:00:03 expires at
13:00:03, missing the candle boundary. Epoch-aligned TTLs always
expire at the next candle open + a 2-second settle buffer.
"""

from __future__ import annotations

import time

# Timeframe to seconds mapping
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

# Per-provider TTL constants (seconds)
FLOW_TTL: int = 300           # 5 minutes (L2)
REGIME_TTL: int = 14400       # 4 hours (L3)
SENTIMENT_TTL: int = 3600     # 1 hour (L3)
NEWS_TTL: int = 1800          # 30 minutes (L3)

# Exchange settle buffer after candle boundary
_SETTLE_BUFFER: float = 2.0


def compute_ttl(timeframe: str) -> float:
    """Calculate TTL aligned to next candle open + 2s settle buffer.

    At 12:30:00 with 1h candles: next open is 13:00:00, TTL = 1802.0s.
    At 12:59:58 with 1h candles: next open is 13:00:00, TTL = 4.0s.
    """
    now = time.time()
    period = TIMEFRAME_SECONDS.get(timeframe)
    if period is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    next_candle_open = ((now // period) + 1) * period
    return (next_candle_open - now) + _SETTLE_BUFFER


def expected_candle_close(timeframe: str) -> float:
    """Return the expected close timestamp of the current candle.

    The close timestamp equals the open of the *next* candle.
    """
    now = time.time()
    period = TIMEFRAME_SECONDS.get(timeframe)
    if period is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    current_candle_open = (now // period) * period
    return current_candle_open + period
