"""Sentinel thresholds, cooldown periods, and budget configuration.

Cooldowns and daily budgets are timeframe-dependent: a 15m bot can
trigger more often than a 4h bot. Defaults are sensible for cost control.
"""

from __future__ import annotations

# Cooldown between triggers (seconds). One candle period per timeframe.
SENTINEL_COOLDOWNS: dict[str, int] = {
    "15m": 900,       # 15 minutes
    "30m": 1800,      # 30 minutes
    "1h": 3600,       # 60 minutes
    "4h": 14400,      # 4 hours
    "1d": 86400,      # 24 hours
}

# Max full-pipeline triggers per day per symbol.
SENTINEL_DAILY_BUDGETS: dict[str, int] = {
    "15m": 16,
    "30m": 12,
    "1h": 8,
    "4h": 4,
    "1d": 2,
}


def get_sentinel_cooldown(timeframe: str) -> int:
    """Get cooldown in seconds for a timeframe. Default: 3600 (1h)."""
    return SENTINEL_COOLDOWNS.get(timeframe, 3600)


def get_sentinel_daily_budget(timeframe: str) -> int:
    """Get max daily triggers for a timeframe. Default: 8."""
    return SENTINEL_DAILY_BUDGETS.get(timeframe, 8)
