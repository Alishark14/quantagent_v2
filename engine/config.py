"""TradingConfig, TimeframeProfiles, and feature flags."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# Timeframe helpers
# ---------------------------------------------------------------------------

_TF_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}


def timeframe_to_seconds(tf: str) -> int:
    """Convert a timeframe string to seconds."""
    if tf not in _TF_SECONDS:
        raise ValueError(f"Unknown timeframe: {tf}")
    return _TF_SECONDS[tf]


def _format_duration(total_seconds: int) -> str:
    """Format seconds into a human-readable approximate duration."""
    if total_seconds < 60:
        return f"~{total_seconds} seconds"
    minutes = total_seconds / 60
    if minutes < 120:
        return f"~{int(minutes)} minutes"
    hours = minutes / 60
    if hours < 48:
        return f"~{int(hours)} hours"
    days = hours / 24
    if days < 14:
        return f"~{int(days)} days"
    weeks = days / 7
    return f"~{int(weeks)} weeks"


def get_lookback_description(timeframe: str, num_candles: int) -> str:
    """Describe how far back a candle count reaches. e.g. 150 x 1h = '~6 days'."""
    total = timeframe_to_seconds(timeframe) * num_candles
    return _format_duration(total)


def get_forecast_description(timeframe: str, forecast_candles: int) -> str:
    """Describe the forecast horizon. e.g. 3 x 1h = '~3 hours'."""
    total = timeframe_to_seconds(timeframe) * forecast_candles
    return _format_duration(total)


# ---------------------------------------------------------------------------
# TradingConfig
# ---------------------------------------------------------------------------


@dataclass
class TradingConfig:
    """Per-bot trading configuration."""

    symbol: str = "BTC-USDC"
    timeframe: str = "1h"
    exchange: str = "hyperliquid"
    account_balance: float = 0  # 0 = fetch from exchange
    atr_length: int = 14
    forecast_candles: int = 3  # dynamic per regime later
    max_concurrent_positions: int = 1
    max_position_pct: float = 1.0
    conviction_threshold: float = 0.5

    @classmethod
    def from_env(cls) -> TradingConfig:
        """Load from environment variables, falling back to defaults."""
        return cls(
            symbol=os.environ.get("SYMBOL", "BTC-USDC"),
            timeframe=os.environ.get("TIMEFRAME", "1h"),
            exchange=os.environ.get("EXCHANGE", "hyperliquid"),
            account_balance=float(os.environ.get("ACCOUNT_BALANCE", "0")),
            atr_length=int(os.environ.get("ATR_LENGTH", "14")),
            forecast_candles=int(os.environ.get("FORECAST_CANDLES", "3")),
            max_concurrent_positions=int(os.environ.get("MAX_CONCURRENT_POSITIONS", "1")),
            max_position_pct=float(os.environ.get("MAX_POSITION_PCT", "1.0")),
            conviction_threshold=float(os.environ.get("CONVICTION_THRESHOLD", "0.5")),
        )


# ---------------------------------------------------------------------------
# TimeframeProfile
# ---------------------------------------------------------------------------


@dataclass
class TimeframeProfile:
    """Base profile for a trading timeframe, adjusted by regime at runtime."""

    timeframe: str
    candles: int
    atr_multiplier: float
    rr_min: float
    rr_max: float
    trailing_enabled: bool


# Expected hold duration by timeframe (hours) — for funding cost estimation
EXPECTED_HOLD_HOURS: dict[str, float] = {
    "15m": 2.0,
    "30m": 4.0,
    "1h": 8.0,
    "4h": 24.0,
    "1d": 72.0,
}

# Maximum fee drag before trade is blocked (10% = costs eat 10% of risk)
MAX_FEE_DRAG_PCT: float = 0.10


DEFAULT_PROFILES: dict[str, TimeframeProfile] = {
    "15m": TimeframeProfile("15m", candles=100, atr_multiplier=2.5, rr_min=0.8, rr_max=1.2, trailing_enabled=False),
    "30m": TimeframeProfile("30m", candles=100, atr_multiplier=2.0, rr_min=1.0, rr_max=1.5, trailing_enabled=False),
    "1h": TimeframeProfile("1h", candles=150, atr_multiplier=1.5, rr_min=1.5, rr_max=2.0, trailing_enabled=False),
    "4h": TimeframeProfile("4h", candles=150, atr_multiplier=1.0, rr_min=3.0, rr_max=5.0, trailing_enabled=True),
    "1d": TimeframeProfile("1d", candles=200, atr_multiplier=1.0, rr_min=3.0, rr_max=5.0, trailing_enabled=True),
}

# Regime multipliers applied by get_dynamic_profile()
_REGIME_MULTIPLIERS: dict[str, dict[str, float]] = {
    "TRENDING": {"atr_mult": 0.8, "rr_min_mult": 1.3, "rr_max_mult": 1.5},
    "TRENDING_UP": {"atr_mult": 0.8, "rr_min_mult": 1.3, "rr_max_mult": 1.5},
    "TRENDING_DOWN": {"atr_mult": 0.8, "rr_min_mult": 1.3, "rr_max_mult": 1.5},
    "RANGING": {"atr_mult": 1.2, "rr_min_mult": 0.7, "rr_max_mult": 0.8},
    "HIGH_VOLATILITY": {"atr_mult": 1.3, "rr_min_mult": 0.8, "rr_max_mult": 1.0},
    "BREAKOUT": {"atr_mult": 0.9, "rr_min_mult": 1.5, "rr_max_mult": 2.0},
}


def get_dynamic_profile(
    base: TimeframeProfile,
    regime: str,
    volatility_percentile: float,
) -> TimeframeProfile:
    """Return a new TimeframeProfile adjusted for regime and volatility."""
    mults = _REGIME_MULTIPLIERS.get(regime, {"atr_mult": 1.0, "rr_min_mult": 1.0, "rr_max_mult": 1.0})

    atr = base.atr_multiplier * mults["atr_mult"]
    rr_min = base.rr_min * mults["rr_min_mult"]
    rr_max = base.rr_max * mults["rr_max_mult"]

    # Volatility scaling
    if volatility_percentile > 80:
        atr *= 1.15
    elif volatility_percentile < 20:
        atr *= 0.85

    return TimeframeProfile(
        timeframe=base.timeframe,
        candles=base.candles,
        atr_multiplier=atr,
        rr_min=rr_min,
        rr_max=rr_max,
        trailing_enabled=base.trailing_enabled,
    )


# ---------------------------------------------------------------------------
# FeatureFlags
# ---------------------------------------------------------------------------


class FeatureFlags:
    """Feature flags loaded from config/features.yaml with env var overrides."""

    def __init__(self, yaml_path: Path | None = None) -> None:
        self._flags: dict[str, bool] = {}
        path = yaml_path or (_CONFIG_DIR / "features.yaml")
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            for key, value in data.items():
                self._flags[key] = bool(value)

    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature flag is enabled. Env vars override YAML."""
        env_key = f"FEATURE_{flag_name.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return env_val.lower() in ("true", "1", "yes")
        return self._flags.get(flag_name, False)

    def all_flags(self) -> dict[str, bool]:
        """Return a copy of all loaded flags (before env overrides)."""
        return dict(self._flags)
