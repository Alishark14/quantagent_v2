"""Unit tests for engine/config.py."""

from pathlib import Path
from unittest.mock import patch

import pytest

from engine.config import (
    DEFAULT_PROFILES,
    FeatureFlags,
    TimeframeProfile,
    TradingConfig,
    get_dynamic_profile,
    get_forecast_description,
    get_lookback_description,
    timeframe_to_seconds,
)


# ---------------------------------------------------------------------------
# TradingConfig
# ---------------------------------------------------------------------------


class TestTradingConfig:
    def test_defaults(self) -> None:
        cfg = TradingConfig()
        assert cfg.symbol == "BTC-USDC"
        assert cfg.timeframe == "1h"
        assert cfg.exchange == "hyperliquid"
        assert cfg.account_balance == 0
        assert cfg.atr_length == 14
        assert cfg.forecast_candles == 3
        assert cfg.max_concurrent_positions == 1
        assert cfg.max_position_pct == 1.0
        assert cfg.conviction_threshold == 0.5

    def test_from_env_with_defaults(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            cfg = TradingConfig.from_env()
        assert cfg.symbol == "BTC-USDC"
        assert cfg.timeframe == "1h"

    def test_from_env_with_overrides(self) -> None:
        env = {
            "SYMBOL": "ETH-USDC",
            "TIMEFRAME": "4h",
            "EXCHANGE": "dydx",
            "ACCOUNT_BALANCE": "10000",
            "ATR_LENGTH": "20",
            "FORECAST_CANDLES": "5",
            "MAX_CONCURRENT_POSITIONS": "3",
            "MAX_POSITION_PCT": "0.5",
            "CONVICTION_THRESHOLD": "0.7",
        }
        with patch.dict("os.environ", env, clear=True):
            cfg = TradingConfig.from_env()
        assert cfg.symbol == "ETH-USDC"
        assert cfg.timeframe == "4h"
        assert cfg.exchange == "dydx"
        assert cfg.account_balance == 10000.0
        assert cfg.atr_length == 20
        assert cfg.forecast_candles == 5
        assert cfg.max_concurrent_positions == 3
        assert cfg.max_position_pct == 0.5
        assert cfg.conviction_threshold == 0.7


# ---------------------------------------------------------------------------
# TimeframeProfile + DEFAULT_PROFILES
# ---------------------------------------------------------------------------


class TestTimeframeProfiles:
    def test_all_five_profiles_exist(self) -> None:
        assert set(DEFAULT_PROFILES.keys()) == {"15m", "30m", "1h", "4h", "1d"}

    def test_15m_profile(self) -> None:
        p = DEFAULT_PROFILES["15m"]
        assert p.candles == 100
        assert p.atr_multiplier == 2.5
        assert p.rr_min == 0.8
        assert p.rr_max == 1.2
        assert p.trailing_enabled is False

    def test_30m_profile(self) -> None:
        p = DEFAULT_PROFILES["30m"]
        assert p.candles == 100
        assert p.atr_multiplier == 2.0
        assert p.rr_min == 1.0
        assert p.rr_max == 1.5
        assert p.trailing_enabled is False

    def test_1h_profile(self) -> None:
        p = DEFAULT_PROFILES["1h"]
        assert p.candles == 150
        assert p.atr_multiplier == 1.5
        assert p.rr_min == 1.5
        assert p.rr_max == 2.0
        assert p.trailing_enabled is False

    def test_4h_profile(self) -> None:
        p = DEFAULT_PROFILES["4h"]
        assert p.candles == 150
        assert p.atr_multiplier == 1.0
        assert p.rr_min == 3.0
        assert p.rr_max == 5.0
        assert p.trailing_enabled is True

    def test_1d_profile(self) -> None:
        p = DEFAULT_PROFILES["1d"]
        assert p.candles == 200
        assert p.atr_multiplier == 1.0
        assert p.rr_min == 3.0
        assert p.rr_max == 5.0
        assert p.trailing_enabled is True


# ---------------------------------------------------------------------------
# get_dynamic_profile — regime multipliers
# ---------------------------------------------------------------------------


class TestDynamicProfile:
    BASE = DEFAULT_PROFILES["1h"]  # atr=1.5, rr_min=1.5, rr_max=2.0

    def test_trending(self) -> None:
        p = get_dynamic_profile(self.BASE, "TRENDING", 50.0)
        assert p.atr_multiplier == pytest.approx(1.5 * 0.8)
        assert p.rr_min == pytest.approx(1.5 * 1.3)
        assert p.rr_max == pytest.approx(2.0 * 1.5)

    def test_trending_up(self) -> None:
        p = get_dynamic_profile(self.BASE, "TRENDING_UP", 50.0)
        assert p.atr_multiplier == pytest.approx(1.5 * 0.8)
        assert p.rr_min == pytest.approx(1.5 * 1.3)

    def test_trending_down(self) -> None:
        p = get_dynamic_profile(self.BASE, "TRENDING_DOWN", 50.0)
        assert p.atr_multiplier == pytest.approx(1.5 * 0.8)

    def test_ranging(self) -> None:
        p = get_dynamic_profile(self.BASE, "RANGING", 50.0)
        assert p.atr_multiplier == pytest.approx(1.5 * 1.2)
        assert p.rr_min == pytest.approx(1.5 * 0.7)
        assert p.rr_max == pytest.approx(2.0 * 0.8)

    def test_high_volatility(self) -> None:
        p = get_dynamic_profile(self.BASE, "HIGH_VOLATILITY", 50.0)
        assert p.atr_multiplier == pytest.approx(1.5 * 1.3)
        assert p.rr_min == pytest.approx(1.5 * 0.8)
        assert p.rr_max == pytest.approx(2.0 * 1.0)

    def test_breakout(self) -> None:
        p = get_dynamic_profile(self.BASE, "BREAKOUT", 50.0)
        assert p.atr_multiplier == pytest.approx(1.5 * 0.9)
        assert p.rr_min == pytest.approx(1.5 * 1.5)
        assert p.rr_max == pytest.approx(2.0 * 2.0)

    def test_unknown_regime_no_change(self) -> None:
        p = get_dynamic_profile(self.BASE, "UNKNOWN_REGIME", 50.0)
        assert p.atr_multiplier == pytest.approx(1.5)
        assert p.rr_min == pytest.approx(1.5)
        assert p.rr_max == pytest.approx(2.0)

    def test_preserves_immutable_fields(self) -> None:
        p = get_dynamic_profile(self.BASE, "TRENDING", 50.0)
        assert p.timeframe == "1h"
        assert p.candles == 150
        assert p.trailing_enabled is False


class TestVolatilityScaling:
    BASE = DEFAULT_PROFILES["1h"]

    def test_high_volatility_percentile(self) -> None:
        p = get_dynamic_profile(self.BASE, "TRENDING", 85.0)
        # TRENDING atr_mult=0.8, then vol scale *1.15
        assert p.atr_multiplier == pytest.approx(1.5 * 0.8 * 1.15)

    def test_low_volatility_percentile(self) -> None:
        p = get_dynamic_profile(self.BASE, "TRENDING", 15.0)
        # TRENDING atr_mult=0.8, then vol scale *0.85
        assert p.atr_multiplier == pytest.approx(1.5 * 0.8 * 0.85)

    def test_mid_volatility_no_scaling(self) -> None:
        p = get_dynamic_profile(self.BASE, "TRENDING", 50.0)
        assert p.atr_multiplier == pytest.approx(1.5 * 0.8)

    def test_boundary_80_no_scaling(self) -> None:
        p = get_dynamic_profile(self.BASE, "TRENDING", 80.0)
        # 80 is NOT > 80, so no high-vol scaling
        assert p.atr_multiplier == pytest.approx(1.5 * 0.8)

    def test_boundary_20_no_scaling(self) -> None:
        p = get_dynamic_profile(self.BASE, "TRENDING", 20.0)
        # 20 is NOT < 20, so no low-vol scaling
        assert p.atr_multiplier == pytest.approx(1.5 * 0.8)

    def test_rr_not_affected_by_volatility(self) -> None:
        p_hi = get_dynamic_profile(self.BASE, "RANGING", 95.0)
        p_lo = get_dynamic_profile(self.BASE, "RANGING", 5.0)
        # rr values same regardless of volatility
        assert p_hi.rr_min == pytest.approx(p_lo.rr_min)
        assert p_hi.rr_max == pytest.approx(p_lo.rr_max)


# ---------------------------------------------------------------------------
# FeatureFlags
# ---------------------------------------------------------------------------


class TestFeatureFlags:
    def test_loads_from_yaml(self) -> None:
        ff = FeatureFlags()
        assert ff.is_enabled("paper_trading_mode") is True
        assert ff.is_enabled("sentinel_enabled") is False
        assert ff.is_enabled("ml_regime_model") is False

    def test_all_flags_loaded(self) -> None:
        ff = FeatureFlags()
        flags = ff.all_flags()
        assert "paper_trading_mode" in flags
        assert "sentinel_enabled" in flags
        assert "flow_signal_agent" in flags
        assert len(flags) == 19

    def test_env_override_true(self) -> None:
        ff = FeatureFlags()
        with patch.dict("os.environ", {"FEATURE_SENTINEL_ENABLED": "true"}):
            assert ff.is_enabled("sentinel_enabled") is True

    def test_env_override_false(self) -> None:
        ff = FeatureFlags()
        # paper_trading_mode is True in yaml, override to false
        with patch.dict("os.environ", {"FEATURE_PAPER_TRADING_MODE": "false"}):
            assert ff.is_enabled("paper_trading_mode") is False

    def test_env_override_numeric(self) -> None:
        ff = FeatureFlags()
        with patch.dict("os.environ", {"FEATURE_ML_REGIME_MODEL": "1"}):
            assert ff.is_enabled("ml_regime_model") is True

    def test_unknown_flag_defaults_false(self) -> None:
        ff = FeatureFlags()
        assert ff.is_enabled("nonexistent_flag") is False

    def test_missing_yaml_file(self, tmp_path: Path) -> None:
        ff = FeatureFlags(yaml_path=tmp_path / "nope.yaml")
        assert ff.is_enabled("anything") is False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestTimeframeToSeconds:
    def test_all_timeframes(self) -> None:
        assert timeframe_to_seconds("1m") == 60
        assert timeframe_to_seconds("5m") == 300
        assert timeframe_to_seconds("15m") == 900
        assert timeframe_to_seconds("30m") == 1800
        assert timeframe_to_seconds("1h") == 3600
        assert timeframe_to_seconds("4h") == 14400
        assert timeframe_to_seconds("1d") == 86400
        assert timeframe_to_seconds("1w") == 604800

    def test_unknown_timeframe_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown timeframe"):
            timeframe_to_seconds("2h")


class TestLookbackDescription:
    def test_15m_100(self) -> None:
        assert get_lookback_description("15m", 100) == "~25 hours"

    def test_30m_100(self) -> None:
        assert get_lookback_description("30m", 100) == "~2 days"

    def test_1h_150(self) -> None:
        assert get_lookback_description("1h", 150) == "~6 days"

    def test_4h_150(self) -> None:
        assert get_lookback_description("4h", 150) == "~3 weeks"

    def test_1d_200(self) -> None:
        assert get_lookback_description("1d", 200) == "~28 weeks"


class TestForecastDescription:
    def test_15m_3(self) -> None:
        assert get_forecast_description("15m", 3) == "~45 minutes"

    def test_30m_3(self) -> None:
        assert get_forecast_description("30m", 3) == "~90 minutes"

    def test_1h_3(self) -> None:
        assert get_forecast_description("1h", 3) == "~3 hours"

    def test_4h_3(self) -> None:
        assert get_forecast_description("4h", 3) == "~12 hours"

    def test_1d_3(self) -> None:
        assert get_forecast_description("1d", 3) == "~3 days"
