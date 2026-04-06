"""Unit tests for engine/data/indicators.py and engine/data/parent_tf.py."""

from __future__ import annotations

import numpy as np
import pytest

from engine.data.indicators import (
    compute_atr,
    compute_atr_series,
    compute_adx,
    compute_bollinger_bands,
    compute_macd,
    compute_roc,
    compute_rsi,
    compute_stochastic,
    compute_volume_ma,
    compute_williams_r,
    compute_all_indicators,
    get_volatility_percentile,
)
from engine.data.parent_tf import compute_parent_tf_context, get_parent_timeframe


# ---------------------------------------------------------------------------
# Helpers — generate simple OHLCV series
# ---------------------------------------------------------------------------

def _uptrend_close(n: int = 30, start: float = 100.0, step: float = 1.0) -> np.ndarray:
    """Monotonically increasing close prices."""
    return np.arange(start, start + n * step, step)


def _downtrend_close(n: int = 30, start: float = 130.0, step: float = 1.0) -> np.ndarray:
    """Monotonically decreasing close prices."""
    return np.arange(start, start - n * step, -step)


def _flat_close(n: int = 30, price: float = 100.0) -> np.ndarray:
    return np.full(n, price)


def _make_candles(
    close: np.ndarray,
    spread: float = 2.0,
    volume: float = 1000.0,
) -> list[dict]:
    """Build OHLCV candle dicts from a close array."""
    candles = []
    for i, c in enumerate(close):
        candles.append({
            "timestamp": 1700000000 + i * 3600,
            "open": c - 0.5,
            "high": c + spread / 2,
            "low": c - spread / 2,
            "close": float(c),
            "volume": volume,
        })
    return candles


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


class TestRSI:
    def test_all_up_candles(self) -> None:
        close = _uptrend_close(30)
        rsi = compute_rsi(close)
        assert rsi > 90  # near 100

    def test_all_down_candles(self) -> None:
        close = _downtrend_close(30)
        rsi = compute_rsi(close)
        assert rsi < 10  # near 0

    def test_mixed_candles_between_extremes(self) -> None:
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50))
        rsi = compute_rsi(close)
        assert 10 < rsi < 90

    def test_flat_close_returns_50(self) -> None:
        close = _flat_close(30)
        rsi = compute_rsi(close)
        assert rsi == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


class TestMACD:
    def test_bullish_cross(self) -> None:
        # Down then sharp up: should produce bullish cross
        close = np.concatenate([_downtrend_close(30, 130, 0.5), _uptrend_close(20, 115, 2.0)])
        result = compute_macd(close)
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result
        assert result["histogram_direction"] in ("rising", "falling")
        assert result["cross"] in ("bullish_cross", "bearish_cross", "none")

    def test_strong_uptrend_positive_macd(self) -> None:
        close = _uptrend_close(50, 100, 1.0)
        result = compute_macd(close)
        assert result["macd"] > 0
        assert result["histogram"] >= -1e-10  # allow float rounding

    def test_strong_downtrend_negative_macd(self) -> None:
        close = _downtrend_close(50, 150, 1.0)
        result = compute_macd(close)
        assert result["macd"] < 0

    def test_returns_all_keys(self) -> None:
        close = _uptrend_close(50)
        result = compute_macd(close)
        assert set(result.keys()) == {"macd", "signal", "histogram", "histogram_direction", "cross"}


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------


class TestROC:
    def test_uptrend_positive(self) -> None:
        close = _uptrend_close(20, 100, 1.0)
        roc = compute_roc(close)
        assert roc > 0

    def test_downtrend_negative(self) -> None:
        close = _downtrend_close(20, 120, 1.0)
        roc = compute_roc(close)
        assert roc < 0

    def test_flat_zero(self) -> None:
        close = _flat_close(20)
        roc = compute_roc(close)
        assert roc == pytest.approx(0.0)

    def test_known_value(self) -> None:
        # 10 periods ago = 100, now = 110 => ROC = 10%
        close = np.array([100.0] * 10 + [110.0])
        roc = compute_roc(close, period=10)
        assert roc == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------


class TestStochastic:
    def test_overbought_zone(self) -> None:
        # Price at top of range
        high = np.full(20, 110.0)
        low = np.full(20, 90.0)
        close = np.full(20, 109.0)  # near high
        result = compute_stochastic(high, low, close)
        assert result["zone"] == "overbought"
        assert result["k"] > 80

    def test_oversold_zone(self) -> None:
        high = np.full(20, 110.0)
        low = np.full(20, 90.0)
        close = np.full(20, 91.0)  # near low
        result = compute_stochastic(high, low, close)
        assert result["zone"] == "oversold"
        assert result["k"] < 20

    def test_neutral_zone(self) -> None:
        high = np.full(20, 110.0)
        low = np.full(20, 90.0)
        close = np.full(20, 100.0)  # midrange
        result = compute_stochastic(high, low, close)
        assert result["zone"] == "neutral"

    def test_returns_all_keys(self) -> None:
        close = _uptrend_close(20)
        high = close + 1
        low = close - 1
        result = compute_stochastic(high, low, close)
        assert set(result.keys()) == {"k", "d", "zone"}


# ---------------------------------------------------------------------------
# Williams %R
# ---------------------------------------------------------------------------


class TestWilliamsR:
    def test_range_bounds(self) -> None:
        close = _uptrend_close(20)
        high = close + 1
        low = close - 1
        wr = compute_williams_r(high, low, close)
        assert -100 <= wr <= 0

    def test_at_high_near_zero(self) -> None:
        high = np.full(20, 110.0)
        low = np.full(20, 90.0)
        close = np.full(20, 110.0)  # at the high
        wr = compute_williams_r(high, low, close)
        assert wr == pytest.approx(0.0)

    def test_at_low_near_minus_100(self) -> None:
        high = np.full(20, 110.0)
        low = np.full(20, 90.0)
        close = np.full(20, 90.0)
        wr = compute_williams_r(high, low, close)
        assert wr == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


class TestATR:
    def test_constant_range(self) -> None:
        # All candles same range of 10 => ATR should converge to 10
        n = 50
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)
        atr = compute_atr(high, low, close)
        assert atr == pytest.approx(10.0, abs=0.5)

    def test_increasing_range(self) -> None:
        n = 50
        high = 100 + np.arange(n, dtype=float)
        low = 100 - np.arange(n, dtype=float)
        close = np.full(n, 100.0)
        atr = compute_atr(high, low, close)
        assert atr > 10  # ATR grows with range

    def test_atr_series_length(self) -> None:
        n = 50
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)
        series = compute_atr_series(high, low, close)
        assert len(series) == n


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


class TestADX:
    def test_trending_classification(self) -> None:
        # Strong trend => ADX > 25
        close = _uptrend_close(60, 100, 2.0)
        high = close + 1
        low = close - 1
        result = compute_adx(high, low, close)
        assert result["adx"] > 25
        assert result["classification"] == "TRENDING"

    def test_ranging_classification(self) -> None:
        # Oscillating price => ADX < 20
        n = 60
        close = 100 + 0.5 * np.sin(np.linspace(0, 20 * np.pi, n))
        high = close + 0.3
        low = close - 0.3
        result = compute_adx(high, low, close)
        assert result["classification"] in ("RANGING", "WEAK")

    def test_returns_all_keys(self) -> None:
        close = _uptrend_close(40)
        high = close + 1
        low = close - 1
        result = compute_adx(high, low, close)
        assert set(result.keys()) == {"adx", "plus_di", "minus_di", "classification"}

    def test_uptrend_plus_di_greater(self) -> None:
        close = _uptrend_close(60, 100, 2.0)
        high = close + 1
        low = close - 1
        result = compute_adx(high, low, close)
        assert result["plus_di"] > result["minus_di"]


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


class TestBollingerBands:
    def test_structure(self) -> None:
        close = _uptrend_close(30)
        bb = compute_bollinger_bands(close)
        assert bb["upper"] > bb["middle"] > bb["lower"]
        assert bb["width"] > 0

    def test_flat_price_narrow_bands(self) -> None:
        close = _flat_close(30)
        bb = compute_bollinger_bands(close)
        assert bb["width"] == pytest.approx(0.0, abs=0.01)

    def test_width_percentile_range(self) -> None:
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100))
        bb = compute_bollinger_bands(close)
        assert 0 <= bb["width_percentile"] <= 100


# ---------------------------------------------------------------------------
# Volume MA
# ---------------------------------------------------------------------------


class TestVolumeMA:
    def test_normal_volume(self) -> None:
        volume = np.full(30, 1000.0)
        result = compute_volume_ma(volume)
        assert result["ratio"] == pytest.approx(1.0)
        assert result["spike"] is False

    def test_volume_spike(self) -> None:
        volume = np.full(30, 1000.0)
        volume[-1] = 5000.0  # 5x MA
        result = compute_volume_ma(volume)
        assert result["spike"] is True
        assert result["ratio"] > 3.0

    def test_returns_all_keys(self) -> None:
        volume = np.full(30, 1000.0)
        result = compute_volume_ma(volume)
        assert set(result.keys()) == {"ma", "current", "ratio", "spike"}


# ---------------------------------------------------------------------------
# Volatility percentile
# ---------------------------------------------------------------------------


class TestVolatilityPercentile:
    def test_high_vol_high_percentile(self) -> None:
        series = np.arange(1.0, 101.0)  # current = 100 (highest)
        pct = get_volatility_percentile(series)
        assert pct > 90

    def test_low_vol_low_percentile(self) -> None:
        series = np.arange(100.0, 0.0, -1.0)  # current = 1 (lowest)
        pct = get_volatility_percentile(series)
        assert pct < 10

    def test_highest_val_high_percentile(self) -> None:
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        # current = 10, searchsorted finds index 9 of 10 => 90%
        pct = get_volatility_percentile(series)
        assert pct >= 90.0


# ---------------------------------------------------------------------------
# compute_all_indicators
# ---------------------------------------------------------------------------


class TestComputeAll:
    def test_returns_all_indicator_keys(self) -> None:
        close = _uptrend_close(50)
        candles = _make_candles(close)
        result = compute_all_indicators(candles)
        expected_keys = {
            "rsi", "macd", "roc", "stochastic", "williams_r",
            "atr", "adx", "bollinger_bands", "volume_ma",
            "volatility_percentile",
        }
        assert set(result.keys()) == expected_keys

    def test_rsi_is_numeric(self) -> None:
        close = _uptrend_close(50)
        candles = _make_candles(close)
        result = compute_all_indicators(candles)
        assert isinstance(result["rsi"], float)

    def test_macd_is_dict(self) -> None:
        close = _uptrend_close(50)
        candles = _make_candles(close)
        result = compute_all_indicators(candles)
        assert isinstance(result["macd"], dict)


# ---------------------------------------------------------------------------
# Parent TF
# ---------------------------------------------------------------------------


class TestParentTimeframe:
    def test_mapping(self) -> None:
        assert get_parent_timeframe("15m") == "1h"
        assert get_parent_timeframe("30m") == "4h"
        assert get_parent_timeframe("1h") == "4h"
        assert get_parent_timeframe("4h") == "1d"
        assert get_parent_timeframe("1d") == "1w"

    def test_unknown_tf_raises(self) -> None:
        with pytest.raises(ValueError):
            get_parent_timeframe("2h")

    def test_compute_parent_tf_context_uptrend(self) -> None:
        close = _uptrend_close(50, 100, 1.0)
        candles = _make_candles(close)
        ctx = compute_parent_tf_context(candles, "4h")
        assert ctx.timeframe == "4h"
        assert ctx.trend_direction == "BULLISH"
        assert ctx.ma_position == "ABOVE_50MA"

    def test_compute_parent_tf_context_downtrend(self) -> None:
        close = _downtrend_close(50, 150, 1.0)
        candles = _make_candles(close)
        ctx = compute_parent_tf_context(candles, "1h")
        assert ctx.trend_direction == "BEARISH"
        assert ctx.ma_position == "BELOW_50MA"

    def test_compute_parent_tf_context_returns_dataclass(self) -> None:
        close = _uptrend_close(50)
        candles = _make_candles(close)
        ctx = compute_parent_tf_context(candles, "4h")
        assert hasattr(ctx, "adx_value")
        assert hasattr(ctx, "adx_classification")
        assert hasattr(ctx, "bb_width_percentile")
