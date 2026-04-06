"""Unit tests for chart generation and grounding header."""

from __future__ import annotations

import pytest

from engine.data.charts import (
    generate_candlestick_chart,
    generate_grounding_header,
    generate_trendline_chart,
)
from engine.types import FlowOutput, ParentTFContext


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _make_candles(n: int = 50, base_price: float = 100.0) -> list[dict]:
    """Generate n uptrending candle dicts."""
    candles = []
    for i in range(n):
        c = base_price + i * 0.5
        candles.append({
            "timestamp": 1700000000 + i * 3600,
            "open": c - 0.3,
            "high": c + 1.0,
            "low": c - 1.0,
            "close": c,
            "volume": 1000.0 + i * 10,
        })
    return candles


def _make_indicators() -> dict:
    """Return a realistic indicator dict matching compute_all_indicators output."""
    return {
        "rsi": 73.2,
        "macd": {
            "macd": 1.5,
            "signal": 1.2,
            "histogram": 0.3,
            "histogram_direction": "rising",
            "cross": "none",
        },
        "roc": 2.45,
        "stochastic": {"k": 82.0, "d": 78.0, "zone": "overbought"},
        "williams_r": -18.0,
        "atr": 1.25,
        "adx": {"adx": 31.0, "plus_di": 28.0, "minus_di": 14.0, "classification": "TRENDING"},
        "bollinger_bands": {
            "upper": 110.0,
            "middle": 105.0,
            "lower": 100.0,
            "width": 10.0,
            "width_percentile": 65.0,
        },
        "volume_ma": {"ma": 1000.0, "current": 1200.0, "ratio": 1.2, "spike": False},
        "volatility_percentile": 58.0,
    }


# PNG magic bytes: \x89PNG\r\n\x1a\n
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Candlestick chart tests
# ---------------------------------------------------------------------------


class TestCandlestickChart:
    def test_returns_valid_png(self) -> None:
        candles = _make_candles(50)
        result = generate_candlestick_chart(candles, "BTC-USDC", "1h")

        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result[:8] == PNG_MAGIC

    def test_with_swing_levels(self) -> None:
        candles = _make_candles(50)
        result = generate_candlestick_chart(
            candles, "BTC-USDC", "1h",
            swing_highs=[120.0, 118.0],
            swing_lows=[95.0, 97.0],
        )

        assert result[:8] == PNG_MAGIC
        assert len(result) > 0

    def test_empty_candles_returns_empty(self) -> None:
        result = generate_candlestick_chart([], "BTC-USDC", "1h")
        assert result == b""

    def test_minimal_candles(self) -> None:
        candles = _make_candles(2)
        result = generate_candlestick_chart(candles, "ETH-USDC", "4h")

        assert result[:8] == PNG_MAGIC

    def test_custom_dimensions(self) -> None:
        candles = _make_candles(20)
        result = generate_candlestick_chart(
            candles, "BTC-USDC", "1h",
            width=800, height=600,
        )

        assert result[:8] == PNG_MAGIC
        assert len(result) > 0

    def test_no_swing_levels(self) -> None:
        candles = _make_candles(30)
        result = generate_candlestick_chart(
            candles, "BTC-USDC", "1h",
            swing_highs=None, swing_lows=None,
        )

        assert result[:8] == PNG_MAGIC


# ---------------------------------------------------------------------------
# Trendline chart tests
# ---------------------------------------------------------------------------


class TestTrendlineChart:
    def test_returns_valid_png(self) -> None:
        candles = _make_candles(50)
        result = generate_trendline_chart(candles, "BTC-USDC", "1h")

        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result[:8] == PNG_MAGIC

    def test_empty_candles_returns_empty(self) -> None:
        result = generate_trendline_chart([], "BTC-USDC", "1h")
        assert result == b""

    def test_fewer_than_20_candles(self) -> None:
        candles = _make_candles(10)
        result = generate_trendline_chart(candles, "ETH-USDC", "15m")

        assert result[:8] == PNG_MAGIC

    def test_many_candles(self) -> None:
        candles = _make_candles(200)
        result = generate_trendline_chart(candles, "BTC-USDC", "1d")

        assert result[:8] == PNG_MAGIC
        assert len(result) > 0

    def test_custom_dimensions(self) -> None:
        candles = _make_candles(30)
        result = generate_trendline_chart(
            candles, "BTC-USDC", "4h",
            width=512, height=384,
        )

        assert result[:8] == PNG_MAGIC


# ---------------------------------------------------------------------------
# Grounding header tests
# ---------------------------------------------------------------------------


class TestGroundingHeader:
    def test_includes_symbol_and_timeframe(self) -> None:
        header = generate_grounding_header(
            symbol="BTC-USDC", timeframe="1h",
            indicators=_make_indicators(), flow=None, parent_tf=None,
            swing_highs=[65400.0], swing_lows=[63100.0],
            forecast_candles=3, forecast_description="~3 hours",
            num_candles=150, lookback_description="~6 days",
        )

        assert "BTC-USDC" in header
        assert "1h" in header
        assert "150 candles" in header
        assert "~6 days" in header
        assert "3 candles" in header
        assert "~3 hours" in header

    def test_includes_indicator_values(self) -> None:
        header = generate_grounding_header(
            symbol="BTC-USDC", timeframe="1h",
            indicators=_make_indicators(), flow=None, parent_tf=None,
            swing_highs=[], swing_lows=[],
            forecast_candles=3, forecast_description="~3 hours",
            num_candles=150, lookback_description="~6 days",
        )

        assert "RSI: 73.2" in header
        assert "overbought" in header
        assert "MACD" in header
        assert "histogram rising" in header
        assert "ROC: 2.45%" in header
        assert "Stochastic: 82" in header
        assert "Williams %R: -18.0" in header
        assert "ATR: 1.25" in header
        assert "ADX: 31.0" in header
        assert "TRENDING" in header
        assert "BB width percentile: 65" in header
        assert "Volume: 1.2x avg" in header
        assert "Volatility percentile: 58" in header

    def test_includes_macd_crossover(self) -> None:
        indicators = _make_indicators()
        indicators["macd"]["cross"] = "bullish_cross"

        header = generate_grounding_header(
            symbol="ETH-USDC", timeframe="4h",
            indicators=indicators, flow=None, parent_tf=None,
            swing_highs=[], swing_lows=[],
            forecast_candles=5, forecast_description="~20 hours",
            num_candles=150, lookback_description="~25 days",
        )

        assert "bullish cross" in header

    def test_handles_none_flow(self) -> None:
        header = generate_grounding_header(
            symbol="BTC-USDC", timeframe="1h",
            indicators=_make_indicators(), flow=None, parent_tf=None,
            swing_highs=[], swing_lows=[],
            forecast_candles=3, forecast_description="~3 hours",
            num_candles=150, lookback_description="~6 days",
        )

        assert "Flow:" not in header

    def test_handles_none_parent_tf(self) -> None:
        header = generate_grounding_header(
            symbol="BTC-USDC", timeframe="1h",
            indicators=_make_indicators(), flow=None, parent_tf=None,
            swing_highs=[], swing_lows=[],
            forecast_candles=3, forecast_description="~3 hours",
            num_candles=150, lookback_description="~6 days",
        )

        assert "Parent TF" not in header

    def test_includes_flow_data(self) -> None:
        flow = FlowOutput(
            funding_rate=0.042,
            funding_signal="CROWDED_LONG",
            oi_change_4h=8.2,
            oi_trend="BUILDING",
            nearest_liquidation_above={"price": 66000, "size": 50000000},
            nearest_liquidation_below={"price": 62000, "size": 30000000},
            gex_regime="POSITIVE_GAMMA",
            gex_flip_level=64500.0,
            data_richness="FULL",
        )

        header = generate_grounding_header(
            symbol="BTC-USDC", timeframe="1h",
            indicators=_make_indicators(), flow=flow, parent_tf=None,
            swing_highs=[], swing_lows=[],
            forecast_candles=3, forecast_description="~3 hours",
            num_candles=150, lookback_description="~6 days",
        )

        assert "Flow:" in header
        assert "Funding rate:" in header
        assert "CROWDED_LONG" in header
        assert "OI change:" in header
        assert "BUILDING" in header
        assert "POSITIVE_GAMMA" in header

    def test_includes_parent_tf(self) -> None:
        parent = ParentTFContext(
            timeframe="4h",
            trend_direction="BEARISH",
            ma_position="BELOW_50MA",
            adx_value=31.0,
            adx_classification="TRENDING",
            bb_width_percentile=72.0,
        )

        header = generate_grounding_header(
            symbol="BTC-USDC", timeframe="1h",
            indicators=_make_indicators(), flow=None, parent_tf=parent,
            swing_highs=[], swing_lows=[],
            forecast_candles=3, forecast_description="~3 hours",
            num_candles=150, lookback_description="~6 days",
        )

        assert "Parent TF (4h)" in header
        assert "BEARISH" in header
        assert "below 50ma" in header
        assert "ADX 31" in header

    def test_includes_swing_levels(self) -> None:
        header = generate_grounding_header(
            symbol="BTC-USDC", timeframe="1h",
            indicators=_make_indicators(), flow=None, parent_tf=None,
            swing_highs=[65400.0, 66200.0],
            swing_lows=[63100.0, 62500.0],
            forecast_candles=3, forecast_description="~3 hours",
            num_candles=150, lookback_description="~6 days",
        )

        assert "Nearest resistance:" in header
        assert "$65,400.00" in header
        assert "Nearest support:" in header
        assert "$63,100.00" in header

    def test_empty_indicators(self) -> None:
        header = generate_grounding_header(
            symbol="BTC-USDC", timeframe="1h",
            indicators={}, flow=None, parent_tf=None,
            swing_highs=[], swing_lows=[],
            forecast_candles=3, forecast_description="~3 hours",
            num_candles=150, lookback_description="~6 days",
        )

        assert "CONTEXT" in header
        assert "BTC-USDC" in header

    def test_context_header_starts_with_context_line(self) -> None:
        header = generate_grounding_header(
            symbol="BTC-USDC", timeframe="1h",
            indicators=_make_indicators(), flow=None, parent_tf=None,
            swing_highs=[], swing_lows=[],
            forecast_candles=3, forecast_description="~3 hours",
            num_candles=150, lookback_description="~6 days",
        )

        assert header.startswith("CONTEXT (do not override with visual impression):")

    def test_volume_spike_flagged(self) -> None:
        indicators = _make_indicators()
        indicators["volume_ma"]["spike"] = True
        indicators["volume_ma"]["ratio"] = 3.5

        header = generate_grounding_header(
            symbol="BTC-USDC", timeframe="1h",
            indicators=indicators, flow=None, parent_tf=None,
            swing_highs=[], swing_lows=[],
            forecast_candles=3, forecast_description="~3 hours",
            num_candles=150, lookback_description="~6 days",
        )

        assert "SPIKE" in header
        assert "3.5x avg" in header

    def test_rsi_oversold_label(self) -> None:
        indicators = _make_indicators()
        indicators["rsi"] = 22.5

        header = generate_grounding_header(
            symbol="BTC-USDC", timeframe="1h",
            indicators=indicators, flow=None, parent_tf=None,
            swing_highs=[], swing_lows=[],
            forecast_candles=3, forecast_description="~3 hours",
            num_candles=150, lookback_description="~6 days",
        )

        assert "RSI: 22.5 (oversold)" in header
