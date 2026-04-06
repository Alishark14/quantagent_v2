"""Unit tests for risk profiles: SL/TP computation and position sizing."""

from __future__ import annotations

import pytest

from engine.config import TimeframeProfile
from engine.execution.risk_profiles import compute_position_size, compute_sl_tp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _1h_profile() -> TimeframeProfile:
    """Standard 1h profile: ATR mult 1.5, RR 1.5-2.0."""
    return TimeframeProfile(
        timeframe="1h", candles=150,
        atr_multiplier=1.5, rr_min=1.5, rr_max=2.0,
        trailing_enabled=False,
    )


def _4h_profile() -> TimeframeProfile:
    return TimeframeProfile(
        timeframe="4h", candles=150,
        atr_multiplier=1.0, rr_min=3.0, rr_max=5.0,
        trailing_enabled=True,
    )


# ---------------------------------------------------------------------------
# compute_sl_tp: LONG
# ---------------------------------------------------------------------------

class TestComputeSlTpLong:
    def test_basic_long_atr_based(self) -> None:
        result = compute_sl_tp(
            entry_price=100.0, direction="LONG", atr=2.0,
            profile=_1h_profile(), swing_highs=[], swing_lows=[],
        )
        # SL = 100 - (2.0 * 1.5) = 97.0
        assert result["sl_price"] == pytest.approx(97.0)
        # Risk distance = 3.0
        assert result["risk_distance"] == pytest.approx(3.0)
        # TP1 = 100 + 3.0 = 103.0 (1:1)
        assert result["tp1_price"] == pytest.approx(103.0)
        # TP2 = 100 + 3.0 * 1.5 = 104.5
        assert result["tp2_price"] == pytest.approx(104.5)
        assert result["rr_ratio"] == pytest.approx(1.5)

    def test_long_snaps_to_swing_low(self) -> None:
        # ATR-based SL = 100 - 3.0 = 97.0
        # Swing low at 98.0 is above 97.0 (tighter) and far enough from entry
        result = compute_sl_tp(
            entry_price=100.0, direction="LONG", atr=2.0,
            profile=_1h_profile(), swing_highs=[], swing_lows=[98.0, 95.0],
        )
        assert result["sl_price"] == pytest.approx(98.0)
        assert result["risk_distance"] == pytest.approx(2.0)

    def test_long_ignores_swing_too_close(self) -> None:
        # Swing low at 99.9 is only 0.1 away from entry
        # min_distance = 0.2 * 2.0 = 0.4, so 0.1 < 0.4 → ignored
        result = compute_sl_tp(
            entry_price=100.0, direction="LONG", atr=2.0,
            profile=_1h_profile(), swing_highs=[], swing_lows=[99.9],
        )
        assert result["sl_price"] == pytest.approx(97.0)

    def test_long_ignores_swing_below_atr_sl(self) -> None:
        # Swing low at 95.0 is below ATR SL 97.0 → wider stop, not tighter
        result = compute_sl_tp(
            entry_price=100.0, direction="LONG", atr=2.0,
            profile=_1h_profile(), swing_highs=[], swing_lows=[95.0],
        )
        assert result["sl_price"] == pytest.approx(97.0)

    def test_long_picks_tightest_valid_swing(self) -> None:
        # ATR SL = 97.0. Swing lows: 98.5 (valid, tightest), 97.5 (valid), 95 (invalid)
        result = compute_sl_tp(
            entry_price=100.0, direction="LONG", atr=2.0,
            profile=_1h_profile(), swing_highs=[], swing_lows=[98.5, 97.5, 95.0],
        )
        assert result["sl_price"] == pytest.approx(98.5)

    def test_long_4h_profile(self) -> None:
        # ATR mult 1.0 → SL = 100 - 2.0 = 98.0
        result = compute_sl_tp(
            entry_price=100.0, direction="LONG", atr=2.0,
            profile=_4h_profile(), swing_highs=[], swing_lows=[],
        )
        assert result["sl_price"] == pytest.approx(98.0)
        assert result["tp2_price"] == pytest.approx(106.0)  # 100 + 2.0 * 3.0
        assert result["rr_ratio"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# compute_sl_tp: SHORT
# ---------------------------------------------------------------------------

class TestComputeSlTpShort:
    def test_basic_short_atr_based(self) -> None:
        result = compute_sl_tp(
            entry_price=100.0, direction="SHORT", atr=2.0,
            profile=_1h_profile(), swing_highs=[], swing_lows=[],
        )
        # SL = 100 + 3.0 = 103.0
        assert result["sl_price"] == pytest.approx(103.0)
        assert result["risk_distance"] == pytest.approx(3.0)
        # TP1 = 100 - 3.0 = 97.0
        assert result["tp1_price"] == pytest.approx(97.0)
        # TP2 = 100 - 3.0 * 1.5 = 95.5
        assert result["tp2_price"] == pytest.approx(95.5)

    def test_short_snaps_to_swing_high(self) -> None:
        # ATR SL = 103.0. Swing high at 101.5 is below 103 (tighter)
        result = compute_sl_tp(
            entry_price=100.0, direction="SHORT", atr=2.0,
            profile=_1h_profile(), swing_highs=[101.5, 105.0], swing_lows=[],
        )
        assert result["sl_price"] == pytest.approx(101.5)
        assert result["risk_distance"] == pytest.approx(1.5)

    def test_short_ignores_swing_too_close(self) -> None:
        # Swing high at 100.1, distance = 0.1 < min_distance 0.4
        result = compute_sl_tp(
            entry_price=100.0, direction="SHORT", atr=2.0,
            profile=_1h_profile(), swing_highs=[100.1], swing_lows=[],
        )
        assert result["sl_price"] == pytest.approx(103.0)

    def test_short_ignores_swing_above_atr_sl(self) -> None:
        # Swing high at 105.0 above ATR SL 103.0 → wider, not tighter
        result = compute_sl_tp(
            entry_price=100.0, direction="SHORT", atr=2.0,
            profile=_1h_profile(), swing_highs=[105.0], swing_lows=[],
        )
        assert result["sl_price"] == pytest.approx(103.0)


# ---------------------------------------------------------------------------
# compute_sl_tp: edge cases
# ---------------------------------------------------------------------------

class TestComputeSlTpEdge:
    def test_invalid_direction_returns_zeros(self) -> None:
        result = compute_sl_tp(
            entry_price=100.0, direction="SKIP", atr=2.0,
            profile=_1h_profile(), swing_highs=[], swing_lows=[],
        )
        assert result["sl_price"] == 0.0
        assert result["tp1_price"] == 0.0

    def test_empty_swings(self) -> None:
        result = compute_sl_tp(
            entry_price=100.0, direction="LONG", atr=2.0,
            profile=_1h_profile(), swing_highs=[], swing_lows=[],
        )
        assert result["sl_price"] == pytest.approx(97.0)


# ---------------------------------------------------------------------------
# compute_position_size
# ---------------------------------------------------------------------------

class TestComputePositionSize:
    def test_basic_sizing(self) -> None:
        # Balance 10000, risk 1%, entry 100, SL 97 → 3% SL distance
        # Risk amount = 100. Size = 100 / 0.03 = 3333.33
        size = compute_position_size(
            account_balance=10000.0, risk_per_trade=0.01,
            entry_price=100.0, sl_price=97.0,
        )
        assert size == pytest.approx(3333.33, rel=0.01)

    def test_capped_at_max_position(self) -> None:
        # Very tight SL → huge size, should be capped
        size = compute_position_size(
            account_balance=10000.0, risk_per_trade=0.05,
            entry_price=100.0, sl_price=99.99,  # 0.01% SL
            max_position_pct=1.0,
        )
        assert size <= 10000.0

    def test_floored_at_min_size(self) -> None:
        # Tiny balance → computed size small, floored at min but then capped at max_position
        # balance=10, risk=1%, SL=3% → size = 3.33, floor to 20 then cap at 10
        size = compute_position_size(
            account_balance=10.0, risk_per_trade=0.01,
            entry_price=100.0, sl_price=97.0,
        )
        assert size == 10.0  # min_size_usd=20 but capped by max_position=10

    def test_custom_min_size(self) -> None:
        size = compute_position_size(
            account_balance=10.0, risk_per_trade=0.01,
            entry_price=100.0, sl_price=97.0,
            min_size_usd=5.0,
        )
        assert size == pytest.approx(5.0)

    def test_zero_balance_returns_zero(self) -> None:
        size = compute_position_size(
            account_balance=0.0, risk_per_trade=0.01,
            entry_price=100.0, sl_price=97.0,
        )
        assert size == 0.0

    def test_zero_sl_distance_returns_zero(self) -> None:
        size = compute_position_size(
            account_balance=10000.0, risk_per_trade=0.01,
            entry_price=100.0, sl_price=100.0,
        )
        assert size == 0.0

    def test_short_sl_above_entry(self) -> None:
        # For short: SL above entry is correct
        size = compute_position_size(
            account_balance=10000.0, risk_per_trade=0.01,
            entry_price=100.0, sl_price=103.0,
        )
        assert size == pytest.approx(3333.33, rel=0.01)

    def test_max_position_pct_limits(self) -> None:
        # max_position_pct = 0.5 → max position = 5000
        size = compute_position_size(
            account_balance=10000.0, risk_per_trade=0.05,
            entry_price=100.0, sl_price=99.5,  # 0.5% SL → large position
            max_position_pct=0.5,
        )
        assert size <= 5000.0
