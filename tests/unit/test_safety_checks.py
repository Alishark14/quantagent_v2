"""Unit tests for mechanical safety checks."""

from __future__ import annotations

import pytest

from engine.execution.safety_checks import SafetyCheckResult, run_safety_checks
from engine.types import Position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _long_position() -> Position:
    return Position(
        symbol="BTC-USDC", direction="long",
        size=0.1, entry_price=100.0,
        unrealized_pnl=5.0, leverage=10.0,
    )


def _short_position() -> Position:
    return Position(
        symbol="BTC-USDC", direction="short",
        size=0.1, entry_price=100.0,
        unrealized_pnl=-2.0, leverage=10.0,
    )


def _safe_defaults(**overrides) -> dict:
    """Default args that pass all checks."""
    defaults = {
        "action": "LONG",
        "current_position": None,
        "daily_pnl": 0.0,
        "max_daily_loss": -500.0,
        "swing_highs": [110.0, 115.0],
        "swing_lows": [90.0, 85.0],
        "atr": 2.0,
        "conviction_score": 0.7,
        "entry_price": 100.0,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# All checks pass
# ---------------------------------------------------------------------------

class TestAllPass:
    def test_long_all_clear(self) -> None:
        result = run_safety_checks(**_safe_defaults())
        assert result.passed is True
        assert result.adjusted_action == "LONG"
        assert result.violations == []

    def test_short_all_clear(self) -> None:
        result = run_safety_checks(**_safe_defaults(action="SHORT"))
        assert result.passed is True
        assert result.adjusted_action == "SHORT"

    def test_hold_always_passes(self) -> None:
        result = run_safety_checks(**_safe_defaults(action="HOLD", conviction_score=0.1))
        assert result.passed is True
        assert result.adjusted_action == "HOLD"

    def test_close_all_always_passes(self) -> None:
        result = run_safety_checks(**_safe_defaults(action="CLOSE_ALL", conviction_score=0.1))
        assert result.passed is True
        assert result.adjusted_action == "CLOSE_ALL"

    def test_skip_always_passes(self) -> None:
        result = run_safety_checks(**_safe_defaults(action="SKIP", conviction_score=0.1))
        assert result.passed is True
        assert result.adjusted_action == "SKIP"


# ---------------------------------------------------------------------------
# Check 1: Conviction floor
# ---------------------------------------------------------------------------

class TestConvictionFloor:
    def test_low_conviction_forces_skip(self) -> None:
        result = run_safety_checks(**_safe_defaults(conviction_score=0.2))
        assert result.passed is False
        assert result.adjusted_action == "SKIP"
        assert any("conviction_floor" in v for v in result.violations)

    def test_conviction_exactly_0_3_passes(self) -> None:
        # 0.3 is NOT < 0.3, so it should pass
        result = run_safety_checks(**_safe_defaults(conviction_score=0.3))
        assert result.adjusted_action == "LONG"

    def test_conviction_0_29_fails(self) -> None:
        result = run_safety_checks(**_safe_defaults(conviction_score=0.29))
        assert result.adjusted_action == "SKIP"

    def test_hold_exempt_from_conviction_floor(self) -> None:
        result = run_safety_checks(**_safe_defaults(action="HOLD", conviction_score=0.1))
        assert result.adjusted_action == "HOLD"

    def test_close_all_exempt_from_conviction_floor(self) -> None:
        result = run_safety_checks(**_safe_defaults(action="CLOSE_ALL", conviction_score=0.1))
        assert result.adjusted_action == "CLOSE_ALL"

    def test_add_long_blocked_by_low_conviction(self) -> None:
        result = run_safety_checks(**_safe_defaults(
            action="ADD_LONG", conviction_score=0.2,
            current_position=_long_position(),
        ))
        assert result.adjusted_action == "SKIP"


# ---------------------------------------------------------------------------
# Check 2: Daily loss limit
# ---------------------------------------------------------------------------

class TestDailyLossLimit:
    def test_loss_exceeded_blocks_long(self) -> None:
        result = run_safety_checks(**_safe_defaults(
            daily_pnl=-600.0, max_daily_loss=-500.0,
        ))
        assert result.passed is False
        assert result.adjusted_action == "SKIP"
        assert any("daily_loss_limit" in v for v in result.violations)

    def test_loss_exactly_at_limit_blocks(self) -> None:
        result = run_safety_checks(**_safe_defaults(
            daily_pnl=-500.0, max_daily_loss=-500.0,
        ))
        assert result.adjusted_action == "SKIP"

    def test_loss_above_limit_allows(self) -> None:
        result = run_safety_checks(**_safe_defaults(
            daily_pnl=-499.0, max_daily_loss=-500.0,
        ))
        assert result.adjusted_action == "LONG"

    def test_loss_exceeded_allows_hold(self) -> None:
        result = run_safety_checks(**_safe_defaults(
            action="HOLD", daily_pnl=-600.0, max_daily_loss=-500.0,
        ))
        assert result.adjusted_action == "HOLD"

    def test_loss_exceeded_allows_close_all(self) -> None:
        result = run_safety_checks(**_safe_defaults(
            action="CLOSE_ALL", daily_pnl=-600.0, max_daily_loss=-500.0,
        ))
        assert result.adjusted_action == "CLOSE_ALL"

    def test_add_long_converts_to_hold_on_loss(self) -> None:
        result = run_safety_checks(**_safe_defaults(
            action="ADD_LONG", daily_pnl=-600.0, max_daily_loss=-500.0,
            current_position=_long_position(),
        ))
        assert result.adjusted_action == "HOLD"


# ---------------------------------------------------------------------------
# Check 3: Pyramid distance-to-resistance gate
# ---------------------------------------------------------------------------

class TestPyramidGate:
    def test_add_long_near_resistance_blocked(self) -> None:
        # Nearest swing high = 100.5, entry = 100.0, distance = 0.5
        # threshold = 0.3 * 2.0 = 0.6 → 0.5 <= 0.6 → HOLD
        result = run_safety_checks(**_safe_defaults(
            action="ADD_LONG", entry_price=100.0,
            swing_highs=[100.5, 110.0], atr=2.0,
            current_position=_long_position(),
        ))
        assert result.adjusted_action == "HOLD"
        assert any("pyramid_gate" in v for v in result.violations)

    def test_add_long_far_from_resistance_allowed(self) -> None:
        # Nearest swing high = 105.0, distance = 5.0 > threshold 0.6
        result = run_safety_checks(**_safe_defaults(
            action="ADD_LONG", entry_price=100.0,
            swing_highs=[105.0, 110.0], atr=2.0,
            current_position=_long_position(),
        ))
        assert result.adjusted_action == "ADD_LONG"

    def test_add_short_near_support_blocked(self) -> None:
        # Nearest swing low (max) = 99.5, entry = 100.0, distance = 0.5
        # threshold = 0.3 * 2.0 = 0.6 → 0.5 <= 0.6 → HOLD
        result = run_safety_checks(**_safe_defaults(
            action="ADD_SHORT", entry_price=100.0,
            swing_lows=[99.5, 90.0], atr=2.0,
            current_position=_short_position(),
        ))
        assert result.adjusted_action == "HOLD"

    def test_add_short_far_from_support_allowed(self) -> None:
        result = run_safety_checks(**_safe_defaults(
            action="ADD_SHORT", entry_price=100.0,
            swing_lows=[95.0, 90.0], atr=2.0,
            current_position=_short_position(),
        ))
        assert result.adjusted_action == "ADD_SHORT"

    def test_regular_long_not_affected_by_pyramid_gate(self) -> None:
        # Pyramid gate only applies to ADD_LONG/ADD_SHORT
        result = run_safety_checks(**_safe_defaults(
            action="LONG", entry_price=100.0,
            swing_highs=[100.5], atr=2.0,
        ))
        assert result.adjusted_action == "LONG"


# ---------------------------------------------------------------------------
# Check 4: Position count limit
# ---------------------------------------------------------------------------

class TestPositionLimit:
    def test_long_blocked_when_position_exists(self) -> None:
        result = run_safety_checks(**_safe_defaults(
            action="LONG", current_position=_long_position(),
        ))
        assert result.adjusted_action == "HOLD"
        assert any("position_limit" in v for v in result.violations)

    def test_short_blocked_when_position_exists(self) -> None:
        result = run_safety_checks(**_safe_defaults(
            action="SHORT", current_position=_short_position(),
        ))
        assert result.adjusted_action == "HOLD"

    def test_long_allowed_when_no_position(self) -> None:
        result = run_safety_checks(**_safe_defaults(
            action="LONG", current_position=None,
        ))
        assert result.adjusted_action == "LONG"

    def test_add_long_not_blocked_by_position_limit(self) -> None:
        # ADD actions expect a position to exist — not blocked by check 4
        result = run_safety_checks(**_safe_defaults(
            action="ADD_LONG", current_position=_long_position(),
        ))
        assert result.adjusted_action == "ADD_LONG"


# ---------------------------------------------------------------------------
# Check 5: SL validation (valid ATR)
# ---------------------------------------------------------------------------

class TestSlValidation:
    def test_zero_atr_blocks_entry(self) -> None:
        result = run_safety_checks(**_safe_defaults(atr=0.0))
        assert result.adjusted_action == "SKIP"
        assert any("sl_validation" in v for v in result.violations)

    def test_negative_atr_blocks_entry(self) -> None:
        result = run_safety_checks(**_safe_defaults(atr=-1.0))
        assert result.adjusted_action == "SKIP"

    def test_positive_atr_allows(self) -> None:
        result = run_safety_checks(**_safe_defaults(atr=0.5))
        assert result.adjusted_action == "LONG"


# ---------------------------------------------------------------------------
# Multiple violations
# ---------------------------------------------------------------------------

class TestMultipleViolations:
    def test_multiple_violations_reported(self) -> None:
        result = run_safety_checks(**_safe_defaults(
            action="LONG",
            conviction_score=0.1,
            daily_pnl=-600.0, max_daily_loss=-500.0,
            current_position=_long_position(),
        ))
        assert result.passed is False
        # conviction_floor sets SKIP, daily_loss sets SKIP, position_limit sets HOLD
        # Final result depends on check order — last writer wins
        assert len(result.violations) >= 2
        assert result.adjusted_action == "HOLD"  # position_limit is last to fire


# ---------------------------------------------------------------------------
# SafetyCheckResult
# ---------------------------------------------------------------------------

class TestSafetyCheckResult:
    def test_to_dict(self) -> None:
        result = SafetyCheckResult(
            passed=False,
            original_action="LONG",
            adjusted_action="SKIP",
            violations=["conviction_floor: 0.2 < 0.3"],
        )
        d = result.to_dict()
        assert d["passed"] is False
        assert d["original_action"] == "LONG"
        assert d["adjusted_action"] == "SKIP"
        assert len(d["violations"]) == 1
