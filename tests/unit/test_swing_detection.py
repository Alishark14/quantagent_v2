"""Unit tests for engine/data/swing_detection.py — detailed edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from engine.data.swing_detection import (
    adjust_sl_to_structure,
    find_swing_highs,
    find_swing_lows,
)


# ---------------------------------------------------------------------------
# find_swing_highs
# ---------------------------------------------------------------------------


class TestFindSwingHighs:
    def test_simple_peak(self) -> None:
        # Clear peak at index 2
        high = np.array([10.0, 11.0, 15.0, 11.0, 10.0])
        swings = find_swing_highs(high, lookback=5, num_swings=3)
        assert 15.0 in swings

    def test_multiple_peaks(self) -> None:
        high = np.array([10, 15, 10, 20, 10, 18, 10], dtype=float)
        swings = find_swing_highs(high, lookback=7, num_swings=3)
        assert len(swings) == 3
        assert set(swings) == {15.0, 20.0, 18.0}

    def test_respects_num_swings(self) -> None:
        high = np.array([10, 15, 10, 20, 10, 18, 10, 16, 10], dtype=float)
        swings = find_swing_highs(high, lookback=9, num_swings=2)
        assert len(swings) == 2

    def test_sorted_by_proximity_to_current(self) -> None:
        # Current price (last high) = 10. Peaks at 15, 20, 12.
        # Nearest to 10: 12, 15, 20
        high = np.array([10, 12, 10, 15, 10, 20, 10], dtype=float)
        swings = find_swing_highs(high, lookback=7, num_swings=3)
        assert swings[0] == 12.0  # nearest to current (10)

    def test_no_peaks_returns_empty(self) -> None:
        high = np.array([10, 11, 12, 13, 14], dtype=float)  # monotonic up, no peak
        swings = find_swing_highs(high, lookback=5, num_swings=3)
        assert swings == []

    def test_lookback_limits_range(self) -> None:
        # Peak at idx 1 is outside lookback=3 (last 3 elements)
        high = np.array([10, 20, 10, 5, 6, 5], dtype=float)
        swings = find_swing_highs(high, lookback=3, num_swings=3)
        # Only looking at [5, 6, 5] => peak at 6
        assert 6.0 in swings
        assert 20.0 not in swings

    def test_flat_prices_no_swings(self) -> None:
        high = np.full(10, 100.0)
        swings = find_swing_highs(high, lookback=10, num_swings=3)
        assert swings == []

    def test_minimum_data(self) -> None:
        # Only 3 bars, one possible pivot
        high = np.array([10.0, 15.0, 10.0])
        swings = find_swing_highs(high, lookback=3, num_swings=3)
        assert swings == [15.0]


# ---------------------------------------------------------------------------
# find_swing_lows
# ---------------------------------------------------------------------------


class TestFindSwingLows:
    def test_simple_trough(self) -> None:
        low = np.array([20.0, 15.0, 10.0, 15.0, 20.0])
        swings = find_swing_lows(low, lookback=5, num_swings=3)
        assert 10.0 in swings

    def test_multiple_troughs(self) -> None:
        low = np.array([20, 10, 20, 5, 20, 8, 20], dtype=float)
        swings = find_swing_lows(low, lookback=7, num_swings=3)
        assert len(swings) == 3
        assert set(swings) == {10.0, 5.0, 8.0}

    def test_no_troughs_returns_empty(self) -> None:
        low = np.array([20, 19, 18, 17, 16], dtype=float)  # monotonic down
        swings = find_swing_lows(low, lookback=5, num_swings=3)
        assert swings == []

    def test_sorted_by_proximity_to_current(self) -> None:
        # Current (last low) = 20. Troughs at 10, 5, 18.
        low = np.array([20, 10, 20, 5, 20, 18, 20], dtype=float)
        swings = find_swing_lows(low, lookback=7, num_swings=3)
        assert swings[0] == 18.0  # nearest to 20


# ---------------------------------------------------------------------------
# adjust_sl_to_structure
# ---------------------------------------------------------------------------


class TestAdjustSLToStructure:
    def test_long_snaps_to_swing_low(self) -> None:
        sl = 100.0
        atr = 10.0
        swing_lows = [99.5]  # within 15% of ATR (1.5) from SL
        adjusted = adjust_sl_to_structure(sl, "LONG", [], swing_lows, atr)
        expected = 99.5 * (1 - 0.002)
        assert adjusted == pytest.approx(expected)

    def test_short_snaps_to_swing_high(self) -> None:
        sl = 110.0
        atr = 10.0
        swing_highs = [110.5]  # within 15% of ATR from SL
        adjusted = adjust_sl_to_structure(sl, "SHORT", swing_highs, [], atr)
        expected = 110.5 * (1 + 0.002)
        assert adjusted == pytest.approx(expected)

    def test_no_nearby_swing_returns_original(self) -> None:
        sl = 100.0
        atr = 10.0
        swing_lows = [85.0]  # way too far (15 away, threshold is 1.5)
        adjusted = adjust_sl_to_structure(sl, "LONG", [], swing_lows, atr)
        assert adjusted == 100.0

    def test_custom_buffer(self) -> None:
        sl = 100.0
        atr = 10.0
        swing_lows = [100.5]
        adjusted = adjust_sl_to_structure(sl, "LONG", [], swing_lows, atr, buffer_pct=0.01)
        expected = 100.5 * (1 - 0.01)
        assert adjusted == pytest.approx(expected)

    def test_empty_swings_returns_original(self) -> None:
        adjusted = adjust_sl_to_structure(100.0, "LONG", [], [], 10.0)
        assert adjusted == 100.0

    def test_picks_nearest_swing(self) -> None:
        sl = 100.0
        atr = 10.0
        # Both within threshold (1.5). 100.3 is closer to 100 than 99.0.
        swing_lows = [99.0, 100.3]
        adjusted = adjust_sl_to_structure(sl, "LONG", [], swing_lows, atr)
        expected = 100.3 * (1 - 0.002)
        assert adjusted == pytest.approx(expected)
