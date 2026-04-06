"""Readiness conditions: indicator cross, level touch, volume anomaly, etc.

All computation is local and deterministic — zero LLM calls.
Each condition returns a bool + weight. The ReadinessScorer sums
active conditions into a 0-1 readiness score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReadinessCondition:
    """A single readiness condition result."""

    name: str
    triggered: bool
    weight: float
    detail: str


class ReadinessScorer:
    """Evaluates 5 weighted readiness conditions from indicator data.

    Conditions:
    1. RSI threshold cross (30/70)     — weight 0.25
    2. Key level touch (BB/swing)      — weight 0.30
    3. Volume anomaly (>3x average)    — weight 0.20
    4. Flow shift (funding > |0.01%|)  — weight 0.15
    5. MACD cross (histogram sign flip) — weight 0.10
    """

    def score(
        self,
        indicators: dict,
        current_price: float,
        swing_highs: list[float],
        swing_lows: list[float],
        funding_rate: float | None = None,
        prev_macd_histogram: float | None = None,
    ) -> tuple[float, list[ReadinessCondition]]:
        """Compute readiness score from indicators and context.

        Args:
            indicators: Dict from compute_all_indicators().
            current_price: Latest close price.
            swing_highs: Nearest swing high levels.
            swing_lows: Nearest swing low levels.
            funding_rate: Current funding rate (None if unavailable).
            prev_macd_histogram: Previous candle's MACD histogram (for cross detection).

        Returns:
            (score, conditions) where score is 0-1 and conditions list
            shows which fired.
        """
        conditions = [
            self._check_rsi_cross(indicators),
            self._check_level_touch(indicators, current_price, swing_highs, swing_lows),
            self._check_volume_anomaly(indicators),
            self._check_flow_shift(funding_rate),
            self._check_macd_cross(indicators, prev_macd_histogram),
        ]

        total = sum(c.weight for c in conditions if c.triggered)
        score = min(1.0, max(0.0, total))

        return score, conditions

    def _check_rsi_cross(self, indicators: dict) -> ReadinessCondition:
        """RSI crossing 30 (oversold) or 70 (overbought)."""
        rsi = indicators.get("rsi")
        if rsi is None:
            return ReadinessCondition("rsi_cross", False, 0.25, "RSI not available")

        if rsi <= 30:
            return ReadinessCondition("rsi_cross", True, 0.25, f"RSI {rsi:.1f} <= 30 (oversold)")
        if rsi >= 70:
            return ReadinessCondition("rsi_cross", True, 0.25, f"RSI {rsi:.1f} >= 70 (overbought)")

        return ReadinessCondition("rsi_cross", False, 0.25, f"RSI {rsi:.1f} (neutral)")

    def _check_level_touch(
        self,
        indicators: dict,
        price: float,
        swing_highs: list[float],
        swing_lows: list[float],
    ) -> ReadinessCondition:
        """Price near Bollinger Band edge or swing level (within 0.3% of price)."""
        bb = indicators.get("bollinger_bands", {})
        atr = indicators.get("atr", 0)
        threshold = price * 0.003 if price > 0 else 1.0  # 0.3% of price

        # Check BB touch
        bb_upper = bb.get("upper", 0)
        bb_lower = bb.get("lower", 0)
        if bb_upper and abs(price - bb_upper) <= threshold:
            return ReadinessCondition(
                "level_touch", True, 0.30,
                f"Price {price:.2f} near BB upper {bb_upper:.2f}",
            )
        if bb_lower and abs(price - bb_lower) <= threshold:
            return ReadinessCondition(
                "level_touch", True, 0.30,
                f"Price {price:.2f} near BB lower {bb_lower:.2f}",
            )

        # Check swing level touch
        for level in swing_highs[:3]:
            if abs(price - level) <= threshold:
                return ReadinessCondition(
                    "level_touch", True, 0.30,
                    f"Price {price:.2f} near swing high {level:.2f}",
                )
        for level in swing_lows[:3]:
            if abs(price - level) <= threshold:
                return ReadinessCondition(
                    "level_touch", True, 0.30,
                    f"Price {price:.2f} near swing low {level:.2f}",
                )

        return ReadinessCondition("level_touch", False, 0.30, "Not near key level")

    def _check_volume_anomaly(self, indicators: dict) -> ReadinessCondition:
        """Volume > 3x moving average."""
        vol = indicators.get("volume_ma", {})
        ratio = vol.get("ratio", 0)

        if ratio >= 3.0:
            return ReadinessCondition(
                "volume_anomaly", True, 0.20,
                f"Volume {ratio:.1f}x average (spike)",
            )

        return ReadinessCondition(
            "volume_anomaly", False, 0.20,
            f"Volume {ratio:.1f}x average (normal)",
        )

    def _check_flow_shift(self, funding_rate: float | None) -> ReadinessCondition:
        """Funding rate magnitude > 0.01% indicates positioning pressure."""
        if funding_rate is None:
            return ReadinessCondition("flow_shift", False, 0.15, "Funding rate unavailable")

        if abs(funding_rate) > 0.0001:  # 0.01%
            direction = "long-heavy" if funding_rate > 0 else "short-heavy"
            return ReadinessCondition(
                "flow_shift", True, 0.15,
                f"Funding {funding_rate:.4%} ({direction})",
            )

        return ReadinessCondition(
            "flow_shift", False, 0.15,
            f"Funding {funding_rate:.4%} (neutral)",
        )

    def _check_macd_cross(
        self, indicators: dict, prev_histogram: float | None
    ) -> ReadinessCondition:
        """MACD histogram sign flip (cross)."""
        macd = indicators.get("macd", {})
        histogram = macd.get("histogram")

        if histogram is None or prev_histogram is None:
            return ReadinessCondition("macd_cross", False, 0.10, "MACD histogram unavailable")

        if (prev_histogram < 0 and histogram >= 0):
            return ReadinessCondition(
                "macd_cross", True, 0.10,
                f"MACD bullish cross (histogram {prev_histogram:.4f} -> {histogram:.4f})",
            )
        if (prev_histogram > 0 and histogram <= 0):
            return ReadinessCondition(
                "macd_cross", True, 0.10,
                f"MACD bearish cross (histogram {prev_histogram:.4f} -> {histogram:.4f})",
            )

        return ReadinessCondition(
            "macd_cross", False, 0.10,
            f"No MACD cross (histogram {histogram:.4f})",
        )
