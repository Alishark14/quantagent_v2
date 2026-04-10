"""CryptoFlowProvider: funding rate, OI, liquidation data.

Fetches crypto-specific positioning data from the exchange adapter.
Code-only, zero LLM cost. All data is factual/deterministic.

OI history buffer
=================

The provider maintains a per-symbol in-memory deque of
``(timestamp_seconds, oi_value)`` snapshots. On each ``fetch()`` call
the current OI is appended. When the buffer spans ≥ 4 hours the
provider computes ``oi_change_4h`` (fractional delta) and derives
``oi_trend`` ("BUILDING" / "DROPPING" / "STABLE"). During the cold-
start warmup window (< 4h of data) both fields stay ``None`` /
``"STABLE"`` — ``FlowSignalAgent`` handles ``None`` correctly by
falling through to NEUTRAL.

The buffer is keyed by symbol so a single provider instance can serve
multiple bots (or a multi-symbol bot) without cross-contamination.
``maxlen=480`` at 30-second Sentinel poll intervals covers exactly 4
hours. Longer poll intervals just mean a longer warmup.
"""

from __future__ import annotations

import logging
import time
from collections import deque

from engine.data.flow.base import FlowProvider
from exchanges.base import ExchangeAdapter

logger = logging.getLogger(__name__)

# Funding rate thresholds for signal classification
_CROWDED_LONG_THRESHOLD = 0.01   # > +0.01% = crowded long
_CROWDED_SHORT_THRESHOLD = -0.01  # < -0.01% = crowded short

# OI history buffer config
_OI_BUFFER_MAXLEN = 480        # 480 × 30s = 4 hours
_OI_LOOKBACK_SECONDS = 14_400  # 4 hours in seconds
_OI_TREND_THRESHOLD = 0.02     # ±2% to classify BUILDING / DROPPING


class CryptoFlowProvider(FlowProvider):
    """Fetches funding rate and open interest from crypto exchange adapters.

    Maintains a per-symbol OI history buffer so ``oi_change_4h`` and
    ``oi_trend`` can be computed from live data rather than being
    hardcoded to ``None`` / ``"STABLE"``.
    """

    def __init__(self) -> None:
        # symbol → deque of (timestamp_seconds, oi_value)
        self._oi_history: dict[str, deque[tuple[float, float]]] = {}
        # Track which symbols have logged the "buffer warm" message
        self._oi_warm_logged: set[str] = set()

    def name(self) -> str:
        return "crypto"

    async def fetch(self, symbol: str, adapter: ExchangeAdapter) -> dict:
        """Fetch funding rate and OI from exchange adapter.

        Each data point is independently try/excepted — one failure
        does not block the other.
        """
        result: dict = {}

        # Funding rate
        try:
            rate = await adapter.get_funding_rate(symbol)
            if rate is not None:
                result["funding_rate"] = rate
                if rate > _CROWDED_LONG_THRESHOLD:
                    result["funding_signal"] = "CROWDED_LONG"
                elif rate < _CROWDED_SHORT_THRESHOLD:
                    result["funding_signal"] = "CROWDED_SHORT"
                else:
                    result["funding_signal"] = "NEUTRAL"
        except Exception as e:
            logger.warning(f"CryptoFlowProvider: funding rate unavailable for {symbol}: {e}")

        # Open Interest
        try:
            oi = await adapter.get_open_interest(symbol)
            if oi is not None:
                result["open_interest"] = oi

                # Record snapshot in the history buffer
                now = time.time()
                if symbol not in self._oi_history:
                    self._oi_history[symbol] = deque(maxlen=_OI_BUFFER_MAXLEN)
                self._oi_history[symbol].append((now, oi))

                # Compute 4h delta from the buffer
                oi_change, oi_trend = self._compute_oi_delta(symbol, now, oi)
                result["oi_change_4h"] = oi_change
                result["oi_trend"] = oi_trend
        except Exception as e:
            logger.warning(f"CryptoFlowProvider: OI unavailable for {symbol}: {e}")

        return result

    def _compute_oi_delta(
        self, symbol: str, now: float, current_oi: float
    ) -> tuple[float | None, str]:
        """Compute oi_change_4h and oi_trend from the history buffer.

        Returns (None, "STABLE") during the warmup window when the
        buffer doesn't span 4 hours yet.
        """
        buf = self._oi_history.get(symbol)
        if not buf or len(buf) < 2:
            return None, "STABLE"

        cutoff = now - _OI_LOOKBACK_SECONDS

        # Find the oldest entry at or after the 4h cutoff.
        # The deque is ordered by time (append-only), so scan from left.
        old_ts, old_oi = buf[0]
        if old_ts > cutoff:
            # Buffer doesn't span 4h yet — cold start
            return None, "STABLE"

        # Walk forward to find the entry closest to (but >= cutoff)
        best_ts, best_oi = old_ts, old_oi
        for ts, oi_val in buf:
            if ts >= cutoff:
                best_ts, best_oi = ts, oi_val
                break

        # Log when buffer first reaches 4h depth
        if symbol not in self._oi_warm_logged:
            self._oi_warm_logged.add(symbol)
            logger.info(
                f"CryptoFlowProvider: OI history buffer warm for {symbol} "
                f"({len(buf)} snapshots)"
            )

        if best_oi == 0:
            return None, "STABLE"

        oi_change = (current_oi - best_oi) / best_oi

        if oi_change > _OI_TREND_THRESHOLD:
            oi_trend = "BUILDING"
        elif oi_change < -_OI_TREND_THRESHOLD:
            oi_trend = "DROPPING"
        else:
            oi_trend = "STABLE"

        return oi_change, oi_trend
