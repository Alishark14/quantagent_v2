"""SentinelMonitor: persistent, code-only, proactive market watcher.

Polls exchange for candles at a configurable interval, computes fast
indicators on a short window, runs ReadinessScorer, and emits
SetupDetected when the score exceeds the threshold.

Enforces:
- Cooldown (default 15 min) between triggers for the same symbol
- Daily budget (default 8 triggers per symbol)

Zero LLM calls. All computation is local and deterministic.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from engine.data.indicators import compute_all_indicators
from engine.data.swing_detection import find_swing_highs, find_swing_lows
from engine.events import EventBus, SetupDetected
from exchanges.base import ExchangeAdapter
from sentinel.conditions import ReadinessScorer
from sentinel.config import get_sentinel_cooldown, get_sentinel_daily_budget

import numpy as np

logger = logging.getLogger(__name__)


class SentinelMonitor:
    """Persistent market monitor that detects tradeable setups.

    Runs in a loop, polling the exchange for fresh candle data,
    computing indicators, and scoring readiness. When readiness
    exceeds the threshold, emits a SetupDetected event.
    """

    def __init__(
        self,
        adapter: ExchangeAdapter,
        event_bus: EventBus,
        symbol: str,
        timeframe: str = "1h",
        check_interval: int = 30,
        threshold: float = 0.7,
        cooldown_seconds: int | None = None,
        daily_budget: int | None = None,
        candle_window: int = 30,
    ) -> None:
        self._adapter = adapter
        self._bus = event_bus
        self._symbol = symbol
        self._timeframe = timeframe
        self._check_interval = check_interval
        self._threshold = threshold
        self._cooldown_seconds = cooldown_seconds if cooldown_seconds is not None else get_sentinel_cooldown(timeframe)
        self._daily_budget = daily_budget if daily_budget is not None else get_sentinel_daily_budget(timeframe)
        self._candle_window = candle_window

        self._scorer = ReadinessScorer()
        self._last_trigger: datetime | None = None
        self._daily_trigger_count: int = 0
        self._current_day: int = -1
        self._prev_macd_histogram: float | None = None
        self._running: bool = False

    async def run(self) -> None:
        """Run the Sentinel loop until stopped."""
        self._running = True
        logger.info(
            f"Sentinel started: {self._symbol}/{self._timeframe} "
            f"(interval={self._check_interval}s, threshold={self._threshold}, "
            f"cooldown={self._cooldown_seconds}s, budget={self._daily_budget}/day)"
        )

        while self._running:
            try:
                await self._check_once()
            except Exception:
                logger.exception(f"Sentinel: check failed for {self._symbol}")

            await asyncio.sleep(self._check_interval)

    def stop(self) -> None:
        """Stop the Sentinel loop."""
        self._running = False
        logger.info(f"Sentinel stopped: {self._symbol}")

    async def check_once(self) -> tuple[float, list]:
        """Run a single readiness check. Returns (score, conditions).

        Public method for testing and manual triggering.
        """
        return await self._check_once()

    async def _check_once(self) -> tuple[float, list]:
        """Internal: fetch data, compute indicators, score readiness."""
        # Reset daily counter at day boundary
        now = datetime.now(timezone.utc)
        if now.day != self._current_day:
            self._current_day = now.day
            self._daily_trigger_count = 0

        # Fetch candles
        candles = await self._adapter.fetch_ohlcv(
            self._symbol, self._timeframe, limit=self._candle_window,
        )
        if not candles or len(candles) < 10:
            logger.debug(f"Sentinel: insufficient candles for {self._symbol}")
            return 0.0, []

        # Compute indicators
        indicators = compute_all_indicators(candles)
        current_price = float(candles[-1]["close"])

        # Swing detection
        highs = np.array([c["high"] for c in candles], dtype=float)
        lows = np.array([c["low"] for c in candles], dtype=float)
        swing_highs = find_swing_highs(highs)
        swing_lows = find_swing_lows(lows)

        # Funding rate (optional)
        funding_rate = None
        try:
            funding_rate = await self._adapter.get_funding_rate(self._symbol)
        except Exception:
            pass

        # Score readiness
        score, conditions = self._scorer.score(
            indicators=indicators,
            current_price=current_price,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            funding_rate=funding_rate,
            prev_macd_histogram=self._prev_macd_histogram,
        )

        # Store current histogram for next check's cross detection
        macd = indicators.get("macd", {})
        self._prev_macd_histogram = macd.get("histogram")

        triggered_names = [c.name for c in conditions if c.triggered]
        logger.debug(
            f"Sentinel: {self._symbol} readiness={score:.2f} "
            f"({len(triggered_names)} conditions: {triggered_names})"
        )

        # Check if we should emit SetupDetected
        if score >= self._threshold:
            if self._can_trigger(now):
                self._last_trigger = now
                self._daily_trigger_count += 1

                logger.info(
                    f"Sentinel: SETUP DETECTED for {self._symbol} "
                    f"(readiness={score:.2f}, triggers today={self._daily_trigger_count})"
                )

                try:
                    await self._bus.publish(SetupDetected(
                        source="sentinel",
                        symbol=self._symbol,
                        readiness=score,
                        conditions=[c.detail for c in conditions if c.triggered],
                    ))
                except Exception:
                    logger.warning("Sentinel: failed to emit SetupDetected", exc_info=True)
            else:
                logger.debug(
                    f"Sentinel: {self._symbol} readiness={score:.2f} above threshold "
                    f"but cooldown/budget prevents trigger"
                )

        return score, conditions

    def _can_trigger(self, now: datetime) -> bool:
        """Check cooldown and daily budget constraints."""
        # Daily budget
        if self._daily_trigger_count >= self._daily_budget:
            return False

        # Cooldown
        if self._last_trigger is not None:
            elapsed = (now - self._last_trigger).total_seconds()
            if elapsed < self._cooldown_seconds:
                return False

        return True

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def daily_triggers_remaining(self) -> int:
        return max(0, self._daily_budget - self._daily_trigger_count)
