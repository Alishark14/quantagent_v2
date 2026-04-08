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
from pathlib import Path

from engine.data.indicators import compute_all_indicators
from engine.data.swing_detection import find_swing_highs, find_swing_lows
from engine.events import EventBus, SetupDetected
from exchanges.base import ExchangeAdapter
from mcp.macro_regime.agent import load_macro_regime
from sentinel.conditions import ReadinessScorer
from sentinel.config import get_sentinel_cooldown, get_sentinel_daily_budget

import numpy as np

logger = logging.getLogger(__name__)


_DEFAULT_MACRO_REGIME_PATH = Path("macro_regime.json")


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
        cache=None,
        macro_regime_path: Path | str = _DEFAULT_MACRO_REGIME_PATH,
        clock=None,
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
        self._cache = cache  # Optional CacheManager — used for L2/L3 reads only
        self._macro_regime_path = Path(macro_regime_path)
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))

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

        # Funding rate (L2 — read from cache if available)
        funding_rate = None
        try:
            if self._cache is not None:
                from storage.cache import funding_key
                from storage.cache.ttl import FLOW_TTL
                funding_rate = await self._cache.get_or_fetch(
                    funding_key(self._symbol),
                    lambda: self._adapter.get_funding_rate(self._symbol),
                    ttl=FLOW_TTL,
                )
            else:
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
            if not self._can_trigger(now):
                logger.debug(
                    f"Sentinel: {self._symbol} readiness={score:.2f} above threshold "
                    f"but cooldown/budget prevents trigger"
                )
                return score, conditions

            # Macro blackout suppression (§13.2.4) — check AFTER cooldown so a
            # blackout doesn't quietly burn the daily budget. The blackout
            # check is read-only filesystem I/O; missing or expired files
            # behave as "no blackout" and the cycle proceeds normally.
            blackout_reason = self._active_blackout_reason()
            if blackout_reason is not None:
                logger.info(
                    f"Sentinel: {self._symbol} setup detected "
                    f"(readiness={score:.2f}) but blackout active "
                    f"({blackout_reason}), suppressing"
                )
                return score, conditions

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

        return score, conditions

    # ------------------------------------------------------------------
    # Macro blackout helper
    # ------------------------------------------------------------------

    def _active_blackout_reason(self) -> str | None:
        """Return the reason of any active blackout window, else None.

        A missing / unparseable / expired `macro_regime.json` is the
        safe default — no suppression. Failures here MUST NOT crash
        the Sentinel loop, so the entire check is wrapped.
        """
        try:
            macro = load_macro_regime(self._macro_regime_path)
            if macro is None or macro.error is not None:
                return None
            now = self._clock()
            # Honour expires — a stale regime overlay shouldn't keep
            # blocking new entries indefinitely.
            if macro.expires:
                try:
                    expires_dt = datetime.fromisoformat(
                        macro.expires.replace("Z", "+00:00")
                    )
                    if expires_dt.tzinfo is None:
                        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                    if now > expires_dt:
                        return None
                except (TypeError, ValueError):
                    return None
            for window in macro.blackout_windows:
                if window.contains(now):
                    return window.reason
            return None
        except Exception:
            logger.warning("Sentinel: macro_regime blackout check failed", exc_info=True)
            return None

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
