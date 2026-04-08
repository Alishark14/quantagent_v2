"""SentinelMonitor: persistent, code-only, proactive market watcher.

Polls exchange for candles at a configurable interval, computes fast
indicators on a short window, runs ReadinessScorer, and emits
SetupDetected when the score exceeds the threshold.

Enforces:
- Cooldown (default 15 min) between triggers for the same symbol
- Daily budget (default 8 triggers per symbol)
- **Escalating readiness threshold after SKIP** (see below)

Zero LLM calls. All computation is local and deterministic.

Escalation contract (per ARCHITECTURE.md §8 + the Task 11 spec)
================================================================

After a SetupDetected fires and the resulting analysis pipeline returns
SKIP (i.e. no trade was opened), the Sentinel:
  1. Raises its readiness threshold for that symbol by ESCALATION_STEP
     (default +0.10), capped at BASE + MAX_ESCALATION (default 0.55).
  2. Switches the cooldown to SKIP_COOLDOWN_SECONDS (default 15 min)
     instead of the full candle period — so the next *higher-quality*
     setup can fire much sooner than the standard candle-cooldown.

After a SetupDetected fires and the pipeline returns TRADE (a position
was opened), the Sentinel:
  1. Resets the readiness threshold for that symbol back to BASE.
  2. Switches the cooldown back to the full candle period (the
     standard "no immediate re-entry on the same candle" guarantee).

On every new candle close (detected via `candles[-1]["timestamp"]`
advancing past the previous tick's value), ALL per-symbol escalations
are cleared — every new candle is a fresh chance.

The feedback that "the pipeline returned SKIP / TRADE" arrives via the
`SetupResult` event published by `BotManager` after each spawned
TraderBot finishes. The Sentinel subscribes via `subscribe_results()`.

Implementation notes
====================

- Per-symbol state is held in `_current_threshold: dict[str, float]`
  even though each `SentinelMonitor` is bound to a single symbol. This
  matches the user spec literally and lets a future refactor lift the
  Sentinel to multi-symbol without changing the data shape.
- The cooldown duration is held in `_active_cooldown_seconds`, which
  flips between the candle-period default and `SKIP_COOLDOWN_SECONDS`
  in response to `SetupResult` events. The standard `_can_trigger`
  check reads the active value, so escalation tuning has no effect
  until the NEXT trigger attempt — which is the right semantics
  (we don't shorten an in-flight cooldown retroactively).
- Candle-close detection uses the timestamp of the latest candle from
  `fetch_ohlcv`. The first observation initializes `_last_candle_ts`
  without resetting (a fresh-start Sentinel shouldn't immediately
  clear escalation state it never built).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from engine.data.indicators import compute_all_indicators
from engine.data.swing_detection import find_swing_highs, find_swing_lows
from engine.events import EventBus, SetupDetected, SetupResult
from exchanges.base import ExchangeAdapter
from mcp.macro_regime.agent import load_macro_regime
from sentinel.conditions import ReadinessScorer
from sentinel.config import (
    BASE_READINESS_THRESHOLD,
    ESCALATION_STEP,
    MAX_ESCALATION,
    SKIP_COOLDOWN_SECONDS,
    get_sentinel_cooldown,
    get_sentinel_daily_budget,
)

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
        # Escalation tunables — defaults pulled from sentinel.config so the
        # constants live in one place. Pass overrides for tests / per-bot
        # tuning. `threshold` (the legacy ctor arg) still controls the
        # *initial* base threshold for back-compat — it overrides the
        # config default for THIS monitor.
        base_threshold: float | None = None,
        escalation_step: float = ESCALATION_STEP,
        max_escalation: float = MAX_ESCALATION,
        skip_cooldown_seconds: int = SKIP_COOLDOWN_SECONDS,
    ) -> None:
        self._adapter = adapter
        self._bus = event_bus
        self._symbol = symbol
        self._timeframe = timeframe
        self._check_interval = check_interval
        # Legacy `threshold` ctor arg still functions as the per-monitor
        # base. If neither is provided, fall back to the config default.
        # The legacy default of 0.7 (from before escalation existed)
        # still wins over the config default when callers don't pass
        # `base_threshold` — that preserves all existing call sites and
        # tests that rely on the 0.7 default behaviour.
        if base_threshold is not None:
            self._base_threshold = base_threshold
        else:
            self._base_threshold = threshold
        self._candle_cooldown_seconds = (
            cooldown_seconds if cooldown_seconds is not None
            else get_sentinel_cooldown(timeframe)
        )
        # Back-compat alias — pre-escalation tests + scripts that read
        # `_cooldown_seconds` see the candle-period default. The active
        # cooldown that `_can_trigger` actually uses lives in
        # `_active_cooldown_seconds` and flips between the default and
        # `SKIP_COOLDOWN_SECONDS` in response to SetupResult events.
        self._cooldown_seconds = self._candle_cooldown_seconds
        self._active_cooldown_seconds = self._candle_cooldown_seconds
        self._daily_budget = daily_budget if daily_budget is not None else get_sentinel_daily_budget(timeframe)
        self._candle_window = candle_window
        self._cache = cache  # Optional CacheManager — used for L2/L3 reads only
        self._macro_regime_path = Path(macro_regime_path)
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))

        # Escalation state — keyed per-symbol per the spec, even though
        # each SentinelMonitor instance is currently bound to a single
        # symbol. The dict shape future-proofs the data layout for a
        # multi-symbol monitor refactor.
        self._current_threshold: dict[str, float] = {}
        self._escalation_step = escalation_step
        self._max_escalation = max_escalation
        self._skip_cooldown_seconds = skip_cooldown_seconds

        # Candle-close detection — set to the latest candle's timestamp
        # on each tick; clears all escalations when it advances.
        self._last_candle_ts: int | None = None

        self._scorer = ReadinessScorer()
        self._last_trigger: datetime | None = None
        self._daily_trigger_count: int = 0
        self._current_day: int = -1
        self._prev_macd_histogram: float | None = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public escalation API — used by tests and (optionally) consumers
    # ------------------------------------------------------------------

    def current_threshold(self, symbol: str | None = None) -> float:
        """Return the current readiness threshold for `symbol`.

        Defaults to this monitor's bound symbol when called without
        arguments. Returns the per-symbol escalated value if one has
        been set, otherwise the base threshold.
        """
        sym = symbol if symbol is not None else self._symbol
        return self._current_threshold.get(sym, self._base_threshold)

    @property
    def base_threshold(self) -> float:
        return self._base_threshold

    @property
    def active_cooldown_seconds(self) -> int:
        """Cooldown duration that the NEXT trigger check will use.

        Flips between the candle-period default and `SKIP_COOLDOWN_SECONDS`
        in response to SetupResult events.
        """
        return self._active_cooldown_seconds

    def subscribe_results(self) -> None:
        """Subscribe this monitor's escalation handler to SetupResult events.

        Call once during wiring (typically right after constructing the
        monitor and before `run()`). Idempotent — calling multiple times
        registers the handler multiple times, which is harmless but
        wastes work, so prefer to call it exactly once.
        """
        self._bus.subscribe(SetupResult, self._on_setup_result)

    async def _on_setup_result(self, event: SetupResult) -> None:
        """Adjust escalation state in response to a SetupResult.

        Filters by symbol — a Sentinel only reacts to its own symbol's
        results. SKIP outcomes escalate the per-symbol threshold and
        switch to the short cooldown; TRADE outcomes reset the
        threshold and switch back to the candle-period cooldown.
        """
        if event.symbol != self._symbol:
            return
        if event.outcome == "TRADE":
            self._reset_threshold_after_trade()
        elif event.outcome == "SKIP":
            self._escalate_threshold_after_skip()
        else:
            logger.warning(
                f"Sentinel: SetupResult for {event.symbol} carries "
                f"unknown outcome {event.outcome!r}, ignoring"
            )

    def _escalate_threshold_after_skip(self) -> None:
        current = self._current_threshold.get(self._symbol, self._base_threshold)
        ceiling = self._base_threshold + self._max_escalation
        new_threshold = min(current + self._escalation_step, ceiling)
        self._current_threshold[self._symbol] = new_threshold
        self._active_cooldown_seconds = self._skip_cooldown_seconds
        logger.info(
            f"Sentinel: SKIP on {self._symbol}: escalating readiness "
            f"threshold from {current:.2f} to {new_threshold:.2f} "
            f"(cooldown switched to {self._skip_cooldown_seconds}s)"
        )

    def _reset_threshold_after_trade(self) -> None:
        self._current_threshold[self._symbol] = self._base_threshold
        self._active_cooldown_seconds = self._candle_cooldown_seconds
        logger.info(
            f"Sentinel: TRADE on {self._symbol}: resetting readiness "
            f"threshold to {self._base_threshold:.2f} "
            f"(cooldown restored to {self._candle_cooldown_seconds}s)"
        )

    def _reset_all_escalations_for_new_candle(self, new_ts: int) -> None:
        """Wipe per-symbol escalation state on a new candle close.

        Called from `_check_once` when `candles[-1]["timestamp"]`
        advances past `_last_candle_ts`. Restores the active cooldown
        to the candle-period default as well — every new candle is a
        fresh chance.
        """
        self._current_threshold.clear()
        self._active_cooldown_seconds = self._candle_cooldown_seconds
        logger.info(
            f"Sentinel: New candle (ts={new_ts}): resetting all readiness "
            f"thresholds to base {self._base_threshold:.2f}"
        )

    async def run(self) -> None:
        """Run the Sentinel loop until stopped."""
        self._running = True
        logger.info(
            f"Sentinel started: {self._symbol}/{self._timeframe} "
            f"(interval={self._check_interval}s, base_threshold={self._base_threshold}, "
            f"cooldown={self._candle_cooldown_seconds}s, budget={self._daily_budget}/day)"
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

        # Candle-close detection — every new candle is a fresh chance, so
        # we wipe any escalation state. The first observation initializes
        # the marker without resetting (a fresh-start Sentinel shouldn't
        # immediately clear escalation state it never built).
        latest_candle_ts = int(candles[-1]["timestamp"])
        if self._last_candle_ts is None:
            self._last_candle_ts = latest_candle_ts
        elif latest_candle_ts > self._last_candle_ts:
            self._reset_all_escalations_for_new_candle(latest_candle_ts)
            self._last_candle_ts = latest_candle_ts

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
        # Read the *escalated* threshold (defaults to base when no SKIP
        # has happened yet on this candle). This is the only place the
        # threshold gate is evaluated — escalation flows through here.
        active_threshold = self.current_threshold()
        logger.debug(
            f"Sentinel: {self._symbol} readiness={score:.2f} "
            f"(threshold={active_threshold:.2f}, "
            f"{len(triggered_names)} conditions: {triggered_names})"
        )

        # Check if we should emit SetupDetected
        if score >= active_threshold:
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
        """Check cooldown and daily budget constraints.

        Reads `_active_cooldown_seconds`, which flips between the candle
        period and `SKIP_COOLDOWN_SECONDS` in response to SetupResult
        events. The standard candle-period cooldown is the default and
        is what gets used until the first SKIP outcome arrives.
        """
        # Daily budget
        if self._daily_trigger_count >= self._daily_budget:
            return False

        # Cooldown
        if self._last_trigger is not None:
            elapsed = (now - self._last_trigger).total_seconds()
            if elapsed < self._active_cooldown_seconds:
                return False

        return True

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def daily_triggers_remaining(self) -> int:
        return max(0, self._daily_budget - self._daily_trigger_count)
