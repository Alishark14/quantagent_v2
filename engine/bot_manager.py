"""BotManager: spawns and manages ephemeral TraderBots.

Subscribes to SetupDetected events from Sentinel. Enforces per-symbol
concurrent limits. Cleans up after bots complete or crash.

Closes the feedback loop back to Sentinel: after every spawned (and
manually-spawned) TraderBot finishes, BotManager publishes a
`SetupResult` event so the Sentinel can ratchet its readiness
threshold up after a SKIP and reset it after a TRADE. The TRADE / SKIP
classification is derived from the bot's result dict — see
`_classify_outcome` below for the canonical rules.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from engine.events import EventBus, SetupDetected, SetupResult
from engine.trader_bot import TraderBot

logger = logging.getLogger(__name__)


# Actions that count as "the pipeline opened a new position" — anything
# else (SKIP / HOLD / CLOSE_ALL / errors / failed orders) is classified
# as a SKIP outcome for Sentinel escalation purposes.
#
# CLOSE_ALL is intentionally NOT in this set: it closes a position
# rather than opening one, so there's no new-entry rationale to reset
# Sentinel's escalation. (Future work: distinguish CLOSE outcomes if
# escalation logic ever needs them.)
_TRADE_ACTIONS: frozenset[str] = frozenset({
    "LONG", "SHORT", "ADD_LONG", "ADD_SHORT",
})


class BotManager:
    """Manages ephemeral TraderBot lifecycle: spawn, track, cleanup."""

    def __init__(
        self,
        event_bus: EventBus,
        bot_factory: callable,
        max_concurrent_per_symbol: int = 1,
    ) -> None:
        """
        Args:
            event_bus: For subscribing to SetupDetected.
            bot_factory: Callable(symbol, bot_id) -> TraderBot.
                The caller wires all dependencies; manager just calls it.
            max_concurrent_per_symbol: Max bots running per symbol at once.
        """
        self._bus = event_bus
        self._bot_factory = bot_factory
        self._max_concurrent = max_concurrent_per_symbol
        self._active: dict[str, set[str]] = {}  # symbol -> set of active bot_ids
        self._results: list[dict] = []
        self._lock = asyncio.Lock()

    def subscribe(self) -> None:
        """Subscribe to SetupDetected events."""
        self._bus.subscribe(SetupDetected, self._on_setup_detected)
        logger.info("BotManager: subscribed to SetupDetected")

    async def _on_setup_detected(self, event: SetupDetected) -> None:
        """Handle a SetupDetected event: spawn a TraderBot if allowed."""
        symbol = event.symbol

        async with self._lock:
            active_ids = self._active.get(symbol, set())
            if len(active_ids) >= self._max_concurrent:
                logger.info(
                    f"BotManager: skipping {symbol} — {len(active_ids)} bots already active "
                    f"(max {self._max_concurrent})"
                )
                return

            bot_id = f"bot-{symbol}-{uuid4().hex[:8]}"
            if symbol not in self._active:
                self._active[symbol] = set()
            self._active[symbol].add(bot_id)

        logger.info(
            f"BotManager: spawning {bot_id} for {symbol} "
            f"(readiness={event.readiness:.2f})"
        )

        # Spawn bot as a background task (non-blocking)
        asyncio.create_task(self._run_bot(symbol, bot_id))

    async def _run_bot(self, symbol: str, bot_id: str) -> None:
        """Run a TraderBot and clean up after it completes."""
        result: dict | None = None
        try:
            bot = self._bot_factory(symbol, bot_id)
            result = await bot.run()
            self._results.append(result)

            logger.info(
                f"BotManager: {bot_id} finished — {result.get('action', '?')} "
                f"({result.get('status', '?')})"
            )

        except Exception as e:
            logger.error(f"BotManager: {bot_id} crashed — {e}", exc_info=True)
            result = {
                "bot_id": bot_id,
                "status": "CRASH",
                "action": "SKIP",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._results.append(result)

        finally:
            async with self._lock:
                active_ids = self._active.get(symbol, set())
                active_ids.discard(bot_id)
                if not active_ids:
                    del self._active[symbol]

        # Emit SetupResult OUTSIDE the lock so the handler chain (which
        # may call into Sentinel state) doesn't sit on our active-set
        # mutex. Failures to publish are logged but never propagate —
        # the bot already finished, the cleanup is done, and the
        # Sentinel feedback loop is best-effort.
        if result is not None:
            await self._publish_setup_result(symbol, bot_id, result)

    async def spawn_bot(self, symbol: str) -> dict:
        """Manually spawn a bot for a symbol. Returns the result dict.

        Unlike event-triggered spawning, this is synchronous (waits for
        the bot to finish). Useful for scheduled cycles.
        """
        bot_id = f"bot-{symbol}-{uuid4().hex[:8]}"

        async with self._lock:
            if symbol not in self._active:
                self._active[symbol] = set()
            active_ids = self._active[symbol]
            if len(active_ids) >= self._max_concurrent:
                # Concurrency-blocked: do NOT publish a SetupResult.
                # The pipeline never ran, so there's no SKIP/TRADE
                # outcome to feed back to the Sentinel — the situation
                # is "we already have a bot for this symbol", which is
                # orthogonal to escalation.
                return {
                    "bot_id": bot_id,
                    "status": "SKIPPED",
                    "action": "SKIP",
                    "reasoning": f"Max concurrent bots ({self._max_concurrent}) for {symbol}",
                }
            active_ids.add(bot_id)

        result: dict
        try:
            bot = self._bot_factory(symbol, bot_id)
            result = await bot.run()
            self._results.append(result)
        except Exception as e:
            logger.error(f"BotManager: manual spawn {bot_id} crashed — {e}", exc_info=True)
            result = {
                "bot_id": bot_id,
                "status": "CRASH",
                "action": "SKIP",
                "error": str(e),
            }
        finally:
            async with self._lock:
                active_ids = self._active.get(symbol, set())
                active_ids.discard(bot_id)
                if not active_ids and symbol in self._active:
                    del self._active[symbol]

        # Publish OUTSIDE the lock — same rationale as `_run_bot`.
        await self._publish_setup_result(symbol, bot_id, result)
        return result

    # ------------------------------------------------------------------
    # SetupResult publication
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_outcome(result: dict) -> str:
        """Map a TraderBot result dict to a SetupResult outcome.

        Returns "TRADE" only when the action is one of LONG / SHORT /
        ADD_LONG / ADD_SHORT AND the order_result reports success.
        Everything else (SKIP, HOLD, CLOSE_ALL, errored bot, failed
        order, missing order_result) classifies as "SKIP".

        See the module-level _TRADE_ACTIONS for the canonical action
        set, and the SetupResult docstring in engine/events.py for the
        contract.
        """
        action = result.get("action", "")
        if action not in _TRADE_ACTIONS:
            return "SKIP"
        # Action is a new-position open. Now check the order result.
        order = result.get("order_result")
        if not isinstance(order, dict):
            return "SKIP"
        return "TRADE" if order.get("success") else "SKIP"

    async def _publish_setup_result(
        self, symbol: str, bot_id: str, result: dict
    ) -> None:
        """Publish a SetupResult event derived from a bot result dict.

        Failures to publish are logged but never propagate — the
        Sentinel feedback loop is best-effort and must not crash the
        bot lifecycle.
        """
        outcome = self._classify_outcome(result)
        try:
            await self._bus.publish(SetupResult(
                source="bot_manager",
                symbol=symbol,
                outcome=outcome,
                action=str(result.get("action", "")),
                bot_id=bot_id,
                conviction_score=float(result.get("conviction_score", 0.0) or 0.0),
            ))
        except Exception:
            logger.warning(
                f"BotManager: failed to publish SetupResult for {bot_id}",
                exc_info=True,
            )

    def active_count(self, symbol: str) -> int:
        """Number of active bots for a symbol."""
        return len(self._active.get(symbol, set()))

    @property
    def total_active(self) -> int:
        return sum(len(ids) for ids in self._active.values())

    @property
    def results(self) -> list[dict]:
        return list(self._results)
