"""BotManager: spawns and manages ephemeral TraderBots.

Subscribes to SetupDetected events from Sentinel. Enforces per-symbol
concurrent limits. Cleans up after bots complete or crash.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from engine.events import EventBus, SetupDetected
from engine.trader_bot import TraderBot

logger = logging.getLogger(__name__)


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
            self._results.append({
                "bot_id": bot_id,
                "status": "CRASH",
                "action": "SKIP",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        finally:
            async with self._lock:
                active_ids = self._active.get(symbol, set())
                active_ids.discard(bot_id)
                if not active_ids:
                    del self._active[symbol]

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
                return {
                    "bot_id": bot_id,
                    "status": "SKIPPED",
                    "action": "SKIP",
                    "reasoning": f"Max concurrent bots ({self._max_concurrent}) for {symbol}",
                }
            active_ids.add(bot_id)

        try:
            bot = self._bot_factory(symbol, bot_id)
            result = await bot.run()
            self._results.append(result)
            return result
        except Exception as e:
            logger.error(f"BotManager: manual spawn {bot_id} crashed — {e}", exc_info=True)
            return {
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

    def active_count(self, symbol: str) -> int:
        """Number of active bots for a symbol."""
        return len(self._active.get(symbol, set()))

    @property
    def total_active(self) -> int:
        return sum(len(ids) for ids in self._active.values())

    @property
    def results(self) -> list[dict]:
        return list(self._results)
