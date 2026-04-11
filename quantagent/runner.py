"""BotRunner: the production service that keeps everything alive.

Loads active bots from DB, starts a SentinelMonitor per unique symbol,
schedules fallback analysis loops, and manages the full lifecycle.

Graceful shutdown: stops sentinels, cancels scheduled tasks, waits
for active TraderBots to finish.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from engine.bot_manager import BotManager
from engine.config import TradingConfig
from engine.events import EventBus
from exchanges.base import ExchangeAdapter
from llm.base import LLMProvider
from sentinel.config import get_sentinel_cooldown
from sentinel.monitor import SentinelMonitor
from storage.repositories.base import BotRepository

logger = logging.getLogger(__name__)

# Exponential backoff limits for sentinel auto-restart
_MIN_BACKOFF_SECONDS = 5
_MAX_BACKOFF_SECONDS = 300  # 5 minutes


class BotRunner:
    """Production service: loads bots, runs sentinels, schedules fallbacks.

    Lifecycle:
        runner = BotRunner(...)
        await runner.start()   # loads bots from DB, starts sentinels
        ...
        await runner.stop()    # graceful shutdown
    """

    def __init__(
        self,
        repos,
        adapter_factory: callable,
        llm_provider: LLMProvider,
        event_bus: EventBus,
        bot_manager: BotManager,
        shadow_mode: bool = False,
    ) -> None:
        self._repos = repos
        self._adapter_factory = adapter_factory
        self._llm_provider = llm_provider
        self._bus = event_bus
        self._bot_manager = bot_manager
        self._shadow_mode = shadow_mode

        self._sentinels: dict[str, SentinelMonitor] = {}
        self._sentinel_tasks: dict[str, asyncio.Task] = {}
        self._scheduled_tasks: dict[str, asyncio.Task] = {}
        self._bot_configs: dict[str, dict] = {}  # bot_id -> bot dict
        self._symbol_bots: dict[str, set[str]] = {}  # symbol -> set of bot_ids
        self._adapters: dict[str, object] = {}  # symbol -> shared adapter
        self._running = False

    @property
    def shadow_mode(self) -> bool:
        return self._shadow_mode

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def sentinel_count(self) -> int:
        return len(self._sentinels)

    @property
    def scheduled_task_count(self) -> int:
        return len(self._scheduled_tasks)

    def get_sentinel(self, symbol: str) -> SentinelMonitor | None:
        return self._sentinels.get(symbol)

    async def start(self) -> None:
        """Load active bots from DB, start sentinels, schedule fallbacks."""
        if self._running:
            logger.warning("BotRunner: already running")
            return

        self._running = True
        self._bot_manager.subscribe()

        # Load all active bots from DB
        bot_repo: BotRepository = self._repos.bots
        # We need to get bots across all users — iterate known users
        # Since BotRepository doesn't have get_all(), we load bots
        # that were registered during add_bot or from a startup scan.
        # For now, load from the bot configs we know about.
        logger.info("BotRunner: started")

    async def start_with_bots(self, bots: list[dict]) -> None:
        """Start the runner with a pre-loaded list of bot configs.

        This is the primary startup path. The caller (main.py) loads
        active bots from DB and passes them here.
        """
        if self._running:
            logger.warning("BotRunner: already running")
            return

        self._running = True
        self._bot_manager.subscribe()

        for bot in bots:
            await self._register_bot(bot)

        logger.info(
            f"BotRunner: started with {len(bots)} bots, "
            f"{len(self._sentinels)} sentinels"
        )

    async def stop(self) -> None:
        """Graceful shutdown: stop sentinels, cancel tasks, wait for bots."""
        if not self._running:
            return

        self._running = False
        logger.info("BotRunner: stopping...")

        # 1. Stop all sentinels
        for symbol, sentinel in self._sentinels.items():
            sentinel.stop()
            logger.debug(f"BotRunner: stopped sentinel for {symbol}")

        # 2. Cancel all sentinel tasks
        for symbol, task in self._sentinel_tasks.items():
            task.cancel()
        if self._sentinel_tasks:
            await asyncio.gather(
                *self._sentinel_tasks.values(), return_exceptions=True
            )

        # 3. Cancel all scheduled tasks
        for bot_id, task in self._scheduled_tasks.items():
            task.cancel()
        if self._scheduled_tasks:
            await asyncio.gather(
                *self._scheduled_tasks.values(), return_exceptions=True
            )

        # 4. Wait for any active TraderBots to finish
        if self._bot_manager.total_active > 0:
            logger.info(
                f"BotRunner: waiting for {self._bot_manager.total_active} "
                f"active bots to finish"
            )
            for _ in range(30):  # max 30 seconds
                if self._bot_manager.total_active == 0:
                    break
                await asyncio.sleep(1)

        self._sentinels.clear()
        self._sentinel_tasks.clear()
        self._scheduled_tasks.clear()
        self._bot_configs.clear()
        self._symbol_bots.clear()
        self._adapters.clear()

        logger.info("BotRunner: stopped")

    async def add_bot(self, bot_config: dict) -> None:
        """Add a bot at runtime. Saves to DB, starts sentinel + schedule.

        Called by the API when a user creates and activates a bot.
        """
        bot_id = bot_config["id"]
        bot_repo: BotRepository = self._repos.bots

        # Save to DB if not already saved
        existing = await bot_repo.get_bot(bot_id)
        if existing is None:
            await bot_repo.save_bot(bot_config)

        await self._register_bot(bot_config)
        logger.info(f"BotRunner: added bot {bot_id}")

    async def remove_bot(self, bot_id: str) -> None:
        """Remove a bot. Stops sentinel if no other bots use that symbol."""
        bot = self._bot_configs.pop(bot_id, None)
        if bot is None:
            logger.warning(f"BotRunner: bot {bot_id} not found for removal")
            return

        symbol = bot.get("symbol", "")

        # Remove from symbol tracking
        if symbol in self._symbol_bots:
            self._symbol_bots[symbol].discard(bot_id)

            # Stop sentinel if no other bots use this symbol
            if not self._symbol_bots[symbol]:
                del self._symbol_bots[symbol]
                await self._stop_sentinel(symbol)

        # Cancel scheduled task
        task = self._scheduled_tasks.pop(bot_id, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        logger.info(f"BotRunner: removed bot {bot_id}")

    async def _register_bot(self, bot: dict) -> None:
        """Register a bot: track it, start sentinel, schedule fallback.

        The bot's ``mode`` field (``"live"`` | ``"shadow"``) is read
        and threaded through to ``adapter_factory(exchange, mode=mode)``
        so each sentinel gets the adapter type that matches the bot it
        belongs to. In a single-mode runner this just echoes the
        process-wide mode (because main.py filters bots by mode at
        load time); in a mixed-mode runner (e.g. the integration test)
        each sentinel gets the right adapter regardless.
        """
        bot_id = bot["id"]
        symbol = bot.get("symbol", "BTC-USDC")
        timeframe = bot.get("timeframe", "1h")
        exchange = bot.get("exchange", "hyperliquid")
        mode = bot.get("mode", "live")

        self._bot_configs[bot_id] = bot

        # Track symbol -> bot mapping
        if symbol not in self._symbol_bots:
            self._symbol_bots[symbol] = set()
        self._symbol_bots[symbol].add(bot_id)

        # Start sentinel for this symbol if not already running.
        # Store the adapter so spawned bots share the same instance —
        # critical for shadow mode where Sentinel feeds candles
        # (triggering SL/TP) and the pipeline opens positions on the
        # same SimulatedExchangeAdapter.
        if symbol not in self._sentinels:
            adapter = self._adapter_factory(exchange, mode=mode)
            self._adapters[symbol] = adapter
            self._bot_manager.set_adapter_for_symbol(symbol, adapter)
            # In shadow mode the Sentinel needs the trades repo so it
            # can monitor open shadow positions for SL/TP breaches on
            # every tick — the simulated adapter has no native SL/TP
            # orders. Live + paper modes get None: their SL/TP orders
            # live on the exchange.
            sentinel_trade_repo = self._repos.trades if self._shadow_mode else None
            sentinel = SentinelMonitor(
                adapter=adapter,
                event_bus=self._bus,
                symbol=symbol,
                timeframe=timeframe,
                trade_repo=sentinel_trade_repo,
            )
            # Activate the Task 11 escalation feedback loop: subscribe
            # this sentinel to SetupResult events so it learns whether
            # the analysis pipeline turned each SetupDetected into a
            # TRADE or a SKIP. Without this call, the per-symbol
            # readiness escalation state machine is dormant — Sentinel
            # behaves as before (full cooldown on every fire). Must be
            # called BEFORE `sentinel.run()` so the handler is registered
            # by the time the first SetupDetected can fire downstream.
            sentinel.subscribe_results()
            self._sentinels[symbol] = sentinel
            self._sentinel_tasks[symbol] = asyncio.create_task(
                self._run_sentinel_safe(symbol, sentinel)
            )

        # Start scheduled fallback for this bot
        cooldown = get_sentinel_cooldown(timeframe)
        interval = max(cooldown, 300)  # at least 5 minutes
        self._scheduled_tasks[bot_id] = asyncio.create_task(
            self._scheduled_loop(bot, interval)
        )

    async def _stop_sentinel(self, symbol: str) -> None:
        """Stop and clean up a sentinel for a symbol."""
        sentinel = self._sentinels.pop(symbol, None)
        if sentinel is not None:
            sentinel.stop()

        task = self._sentinel_tasks.pop(symbol, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _run_sentinel_safe(
        self, symbol: str, sentinel: SentinelMonitor
    ) -> None:
        """Run sentinel with auto-restart on crash (exponential backoff)."""
        backoff = _MIN_BACKOFF_SECONDS

        while self._running:
            try:
                await sentinel.run()
                # If run() returns normally (sentinel stopped), exit
                if not self._running:
                    break
                # Sentinel stopped itself — restart
                logger.warning(
                    f"BotRunner: sentinel for {symbol} stopped unexpectedly, "
                    f"restarting in {backoff}s"
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"BotRunner: sentinel for {symbol} crashed: {e}, "
                    f"restarting in {backoff}s",
                    exc_info=True,
                )

            if not self._running:
                break

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)

            # Reset sentinel state for restart
            sentinel._running = False

        logger.debug(f"BotRunner: sentinel loop for {symbol} exited")

    async def _scheduled_loop(self, bot: dict, interval: int) -> None:
        """Fallback: trigger analysis on schedule if sentinel didn't fire.

        This ensures bots run at least once per interval even if the
        sentinel's readiness score never reaches threshold.
        """
        bot_id = bot["id"]
        symbol = bot.get("symbol", "BTC-USDC")

        logger.info(
            f"BotRunner: scheduled loop for {bot_id} ({symbol}) "
            f"every {interval}s"
        )

        while self._running:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

            if not self._running:
                break

            logger.info(f"BotRunner: scheduled trigger for {bot_id} ({symbol})")
            try:
                await self._bot_manager.spawn_bot(symbol, source="scheduled")
            except Exception:
                logger.error(
                    f"BotRunner: scheduled spawn failed for {bot_id}",
                    exc_info=True,
                )
