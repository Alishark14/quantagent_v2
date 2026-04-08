"""Tests for the BotRunner production service.

All tests use mock repos, mock adapters, and mock sentinels to test
lifecycle management without real exchange or DB connections.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from engine.bot_manager import BotManager
from engine.events import InProcessBus, SetupDetected
from quantagent.runner import BotRunner, _MAX_BACKOFF_SECONDS, _MIN_BACKOFF_SECONDS


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------


class MockBotRepository:
    """In-memory bot repository for testing."""

    def __init__(self) -> None:
        self._bots: dict[str, dict] = {}

    async def save_bot(self, bot: dict) -> str:
        bot_id = bot.get("id") or str(uuid4())
        self._bots[bot_id] = {**bot, "id": bot_id}
        return bot_id

    async def get_bot(self, bot_id: str) -> dict | None:
        return self._bots.get(bot_id)

    async def get_bots_by_user(self, user_id: str) -> list[dict]:
        return [b for b in self._bots.values() if b.get("user_id") == user_id]

    async def update_bot_health(self, bot_id: str, health: dict) -> bool:
        if bot_id in self._bots:
            self._bots[bot_id]["last_health"] = health
            return True
        return False


class MockRepos:
    """Container matching the repos interface."""

    def __init__(self) -> None:
        self.bots = MockBotRepository()
        self.trades = MagicMock()
        self.cycles = MagicMock()
        self.rules = MagicMock()
        self.cross_bot = MagicMock()


def _make_bot_config(
    bot_id: str | None = None,
    symbol: str = "BTC-USDC",
    timeframe: str = "1h",
    exchange: str = "hyperliquid",
    user_id: str = "test-user",
) -> dict:
    """Create a bot config dict."""
    return {
        "id": bot_id or f"bot-{uuid4().hex[:8]}",
        "user_id": user_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange": exchange,
        "status": "active",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus():
    return InProcessBus()


@pytest.fixture
def repos():
    return MockRepos()


@pytest.fixture
def bot_manager(event_bus):
    def factory(symbol, bot_id):
        bot = MagicMock()
        bot.run = AsyncMock(return_value={
            "bot_id": bot_id,
            "status": "OK",
            "action": "SKIP",
        })
        return bot
    return BotManager(event_bus=event_bus, bot_factory=factory)


@pytest.fixture
def adapter_factory():
    """Returns a factory that creates mock exchange adapters."""
    adapters: dict[str, MagicMock] = {}

    def factory(exchange: str) -> MagicMock:
        if exchange not in adapters:
            adapter = MagicMock()
            adapter.fetch_ohlcv = AsyncMock(return_value=[])
            adapter.get_funding_rate = AsyncMock(return_value=0.0001)
            adapters[exchange] = adapter
        return adapters[exchange]

    return factory


@pytest.fixture
def runner(repos, adapter_factory, event_bus, bot_manager):
    return BotRunner(
        repos=repos,
        adapter_factory=adapter_factory,
        llm_provider=MagicMock(),
        event_bus=event_bus,
        bot_manager=bot_manager,
    )


# ---------------------------------------------------------------------------
# Start / lifecycle tests
# ---------------------------------------------------------------------------


class TestStartStop:
    """Test BotRunner start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, runner):
        await runner.start()
        assert runner.is_running is True
        await runner.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, runner):
        await runner.start()
        await runner.stop()
        assert runner.is_running is False

    @pytest.mark.asyncio
    async def test_start_twice_is_idempotent(self, runner):
        await runner.start()
        await runner.start()  # should warn but not crash
        assert runner.is_running is True
        await runner.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_safe(self, runner):
        await runner.stop()  # should be no-op


class TestStartWithBots:
    """Test loading bots at startup."""

    @pytest.mark.asyncio
    async def test_start_with_bots_creates_sentinels(self, runner):
        bots = [
            _make_bot_config(symbol="BTC-USDC"),
            _make_bot_config(symbol="ETH-USDC"),
        ]

        await runner.start_with_bots(bots)

        assert runner.sentinel_count == 2
        assert runner.get_sentinel("BTC-USDC") is not None
        assert runner.get_sentinel("ETH-USDC") is not None

        await runner.stop()

    @pytest.mark.asyncio
    async def test_start_with_bots_creates_scheduled_tasks(self, runner):
        bots = [
            _make_bot_config(bot_id="bot-1", symbol="BTC-USDC"),
            _make_bot_config(bot_id="bot-2", symbol="ETH-USDC"),
        ]

        await runner.start_with_bots(bots)

        assert runner.scheduled_task_count == 2

        await runner.stop()

    @pytest.mark.asyncio
    async def test_same_symbol_shares_sentinel(self, runner):
        """Two bots on the same symbol should share one sentinel."""
        bots = [
            _make_bot_config(bot_id="bot-1", symbol="BTC-USDC"),
            _make_bot_config(bot_id="bot-2", symbol="BTC-USDC"),
        ]

        await runner.start_with_bots(bots)

        # One sentinel for the shared symbol
        assert runner.sentinel_count == 1
        # But two scheduled tasks (one per bot)
        assert runner.scheduled_task_count == 2

        await runner.stop()

    @pytest.mark.asyncio
    async def test_start_with_empty_list(self, runner):
        await runner.start_with_bots([])
        assert runner.is_running is True
        assert runner.sentinel_count == 0
        await runner.stop()


# ---------------------------------------------------------------------------
# Add/remove bot tests
# ---------------------------------------------------------------------------


class TestAddBot:
    """Test dynamic bot addition."""

    @pytest.mark.asyncio
    async def test_add_bot_creates_sentinel_and_task(self, runner):
        await runner.start()

        bot = _make_bot_config(bot_id="new-bot", symbol="SOL-USDC")
        await runner.add_bot(bot)

        assert runner.sentinel_count == 1
        assert runner.get_sentinel("SOL-USDC") is not None
        assert runner.scheduled_task_count == 1

        await runner.stop()

    @pytest.mark.asyncio
    async def test_add_bot_saves_to_db(self, runner, repos):
        await runner.start()

        bot = _make_bot_config(bot_id="save-test", symbol="DOGE-USDC")
        await runner.add_bot(bot)

        saved = await repos.bots.get_bot("save-test")
        assert saved is not None
        assert saved["symbol"] == "DOGE-USDC"

        await runner.stop()

    @pytest.mark.asyncio
    async def test_add_bot_skips_save_if_exists(self, runner, repos):
        """If bot already exists in DB, don't duplicate."""
        await runner.start()

        bot = _make_bot_config(bot_id="existing-bot", symbol="BTC-USDC")
        await repos.bots.save_bot(bot)

        # Add same bot — should not crash
        await runner.add_bot(bot)

        assert runner.sentinel_count == 1
        await runner.stop()


class TestRemoveBot:
    """Test dynamic bot removal."""

    @pytest.mark.asyncio
    async def test_remove_bot_cleans_up(self, runner):
        await runner.start()

        bot = _make_bot_config(bot_id="rm-bot", symbol="BTC-USDC")
        await runner.add_bot(bot)
        assert runner.sentinel_count == 1

        await runner.remove_bot("rm-bot")

        # Sentinel and task should be gone
        assert runner.sentinel_count == 0
        assert runner.scheduled_task_count == 0

        await runner.stop()

    @pytest.mark.asyncio
    async def test_remove_bot_keeps_sentinel_if_shared(self, runner):
        """Removing one bot should keep the sentinel if another bot uses it."""
        await runner.start()

        bot1 = _make_bot_config(bot_id="bot-a", symbol="BTC-USDC")
        bot2 = _make_bot_config(bot_id="bot-b", symbol="BTC-USDC")
        await runner.add_bot(bot1)
        await runner.add_bot(bot2)
        assert runner.sentinel_count == 1
        assert runner.scheduled_task_count == 2

        await runner.remove_bot("bot-a")

        # Sentinel still running for bot-b
        assert runner.sentinel_count == 1
        assert runner.scheduled_task_count == 1

        await runner.stop()

    @pytest.mark.asyncio
    async def test_remove_nonexistent_bot_is_safe(self, runner):
        await runner.start()
        await runner.remove_bot("does-not-exist")  # should not crash
        await runner.stop()

    @pytest.mark.asyncio
    async def test_remove_last_bot_stops_sentinel(self, runner):
        await runner.start()

        bot = _make_bot_config(bot_id="only-bot", symbol="ETH-USDC")
        await runner.add_bot(bot)
        assert runner.get_sentinel("ETH-USDC") is not None

        await runner.remove_bot("only-bot")
        assert runner.get_sentinel("ETH-USDC") is None

        await runner.stop()


# ---------------------------------------------------------------------------
# Graceful shutdown tests
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    """Test that shutdown stops all sentinels and cancels tasks."""

    @pytest.mark.asyncio
    async def test_shutdown_stops_all_sentinels(self, runner):
        bots = [
            _make_bot_config(symbol="BTC-USDC"),
            _make_bot_config(symbol="ETH-USDC"),
            _make_bot_config(symbol="SOL-USDC"),
        ]
        await runner.start_with_bots(bots)
        assert runner.sentinel_count == 3

        await runner.stop()

        assert runner.sentinel_count == 0
        assert runner.scheduled_task_count == 0
        assert runner.is_running is False

    @pytest.mark.asyncio
    async def test_shutdown_cancels_scheduled_tasks(self, runner):
        bots = [
            _make_bot_config(bot_id="bot-1"),
            _make_bot_config(bot_id="bot-2"),
        ]
        await runner.start_with_bots(bots)

        await runner.stop()

        assert runner.scheduled_task_count == 0


# ---------------------------------------------------------------------------
# Sentinel auto-restart tests
# ---------------------------------------------------------------------------


class TestSentinelAutoRestart:
    """Test that sentinels auto-restart on crash with backoff."""

    @pytest.mark.asyncio
    async def test_sentinel_restarts_after_crash(self, runner):
        """Sentinel should restart after an exception."""
        crash_count = 0

        class CrashingSentinel:
            """Sentinel that crashes once then stops."""

            def __init__(self):
                self._running = False

            async def run(self):
                nonlocal crash_count
                self._running = True
                crash_count += 1
                if crash_count == 1:
                    raise RuntimeError("Simulated crash")
                # Second call: run briefly then stop
                await asyncio.sleep(0.05)
                self._running = False

            def stop(self):
                self._running = False

            @property
            def is_running(self):
                return self._running

        sentinel = CrashingSentinel()

        # Patch _MIN_BACKOFF to make test fast
        with patch("quantagent.runner._MIN_BACKOFF_SECONDS", 0.05):
            task = asyncio.create_task(
                runner._run_sentinel_safe("BTC-USDC", sentinel)
            )
            runner._running = True

            # Wait for crash + restart + normal exit
            await asyncio.sleep(0.3)
            runner._running = False
            sentinel.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert crash_count >= 2  # crashed once, restarted at least once

    @pytest.mark.asyncio
    async def test_sentinel_does_not_restart_after_stop(self, runner):
        """When runner._running is False, sentinel should not restart."""
        call_count = 0

        class OneRunSentinel:
            def __init__(self):
                self._running = False

            async def run(self):
                nonlocal call_count
                self._running = True
                call_count += 1
                await asyncio.sleep(0.02)
                self._running = False

            def stop(self):
                self._running = False

            @property
            def is_running(self):
                return self._running

        sentinel = OneRunSentinel()
        runner._running = False  # Not running — should not restart

        task = asyncio.create_task(
            runner._run_sentinel_safe("BTC-USDC", sentinel)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have exited immediately without running
        assert call_count == 0


# ---------------------------------------------------------------------------
# Scheduled fallback tests
# ---------------------------------------------------------------------------


class TestScheduledFallback:
    """Test the scheduled analysis loop."""

    @pytest.mark.asyncio
    async def test_scheduled_loop_fires_at_interval(self, runner, bot_manager):
        """Scheduled loop should trigger spawn_bot at the interval."""
        spawn_count = 0
        original_spawn = bot_manager.spawn_bot

        async def counting_spawn(symbol):
            nonlocal spawn_count
            spawn_count += 1
            return {"status": "OK", "action": "SKIP", "bot_id": "test"}

        bot_manager.spawn_bot = counting_spawn
        runner._running = True

        bot = _make_bot_config(symbol="BTC-USDC")

        task = asyncio.create_task(runner._scheduled_loop(bot, interval=0.1))

        # Wait for a few intervals
        await asyncio.sleep(0.35)
        runner._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have fired 2-3 times in 0.35s with 0.1s interval
        assert spawn_count >= 2

    @pytest.mark.asyncio
    async def test_scheduled_loop_stops_on_cancel(self, runner):
        runner._running = True

        bot = _make_bot_config(symbol="BTC-USDC")
        task = asyncio.create_task(runner._scheduled_loop(bot, interval=60))

        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have exited cleanly

    @pytest.mark.asyncio
    async def test_scheduled_loop_handles_spawn_error(self, runner, bot_manager):
        """Errors in spawn_bot should not crash the scheduled loop."""
        error_count = 0

        async def error_spawn(symbol):
            nonlocal error_count
            error_count += 1
            raise RuntimeError("spawn failed")

        bot_manager.spawn_bot = error_spawn
        runner._running = True

        bot = _make_bot_config(symbol="BTC-USDC")
        task = asyncio.create_task(runner._scheduled_loop(bot, interval=0.05))

        await asyncio.sleep(0.2)
        runner._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have attempted multiple spawns despite errors
        assert error_count >= 2


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """Test the full add → run → remove → stop lifecycle."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, runner, repos):
        # Start empty
        await runner.start()
        assert runner.sentinel_count == 0

        # Add two bots on different symbols
        bot1 = _make_bot_config(bot_id="bot-1", symbol="BTC-USDC")
        bot2 = _make_bot_config(bot_id="bot-2", symbol="ETH-USDC")
        await runner.add_bot(bot1)
        await runner.add_bot(bot2)
        assert runner.sentinel_count == 2
        assert runner.scheduled_task_count == 2

        # Both saved to DB
        assert await repos.bots.get_bot("bot-1") is not None
        assert await repos.bots.get_bot("bot-2") is not None

        # Remove one
        await runner.remove_bot("bot-1")
        assert runner.sentinel_count == 1  # ETH sentinel remains
        assert runner.scheduled_task_count == 1

        # Stop everything
        await runner.stop()
        assert runner.sentinel_count == 0
        assert runner.scheduled_task_count == 0
        assert runner.is_running is False

    @pytest.mark.asyncio
    async def test_add_multiple_then_stop_all(self, runner):
        await runner.start()

        for i in range(5):
            bot = _make_bot_config(bot_id=f"bot-{i}", symbol=f"SYM{i}-USDC")
            await runner.add_bot(bot)

        assert runner.sentinel_count == 5
        assert runner.scheduled_task_count == 5

        await runner.stop()

        assert runner.sentinel_count == 0
        assert runner.scheduled_task_count == 0


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Test the CLI entry point."""

    def test_main_help(self, capsys):
        import sys
        sys.argv = ["quantagent", "--help"]

        from quantagent.main import main
        main()

        captured = capsys.readouterr()
        assert "QuantAgent v2" in captured.out
        assert "run" in captured.out

    def test_main_unknown_command(self, capsys):
        import sys
        sys.argv = ["quantagent", "foobar"]

        from quantagent.main import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Sentinel escalation wiring (Task 11 production-mode activation)
# ---------------------------------------------------------------------------


class TestSentinelEscalationWiring:
    """Verify BotRunner activates Sentinel's SetupResult feedback loop.

    The Sentinel's escalating-readiness state machine is dormant until
    `subscribe_results()` is called. Without it, every spawned bot's
    SKIP outcome would be invisible to the Sentinel and the threshold
    would never ratchet up. These tests verify the wiring lives in
    `BotRunner._register_bot` so production runs get escalation
    automatically — a black-box check that publishing a SetupResult
    on the bus actually mutates the registered sentinel's threshold,
    which can only happen if `subscribe_results()` ran during
    construction.
    """

    @pytest.mark.asyncio
    async def test_add_bot_activates_escalation_feedback(self, runner, event_bus):
        from engine.events import SetupResult

        await runner.start()
        bot = _make_bot_config(bot_id="esc-bot", symbol="LINK-USDC")
        await runner.add_bot(bot)

        sentinel = runner.get_sentinel("LINK-USDC")
        assert sentinel is not None
        baseline = sentinel.current_threshold()

        # Publish a SKIP result on the bus. If subscribe_results() ran,
        # the sentinel's threshold ratchets up; if it didn't, the
        # threshold stays at the baseline.
        await event_bus.publish(SetupResult(
            source="test", symbol="LINK-USDC", outcome="SKIP",
            action="SKIP", bot_id="esc-bot", conviction_score=0.2,
        ))

        escalated = sentinel.current_threshold()
        assert escalated > baseline, (
            "BotRunner._register_bot must call sentinel.subscribe_results() "
            "to activate the Task 11 escalation feedback loop"
        )

        await runner.stop()

    @pytest.mark.asyncio
    async def test_setup_result_for_other_symbol_does_not_escalate(
        self, runner, event_bus
    ):
        """Per-symbol filter still works through the bus dispatch."""
        from engine.events import SetupResult

        await runner.start()
        await runner.add_bot(_make_bot_config(
            bot_id="btc-bot", symbol="BTC-USDC",
        ))
        await runner.add_bot(_make_bot_config(
            bot_id="eth-bot", symbol="ETH-USDC",
        ))

        btc = runner.get_sentinel("BTC-USDC")
        eth = runner.get_sentinel("ETH-USDC")
        eth_baseline = eth.current_threshold()

        # SKIP only on BTC
        await event_bus.publish(SetupResult(
            source="test", symbol="BTC-USDC", outcome="SKIP",
            action="SKIP", bot_id="btc-bot", conviction_score=0.2,
        ))

        # BTC escalated, ETH untouched
        assert btc.current_threshold() > eth_baseline
        assert eth.current_threshold() == pytest.approx(eth_baseline)

        await runner.stop()

    @pytest.mark.asyncio
    async def test_trade_outcome_resets_after_escalation(self, runner, event_bus):
        """End-to-end SKIP -> escalate -> TRADE -> reset via the runner's bus."""
        from engine.events import SetupResult

        await runner.start()
        await runner.add_bot(_make_bot_config(
            bot_id="reset-bot", symbol="AVAX-USDC",
        ))
        sentinel = runner.get_sentinel("AVAX-USDC")
        baseline = sentinel.current_threshold()

        # Two SKIPs -> escalated twice
        for _ in range(2):
            await event_bus.publish(SetupResult(
                source="test", symbol="AVAX-USDC", outcome="SKIP",
                action="SKIP", bot_id="reset-bot", conviction_score=0.2,
            ))
        assert sentinel.current_threshold() > baseline

        # TRADE -> reset
        await event_bus.publish(SetupResult(
            source="test", symbol="AVAX-USDC", outcome="TRADE",
            action="LONG", bot_id="reset-bot", conviction_score=0.78,
        ))
        assert sentinel.current_threshold() == pytest.approx(baseline)

        await runner.stop()
