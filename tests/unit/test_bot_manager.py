"""Unit tests for BotManager — spawning and lifecycle management."""

from __future__ import annotations

import asyncio

import pytest

from engine.bot_manager import BotManager
from engine.config import TradingConfig
from engine.events import InProcessBus, SetupDetected
from engine.execution.executor import Executor
from engine.trader_bot import TraderBot
from engine.types import TradeAction
from tests.unit.test_trader_bot import MockPipeline, MockSLAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_action() -> TradeAction:
    return TradeAction(
        action="SKIP", conviction_score=0.3, position_size=None,
        sl_price=None, tp1_price=None, tp2_price=None,
        rr_ratio=None, atr_multiplier=None, reasoning="test", raw_output="",
    )


def _make_factory(action: TradeAction | None = None, raises: bool = False):
    """Create a bot factory that builds TraderBots with mock pipeline."""
    def factory(symbol: str, bot_id: str) -> TraderBot:
        adapter = MockSLAdapter()
        pipeline = MockPipeline(action=action or _skip_action(), raises=raises)
        executor = Executor(adapter, InProcessBus(), TradingConfig(symbol=symbol))
        return TraderBot(bot_id, pipeline, executor)
    return factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBotManager:

    @pytest.mark.asyncio
    async def test_spawn_bot_returns_result(self) -> None:
        bus = InProcessBus()
        manager = BotManager(bus, _make_factory())

        result = await manager.spawn_bot("BTC-USDC")

        assert result["status"] == "OK"
        assert result["action"] == "SKIP"

    @pytest.mark.asyncio
    async def test_spawn_bot_stored_in_results(self) -> None:
        bus = InProcessBus()
        manager = BotManager(bus, _make_factory())

        await manager.spawn_bot("BTC-USDC")

        assert len(manager.results) == 1

    @pytest.mark.asyncio
    async def test_concurrent_limit_blocks_second(self) -> None:
        bus = InProcessBus()

        # Use a slow factory to simulate a long-running bot
        slow_event = asyncio.Event()

        def slow_factory(symbol, bot_id):
            class SlowPipeline:
                _config = TradingConfig(symbol=symbol)
                async def run_cycle(self):
                    await slow_event.wait()
                    return _skip_action()
            adapter = MockSLAdapter()
            executor = Executor(adapter, InProcessBus(), TradingConfig(symbol=symbol))
            return TraderBot(bot_id, SlowPipeline(), executor)

        manager = BotManager(bus, slow_factory, max_concurrent_per_symbol=1)

        # Start first bot (it blocks on slow_event)
        task = asyncio.create_task(manager.spawn_bot("BTC-USDC"))
        await asyncio.sleep(0.01)

        # Second spawn should be skipped
        result2 = await manager.spawn_bot("BTC-USDC")
        assert result2["status"] == "SKIPPED"

        # Unblock first bot
        slow_event.set()
        await task

    @pytest.mark.asyncio
    async def test_different_symbols_independent(self) -> None:
        bus = InProcessBus()
        manager = BotManager(bus, _make_factory(), max_concurrent_per_symbol=1)

        r1 = await manager.spawn_bot("BTC-USDC")
        r2 = await manager.spawn_bot("ETH-USDC")

        assert r1["status"] == "OK"
        assert r2["status"] == "OK"

    @pytest.mark.asyncio
    async def test_cleanup_after_completion(self) -> None:
        bus = InProcessBus()
        manager = BotManager(bus, _make_factory())

        await manager.spawn_bot("BTC-USDC")

        assert manager.active_count("BTC-USDC") == 0
        assert manager.total_active == 0

    @pytest.mark.asyncio
    async def test_error_does_not_leak_active_slot(self) -> None:
        """TraderBot catches pipeline errors and returns ERROR status.
        Active slot must still be cleaned up."""
        bus = InProcessBus()
        manager = BotManager(bus, _make_factory(raises=True))

        result = await manager.spawn_bot("BTC-USDC")

        assert result["status"] == "ERROR"  # TraderBot catches, returns ERROR
        assert manager.active_count("BTC-USDC") == 0

    @pytest.mark.asyncio
    async def test_setup_detected_triggers_spawn(self) -> None:
        bus = InProcessBus()
        manager = BotManager(bus, _make_factory())
        manager.subscribe()

        await bus.publish(SetupDetected(
            source="test", symbol="BTC-USDC", readiness=0.82, conditions=["RSI cross"],
        ))

        # Give the spawned task time to complete
        await asyncio.sleep(0.1)

        assert len(manager.results) == 1
        assert manager.results[0]["status"] == "OK"
