"""Unit tests for TraderBot — ephemeral analysis + execution worker."""

from __future__ import annotations

import pytest

from engine.config import TradingConfig
from engine.events import InProcessBus
from engine.execution.executor import Executor
from engine.trader_bot import TraderBot
from engine.types import OrderResult, TradeAction
from sentinel.position_manager import PositionManager
from tests.unit.test_position_manager import MockSLAdapter


# ---------------------------------------------------------------------------
# Mock pipeline
# ---------------------------------------------------------------------------

class MockPipeline:
    """Minimal pipeline mock that returns a configurable TradeAction."""

    def __init__(self, action: TradeAction | None = None, raises: bool = False) -> None:
        self._action = action or TradeAction(
            action="SKIP", conviction_score=0.3, position_size=None,
            sl_price=None, tp1_price=None, tp2_price=None,
            rr_ratio=None, atr_multiplier=None, reasoning="mock", raw_output="",
        )
        self._raises = raises
        self._config = TradingConfig(symbol="BTC-USDC", timeframe="1h")

    async def run_cycle(self) -> TradeAction:
        if self._raises:
            raise RuntimeError("pipeline crashed")
        return self._action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _long_action() -> TradeAction:
    return TradeAction(
        action="LONG", conviction_score=0.72, position_size=650.0,
        sl_price=64000.0, tp1_price=66000.0, tp2_price=67500.0,
        rr_ratio=1.5, atr_multiplier=1.2, reasoning="test", raw_output="",
    )


def _skip_action() -> TradeAction:
    return TradeAction(
        action="SKIP", conviction_score=0.35, position_size=None,
        sl_price=None, tp1_price=None, tp2_price=None,
        rr_ratio=None, atr_multiplier=None, reasoning="low conviction", raw_output="",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTraderBot:

    @pytest.mark.asyncio
    async def test_skip_action_no_execution(self) -> None:
        adapter = MockSLAdapter()
        pipeline = MockPipeline(action=_skip_action())
        executor = Executor(adapter, InProcessBus(), TradingConfig())
        bot = TraderBot("bot-001", pipeline, executor)

        result = await bot.run()

        assert result["status"] == "OK"
        assert result["action"] == "SKIP"
        assert result["order_result"] is None
        assert len(adapter.modify_sl_calls) == 0

    @pytest.mark.asyncio
    async def test_long_action_executes(self) -> None:
        adapter = MockSLAdapter()
        pipeline = MockPipeline(action=_long_action())
        executor = Executor(adapter, InProcessBus(), TradingConfig())
        bot = TraderBot("bot-002", pipeline, executor)

        result = await bot.run()

        assert result["status"] == "OK"
        assert result["action"] == "LONG"
        assert result["order_result"] is not None
        assert result["order_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_pipeline_crash_returns_error(self) -> None:
        adapter = MockSLAdapter()
        pipeline = MockPipeline(raises=True)
        executor = Executor(adapter, InProcessBus(), TradingConfig())
        bot = TraderBot("bot-003", pipeline, executor)

        result = await bot.run()

        assert result["status"] == "ERROR"
        assert result["action"] == "SKIP"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_result_has_duration(self) -> None:
        pipeline = MockPipeline(action=_skip_action())
        executor = Executor(MockSLAdapter(), InProcessBus(), TradingConfig())
        bot = TraderBot("bot-004", pipeline, executor)

        result = await bot.run()

        assert "duration_ms" in result
        assert result["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_result_has_bot_id(self) -> None:
        pipeline = MockPipeline(action=_skip_action())
        executor = Executor(MockSLAdapter(), InProcessBus(), TradingConfig())
        bot = TraderBot("my-bot-id", pipeline, executor)

        result = await bot.run()

        assert result["bot_id"] == "my-bot-id"

    @pytest.mark.asyncio
    async def test_long_registers_with_position_manager(self) -> None:
        adapter = MockSLAdapter()
        bus = InProcessBus()
        pm = PositionManager(adapter, bus)
        pipeline = MockPipeline(action=_long_action())
        executor = Executor(adapter, bus, TradingConfig())
        bot = TraderBot("bot-005", pipeline, executor, position_manager=pm)

        await bot.run()

        pos = pm.get_position("BTC-USDC")
        assert pos is not None
        assert pos.direction == "long"

    @pytest.mark.asyncio
    async def test_skip_does_not_register(self) -> None:
        adapter = MockSLAdapter()
        bus = InProcessBus()
        pm = PositionManager(adapter, bus)
        pipeline = MockPipeline(action=_skip_action())
        executor = Executor(adapter, bus, TradingConfig())
        bot = TraderBot("bot-006", pipeline, executor, position_manager=pm)

        await bot.run()

        assert pm.get_position("BTC-USDC") is None
