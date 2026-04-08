"""Unit tests for BotManager — spawning and lifecycle management."""

from __future__ import annotations

import asyncio

import pytest

from engine.bot_manager import BotManager
from engine.config import TradingConfig
from engine.events import InProcessBus, SetupDetected, SetupResult
from engine.execution.executor import Executor
from engine.trader_bot import TraderBot
from engine.types import TradeAction
from tests.unit.test_trader_bot import MockPipeline, MockSLAdapter


# ---------------------------------------------------------------------------
# FakeBot: returns a pre-baked result dict, bypassing TraderBot entirely.
#
# Used to test BotManager's SetupResult emission classification logic
# without having to fully wire an Executor + adapter to produce the
# right action / order_result combinations.
# ---------------------------------------------------------------------------


class FakeBot:
    """Bot stub that returns a pre-built result dict on `.run()`.

    BotManager doesn't actually require a TraderBot instance — it just
    calls `await bot.run()` and treats the dict as the result. This
    keeps the SetupResult tests focused on the classification logic,
    not the executor wiring.
    """

    def __init__(self, result: dict, raises: bool = False) -> None:
        self._result = result
        self._raises = raises

    async def run(self) -> dict:
        if self._raises:
            raise RuntimeError("synthetic crash from FakeBot")
        return self._result


def _fake_factory(result: dict, raises: bool = False):
    def factory(symbol: str, bot_id: str) -> FakeBot:
        # Stamp the bot_id onto the result so the SetupResult event
        # carries the right id (production TraderBot does the same).
        stamped = dict(result)
        stamped.setdefault("bot_id", bot_id)
        return FakeBot(stamped, raises=raises)
    return factory


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


# ---------------------------------------------------------------------------
# Tests: SetupResult emission (Task 11 — Sentinel feedback loop)
# ---------------------------------------------------------------------------


class TestBotManagerSetupResultClassification:
    """Pure-function tests of `_classify_outcome`.

    No bus dispatch — just verifies the TRADE / SKIP mapping rules.
    """

    def test_long_with_successful_order_is_trade(self) -> None:
        result = {"action": "LONG", "order_result": {"success": True}}
        assert BotManager._classify_outcome(result) == "TRADE"

    def test_short_with_successful_order_is_trade(self) -> None:
        result = {"action": "SHORT", "order_result": {"success": True}}
        assert BotManager._classify_outcome(result) == "TRADE"

    def test_add_long_with_successful_order_is_trade(self) -> None:
        result = {"action": "ADD_LONG", "order_result": {"success": True}}
        assert BotManager._classify_outcome(result) == "TRADE"

    def test_add_short_with_successful_order_is_trade(self) -> None:
        result = {"action": "ADD_SHORT", "order_result": {"success": True}}
        assert BotManager._classify_outcome(result) == "TRADE"

    def test_long_with_failed_order_is_skip(self) -> None:
        result = {"action": "LONG", "order_result": {"success": False}}
        assert BotManager._classify_outcome(result) == "SKIP"

    def test_long_with_no_order_is_skip(self) -> None:
        result = {"action": "LONG", "order_result": None}
        assert BotManager._classify_outcome(result) == "SKIP"

    def test_skip_action_is_skip(self) -> None:
        result = {"action": "SKIP", "order_result": None}
        assert BotManager._classify_outcome(result) == "SKIP"

    def test_hold_action_is_skip(self) -> None:
        result = {"action": "HOLD", "order_result": None}
        assert BotManager._classify_outcome(result) == "SKIP"

    def test_close_all_is_skip_not_trade(self) -> None:
        """CLOSE_ALL closes a position rather than opening one — no rationale
        to reset Sentinel's escalation, so it classifies as SKIP."""
        result = {"action": "CLOSE_ALL", "order_result": {"success": True}}
        assert BotManager._classify_outcome(result) == "SKIP"

    def test_crashed_bot_is_skip(self) -> None:
        result = {"action": "SKIP", "status": "CRASH", "error": "boom"}
        assert BotManager._classify_outcome(result) == "SKIP"


class TestBotManagerSetupResultEmission:
    """End-to-end SetupResult publication via the bus."""

    @pytest.mark.asyncio
    async def test_spawn_bot_emits_skip_for_skip_action(self) -> None:
        bus = InProcessBus()
        events: list[SetupResult] = []
        bus.subscribe(SetupResult, lambda e: events.append(e))

        manager = BotManager(bus, _fake_factory({
            "status": "OK",
            "action": "SKIP",
            "conviction_score": 0.25,
            "order_result": None,
        }))
        await manager.spawn_bot("BTC-USDC")

        assert len(events) == 1
        evt = events[0]
        assert evt.symbol == "BTC-USDC"
        assert evt.outcome == "SKIP"
        assert evt.action == "SKIP"
        assert evt.conviction_score == pytest.approx(0.25)
        assert evt.bot_id.startswith("bot-BTC-USDC-")
        assert evt.source == "bot_manager"

    @pytest.mark.asyncio
    async def test_spawn_bot_emits_trade_for_long_with_success(self) -> None:
        bus = InProcessBus()
        events: list[SetupResult] = []
        bus.subscribe(SetupResult, lambda e: events.append(e))

        manager = BotManager(bus, _fake_factory({
            "status": "OK",
            "action": "LONG",
            "conviction_score": 0.78,
            "order_result": {"success": True, "order_id": "x", "fill_price": 65000.0,
                             "fill_size": 0.1, "error": None},
        }))
        await manager.spawn_bot("BTC-USDC")

        assert len(events) == 1
        assert events[0].outcome == "TRADE"
        assert events[0].action == "LONG"

    @pytest.mark.asyncio
    async def test_spawn_bot_emits_skip_for_long_with_failed_order(self) -> None:
        bus = InProcessBus()
        events: list[SetupResult] = []
        bus.subscribe(SetupResult, lambda e: events.append(e))

        manager = BotManager(bus, _fake_factory({
            "status": "OK",
            "action": "LONG",
            "conviction_score": 0.78,
            "order_result": {"success": False, "order_id": None, "fill_price": None,
                             "fill_size": None, "error": "rejected"},
        }))
        await manager.spawn_bot("BTC-USDC")

        assert len(events) == 1
        assert events[0].outcome == "SKIP"
        assert events[0].action == "LONG"

    @pytest.mark.asyncio
    async def test_spawn_bot_emits_skip_for_close_all(self) -> None:
        bus = InProcessBus()
        events: list[SetupResult] = []
        bus.subscribe(SetupResult, lambda e: events.append(e))

        manager = BotManager(bus, _fake_factory({
            "status": "OK",
            "action": "CLOSE_ALL",
            "conviction_score": 0.4,
            "order_result": {"success": True, "order_id": "c", "fill_price": 65000.0,
                             "fill_size": 0.1, "error": None},
        }))
        await manager.spawn_bot("BTC-USDC")

        assert len(events) == 1
        assert events[0].outcome == "SKIP"  # CLOSE_ALL is not TRADE
        assert events[0].action == "CLOSE_ALL"

    @pytest.mark.asyncio
    async def test_spawn_bot_emits_skip_when_bot_crashes(self) -> None:
        bus = InProcessBus()
        events: list[SetupResult] = []
        bus.subscribe(SetupResult, lambda e: events.append(e))

        manager = BotManager(bus, _fake_factory(
            result={"action": "LONG"},  # not used; FakeBot raises
            raises=True,
        ))
        await manager.spawn_bot("BTC-USDC")

        assert len(events) == 1
        assert events[0].outcome == "SKIP"
        # Crash path: action ends up as "SKIP" in the synthesized result
        assert events[0].action == "SKIP"

    @pytest.mark.asyncio
    async def test_spawn_bot_does_not_emit_when_concurrency_blocked(self) -> None:
        """A spawn rejected for concurrency reasons must NOT publish SetupResult.

        The pipeline never ran, so there's no SKIP/TRADE outcome to feed
        back to the Sentinel — the situation is "we already have a bot
        for this symbol", which is orthogonal to escalation.
        """
        bus = InProcessBus()
        events: list[SetupResult] = []
        bus.subscribe(SetupResult, lambda e: events.append(e))

        # Use a slow first bot so the second hits the concurrency cap.
        slow_event = asyncio.Event()

        async def slow_run(self):
            await slow_event.wait()
            return {"action": "SKIP", "conviction_score": 0.1, "order_result": None}

        # Patch FakeBot.run for the slow first bot only
        def slow_factory(symbol, bot_id):
            bot = FakeBot({"action": "SKIP", "order_result": None})
            bot.run = lambda: slow_run(bot)
            return bot

        manager = BotManager(bus, slow_factory, max_concurrent_per_symbol=1)
        first_task = asyncio.create_task(manager.spawn_bot("BTC-USDC"))
        await asyncio.sleep(0.01)  # let first task start

        # Second spawn — should be rejected by the concurrency limit
        result2 = await manager.spawn_bot("BTC-USDC")
        assert result2["status"] == "SKIPPED"

        # No SetupResult should have fired yet — first bot still running,
        # second was concurrency-blocked.
        assert len(events) == 0

        # Unblock first bot — that one SHOULD emit a SetupResult.
        slow_event.set()
        await first_task
        assert len(events) == 1
        assert events[0].outcome == "SKIP"

    @pytest.mark.asyncio
    async def test_event_triggered_spawn_emits_setup_result(self) -> None:
        """SetupDetected → BotManager spawns bot → SetupResult emitted."""
        bus = InProcessBus()
        results: list[SetupResult] = []
        bus.subscribe(SetupResult, lambda e: results.append(e))

        manager = BotManager(bus, _fake_factory({
            "status": "OK",
            "action": "LONG",
            "conviction_score": 0.7,
            "order_result": {"success": True, "order_id": "x", "fill_price": 65000.0,
                             "fill_size": 0.1, "error": None},
        }))
        manager.subscribe()

        await bus.publish(SetupDetected(
            source="test", symbol="BTC-USDC", readiness=0.82, conditions=["RSI cross"],
        ))
        # Give the spawned task time to complete
        for _ in range(50):
            if results:
                break
            await asyncio.sleep(0.01)

        assert len(results) == 1
        assert results[0].outcome == "TRADE"
        assert results[0].symbol == "BTC-USDC"
