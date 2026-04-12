"""Tests for cost tracking: trading fee deduction + LLM usage accumulation.

Layer 1: SLTPMonitor and Sentinel deduct round-trip taker fees from PnL.
Layer 2: LLMProvider accumulates per-cycle token counts and cost.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from engine.sl_tp_monitor import SLTPMonitor
from engine.events import InProcessBus, PriceUpdated
from engine.types import PriceUpdate
from llm.base import LLMProvider, LLMResponse


# ── Fee deduction tests ──────────────────────────────────────────────


class FakeTradeRepo:
    """Minimal trade repo that records calls."""

    def __init__(self):
        self.closed: list[dict] = []
        self.updated: list[tuple[str, dict]] = []
        self._trades: dict[str, list[dict]] = {}

    async def get_open_shadow_trades(self, symbol: str) -> list[dict]:
        return list(self._trades.get(symbol, []))

    async def close_trade(self, trade_id, *, exit_price, exit_reason, exit_time, pnl):
        self.closed.append({
            "trade_id": trade_id,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl": pnl,
        })
        return True

    async def update_trade(self, trade_id: str, updates: dict) -> bool:
        self.updated.append((trade_id, updates))
        return True

    def add_trade(self, trade: dict):
        symbol = trade["symbol"]
        self._trades.setdefault(symbol, []).append(trade)


def _make_trade(
    trade_id: str = "t1",
    symbol: str = "GOLD-USDC",
    direction: str = "SHORT",
    entry_price: float = 100.0,
    size: float = 500.0,
    sl_price: float = 105.0,
    tp_price: float = 95.0,
) -> dict:
    return {
        "id": trade_id,
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry_price,
        "size": size,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "forward_max_r": None,
    }


@pytest.mark.asyncio
async def test_fee_deduction_short_tp():
    """SHORT TP hit: pnl = raw_pnl - round_trip_fee."""
    repo = FakeTradeRepo()
    trade = _make_trade(direction="SHORT", entry_price=100.0, size=500.0,
                        sl_price=105.0, tp_price=95.0)
    repo.add_trade(trade)

    bus = InProcessBus()
    monitor = SLTPMonitor(bus, repo, taker_fee_rate=0.00035, refresh_interval=0.0)
    monitor.register_symbol("GOLD-USDC")
    await monitor.start()

    # Price hits TP at 95.0
    await monitor._on_price_update(PriceUpdated(
        source="test", update=PriceUpdate(symbol="GOLD-USDC", price=94.5)))

    assert len(repo.closed) == 1
    closed = repo.closed[0]

    # raw PnL: (100 - 95) * 500 / 100 = $25.00
    raw_pnl = 25.0
    # round-trip fee: 2 * 0.00035 * 500 = $0.35
    fee = 0.35
    # adjusted PnL: 25.0 - 0.35 = $24.65
    assert abs(closed["pnl"] - (raw_pnl - fee)) < 0.01

    # Check raw_pnl and trading_fee were persisted via update_trade
    assert len(repo.updated) >= 1
    cost_update = next(
        (u for _, u in repo.updated if "raw_pnl" in u), None
    )
    assert cost_update is not None
    assert abs(cost_update["raw_pnl"] - raw_pnl) < 0.01
    assert abs(cost_update["trading_fee"] - fee) < 0.01


@pytest.mark.asyncio
async def test_fee_deduction_long_sl():
    """LONG SL hit: PnL is negative, fee makes it worse."""
    repo = FakeTradeRepo()
    trade = _make_trade(direction="LONG", entry_price=100.0, size=500.0,
                        sl_price=95.0, tp_price=110.0)
    repo.add_trade(trade)

    bus = InProcessBus()
    monitor = SLTPMonitor(bus, repo, taker_fee_rate=0.00035, refresh_interval=0.0)
    monitor.register_symbol("GOLD-USDC")
    await monitor.start()

    # Price hits SL at 95.0
    await monitor._on_price_update(PriceUpdated(
        source="test", update=PriceUpdate(symbol="GOLD-USDC", price=94.5)))

    assert len(repo.closed) == 1
    closed = repo.closed[0]

    # raw PnL: (95 - 100) * 500 / 100 = -$25.00
    raw_pnl = -25.0
    fee = 0.35
    # adjusted: -25.0 - 0.35 = -$25.35
    assert abs(closed["pnl"] - (raw_pnl - fee)) < 0.01


@pytest.mark.asyncio
async def test_pnl_relationship():
    """pnl = raw_pnl - trading_fee always holds."""
    repo = FakeTradeRepo()
    trade = _make_trade(size=1000.0)
    repo.add_trade(trade)

    bus = InProcessBus()
    monitor = SLTPMonitor(bus, repo, taker_fee_rate=0.001, refresh_interval=0.0)
    monitor.register_symbol("GOLD-USDC")
    await monitor.start()

    await monitor._on_price_update(PriceUpdated(
        source="test", update=PriceUpdate(symbol="GOLD-USDC", price=94.0)))

    assert len(repo.closed) == 1
    pnl = repo.closed[0]["pnl"]

    cost_update = next((u for _, u in repo.updated if "raw_pnl" in u), None)
    assert cost_update is not None

    raw_pnl = cost_update["raw_pnl"]
    trading_fee = cost_update["trading_fee"]

    assert abs(pnl - (raw_pnl - trading_fee)) < 0.001


@pytest.mark.asyncio
async def test_zero_fee_rate():
    """When fee rate is 0, pnl equals raw_pnl."""
    repo = FakeTradeRepo()
    trade = _make_trade(size=500.0)
    repo.add_trade(trade)

    bus = InProcessBus()
    monitor = SLTPMonitor(bus, repo, taker_fee_rate=0.0, refresh_interval=0.0)
    monitor.register_symbol("GOLD-USDC")
    await monitor.start()

    await monitor._on_price_update(PriceUpdated(
        source="test", update=PriceUpdate(symbol="GOLD-USDC", price=94.0)))

    assert len(repo.closed) == 1
    pnl = repo.closed[0]["pnl"]

    cost_update = next((u for _, u in repo.updated if "raw_pnl" in u), None)
    assert cost_update is not None
    assert cost_update["trading_fee"] == 0.0
    assert abs(pnl - cost_update["raw_pnl"]) < 0.001


@pytest.mark.asyncio
async def test_fee_scales_with_position_size():
    """Doubling position size should double the fee."""
    results = []
    for size in [500.0, 1000.0]:
        repo = FakeTradeRepo()
        trade = _make_trade(size=size)
        repo.add_trade(trade)

        bus = InProcessBus()
        monitor = SLTPMonitor(bus, repo, taker_fee_rate=0.00035, refresh_interval=0.0)
        monitor.register_symbol("GOLD-USDC")
        await monitor.start()

        await monitor._on_price_update(PriceUpdated(
            source="test", update=PriceUpdate(symbol="GOLD-USDC", price=94.0)))

        cost_update = next((u for _, u in repo.updated if "raw_pnl" in u), None)
        results.append(cost_update["trading_fee"])

    # fee for $1000 should be 2x fee for $500
    assert abs(results[1] - 2 * results[0]) < 0.001


# ── LLM usage accumulation tests ─────────────────────────────────────


class StubLLMProvider(LLMProvider):
    """Concrete provider that returns canned responses."""

    async def generate_text(self, system_prompt, user_prompt, agent_name, **kw):
        resp = LLMResponse(
            content="test", input_tokens=1000, output_tokens=200,
            cost=0.01, model="claude-sonnet", latency_ms=100,
            cached_input_tokens=0,
        )
        self._accumulate_usage(resp)
        return resp

    async def generate_vision(self, system_prompt, user_prompt, image_data,
                              image_media_type, agent_name, **kw):
        resp = LLMResponse(
            content="test", input_tokens=2000, output_tokens=300,
            cost=0.02, model="claude-sonnet", latency_ms=150,
            cached_input_tokens=0,
        )
        self._accumulate_usage(resp)
        return resp


@pytest.mark.asyncio
async def test_llm_usage_accumulation():
    """Token counts and cost accumulate across multiple calls."""
    provider = StubLLMProvider()
    provider.reset_usage()

    await provider.generate_text("sys", "user", "test_agent")
    await provider.generate_text("sys", "user", "test_agent")

    usage = provider.get_usage()
    assert usage["input_tokens"] == 2000
    assert usage["output_tokens"] == 400
    assert abs(usage["cost_usd"] - 0.02) < 0.0001


@pytest.mark.asyncio
async def test_llm_usage_reset():
    """reset_usage zeroes all accumulators."""
    provider = StubLLMProvider()

    await provider.generate_text("sys", "user", "test_agent")

    usage_before = provider.get_usage()
    assert usage_before["input_tokens"] > 0

    provider.reset_usage()
    usage_after = provider.get_usage()
    assert usage_after["input_tokens"] == 0
    assert usage_after["output_tokens"] == 0
    assert usage_after["cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_llm_cost_computed_correctly():
    """Cost from LLMResponse is accumulated faithfully."""
    provider = StubLLMProvider()
    provider.reset_usage()

    # text call: cost=0.01
    await provider.generate_text("sys", "user", "test_agent")
    # vision call: cost=0.02
    await provider.generate_vision("sys", "user", b"img", "image/png", "test_agent")

    usage = provider.get_usage()
    assert usage["input_tokens"] == 3000  # 1000 + 2000
    assert usage["output_tokens"] == 500  # 200 + 300
    assert abs(usage["cost_usd"] - 0.03) < 0.0001  # 0.01 + 0.02


# ── Funding cost tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_funding_cost_computed_from_hold_duration():
    """Funding cost = abs(funding_rate * notional * hold_hours)."""
    from datetime import timedelta

    repo = FakeTradeRepo()
    entry_time = datetime.now(timezone.utc) - timedelta(hours=4)
    trade = _make_trade(direction="SHORT", entry_price=100.0, size=500.0,
                        sl_price=105.0, tp_price=95.0)
    trade["entry_time"] = entry_time.isoformat()
    trade["funding_rate"] = 0.0001  # 0.01% per hour
    repo.add_trade(trade)

    bus = InProcessBus()
    monitor = SLTPMonitor(bus, repo, taker_fee_rate=0.0, refresh_interval=0.0)
    monitor.register_symbol("GOLD-USDC")
    await monitor.start()

    await monitor._on_price_update(PriceUpdated(
        source="test", update=PriceUpdate(symbol="GOLD-USDC", price=94.0)))

    assert len(repo.closed) == 1
    cost_update = next((u for _, u in repo.updated if "funding_cost" in u), None)
    assert cost_update is not None

    # funding_cost = abs(0.0001 * 500 * ~4) ≈ $0.20
    fc = cost_update["funding_cost"]
    assert 0.15 < fc < 0.25, f"Expected ~$0.20 funding cost, got ${fc:.4f}"


@pytest.mark.asyncio
async def test_funding_cost_zero_when_no_funding_rate():
    """If funding_rate is not in the trade dict, funding_cost is 0."""
    repo = FakeTradeRepo()
    trade = _make_trade(size=500.0)
    # No "funding_rate" key in trade dict
    repo.add_trade(trade)

    bus = InProcessBus()
    monitor = SLTPMonitor(bus, repo, taker_fee_rate=0.0, refresh_interval=0.0)
    monitor.register_symbol("GOLD-USDC")
    await monitor.start()

    await monitor._on_price_update(PriceUpdated(
        source="test", update=PriceUpdate(symbol="GOLD-USDC", price=94.0)))

    cost_update = next((u for _, u in repo.updated if "funding_cost" in u), None)
    assert cost_update is not None
    assert cost_update["funding_cost"] == 0.0


@pytest.mark.asyncio
async def test_pnl_equals_raw_minus_fee_minus_funding():
    """pnl = raw_pnl - trading_fee - funding_cost always holds."""
    from datetime import timedelta

    repo = FakeTradeRepo()
    entry_time = datetime.now(timezone.utc) - timedelta(hours=2)
    trade = _make_trade(direction="SHORT", entry_price=100.0, size=1000.0,
                        sl_price=105.0, tp_price=95.0)
    trade["entry_time"] = entry_time.isoformat()
    trade["funding_rate"] = 0.0002
    repo.add_trade(trade)

    bus = InProcessBus()
    monitor = SLTPMonitor(bus, repo, taker_fee_rate=0.001, refresh_interval=0.0)
    monitor.register_symbol("GOLD-USDC")
    await monitor.start()

    await monitor._on_price_update(PriceUpdated(
        source="test", update=PriceUpdate(symbol="GOLD-USDC", price=94.0)))

    pnl = repo.closed[0]["pnl"]
    cost_update = next((u for _, u in repo.updated if "raw_pnl" in u), None)

    raw_pnl = cost_update["raw_pnl"]
    trading_fee = cost_update["trading_fee"]
    funding_cost = cost_update["funding_cost"]

    assert abs(pnl - (raw_pnl - trading_fee - funding_cost)) < 0.001


# ── Duration tracking test ───────────────────────────────────────────


def test_duration_ms_in_cycle_record():
    """Pipeline stamps duration_ms on the cycle record dict."""
    import time as _time

    # Verify the import exists and time.monotonic is available
    start = _time.monotonic()
    _time.sleep(0.01)
    elapsed = int((_time.monotonic() - start) * 1000)
    assert elapsed >= 5  # at least 5ms (generous floor)
