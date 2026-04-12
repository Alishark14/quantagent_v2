"""Unit tests for engine/sl_tp_monitor.py.

Sprint Week 7 Task 5. Drives the SLTPMonitor with a `FakeTradeRepo`
that records every method call so we can assert exact behaviour:

  * LONG / SHORT × SL / TP gates trigger at the right tick price
  * No trades for the symbol = single dict lookup, zero DB calls
  * Closed trades emit `TradeClosed` on the bus
  * P&L matches Sentinel's legacy formula byte-for-byte
  * `forward_max_r` tracks the in-memory high-water mark and only
    flushes to DB on trade close (and only when it improves the prior)
  * Periodic refresh picks up new trades after the interval
  * Tick price between SL and TP = no DB write, no event
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from engine.events import InProcessBus, PriceUpdated, TradeClosed
from engine.sl_tp_monitor import SLTPMonitor
from engine.types import PriceUpdate


# ---------------------------------------------------------------------------
# FakeTradeRepo
# ---------------------------------------------------------------------------


class FakeTradeRepo:
    """In-memory TradeRepository stand-in.

    Holds open trades keyed by symbol. Records every call to
    `get_open_shadow_trades`, `close_trade`, and `update_trade` so the
    tests can assert on call counts + arguments without monkey-patching.
    """

    def __init__(self, open_trades_by_symbol: dict[str, list[dict]] | None = None) -> None:
        self._open: dict[str, list[dict]] = {
            s: list(trades) for s, trades in (open_trades_by_symbol or {}).items()
        }
        self.get_calls: list[str] = []
        self.close_calls: list[dict] = []
        self.update_calls: list[dict] = []

    async def get_open_shadow_trades(self, symbol: str) -> list[dict]:
        self.get_calls.append(symbol)
        return [dict(t) for t in self._open.get(symbol, [])]

    async def close_trade(
        self,
        trade_id: str,
        *,
        exit_price: float,
        exit_reason: str,
        exit_time,
        pnl: float,
    ) -> bool:
        self.close_calls.append(
            {
                "trade_id": trade_id,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "exit_time": exit_time,
                "pnl": pnl,
            }
        )
        # Drop the trade from any open list so a re-fetch wouldn't see it.
        for trades in self._open.values():
            trades[:] = [t for t in trades if str(t.get("id")) != trade_id]
        return True

    async def update_trade(self, trade_id: str, updates: dict) -> bool:
        self.update_calls.append({"trade_id": trade_id, "updates": updates})
        return True

    # Helper for tests — inject a new trade after start().
    def add_trade(self, symbol: str, trade: dict) -> None:
        self._open.setdefault(symbol, []).append(dict(trade))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.events: list = []

    def __call__(self, event) -> None:
        self.events.append(event)


def _trade(
    *,
    id: str = "t1",
    direction: str = "LONG",
    entry: float = 100.0,
    sl: float = 95.0,
    tp: float = 110.0,
    size: float = 1000.0,
    forward_max_r: float | None = None,
) -> dict:
    return {
        "id": id,
        "direction": direction,
        "entry_price": entry,
        "sl_price": sl,
        "tp_price": tp,
        "size": size,
        "forward_max_r": forward_max_r,
    }


async def _publish_tick(
    bus: InProcessBus, symbol: str, price: float
) -> None:
    await bus.publish(
        PriceUpdated(
            source="test",
            update=PriceUpdate(
                symbol=symbol,
                price=price,
                exchange="hyperliquid",
                timestamp=datetime.now(timezone.utc),
            ),
        )
    )


@pytest.fixture
def bus() -> InProcessBus:
    return InProcessBus()


# ---------------------------------------------------------------------------
# Gate-trigger tests
# ---------------------------------------------------------------------------


class TestLongGates:
    async def test_long_sl_hit(self, bus: InProcessBus) -> None:
        repo = FakeTradeRepo({"BTC-USDC": [_trade(direction="LONG", entry=100, sl=95, tp=110)]})
        mgr = SLTPMonitor(bus, repo)
        mgr.register_symbol("BTC-USDC")
        await mgr.start()

        await _publish_tick(bus, "BTC-USDC", 94.5)

        assert len(repo.close_calls) == 1
        call = repo.close_calls[0]
        assert call["trade_id"] == "t1"
        assert call["exit_price"] == 95.0
        assert call["exit_reason"] == "SL"
        # Long PnL: (95 - 100) * 1000 / 100 = -50
        assert call["pnl"] == pytest.approx(-50.0)
        await mgr.stop()

    async def test_long_tp_hit(self, bus: InProcessBus) -> None:
        repo = FakeTradeRepo({"BTC-USDC": [_trade(direction="LONG", entry=100, sl=95, tp=110)]})
        mgr = SLTPMonitor(bus, repo)
        mgr.register_symbol("BTC-USDC")
        await mgr.start()

        await _publish_tick(bus, "BTC-USDC", 110.5)

        assert len(repo.close_calls) == 1
        call = repo.close_calls[0]
        assert call["exit_price"] == 110.0
        assert call["exit_reason"] == "TP"
        # Long PnL: (110 - 100) * 1000 / 100 = 100
        assert call["pnl"] == pytest.approx(100.0)
        await mgr.stop()


class TestShortGates:
    async def test_short_sl_hit(self, bus: InProcessBus) -> None:
        repo = FakeTradeRepo({"BTC-USDC": [_trade(direction="SHORT", entry=100, sl=105, tp=90)]})
        mgr = SLTPMonitor(bus, repo)
        mgr.register_symbol("BTC-USDC")
        await mgr.start()

        await _publish_tick(bus, "BTC-USDC", 105.5)

        assert len(repo.close_calls) == 1
        call = repo.close_calls[0]
        assert call["exit_price"] == 105.0
        assert call["exit_reason"] == "SL"
        # Short PnL: (100 - 105) * 1000 / 100 = -50
        assert call["pnl"] == pytest.approx(-50.0)
        await mgr.stop()

    async def test_short_tp_hit(self, bus: InProcessBus) -> None:
        repo = FakeTradeRepo({"BTC-USDC": [_trade(direction="SHORT", entry=100, sl=105, tp=90)]})
        mgr = SLTPMonitor(bus, repo)
        mgr.register_symbol("BTC-USDC")
        await mgr.start()

        await _publish_tick(bus, "BTC-USDC", 89.5)

        assert len(repo.close_calls) == 1
        call = repo.close_calls[0]
        assert call["exit_price"] == 90.0
        assert call["exit_reason"] == "TP"
        # Short PnL: (100 - 90) * 1000 / 100 = 100
        assert call["pnl"] == pytest.approx(100.0)
        await mgr.stop()


class TestNoOpPaths:
    async def test_no_trades_for_symbol_is_fast_path(self, bus: InProcessBus) -> None:
        # Empty repo. Subscribe but never register any symbol.
        repo = FakeTradeRepo({})
        mgr = SLTPMonitor(bus, repo)
        await mgr.start()
        # Initial start() refresh hits zero symbols (none registered).
        initial_get_calls = len(repo.get_calls)

        # Tick on a symbol we have no trades for.
        await _publish_tick(bus, "DOGE-USDC", 0.15)

        # Hot path: no trades for symbol → fast return → NO additional
        # DB calls and definitely no close_trade.
        assert len(repo.get_calls) == initial_get_calls
        assert repo.close_calls == []
        await mgr.stop()

    async def test_price_between_sl_and_tp_no_action(self, bus: InProcessBus) -> None:
        repo = FakeTradeRepo({"BTC-USDC": [_trade(direction="LONG", entry=100, sl=95, tp=110)]})
        mgr = SLTPMonitor(bus, repo)
        mgr.register_symbol("BTC-USDC")
        await mgr.start()

        await _publish_tick(bus, "BTC-USDC", 102.0)
        await _publish_tick(bus, "BTC-USDC", 105.0)

        assert repo.close_calls == []
        # The trade is still in the in-memory map.
        assert mgr.open_trade_count("BTC-USDC") == 1
        await mgr.stop()


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


class TestTradeClosedEvent:
    async def test_close_emits_trade_closed_event(self, bus: InProcessBus) -> None:
        rec = _Recorder()
        bus.subscribe(TradeClosed, rec)

        repo = FakeTradeRepo({"BTC-USDC": [_trade(direction="LONG", entry=100, sl=95, tp=110)]})
        mgr = SLTPMonitor(bus, repo)
        mgr.register_symbol("BTC-USDC")
        await mgr.start()

        await _publish_tick(bus, "BTC-USDC", 109.99)
        await _publish_tick(bus, "BTC-USDC", 110.0)

        assert len(rec.events) == 1
        evt = rec.events[0]
        assert isinstance(evt, TradeClosed)
        assert evt.symbol == "BTC-USDC"
        assert evt.exit_reason == "TP"
        assert evt.pnl == pytest.approx(100.0)
        assert evt.trade_id == "t1"
        assert evt.direction == "LONG"
        assert evt.entry_price == 100.0
        assert evt.sl_price == 95.0
        await mgr.stop()


# ---------------------------------------------------------------------------
# forward_max_r
# ---------------------------------------------------------------------------


class TestForwardMaxR:
    async def test_high_water_mark_tracks_per_tick_then_flushes_on_close(
        self, bus: InProcessBus
    ) -> None:
        # LONG, entry 100, SL 95 → risk = 5
        # Tick to 102 → R = (102-100)/5 = 0.4
        # Tick to 108 → R = 1.6 (new HWM)
        # Tick to 105 → R = 1.0 (no update — HWM stays at 1.6)
        # Tick to 110 → TP hit → R at exit = 2.0 (new HWM, then flushed)
        repo = FakeTradeRepo(
            {"BTC-USDC": [_trade(direction="LONG", entry=100, sl=95, tp=110, forward_max_r=None)]}
        )
        mgr = SLTPMonitor(bus, repo)
        mgr.register_symbol("BTC-USDC")
        await mgr.start()

        await _publish_tick(bus, "BTC-USDC", 102.0)
        # No DB writes mid-trade — only in-memory.
        assert repo.update_calls == []

        await _publish_tick(bus, "BTC-USDC", 108.0)
        assert repo.update_calls == []

        await _publish_tick(bus, "BTC-USDC", 105.0)
        assert repo.update_calls == []

        # Trip TP — close + flush forward_max_r in one shot.
        await _publish_tick(bus, "BTC-USDC", 110.0)

        assert len(repo.close_calls) == 1
        # Final HWM = 2.0 (the TP price itself).
        assert len(repo.update_calls) == 1
        assert repo.update_calls[0]["trade_id"] == "t1"
        assert repo.update_calls[0]["updates"]["forward_max_r"] == pytest.approx(2.0)
        await mgr.stop()

    async def test_flush_skipped_when_prior_persisted_already_higher(
        self, bus: InProcessBus
    ) -> None:
        # SHORT, entry 100, SL 105, TP 90, prior_persisted = 5.0
        # Tick to 91 → R = (100-91)/5 = 1.8 (in-memory HWM, but < 5.0 persisted)
        # Tick to 90 → TP hit → R = 2.0, still < 5.0 → no flush
        repo = FakeTradeRepo(
            {"BTC-USDC": [_trade(direction="SHORT", entry=100, sl=105, tp=90, forward_max_r=5.0)]}
        )
        mgr = SLTPMonitor(bus, repo)
        mgr.register_symbol("BTC-USDC")
        await mgr.start()

        await _publish_tick(bus, "BTC-USDC", 91.0)
        await _publish_tick(bus, "BTC-USDC", 90.0)

        assert len(repo.close_calls) == 1
        # Prior persisted (5.0) is higher than new in-memory HWM — no update_trade call.
        assert repo.update_calls == []
        await mgr.stop()


# ---------------------------------------------------------------------------
# Periodic refresh
# ---------------------------------------------------------------------------


class TestPeriodicRefresh:
    async def test_refresh_picks_up_new_trade_after_interval(
        self, bus: InProcessBus
    ) -> None:
        # t1 has a wide SL/TP so neither tick in the test will close it —
        # we want the refresh assertion to be about t2 alone.
        repo = FakeTradeRepo(
            {"BTC-USDC": [_trade(id="t1", direction="LONG", entry=100, sl=50, tp=150)]}
        )
        mgr = SLTPMonitor(bus, repo, refresh_interval=0.0)  # always refresh
        mgr.register_symbol("BTC-USDC")
        await mgr.start()
        assert mgr.open_trade_count("BTC-USDC") == 1

        # Inject a new trade for the SAME symbol mid-flight. Tight SL so
        # the next-but-one tick hits it cleanly.
        repo.add_trade(
            "BTC-USDC",
            _trade(id="t2", direction="LONG", entry=100, sl=98, tp=102),
        )

        # First tick at 101 — neither trade hits, but the refresh runs
        # (refresh_interval=0) and picks up t2.
        await _publish_tick(bus, "BTC-USDC", 101.0)
        assert mgr.open_trade_count("BTC-USDC") == 2

        # Tick at 97 — hits t2's SL (97 < 98) but stays clear of t1's
        # 50 SL. Only t2 closes.
        await _publish_tick(bus, "BTC-USDC", 97.0)
        assert len(repo.close_calls) == 1
        assert repo.close_calls[0]["trade_id"] == "t2"
        assert mgr.open_trade_count("BTC-USDC") == 1
        await mgr.stop()

    async def test_refresh_skipped_within_interval(self, bus: InProcessBus) -> None:
        repo = FakeTradeRepo({"BTC-USDC": [_trade()]})
        mgr = SLTPMonitor(bus, repo, refresh_interval=999.0)  # never refresh again
        mgr.register_symbol("BTC-USDC")
        await mgr.start()
        baseline_get_calls = len(repo.get_calls)

        # Multiple ticks — none should trigger a refresh because the
        # interval hasn't elapsed.
        for px in (101.0, 102.0, 103.0, 104.0):
            await _publish_tick(bus, "BTC-USDC", px)

        assert len(repo.get_calls) == baseline_get_calls
        await mgr.stop()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_start_is_idempotent(self, bus: InProcessBus) -> None:
        repo = FakeTradeRepo({})
        mgr = SLTPMonitor(bus, repo)
        await mgr.start()
        await mgr.start()
        assert mgr.is_running() is True
        await mgr.stop()
        assert mgr.is_running() is False

    async def test_stop_unsubscribes(self, bus: InProcessBus) -> None:
        repo = FakeTradeRepo({"BTC-USDC": [_trade()]})
        mgr = SLTPMonitor(bus, repo)
        mgr.register_symbol("BTC-USDC")
        await mgr.start()
        await mgr.stop()

        # After stop, ticks should not trigger any DB activity.
        baseline_close = len(repo.close_calls)
        await _publish_tick(bus, "BTC-USDC", 90.0)
        assert len(repo.close_calls) == baseline_close
