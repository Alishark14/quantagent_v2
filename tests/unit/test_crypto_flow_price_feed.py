"""Tests for CryptoFlowProvider ← PriceFeed integration.

Verifies:
  * With PriceFeed wired, funding reads from memory (adapter NOT called)
  * With PriceFeed wired, OI reads from memory (adapter NOT called)
  * With price_feed=None, falls back to REST adapter (existing behaviour)
  * OpenInterestUpdated event subscription populates the OI history buffer
  * oi_change_pct computed correctly from pushed OI snapshots
  * Mixed scenario — PriceFeed has funding but no OI yet → funding from
    memory, OI from REST fallback
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone

import pytest

from engine.data.flow.crypto import CryptoFlowProvider
from engine.data.price_feed.base import PriceFeed, SymbolState
from engine.events import InProcessBus, OpenInterestUpdated
from engine.types import OIUpdate


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakePriceFeed(PriceFeed):
    """Minimal PriceFeed that serves canned funding / OI values."""

    def __init__(
        self,
        event_bus,
        *,
        funding_by_symbol: dict[str, float] | None = None,
        oi_by_symbol: dict[str, float] | None = None,
    ) -> None:
        super().__init__(event_bus, exchange_name="fake")
        for sym, rate in (funding_by_symbol or {}).items():
            state = self._symbols.setdefault(sym, SymbolState(symbol=sym))
            state.funding_rate = rate
        for sym, oi in (oi_by_symbol or {}).items():
            state = self._symbols.setdefault(sym, SymbolState(symbol=sym))
            state.open_interest = oi

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def _listen(self) -> None:
        return None

    async def subscribe(self, symbols) -> None:
        for s in symbols:
            self._subscribed_symbols.add(s)

    async def unsubscribe(self, symbols) -> None:
        for s in symbols:
            self._subscribed_symbols.discard(s)


class FakeAdapter:
    """Records every REST call so tests can assert on call counts."""

    def __init__(
        self,
        funding: float | None = 0.0001,
        oi: float | None = 1_000_000_000.0,
    ) -> None:
        self._funding = funding
        self._oi = oi
        self.funding_calls: list[str] = []
        self.oi_calls: list[str] = []

    async def get_funding_rate(self, symbol: str) -> float | None:
        self.funding_calls.append(symbol)
        return self._funding

    async def get_open_interest(self, symbol: str) -> float | None:
        self.oi_calls.append(symbol)
        return self._oi


@pytest.fixture
def bus() -> InProcessBus:
    return InProcessBus()


# ---------------------------------------------------------------------------
# PriceFeed-first reads
# ---------------------------------------------------------------------------


class TestPriceFeedFirstReads:
    async def test_funding_from_price_feed_memory_no_rest(
        self, bus: InProcessBus
    ) -> None:
        pf = FakePriceFeed(bus, funding_by_symbol={"BTC-USDC": 0.0005})
        adapter = FakeAdapter(funding=0.9999)
        provider = CryptoFlowProvider(price_feed=pf)

        result = await provider.fetch("BTC-USDC", adapter)

        assert result["funding_rate"] == 0.0005
        assert adapter.funding_calls == []  # adapter was NOT called

    async def test_oi_from_price_feed_memory_no_rest(
        self, bus: InProcessBus
    ) -> None:
        pf = FakePriceFeed(bus, oi_by_symbol={"BTC-USDC": 2_000_000_000.0})
        adapter = FakeAdapter(oi=9_999.0)
        provider = CryptoFlowProvider(price_feed=pf)

        result = await provider.fetch("BTC-USDC", adapter)

        assert result["open_interest"] == 2_000_000_000.0
        assert adapter.oi_calls == []  # adapter was NOT called

    async def test_fallback_to_rest_when_no_price_feed(
        self, bus: InProcessBus
    ) -> None:
        adapter = FakeAdapter(funding=0.0001, oi=1_500_000_000.0)
        provider = CryptoFlowProvider()  # no price_feed

        result = await provider.fetch("BTC-USDC", adapter)

        assert result["funding_rate"] == 0.0001
        assert result["open_interest"] == 1_500_000_000.0
        assert len(adapter.funding_calls) == 1
        assert len(adapter.oi_calls) == 1

    async def test_mixed_funding_from_memory_oi_from_rest(
        self, bus: InProcessBus
    ) -> None:
        # PriceFeed has funding but no OI for this symbol.
        pf = FakePriceFeed(bus, funding_by_symbol={"BTC-USDC": 0.0003})
        adapter = FakeAdapter(funding=0.9999, oi=1_500_000_000.0)
        provider = CryptoFlowProvider(price_feed=pf)

        result = await provider.fetch("BTC-USDC", adapter)

        # Funding came from PriceFeed → adapter NOT called for funding.
        assert result["funding_rate"] == 0.0003
        assert adapter.funding_calls == []
        # OI came from REST → adapter WAS called.
        assert result["open_interest"] == 1_500_000_000.0
        assert len(adapter.oi_calls) == 1

    async def test_funding_signal_classification_works_with_price_feed(
        self, bus: InProcessBus
    ) -> None:
        pf = FakePriceFeed(bus, funding_by_symbol={"BTC-USDC": 0.02})
        adapter = FakeAdapter()
        provider = CryptoFlowProvider(price_feed=pf)

        result = await provider.fetch("BTC-USDC", adapter)
        assert result["funding_signal"] == "CROWDED_LONG"


# ---------------------------------------------------------------------------
# OI event subscription
# ---------------------------------------------------------------------------


class TestOIEventSubscription:
    async def test_oi_update_event_populates_history_buffer(
        self, bus: InProcessBus
    ) -> None:
        pf = FakePriceFeed(bus)
        provider = CryptoFlowProvider(price_feed=pf, event_bus=bus)

        ts = datetime(2026, 4, 12, 10, 0, 0, tzinfo=timezone.utc)
        await bus.publish(
            OpenInterestUpdated(
                source="price_feed:hyperliquid",
                update=OIUpdate(
                    symbol="BTC-USDC",
                    open_interest=1_000_000_000.0,
                    exchange="hyperliquid",
                    timestamp=ts,
                ),
            )
        )

        buf = provider._oi_history.get("BTC-USDC")
        assert buf is not None
        assert len(buf) == 1
        assert buf[0][1] == 1_000_000_000.0

    async def test_oi_change_computed_from_pushed_snapshots(
        self, bus: InProcessBus
    ) -> None:
        pf = FakePriceFeed(bus)
        provider = CryptoFlowProvider(
            price_feed=pf, event_bus=bus, lookback_seconds=3600
        )

        now = time.time()
        # Three entries: one before the cutoff (proves the buffer spans
        # the lookback), one just after the cutoff (picked as best_oi),
        # and the current observation passed to _compute_oi_delta.
        await provider._record_oi_snapshot("BTC-USDC", 1_000.0, now - 4000)
        await provider._record_oi_snapshot("BTC-USDC", 1_000.0, now - 3500)

        oi_change, oi_trend = provider._compute_oi_delta("BTC-USDC", now, 1_050.0)

        assert oi_change == pytest.approx(0.05)  # +5%
        assert oi_trend == "BUILDING"

    async def test_oi_buffer_fills_from_events_then_fetch_uses_it(
        self, bus: InProcessBus
    ) -> None:
        pf = FakePriceFeed(
            bus,
            funding_by_symbol={"BTC-USDC": 0.0001},
            oi_by_symbol={"BTC-USDC": 1_050_000_000.0},
        )
        adapter = FakeAdapter()
        provider = CryptoFlowProvider(
            price_feed=pf, event_bus=bus, lookback_seconds=3600
        )

        now = time.time()
        # Two pre-existing snapshots so the buffer spans the lookback:
        # one before the cutoff (proves depth), one just after (picked
        # as best_oi by the walk). Both at 1.0B.
        await provider._record_oi_snapshot("BTC-USDC", 1_000_000_000.0, now - 4000)
        await provider._record_oi_snapshot("BTC-USDC", 1_000_000_000.0, now - 3500)

        # fetch() reads OI from PriceFeed (1.05B) and appends it to the
        # buffer. The delta is computed against the snapshot at now-3500
        # (1.0B) → +5% = BUILDING.
        result = await provider.fetch("BTC-USDC", adapter)

        assert result["open_interest"] == 1_050_000_000.0
        assert result["oi_change_4h"] == pytest.approx(0.05)
        assert result["oi_trend"] == "BUILDING"
        # Neither REST call was made.
        assert adapter.funding_calls == []
        assert adapter.oi_calls == []


class TestSetPriceFeedSetter:
    async def test_set_price_feed_wires_oi_subscription(
        self, bus: InProcessBus
    ) -> None:
        provider = CryptoFlowProvider()
        assert provider._price_feed is None

        pf = FakePriceFeed(bus, funding_by_symbol={"BTC-USDC": 0.0005})
        provider.set_price_feed(pf, bus)

        assert provider._price_feed is pf

        # Publishing an OI event should now populate the buffer.
        ts = datetime(2026, 4, 12, 11, 0, 0, tzinfo=timezone.utc)
        await bus.publish(
            OpenInterestUpdated(
                source="test",
                update=OIUpdate(
                    symbol="ETH-USDC",
                    open_interest=500_000.0,
                    timestamp=ts,
                ),
            )
        )
        buf = provider._oi_history.get("ETH-USDC")
        assert buf is not None
        assert len(buf) == 1

    async def test_set_price_feed_idempotent(self, bus: InProcessBus) -> None:
        pf = FakePriceFeed(bus)
        provider = CryptoFlowProvider()
        provider.set_price_feed(pf, bus)
        provider.set_price_feed(pf, bus)  # second call — no-op
        assert provider._price_feed is pf
