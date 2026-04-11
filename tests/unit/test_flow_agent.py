"""Unit tests for FlowAgent and CryptoFlowProvider."""

from __future__ import annotations

import time
from collections import deque

import pytest

from engine.data.flow import FlowAgent
from engine.data.flow.base import FlowProvider
from engine.data.flow.crypto import (
    CryptoFlowProvider,
    _OI_BUFFER_MAXLEN,
    _OI_LOOKBACK_SECONDS,
)
from engine.types import AdapterCapabilities, FlowOutput, OrderResult, Position
from exchanges.base import ExchangeAdapter


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------

class MockFlowAdapter(ExchangeAdapter):
    """Adapter returning configurable funding rate and OI."""

    def __init__(
        self,
        funding_rate: float | None = None,
        open_interest: float | None = None,
        funding_raises: bool = False,
        oi_raises: bool = False,
    ) -> None:
        self._funding_rate = funding_rate
        self._open_interest = open_interest
        self._funding_raises = funding_raises
        self._oi_raises = oi_raises

    def name(self) -> str:
        return "mock_flow"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            native_sl_tp=False, supports_short=True, market_hours=None,
            asset_types=["perpetual"], margin_type="cross", has_funding_rate=True,
            has_oi_data=True, max_leverage=10.0, order_types=["market"],
            supports_partial_close=False,
        )

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> list[dict]:
        return []

    async def get_ticker(self, symbol: str) -> dict:
        return {}

    async def get_balance(self) -> float:
        return 0.0

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        return []

    async def place_market_order(self, symbol: str, side: str, size: float) -> OrderResult:
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="mock")

    async def place_limit_order(self, symbol: str, side: str, size: float, price: float) -> OrderResult:
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="mock")

    async def place_sl_order(self, symbol: str, side: str, size: float, trigger_price: float) -> OrderResult:
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="mock")

    async def place_tp_order(self, symbol: str, side: str, size: float, trigger_price: float) -> OrderResult:
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="mock")

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return False

    async def cancel_all_orders(self, symbol: str) -> int:
        return 0

    async def close_position(self, symbol: str) -> OrderResult:
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="mock")

    async def modify_sl(self, symbol: str, new_price: float) -> OrderResult:
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="mock")

    async def modify_tp(self, symbol: str, new_price: float) -> OrderResult:
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="mock")

    async def get_funding_rate(self, symbol: str) -> float | None:
        if self._funding_raises:
            raise ConnectionError("Exchange API unavailable")
        return self._funding_rate

    async def get_open_interest(self, symbol: str) -> float | None:
        if self._oi_raises:
            raise ConnectionError("Exchange API unavailable")
        return self._open_interest


# ---------------------------------------------------------------------------
# Extra mock provider for multi-provider tests
# ---------------------------------------------------------------------------

class MockGexProvider(FlowProvider):
    """Simulates a GEX/options flow provider."""

    def name(self) -> str:
        return "gex"

    async def fetch(self, symbol: str, adapter: ExchangeAdapter) -> dict:
        return {
            "gex_regime": "POSITIVE_GAMMA",
            "gex_flip_level": 64500.0,
        }


class FailingProvider(FlowProvider):
    """Always raises an exception."""

    def name(self) -> str:
        return "failing"

    async def fetch(self, symbol: str, adapter: ExchangeAdapter) -> dict:
        raise RuntimeError("Provider down")


# ---------------------------------------------------------------------------
# CryptoFlowProvider tests
# ---------------------------------------------------------------------------

class TestCryptoFlowProvider:
    @pytest.mark.asyncio
    async def test_crowded_long(self) -> None:
        adapter = MockFlowAdapter(funding_rate=0.05, open_interest=1_000_000.0)
        provider = CryptoFlowProvider()

        data = await provider.fetch("BTC-USDC", adapter)

        assert data["funding_rate"] == 0.05
        assert data["funding_signal"] == "CROWDED_LONG"

    @pytest.mark.asyncio
    async def test_crowded_short(self) -> None:
        adapter = MockFlowAdapter(funding_rate=-0.03, open_interest=500_000.0)
        provider = CryptoFlowProvider()

        data = await provider.fetch("BTC-USDC", adapter)

        assert data["funding_rate"] == -0.03
        assert data["funding_signal"] == "CROWDED_SHORT"

    @pytest.mark.asyncio
    async def test_neutral_funding(self) -> None:
        adapter = MockFlowAdapter(funding_rate=0.005, open_interest=500_000.0)
        provider = CryptoFlowProvider()

        data = await provider.fetch("BTC-USDC", adapter)

        assert data["funding_signal"] == "NEUTRAL"

    @pytest.mark.asyncio
    async def test_exactly_at_threshold(self) -> None:
        # 0.01 is exactly at threshold — not greater, so NEUTRAL
        adapter = MockFlowAdapter(funding_rate=0.01, open_interest=100.0)
        provider = CryptoFlowProvider()

        data = await provider.fetch("BTC-USDC", adapter)

        assert data["funding_signal"] == "NEUTRAL"

    @pytest.mark.asyncio
    async def test_negative_threshold(self) -> None:
        adapter = MockFlowAdapter(funding_rate=-0.01)
        provider = CryptoFlowProvider()

        data = await provider.fetch("BTC-USDC", adapter)

        assert data["funding_signal"] == "NEUTRAL"

    @pytest.mark.asyncio
    async def test_oi_returned(self) -> None:
        adapter = MockFlowAdapter(funding_rate=0.002, open_interest=1_500_000.0)
        provider = CryptoFlowProvider()

        data = await provider.fetch("BTC-USDC", adapter)

        assert data["open_interest"] == 1_500_000.0
        assert data["oi_trend"] == "STABLE"

    @pytest.mark.asyncio
    async def test_none_funding_rate(self) -> None:
        adapter = MockFlowAdapter(funding_rate=None, open_interest=100.0)
        provider = CryptoFlowProvider()

        data = await provider.fetch("BTC-USDC", adapter)

        assert "funding_rate" not in data
        assert "funding_signal" not in data
        assert data["open_interest"] == 100.0

    @pytest.mark.asyncio
    async def test_none_oi(self) -> None:
        adapter = MockFlowAdapter(funding_rate=0.002, open_interest=None)
        provider = CryptoFlowProvider()

        data = await provider.fetch("BTC-USDC", adapter)

        assert data["funding_rate"] == 0.002
        assert "open_interest" not in data

    @pytest.mark.asyncio
    async def test_funding_error_graceful(self) -> None:
        adapter = MockFlowAdapter(funding_raises=True, open_interest=100.0)
        provider = CryptoFlowProvider()

        data = await provider.fetch("BTC-USDC", adapter)

        assert "funding_rate" not in data
        assert data["open_interest"] == 100.0

    @pytest.mark.asyncio
    async def test_oi_error_graceful(self) -> None:
        adapter = MockFlowAdapter(funding_rate=0.002, oi_raises=True)
        provider = CryptoFlowProvider()

        data = await provider.fetch("BTC-USDC", adapter)

        assert data["funding_rate"] == 0.002
        assert "open_interest" not in data

    @pytest.mark.asyncio
    async def test_both_error_returns_empty(self) -> None:
        adapter = MockFlowAdapter(funding_raises=True, oi_raises=True)
        provider = CryptoFlowProvider()

        data = await provider.fetch("BTC-USDC", adapter)

        assert data == {}

    def test_name(self) -> None:
        assert CryptoFlowProvider().name() == "crypto"


# ---------------------------------------------------------------------------
# OI history buffer tests
# ---------------------------------------------------------------------------


class TestOIHistoryBuffer:
    """Tests for the per-symbol rolling OI history buffer."""

    @pytest.mark.asyncio
    async def test_first_call_oi_change_is_none(self) -> None:
        """Fresh provider, first call: oi_change_4h is None (buffer empty)."""
        adapter = MockFlowAdapter(funding_rate=0.001, open_interest=1_000_000.0)
        provider = CryptoFlowProvider()

        data = await provider.fetch("BTC-USDC", adapter)

        assert data["open_interest"] == 1_000_000.0
        assert data.get("oi_change_4h") is None
        assert data["oi_trend"] == "STABLE"

    @pytest.mark.asyncio
    async def test_buffer_warmup_returns_none(self) -> None:
        """Buffer with < 4h of data: oi_change_4h stays None."""
        provider = CryptoFlowProvider()
        now = time.time()

        # Inject 10 snapshots spanning only 5 minutes
        provider._oi_history["BTC-USDC"] = deque(maxlen=_OI_BUFFER_MAXLEN)
        for i in range(10):
            provider._oi_history["BTC-USDC"].append((now - 300 + i * 30, 1_000_000.0))

        adapter = MockFlowAdapter(funding_rate=0.001, open_interest=1_050_000.0)
        data = await provider.fetch("BTC-USDC", adapter)

        assert data.get("oi_change_4h") is None
        assert data["oi_trend"] == "STABLE"

    @pytest.mark.asyncio
    async def test_4h_buffer_computes_delta(self) -> None:
        """Provider with 4h of synthetic snapshots: oi_change_4h computed correctly."""
        provider = CryptoFlowProvider()
        now = time.time()

        old_oi = 1_000_000.0
        new_oi = 1_050_000.0  # +5%

        # Inject: one entry before the 4h window, one 10s inside it.
        # The +10 margin ensures the entry is inside the window even
        # after the small time delta between now and fetch's time.time().
        provider._oi_history["BTC-USDC"] = deque(maxlen=_OI_BUFFER_MAXLEN)
        provider._oi_history["BTC-USDC"].append((now - _OI_LOOKBACK_SECONDS - 60, old_oi))
        provider._oi_history["BTC-USDC"].append((now - _OI_LOOKBACK_SECONDS + 10, old_oi))

        adapter = MockFlowAdapter(funding_rate=0.001, open_interest=new_oi)
        data = await provider.fetch("BTC-USDC", adapter)

        expected_change = (new_oi - old_oi) / old_oi  # 0.05
        assert data["oi_change_4h"] is not None
        assert abs(data["oi_change_4h"] - expected_change) < 1e-9

    @pytest.mark.asyncio
    async def test_oi_building(self) -> None:
        """OI rose 5% over 4h: oi_trend = BUILDING."""
        provider = CryptoFlowProvider()
        now = time.time()
        old_oi = 1_000_000.0
        new_oi = 1_050_000.0  # +5%

        provider._oi_history["BTC-USDC"] = deque(maxlen=_OI_BUFFER_MAXLEN)
        provider._oi_history["BTC-USDC"].append((now - _OI_LOOKBACK_SECONDS - 60, old_oi))
        provider._oi_history["BTC-USDC"].append((now - _OI_LOOKBACK_SECONDS + 10, old_oi))

        adapter = MockFlowAdapter(funding_rate=0.001, open_interest=new_oi)
        data = await provider.fetch("BTC-USDC", adapter)

        assert data["oi_trend"] == "BUILDING"
        assert data["oi_change_4h"] > 0.02

    @pytest.mark.asyncio
    async def test_oi_dropping(self) -> None:
        """OI dropped 3% over 4h: oi_trend = DROPPING."""
        provider = CryptoFlowProvider()
        now = time.time()
        old_oi = 1_000_000.0
        new_oi = 970_000.0  # -3%

        provider._oi_history["BTC-USDC"] = deque(maxlen=_OI_BUFFER_MAXLEN)
        provider._oi_history["BTC-USDC"].append((now - _OI_LOOKBACK_SECONDS - 60, old_oi))
        provider._oi_history["BTC-USDC"].append((now - _OI_LOOKBACK_SECONDS + 10, old_oi))

        adapter = MockFlowAdapter(funding_rate=0.001, open_interest=new_oi)
        data = await provider.fetch("BTC-USDC", adapter)

        assert data["oi_trend"] == "DROPPING"
        assert data["oi_change_4h"] < -0.02

    @pytest.mark.asyncio
    async def test_oi_stable(self) -> None:
        """OI changed 1% over 4h: oi_trend = STABLE."""
        provider = CryptoFlowProvider()
        now = time.time()
        old_oi = 1_000_000.0
        new_oi = 1_010_000.0  # +1% — within ±2% threshold

        provider._oi_history["BTC-USDC"] = deque(maxlen=_OI_BUFFER_MAXLEN)
        provider._oi_history["BTC-USDC"].append((now - _OI_LOOKBACK_SECONDS - 60, old_oi))
        provider._oi_history["BTC-USDC"].append((now - _OI_LOOKBACK_SECONDS + 10, old_oi))

        adapter = MockFlowAdapter(funding_rate=0.001, open_interest=new_oi)
        data = await provider.fetch("BTC-USDC", adapter)

        assert data["oi_trend"] == "STABLE"
        assert data["oi_change_4h"] is not None
        assert abs(data["oi_change_4h"] - 0.01) < 1e-9

    def test_buffer_maxlen_respected(self) -> None:
        """Adding 500 entries doesn't grow beyond 480."""
        provider = CryptoFlowProvider()
        provider._oi_history["BTC-USDC"] = deque(maxlen=_OI_BUFFER_MAXLEN)

        for i in range(500):
            provider._oi_history["BTC-USDC"].append((float(i), 1_000_000.0 + i))

        assert len(provider._oi_history["BTC-USDC"]) == _OI_BUFFER_MAXLEN

    @pytest.mark.asyncio
    async def test_multiple_symbols_tracked_independently(self) -> None:
        """Multiple symbols tracked independently in the same provider instance."""
        provider = CryptoFlowProvider()
        now = time.time()

        # Warm up BTC with 4h+ of data showing OI building
        provider._oi_history["BTC-USDC"] = deque(maxlen=_OI_BUFFER_MAXLEN)
        provider._oi_history["BTC-USDC"].append((now - _OI_LOOKBACK_SECONDS - 60, 1_000_000.0))
        provider._oi_history["BTC-USDC"].append((now - _OI_LOOKBACK_SECONDS + 10, 1_000_000.0))

        # ETH has no history — should be cold start
        adapter_btc = MockFlowAdapter(funding_rate=0.001, open_interest=1_050_000.0)
        adapter_eth = MockFlowAdapter(funding_rate=0.001, open_interest=500_000.0)

        data_btc = await provider.fetch("BTC-USDC", adapter_btc)
        data_eth = await provider.fetch("ETH-USDC", adapter_eth)

        # BTC should have computed delta (warm buffer)
        assert data_btc["oi_change_4h"] is not None
        assert data_btc["oi_trend"] == "BUILDING"

        # ETH should be cold start (None)
        assert data_eth.get("oi_change_4h") is None
        assert data_eth["oi_trend"] == "STABLE"

        # Both symbols should have separate buffers
        assert "BTC-USDC" in provider._oi_history
        assert "ETH-USDC" in provider._oi_history
        assert len(provider._oi_history["BTC-USDC"]) == 3  # 2 injected + 1 from fetch
        assert len(provider._oi_history["ETH-USDC"]) == 1  # just from fetch


# ---------------------------------------------------------------------------
# FlowAgent tests
# ---------------------------------------------------------------------------

class TestFlowAgent:
    @pytest.mark.asyncio
    async def test_full_richness(self) -> None:
        adapter = MockFlowAdapter(funding_rate=0.05, open_interest=1_000_000.0)
        agent = FlowAgent(providers=[CryptoFlowProvider()])

        result = await agent.fetch_flow("BTC-USDC", adapter)

        assert isinstance(result, FlowOutput)
        assert result.data_richness == "FULL"
        assert result.funding_rate == 0.05
        assert result.funding_signal == "CROWDED_LONG"
        assert result.oi_trend == "STABLE"

    @pytest.mark.asyncio
    async def test_partial_richness_funding_only(self) -> None:
        adapter = MockFlowAdapter(funding_rate=0.002, open_interest=None)
        agent = FlowAgent(providers=[CryptoFlowProvider()])

        result = await agent.fetch_flow("BTC-USDC", adapter)

        assert result.data_richness == "PARTIAL"
        assert result.funding_rate == 0.002

    @pytest.mark.asyncio
    async def test_partial_richness_oi_only(self) -> None:
        adapter = MockFlowAdapter(funding_rate=None, open_interest=500_000.0)
        agent = FlowAgent(providers=[CryptoFlowProvider()])

        result = await agent.fetch_flow("BTC-USDC", adapter)

        assert result.data_richness == "PARTIAL"

    @pytest.mark.asyncio
    async def test_minimal_richness(self) -> None:
        adapter = MockFlowAdapter(funding_rate=None, open_interest=None)
        agent = FlowAgent(providers=[CryptoFlowProvider()])

        result = await agent.fetch_flow("BTC-USDC", adapter)

        assert result.data_richness == "MINIMAL"
        assert result.funding_rate is None
        assert result.funding_signal == "NEUTRAL"
        assert result.oi_trend == "STABLE"

    @pytest.mark.asyncio
    async def test_no_providers_returns_minimal(self) -> None:
        adapter = MockFlowAdapter()
        agent = FlowAgent(providers=[])

        result = await agent.fetch_flow("BTC-USDC", adapter)

        assert result.data_richness == "MINIMAL"

    @pytest.mark.asyncio
    async def test_merges_multiple_providers(self) -> None:
        adapter = MockFlowAdapter(funding_rate=0.05, open_interest=1_000_000.0)
        agent = FlowAgent(providers=[CryptoFlowProvider(), MockGexProvider()])

        result = await agent.fetch_flow("BTC-USDC", adapter)

        assert result.data_richness == "FULL"
        assert result.funding_rate == 0.05
        assert result.gex_regime == "POSITIVE_GAMMA"
        assert result.gex_flip_level == 64500.0

    @pytest.mark.asyncio
    async def test_failing_provider_does_not_block_others(self) -> None:
        adapter = MockFlowAdapter(funding_rate=0.05, open_interest=1_000_000.0)
        agent = FlowAgent(providers=[FailingProvider(), CryptoFlowProvider()])

        result = await agent.fetch_flow("BTC-USDC", adapter)

        assert result.data_richness == "FULL"
        assert result.funding_rate == 0.05

    @pytest.mark.asyncio
    async def test_all_providers_fail_returns_minimal(self) -> None:
        adapter = MockFlowAdapter()
        agent = FlowAgent(providers=[FailingProvider()])

        result = await agent.fetch_flow("BTC-USDC", adapter)

        assert result.data_richness == "MINIMAL"

    def test_add_provider(self) -> None:
        agent = FlowAgent()
        assert len(agent.providers) == 0

        agent.add_provider(CryptoFlowProvider())
        assert len(agent.providers) == 1
        assert agent.providers[0].name() == "crypto"

    @pytest.mark.asyncio
    async def test_defaults_for_missing_fields(self) -> None:
        adapter = MockFlowAdapter(funding_rate=0.002, open_interest=100.0)
        agent = FlowAgent(providers=[CryptoFlowProvider()])

        result = await agent.fetch_flow("BTC-USDC", adapter)

        # Fields not provided by CryptoFlowProvider default properly
        assert result.oi_change_4h is None
        assert result.nearest_liquidation_above is None
        assert result.nearest_liquidation_below is None
        assert result.gex_regime is None
        assert result.gex_flip_level is None


# ---------------------------------------------------------------------------
# Configurable lookback + persistence
# ---------------------------------------------------------------------------


class _FakeOIRepo:
    """In-memory stand-in for OISnapshotRepository."""

    def __init__(self, preload: list[dict] | None = None) -> None:
        self.snapshots: list[tuple[str, float, float]] = []
        if preload:
            for row in preload:
                self.snapshots.append(
                    (row["symbol"], float(row["timestamp"]), float(row["oi_value"]))
                )

    async def insert_snapshot(self, symbol, timestamp, oi_value):
        # Coerce datetime → epoch like the real repo would store
        if hasattr(timestamp, "timestamp"):
            ts_epoch = timestamp.timestamp()
        else:
            ts_epoch = float(timestamp)
        self.snapshots.append((symbol, ts_epoch, float(oi_value)))

    async def get_recent_snapshots(self, lookback_seconds):
        cutoff = time.time() - float(lookback_seconds)
        rows = [
            {"symbol": s, "timestamp": ts, "oi_value": oi}
            for (s, ts, oi) in self.snapshots
            if ts > cutoff
        ]
        rows.sort(key=lambda r: (r["symbol"], r["timestamp"]))
        return rows

    async def cleanup_older_than(self, seconds):
        cutoff = time.time() - float(seconds)
        before = len(self.snapshots)
        self.snapshots = [s for s in self.snapshots if s[1] >= cutoff]
        return before - len(self.snapshots)


class TestCryptoFlowProviderLookback:
    """Configurable lookback (Change 7)."""

    def test_default_lookback_is_two_hours(self) -> None:
        provider = CryptoFlowProvider()
        assert provider.lookback_seconds == 7_200
        assert provider._buffer_maxlen == 240

    def test_explicit_lookback_in_constructor(self) -> None:
        provider = CryptoFlowProvider(lookback_seconds=3_600)
        assert provider.lookback_seconds == 3_600
        assert provider._buffer_maxlen == 120

    def test_set_lookback_for_known_timeframes(self) -> None:
        provider = CryptoFlowProvider()
        for tf, expected in [
            ("15m", 1_800),
            ("30m", 3_600),
            ("1h", 7_200),
            ("4h", 28_800),
            ("1d", 172_800),
        ]:
            provider.set_lookback_for_timeframe(tf)
            assert provider.lookback_seconds == expected
            assert provider._buffer_maxlen == max(60, expected // 30)

    def test_set_lookback_for_unknown_timeframe_falls_back(self) -> None:
        provider = CryptoFlowProvider()
        provider.set_lookback_for_timeframe("7m")  # not in the map
        assert provider.lookback_seconds == 7_200  # default

    def test_set_lookback_resets_history(self) -> None:
        """Changing the lookback wipes existing deques (their maxlen
        is now wrong) so they refill cleanly from the next fetch."""
        provider = CryptoFlowProvider()
        provider._oi_history["BTC-USDC"] = deque(maxlen=480)
        provider._oi_history["BTC-USDC"].append((time.time(), 1.0))
        provider._oi_warm_logged.add("BTC-USDC")

        provider.set_lookback_for_timeframe("30m")

        assert "BTC-USDC" not in provider._oi_history
        assert "BTC-USDC" not in provider._oi_warm_logged

    @pytest.mark.asyncio
    async def test_compute_uses_configured_lookback(self) -> None:
        """A provider with a 30-minute lookback computes deltas off the
        30-minute window, not the 4-hour window."""
        provider = CryptoFlowProvider(lookback_seconds=1_800)
        now = time.time()
        # Inject one entry just outside the 30-minute window and one
        # just inside it.
        provider._oi_history["BTC-USDC"] = deque(maxlen=120)
        provider._oi_history["BTC-USDC"].append((now - 1_900, 1_000_000.0))
        provider._oi_history["BTC-USDC"].append((now - 1_750, 1_000_000.0))

        adapter = MockFlowAdapter(funding_rate=0.001, open_interest=1_050_000.0)
        data = await provider.fetch("BTC-USDC", adapter)

        assert data["oi_change_4h"] is not None
        assert abs(data["oi_change_4h"] - 0.05) < 1e-9
        assert data["oi_trend"] == "BUILDING"


class TestCryptoFlowProviderPersistence:
    """Repository wiring (Change 6)."""

    @pytest.mark.asyncio
    async def test_warmup_loads_recent_snapshots_into_deques(self) -> None:
        now = time.time()
        repo = _FakeOIRepo(preload=[
            {"symbol": "BTC-USDC", "timestamp": now - 1_500, "oi_value": 1_000_000.0},
            {"symbol": "BTC-USDC", "timestamp": now - 1_400, "oi_value": 1_010_000.0},
            {"symbol": "ETH-USDC", "timestamp": now - 1_400, "oi_value": 500_000.0},
        ])
        provider = CryptoFlowProvider(lookback_seconds=1_800, oi_repo=repo)

        loaded = await provider.warmup_from_repo()
        assert loaded == 3
        assert len(provider._oi_history["BTC-USDC"]) == 2
        assert len(provider._oi_history["ETH-USDC"]) == 1

    @pytest.mark.asyncio
    async def test_warmup_no_repo_is_no_op(self) -> None:
        provider = CryptoFlowProvider()
        assert await provider.warmup_from_repo() == 0

    @pytest.mark.asyncio
    async def test_warmup_swallows_repo_errors(self) -> None:
        class _BoomRepo:
            async def get_recent_snapshots(self, lookback_seconds):
                raise RuntimeError("DB blew up")

            async def insert_snapshot(self, *a, **k):
                pass

            async def cleanup_older_than(self, *a, **k):
                return 0

        provider = CryptoFlowProvider(oi_repo=_BoomRepo())
        # Must NOT raise — provider falls back to cold start
        assert await provider.warmup_from_repo() == 0
        assert provider._oi_history == {}

    @pytest.mark.asyncio
    async def test_fetch_persists_snapshot_to_repo(self) -> None:
        repo = _FakeOIRepo()
        provider = CryptoFlowProvider(oi_repo=repo)
        adapter = MockFlowAdapter(funding_rate=0.001, open_interest=1_000_000.0)

        await provider.fetch("BTC-USDC", adapter)
        await provider.fetch("BTC-USDC", adapter)

        assert len(repo.snapshots) == 2
        assert repo.snapshots[0][0] == "BTC-USDC"
        assert repo.snapshots[0][2] == 1_000_000.0

    @pytest.mark.asyncio
    async def test_fetch_repo_failure_does_not_break_data_layer(self) -> None:
        class _InsertBoomRepo:
            async def get_recent_snapshots(self, lookback_seconds):
                return []

            async def insert_snapshot(self, *a, **k):
                raise ConnectionError("DB unreachable")

            async def cleanup_older_than(self, *a, **k):
                return 0

        provider = CryptoFlowProvider(oi_repo=_InsertBoomRepo())
        adapter = MockFlowAdapter(funding_rate=0.001, open_interest=1_000_000.0)
        # Must not raise — DB blip cannot take down the data layer
        data = await provider.fetch("BTC-USDC", adapter)
        assert data["open_interest"] == 1_000_000.0

    @pytest.mark.asyncio
    async def test_warmup_then_fetch_yields_immediate_delta(self) -> None:
        """The whole point of persistence: a fresh process can compute
        deltas on its very first fetch instead of waiting for warmup."""
        now = time.time()
        repo = _FakeOIRepo(preload=[
            # One entry just before the lookback window, one just inside
            {"symbol": "BTC-USDC", "timestamp": now - 1_900, "oi_value": 1_000_000.0},
            {"symbol": "BTC-USDC", "timestamp": now - 1_750, "oi_value": 1_000_000.0},
        ])
        provider = CryptoFlowProvider(lookback_seconds=1_800, oi_repo=repo)
        await provider.warmup_from_repo()

        adapter = MockFlowAdapter(funding_rate=0.001, open_interest=1_050_000.0)
        data = await provider.fetch("BTC-USDC", adapter)

        assert data["oi_change_4h"] is not None
        assert data["oi_change_4h"] > 0.02
        assert data["oi_trend"] == "BUILDING"
