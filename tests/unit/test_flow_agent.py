"""Unit tests for FlowAgent and CryptoFlowProvider."""

from __future__ import annotations

import pytest

from engine.data.flow import FlowAgent
from engine.data.flow.base import FlowProvider
from engine.data.flow.crypto import CryptoFlowProvider
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
