"""Unit tests for ExchangeAdapter base and ExchangeFactory."""

from __future__ import annotations

import pytest

from engine.types import AdapterCapabilities, OrderResult, Position
from exchanges.base import ExchangeAdapter
from exchanges.factory import ExchangeFactory


# ---------------------------------------------------------------------------
# MockAdapter — implements all abstract methods
# ---------------------------------------------------------------------------


class MockAdapter(ExchangeAdapter):
    def name(self) -> str:
        return "mock"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            native_sl_tp=True,
            supports_short=True,
            market_hours=None,
            asset_types=["crypto"],
            margin_type="cross",
            has_funding_rate=True,
            has_oi_data=False,
            max_leverage=50.0,
            order_types=["market", "limit"],
            supports_partial_close=True,
        )

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> list[dict]:
        return []

    async def get_ticker(self, symbol: str) -> dict:
        return {"bid": 100.0, "ask": 100.1}

    async def get_balance(self) -> float:
        return 10000.0

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        return []

    async def place_market_order(self, symbol: str, side: str, size: float) -> OrderResult:
        return OrderResult(success=True, order_id="M1", fill_price=100.0, fill_size=size, error=None)

    async def place_limit_order(self, symbol: str, side: str, size: float, price: float) -> OrderResult:
        return OrderResult(success=True, order_id="L1", fill_price=price, fill_size=size, error=None)

    async def place_sl_order(self, symbol: str, side: str, size: float, trigger_price: float) -> OrderResult:
        return OrderResult(success=True, order_id="SL1", fill_price=None, fill_size=None, error=None)

    async def place_tp_order(self, symbol: str, side: str, size: float, trigger_price: float) -> OrderResult:
        return OrderResult(success=True, order_id="TP1", fill_price=None, fill_size=None, error=None)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True

    async def cancel_all_orders(self, symbol: str) -> int:
        return 0

    async def close_position(self, symbol: str) -> OrderResult:
        return OrderResult(success=True, order_id="CL1", fill_price=100.0, fill_size=1.0, error=None)

    async def modify_sl(self, symbol: str, new_price: float) -> OrderResult:
        return OrderResult(success=True, order_id="SL2", fill_price=None, fill_size=None, error=None)

    async def modify_tp(self, symbol: str, new_price: float) -> OrderResult:
        return OrderResult(success=True, order_id="TP2", fill_price=None, fill_size=None, error=None)


# ---------------------------------------------------------------------------
# ExchangeAdapter ABC
# ---------------------------------------------------------------------------


class TestExchangeAdapterABC:
    def test_mock_adapter_instantiates(self) -> None:
        adapter = MockAdapter()
        assert adapter.name() == "mock"

    def test_capabilities(self) -> None:
        adapter = MockAdapter()
        caps = adapter.capabilities()
        assert caps.native_sl_tp is True
        assert caps.max_leverage == 50.0

    @pytest.mark.asyncio
    async def test_default_funding_rate_returns_none(self) -> None:
        adapter = MockAdapter()
        result = await adapter.get_funding_rate("BTC-USDC")
        assert result is None

    @pytest.mark.asyncio
    async def test_default_open_interest_returns_none(self) -> None:
        adapter = MockAdapter()
        result = await adapter.get_open_interest("BTC-USDC")
        assert result is None

    @pytest.mark.asyncio
    async def test_place_market_order(self) -> None:
        adapter = MockAdapter()
        result = await adapter.place_market_order("BTC-USDC", "buy", 0.5)
        assert result.success is True
        assert result.fill_size == 0.5


# ---------------------------------------------------------------------------
# ExchangeFactory
# ---------------------------------------------------------------------------


class TestExchangeFactory:
    def setup_method(self) -> None:
        ExchangeFactory.reset()

    def test_register_and_get_adapter(self) -> None:
        ExchangeFactory.register("mock", MockAdapter)
        adapter = ExchangeFactory.get_adapter("mock")
        assert isinstance(adapter, MockAdapter)
        assert adapter.name() == "mock"

    def test_singleton_caching(self) -> None:
        ExchangeFactory.register("mock", MockAdapter)
        a1 = ExchangeFactory.get_adapter("mock")
        a2 = ExchangeFactory.get_adapter("mock")
        assert a1 is a2

    def test_unknown_exchange_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown exchange"):
            ExchangeFactory.get_adapter("nonexistent")

    def test_reset_clears_cache(self) -> None:
        ExchangeFactory.register("mock", MockAdapter)
        ExchangeFactory.get_adapter("mock")
        ExchangeFactory.reset()
        with pytest.raises(ValueError):
            ExchangeFactory.get_adapter("mock")

    def test_multiple_exchanges(self) -> None:
        ExchangeFactory.register("mock_a", MockAdapter)
        ExchangeFactory.register("mock_b", MockAdapter)
        a = ExchangeFactory.get_adapter("mock_a")
        b = ExchangeFactory.get_adapter("mock_b")
        assert a is not b
