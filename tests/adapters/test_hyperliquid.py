"""Tests for HyperliquidAdapter — all CCXT calls mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from engine.types import AdapterCapabilities, OrderResult, Position
from exchanges.hyperliquid import (
    HIP3_SYMBOLS,
    SYMBOL_MAP,
    HyperliquidAdapter,
    _pos_size,
    _REVERSE_MAP,
)


# ---------------------------------------------------------------------------
# Fixture: adapter with fully mocked ccxt exchange
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter() -> HyperliquidAdapter:
    with patch("exchanges.hyperliquid.ccxt") as mock_ccxt:
        mock_exchange = MagicMock()
        mock_ccxt.hyperliquid.return_value = mock_exchange
        a = HyperliquidAdapter(wallet_address="0xtest", private_key="testkey")
        a._exchange = mock_exchange
        return a


# ---------------------------------------------------------------------------
# Symbol conversion
# ---------------------------------------------------------------------------


class TestSymbolConversion:
    def test_standard_to_ccxt(self, adapter: HyperliquidAdapter) -> None:
        assert adapter._to_ccxt_symbol("BTC-USDC") == "BTC/USDC:USDC"
        assert adapter._to_ccxt_symbol("ETH-USDC") == "ETH/USDC:USDC"
        assert adapter._to_ccxt_symbol("SOL-USDC") == "SOL/USDC:USDC"

    def test_hip3_to_ccxt(self, adapter: HyperliquidAdapter) -> None:
        assert adapter._to_ccxt_symbol("GOLD-USDC") == "XYZ-GOLD/USDC:USDC"
        assert adapter._to_ccxt_symbol("SP500-USDC") == "XYZ-SP500/USDC:USDC"
        assert adapter._to_ccxt_symbol("TSLA-USDC") == "XYZ-TSLA/USDC:USDC"
        assert adapter._to_ccxt_symbol("EUR-USDC") == "XYZ-EUR/USDC:USDC"

    def test_already_ccxt_format_passthrough(self, adapter: HyperliquidAdapter) -> None:
        assert adapter._to_ccxt_symbol("BTC/USDC:USDC") == "BTC/USDC:USDC"

    def test_unknown_symbol_fallback(self, adapter: HyperliquidAdapter) -> None:
        assert adapter._to_ccxt_symbol("NEWCOIN-USDC") == "NEWCOIN/USDC:USDC"

    def test_from_ccxt_standard(self, adapter: HyperliquidAdapter) -> None:
        assert adapter._from_ccxt_symbol("BTC/USDC:USDC") == "BTC-USDC"
        assert adapter._from_ccxt_symbol("ETH/USDC:USDC") == "ETH-USDC"

    def test_from_ccxt_hip3(self, adapter: HyperliquidAdapter) -> None:
        assert adapter._from_ccxt_symbol("XYZ-GOLD/USDC:USDC") == "GOLD-USDC"
        assert adapter._from_ccxt_symbol("XYZ-TSLA/USDC:USDC") == "TSLA-USDC"

    def test_from_ccxt_unknown(self, adapter: HyperliquidAdapter) -> None:
        assert adapter._from_ccxt_symbol("ABC/USDC:USDC") == "ABC-USDC"
        assert adapter._from_ccxt_symbol("XYZ-NEW/USDC:USDC") == "NEW-USDC"

    def test_hip3_symbols_set(self) -> None:
        assert "XYZ-GOLD/USDC:USDC" in HIP3_SYMBOLS
        assert "XYZ-TSLA/USDC:USDC" in HIP3_SYMBOLS
        assert "BTC/USDC:USDC" not in HIP3_SYMBOLS

    def test_hip3_params(self, adapter: HyperliquidAdapter) -> None:
        assert adapter._hip3_params("XYZ-GOLD/USDC:USDC") == {"dex": "xyz"}
        assert adapter._hip3_params("BTC/USDC:USDC") == {}

    def test_reverse_map_completeness(self) -> None:
        assert len(_REVERSE_MAP) == len(SYMBOL_MAP)


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_capabilities(self, adapter: HyperliquidAdapter) -> None:
        caps = adapter.capabilities()
        assert isinstance(caps, AdapterCapabilities)
        assert caps.native_sl_tp is True
        assert caps.supports_short is True
        assert caps.market_hours is None
        assert caps.has_funding_rate is True
        assert caps.has_oi_data is True
        assert caps.max_leverage == 50.0

    def test_name(self, adapter: HyperliquidAdapter) -> None:
        assert adapter.name() == "hyperliquid"


# ---------------------------------------------------------------------------
# Data methods
# ---------------------------------------------------------------------------


class TestFetchOHLCV:
    @pytest.mark.asyncio
    async def test_fetch_ohlcv_success(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.fetch_ohlcv.return_value = [
            [1700000000, 67000, 67500, 66800, 67200, 1500],
            [1700003600, 67200, 67800, 67100, 67600, 1200],
        ]
        result = await adapter.fetch_ohlcv("BTC-USDC", "1h", 2)
        assert len(result) == 2
        assert result[0]["close"] == 67200
        assert result[1]["open"] == 67200
        adapter._exchange.fetch_ohlcv.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_error_returns_empty(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.fetch_ohlcv.side_effect = Exception("API down")
        result = await adapter.fetch_ohlcv("BTC-USDC", "1h")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_hip3_passes_dex_param(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.fetch_ohlcv.return_value = []
        await adapter.fetch_ohlcv("GOLD-USDC", "1h")
        call_kwargs = adapter._exchange.fetch_ohlcv.call_args
        assert call_kwargs.kwargs.get("params") == {"dex": "xyz"} or call_kwargs[1].get("params") == {"dex": "xyz"}


# ---------------------------------------------------------------------------
# Order methods
# ---------------------------------------------------------------------------


class TestPlaceOrders:
    @pytest.mark.asyncio
    async def test_place_market_order_success(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.fetch_ticker.return_value = {"last": 67200, "bid": 67190, "ask": 67210}
        adapter._exchange.create_order.return_value = {
            "id": "HL-001",
            "average": 67200.5,
            "filled": 0.5,
            "status": "filled",
        }
        result = await adapter.place_market_order("BTC-USDC", "buy", 0.5)
        assert isinstance(result, OrderResult)
        assert result.success is True
        assert result.order_id == "HL-001"
        assert result.fill_price == 67200.5
        assert result.fill_size == 0.5

    @pytest.mark.asyncio
    async def test_place_market_order_failure(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.fetch_ticker.side_effect = Exception("Connection lost")
        result = await adapter.place_market_order("BTC-USDC", "buy", 0.5)
        assert result.success is False
        assert result.error is not None
        assert "Connection lost" in result.error

    @pytest.mark.asyncio
    async def test_place_sl_order_success(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.create_order.return_value = {"id": "SL-001", "status": "pending"}
        result = await adapter.place_sl_order("BTC-USDC", "sell", 0.5, 66000.0)
        assert result.success is True
        assert result.order_id == "SL-001"
        call_args = adapter._exchange.create_order.call_args
        params = call_args[0][5]
        assert params["stopPrice"] == 66000.0
        assert params["reduceOnly"] is True

    @pytest.mark.asyncio
    async def test_place_tp_order_success(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.create_order.return_value = {"id": "TP-001", "status": "pending"}
        result = await adapter.place_tp_order("BTC-USDC", "sell", 0.25, 70000.0)
        assert result.success is True
        call_args = adapter._exchange.create_order.call_args
        params = call_args[0][5]
        assert params["takeProfitPrice"] == 70000.0
        assert params["reduceOnly"] is True

    @pytest.mark.asyncio
    async def test_place_sl_order_failure(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.create_order.side_effect = Exception("Insufficient margin")
        result = await adapter.place_sl_order("BTC-USDC", "sell", 0.5, 66000.0)
        assert result.success is False
        assert "Insufficient margin" in result.error

    @pytest.mark.asyncio
    async def test_place_limit_order_success(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.create_order.return_value = {"id": "LIM-001", "price": 65000.0, "filled": 0}
        result = await adapter.place_limit_order("BTC-USDC", "buy", 0.5, 65000.0)
        assert result.success is True
        assert result.order_id == "LIM-001"


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


class TestPositions:
    @pytest.mark.asyncio
    async def test_get_positions_with_symbol(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.fetch_positions.return_value = [
            {
                "symbol": "BTC/USDC:USDC",
                "side": "long",
                "contracts": 0.5,
                "entryPrice": 67000.0,
                "unrealizedPnl": 100.0,
                "leverage": 5,
                "info": {},
            }
        ]
        positions = await adapter.get_positions("BTC-USDC")
        assert len(positions) == 1
        assert positions[0].symbol == "BTC-USDC"
        assert positions[0].direction == "long"
        assert positions[0].size == 0.5

    @pytest.mark.asyncio
    async def test_get_positions_empty(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.fetch_positions.return_value = []
        positions = await adapter.get_positions("BTC-USDC")
        assert positions == []

    @pytest.mark.asyncio
    async def test_get_positions_error(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.fetch_positions.side_effect = Exception("API error")
        positions = await adapter.get_positions("BTC-USDC")
        assert positions == []


# ---------------------------------------------------------------------------
# Cancel + close
# ---------------------------------------------------------------------------


class TestCancelAndClose:
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.cancel_order.return_value = {"status": "canceled"}
        result = await adapter.cancel_order("BTC-USDC", "ORD-123")
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.cancel_order.side_effect = Exception("Not found")
        result = await adapter.cancel_order("BTC-USDC", "ORD-123")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.fetch_open_orders.return_value = [
            {"id": "O1"}, {"id": "O2"}, {"id": "O3"},
        ]
        count = await adapter.cancel_all_orders("BTC-USDC")
        assert count == 3

    @pytest.mark.asyncio
    async def test_close_position(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.fetch_positions.return_value = [
            {"symbol": "BTC/USDC:USDC", "side": "long", "contracts": 0.5,
             "entryPrice": 67000, "unrealizedPnl": 0, "leverage": 5, "info": {}},
        ]
        adapter._exchange.fetch_ticker.return_value = {"last": 67500}
        adapter._exchange.create_order.return_value = {"id": "CL1", "average": 67500, "filled": 0.5}
        result = await adapter.close_position("BTC-USDC")
        assert result.success is True
        # Should place a sell order to close long
        call_args = adapter._exchange.create_order.call_args[0]
        assert call_args[2] == "sell"


# ---------------------------------------------------------------------------
# _pos_size helper
# ---------------------------------------------------------------------------


class TestPosSize:
    def test_from_contracts(self) -> None:
        assert _pos_size({"contracts": 1.5}) == 1.5

    def test_from_szi(self) -> None:
        assert _pos_size({"contracts": 0, "info": {"szi": "-2.5"}}) == 2.5

    def test_zero(self) -> None:
        assert _pos_size({"contracts": 0, "info": {}}) == 0.0

    def test_negative_contracts(self) -> None:
        assert _pos_size({"contracts": -1.0}) == 1.0


# ---------------------------------------------------------------------------
# Flow data
# ---------------------------------------------------------------------------


class TestFlowData:
    @pytest.mark.asyncio
    async def test_get_funding_rate(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.fetch_funding_rate.return_value = {"fundingRate": 0.0001}
        rate = await adapter.get_funding_rate("BTC-USDC")
        assert rate == 0.0001

    @pytest.mark.asyncio
    async def test_get_funding_rate_failure(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.fetch_funding_rate.side_effect = Exception("fail")
        rate = await adapter.get_funding_rate("BTC-USDC")
        assert rate is None

    @pytest.mark.asyncio
    async def test_get_open_interest(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.fetch_open_interest.return_value = {"openInterestAmount": 5000000}
        oi = await adapter.get_open_interest("BTC-USDC")
        assert oi == 5000000

    @pytest.mark.asyncio
    async def test_get_open_interest_failure(self, adapter: HyperliquidAdapter) -> None:
        adapter._exchange.fetch_open_interest.side_effect = Exception("fail")
        oi = await adapter.get_open_interest("BTC-USDC")
        assert oi is None
