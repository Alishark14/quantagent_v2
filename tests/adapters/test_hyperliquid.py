"""Tests for HyperliquidAdapter — all CCXT and HTTP calls mocked."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from engine.types import AdapterCapabilities, OrderResult, Position
from exchanges.hyperliquid import (
    HIP3_SYMBOLS,
    SYMBOL_MAP,
    HyperliquidAdapter,
    _STATIC_COIN_MAP,
    _build_static_coin_map,
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


# ---------------------------------------------------------------------------
# Static coin map
# ---------------------------------------------------------------------------


class TestStaticCoinMap:
    def test_native_perp_maps_to_bare_name(self) -> None:
        assert _STATIC_COIN_MAP["BTC-USDC"] == "BTC"
        assert _STATIC_COIN_MAP["ETH-USDC"] == "ETH"
        assert _STATIC_COIN_MAP["SOL-USDC"] == "SOL"

    def test_hip3_maps_to_xyz_prefixed_name(self) -> None:
        assert _STATIC_COIN_MAP["GOLD-USDC"] == "xyz:GOLD"
        assert _STATIC_COIN_MAP["TSLA-USDC"] == "xyz:TSLA"
        assert _STATIC_COIN_MAP["SP500-USDC"] == "xyz:SP500"

    def test_wtioil_maps_to_cl(self) -> None:
        """WTIOIL→CL is the canonical non-obvious mapping."""
        assert _STATIC_COIN_MAP["WTIOIL-USDC"] == "xyz:CL"

    def test_all_symbol_map_entries_have_coin(self) -> None:
        for sym in SYMBOL_MAP:
            assert sym in _STATIC_COIN_MAP, f"Missing coin map entry for {sym}"


# ---------------------------------------------------------------------------
# Helpers for mocking httpx responses
# ---------------------------------------------------------------------------


def _mock_response(json_data, status_code: int = 200) -> httpx.Response:
    """Build a mock httpx.Response with .json() and .raise_for_status()."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp,
        )
    return resp


# Sample meta responses for tests
_NATIVE_META = {
    "universe": [
        {"name": "BTC", "szDecimals": 5},
        {"name": "ETH", "szDecimals": 4},
    ]
}

_HIP3_META = {
    "universe": [
        {"name": "xyz:GOLD", "szDecimals": 2},
        {"name": "xyz:CL", "szDecimals": 3},
    ]
}


# ---------------------------------------------------------------------------
# Asset registry + coin resolver
# ---------------------------------------------------------------------------


class TestAssetRegistry:
    @pytest.mark.asyncio
    async def test_build_registry_from_meta(self, adapter: HyperliquidAdapter) -> None:
        """Registry merges native + HIP-3 meta into _coin_map."""
        async def _mock_post(url, json=None):
            if json.get("dex") == "xyz":
                return _mock_response(_HIP3_META)
            return _mock_response(_NATIVE_META)

        mock_client = AsyncMock()
        mock_client.post = _mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exchanges.hyperliquid.httpx.AsyncClient", return_value=mock_client):
            await adapter._build_asset_registry()

        assert adapter._coin_map is not None
        # Native perps from meta
        assert adapter._coin_map["BTC-USDC"] == "BTC"
        assert adapter._coin_map["ETH-USDC"] == "ETH"
        # HIP-3 from meta — base is stripped from "xyz:GOLD" → canonical "GOLD-USDC"
        assert adapter._coin_map["GOLD-USDC"] == "xyz:GOLD"
        assert adapter._coin_map["CL-USDC"] == "xyz:CL"
        # Static map entries survive for aliases
        assert adapter._coin_map["WTIOIL-USDC"] == "xyz:CL"

    @pytest.mark.asyncio
    async def test_registry_caches_for_24h(self, adapter: HyperliquidAdapter) -> None:
        """Once built, the registry isn't rebuilt within the TTL window."""
        adapter._coin_map = {"BTC-USDC": "BTC"}
        adapter._registry_expires_at = time.time() + 86400  # future

        coin = await adapter._resolve_coin("BTC-USDC")
        assert coin == "BTC"
        # _build_asset_registry should NOT have been called (no HTTP)

    @pytest.mark.asyncio
    async def test_registry_rebuilds_after_expiry(self, adapter: HyperliquidAdapter) -> None:
        """Expired registry triggers a rebuild."""
        adapter._coin_map = {"BTC-USDC": "BTC"}
        adapter._registry_expires_at = time.time() - 1  # expired

        async def _mock_post(url, json=None):
            if json.get("dex") == "xyz":
                return _mock_response(_HIP3_META)
            return _mock_response(_NATIVE_META)

        mock_client = AsyncMock()
        mock_client.post = _mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exchanges.hyperliquid.httpx.AsyncClient", return_value=mock_client):
            coin = await adapter._resolve_coin("BTC-USDC")

        assert coin == "BTC"
        assert adapter._registry_expires_at > time.time()

    @pytest.mark.asyncio
    async def test_registry_meta_failure_falls_back_to_static(self, adapter: HyperliquidAdapter) -> None:
        """If the meta endpoint fails, the static map still works."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("offline"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exchanges.hyperliquid.httpx.AsyncClient", return_value=mock_client):
            await adapter._build_asset_registry()

        # Static map entries should still be present
        assert adapter._coin_map["BTC-USDC"] == "BTC"
        assert adapter._coin_map["GOLD-USDC"] == "xyz:GOLD"


class TestResolveCoin:
    @pytest.mark.asyncio
    async def test_resolve_native_perp(self, adapter: HyperliquidAdapter) -> None:
        adapter._coin_map = dict(_STATIC_COIN_MAP)
        adapter._registry_expires_at = time.time() + 86400
        assert await adapter._resolve_coin("BTC-USDC") == "BTC"

    @pytest.mark.asyncio
    async def test_resolve_hip3_gold(self, adapter: HyperliquidAdapter) -> None:
        adapter._coin_map = dict(_STATIC_COIN_MAP)
        adapter._registry_expires_at = time.time() + 86400
        assert await adapter._resolve_coin("GOLD-USDC") == "xyz:GOLD"

    @pytest.mark.asyncio
    async def test_resolve_wtioil_to_cl(self, adapter: HyperliquidAdapter) -> None:
        adapter._coin_map = dict(_STATIC_COIN_MAP)
        adapter._registry_expires_at = time.time() + 86400
        assert await adapter._resolve_coin("WTIOIL-USDC") == "xyz:CL"

    @pytest.mark.asyncio
    async def test_resolve_unknown_falls_back_to_base(self, adapter: HyperliquidAdapter) -> None:
        adapter._coin_map = {"BTC-USDC": "BTC"}
        adapter._registry_expires_at = time.time() + 86400
        coin = await adapter._resolve_coin("NEWCOIN-USDC")
        assert coin == "NEWCOIN"


# ---------------------------------------------------------------------------
# Funding rate (direct API)
# ---------------------------------------------------------------------------


class TestFundingRate:
    @pytest.mark.asyncio
    async def test_get_funding_rate_native(self, adapter: HyperliquidAdapter) -> None:
        """Native perp funding rate fetched with bare coin name."""
        adapter._coin_map = dict(_STATIC_COIN_MAP)
        adapter._registry_expires_at = time.time() + 86400

        funding_data = [
            {"coin": "BTC", "fundingRate": "0.00005", "time": 1700000000},
            {"coin": "BTC", "fundingRate": "0.00012", "time": 1700003600},
        ]

        async def _mock_post(url, json=None):
            return _mock_response(funding_data)

        mock_client = AsyncMock()
        mock_client.post = _mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exchanges.hyperliquid.httpx.AsyncClient", return_value=mock_client):
            rate = await adapter.get_funding_rate("BTC-USDC")

        assert rate == 0.00012  # latest entry

    @pytest.mark.asyncio
    async def test_get_funding_rate_hip3_gold(self, adapter: HyperliquidAdapter) -> None:
        """HIP-3 funding rate uses xyz:GOLD as the coin name."""
        adapter._coin_map = dict(_STATIC_COIN_MAP)
        adapter._registry_expires_at = time.time() + 86400

        funding_data = [
            {"coin": "xyz:GOLD", "fundingRate": "-0.0003", "time": 1700000000},
        ]
        captured_payloads: list[dict] = []

        async def _mock_post(url, json=None):
            captured_payloads.append(json)
            return _mock_response(funding_data)

        mock_client = AsyncMock()
        mock_client.post = _mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exchanges.hyperliquid.httpx.AsyncClient", return_value=mock_client):
            rate = await adapter.get_funding_rate("GOLD-USDC")

        assert rate == -0.0003
        # Verify the API was called with the correct coin name
        assert any(p.get("coin") == "xyz:GOLD" for p in captured_payloads)

    @pytest.mark.asyncio
    async def test_get_funding_rate_empty_response(self, adapter: HyperliquidAdapter) -> None:
        adapter._coin_map = dict(_STATIC_COIN_MAP)
        adapter._registry_expires_at = time.time() + 86400

        async def _mock_post(url, json=None):
            return _mock_response([])

        mock_client = AsyncMock()
        mock_client.post = _mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exchanges.hyperliquid.httpx.AsyncClient", return_value=mock_client):
            rate = await adapter.get_funding_rate("BTC-USDC")

        assert rate is None

    @pytest.mark.asyncio
    async def test_get_funding_rate_http_failure(self, adapter: HyperliquidAdapter) -> None:
        adapter._coin_map = dict(_STATIC_COIN_MAP)
        adapter._registry_expires_at = time.time() + 86400

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("offline"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exchanges.hyperliquid.httpx.AsyncClient", return_value=mock_client):
            rate = await adapter.get_funding_rate("BTC-USDC")

        assert rate is None


# ---------------------------------------------------------------------------
# Open interest (direct API)
# ---------------------------------------------------------------------------


class TestOpenInterest:
    @pytest.mark.asyncio
    async def test_get_oi_native(self, adapter: HyperliquidAdapter) -> None:
        adapter._coin_map = dict(_STATIC_COIN_MAP)
        adapter._registry_expires_at = time.time() + 86400

        meta_ctx = [
            {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
            [{"openInterest": "5000000.5"}, {"openInterest": "2000000"}],
        ]

        async def _mock_post(url, json=None):
            return _mock_response(meta_ctx)

        mock_client = AsyncMock()
        mock_client.post = _mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exchanges.hyperliquid.httpx.AsyncClient", return_value=mock_client):
            oi = await adapter.get_open_interest("BTC-USDC")

        assert oi == 5000000.5

    @pytest.mark.asyncio
    async def test_get_oi_hip3(self, adapter: HyperliquidAdapter) -> None:
        """HIP-3 OI query sends dex=xyz and looks up by stripped name."""
        adapter._coin_map = dict(_STATIC_COIN_MAP)
        adapter._registry_expires_at = time.time() + 86400

        meta_ctx = [
            {"universe": [{"name": "GOLD"}, {"name": "CL"}]},
            [{"openInterest": "1234"}, {"openInterest": "5678"}],
        ]
        captured_payloads: list[dict] = []

        async def _mock_post(url, json=None):
            captured_payloads.append(json)
            return _mock_response(meta_ctx)

        mock_client = AsyncMock()
        mock_client.post = _mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exchanges.hyperliquid.httpx.AsyncClient", return_value=mock_client):
            oi = await adapter.get_open_interest("GOLD-USDC")

        assert oi == 1234.0
        # Verify the payload had dex=xyz for HIP-3
        assert any(p.get("dex") == "xyz" for p in captured_payloads)

    @pytest.mark.asyncio
    async def test_get_oi_coin_not_in_response(self, adapter: HyperliquidAdapter) -> None:
        adapter._coin_map = dict(_STATIC_COIN_MAP)
        adapter._registry_expires_at = time.time() + 86400

        meta_ctx = [
            {"universe": [{"name": "ETH"}]},
            [{"openInterest": "2000000"}],
        ]

        async def _mock_post(url, json=None):
            return _mock_response(meta_ctx)

        mock_client = AsyncMock()
        mock_client.post = _mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exchanges.hyperliquid.httpx.AsyncClient", return_value=mock_client):
            oi = await adapter.get_open_interest("BTC-USDC")

        assert oi is None

    @pytest.mark.asyncio
    async def test_get_oi_http_failure(self, adapter: HyperliquidAdapter) -> None:
        adapter._coin_map = dict(_STATIC_COIN_MAP)
        adapter._registry_expires_at = time.time() + 86400

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("offline"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("exchanges.hyperliquid.httpx.AsyncClient", return_value=mock_client):
            oi = await adapter.get_open_interest("BTC-USDC")

        assert oi is None


# ---------------------------------------------------------------------------
# Credential resolution — testnet vs mainnet env var precedence
# ---------------------------------------------------------------------------


class TestCredentialResolution:
    """Verify the constructor reads the right env vars per mode.

    Regression: testnet runs used to share mainnet credentials because the
    constructor read ``HYPERLIQUID_WALLET_ADDRESS`` / ``HYPERLIQUID_PRIVATE_KEY``
    unconditionally. Operators wanted dedicated testnet vars so the wrong key
    can never be sent to the wrong endpoint.
    """

    def test_testnet_prefers_dedicated_env_vars(self, monkeypatch) -> None:
        """When ``testnet=True`` and BOTH pairs are set, the testnet pair wins."""
        monkeypatch.setenv("HYPERLIQUID_WALLET_ADDRESS", "0xMAINNET_WALLET")
        monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "MAINNET_KEY")
        monkeypatch.setenv("HYPERLIQUID_TESTNET_WALLET_ADDRESS", "0xTESTNET_WALLET")
        monkeypatch.setenv("HYPERLIQUID_TESTNET_PRIVATE_KEY", "TESTNET_KEY")

        captured: dict = {}
        with patch("exchanges.hyperliquid.ccxt") as mock_ccxt:
            def _capture(config):
                captured.update(config)
                return MagicMock()
            mock_ccxt.hyperliquid.side_effect = _capture
            HyperliquidAdapter(testnet=True)

        assert captured["walletAddress"] == "0xTESTNET_WALLET"
        assert captured["privateKey"] == "TESTNET_KEY"

    def test_testnet_falls_back_to_mainnet_vars(self, monkeypatch) -> None:
        """If only the mainnet pair is set, testnet mode falls back to it (BC)."""
        monkeypatch.delenv("HYPERLIQUID_TESTNET_WALLET_ADDRESS", raising=False)
        monkeypatch.delenv("HYPERLIQUID_TESTNET_PRIVATE_KEY", raising=False)
        monkeypatch.setenv("HYPERLIQUID_WALLET_ADDRESS", "0xMAINNET_WALLET")
        monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "MAINNET_KEY")

        captured: dict = {}
        with patch("exchanges.hyperliquid.ccxt") as mock_ccxt:
            def _capture(config):
                captured.update(config)
                return MagicMock()
            mock_ccxt.hyperliquid.side_effect = _capture
            HyperliquidAdapter(testnet=True)

        assert captured["walletAddress"] == "0xMAINNET_WALLET"
        assert captured["privateKey"] == "MAINNET_KEY"

    def test_live_mode_ignores_testnet_vars(self, monkeypatch) -> None:
        """Live mode reads ONLY the mainnet vars even if testnet vars are set."""
        monkeypatch.setenv("HYPERLIQUID_WALLET_ADDRESS", "0xMAINNET_WALLET")
        monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "MAINNET_KEY")
        monkeypatch.setenv("HYPERLIQUID_TESTNET_WALLET_ADDRESS", "0xTESTNET_WALLET")
        monkeypatch.setenv("HYPERLIQUID_TESTNET_PRIVATE_KEY", "TESTNET_KEY")
        monkeypatch.delenv("HYPERLIQUID_TESTNET", raising=False)

        captured: dict = {}
        with patch("exchanges.hyperliquid.ccxt") as mock_ccxt:
            def _capture(config):
                captured.update(config)
                return MagicMock()
            mock_ccxt.hyperliquid.side_effect = _capture
            HyperliquidAdapter(testnet=False)

        assert captured["walletAddress"] == "0xMAINNET_WALLET"
        assert captured["privateKey"] == "MAINNET_KEY"

    def test_explicit_kwargs_override_env(self, monkeypatch) -> None:
        """Explicitly-passed wallet/key override the env-var resolution path."""
        monkeypatch.setenv("HYPERLIQUID_TESTNET_WALLET_ADDRESS", "0xTESTNET_WALLET")
        monkeypatch.setenv("HYPERLIQUID_TESTNET_PRIVATE_KEY", "TESTNET_KEY")

        captured: dict = {}
        with patch("exchanges.hyperliquid.ccxt") as mock_ccxt:
            def _capture(config):
                captured.update(config)
                return MagicMock()
            mock_ccxt.hyperliquid.side_effect = _capture
            HyperliquidAdapter(
                wallet_address="0xEXPLICIT",
                private_key="EXPLICIT_KEY",
                testnet=True,
            )

        assert captured["walletAddress"] == "0xEXPLICIT"
        assert captured["privateKey"] == "EXPLICIT_KEY"
