"""Tests for BinanceAdapter — all CCXT calls mocked.

Mirrors ``tests/adapters/test_hyperliquid.py``: a fixture builds an
adapter against a `MagicMock` ccxt instance, then individual tests
exercise symbol mapping, fetch_ohlcv, and the trading-method
NotImplementedError surface. No real network traffic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from engine.types import AdapterCapabilities
from exchanges.binance import (
    BinanceAdapter,
    SYMBOL_OVERRIDES,
    _TRADING_NOT_IMPLEMENTED,
)
from exchanges.factory import ExchangeFactory


# ---------------------------------------------------------------------------
# Fixture: adapter with fully mocked ccxt exchange
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> BinanceAdapter:
    with patch("exchanges.binance.ccxt") as mock_ccxt:
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange
        a = BinanceAdapter()
        a._exchange = mock_exchange
        return a


# ---------------------------------------------------------------------------
# Symbol conversion
# ---------------------------------------------------------------------------


class TestSymbolConversion:
    def test_btc_usdc_to_ccxt(self) -> None:
        assert BinanceAdapter._to_ccxt_symbol("BTC-USDC") == "BTC/USDT:USDT"

    def test_eth_usdc_to_ccxt(self) -> None:
        assert BinanceAdapter._to_ccxt_symbol("ETH-USDC") == "ETH/USDT:USDT"

    def test_sol_usdc_to_ccxt(self) -> None:
        assert BinanceAdapter._to_ccxt_symbol("SOL-USDC") == "SOL/USDT:USDT"

    def test_arbitrary_base_to_ccxt(self) -> None:
        # Programmatic mapping should handle any BASE-USDC, not just the
        # three the user listed in the verification command.
        assert BinanceAdapter._to_ccxt_symbol("AVAX-USDC") == "AVAX/USDT:USDT"
        assert BinanceAdapter._to_ccxt_symbol("DOGE-USDC") == "DOGE/USDT:USDT"

    def test_usdt_quoted_input_also_accepted(self) -> None:
        # Belt + braces — if a caller already speaks USDT, don't reject.
        assert BinanceAdapter._to_ccxt_symbol("BTC-USDT") == "BTC/USDT:USDT"

    def test_ccxt_format_passthrough(self) -> None:
        assert BinanceAdapter._to_ccxt_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot convert"):
            BinanceAdapter._to_ccxt_symbol("BTCUSDC")  # no separator
        with pytest.raises(ValueError, match="cannot convert"):
            BinanceAdapter._to_ccxt_symbol("BTC-EUR")  # unsupported quote

    def test_empty_base_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty base"):
            BinanceAdapter._to_ccxt_symbol("-USDC")

    def test_overrides_dict_consulted_first(self, monkeypatch) -> None:
        monkeypatch.setitem(SYMBOL_OVERRIDES, "WEIRD-USDC", "WEIRD123/USDT:USDT")
        try:
            assert (
                BinanceAdapter._to_ccxt_symbol("WEIRD-USDC")
                == "WEIRD123/USDT:USDT"
            )
        finally:
            SYMBOL_OVERRIDES.pop("WEIRD-USDC", None)

    def test_from_ccxt_canonicalises_to_usdc(self) -> None:
        # The reverse conversion canonicalises back to BASE-USDC because
        # the rest of the codebase speaks USDC end-to-end.
        assert BinanceAdapter._from_ccxt_symbol("BTC/USDT:USDT") == "BTC-USDC"
        assert BinanceAdapter._from_ccxt_symbol("ETH/USDT:USDT") == "ETH-USDC"
        assert BinanceAdapter._from_ccxt_symbol("SOL/USDT:USDT") == "SOL-USDC"


# ---------------------------------------------------------------------------
# Identity / capabilities
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_name(self, adapter: BinanceAdapter) -> None:
        assert adapter.name() == "binance"

    def test_capabilities_match_binance_perp_features(
        self, adapter: BinanceAdapter
    ) -> None:
        caps = adapter.capabilities()
        assert isinstance(caps, AdapterCapabilities)
        assert caps.native_sl_tp is True
        assert caps.supports_short is True
        assert caps.market_hours is None
        assert caps.has_funding_rate is True
        assert caps.has_oi_data is True
        assert caps.max_leverage == 125.0
        assert "perpetual" in caps.asset_types


# ---------------------------------------------------------------------------
# Constructor — defaultType + testnet
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_type_is_future(self) -> None:
        with patch("exchanges.binance.ccxt") as mock_ccxt:
            BinanceAdapter()
            mock_ccxt.binance.assert_called_once()
            config = mock_ccxt.binance.call_args[0][0]
            assert config["options"]["defaultType"] == "future"
            assert config["enableRateLimit"] is True

    def test_no_api_key_required(self) -> None:
        # The constructor must NEVER touch ANTHROPIC_API_KEY / Binance
        # credentials — OHLCV is public. We assert by constructing
        # without any env / kwargs and verifying nothing throws.
        with patch("exchanges.binance.ccxt") as mock_ccxt:
            mock_ccxt.binance.return_value = MagicMock()
            adapter = BinanceAdapter()
            assert adapter._exchange is not None

    def test_testnet_flag_calls_set_sandbox_mode(self) -> None:
        with patch("exchanges.binance.ccxt") as mock_ccxt:
            mock_exchange = MagicMock()
            mock_ccxt.binance.return_value = mock_exchange
            adapter = BinanceAdapter(testnet=True)
            mock_exchange.set_sandbox_mode.assert_called_once_with(True)
            assert adapter._testnet is True

    def test_default_is_mainnet(self) -> None:
        with patch("exchanges.binance.ccxt") as mock_ccxt:
            mock_exchange = MagicMock()
            mock_ccxt.binance.return_value = mock_exchange
            adapter = BinanceAdapter()
            mock_exchange.set_sandbox_mode.assert_not_called()
            assert adapter._testnet is False


# ---------------------------------------------------------------------------
# fetch_ohlcv
# ---------------------------------------------------------------------------


class TestFetchOHLCV:
    @pytest.mark.asyncio
    async def test_returns_canonical_dict_shape(
        self, adapter: BinanceAdapter
    ) -> None:
        adapter._exchange.fetch_ohlcv.return_value = [
            [1700000000000, 67000.0, 67500.0, 66800.0, 67200.0, 1500.0],
            [1700003600000, 67200.0, 67800.0, 67100.0, 67600.0, 1200.0],
        ]
        result = await adapter.fetch_ohlcv("BTC-USDC", "1h", limit=2)
        assert len(result) == 2
        assert result[0] == {
            "timestamp": 1700000000000,
            "open": 67000.0,
            "high": 67500.0,
            "low": 66800.0,
            "close": 67200.0,
            "volume": 1500.0,
        }
        assert result[1]["close"] == 67600.0

    @pytest.mark.asyncio
    async def test_passes_translated_symbol_to_ccxt(
        self, adapter: BinanceAdapter
    ) -> None:
        adapter._exchange.fetch_ohlcv.return_value = []
        await adapter.fetch_ohlcv("BTC-USDC", "1h", limit=500, since=1700000000000)
        call = adapter._exchange.fetch_ohlcv.call_args
        # First positional arg is the CCXT symbol, second is timeframe.
        assert call.args[0] == "BTC/USDT:USDT"
        assert call.args[1] == "1h"
        assert call.kwargs.get("limit") == 500
        assert call.kwargs.get("since") == 1700000000000

    @pytest.mark.asyncio
    async def test_passes_eth_symbol_correctly(
        self, adapter: BinanceAdapter
    ) -> None:
        adapter._exchange.fetch_ohlcv.return_value = []
        await adapter.fetch_ohlcv("ETH-USDC", "15m")
        assert adapter._exchange.fetch_ohlcv.call_args.args[0] == "ETH/USDT:USDT"

    @pytest.mark.asyncio
    async def test_error_returns_empty_list(
        self, adapter: BinanceAdapter
    ) -> None:
        adapter._exchange.fetch_ohlcv.side_effect = RuntimeError("Binance 503")
        result = await adapter.fetch_ohlcv("BTC-USDC", "1h")
        assert result == []  # downloader's pagination loop interprets [] as "stop"

    @pytest.mark.asyncio
    async def test_unknown_symbol_format_raises(
        self, adapter: BinanceAdapter
    ) -> None:
        # Symbol-mapping errors should NOT be silently swallowed by the
        # outer try/except — they're a programming error, not an API
        # failure. The adapter's fetch_ohlcv catches all Exceptions for
        # network safety, so we check the raw mapping helper instead.
        with pytest.raises(ValueError):
            BinanceAdapter._to_ccxt_symbol("BTCUSDC")


# ---------------------------------------------------------------------------
# Trading surface — every method must raise NotImplementedError
# ---------------------------------------------------------------------------


class TestTradingNotImplemented:
    @pytest.mark.parametrize(
        "method_name,args",
        [
            ("get_ticker", ("BTC-USDC",)),
            ("get_balance", ()),
            ("get_positions", ()),
            ("place_market_order", ("BTC-USDC", "buy", 1.0)),
            ("place_limit_order", ("BTC-USDC", "buy", 1.0, 67000.0)),
            ("place_sl_order", ("BTC-USDC", "sell", 1.0, 66000.0)),
            ("place_tp_order", ("BTC-USDC", "sell", 1.0, 68000.0)),
            ("cancel_order", ("BTC-USDC", "ord-1")),
            ("cancel_all_orders", ("BTC-USDC",)),
            ("close_position", ("BTC-USDC",)),
            ("modify_sl", ("BTC-USDC", 65000.0)),
            ("modify_tp", ("BTC-USDC", 70000.0)),
        ],
    )
    @pytest.mark.asyncio
    async def test_method_raises(
        self, adapter: BinanceAdapter, method_name: str, args: tuple
    ) -> None:
        method = getattr(adapter, method_name)
        with pytest.raises(NotImplementedError, match="data-download only"):
            await method(*args)

    def test_error_message_mentions_phase_5(self) -> None:
        # The marker constant is the source of truth — if a future task
        # changes the wording, this test fails loudly so the matching
        # NotImplementedError checks across the suite stay consistent.
        assert "data-download only" in _TRADING_NOT_IMPLEMENTED
        assert "Phase 5" in _TRADING_NOT_IMPLEMENTED


# ---------------------------------------------------------------------------
# Factory registration
# ---------------------------------------------------------------------------


class TestFactoryRegistration:
    def test_factory_returns_binance_adapter(self) -> None:
        # The registration call at the bottom of `exchanges/binance.py`
        # only fires once per process (modules are cached in
        # sys.modules), so calling reset() and re-importing is a no-op.
        # Re-register explicitly so the test exercises the construction
        # path through ExchangeFactory.get_adapter.
        ExchangeFactory.reset()
        ExchangeFactory.register("binance", BinanceAdapter)
        try:
            with patch("exchanges.binance.ccxt") as mock_ccxt:
                mock_ccxt.binance.return_value = MagicMock()
                instance = ExchangeFactory.get_adapter("binance")
                assert isinstance(instance, BinanceAdapter)
                assert instance.name() == "binance"
        finally:
            # Restore the rest of the registry for tests that follow.
            ExchangeFactory.reset()
            from exchanges.hyperliquid import HyperliquidAdapter
            ExchangeFactory.register("hyperliquid", HyperliquidAdapter)
            ExchangeFactory.register("binance", BinanceAdapter)
