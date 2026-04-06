"""Unit tests for OHLCVFetcher."""

from __future__ import annotations

import pytest

from engine.config import TradingConfig
from engine.data.ohlcv import OHLCVFetcher
from engine.types import AdapterCapabilities, MarketData, OrderResult, Position
from exchanges.base import ExchangeAdapter


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------

def _make_candles(n: int = 50, base_price: float = 100.0) -> list[dict]:
    """Generate n uptrending candle dicts."""
    candles = []
    for i in range(n):
        c = base_price + i * 0.5
        candles.append({
            "timestamp": 1700000000 + i * 3600,
            "open": c - 0.3,
            "high": c + 1.0,
            "low": c - 1.0,
            "close": c,
            "volume": 1000.0,
        })
    return candles


class MockAdapter(ExchangeAdapter):
    """Returns predetermined candle data for testing."""

    def __init__(self, candles: list[dict] | None = None, parent_candles: list[dict] | None = None) -> None:
        self._responses: list[list[dict]] = [
            candles if candles is not None else _make_candles(150),
            parent_candles if parent_candles is not None else _make_candles(50, base_price=95.0),
        ]
        self._fetch_calls: list[tuple[str, str, int]] = []
        self._call_idx = 0

    def name(self) -> str:
        return "mock"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            native_sl_tp=False, supports_short=True, market_hours=None,
            asset_types=["crypto"], margin_type="cross", has_funding_rate=False,
            has_oi_data=False, max_leverage=10.0, order_types=["market"],
            supports_partial_close=False,
        )

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> list[dict]:
        self._fetch_calls.append((symbol, timeframe, limit))
        idx = self._call_idx
        self._call_idx += 1
        if idx < len(self._responses):
            return self._responses[idx][:limit]
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOHLCVFetcher:
    @pytest.mark.asyncio
    async def test_fetch_returns_market_data(self) -> None:
        adapter = MockAdapter()
        config = TradingConfig(symbol="BTC-USDC", timeframe="1h")
        fetcher = OHLCVFetcher(adapter, config)

        result = await fetcher.fetch("BTC-USDC", "1h")

        assert isinstance(result, MarketData)
        assert result.symbol == "BTC-USDC"
        assert result.timeframe == "1h"
        assert result.num_candles == 150
        assert len(result.candles) == 150

    @pytest.mark.asyncio
    async def test_indicators_computed(self) -> None:
        adapter = MockAdapter()
        fetcher = OHLCVFetcher(adapter, TradingConfig())

        result = await fetcher.fetch("BTC-USDC", "1h")

        assert "rsi" in result.indicators
        assert "macd" in result.indicators
        assert "atr" in result.indicators
        assert "bollinger_bands" in result.indicators
        assert isinstance(result.indicators["rsi"], float)

    @pytest.mark.asyncio
    async def test_swings_detected(self) -> None:
        adapter = MockAdapter()
        fetcher = OHLCVFetcher(adapter, TradingConfig())

        result = await fetcher.fetch("BTC-USDC", "1h")

        assert isinstance(result.swing_highs, list)
        assert isinstance(result.swing_lows, list)

    @pytest.mark.asyncio
    async def test_parent_tf_fetched(self) -> None:
        adapter = MockAdapter()
        fetcher = OHLCVFetcher(adapter, TradingConfig())

        result = await fetcher.fetch("BTC-USDC", "1h")

        # Should have made 2 fetch_ohlcv calls: 1h and parent (4h)
        assert len(adapter._fetch_calls) == 2
        assert adapter._fetch_calls[0] == ("BTC-USDC", "1h", 150)
        assert adapter._fetch_calls[1] == ("BTC-USDC", "4h", 50)
        assert result.parent_tf is not None
        assert result.parent_tf.timeframe == "4h"

    @pytest.mark.asyncio
    async def test_15m_parent_is_1h(self) -> None:
        adapter = MockAdapter(candles=_make_candles(100))
        fetcher = OHLCVFetcher(adapter, TradingConfig())

        await fetcher.fetch("ETH-USDC", "15m")

        assert adapter._fetch_calls[0] == ("ETH-USDC", "15m", 100)
        assert adapter._fetch_calls[1] == ("ETH-USDC", "1h", 50)

    @pytest.mark.asyncio
    async def test_lookback_and_forecast_descriptions(self) -> None:
        adapter = MockAdapter()
        fetcher = OHLCVFetcher(adapter, TradingConfig(forecast_candles=3))

        result = await fetcher.fetch("BTC-USDC", "1h")

        assert "~6 days" in result.lookback_description
        assert "~3 hours" in result.forecast_description

    @pytest.mark.asyncio
    async def test_empty_candles_returns_empty_market_data(self) -> None:
        adapter = MockAdapter(candles=[])
        fetcher = OHLCVFetcher(adapter, TradingConfig())

        result = await fetcher.fetch("BTC-USDC", "1h")

        assert result.num_candles == 0
        assert result.candles == []
        assert result.indicators == {}
        assert result.parent_tf is None

    @pytest.mark.asyncio
    async def test_candle_count_from_profile(self) -> None:
        adapter = MockAdapter(candles=_make_candles(200))
        fetcher = OHLCVFetcher(adapter, TradingConfig())

        await fetcher.fetch("BTC-USDC", "1d")

        # 1d profile has 200 candles
        assert adapter._fetch_calls[0][2] == 200

    @pytest.mark.asyncio
    async def test_4h_profile_candle_count(self) -> None:
        adapter = MockAdapter()
        fetcher = OHLCVFetcher(adapter, TradingConfig())

        await fetcher.fetch("BTC-USDC", "4h")

        assert adapter._fetch_calls[0][2] == 150
