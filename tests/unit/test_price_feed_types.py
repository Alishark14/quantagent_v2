"""Unit tests for PriceFeed payload dataclasses and the paired Event wrappers.

Sprint Week 7 Task 1 — covers round-trip serialization, optional-field
defaults, and timestamp handling for PriceUpdate / CandleClose /
FundingUpdate / OIUpdate, plus the four Event subclasses in
`engine.events` that carry them on the bus.
"""

from datetime import datetime, timezone

from engine.events import (
    CandleClosed,
    FundingUpdated,
    OpenInterestUpdated,
    PriceUpdated,
)
from engine.types import (
    CandleClose,
    FundingUpdate,
    OIUpdate,
    PriceUpdate,
)


# ---------------------------------------------------------------------------
# PriceUpdate
# ---------------------------------------------------------------------------


class TestPriceUpdate:
    def test_required_fields_only(self) -> None:
        update = PriceUpdate(symbol="BTC-USDC", price=65432.10)
        assert update.symbol == "BTC-USDC"
        assert update.price == 65432.10
        assert update.bid is None
        assert update.ask is None
        assert update.size is None
        assert update.exchange == ""
        assert isinstance(update.timestamp, datetime)
        assert update.timestamp.tzinfo is timezone.utc

    def test_all_fields_populated(self) -> None:
        ts = datetime(2026, 4, 11, 12, 30, 0, tzinfo=timezone.utc)
        update = PriceUpdate(
            symbol="ETH-USDC",
            price=3200.5,
            bid=3200.25,
            ask=3200.75,
            size=0.42,
            exchange="hyperliquid",
            timestamp=ts,
        )
        assert update.bid == 3200.25
        assert update.ask == 3200.75
        assert update.size == 0.42
        assert update.exchange == "hyperliquid"
        assert update.timestamp == ts

    def test_to_dict_round_trip(self) -> None:
        ts = datetime(2026, 4, 11, 12, 30, 0, tzinfo=timezone.utc)
        update = PriceUpdate(symbol="SOL-USDC", price=142.0, exchange="hyperliquid", timestamp=ts)
        d = update.to_dict()
        assert d == {
            "symbol": "SOL-USDC",
            "price": 142.0,
            "bid": None,
            "ask": None,
            "size": None,
            "exchange": "hyperliquid",
            "timestamp": ts,
        }
        assert isinstance(d["timestamp"], datetime)


# ---------------------------------------------------------------------------
# CandleClose
# ---------------------------------------------------------------------------


class TestCandleClose:
    def test_required_fields_only(self) -> None:
        candle = CandleClose(
            symbol="BTC-USDC",
            timeframe="1h",
            open=65000.0,
            high=65500.0,
            low=64800.0,
            close=65400.0,
            volume=123.45,
        )
        assert candle.symbol == "BTC-USDC"
        assert candle.timeframe == "1h"
        assert candle.open == 65000.0
        assert candle.high == 65500.0
        assert candle.low == 64800.0
        assert candle.close == 65400.0
        assert candle.volume == 123.45
        assert candle.exchange == ""
        assert isinstance(candle.timestamp, datetime)
        assert candle.timestamp.tzinfo is timezone.utc

    def test_to_dict_round_trip(self) -> None:
        ts = datetime(2026, 4, 11, 13, 0, 0, tzinfo=timezone.utc)
        candle = CandleClose(
            symbol="ETH-USDC",
            timeframe="4h",
            open=3100.0,
            high=3250.0,
            low=3080.0,
            close=3200.0,
            volume=987.6,
            exchange="hyperliquid",
            timestamp=ts,
        )
        d = candle.to_dict()
        assert d == {
            "symbol": "ETH-USDC",
            "timeframe": "4h",
            "open": 3100.0,
            "high": 3250.0,
            "low": 3080.0,
            "close": 3200.0,
            "volume": 987.6,
            "exchange": "hyperliquid",
            "timestamp": ts,
        }


# ---------------------------------------------------------------------------
# FundingUpdate
# ---------------------------------------------------------------------------


class TestFundingUpdate:
    def test_required_fields_only(self) -> None:
        update = FundingUpdate(symbol="BTC-USDC", funding_rate=0.0001)
        assert update.symbol == "BTC-USDC"
        assert update.funding_rate == 0.0001
        assert update.next_funding_time is None
        assert update.exchange == ""
        assert isinstance(update.timestamp, datetime)
        assert update.timestamp.tzinfo is timezone.utc

    def test_to_dict_round_trip(self) -> None:
        ts = datetime(2026, 4, 11, 14, 0, 0, tzinfo=timezone.utc)
        next_ts = datetime(2026, 4, 11, 16, 0, 0, tzinfo=timezone.utc)
        update = FundingUpdate(
            symbol="ETH-USDC",
            funding_rate=-0.00025,
            next_funding_time=next_ts,
            exchange="hyperliquid",
            timestamp=ts,
        )
        d = update.to_dict()
        assert d == {
            "symbol": "ETH-USDC",
            "funding_rate": -0.00025,
            "next_funding_time": next_ts,
            "exchange": "hyperliquid",
            "timestamp": ts,
        }


# ---------------------------------------------------------------------------
# OIUpdate
# ---------------------------------------------------------------------------


class TestOIUpdate:
    def test_required_fields_only(self) -> None:
        update = OIUpdate(symbol="BTC-USDC", open_interest=1_500_000_000.0)
        assert update.symbol == "BTC-USDC"
        assert update.open_interest == 1_500_000_000.0
        assert update.oi_change_pct is None
        assert update.exchange == ""
        assert isinstance(update.timestamp, datetime)
        assert update.timestamp.tzinfo is timezone.utc

    def test_to_dict_round_trip(self) -> None:
        ts = datetime(2026, 4, 11, 15, 0, 0, tzinfo=timezone.utc)
        update = OIUpdate(
            symbol="SOL-USDC",
            open_interest=250_000_000.0,
            oi_change_pct=0.0135,
            exchange="hyperliquid",
            timestamp=ts,
        )
        d = update.to_dict()
        assert d == {
            "symbol": "SOL-USDC",
            "open_interest": 250_000_000.0,
            "oi_change_pct": 0.0135,
            "exchange": "hyperliquid",
            "timestamp": ts,
        }


# ---------------------------------------------------------------------------
# Event wrappers — make sure the bus-side classes exist and carry the
# correct payload field, matching the existing DataReady/TradeClosed pattern.
# ---------------------------------------------------------------------------


class TestPriceFeedEvents:
    def test_price_updated_wraps_payload(self) -> None:
        payload = PriceUpdate(symbol="BTC-USDC", price=65000.0)
        event = PriceUpdated(source="price_feed", update=payload)
        assert event.source == "price_feed"
        assert event.update is payload
        assert isinstance(event.timestamp, datetime)

    def test_candle_closed_wraps_payload(self) -> None:
        payload = CandleClose(
            symbol="ETH-USDC",
            timeframe="1h",
            open=3100.0,
            high=3250.0,
            low=3080.0,
            close=3200.0,
            volume=987.6,
        )
        event = CandleClosed(source="price_feed", candle=payload)
        assert event.candle is payload
        assert event.candle.timeframe == "1h"

    def test_funding_updated_wraps_payload(self) -> None:
        payload = FundingUpdate(symbol="BTC-USDC", funding_rate=0.0001)
        event = FundingUpdated(source="price_feed", update=payload)
        assert event.update is payload
        assert event.update.funding_rate == 0.0001

    def test_open_interest_updated_wraps_payload(self) -> None:
        payload = OIUpdate(symbol="BTC-USDC", open_interest=1_500_000_000.0)
        event = OpenInterestUpdated(source="price_feed", update=payload)
        assert event.update is payload
        assert event.update.open_interest == 1_500_000_000.0
