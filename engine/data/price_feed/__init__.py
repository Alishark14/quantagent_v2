"""PriceFeed: real-time, exchange-agnostic market-data feeds.

Sprint Week 7 — Event-Driven Refactor Phase 1. Replaces the 30-second
REST poll loop in Sentinel with a WebSocket-backed in-memory state that
emits ``PriceUpdated`` / ``CandleClosed`` / ``FundingUpdated`` /
``OpenInterestUpdated`` events on the Event Bus.

Consumers (``Sentinel``, ``SLTPMonitor``, signal agents) never reach
into an exchange adapter directly — they subscribe to the bus and read
from the PriceFeed's memory. Zero REST overhead, tick-level SL/TP.
"""

from __future__ import annotations

from engine.data.price_feed.base import PriceFeed, SymbolState
from engine.data.price_feed.fallback import ConnectionState, RESTFallbackManager
from engine.data.price_feed.hyperliquid import HyperliquidPriceFeed

__all__ = [
    "ConnectionState",
    "HyperliquidPriceFeed",
    "PriceFeed",
    "RESTFallbackManager",
    "SymbolState",
]
