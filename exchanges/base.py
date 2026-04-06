"""Abstract ExchangeAdapter and AdapterCapabilities."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.types import AdapterCapabilities, OrderResult, Position


class ExchangeAdapter(ABC):
    """Abstract base for all exchange adapters. One file per exchange."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def capabilities(self) -> AdapterCapabilities: ...

    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> list[dict]: ...

    @abstractmethod
    async def get_ticker(self, symbol: str) -> dict: ...

    @abstractmethod
    async def get_balance(self) -> float: ...

    @abstractmethod
    async def get_positions(self, symbol: str | None = None) -> list[Position]: ...

    @abstractmethod
    async def place_market_order(self, symbol: str, side: str, size: float) -> OrderResult: ...

    @abstractmethod
    async def place_limit_order(self, symbol: str, side: str, size: float, price: float) -> OrderResult: ...

    @abstractmethod
    async def place_sl_order(self, symbol: str, side: str, size: float, trigger_price: float) -> OrderResult: ...

    @abstractmethod
    async def place_tp_order(self, symbol: str, side: str, size: float, trigger_price: float) -> OrderResult: ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool: ...

    @abstractmethod
    async def cancel_all_orders(self, symbol: str) -> int: ...

    @abstractmethod
    async def close_position(self, symbol: str) -> OrderResult: ...

    @abstractmethod
    async def modify_sl(self, symbol: str, new_price: float) -> OrderResult: ...

    @abstractmethod
    async def modify_tp(self, symbol: str, new_price: float) -> OrderResult: ...

    # Optional flow data — return None if exchange doesn't support it
    async def get_funding_rate(self, symbol: str) -> float | None:
        return None

    async def get_open_interest(self, symbol: str) -> float | None:
        return None

    # Optional cost model data — return defaults if exchange doesn't support it
    async def fetch_meta(self) -> list[dict]:
        """Fetch all asset metadata including fee parameters."""
        return []

    async def fetch_orderbook(self, symbol: str, limit: int = 10) -> dict:
        """Fetch orderbook snapshot. Returns {bids: [[price, size], ...], asks: ...}."""
        return {"bids": [], "asks": []}

    async def fetch_user_fees(self) -> dict:
        """Fetch user's current fee tier and discounts."""
        return {"tier": 0, "staking_discount": 0.0, "referral_discount": 0.0}
