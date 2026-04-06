"""Abstract FlowProvider interface.

FlowProviders fetch market positioning data (funding, OI, GEX, etc.)
from exchange adapters or external APIs. Each returns a partial dict
that FlowAgent merges into a FlowOutput.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from exchanges.base import ExchangeAdapter


class FlowProvider(ABC):
    """Base class for all flow data providers."""

    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this provider."""
        ...

    @abstractmethod
    async def fetch(self, symbol: str, adapter: ExchangeAdapter) -> dict:
        """Fetch flow data and return a partial dict.

        Keys may include: funding_rate, funding_signal, open_interest,
        oi_change_4h, oi_trend, nearest_liquidation_above,
        nearest_liquidation_below, gex_regime, gex_flip_level.

        FlowAgent merges dicts from all providers.
        """
        ...
