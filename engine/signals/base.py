"""Abstract SignalProducer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.types import MarketData, SignalOutput


class SignalProducer(ABC):
    """Base class for all signal producers (LLM agents and ML models)."""

    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this producer."""
        ...

    @abstractmethod
    def signal_type(self) -> str:
        """One of: 'llm', 'ml', 'flow'.

        - ``llm``: Claude / GPT-style text or vision agents.
        - ``ml``: trained statistical / neural models.
        - ``flow``: code-only rules-based agents that interpret order
          flow (funding, OI, liquidations) without an LLM call.

        New strings can be added in the future without breaking the ABC,
        but consumers that filter by ``signal_type`` should treat unknown
        values defensively.
        """
        ...

    @abstractmethod
    def is_enabled(self) -> bool:
        """Check feature flag. Disabled producers are skipped."""
        ...

    @abstractmethod
    async def analyze(self, data: MarketData) -> SignalOutput | None:
        """Produce a signal from market data. Return None if unable."""
        ...

    def requires_vision(self) -> bool:
        """Override to True for vision-based agents."""
        return False
