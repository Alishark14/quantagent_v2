"""FlowAgent: aggregates positioning data from all enabled providers.

Code-only, zero LLM cost. Merges partial dicts from FlowProviders
into a single FlowOutput with data_richness classification.

Re-exports :class:`FlowSignalAgent` from
``engine.data.flow.signal_agent`` for ergonomic ``from engine.data.flow
import FlowSignalAgent`` imports — the signal agent is the consumer of
``FlowAgent``'s output, but the two live side by side because they share
the same domain.
"""

from __future__ import annotations

import logging

from engine.data.flow.base import FlowProvider
from engine.data.flow.signal_agent import FlowSignalAgent
from engine.types import FlowOutput
from exchanges.base import ExchangeAdapter

logger = logging.getLogger(__name__)

__all__ = ["FlowAgent", "FlowProvider", "FlowSignalAgent"]


class FlowAgent:
    """Aggregates flow data from all enabled providers. Code-only, zero LLM cost."""

    def __init__(
        self,
        providers: list[FlowProvider] | None = None,
        cache=None,
    ) -> None:
        self._providers: list[FlowProvider] = list(providers) if providers else []
        self._cache = cache  # Optional CacheManager

    def add_provider(self, provider: FlowProvider) -> None:
        """Register an additional flow data provider."""
        self._providers.append(provider)

    @property
    def providers(self) -> list[FlowProvider]:
        return list(self._providers)

    async def _fetch_and_merge(self, symbol: str, adapter: ExchangeAdapter) -> FlowOutput:
        """Fetch from all providers, merge, classify richness."""
        merged: dict = {}

        for provider in self._providers:
            try:
                data = await provider.fetch(symbol, adapter)
                merged.update(data)
            except Exception as e:
                logger.warning(f"FlowProvider {provider.name()} failed for {symbol}: {e}")

        has_funding = "funding_rate" in merged
        has_oi = "open_interest" in merged
        if has_funding and has_oi:
            richness = "FULL"
        elif has_funding or has_oi:
            richness = "PARTIAL"
        else:
            richness = "MINIMAL"

        return FlowOutput(
            funding_rate=merged.get("funding_rate"),
            funding_signal=merged.get("funding_signal", "NEUTRAL"),
            oi_change_4h=merged.get("oi_change_4h"),
            oi_trend=merged.get("oi_trend", "STABLE"),
            nearest_liquidation_above=merged.get("nearest_liquidation_above"),
            nearest_liquidation_below=merged.get("nearest_liquidation_below"),
            gex_regime=merged.get("gex_regime"),
            gex_flip_level=merged.get("gex_flip_level"),
            data_richness=richness,
        )

    async def fetch_flow(self, symbol: str, adapter: ExchangeAdapter) -> FlowOutput:
        """Fetch flow data with cache-aside pattern (thundering herd safe).

        Flow data is cached for 5 minutes (FLOW_TTL=300).
        Uses get_or_fetch() for thundering herd protection.
        """
        if self._cache is not None:
            from storage.cache import flow_key, TTL_FLOW
            return await self._cache.get_or_fetch(
                flow_key(symbol),
                lambda: self._fetch_and_merge(symbol, adapter),
                ttl=TTL_FLOW,
            )

        return await self._fetch_and_merge(symbol, adapter)
