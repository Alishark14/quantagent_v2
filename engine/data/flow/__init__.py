"""FlowAgent: aggregates positioning data from all enabled providers.

Code-only, zero LLM cost. Merges partial dicts from FlowProviders
into a single FlowOutput with data_richness classification.
"""

from __future__ import annotations

import logging

from engine.data.flow.base import FlowProvider
from engine.types import FlowOutput
from exchanges.base import ExchangeAdapter

logger = logging.getLogger(__name__)


class FlowAgent:
    """Aggregates flow data from all enabled providers. Code-only, zero LLM cost."""

    def __init__(self, providers: list[FlowProvider] | None = None) -> None:
        self._providers: list[FlowProvider] = list(providers) if providers else []

    def add_provider(self, provider: FlowProvider) -> None:
        """Register an additional flow data provider."""
        self._providers.append(provider)

    @property
    def providers(self) -> list[FlowProvider]:
        return list(self._providers)

    async def fetch_flow(self, symbol: str, adapter: ExchangeAdapter) -> FlowOutput:
        """Fetch flow data from all providers and merge into FlowOutput.

        Each provider is independently try/excepted — one provider failing
        does not block others. The merged dict is classified by data richness.
        """
        merged: dict = {}

        for provider in self._providers:
            try:
                data = await provider.fetch(symbol, adapter)
                merged.update(data)
            except Exception as e:
                logger.warning(f"FlowProvider {provider.name()} failed for {symbol}: {e}")

        # Classify data richness
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
