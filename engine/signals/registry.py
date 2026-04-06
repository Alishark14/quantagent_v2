"""SignalProducer registry for config-driven agent management."""

from __future__ import annotations

import asyncio
import logging

from engine.signals.base import SignalProducer
from engine.types import MarketData, SignalOutput

logger = logging.getLogger(__name__)


class SignalRegistry:
    """Manages all registered SignalProducers. Config-driven."""

    def __init__(self) -> None:
        self._producers: list[SignalProducer] = []

    def register(self, producer: SignalProducer) -> None:
        """Add a producer to the registry."""
        self._producers.append(producer)

    def unregister(self, name: str) -> None:
        """Remove a producer by name."""
        self._producers = [p for p in self._producers if p.name() != name]

    def get_enabled(self) -> list[SignalProducer]:
        """Return only producers where is_enabled() is True."""
        return [p for p in self._producers if p.is_enabled()]

    def get_by_type(self, signal_type: str) -> list[SignalProducer]:
        """Filter by 'llm' or 'ml'."""
        return [p for p in self._producers if p.signal_type() == signal_type]

    async def run_all(self, data: MarketData) -> list[SignalOutput]:
        """Run all enabled producers in parallel.

        Uses asyncio.gather(). If a producer raises, log the full traceback
        and continue. Returns list of non-None results.
        """
        enabled = self.get_enabled()
        if not enabled:
            logger.warning("SignalRegistry: no enabled producers")
            return []

        async def _safe_run(producer: SignalProducer) -> SignalOutput | None:
            name = producer.name()
            try:
                logger.info(f"SignalRegistry: running '{name}'...")
                result = await producer.analyze(data)
                if result is None:
                    logger.warning(
                        f"SignalRegistry: '{name}' returned None "
                        f"(agent handled error internally or data insufficient)"
                    )
                else:
                    logger.info(
                        f"SignalRegistry: '{name}' -> {result.direction} "
                        f"(confidence={result.confidence:.2f})"
                    )
                return result
            except Exception:
                logger.exception(
                    f"SignalRegistry: '{name}' raised an unhandled exception"
                )
                return None

        results = await asyncio.gather(*[_safe_run(p) for p in enabled])
        successful = [r for r in results if r is not None]
        logger.info(
            f"SignalRegistry: {len(successful)}/{len(enabled)} producers returned signals"
        )
        return successful
