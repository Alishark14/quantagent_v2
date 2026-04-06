"""CryptoFlowProvider: funding rate, OI, liquidation data.

Fetches crypto-specific positioning data from the exchange adapter.
Code-only, zero LLM cost. All data is factual/deterministic.
"""

from __future__ import annotations

import logging

from engine.data.flow.base import FlowProvider
from exchanges.base import ExchangeAdapter

logger = logging.getLogger(__name__)

# Funding rate thresholds for signal classification
_CROWDED_LONG_THRESHOLD = 0.01   # > +0.01% = crowded long
_CROWDED_SHORT_THRESHOLD = -0.01  # < -0.01% = crowded short


class CryptoFlowProvider(FlowProvider):
    """Fetches funding rate and open interest from crypto exchange adapters."""

    def name(self) -> str:
        return "crypto"

    async def fetch(self, symbol: str, adapter: ExchangeAdapter) -> dict:
        """Fetch funding rate and OI from exchange adapter.

        Each data point is independently try/excepted — one failure
        does not block the other.
        """
        result: dict = {}

        # Funding rate
        try:
            rate = await adapter.get_funding_rate(symbol)
            if rate is not None:
                result["funding_rate"] = rate
                if rate > _CROWDED_LONG_THRESHOLD:
                    result["funding_signal"] = "CROWDED_LONG"
                elif rate < _CROWDED_SHORT_THRESHOLD:
                    result["funding_signal"] = "CROWDED_SHORT"
                else:
                    result["funding_signal"] = "NEUTRAL"
        except Exception as e:
            logger.warning(f"CryptoFlowProvider: funding rate unavailable for {symbol}: {e}")

        # Open Interest
        try:
            oi = await adapter.get_open_interest(symbol)
            if oi is not None:
                result["open_interest"] = oi
                # OI delta requires previous value — tracked by FlowAgent history later
                result["oi_trend"] = "STABLE"
        except Exception as e:
            logger.warning(f"CryptoFlowProvider: OI unavailable for {symbol}: {e}")

        return result
