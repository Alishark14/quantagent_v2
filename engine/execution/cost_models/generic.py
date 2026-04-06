"""GenericCostModel: conservative defaults for unknown exchanges.

Used when no exchange-specific cost model exists. Overestimates
costs to err on the side of safety.
"""

from __future__ import annotations

from engine.execution.cost_model import ExecutionCostModel


class GenericCostModel(ExecutionCostModel):
    """Conservative cost model with static defaults."""

    def __init__(
        self,
        taker_rate: float = 0.001,     # 0.10%
        maker_rate: float = 0.0005,    # 0.05%
        slippage: float = 0.0005,      # 0.05%
        spread: float = 0.001,         # 0.10%
    ) -> None:
        self._taker = taker_rate
        self._maker = maker_rate
        self._slippage = slippage
        self._spread = spread

    async def refresh(self, adapter) -> None:
        pass  # static model, nothing to refresh

    def get_taker_rate(self, symbol: str) -> float:
        return self._taker

    def get_maker_rate(self, symbol: str) -> float:
        return self._maker

    def estimate_slippage(self, symbol: str, size_usd: float, side: str) -> float:
        return self._slippage

    def estimate_spread_cost(self, symbol: str) -> float:
        return self._spread

    def estimate_funding_cost(
        self, symbol: str, direction: str, hold_hours: float
    ) -> float:
        return 0.0  # unknown exchange, assume no funding
