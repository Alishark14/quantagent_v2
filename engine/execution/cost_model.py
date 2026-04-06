"""ExecutionCostModel: comprehensive cost-aware execution.

Computes true trade cost (fees + slippage + spread + funding) BEFORE
any order is placed. Prevents the engine from trading when execution
costs make the trade unprofitable.

CRITICAL SAFETY: Without cost awareness, bots on HIP-3 assets (GOLD,
OIL, stocks) with small positions can have 20-50% of risk eaten by costs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExecutionCost:
    """Full breakdown of execution costs for a trade."""

    entry_fee: float
    exit_fee: float
    entry_slippage: float
    exit_slippage: float
    spread_cost: float
    estimated_funding: float
    total_cost: float
    total_as_pct: float       # total as % of position value
    taker_rate: float
    maker_rate: float

    def to_dict(self) -> dict:
        return {
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee,
            "entry_slippage": self.entry_slippage,
            "exit_slippage": self.exit_slippage,
            "spread_cost": self.spread_cost,
            "estimated_funding": self.estimated_funding,
            "total_cost": self.total_cost,
            "total_as_pct": self.total_as_pct,
            "taker_rate": self.taker_rate,
            "maker_rate": self.maker_rate,
        }


@dataclass
class PositionSizeResult:
    """Result of cost-aware position sizing."""

    size: float               # position size in USD
    viable: bool              # is this trade worth the costs?
    reason: str | None        # why not viable (if not)
    cost: ExecutionCost       # full cost breakdown at this size
    fee_drag_pct: float       # costs as % of risk

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "viable": self.viable,
            "reason": self.reason,
            "cost": self.cost.to_dict(),
            "fee_drag_pct": self.fee_drag_pct,
        }


class ExecutionCostModel(ABC):
    """Abstract cost model. Each exchange provides its own implementation."""

    @abstractmethod
    async def refresh(self, adapter) -> None:
        """Refresh fee tier, asset metadata, orderbook snapshots."""
        ...

    @abstractmethod
    def get_taker_rate(self, symbol: str) -> float:
        """Effective taker fee rate for this symbol."""
        ...

    @abstractmethod
    def get_maker_rate(self, symbol: str) -> float:
        """Effective maker fee rate (may be negative = rebate)."""
        ...

    @abstractmethod
    def estimate_slippage(self, symbol: str, size_usd: float, side: str) -> float:
        """Estimate slippage as a fraction based on orderbook depth."""
        ...

    @abstractmethod
    def estimate_spread_cost(self, symbol: str) -> float:
        """Current bid-ask spread as a fraction of mid price."""
        ...

    @abstractmethod
    def estimate_funding_cost(
        self, symbol: str, direction: str, hold_hours: float
    ) -> float:
        """Estimated funding fee as a fraction over expected hold duration."""
        ...

    def compute_total_cost(
        self,
        symbol: str,
        position_value: float,
        direction: str,
        hold_hours: float,
    ) -> ExecutionCost:
        """Compute full execution cost breakdown."""
        taker = self.get_taker_rate(symbol)
        maker = self.get_maker_rate(symbol)
        slippage = self.estimate_slippage(symbol, position_value, direction)
        spread = self.estimate_spread_cost(symbol)
        funding = self.estimate_funding_cost(symbol, direction, hold_hours)

        entry_cost = position_value * (taker + slippage + spread / 2)
        exit_cost = position_value * (taker + slippage + spread / 2)
        funding_cost = position_value * abs(funding)
        total = entry_cost + exit_cost + funding_cost

        return ExecutionCost(
            entry_fee=position_value * taker,
            exit_fee=position_value * taker,
            entry_slippage=position_value * slippage,
            exit_slippage=position_value * slippage,
            spread_cost=position_value * spread,
            estimated_funding=funding_cost,
            total_cost=total,
            total_as_pct=total / position_value if position_value > 0 else 0,
            taker_rate=taker,
            maker_rate=maker,
        )

    def compute_fee_adjusted_rr(
        self,
        symbol: str,
        position_value: float,
        sl_distance_pct: float,
        tp_distance_pct: float,
        direction: str,
        hold_hours: float,
    ) -> float:
        """True R:R after all execution costs."""
        cost = self.compute_total_cost(symbol, position_value, direction, hold_hours)
        actual_risk = (position_value * sl_distance_pct) + cost.total_cost
        actual_reward = (position_value * tp_distance_pct) - cost.total_cost
        if actual_risk <= 0:
            return 0.0
        return actual_reward / actual_risk

    def compute_cost_aware_position_size(
        self,
        symbol: str,
        account_balance: float,
        risk_per_trade: float,
        sl_distance_pct: float,
        direction: str,
        hold_hours: float,
        max_position_pct: float,
        max_fee_drag_pct: float = 0.10,
    ) -> PositionSizeResult:
        """Position size where (trade_loss + ALL_costs) = max_risk_amount.

        Solves circular math: fees depend on size, size depends on fees.
        Converges in 2-3 iterations.
        """
        if sl_distance_pct <= 0 or account_balance <= 0:
            zero_cost = self.compute_total_cost(symbol, 0, direction, hold_hours)
            return PositionSizeResult(
                size=0, viable=False,
                reason="Invalid SL distance or balance",
                cost=zero_cost, fee_drag_pct=100.0,
            )

        max_risk = account_balance * risk_per_trade
        size = max_risk / sl_distance_pct  # initial guess (no fees)

        for _ in range(5):
            cost = self.compute_total_cost(symbol, size, direction, hold_hours)
            new_size = (max_risk - cost.total_cost) / sl_distance_pct
            if new_size <= 0:
                return PositionSizeResult(
                    size=0, viable=False,
                    reason="Execution costs exceed maximum risk budget",
                    cost=cost, fee_drag_pct=100.0,
                )
            if abs(new_size - size) < 0.01:
                break
            size = new_size

        size = min(size, account_balance * max_position_pct)
        final_cost = self.compute_total_cost(symbol, size, direction, hold_hours)
        risk_amount = size * sl_distance_pct
        fee_drag = (final_cost.total_cost / risk_amount * 100) if risk_amount > 0 else 100.0
        viable = fee_drag <= (max_fee_drag_pct * 100)

        return PositionSizeResult(
            size=round(size, 2),
            viable=viable,
            reason=None if viable else f"Fee drag {fee_drag:.1f}% exceeds {max_fee_drag_pct*100:.0f}% limit",
            cost=final_cost,
            fee_drag_pct=round(fee_drag, 2),
        )

    def is_trade_viable(
        self,
        symbol: str,
        position_value: float,
        sl_distance_pct: float,
        tp_distance_pct: float,
        direction: str,
        hold_hours: float,
        min_rr: float = 1.0,
    ) -> tuple[bool, str]:
        """Final viability check: does fee-adjusted R:R meet minimum?"""
        adjusted_rr = self.compute_fee_adjusted_rr(
            symbol, position_value, sl_distance_pct, tp_distance_pct,
            direction, hold_hours,
        )
        if adjusted_rr < min_rr:
            cost = self.compute_total_cost(symbol, position_value, direction, hold_hours)
            return False, (
                f"Net R:R {adjusted_rr:.2f} below minimum {min_rr}. "
                f"Costs: {cost.total_as_pct*100:.2f}% of position "
                f"(fees:{cost.entry_fee+cost.exit_fee:.2f}, "
                f"slip:{cost.entry_slippage+cost.exit_slippage:.2f}, "
                f"funding:{cost.estimated_funding:.2f})"
            )
        return True, f"Trade viable. Net R:R: {adjusted_rr:.2f}"
