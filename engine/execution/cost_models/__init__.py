"""Exchange-specific cost model implementations.

The :func:`get_cost_model` factory selects the right implementation by
exchange name. Falls back to :class:`GenericCostModel` for unknown
exchanges so callers (especially the backtest sim) always get a
non-None cost model — fees are only zero when explicitly disabled.
"""

from engine.execution.cost_model import ExecutionCostModel
from engine.execution.cost_models.generic import GenericCostModel
from engine.execution.cost_models.hyperliquid import HyperliquidCostModel


def get_cost_model(exchange: str) -> ExecutionCostModel:
    """Return a cost model instance for the named exchange.

    Unknown exchange names get :class:`GenericCostModel` (conservative
    defaults: 0.10% taker, 0.05% slippage, 0.10% spread). Hyperliquid
    perps get the full HIP-3-aware model at fee tier 0 with no
    metadata loaded — that's the right default for backtests of the
    main perp markets (~4.5 bps taker), and async ``refresh()`` can be
    called separately if a live tier / metadata snapshot is needed.
    """
    name = (exchange or "").lower().strip()
    if name == "hyperliquid":
        return HyperliquidCostModel()
    return GenericCostModel()


__all__ = ["GenericCostModel", "HyperliquidCostModel", "get_cost_model"]
