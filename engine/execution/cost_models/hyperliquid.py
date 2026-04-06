"""HyperliquidCostModel: Hyperliquid-specific execution cost model.

Fetches fee tiers, HIP-3 deployer scales, orderbook data, and funding
rates dynamically. Asset metadata is NEVER hardcoded — fetched from
the exchange API and cached for 24 hours.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from engine.execution.cost_model import ExecutionCostModel

logger = logging.getLogger(__name__)

# Hyperliquid perps base taker rates by tier
PERP_TAKER_RATES: dict[int, float] = {
    0: 0.00045,   # 0.045%
    1: 0.00040,   # 0.040% (>$5M 14d vol)
    2: 0.00035,   # 0.035% (>$25M)
    3: 0.00030,   # 0.030% (>$100M)
    4: 0.00028,   # 0.028% (>$500M)
    5: 0.00026,   # 0.026% (>$2B)
    6: 0.00024,   # 0.024% (>$7B)
}

PERP_MAKER_RATES: dict[int, float] = {
    0: 0.00015,   # 0.015%
    1: 0.00012,
    2: 0.00008,
    3: 0.00004,
    4: 0.00000,   # Free at tier 4+
    5: 0.00000,
    6: 0.00000,
}


class HyperliquidCostModel(ExecutionCostModel):
    """Hyperliquid-specific cost model with HIP-3 deployer fees.

    Fetches fee tier, staking/referral discounts, asset metadata,
    orderbook snapshots, and funding rates dynamically.
    """

    def __init__(self) -> None:
        self._fee_tier: int = 0
        self._staking_discount: float = 0.0
        self._referral_discount: float = 0.0
        self._asset_meta: dict[str, dict] = {}
        self._orderbook_cache: dict[str, dict] = {}
        self._funding_cache: dict[str, dict] = {}
        self._last_meta_refresh: datetime | None = None

    async def refresh(self, adapter) -> None:
        """Refresh fee tier and asset metadata from Hyperliquid APIs."""
        now = datetime.now(timezone.utc)

        # Asset metadata (refresh every 24 hours)
        if (
            self._last_meta_refresh is None
            or (now - self._last_meta_refresh).total_seconds() > 86400
        ):
            try:
                meta = await adapter.fetch_meta()
                for asset in meta:
                    sym = asset.get("symbol", "")
                    self._asset_meta[sym] = {
                        "deployer_scale": asset.get("deployer_fee_scale", 0),
                        "growth_mode": asset.get("growth_mode", False),
                        "is_hip3": asset.get("is_hip3", False),
                    }
                self._last_meta_refresh = now
                logger.info(f"HyperliquidCostModel: refreshed metadata for {len(meta)} assets")
            except Exception:
                logger.warning("HyperliquidCostModel: meta refresh failed", exc_info=True)

        # Fee tier
        try:
            user_fees = await adapter.fetch_user_fees()
            self._fee_tier = user_fees.get("tier", 0)
            self._staking_discount = user_fees.get("staking_discount", 0.0)
            self._referral_discount = user_fees.get("referral_discount", 0.0)
        except Exception:
            logger.warning("HyperliquidCostModel: fee tier refresh failed", exc_info=True)

    def get_taker_rate(self, symbol: str) -> float:
        """Apply Hyperliquid's fee formula: base rate * HIP-3 scaling * discounts."""
        base_rate = PERP_TAKER_RATES.get(self._fee_tier, PERP_TAKER_RATES[0])
        meta = self._asset_meta.get(symbol, {})

        # HIP-3 deployer fee scaling
        deployer_scale = meta.get("deployer_scale", 0)
        if deployer_scale > 0:
            if deployer_scale < 1:
                hip3_mult = deployer_scale + 1
            else:
                hip3_mult = deployer_scale * 2
            base_rate *= hip3_mult

        # Growth mode (90% reduction)
        if meta.get("growth_mode", False):
            base_rate *= 0.1

        # Staking + referral discounts
        base_rate *= (1 - self._staking_discount)
        base_rate *= (1 - self._referral_discount)

        return base_rate

    def get_maker_rate(self, symbol: str) -> float:
        """Maker rate — may be negative (rebate) at high tiers."""
        return PERP_MAKER_RATES.get(self._fee_tier, PERP_MAKER_RATES[0])

    def estimate_slippage(self, symbol: str, size_usd: float, side: str) -> float:
        """Estimate market impact from cached orderbook depth."""
        book = self._orderbook_cache.get(symbol)
        if not book or not book.get("best_bid"):
            # Conservative default: 0.05% for crypto, 0.10% for HIP-3
            return 0.001 if not self._is_hip3(symbol) else 0.002

        mid_price = (book["best_bid"] + book["best_ask"]) / 2
        if mid_price <= 0:
            return 0.001

        levels = book["asks"] if side in ("buy", "LONG", "long") else book["bids"]
        remaining = size_usd
        weighted_price = 0.0

        for price, qty_usd in levels:
            fill = min(remaining, qty_usd)
            weighted_price += price * fill
            remaining -= fill
            if remaining <= 0:
                break

        if size_usd > 0 and weighted_price > 0:
            avg_fill = weighted_price / (size_usd - remaining) if (size_usd - remaining) > 0 else mid_price
            slippage = abs(avg_fill - mid_price) / mid_price
            return slippage

        # Size exceeds visible book
        return 0.005

    def estimate_spread_cost(self, symbol: str) -> float:
        """Current bid-ask spread as fraction of mid price."""
        book = self._orderbook_cache.get(symbol)
        if not book or not book.get("best_bid"):
            return 0.001 if not self._is_hip3(symbol) else 0.003
        if book["best_bid"] <= 0:
            return 0.001
        return (book["best_ask"] - book["best_bid"]) / book["best_bid"]

    def estimate_funding_cost(
        self, symbol: str, direction: str, hold_hours: float
    ) -> float:
        """Estimated cumulative funding over hold period.

        Funding paid every 8 hours on Hyperliquid.
        Long pays positive funding, short pays negative.
        """
        funding = self._funding_cache.get(symbol, {})
        rate = funding.get("rate", 0)
        num_periods = hold_hours / 8
        if direction in ("LONG", "long", "buy"):
            return rate * num_periods
        else:
            return -rate * num_periods

    def _is_hip3(self, symbol: str) -> bool:
        meta = self._asset_meta.get(symbol, {})
        return meta.get("is_hip3", False)

    async def refresh_orderbook(self, adapter, symbol: str) -> None:
        """Refresh orderbook snapshot for slippage estimation."""
        try:
            book = await adapter.fetch_orderbook(symbol, limit=10)
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            self._orderbook_cache[symbol] = {
                "bids": bids,
                "asks": asks,
                "best_bid": bids[0][0] if bids else 0,
                "best_ask": asks[0][0] if asks else 0,
                "timestamp": datetime.now(timezone.utc),
            }
        except Exception:
            logger.warning(f"HyperliquidCostModel: orderbook refresh failed for {symbol}", exc_info=True)

    async def refresh_funding(self, adapter, symbol: str) -> None:
        """Refresh current funding rate."""
        try:
            rate = await adapter.get_funding_rate(symbol)
            self._funding_cache[symbol] = {
                "rate": rate or 0,
                "timestamp": datetime.now(timezone.utc),
            }
        except Exception:
            logger.warning(f"HyperliquidCostModel: funding refresh failed for {symbol}", exc_info=True)

    def set_meta(self, symbol: str, deployer_scale: float = 0, growth_mode: bool = False, is_hip3: bool = False) -> None:
        """Manually set asset metadata (for testing or overrides)."""
        self._asset_meta[symbol] = {
            "deployer_scale": deployer_scale,
            "growth_mode": growth_mode,
            "is_hip3": is_hip3,
        }

    def set_orderbook(self, symbol: str, bids: list, asks: list) -> None:
        """Manually set orderbook (for testing)."""
        self._orderbook_cache[symbol] = {
            "bids": bids,
            "asks": asks,
            "best_bid": bids[0][0] if bids else 0,
            "best_ask": asks[0][0] if asks else 0,
            "timestamp": datetime.now(timezone.utc),
        }

    def set_funding(self, symbol: str, rate: float) -> None:
        """Manually set funding rate (for testing)."""
        self._funding_cache[symbol] = {
            "rate": rate,
            "timestamp": datetime.now(timezone.utc),
        }
