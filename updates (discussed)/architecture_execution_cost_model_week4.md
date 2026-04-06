# Architecture Update: Execution Cost Model

> Replaces the original "FeeCalculator" concept with a comprehensive cost-aware execution system.
> Affects: Architecture sections 7, 8, 6, 28, 16, 25, 14
> Status: APPROVED — implement in Week 4 or Week 5

---

## 1. The Problem

Without cost awareness, the engine reports a 2:1 R:R trade that actually nets 1.52:1 after execution costs. On HIP-3 assets with high deployer fees, small positions can have 20-50% of their risk eaten by costs. The bot would grind itself to zero while reporting profitable signals.

Execution cost is not just taker fees. It's the total tax on every trade:

```
Total Execution Cost = Taker Fee (Entry)
                     + Estimated Slippage (based on orderbook depth)
                     + Estimated Spread Impact (bid-ask)
                     + Estimated Funding Cost (based on expected hold time)
                     + Taker Fee (Exit)
```

---

## 2. ExecutionCostModel (replaces FeeCalculator)

A pure-code module in `engine/execution/cost_model.py` that computes the true cost of any trade before it happens. Exchange-agnostic — works with any adapter that provides the required data.

### 2.1 Interface

```python
class ExecutionCostModel(ABC):
    """Abstract cost model — each exchange provides its own implementation."""

    @abstractmethod
    async def refresh(self, adapter: ExchangeAdapter) -> None:
        """Refresh fee tier, asset metadata, orderbook snapshots.
        Called once per cycle or on a cache TTL (e.g., every 24h for metadata)."""
        ...

    @abstractmethod
    def get_taker_rate(self, symbol: str) -> float:
        """Effective taker fee rate for this symbol, including all exchange-specific
        modifiers (deployer fees, growth mode, staking discounts, referral discounts)."""
        ...

    @abstractmethod
    def get_maker_rate(self, symbol: str) -> float:
        """Effective maker fee rate (may be negative = rebate)."""
        ...

    @abstractmethod
    def estimate_slippage(self, symbol: str, size_usd: float,
                          side: str) -> float:
        """Estimate slippage as a percentage based on orderbook depth.
        Uses cached orderbook snapshot."""
        ...

    @abstractmethod
    def estimate_spread_cost(self, symbol: str) -> float:
        """Current bid-ask spread as a percentage of mid price."""
        ...

    @abstractmethod
    def estimate_funding_cost(self, symbol: str, direction: str,
                               hold_hours: float) -> float:
        """Estimated funding fee over expected hold duration.
        Uses current funding rate, assumes it stays constant."""
        ...

    def compute_total_cost(self, symbol: str, position_value: float,
                           direction: str, hold_hours: float) -> ExecutionCost:
        """Compute full execution cost breakdown."""
        taker = self.get_taker_rate(symbol)
        slippage = self.estimate_slippage(symbol, position_value, direction)
        spread = self.estimate_spread_cost(symbol)
        funding = self.estimate_funding_cost(symbol, direction, hold_hours)

        entry_cost = position_value * (taker + slippage + spread / 2)
        exit_cost = position_value * (taker + slippage + spread / 2)
        funding_cost = position_value * funding
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
            maker_rate=self.get_maker_rate(symbol),
        )

    def compute_fee_adjusted_rr(self, symbol: str, position_value: float,
                                 sl_distance_pct: float, tp_distance_pct: float,
                                 direction: str, hold_hours: float) -> float:
        """True R:R after all execution costs."""
        cost = self.compute_total_cost(symbol, position_value, direction, hold_hours)
        actual_risk = (position_value * sl_distance_pct) + cost.total_cost
        actual_reward = (position_value * tp_distance_pct) - cost.total_cost
        if actual_risk <= 0:
            return 0
        return actual_reward / actual_risk

    def compute_cost_aware_position_size(self, symbol: str,
                                          account_balance: float,
                                          risk_per_trade: float,
                                          sl_distance_pct: float,
                                          direction: str,
                                          hold_hours: float,
                                          max_position_pct: float,
                                          max_fee_drag_pct: float = 0.10
                                          ) -> PositionSizeResult:
        """Position size where (trade_loss + ALL_costs) = max_risk_amount.
        Solves the circular math: fees depend on size, size depends on fees."""
        max_risk = account_balance * risk_per_trade

        # Iterative solver (converges in 2-3 iterations)
        size = max_risk / sl_distance_pct  # initial guess (no fees)
        for _ in range(5):
            cost = self.compute_total_cost(symbol, size, direction, hold_hours)
            # True risk = price_risk + cost
            # We want: size * sl_distance_pct + cost.total_cost = max_risk
            # So: size = (max_risk - cost.total_cost) / sl_distance_pct
            new_size = (max_risk - cost.total_cost) / sl_distance_pct
            if new_size <= 0:
                return PositionSizeResult(
                    size=0, viable=False,
                    reason="Execution costs exceed maximum risk budget",
                    cost=cost, fee_drag_pct=100.0
                )
            if abs(new_size - size) < 0.01:
                break
            size = new_size

        # Cap at max position
        size = min(size, account_balance * max_position_pct)

        # Compute final cost at actual size
        final_cost = self.compute_total_cost(symbol, size, direction, hold_hours)
        fee_drag = final_cost.total_cost / (size * sl_distance_pct) * 100

        viable = fee_drag <= (max_fee_drag_pct * 100)

        return PositionSizeResult(
            size=size,
            viable=viable,
            reason=None if viable else f"Fee drag {fee_drag:.1f}% exceeds {max_fee_drag_pct*100:.0f}% limit",
            cost=final_cost,
            fee_drag_pct=fee_drag,
        )

    def is_trade_viable(self, symbol: str, position_value: float,
                         sl_distance_pct: float, tp_distance_pct: float,
                         direction: str, hold_hours: float,
                         min_rr: float = 1.0) -> tuple[bool, str]:
        """Final viability check: does fee-adjusted R:R meet minimum?"""
        adjusted_rr = self.compute_fee_adjusted_rr(
            symbol, position_value, sl_distance_pct, tp_distance_pct,
            direction, hold_hours)
        if adjusted_rr < min_rr:
            cost = self.compute_total_cost(symbol, position_value, direction, hold_hours)
            return False, (f"Net R:R {adjusted_rr:.2f} below minimum {min_rr}. "
                          f"Execution costs: {cost.total_as_pct*100:.2f}% of position "
                          f"(fees:{cost.entry_fee+cost.exit_fee:.2f}, "
                          f"slippage:{cost.entry_slippage+cost.exit_slippage:.2f}, "
                          f"funding:{cost.estimated_funding:.2f})")
        return True, f"Trade viable. Net R:R: {adjusted_rr:.2f}"
```

### 2.2 Data Types

```python
@dataclass
class ExecutionCost:
    entry_fee: float          # taker fee on entry
    exit_fee: float           # taker fee on exit
    entry_slippage: float     # estimated slippage on entry
    exit_slippage: float      # estimated slippage on exit
    spread_cost: float        # bid-ask spread cost
    estimated_funding: float  # estimated funding over hold period
    total_cost: float         # sum of all costs
    total_as_pct: float       # total as % of position value
    taker_rate: float         # effective taker rate used
    maker_rate: float         # effective maker rate (for reference)

@dataclass
class PositionSizeResult:
    size: float               # position size in quote currency
    viable: bool              # is this trade worth the costs?
    reason: str | None        # why not viable (if not)
    cost: ExecutionCost       # full cost breakdown at this size
    fee_drag_pct: float       # costs as % of risk
```

### 2.3 Hyperliquid Implementation

```python
class HyperliquidCostModel(ExecutionCostModel):
    """Hyperliquid-specific cost model.
    Fetches fee tiers, HIP-3 deployer scales, and orderbook data dynamically."""

    def __init__(self):
        self._fee_tier: int = 0
        self._staking_discount: float = 0.0
        self._referral_discount: float = 0.0
        self._asset_meta: dict = {}        # symbol -> {deployer_scale, growth_mode, ...}
        self._orderbook_cache: dict = {}   # symbol -> {bids, asks, timestamp}
        self._funding_cache: dict = {}     # symbol -> {rate, timestamp}
        self._last_meta_refresh: datetime | None = None

    async def refresh(self, adapter: ExchangeAdapter) -> None:
        """Refresh from Hyperliquid APIs."""
        now = datetime.utcnow()

        # 1. Asset metadata (refresh every 24 hours)
        #    Fetch from Hyperliquid meta endpoint — returns ALL assets
        #    with deployer_scale, growth_mode, etc.
        if (self._last_meta_refresh is None or
            (now - self._last_meta_refresh).total_seconds() > 86400):
            meta = await adapter.fetch_meta()  # New adapter method
            for asset in meta:
                self._asset_meta[asset['symbol']] = {
                    'deployer_scale': asset.get('deployer_fee_scale', 0),
                    'growth_mode': asset.get('growth_mode', False),
                    'is_hip3': asset.get('is_hip3', False),
                }
            self._last_meta_refresh = now

        # 2. Fee tier (refresh every 24 hours, same as meta)
        #    Fetch from userFees endpoint
        user_fees = await adapter.fetch_user_fees()
        self._fee_tier = user_fees.get('tier', 0)
        self._staking_discount = user_fees.get('staking_discount', 0)
        self._referral_discount = user_fees.get('referral_discount', 0)

    def get_taker_rate(self, symbol: str) -> float:
        """Apply Hyperliquid's fee formula exactly."""
        base_rate = PERP_TAKER_RATES[self._fee_tier]
        meta = self._asset_meta.get(symbol, {})

        # HIP-3 scaling (from HL docs)
        deployer_scale = meta.get('deployer_scale', 0)
        if deployer_scale > 0:
            if deployer_scale < 1:
                hip3_mult = deployer_scale + 1
            else:
                hip3_mult = deployer_scale * 2
            base_rate *= hip3_mult

        # Growth mode (90% reduction)
        if meta.get('growth_mode', False):
            base_rate *= 0.1

        # Staking + referral discounts
        base_rate *= (1 - self._staking_discount)
        base_rate *= (1 - self._referral_discount)

        return base_rate

    def estimate_slippage(self, symbol: str, size_usd: float,
                          side: str) -> float:
        """Estimate market impact from orderbook depth."""
        book = self._orderbook_cache.get(symbol)
        if not book:
            # Conservative default: 0.05% for crypto, 0.10% for HIP-3
            return 0.001 if not self._is_hip3(symbol) else 0.002

        # Walk the orderbook to find fill price for this size
        levels = book['asks'] if side == 'buy' else book['bids']
        remaining = size_usd
        weighted_price = 0
        mid_price = (book['best_bid'] + book['best_ask']) / 2

        for price, qty_usd in levels:
            fill = min(remaining, qty_usd)
            weighted_price += price * fill
            remaining -= fill
            if remaining <= 0:
                break

        if size_usd > 0 and weighted_price > 0:
            avg_fill = weighted_price / size_usd
            slippage = abs(avg_fill - mid_price) / mid_price
            return slippage

        # Size exceeds visible book — high slippage
        return 0.005

    def estimate_spread_cost(self, symbol: str) -> float:
        """Current bid-ask spread as percentage."""
        book = self._orderbook_cache.get(symbol)
        if not book:
            return 0.001 if not self._is_hip3(symbol) else 0.003
        spread = (book['best_ask'] - book['best_bid']) / book['best_bid']
        return spread

    def estimate_funding_cost(self, symbol: str, direction: str,
                               hold_hours: float) -> float:
        """Estimated cumulative funding over hold period."""
        funding = self._funding_cache.get(symbol, {})
        rate = funding.get('rate', 0)
        # Funding paid every 8 hours on Hyperliquid
        num_funding_periods = hold_hours / 8
        # Long pays positive funding, short pays negative
        if direction in ('LONG', 'long', 'buy'):
            return rate * num_funding_periods
        else:
            return -rate * num_funding_periods

    def _is_hip3(self, symbol: str) -> bool:
        meta = self._asset_meta.get(symbol, {})
        return meta.get('is_hip3', False)

    async def refresh_orderbook(self, adapter: ExchangeAdapter,
                                 symbol: str) -> None:
        """Refresh orderbook snapshot for slippage estimation.
        Called before each trade decision, cached briefly."""
        # Fetch top 10 levels of orderbook
        book = await adapter.fetch_orderbook(symbol, limit=10)
        self._orderbook_cache[symbol] = {
            'bids': book['bids'],
            'asks': book['asks'],
            'best_bid': book['bids'][0][0] if book['bids'] else 0,
            'best_ask': book['asks'][0][0] if book['asks'] else 0,
            'timestamp': datetime.utcnow(),
        }

    async def refresh_funding(self, adapter: ExchangeAdapter,
                               symbol: str) -> None:
        """Refresh current funding rate."""
        rate = await adapter.get_funding_rate(symbol)
        self._funding_cache[symbol] = {
            'rate': rate or 0,
            'timestamp': datetime.utcnow(),
        }
```

### 2.4 Fee Tier Constants

```python
# Hyperliquid perps base taker rates by tier (no staking, no referral)
PERP_TAKER_RATES = {
    0: 0.00045,   # 0.045%
    1: 0.00040,   # 0.040% (>$5M 14d vol)
    2: 0.00035,   # 0.035% (>$25M)
    3: 0.00030,   # 0.030% (>$100M)
    4: 0.00028,   # 0.028% (>$500M)
    5: 0.00026,   # 0.026% (>$2B)
    6: 0.00024,   # 0.024% (>$7B)
}

PERP_MAKER_RATES = {
    0: 0.00015,   # 0.015%
    1: 0.00012,
    2: 0.00008,
    3: 0.00004,
    4: 0.00000,   # Free at tier 4+
    5: 0.00000,
    6: 0.00000,
}
```

---

## 3. Dynamic Asset Metadata (No Hardcoding)

Asset fee profiles are NEVER hardcoded. They're fetched from the exchange API and cached.

### 3.1 ExchangeAdapter additions

Two new optional methods on the ExchangeAdapter interface:

```python
class ExchangeAdapter(ABC):
    # ... existing methods ...

    async def fetch_meta(self) -> list[dict]:
        """Fetch all asset metadata including fee parameters.
        Returns list of dicts with: symbol, deployer_fee_scale, growth_mode, is_hip3, etc.
        Default: returns empty list (exchange doesn't support meta)."""
        return []

    async def fetch_orderbook(self, symbol: str, limit: int = 10) -> dict:
        """Fetch orderbook snapshot. Returns {bids: [[price, size], ...], asks: ...}
        Default: returns empty book."""
        return {"bids": [], "asks": []}

    async def fetch_user_fees(self) -> dict:
        """Fetch user's current fee tier and discounts.
        Returns {tier: int, staking_discount: float, referral_discount: float}
        Default: returns tier 0, no discounts."""
        return {"tier": 0, "staking_discount": 0.0, "referral_discount": 0.0}
```

### 3.2 Cache Strategy

```
Cache Layer 3 (External APIs): TTL 24 hours
  └─ Asset metadata (deployer scales, growth mode) — from meta endpoint
  └─ User fee tier — from userFees endpoint

Cache Layer 2 (Flow Data): TTL 30 seconds
  └─ Orderbook snapshots — refreshed before each trade decision
  └─ Funding rates — already cached by FlowAgent
```

The meta endpoint returns ALL assets on the exchange. When Hyperliquid adds a new asset or changes a deployer fee, we pick it up automatically within 24 hours. Zero maintenance.

---

## 4. Integration Points

### 4.1 Safety Checks — New Check #6: Cost Viability

```python
# In safety_checks.py — add as check 6

# Check 6: Execution cost viability
if action.action in ("LONG", "SHORT"):
    viable, reason = cost_model.is_trade_viable(
        symbol=symbol,
        position_value=action.position_size,
        sl_distance_pct=abs(action.sl_price - entry_price) / entry_price,
        tp_distance_pct=abs(action.tp2_price - entry_price) / entry_price,
        direction=action.action,
        hold_hours=expected_hold_hours,  # from timeframe profile
        min_rr=profile.rr_min,
    )
    if not viable:
        return SafetyCheckResult(False, action.action, "SKIP", reason)
```

This is a mechanical gate. No LLM can override it. If the math doesn't work, the trade is blocked.

### 4.2 Position Sizing — Cost-Aware (Solves Circular Math)

Replace the current `compute_position_size()` in risk_profiles.py:

```python
# OLD (fee-blind):
def compute_position_size(balance, risk_pct, entry, sl, max_pct):
    risk_amount = balance * risk_pct
    sl_distance = abs(entry - sl) / entry
    size = risk_amount / sl_distance
    return min(size, balance * max_pct)

# NEW (cost-aware):
def compute_position_size(balance, risk_pct, entry, sl, max_pct,
                          cost_model, symbol, direction, hold_hours):
    sl_distance = abs(entry - sl) / entry
    result = cost_model.compute_cost_aware_position_size(
        symbol=symbol,
        account_balance=balance,
        risk_per_trade=risk_pct,
        sl_distance_pct=sl_distance,
        direction=direction,
        hold_hours=hold_hours,
        max_position_pct=max_pct,
    )
    if not result.viable:
        return 0  # Trade blocked by cost constraint
    return result.size
```

### 4.3 ConvictionAgent Grounding — Distilled Cost Line

One line in the grounding header, no implementation details:

```
COST_AWARE_RR: SL 1.5% / TP 3.0% → raw RR 2.00 → net RR 1.52 after execution costs. Fee drag: HIGH (24%).
```

Three severity levels:
- **LOW** (fee drag < 5%): most crypto majors with decent size
- **MEDIUM** (5-15%): smaller positions, some HIP-3 with growth mode
- **HIGH** (> 15%): small positions on HIP-3, illiquid assets

The LLM sees the mathematical reality without needing to understand why. It naturally lowers conviction for HIGH fee drag trades because the edge must be larger.

### 4.4 Risk Profiles — Return Cost Breakdown

```python
def compute_sl_tp(..., cost_model, symbol, position_value, direction):
    # ... existing SL/TP computation ...

    # Expected hold time based on timeframe
    hold_hours = EXPECTED_HOLD_HOURS[timeframe]  # 15m=2h, 1h=8h, 4h=24h, 1d=72h

    cost = cost_model.compute_total_cost(symbol, position_value, direction, hold_hours)
    adjusted_rr = cost_model.compute_fee_adjusted_rr(
        symbol, position_value, sl_distance_pct, tp_distance_pct,
        direction, hold_hours)

    return {
        "sl": sl, "tp1": tp1, "tp2": tp2,
        "rr_ratio": rr,                          # raw R:R
        "fee_adjusted_rr": adjusted_rr,           # net R:R
        "execution_cost": cost,                   # full breakdown
        "fee_drag_pct": cost.total_cost / (position_value * sl_distance_pct) * 100,
        "fee_drag_severity": "LOW" if drag < 5 else "MEDIUM" if drag < 15 else "HIGH",
    }
```

### 4.5 Tracking — Record Actual Costs

Every trade record includes:

```python
trade_record = {
    # ... existing fields ...
    "estimated_entry_fee": cost.entry_fee,
    "estimated_exit_fee": cost.exit_fee,
    "estimated_slippage": cost.entry_slippage + cost.exit_slippage,
    "estimated_spread_cost": cost.spread_cost,
    "estimated_funding_cost": cost.estimated_funding,
    "estimated_total_cost": cost.total_cost,
    "fee_adjusted_rr": adjusted_rr,
    "fee_drag_pct": fee_drag,
    "taker_rate_used": cost.taker_rate,
}
```

The Overnight Quant can analyze: which assets are only profitable above certain position sizes, which assets have hidden cost drag, and whether slippage estimates match actual fills.

---

## 5. Dual Execution Mode

### 5.1 Sentinel Triggers → Pure Taker (Market Orders)

When Sentinel fires SetupDetected, momentum is happening. Speed matters more than fee savings. Entry uses market orders (taker).

### 5.2 Post-Trade SL/TP → Maker Limit Orders

After entry, the Sentinel Position Manager replaces SL/TP trigger orders with maker limit orders sitting on the book. This earns maker rebates instead of paying taker fees on exit. At Tier 0: saves 0.06% per exit (0.045% taker → -0.015% maker rebate).

Risk: if price gaps through the limit, the order won't fill. Mitigation: keep a backup trigger order slightly behind the limit. If the limit misses, the trigger catches it.

### 5.3 B2B API → Client Choice

The B2B API response includes both taker and maker cost estimates. Institutional clients choose their own execution routing.

```json
{
    "signal": { ... },
    "execution_costs": {
        "taker_entry": 0.00045,
        "maker_entry": -0.00015,
        "estimated_slippage": 0.00012,
        "recommended_mode": "taker"  // or "maker" for non-urgent
    }
}
```

---

## 6. Exchange-Agnostic Design

The ExecutionCostModel is abstract. Each exchange provides its own implementation:

```
engine/execution/
├── cost_model.py            # Abstract ExecutionCostModel + data types
└── cost_models/
    ├── __init__.py
    ├── hyperliquid.py       # HyperliquidCostModel (HIP-3, deployer fees, growth mode)
    ├── binance.py           # BinanceCostModel (BNB discount, VIP tiers)
    ├── ibkr.py              # IBKRCostModel (commission-based, no funding)
    └── generic.py           # GenericCostModel (conservative defaults)
```

When a new exchange adapter is added, it optionally provides a cost model. If no cost model exists, GenericCostModel uses conservative defaults (0.1% taker, 0.05% slippage, no funding).

The ExchangeAdapter declares which cost model to use:

```python
class ExchangeAdapter(ABC):
    def get_cost_model(self) -> ExecutionCostModel:
        """Return the cost model for this exchange.
        Default: GenericCostModel with conservative estimates."""
        return GenericCostModel()
```

---

## 7. Expected Hold Hours by Timeframe

The funding cost estimate needs expected hold duration:

| Timeframe | Expected Hold | Funding Periods (8h) | Rationale |
|-----------|--------------|---------------------|-----------|
| 15m | 2 hours | 0.25 | Quick scalp, rarely held overnight |
| 30m | 4 hours | 0.5 | Short-term swing |
| 1h | 8 hours | 1.0 | One funding period typical |
| 4h | 24 hours | 3.0 | Multi-day hold |
| 1d | 72 hours | 9.0 | Position trade, significant funding exposure |

For 4h and 1d bots, funding cost can be substantial. A 0.03% funding rate over 72 hours = 0.27% position cost just from funding. The cost model captures this.

---

## 8. Marketing Angle

> **"Zero-Leakage Execution"**
>
> QuantAgent calculates exchange fees, bid-ask spread, market impact, and funding costs before every trade. If the math doesn't work, the trade is physically blocked. Your edge is protected from execution drag.
>
> No other retail AI trading bot does this.

This goes on the landing page, in the Consumer Dashboard (show users their "costs saved" metric), and in the B2B API documentation.

---

## 9. Summary of Architecture Changes

| Component | Change | Section |
|-----------|--------|---------|
| NEW: `engine/execution/cost_model.py` | Abstract ExecutionCostModel + data types | Engine |
| NEW: `engine/execution/cost_models/hyperliquid.py` | HyperliquidCostModel with dynamic meta | Engine |
| NEW: `engine/execution/cost_models/generic.py` | Conservative fallback for unknown exchanges | Engine |
| UPDATE: `exchanges/base.py` | Add fetch_meta(), fetch_orderbook(), fetch_user_fees() | Adapters |
| UPDATE: `exchanges/hyperliquid.py` | Implement 3 new methods | Adapters |
| UPDATE: `engine/execution/risk_profiles.py` | Cost-aware position sizing, fee-adjusted R:R | Execution |
| UPDATE: `engine/execution/safety_checks.py` | Add Check #6: cost viability gate | Execution |
| UPDATE: `engine/data/charts.py` (grounding header) | Add distilled COST_AWARE_RR line | Data |
| UPDATE: Tracking | Record estimated costs per trade | Tracking |
| UPDATE: Config | Add expected_hold_hours per timeframe, max_fee_drag_pct | Config |
| UPDATE: Pipeline | Pass cost_model through to risk_profiles and safety checks | Pipeline |

---

## 10. Claude Code Prompt (Implementation)

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to MAX for this task.

Implement the ExecutionCostModel — comprehensive cost-aware execution that prevents
the engine from trading when execution costs make the trade unprofitable.

This is a CRITICAL SAFETY feature. Without it, bots on HIP-3 assets (GOLD, OIL, stocks)
can grind themselves to zero just on fees.

FILE 1: engine/execution/cost_model.py
- ExecutionCost dataclass (entry_fee, exit_fee, slippage, spread, funding, total, pct)
- PositionSizeResult dataclass (size, viable, reason, cost, fee_drag_pct)
- Abstract ExecutionCostModel with:
  - refresh(), get_taker_rate(), get_maker_rate()
  - estimate_slippage(), estimate_spread_cost(), estimate_funding_cost()
  - compute_total_cost() -> ExecutionCost
  - compute_fee_adjusted_rr() -> float
  - compute_cost_aware_position_size() -> PositionSizeResult (iterative solver for circular math)
  - is_trade_viable() -> (bool, str)

FILE 2: engine/execution/cost_models/hyperliquid.py
- HyperliquidCostModel implementing ExecutionCostModel
- Fetches fee tier, staking/referral discounts from adapter
- Fetches asset metadata from HL meta endpoint (deployer_scale, growth_mode, is_hip3)
  — cached 24 hours, NOT hardcoded
- Fetches orderbook snapshots for slippage estimation
- Uses current funding rate for hold-time funding cost
- Applies HL fee formula exactly (from their docs)
- PERP_TAKER_RATES and PERP_MAKER_RATES dicts (tiers 0-6)

FILE 3: engine/execution/cost_models/generic.py
- GenericCostModel with conservative defaults (0.1% taker, 0.05% slippage)
- Used when no exchange-specific model exists

FILE 4: Update exchanges/base.py
- Add optional methods: fetch_meta(), fetch_orderbook(symbol, limit), fetch_user_fees()
- Default implementations return empty/default values

FILE 5: Update exchanges/hyperliquid.py
- Implement fetch_meta() using HL info endpoint
- Implement fetch_orderbook() using HL L2 book endpoint
- Implement fetch_user_fees() using HL userFees endpoint

FILE 6: Update engine/execution/risk_profiles.py
- compute_position_size() now takes cost_model parameter
- Uses compute_cost_aware_position_size() instead of naive division
- compute_sl_tp() returns fee_adjusted_rr and execution_cost alongside raw values

FILE 7: Update engine/execution/safety_checks.py
- Add Check #6: cost viability — calls cost_model.is_trade_viable()
- Blocks trade if fee-adjusted R:R < minimum

FILE 8: Update engine/data/charts.py (generate_grounding_header)
- Add COST_AWARE_RR line to grounding header:
  "COST_AWARE_RR: SL X% / TP Y% -> raw RR Z -> net RR W. Fee drag: LOW/MEDIUM/HIGH."

FILE 9: Add to engine/config.py
EXPECTED_HOLD_HOURS = {"15m": 2, "30m": 4, "1h": 8, "4h": 24, "1d": 72}
MAX_FEE_DRAG_PCT = 0.10  # 10% — trades with higher drag are blocked

Write tests in tests/unit/test_cost_model.py:
- Test HyperliquidCostModel taker rate calculation (standard crypto vs HIP-3)
- Test HIP-3 deployer fee scaling (scale < 1 vs scale >= 1)
- Test growth mode 90% reduction
- Test staking discount applied correctly
- Test slippage estimation from mock orderbook
- Test funding cost over different hold periods
- Test compute_total_cost sums all components
- Test fee_adjusted_rr is lower than raw rr
- Test cost_aware_position_size converges (iterative solver)
- Test position_size = 0 when costs exceed risk budget
- Test is_trade_viable blocks high-cost trades
- Test GenericCostModel uses conservative defaults
- Test with real-world scenarios:
  - BTC $1000 position, 1h: LOW fee drag
  - GOLD $200 position, 1h: HIGH fee drag (should be blocked)
  - BTC $50 position, 15m: MEDIUM fee drag

Update PROJECT_CONTEXT.md sections 2, 14. Update CHANGELOG.md.
```
