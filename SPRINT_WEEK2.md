# SPRINT.md — Week 2 (Phase 1a continued: First Live Analysis)

> Theme: Connect to real market data and run the first LLM analysis cycle.
> Start: April 14, 2026
> Target: April 18, 2026
> Tasks: 8 (ordered by dependency)
> Foundation: 196 tests passing, Event Bus, Types, Config, LLM Provider, Indicators, SignalProducer all working.

---

## Task 1: Exchange Adapter Base + Factory [S]

**Status:** [ ] Not Started
**Depends on:** Types (Week 1)
**Estimated time:** 20 minutes

**Acceptance criteria:**
- [ ] Abstract ExchangeAdapter with all methods from CLAUDE.md interface
- [ ] AdapterCapabilities dataclass used by each adapter to declare features
- [ ] ExchangeFactory with singleton caching
- [ ] Unit tests for factory registration and caching
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Implement the exchange adapter abstraction layer.

FILE 1: exchanges/base.py

Import AdapterCapabilities from engine/types.py (already defined in Week 1).

class ExchangeAdapter(ABC):
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities: ...
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> list[dict]: ...
    @abstractmethod
    async def get_ticker(self, symbol: str) -> dict: ...
    @abstractmethod
    async def get_balance(self) -> float: ...
    @abstractmethod
    async def get_positions(self, symbol: str | None = None) -> list[Position]: ...
    @abstractmethod
    async def place_market_order(self, symbol: str, side: str, size: float) -> OrderResult: ...
    @abstractmethod
    async def place_limit_order(self, symbol: str, side: str, size: float, price: float) -> OrderResult: ...
    @abstractmethod
    async def place_sl_order(self, symbol: str, side: str, size: float, trigger_price: float) -> OrderResult: ...
    @abstractmethod
    async def place_tp_order(self, symbol: str, side: str, size: float, trigger_price: float) -> OrderResult: ...
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool: ...
    @abstractmethod
    async def cancel_all_orders(self, symbol: str) -> int: ...
    @abstractmethod
    async def close_position(self, symbol: str) -> OrderResult: ...
    @abstractmethod
    async def modify_sl(self, symbol: str, new_price: float) -> OrderResult: ...
    @abstractmethod
    async def modify_tp(self, symbol: str, new_price: float) -> OrderResult: ...

    # Optional flow data (return None if not supported)
    async def get_funding_rate(self, symbol: str) -> float | None:
        return None
    async def get_open_interest(self, symbol: str) -> float | None:
        return None

FILE 2: exchanges/factory.py

class ExchangeFactory:
    _instances: dict[str, ExchangeAdapter] = {}
    _registry: dict[str, type[ExchangeAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_class: type[ExchangeAdapter]) -> None: ...
    @classmethod
    def get_adapter(cls, name: str, **kwargs) -> ExchangeAdapter: ...
    @classmethod
    def reset(cls) -> None: ...

Write tests in tests/unit/test_exchange_base.py:
- Test factory register + get_adapter returns instance
- Test singleton caching (same instance returned twice)
- Test unknown exchange raises ValueError
- Test reset clears cache
- Create a MockAdapter for tests

Update PROJECT_CONTEXT.md sections 2, 6, 14.
```

---

## Task 2: Hyperliquid Adapter [L]

**Status:** [ ] Not Started
**Depends on:** Task 1
**Estimated time:** 90 minutes

**Acceptance criteria:**
- [ ] Full HyperliquidAdapter implementing ExchangeAdapter
- [ ] Capabilities declared (native_sl_tp=True, supports_short=True, 24/7, perpetual+spot)
- [ ] CCXT used internally — engine never sees CCXT
- [ ] Symbol conversion: BTC-USDC internal to CCXT format
- [ ] SL/TP placement using native trigger orders
- [ ] HIP-3 symbol support (commodities, indices, stocks, forex)
- [ ] Funding rate and OI fetching
- [ ] Unit tests with mock CCXT responses
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Implement the Hyperliquid exchange adapter. PRIMARY exchange.

Reference: v1 codebase at ../quantagent-v1/ has a working adapter
in exchanges/hyperliquid_adapter.py. Port the LOGIC but rewrite
to fit the v2 ExchangeAdapter interface.

FILE: exchanges/hyperliquid.py

class HyperliquidAdapter(ExchangeAdapter):
    def __init__(self, wallet_address: str = None, private_key: str = None, testnet: bool = False):
        self._exchange = ccxt.hyperliquid({...})
        if testnet: self._exchange.setSandboxMode(True)

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            native_sl_tp=True, supports_short=True, market_hours=None,
            asset_types=["perpetual", "spot"], margin_type="cross",
            has_funding_rate=True, has_oi_data=True, max_leverage=50.0,
            order_types=["market", "limit", "stopMarket", "takeProfit"],
            supports_partial_close=True,
        )

    Symbol conversion helpers:
    - _to_ccxt_symbol: BTC-USDC -> BTC/USDC:USDC, GOLD-USDC -> XYZ-GOLD/USDC:USDC
    - _from_ccxt_symbol: reverse
    - HIP3_SYMBOLS set for prefix detection (GOLD, SILVER, OIL, SP500, TSLA, etc.)

    All methods: wrap in try/except, return OrderResult with success=False on error.
    Never raise raw CCXT exceptions.

    Register: ExchangeFactory.register("hyperliquid", HyperliquidAdapter)

Write tests in tests/adapters/test_hyperliquid.py:
- Mock ccxt.hyperliquid entirely
- Test symbol conversion both directions (standard + HIP-3)
- Test fetch_ohlcv, place_market_order, SL/TP orders
- Test error handling (mock failure -> OrderResult success=False)

Update PROJECT_CONTEXT.md sections 2, 6, 14.
```

---

## Task 3: OHLCV Fetcher [S]

**Status:** [ ] Not Started
**Depends on:** Task 1, Config (Week 1)
**Estimated time:** 20 minutes

**Acceptance criteria:**
- [ ] OHLCVFetcher fetches candles via exchange adapter
- [ ] Computes all indicators, swings, parent TF in one call
- [ ] Returns complete MarketData ready for agents
- [ ] Unit tests with mock adapter
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

FILE: engine/data/ohlcv.py

class OHLCVFetcher:
    def __init__(self, adapter: ExchangeAdapter, config: TradingConfig): ...

    async def fetch(self, symbol: str, timeframe: str) -> MarketData:
        # 1. Get candle count from DEFAULT_PROFILES[timeframe]
        # 2. Fetch candles via adapter.fetch_ohlcv()
        # 3. Compute all indicators via compute_all_indicators()
        # 4. Detect swings via find_swing_highs/lows
        # 5. Fetch parent TF candles and compute parent context
        # 6. Return complete MarketData with all fields populated

Write tests in tests/unit/test_ohlcv.py:
- MockAdapter returning predetermined candle data
- Test fetch() returns MarketData with indicators computed
- Test parent TF fetched with correct timeframe mapping

Update PROJECT_CONTEXT.md sections 2, 14.
```

---

## Task 4: Chart Generation [M]

**Status:** [ ] Not Started
**Depends on:** Types (Week 1)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] Candlestick chart with volume bars (dark trading theme)
- [ ] OLS trendline overlay chart
- [ ] Charts render to PNG bytes (no file I/O)
- [ ] Grounding context header builder function
- [ ] Unit tests verify valid PNG output
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Reference: v1 chart generation in ../quantagent-v1/chart_generator.py

FILE: engine/data/charts.py

Dark trading theme: bg=#1a1a2e, up=#00d4aa, down=#ff6b6b

def generate_candlestick_chart(candles, symbol, timeframe,
    swing_highs=None, swing_lows=None, width=1024, height=768) -> bytes:
    # Candlesticks + volume bars + optional swing level lines
    # Return PNG bytes via BytesIO

def generate_trendline_chart(candles, symbol, timeframe,
    width=1024, height=768) -> bytes:
    # Candlesticks + OLS trendline (full + last 20) + Bollinger Bands shaded
    # Return PNG bytes

def generate_grounding_header(symbol, timeframe, indicators, flow,
    parent_tf, swing_highs, swing_lows, forecast_candles,
    forecast_description, num_candles, lookback_description) -> str:
    # Build the "CONTEXT (do not override with visual impression):" text block
    # Include all indicator values, flow if available, parent TF if available

Write tests in tests/unit/test_charts.py:
- Test PNG magic bytes in output
- Test grounding header includes indicator values
- Test grounding header handles None flow and parent_tf

Update PROJECT_CONTEXT.md sections 2, 14.
```

---

## Task 5: IndicatorAgent [M]

**Status:** [ ] Not Started
**Depends on:** LLM Provider + SignalProducer (Week 1), Charts (Task 4)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] Implements SignalProducer interface
- [ ] System prompt with structured JSON output instructions
- [ ] Grounding context header injected
- [ ] Parses LLM response into SignalOutput
- [ ] Parse failures return None (SKIP is safe)
- [ ] Prompt in engine/signals/prompts/indicator_v1.py
- [ ] Unit tests with mock LLM
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

First LLM signal agent. Text-only (no vision). Analyzes indicator values.

FILE 1: engine/signals/prompts/indicator_v1.py
System prompt instructs Claude to analyze indicator values and respond in JSON:
direction (BULLISH/BEARISH/NEUTRAL), confidence (0-1), reasoning, contradictions, key_levels.
Include rules: RSI divergence, ADX strength, volume confirmation.
Include {grounding_header} placeholder.

FILE 2: engine/signals/indicator_agent.py
class IndicatorAgent(SignalProducer):
    - name() = "indicator_agent"
    - signal_type() = "llm", requires_vision() = False
    - analyze(): build grounding header, format prompts, call LLM, parse JSON response
    - _parse_response(): extract JSON (handle markdown code blocks), return SignalOutput or None

Write tests in tests/unit/test_indicator_agent.py:
- MockLLMProvider returning predetermined JSON
- Test successful parse into SignalOutput
- Test parse failure returns None
- Test grounding header in system prompt

Update PROJECT_CONTEXT.md sections 2, 5, 14.
```

---

## Task 6: PatternAgent (Vision) [M]

**Status:** [ ] Not Started
**Depends on:** LLM Provider + SignalProducer (Week 1), Charts (Task 4)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] Implements SignalProducer with requires_vision() = True
- [ ] System prompt includes 16-pattern library
- [ ] Sends candlestick chart image to Claude Vision
- [ ] Grounding context prevents hallucination
- [ ] Prompt in engine/signals/prompts/pattern_v1.py
- [ ] Unit tests with mock vision responses
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

First VISION agent. Sends candlestick chart to Claude Vision for pattern detection.

FILE 1: engine/signals/prompts/pattern_v1.py
System prompt with 16-pattern library:
Bullish: ascending_triangle, bull_flag, double_bottom, inverse_head_shoulders,
         cup_and_handle, falling_wedge, bullish_engulfing, morning_star
Bearish: descending_triangle, bear_flag, double_top, head_and_shoulders,
         rising_wedge, bearish_engulfing, evening_star, dark_cloud_cover
Includes {grounding_header} with emphasis: "indicator values are MATHEMATICAL FACTS —
if visual impression conflicts with numbers, numbers are correct."

FILE 2: engine/signals/pattern_agent.py
class PatternAgent(SignalProducer):
    - requires_vision() = True
    - analyze(): generate candlestick chart, build grounding, call generate_vision(), parse
    - Same JSON parse as IndicatorAgent but includes pattern_detected field

Write tests in tests/unit/test_pattern_agent.py

Update PROJECT_CONTEXT.md sections 2, 5, 14.
```

---

## Task 7: TrendAgent (Vision) [M]

**Status:** [ ] Not Started
**Depends on:** LLM Provider + SignalProducer (Week 1), Charts (Task 4)
**Estimated time:** 30 minutes

**Acceptance criteria:**
- [ ] Implements SignalProducer with requires_vision() = True
- [ ] Sends OLS trendline chart to Claude Vision
- [ ] Focuses on trend direction, strength, reversal signals
- [ ] Prompt in engine/signals/prompts/trend_v1.py
- [ ] Unit tests with mock vision responses
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Second vision agent. Analyzes OLS trendlines for trend direction and reversals.

FILE 1: engine/signals/prompts/trend_v1.py
System prompt: analyze primary trendline (full OLS), short-term (last 20 candles),
Bollinger Bands. Rules: slope = bias, divergence = reversal signal, ADX confirms strength.

FILE 2: engine/signals/trend_agent.py
Same structure as PatternAgent but uses generate_trendline_chart() instead of candlestick.
name() = "trend_agent"

Write tests in tests/unit/test_trend_agent.py

Update PROJECT_CONTEXT.md sections 2, 5, 14.
```

---

## Task 8: Risk Profiles + Safety Checks [M]

**Status:** [ ] Not Started
**Depends on:** Config + Types (Week 1)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] compute_sl_tp() from entry, ATR, regime profile, swing levels
- [ ] compute_position_size() from balance, risk%, SL distance
- [ ] 5 mechanical safety checks (conviction floor, daily loss, pyramid gate, position limit, SL validation)
- [ ] Unit tests for every calculation
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

FILE 1: engine/execution/risk_profiles.py
def compute_sl_tp(entry_price, direction, atr, profile, swing_highs, swing_lows) -> dict:
    # ATR-based SL -> snap to structure -> compute TP1 (1:1) + TP2 (full RR)
def compute_position_size(account_balance, risk_per_trade, entry_price, sl_price, max_position_pct) -> float:
    # Risk amount / SL distance, capped at max position

FILE 2: engine/execution/safety_checks.py
def run_safety_checks(action, current_position, daily_pnl, max_daily_loss,
    swing_highs, swing_lows, atr, conviction_score) -> SafetyCheckResult:
    # 5 checks: conviction floor (<0.3), daily loss limit, pyramid gate (ADD near S/R),
    # position limit (1 per symbol), SL validation

Write tests in tests/unit/test_risk_profiles.py and tests/unit/test_safety_checks.py

Update PROJECT_CONTEXT.md sections 2, 14.
```

---

## End of Week 2 Checklist

- [ ] pytest passes all tests (target: ~260+)
- [ ] Hyperliquid adapter fetches real OHLCV (manual test)
- [ ] Charts produce valid PNG images
- [ ] IndicatorAgent returns SignalOutput from real data (manual test)
- [ ] PatternAgent sends chart to Vision and gets response (manual test)
- [ ] TrendAgent sends trendline chart and gets response (manual test)
- [ ] Risk profiles compute correct SL/TP
- [ ] Safety checks block invalid trades
- [ ] All 3 agents registered in SignalRegistry
- [ ] Registry.run_all() runs all 3 in parallel
- [ ] No SQL imports in engine/
- [ ] No FastAPI imports in engine/

---

## Week 3 Preview

- ConvictionAgent (meta-evaluator, fact/subjective labeling)
- DecisionAgent (action selection from ConvictionOutput)
- Pipeline orchestrator (full Data -> Signal -> Conviction -> Execution flow)
- Repository pattern base + PostgreSQL implementation
- FlowAgent with CryptoFlowProvider
- First end-to-end analysis cycle on live data
