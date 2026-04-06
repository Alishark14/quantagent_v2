# SPRINT.md — Week 1 (Phase 1a: Core Engine Foundation)

> Theme: Build the skeleton and foundational interfaces that everything else depends on.
> Start: April 7, 2026
> Target: April 11, 2026
> Tasks: 7 (ordered by dependency — do them in sequence)

---

## Task 1: Project Skeleton [S]

**Status:** [ ] Not Started
**Depends on:** Nothing (first task)
**Estimated time:** 15 minutes

**Acceptance criteria:**
- [ ] `pyproject.toml` exists with all dependencies listed
- [ ] Full folder structure created with `__init__.py` files
- [ ] `version.py` exists with CalVer+SemVer version string
- [ ] `pytest` runs successfully (0 tests, no errors)
- [ ] `.gitignore` covers `.env`, `__pycache__`, `*.pyc`, `.venv/`, `node_modules/`
- [ ] PROJECT_CONTEXT.md section 3 (Project Structure) updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Create the full project skeleton for QuantAgent v2. This is a brand new project — not a modification of v1.

1. Create `pyproject.toml` with:
   - Project name: quantagent
   - Python >= 3.12
   - All dependencies from PROJECT_CONTEXT.md section 10
   - Dev dependencies: pytest, pytest-asyncio, pytest-cov, ruff
   - Entry point: quantagent.main:main

2. Create the ENTIRE folder structure from CLAUDE.md "Project Structure" section.
   Every directory needs an `__init__.py` (empty is fine).
   Every .py file listed in the structure should be created as an empty file
   with just a module docstring explaining its purpose (one line).
   Do NOT write any implementation yet — just the skeleton.

3. Create `version.py` at the project root:
   ENGINE_VERSION = "2026.04.2.0.0-alpha.1"
   API_VERSION = "v1"
   PROMPT_VERSIONS = dict with all 6 agents at "1.0"
   ML_MODEL_VERSIONS = dict with 3 models at None

4. Create `.gitignore` covering: .env, __pycache__/, *.pyc, .venv/,
   node_modules/, dist/, *.egg-info/, .pytest_cache/, .ruff_cache/,
   *.db, *.sqlite3, trade_logs/, *.log

5. Create empty config files:
   - config/features.yaml (all flags from PROJECT_CONTEXT.md section 8)
   - config/sentinel.yaml (placeholder with comments)
   - config/profiles.yaml (timeframe profiles from PROJECT_CONTEXT.md section 9)

6. Verify: `pip install -e ".[dev]"` works and `pytest` runs with 0 tests collected.

7. Update PROJECT_CONTEXT.md:
   - Section 3: update file tree to show all created files
   - Section 14 changelog: add entry
```

---

## Task 2: Core Types [S]

**Status:** [ ] Not Started
**Depends on:** Task 1 (skeleton)
**Estimated time:** 30 minutes

**Acceptance criteria:**
- [ ] All shared dataclasses defined in `engine/types.py`
- [ ] All event types defined in `engine/events.py` (types only, no bus yet)
- [ ] Type imports work from any module (`from engine.types import MarketData`)
- [ ] Unit tests for dataclass construction and validation
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Implement all core data types for QuantAgent v2.

FILE 1: engine/types.py
Define these dataclasses (use @dataclass or Pydantic BaseModel — prefer dataclass for simplicity, Pydantic only where validation is needed):

MarketData:
  symbol: str
  timeframe: str
  candles: list[dict]          # OHLCV dicts with timestamp, open, high, low, close, volume
  num_candles: int
  lookback_description: str    # e.g., "~6 days"
  forecast_candles: int
  forecast_description: str    # e.g., "~3 hours"
  indicators: dict             # computed indicator values (RSI, MACD, etc.)
  swing_highs: list[float]
  swing_lows: list[float]
  parent_tf: ParentTFContext | None
  flow: FlowOutput | None
  external_signals: dict       # provider_name -> signal value

ParentTFContext:
  timeframe: str
  trend_direction: str         # "BULLISH" | "BEARISH" | "NEUTRAL"
  ma_position: str             # "ABOVE_50MA" | "BELOW_50MA"
  adx_value: float
  adx_classification: str      # "TRENDING" | "RANGING" | "WEAK"
  bb_width_percentile: float

FlowOutput:
  funding_rate: float | None
  funding_signal: str          # "CROWDED_LONG" | "CROWDED_SHORT" | "NEUTRAL"
  oi_change_4h: float | None
  oi_trend: str                # "BUILDING" | "DECLINING" | "STABLE"
  nearest_liquidation_above: dict | None  # {price, size}
  nearest_liquidation_below: dict | None  # {price, size}
  gex_regime: str | None       # "POSITIVE_GAMMA" | "NEGATIVE_GAMMA" | None
  gex_flip_level: float | None
  data_richness: str           # "FULL" | "PARTIAL" | "MINIMAL"

SignalOutput:
  agent_name: str
  signal_type: str             # "llm" | "ml"
  direction: str | None        # "BULLISH" | "BEARISH" | "NEUTRAL" | None
  confidence: float            # 0.0 to 1.0
  reasoning: str
  signal_category: str         # "directional" | "regime" | "anomaly"
  data_richness: str           # "full" | "partial" | "minimal"
  contradictions: str          # noted conflicts with grounding data
  key_levels: dict             # {"resistance": float, "support": float}
  pattern_detected: str | None # e.g., "ascending_triangle"
  raw_output: str              # full LLM response for data moat

ConvictionOutput:
  conviction_score: float      # 0.0 to 1.0
  direction: str               # "LONG" | "SHORT" | "SKIP"
  regime: str                  # "TRENDING_UP" | "TRENDING_DOWN" | "RANGING" | "HIGH_VOLATILITY" | "BREAKOUT"
  regime_confidence: float
  signal_quality: str          # "HIGH" | "MEDIUM" | "LOW" | "CONFLICTING"
  contradictions: list[str]
  reasoning: str
  factual_weight: float
  subjective_weight: float
  raw_output: str              # full LLM response for data moat

TradeAction:
  action: str                  # "LONG" | "SHORT" | "ADD_LONG" | "ADD_SHORT" | "CLOSE_ALL" | "HOLD" | "SKIP"
  conviction_score: float
  position_size: float | None
  sl_price: float | None
  tp1_price: float | None
  tp2_price: float | None
  rr_ratio: float | None
  atr_multiplier: float | None
  reasoning: str
  raw_output: str

OrderResult:
  success: bool
  order_id: str | None
  fill_price: float | None
  fill_size: float | None
  error: str | None

Position:
  symbol: str
  direction: str               # "long" | "short"
  size: float
  entry_price: float
  unrealized_pnl: float
  leverage: float | None

AdapterCapabilities:
  native_sl_tp: bool
  supports_short: bool
  market_hours: dict | None    # None = 24/7
  asset_types: list[str]
  margin_type: str
  has_funding_rate: bool
  has_oi_data: bool
  max_leverage: float
  order_types: list[str]
  supports_partial_close: bool

FILE 2: engine/events.py (types only, Event Bus implementation is Task 3)
Define event dataclasses, all inheriting from a base Event:

Event (base):
  timestamp: datetime (auto-set to now)
  source: str

DataReady(Event): market_data: MarketData
SignalsReady(Event): signals: list[SignalOutput]
ConvictionScored(Event): conviction: ConvictionOutput
TradeOpened(Event): trade_action: TradeAction, order_result: OrderResult
TradeClosed(Event): symbol: str, pnl: float, exit_reason: str
PositionUpdated(Event): symbol: str, position: Position
SetupDetected(Event): symbol: str, readiness: float, conditions: list[str]
RuleGenerated(Event): rule: dict
FactorsUpdated(Event): filepath: str
MacroUpdated(Event): filepath: str
CycleCompleted(Event): symbol: str, action: str, conviction: float

Write unit tests in tests/unit/test_types.py:
- Test construction of every dataclass with valid data
- Test that required fields raise errors when missing
- Test serialization to dict (for database storage)

Update PROJECT_CONTEXT.md:
- Section 2: mark Types as IMPLEMENTED
- Section 4: mark event types as DEFINED (not bus implementation)
- Section 14 changelog
```

---

## Task 3: Event Bus [M]

**Status:** [ ] Not Started
**Depends on:** Task 2 (types)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] EventBus abstract class with publish/subscribe
- [ ] InProcessBus implementation (asyncio, for single-server)
- [ ] Type-safe subscriptions (subscribe to DataReady, only receive DataReady)
- [ ] Multiple handlers per event type
- [ ] Fire-and-forget: handler errors don't crash the publisher
- [ ] Unit tests with mock handlers
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Implement the Event Bus — the central nervous system of the modular architecture.

FILE: engine/events.py (extend the event types already defined in Task 2)

Add the EventBus abstract class and InProcessBus implementation:

class EventBus(ABC):
    @abstractmethod
    async def publish(self, event: Event) -> None: ...

    @abstractmethod
    def subscribe(self, event_type: type[Event], handler: Callable) -> None: ...

    @abstractmethod
    def unsubscribe(self, event_type: type[Event], handler: Callable) -> None: ...

class InProcessBus(EventBus):
    - Uses a dict[type[Event], list[Callable]] for subscriptions
    - publish() calls all handlers for that event type
    - Each handler is wrapped in try/except — if a handler raises,
      log the error but continue to the next handler. Never crash the publisher.
    - Handlers are called with asyncio.gather() for parallel execution
    - Include a metrics counter: total events published, per-type counts, handler errors
    - Thread-safe (use asyncio.Lock if needed)

Also add a convenience function:
def create_event_bus(backend: str = "memory") -> EventBus:
    if backend == "memory":
        return InProcessBus()
    elif backend == "redis":
        raise NotImplementedError("Redis EventBus planned for multi-server")
    else:
        raise ValueError(f"Unknown bus backend: {backend}")

Write tests in tests/unit/test_event_bus.py:
- Test subscribe + publish delivers event to handler
- Test multiple handlers for same event type all receive it
- Test handler error doesn't crash publisher (other handlers still run)
- Test unsubscribe stops delivery
- Test event type filtering (subscribe to DataReady, publish SignalsReady — handler NOT called)
- Test publish with no subscribers doesn't error
- Test metrics counting

Update PROJECT_CONTEXT.md:
- Section 2: mark Event Bus as IMPLEMENTED
- Section 4: mark events as IMPLEMENTED
- Section 14 changelog
```

---

## Task 4: Config System [S]

**Status:** [ ] Not Started
**Depends on:** Task 1 (skeleton)
**Estimated time:** 30 minutes

**Acceptance criteria:**
- [ ] `engine/config.py` loads from env vars + YAML files
- [ ] TradingConfig dataclass with all per-bot parameters
- [ ] TimeframeProfiles with base profiles + regime multiplier methods
- [ ] FeatureFlags loaded from `config/features.yaml` with env var overrides
- [ ] Unit tests for config loading, profile multiplication, flag resolution
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Implement the configuration system.

FILE: engine/config.py

1. TradingConfig dataclass:
   symbol: str = "BTC-USDC"
   timeframe: str = "1h"
   exchange: str = "hyperliquid"
   account_balance: float = 0  # 0 = fetch from exchange
   atr_length: int = 14
   forecast_candles: int = 3   # dynamic per regime later
   max_concurrent_positions: int = 1
   max_position_pct: float = 1.0
   conviction_threshold: float = 0.5

   @classmethod
   def from_env(cls) -> "TradingConfig":
       # Load from environment variables with defaults

2. TimeframeProfile dataclass:
   timeframe: str
   candles: int
   atr_multiplier: float
   rr_min: float
   rr_max: float
   trailing_enabled: bool  # True for 4h+

   DEFAULT_PROFILES dict mapping timeframe -> TimeframeProfile:
   - 15m: candles=100, atr_mult=2.5, rr=0.8/1.2, trailing=False
   - 30m: candles=100, atr_mult=2.0, rr=1.0/1.5, trailing=False
   - 1h: candles=150, atr_mult=1.5, rr=1.5/2.0, trailing=False
   - 4h: candles=150, atr_mult=1.0, rr=3.0/5.0, trailing=True
   - 1d: candles=200, atr_mult=1.0, rr=3.0/5.0, trailing=True

3. get_dynamic_profile(base: TimeframeProfile, regime: str,
                       volatility_percentile: float) -> TimeframeProfile:
   Apply regime multipliers:
   - TRENDING / TRENDING_UP / TRENDING_DOWN: atr_mult *= 0.8, rr_min *= 1.3, rr_max *= 1.5
   - RANGING: atr_mult *= 1.2, rr_min *= 0.7, rr_max *= 0.8
   - HIGH_VOLATILITY: atr_mult *= 1.3, rr_min *= 0.8, rr_max *= 1.0
   - BREAKOUT: atr_mult *= 0.9, rr_min *= 1.5, rr_max *= 2.0

   Apply volatility scaling:
   - percentile > 80: atr_mult *= 1.15
   - percentile < 20: atr_mult *= 0.85

   Return new TimeframeProfile with adjusted values.

4. FeatureFlags class:
   Load from config/features.yaml.
   Every flag can be overridden by env var: FEATURE_SENTINEL_ENABLED=true
   Env vars take priority over YAML.
   Method: is_enabled(flag_name: str) -> bool

5. Helper functions:
   get_lookback_description(timeframe: str, num_candles: int) -> str
     - 100 candles × 15m = "~25 hours"
     - 150 candles × 1h = "~6 days"
     - etc.

   get_forecast_description(timeframe: str, forecast_candles: int) -> str
     - 3 × 15m = "~45 minutes"
     - 3 × 1h = "~3 hours"
     - etc.

   timeframe_to_seconds(tf: str) -> int
     - "15m" -> 900, "1h" -> 3600, etc.

Write tests in tests/unit/test_config.py:
- Test TradingConfig.from_env() with mock env
- Test all 5 timeframe profiles exist with correct values
- Test get_dynamic_profile with each regime type
- Test volatility scaling at edge percentiles
- Test FeatureFlags loads YAML and respects env overrides
- Test lookback_description and forecast_description for each timeframe

Update config/profiles.yaml with the profiles (used by the YAML loader).
Update PROJECT_CONTEXT.md section 2, 14.
```

---

## Task 5: LLM Provider Abstraction [M]

**Status:** [ ] Not Started
**Depends on:** Task 1 (skeleton)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] Abstract `LLMProvider` with `generate_text()` and `generate_vision()` methods
- [ ] `ClaudeProvider` implementation using Anthropic SDK
- [ ] Prompt caching support on system prompts
- [ ] LangSmith tracing integration
- [ ] Structured output parsing (extract JSON from LLM response)
- [ ] Retry logic with exponential backoff (3 retries)
- [ ] Token usage tracking (input/output tokens per call)
- [ ] Unit tests with mock responses
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Implement the LLM provider abstraction layer.

FILE 1: llm/base.py

class LLMProvider(ABC):
    @abstractmethod
    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,           # for LangSmith tracing
        max_tokens: int = 1024,
        temperature: float = 0.3,
        cache_system_prompt: bool = True,
    ) -> LLMResponse: ...

    @abstractmethod
    async def generate_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data: bytes,         # PNG image bytes
        image_media_type: str,     # "image/png"
        agent_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        cache_system_prompt: bool = True,
    ) -> LLMResponse: ...

@dataclass
class LLMResponse:
    content: str                   # raw text response
    input_tokens: int
    output_tokens: int
    cost: float                    # estimated cost in USD
    model: str
    latency_ms: float
    cached_input_tokens: int       # tokens served from cache

FILE 2: llm/claude.py

class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    Implementation notes:
    - Use anthropic SDK async client
    - For generate_text: build messages with system prompt + user prompt
    - For generate_vision: build messages with image content block (base64)
    - If cache_system_prompt=True, add cache_control={"type": "ephemeral"}
      to the system prompt content block
    - Wrap in try/except with 3 retries, exponential backoff (1s, 2s, 4s)
    - Track timing with time.perf_counter()
    - Calculate cost using MODEL_COSTS dict:
      claude-sonnet-4: input=$3/M, output=$15/M, cached_input=$0.30/M
    - Return LLMResponse with all fields populated
    - Log every call: logger.info(f"LLM call: agent={agent_name}, "
      f"tokens={input}/{output}, cost=${cost:.4f}, latency={ms}ms")

FILE 3: llm/cache.py (simple utility)

class PromptCache:
    """Track which system prompts are warm in Anthropic's cache."""
    def __init__(self):
        self._warm_prompts: dict[str, datetime] = {}

    def mark_warm(self, prompt_hash: str) -> None: ...
    def is_warm(self, prompt_hash: str) -> bool: ...
    # Anthropic cache has ~5 min TTL, track staleness

Write tests in tests/unit/test_llm.py:
- Mock the anthropic client
- Test generate_text returns correct LLMResponse
- Test generate_vision includes image content block
- Test retry logic (mock first call fails, second succeeds)
- Test cost calculation accuracy
- Test cache_control header is set when cache_system_prompt=True

Update PROJECT_CONTEXT.md section 2, 14.
```

---

## Task 6: Indicator Calculator [M]

**Status:** [ ] Not Started
**Depends on:** Task 1 (skeleton), Task 2 (types)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] All 9 indicators computed correctly: RSI, MACD, ROC, Stochastic, Williams %R, ATR, ADX, Bollinger Bands, Volume MA
- [ ] Pure functions, no external dependencies beyond numpy
- [ ] Comprehensive unit tests with known input → expected output
- [ ] Swing detection (high/low from last N candles)
- [ ] Parent timeframe trend computation
- [ ] Helper: get_volatility_percentile (current ATR vs last 100 ATR values)
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Implement all technical indicator computation. These are pure math functions
with no external dependencies beyond numpy. They are the FACTUAL ground truth
that grounds all LLM analysis.

FILE 1: engine/data/indicators.py

All functions take numpy arrays (close, high, low, volume) and return values.

def compute_rsi(close: np.ndarray, period: int = 14) -> float:
    # Standard RSI using Wilder's smoothing
    # Return the latest value only (not the series)

def compute_macd(close: np.ndarray, fast: int = 12, slow: int = 26,
                 signal: int = 9) -> dict:
    # Return {"macd": float, "signal": float, "histogram": float,
    #         "histogram_direction": "rising"|"falling",
    #         "cross": "bullish_cross"|"bearish_cross"|"none"}

def compute_roc(close: np.ndarray, period: int = 10) -> float:
    # Rate of change: (close[-1] - close[-period]) / close[-period] * 100

def compute_stochastic(high: np.ndarray, low: np.ndarray,
                       close: np.ndarray, k_period: int = 14,
                       d_period: int = 3, smooth: int = 3) -> dict:
    # Return {"k": float, "d": float, "zone": "overbought"|"oversold"|"neutral"}

def compute_williams_r(high: np.ndarray, low: np.ndarray,
                       close: np.ndarray, period: int = 14) -> float:
    # Williams %R: -100 to 0

def compute_atr(high: np.ndarray, low: np.ndarray,
                close: np.ndarray, period: int = 14) -> float:
    # Average True Range using Wilder's smoothing

def compute_adx(high: np.ndarray, low: np.ndarray,
                close: np.ndarray, period: int = 14) -> dict:
    # Return {"adx": float, "plus_di": float, "minus_di": float,
    #         "classification": "TRENDING"|"RANGING"|"WEAK"}
    # TRENDING: ADX > 25, RANGING: ADX < 20, WEAK: 20-25

def compute_bollinger_bands(close: np.ndarray, period: int = 20,
                            std_dev: float = 2.0) -> dict:
    # Return {"upper": float, "middle": float, "lower": float,
    #         "width": float, "width_percentile": float}
    # width_percentile: current width vs last 100 widths

def compute_volume_ma(volume: np.ndarray, period: int = 20) -> dict:
    # Return {"ma": float, "current": float,
    #         "ratio": float, "spike": bool}
    # spike: current > 3x MA

def compute_all_indicators(candles: list[dict]) -> dict:
    # Extract OHLCV arrays from candle dicts
    # Call all indicator functions
    # Return unified dict with all values
    # This is what gets injected into agent prompts

def get_volatility_percentile(atr_series: np.ndarray) -> float:
    # Where does current ATR sit in the last 100 ATR values?
    # Returns 0-100 percentile

FILE 2: engine/data/swing_detection.py

def find_swing_highs(high: np.ndarray, lookback: int = 50,
                     num_swings: int = 3) -> list[float]:
    # Find the N most significant swing highs in the last `lookback` candles
    # A swing high: high[i] > high[i-1] AND high[i] > high[i+1] (basic)
    # For robustness: use 2-bar or 3-bar pivot detection
    # Return prices sorted by proximity to current price (nearest first)

def find_swing_lows(low: np.ndarray, lookback: int = 50,
                    num_swings: int = 3) -> list[float]:
    # Same logic for lows

def adjust_sl_to_structure(sl_price: float, direction: str,
                           swing_highs: list[float],
                           swing_lows: list[float],
                           atr: float, buffer_pct: float = 0.002) -> float:
    # If a swing level exists within ±15% of the ATR-based SL,
    # snap SL just beyond it (buffer_pct beyond the swing level)
    # For LONG: snap to nearest swing low below entry
    # For SHORT: snap to nearest swing high above entry
    # Return adjusted SL price

FILE 3: engine/data/parent_tf.py

def compute_parent_tf_context(candles: list[dict],
                               timeframe: str) -> ParentTFContext:
    # From 50 parent TF candles, compute:
    # - MA direction: is price above or below 50-period SMA?
    # - ADX value and classification
    # - Bollinger Band width percentile
    # Return ParentTFContext dataclass

def get_parent_timeframe(trading_tf: str) -> str:
    # 15m -> 1h, 30m -> 4h, 1h -> 4h, 4h -> 1d, 1d -> 1w

Write comprehensive tests in tests/unit/test_indicators.py:
- Test RSI with known values (e.g., 14 up candles = RSI ~100,
  14 down = RSI ~0, mixed = between)
- Test MACD crossover detection
- Test Stochastic overbought/oversold zones
- Test ATR with a simple series (all candles same range = ATR equals that range)
- Test ADX classification thresholds
- Test swing detection finds correct peaks/troughs
- Test adjust_sl_to_structure snaps to nearby swing level
- Test parent TF context computation
- Test get_parent_timeframe mapping

Write tests in tests/unit/test_swing_detection.py:
- Separate detailed tests for swing detection edge cases

Update PROJECT_CONTEXT.md section 2, 14.
```

---

## Task 7: SignalProducer Interface + Registry [S]

**Status:** [ ] Not Started
**Depends on:** Task 2 (types), Task 4 (config)
**Estimated time:** 30 minutes

**Acceptance criteria:**
- [ ] Abstract `SignalProducer` class in `engine/signals/base.py`
- [ ] `SignalRegistry` that manages producers from config
- [ ] ML model base class that returns `None` (slot pattern)
- [ ] Unit tests for registry: add, remove, list, filter by type
- [ ] Integration test: registry runs mock producers in parallel
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Implement the SignalProducer interface and registry that manages all
signal producers (LLM agents and ML models) as pluggable components.

FILE 1: engine/signals/base.py

from abc import ABC, abstractmethod
from engine.types import MarketData, SignalOutput

class SignalProducer(ABC):
    """Base class for all signal producers (LLM agents and ML models)."""

    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this producer."""
        ...

    @abstractmethod
    def signal_type(self) -> str:
        """'llm' or 'ml'"""
        ...

    @abstractmethod
    def is_enabled(self) -> bool:
        """Check feature flag. Disabled producers are skipped."""
        ...

    @abstractmethod
    async def analyze(self, data: MarketData) -> SignalOutput | None:
        """Produce a signal from market data. Return None if unable."""
        ...

    def requires_vision(self) -> bool:
        """Override to True for vision-based agents."""
        return False


FILE 2: engine/signals/registry.py

class SignalRegistry:
    """Manages all registered SignalProducers. Config-driven."""

    def __init__(self):
        self._producers: list[SignalProducer] = []

    def register(self, producer: SignalProducer) -> None:
        """Add a producer to the registry."""

    def unregister(self, name: str) -> None:
        """Remove a producer by name."""

    def get_enabled(self) -> list[SignalProducer]:
        """Return only producers where is_enabled() is True."""

    def get_by_type(self, signal_type: str) -> list[SignalProducer]:
        """Filter by 'llm' or 'ml'."""

    async def run_all(self, data: MarketData) -> list[SignalOutput]:
        """Run all enabled producers in parallel.
        Use asyncio.gather(). If a producer raises, log the error
        and continue (do not fail the entire batch).
        Return list of non-None results."""


FILE 3: engine/signals/ml/__init__.py (or ml/base.py)

class MLModelSlot(SignalProducer):
    """Base class for ML model slots. Returns None until a trained
    model is loaded. Subclasses override analyze() when trained."""

    def __init__(self, name: str, feature_flag: str):
        self._name = name
        self._flag = feature_flag
        self._model = None  # loaded later

    def name(self) -> str:
        return self._name

    def signal_type(self) -> str:
        return "ml"

    def is_enabled(self) -> bool:
        return self._model is not None and FeatureFlags.is_enabled(self._flag)

    async def analyze(self, data: MarketData) -> SignalOutput | None:
        if not self.is_enabled():
            return None
        return self._predict(data)

    def _predict(self, data: MarketData) -> SignalOutput:
        raise NotImplementedError("Model not trained yet")

    def load_model(self, model_path: str) -> None:
        """Load a trained model from disk."""
        raise NotImplementedError


FILE 4: Create placeholder ML slots:
- engine/signals/ml/direction.py: DirectionModel(MLModelSlot) with name="direction_model"
- engine/signals/ml/regime.py: RegimeModel(MLModelSlot) with name="regime_model"
- engine/signals/ml/anomaly.py: AnomalyDetector(MLModelSlot) with name="anomaly_detector"

Each is a minimal subclass that just sets the name and feature flag.
They all return None until trained.

Write tests in tests/unit/test_signal_registry.py:
- Test register and get_enabled
- Test disabled producers are excluded from get_enabled
- Test run_all executes producers in parallel
- Test run_all handles producer errors gracefully (one fails, others still return)
- Test ML slots return None when no model loaded
- Test get_by_type filters correctly

Write tests in tests/integration/test_signal_registry.py:
- Create 3 mock SignalProducers with different results
- Run registry.run_all() and verify all 3 results returned
- Create 1 mock that raises, verify other 2 still return

Update PROJECT_CONTEXT.md:
- Section 2: mark SignalProducer Base, Signal Registry, ML Model Slots as IMPLEMENTED
- Section 14 changelog
```

---

## End of Week 1 Checklist

After completing all 7 tasks, verify:

- [ ] `pytest` passes with all tests green
- [ ] The Event Bus can publish and subscribe events
- [ ] Config loads from env + YAML correctly
- [ ] LLM provider can make a real Claude API call (manual test, not automated)
- [ ] All indicators compute correct values
- [ ] SignalRegistry can run producers in parallel
- [ ] PROJECT_CONTEXT.md is fully updated with all implemented modules
- [ ] All new code has type hints
- [ ] No SQL imports in the engine/ directory
- [ ] No FastAPI imports in the engine/ directory

**If all pass:** the foundation is solid. Week 2 builds the actual agents and Sentinel on top of these interfaces.

---

## Week 2 Preview (Phase 1a continued)

> Not yet detailed. Will be written during Saturday review.

- Exchange adapter base + Hyperliquid adapter
- Repository pattern base + PostgreSQL implementation
- IndicatorAgent (first LLM agent on SignalProducer interface)
- PatternAgent (first vision agent)
- TrendAgent
- Chart generation (matplotlib candlestick + trendlines)
- OHLCV fetcher with DataProvider registry
