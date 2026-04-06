# CLAUDE.md — Instructions for Claude Code (QuantAgent v2)

## First Step (Every Task)

Read `PROJECT_CONTEXT.md` before writing any code. It contains the current implementation state, what's built, what's not, and active decisions. If you skip this, you will repeat solved problems or break working systems.

For full architectural understanding, read `ARCHITECTURE.md` (the v2 architecture specification). PROJECT_CONTEXT.md tells you where we are. ARCHITECTURE.md tells you where we're going.

---

## Core Rules

1. **Modular via Event Bus.** Modules communicate through typed events, never direct imports. DataModule emits `DATA_READY`, SignalModule subscribes to it. If you're tempted to import one module inside another, you're doing it wrong — emit an event instead.

2. **SignalProducer interface is sacred.** Every signal agent (LLM or ML) implements `SignalProducer`. Adding a new agent = one new file + register in config. Zero changes to the pipeline.

3. **Exchange adapter pattern is mandatory.** The engine has zero CCXT imports. All exchange logic lives in `exchanges/*_adapter.py` behind `ExchangeAdapter`. Adding a new exchange = one new file. Zero changes to engine.

4. **Repository pattern is mandatory.** The engine has zero SQL imports. All database access goes through repository interfaces (`TradeRepository`, `BotRepository`, `RuleRepository`). PostgreSQL is the standard backend. SQLite exists only for local dev fallback.

5. **Exchange is the only source of truth** for position status. Never trust internal records over what the exchange reports. Position sync corrects mismatches.

6. **SKIP is always safe.** On any error, ambiguity, parse failure, or low conviction in the trading pipeline, default to SKIP. Never default to LONG or SHORT. Never default to CLOSE_ALL on parse failure for open positions — default to HOLD.

7. **NEVER modify agent prompt files** (`agents/prompts/*.py` or `*.md`) unless explicitly asked. Prompt changes require explicit approval because they affect signal quality. Every prompt change must be regression-tested against historical data before deployment.

8. **Fact vs. Subjective labeling.** Every input to ConvictionAgent must be explicitly labeled as factual (computed, deterministic) or subjective (LLM-interpreted). If you add a new data source, label it.

9. **Feature flags for everything.** New features, agents, data sources, and integrations must be behind feature flags. Nothing is hardcoded as always-on. The system must function with any feature disabled.

10. **Data moat capture is non-negotiable.** Every analysis cycle must record all 6 layers of data (raw market, sensory inputs, cognitive process, action, outcome, reflection). If you add a new agent or data source, ensure its inputs and outputs are captured in the cycle record.

---

## Architecture Quick Reference

```
SENTINEL (persistent, code-only, per symbol)
  │ monitors price, computes readiness, manages SL/TP
  │
  ├── SETUP_DETECTED or SCHEDULED_TIMER
  │
  └── ANALYSIS PIPELINE (spawns TraderBot)
        │
        ├── 1. DATA LAYER (code: OHLCV, indicators, flow, parent TF)
        ├── 2. SIGNAL LAYER (parallel: 3 LLM + N ML via SignalProducer)
        ├── 3. CONVICTION LAYER (1 LLM: ConvictionAgent scores 0-1)
        ├── 4. EXECUTION LAYER (1 LLM + mechanical checks)
        └── 5. REFLECTION (async, after trade closes)
```

Total per cycle: 5 LLM calls (~$0.045). FlowAgent and ML models are code-only ($0.00).

---

## Code Conventions

- **Python 3.12+**
- **Type hints** on all functions — no exceptions
- **Dataclasses or Pydantic** for all data structures (SignalOutput, ConvictionOutput, MarketData, FlowOutput, etc.)
- **Async/await** for all I/O operations (exchange calls, database, LLM calls)
- **Logging** with module-level `logger = logging.getLogger(__name__)`
- **All external API calls** wrapped in try/except with structured error logging — never crash on API failure
- **Event emission** is fire-and-forget — always wrapped in try/except, never blocks the pipeline
- **No global mutable state** — all state flows through function arguments or the Event Bus
- **f-strings** for formatting, never `.format()` or `%`
- **Imports** organized: stdlib → third-party → local, separated by blank lines

---

## Project Structure

```
quantagent-v2/
├── ARCHITECTURE.md              # Full system specification (read-only reference)
├── PROJECT_CONTEXT.md           # Current implementation state (update after every task)
├── CLAUDE.md                    # This file (rules for Claude Code)
├── SPRINT.md                    # Current week's tasks
├── BACKLOG.md                   # Prioritized future tasks
├── CHANGELOG.md                 # What changed and when
├── pyproject.toml               # Project dependencies and metadata
├── version.py                   # CalVer+SemVer, model costs, prompt versions
│
├── engine/                      # THE CORE — pure library, no web/DB dependencies
│   ├── __init__.py
│   ├── pipeline.py              # Orchestrates Data → Signal → Conviction → Execution
│   ├── events.py                # Event Bus: typed events, publish/subscribe
│   ├── types.py                 # All shared types: MarketData, SignalOutput, ConvictionOutput, etc.
│   ├── config.py                # TradingConfig, TimeframeProfiles, feature flags
│   │
│   ├── data/                    # DATA LAYER — all code, zero LLM
│   │   ├── __init__.py
│   │   ├── ohlcv.py             # OHLCV fetcher with DataProvider registry
│   │   ├── indicators.py        # RSI, MACD, ROC, Stoch, WillR, ATR, ADX, BB
│   │   ├── swing_detection.py   # Swing high/low detection
│   │   ├── charts.py            # Candlestick + trendline chart generation (matplotlib)
│   │   ├── parent_tf.py         # Parent timeframe trend computation
│   │   └── flow/                # FlowAgent — pluggable providers
│   │       ├── __init__.py
│   │       ├── base.py          # Abstract FlowProvider
│   │       ├── crypto.py        # CryptoFlowProvider (funding, OI, liquidations)
│   │       ├── options.py       # OptionsEnrichment (GEX, skew — BTC/ETH/equities)
│   │       ├── equity.py        # EquityFlowProvider (dark pool, institutional flow)
│   │       └── forex.py         # ForexFlowProvider (COT, DXY)
│   │
│   ├── signals/                 # SIGNAL LAYER — SignalProducer implementations
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract SignalProducer interface
│   │   ├── registry.py          # SignalProducer registry (config-driven)
│   │   ├── indicator_agent.py   # IndicatorAgent (LLM text)
│   │   ├── pattern_agent.py     # PatternAgent (LLM vision)
│   │   ├── trend_agent.py       # TrendAgent (LLM vision)
│   │   ├── prompts/             # Agent prompt templates (versioned)
│   │   │   ├── indicator_v1.py
│   │   │   ├── pattern_v1.py
│   │   │   └── trend_v1.py
│   │   └── ml/                  # ML model slots
│   │       ├── __init__.py
│   │       ├── direction.py     # DirectionModel slot (returns null until trained)
│   │       ├── regime.py        # RegimeModel slot
│   │       └── anomaly.py       # AnomalyDetector slot
│   │
│   ├── conviction/              # CONVICTION LAYER
│   │   ├── __init__.py
│   │   ├── agent.py             # ConvictionAgent (LLM)
│   │   └── prompts/
│   │       └── conviction_v1.py
│   │
│   ├── execution/               # EXECUTION LAYER
│   │   ├── __init__.py
│   │   ├── agent.py             # DecisionAgent (LLM)
│   │   ├── executor.py          # Order placement, SL/TP, position sizing
│   │   ├── risk_profiles.py     # Dynamic regime-driven profiles
│   │   ├── safety_checks.py     # Mechanical checks (pyramid gate, daily loss, etc.)
│   │   └── prompts/
│   │       └── decision_v1.py
│   │
│   ├── reflection/              # REFLECTION LAYER (async)
│   │   ├── __init__.py
│   │   ├── agent.py             # ReflectionAgent (LLM, post-trade)
│   │   └── prompts/
│   │       └── reflection_v1.py
│   │
│   └── memory/                  # MEMORY SYSTEM
│       ├── __init__.py
│       ├── cycle_memory.py      # Loop 1: recent cycles, position state
│       ├── reflection_rules.py  # Loop 2: learned rules with self-correcting counters
│       ├── cross_bot.py         # Loop 3: cross-bot signal sharing (user_id scoped)
│       └── regime_history.py    # Loop 4: regime ring buffer
│
├── sentinel/                    # SENTINEL SYSTEM — persistent, code-only
│   ├── __init__.py
│   ├── monitor.py               # WebSocket price feed, readiness scoring
│   ├── conditions.py            # Readiness conditions (indicator cross, level touch, etc.)
│   ├── position_manager.py      # SL/TP adjustment between TraderBot lifecycles
│   ├── reaper.py                # Orphan position detection and emergency SL
│   └── config.py                # Sentinel thresholds, cooldown, budget
│
├── exchanges/                   # EXCHANGE ADAPTERS — one file per exchange
│   ├── __init__.py
│   ├── base.py                  # Abstract ExchangeAdapter + AdapterCapabilities
│   ├── factory.py               # get_adapter(name) → singleton cache
│   ├── hyperliquid.py           # PRIMARY — native SL/TP, HIP-3, WalletConnect
│   ├── dydx.py                  # SECONDARY — IOC orders, patches
│   └── deribit.py               # LEGACY — options data source
│
├── llm/                         # LLM PROVIDER ABSTRACTION
│   ├── __init__.py
│   ├── base.py                  # Abstract LLMProvider (text + vision)
│   ├── claude.py                # ClaudeProvider (primary — text + vision)
│   ├── groq.py                  # GroqProvider (cost optimization — text only)
│   └── cache.py                 # Prompt caching management
│
├── storage/                     # STORAGE ABSTRACTION
│   ├── __init__.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract repository interfaces
│   │   ├── postgres.py          # PostgreSQL implementations
│   │   └── sqlite.py            # SQLite fallback (local dev only)
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract CacheManager
│   │   ├── memory.py            # In-process cache (cachetools)
│   │   └── redis.py             # Redis distributed cache
│   └── object_store/
│       ├── __init__.py
│       ├── base.py              # Abstract ObjectStore
│       ├── local.py             # Local filesystem
│       └── s3.py                # S3-compatible (MinIO / AWS)
│
├── tracking/                    # TRACKING & OBSERVABILITY
│   ├── __init__.py
│   ├── financial.py             # Per-trade, per-bot, per-portfolio metrics
│   ├── decision.py              # Per-cycle decision capture, signal quality
│   ├── health.py                # Bot health, API health, infrastructure
│   ├── data_moat.py             # 6-layer data capture for training
│   └── audit.py                 # Full trade audit trail
│
├── distribution/                # SIGNAL DISTRIBUTION
│   ├── __init__.py
│   ├── base.py                  # Abstract NotificationChannel
│   ├── discord.py               # Discord webhook
│   ├── telegram.py              # Telegram Bot API
│   └── formatter.py             # Signal message formatting
│
├── mcp/                         # OFFLINE MCP AGENTS
│   ├── __init__.py
│   ├── overnight_quant.py       # Alpha mining (nightly cron)
│   └── macro_regime.py          # Macro regime assessment (daily)
│
├── api/                         # FASTAPI — web layer (NOT part of engine)
│   ├── __init__.py
│   ├── app.py                   # FastAPI application factory
│   ├── auth.py                  # JWT + API key authentication
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── bots.py              # Bot management endpoints
│   │   ├── trading.py           # B2B API: /v1/analyze, /v1/conviction, etc.
│   │   ├── portfolio.py         # Portfolio, trades, positions
│   │   ├── health.py            # System health endpoints
│   │   └── websocket.py         # WebSocket streams (Sentinel, trades, etc.)
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── rate_limit.py        # Per-user / per-API-key rate limiting
│   │   └── tenant.py            # Multi-tenant user_id injection
│   └── dependencies.py          # FastAPI dependency injection
│
├── dashboard/                   # FRONTEND (SolidJS — Phase 4)
│   └── (empty until Phase 4)
│
├── tests/                       # TEST SUITE
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_indicators.py
│   │   ├── test_position_sizing.py
│   │   ├── test_risk_profiles.py
│   │   ├── test_swing_detection.py
│   │   └── test_safety_checks.py
│   ├── integration/
│   │   ├── test_pipeline.py     # Full pipeline with mock LLM
│   │   ├── test_event_bus.py
│   │   └── test_repositories.py
│   ├── adapters/
│   │   └── test_hyperliquid.py
│   └── fixtures/                # Recorded LLM responses, sample OHLCV data
│       ├── sample_ohlcv.json
│       └── mock_llm_responses.json
│
├── scripts/                     # OPERATIONAL SCRIPTS
│   ├── migrate_db.py            # Database migrations
│   ├── collect_metrics.py       # Cron: query DB → METRICS.md
│   └── health_check.py          # Quick system health verification
│
├── config/                      # CONFIGURATION
│   ├── features.yaml            # Feature flags
│   ├── sentinel.yaml            # Sentinel thresholds and conditions
│   └── profiles.yaml            # Timeframe profiles and regime multipliers
│
└── .env                         # Secrets only (gitignored)
```

---

## Event Bus Conventions

Events are typed dataclasses. Every event has a timestamp and source module.

```python
@dataclass
class Event:
    timestamp: datetime
    source: str

@dataclass
class DataReady(Event):
    market_data: MarketData

@dataclass
class SignalsReady(Event):
    signals: list[SignalOutput]

@dataclass
class ConvictionScored(Event):
    conviction: ConvictionOutput

@dataclass
class TradeOpened(Event):
    trade: Trade

@dataclass
class TradeClosed(Event):
    trade: Trade
    outcome: TradeOutcome

@dataclass
class SetupDetected(Event):
    symbol: str
    readiness: float
    conditions: list[str]
```

**Rules:**
- Never import a module to call it directly. Emit an event.
- Event handlers must be idempotent — the same event processed twice produces the same result.
- Event handlers must not block — use async, fire-and-forget for non-critical handlers.
- TrackingModule subscribes to ALL events. If you create a new event type, ensure tracking captures it.

---

## Symbol Convention

Internal format: `BASE-QUOTE` (e.g., `BTC-USDC`, `GOLD-USDC`). Exchange adapters handle conversion.

| Category | Internal | CCXT Format | Data Source |
|----------|----------|-------------|-------------|
| Crypto | BTC-USDC | `BTC/USDC:USDC` | Bybit / Binance |
| Commodities (HIP-3) | GOLD-USDC | `XYZ-GOLD/USDC:USDC` | Hyperliquid |
| Indices (HIP-3) | SP500-USDC | `XYZ-SP500/USDC:USDC` | Hyperliquid |
| Stocks (HIP-3) | TSLA-USDC | `XYZ-TSLA/USDC:USDC` | Hyperliquid |
| Forex (HIP-3) | EUR-USDC | `XYZ-EUR/USDC:USDC` | Hyperliquid |

---

## Key Interfaces (Do Not Break)

These are the contracts that the entire system depends on. Changing their signatures requires updating every consumer.

```python
# engine/signals/base.py
class SignalProducer(ABC):
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def signal_type(self) -> Literal["llm", "ml"]: ...
    @abstractmethod
    async def analyze(self, data: MarketData) -> SignalOutput | None: ...

# exchanges/base.py
class ExchangeAdapter(ABC):
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities: ...
    @abstractmethod
    async def place_market_order(self, symbol, side, size) -> OrderResult: ...
    @abstractmethod
    async def place_sl_order(self, symbol, side, size, price) -> OrderResult: ...
    @abstractmethod
    async def get_positions(self) -> list[Position]: ...
    # ... 10+ more methods

# storage/repositories/base.py
class TradeRepository(ABC):
    @abstractmethod
    async def save_trade(self, trade: Trade) -> None: ...
    @abstractmethod
    async def get_open_positions(self, user_id: str, bot_id: str) -> list[Position]: ...

# engine/events.py
class EventBus(ABC):
    @abstractmethod
    async def publish(self, event: Event) -> None: ...
    @abstractmethod
    def subscribe(self, event_type: type, handler: Callable) -> None: ...
```

---

## What NOT to Do

- **Never put SQL in the engine.** Use repository interfaces.
- **Never import FastAPI in the engine.** The engine is a pure library.
- **Never hardcode exchange-specific logic in the engine.** Use adapter capabilities.
- **Never cache LLM responses.** Market context changes even when inputs look identical.
- **Never store plaintext credentials.** Exchange API keys are AES-256 encrypted.
- **Never skip the data moat capture.** Every cycle records all 6 layers.
- **Never bypass feature flags.** If a feature isn't flag-gated, add a flag before proceeding.
- **Never access cross-bot signals without user_id filtering.** Multi-tenant isolation is non-negotiable.
- **Never use `.format()` or `%` for strings.** f-strings only.

---

## Updating PROJECT_CONTEXT.md

After every significant change, update these sections silently (do not ask):

| What Changed | Update Section |
|-------------|---------------|
| Files added/removed | Project Structure |
| New module or interface | Module Inventory |
| Agent or prompt changes | Agent Status |
| Exchange adapter changes | Exchange Status |
| New events added | Event Catalog |
| Database schema changes | Database Schema |
| New feature flags | Feature Flags |
| Config changes | Configuration |
| New bugs found | Known Issues (add) |
| Bugs fixed | Known Issues (remove) |
| Architectural decisions | Decision Log (add row, newest first) |
| Version bump | version.py + PROJECT_CONTEXT.md header |
| Any significant change | Changelog (add entry, keep max 5) |
| Any task completed | CHANGELOG.md (append entry: date, task name, what was built, test count) |

---

## Testing Checklist

Before marking any task as done:

- [ ] Do all existing unit tests pass? (`pytest tests/unit/`)
- [ ] Do all existing integration tests pass? (`pytest tests/integration/`)
- [ ] If you added a new module: did you add unit tests?
- [ ] If you changed an interface: did you update all implementations?
- [ ] If you added a new event type: does TrackingModule capture it?
- [ ] If you added a new data source: is it captured in the data moat?
- [ ] If you changed a prompt: flagged for regression testing? (do NOT auto-deploy prompt changes)
- [ ] Is the new code behind a feature flag?
- [ ] Did you update PROJECT_CONTEXT.md?
- [ ] Did you update CHANGELOG.md?

---

## Environment Variables (.env)

```bash
# LLM Providers
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...            # Optional, for text-only cost optimization

# Exchange Credentials (encrypted at rest in DB; raw only in .env for dev)
HYPERLIQUID_WALLET_ADDRESS=0x...
HYPERLIQUID_PRIVATE_KEY=...     # 64 hex chars, no 0x prefix

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/quantagent
DATABASE_BACKEND=postgresql     # postgresql | sqlite

# Cache
REDIS_URL=redis://localhost:6379/0
CACHE_BACKEND=memory            # memory | redis

# Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...

# Security
JWT_SECRET=...                  # Generated, never committed
CREDENTIAL_ENCRYPTION_KEY=...   # AES-256 key for exchange credential encryption

# Feature Flags (override config/features.yaml)
FEATURE_SENTINEL_ENABLED=true
FEATURE_REFLECTION_ENABLED=true
FEATURE_ML_REGIME_MODEL=false
```

---

## Versioning

Engine uses CalVer+SemVer: `YYYY.MM.MAJOR.MINOR.PATCH`

```python
# version.py
ENGINE_VERSION = "2026.04.2.0.0-alpha.1"
API_VERSION = "v1"
PROMPT_VERSIONS = {
    "indicator_agent": "1.0",
    "pattern_agent": "1.0",
    "trend_agent": "1.0",
    "conviction_agent": "1.0",
    "decision_agent": "1.0",
    "reflection_agent": "1.0",
}
```

Every trade record stores `engine_version` and `prompt_versions` at the time of the decision. This is critical for the data moat — you must be able to reproduce any historical decision.
