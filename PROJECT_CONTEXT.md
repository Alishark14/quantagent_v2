# PROJECT_CONTEXT.md — QuantAgent v2

> Current state of the v2 codebase. Claude Code: read this before every task.
> Engine Version: 2026.04.2.0.0-alpha.1 (not yet released)
> Last Updated: 2026-04-05

---

## 1. Current State

**Status: PRE-BUILD.** No code has been written yet. This document describes the target state and will be updated as each module is implemented.

**What exists:**
- `ARCHITECTURE.md` — Full 36-section specification (7 Parts, 2,618 paragraphs)
- `CLAUDE.md` — Rules and conventions for Claude Code
- `PROJECT_CONTEXT.md` — This file
- `SPRINT.md` — Week 1 tasks (Phase 1a: Core Engine)

**What does NOT exist yet:**
- Any Python code
- Any tests
- Any configuration files
- Database schema
- Frontend / dashboard
- Deployment configuration

**Previous version:** v1.1.0 codebase is archived at `../quantagent-v1/` (reference only, not modified). Key lessons carried forward:
- Prompt engineering patterns (grounding context, structured output)
- Exchange adapter concept (abstract base → one file per exchange)
- Risk management math (ATR-based SL, volatility-adjusted sizing)
- Indicator computation (RSI, MACD, ROC, Stochastic, Williams %R)
- Swing detection algorithm (find_swing_lows/highs from 50 candles)
- Chart generation approach (matplotlib candlestick rendering)

---

## 2. Module Inventory

| Module | Location | Status | Dependencies | Notes |
|--------|----------|--------|-------------|-------|
| Event Bus | `engine/events.py` | IMPLEMENTED | None | EventBus ABC + InProcessBus (asyncio.gather, error isolation, metrics) |
| Types | `engine/types.py` | IMPLEMENTED | None | All 9 shared dataclasses with to_dict() |
| Config | `engine/config.py` | IMPLEMENTED | pyyaml | TradingConfig, TimeframeProfile, FeatureFlags, helpers |
| Pipeline | `engine/pipeline.py` | IMPLEMENTED | Events, all layers, memory | AnalysisPipeline: 4-stage orchestrator (Data→Signal→Conviction→Execution), event emission, cycle persistence |
| TraderBot | `engine/trader_bot.py` | IMPLEMENTED | pipeline, executor, position_manager | Ephemeral worker: analyze→execute→register→die. Error-safe (never crashes). |
| BotManager | `engine/bot_manager.py` | IMPLEMENTED | events, TraderBot | Spawns bots on SetupDetected, per-symbol concurrency limit, cleanup in finally. |
| OHLCV Fetcher | `engine/data/ohlcv.py` | IMPLEMENTED | exchanges/, indicators, swings, parent_tf | Assembles complete MarketData |
| Indicators | `engine/data/indicators.py` | IMPLEMENTED | numpy | 9 indicators + compute_all + volatility_percentile |
| Swing Detection | `engine/data/swing_detection.py` | IMPLEMENTED | numpy | Pivot detection, SL structure snapping |
| Charts | `engine/data/charts.py` | IMPLEMENTED | matplotlib, numpy | Candlestick + trendline (OLS + BB) charts, grounding header builder |
| Parent TF | `engine/data/parent_tf.py` | IMPLEMENTED | indicators | SMA trend, ADX, BB width percentile |
| FlowAgent | `engine/data/flow/` | IMPLEMENTED | exchanges/ | FlowProvider ABC, CryptoFlowProvider (funding+OI), FlowAgent aggregator |
| SignalProducer Base | `engine/signals/base.py` | IMPLEMENTED | types | ABC with name/signal_type/is_enabled/analyze/requires_vision |
| Signal Registry | `engine/signals/registry.py` | IMPLEMENTED | base | register/unregister/get_enabled/get_by_type/run_all (parallel, error-safe) |
| IndicatorAgent | `engine/signals/indicator_agent.py` | IMPLEMENTED | llm/, data, charts | LLM text agent (JSON output, grounding header) |
| PatternAgent | `engine/signals/pattern_agent.py` | IMPLEMENTED | llm/, charts | LLM vision agent (16-pattern library, JSON output) |
| TrendAgent | `engine/signals/trend_agent.py` | IMPLEMENTED | llm/, charts | LLM vision agent (OLS trendlines + BB, trend regime) |
| ML Model Slots | `engine/signals/ml/` | IMPLEMENTED | base, config | MLModelSlot + DirectionModel, RegimeModel, AnomalyDetector (all return None) |
| ConvictionAgent | `engine/conviction/agent.py` | IMPLEMENTED | llm/, signals, charts | LLM meta-evaluator: conviction 0-1, regime, fact/subj weighting, parse-safe |
| DecisionAgent | `engine/execution/agent.py` | IMPLEMENTED | llm/, conviction, risk_profiles, safety_checks | LLM action selection: 7 actions, SL/TP, sizing, safety overrides, parse-safe |
| Executor | `engine/execution/executor.py` | IMPLEMENTED | exchanges/, events | Market orders + SL + TP1/TP2, pyramid, close_all, SL failure emergency close, event emission |
| Cost Model ABC | `engine/execution/cost_model.py` | IMPLEMENTED | None | ExecutionCost, PositionSizeResult, abstract ExecutionCostModel with compute_total_cost, fee_adjusted_rr, cost_aware_position_size, is_trade_viable |
| HL Cost Model | `engine/execution/cost_models/hyperliquid.py` | IMPLEMENTED | adapter | HyperliquidCostModel: fee tiers, HIP-3 deployer scaling, growth mode, orderbook slippage, funding costs |
| Generic Cost Model | `engine/execution/cost_models/generic.py` | IMPLEMENTED | None | Conservative defaults (0.1% taker, 0.05% slippage) for unknown exchanges |
| Risk Profiles | `engine/execution/risk_profiles.py` | IMPLEMENTED | config | SL/TP (ATR + swing snap), position sizing |
| Safety Checks | `engine/execution/safety_checks.py` | IMPLEMENTED | types | 5 mechanical checks + SafetyCheckResult |
| ReflectionAgent | `engine/reflection/agent.py` | IMPLEMENTED | llm/, ReflectionRules, events | Post-trade rule distillation, TradeClosed handler, RuleGenerated event |
| Cycle Memory | `engine/memory/cycle_memory.py` | IMPLEMENTED | CycleRepository | Loop 1: save/retrieve recent cycles, prompt formatting |
| Reflection Rules | `engine/memory/reflection_rules.py` | IMPLEMENTED | RuleRepository | Loop 2: rules with self-correcting counters, auto-deactivation |
| Cross-Bot | `engine/memory/cross_bot.py` | IMPLEMENTED | CrossBotRepository | Loop 3: user_id-scoped signal sharing, prompt formatting |
| Regime History | `engine/memory/regime_history.py` | IMPLEMENTED | None (in-memory) | Loop 4: ring buffer, transition detection, streak counting |
| Memory Context | `engine/memory/__init__.py` | IMPLEMENTED | all 4 loops | build_memory_context() assembles all loops for prompt injection |
| Sentinel Monitor | `sentinel/monitor.py` | IMPLEMENTED | exchanges/, events, conditions | Poll-based monitor, cooldown, daily budget, SetupDetected emission |
| Sentinel Conditions | `sentinel/conditions.py` | IMPLEMENTED | indicators | ReadinessScorer: 5 weighted conditions (RSI/level/volume/flow/MACD), 0-1 score |
| Position Manager | `sentinel/position_manager.py` | IMPLEMENTED | adapter, events | Only tightens SL: trailing, BE after TP1, funding tighten. Calls adapter.modify_sl() + emits PositionUpdated. |
| Orphan Reaper | `sentinel/reaper.py` | IMPLEMENTED | adapter, position_manager | Detects orphans, emergency SL at 2x ATR for unprotected positions |
| Exchange Base | `exchanges/base.py` | IMPLEMENTED | None | ExchangeAdapter ABC (15 abstract + 2 optional methods) |
| Exchange Factory | `exchanges/factory.py` | IMPLEMENTED | base | Singleton-cached factory with register/get_adapter/reset |
| Hyperliquid Adapter | `exchanges/hyperliquid.py` | IMPLEMENTED | CCXT | Native SL/TP, HIP-3, symbol conversion, flow data, testnet support (env-driven) |
| LLM Base | `llm/base.py` | IMPLEMENTED | None | LLMProvider ABC + LLMResponse dataclass |
| Claude Provider | `llm/claude.py` | IMPLEMENTED | anthropic SDK, langsmith | Retry, cost calc, prompt caching, vision, LangSmith tracing (conditional) |
| Repository Base | `storage/repositories/base.py` | IMPLEMENTED | None | 5 ABCs: Trade, Cycle, Rule, Bot, CrossBot |
| SQLite Repo | `storage/repositories/sqlite.py` | IMPLEMENTED | aiosqlite | Local dev fallback, all 5 repos + container |
| PostgreSQL Repo | `storage/repositories/postgres.py` | IMPLEMENTED | asyncpg | Standard backend, all 5 repos + pool container |
| Repo Factory | `storage/repositories/__init__.py` | IMPLEMENTED | base, sqlite, postgres | get_repositories() factory, env-driven backend |
| Alembic Migrations | `alembic/` | IMPLEMENTED | alembic, sqlalchemy, asyncpg | async env.py, 001_initial (5 tables + 14 indexes), script.py.mako |
| Seed Script | `scripts/seed_dev.py` | IMPLEMENTED | repos | 3 dev bots + 2 reflection rules, idempotent |
| Cache Base | `storage/cache/base.py` | NOT BUILT | None | Abstract cache |
| Memory Cache | `storage/cache/memory.py` | NOT BUILT | cachetools | Single-server |
| Tracking | `tracking/` | IMPLEMENTED | events | TrackingModule + FinancialTracker, DecisionTracker, HealthTracker. _safe() wrappers. |
| Data Moat | `tracking/data_moat.py` | IMPLEMENTED | events | DataMoatCapture: 6-layer capture (L0-L5), links cycles and trades |
| FastAPI App | `api/app.py` | IMPLEMENTED | FastAPI | Lifespan (startup/shutdown), router includes |
| API Auth | `api/auth.py` | IMPLEMENTED | None | X-API-Key header auth, SHA-256 user_id derivation |
| API Schemas | `api/schemas.py` | IMPLEMENTED | Pydantic | Request/response models for all endpoints |
| API Dependencies | `api/dependencies.py` | IMPLEMENTED | FastAPI | DI providers for repos + health tracker |
| Bot Routes | `api/routes/bots.py` | IMPLEMENTED | FastAPI | CRUD + /analyze (manual cycle trigger) |
| Trade Routes | `api/routes/trades.py` | IMPLEMENTED | FastAPI | List + detail with filters |
| Health Route | `api/routes/health.py` | IMPLEMENTED | FastAPI | System health snapshot (no auth required) |
| Position Routes | `api/routes/positions.py` | IMPLEMENTED | FastAPI | Open positions across all bots |
| Rule Routes | `api/routes/rules.py` | IMPLEMENTED | FastAPI | Reflection rules by symbol/timeframe |
| BotRunner | `quantagent/runner.py` | IMPLEMENTED | sentinel, bot_manager, repos | Production service: sentinel per symbol, scheduled fallbacks, auto-restart with backoff, graceful shutdown |
| CLI | `quantagent/main.py` | IMPLEMENTED | runner, api, uvicorn | `python -m quantagent run` starts BotRunner + FastAPI together, signal-based graceful shutdown |
| CI/CD | `.github/workflows/ci.yml` | IMPLEMENTED | GitHub Actions | Unit+API tests on push, integration on main, import violation checks |
| Distribution | `distribution/` | NOT BUILT | — | Discord, Telegram |
| MCP Agents | `mcp/` | NOT BUILT | — | Overnight Quant, Macro |

---

## 3. Project Structure (Actual Files)

```
quantagent-v2/
├── ARCHITECTURE.md
├── PROJECT_CONTEXT.md           ← this file
├── CLAUDE.md
├── SPRINT.md
├── pyproject.toml               # Project dependencies and metadata
├── version.py                   # CalVer+SemVer, model costs, prompt versions
├── alembic.ini                  # Alembic config (reads DATABASE_URL from env)
├── .gitignore
│
├── alembic/                     # Database migrations (Alembic)
│   ├── env.py                   # Async migration env (asyncpg)
│   ├── script.py.mako           # Migration template
│   └── versions/
│       └── 001_initial.py       # Initial schema: 5 tables + 14 indexes
│
├── quantagent/                  # CLI entry point package
│   ├── __init__.py
│   ├── main.py                  # CLI: run, migrate, seed commands
│   └── runner.py                # BotRunner: sentinels, scheduled loops, auto-restart
│
├── engine/                      # THE CORE — pure library
│   ├── __init__.py
│   ├── pipeline.py
│   ├── events.py
│   ├── types.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ohlcv.py
│   │   ├── indicators.py
│   │   ├── swing_detection.py
│   │   ├── charts.py
│   │   ├── parent_tf.py
│   │   └── flow/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── crypto.py
│   │       ├── options.py
│   │       ├── equity.py
│   │       └── forex.py
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── indicator_agent.py
│   │   ├── pattern_agent.py
│   │   ├── trend_agent.py
│   │   ├── prompts/
│   │   │   ├── __init__.py
│   │   │   ├── indicator_v1.py
│   │   │   ├── pattern_v1.py
│   │   │   └── trend_v1.py
│   │   └── ml/
│   │       ├── __init__.py
│   │       ├── direction.py
│   │       ├── regime.py
│   │       └── anomaly.py
│   ├── conviction/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── prompts/
│   │       ├── __init__.py
│   │       └── conviction_v1.py
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── executor.py
│   │   ├── risk_profiles.py
│   │   ├── safety_checks.py
│   │   └── prompts/
│   │       ├── __init__.py
│   │       └── decision_v1.py
│   ├── reflection/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── prompts/
│   │       ├── __init__.py
│   │       └── reflection_v1.py
│   └── memory/
│       ├── __init__.py
│       ├── cycle_memory.py
│       ├── reflection_rules.py
│       ├── cross_bot.py
│       └── regime_history.py
│
├── sentinel/
│   ├── __init__.py
│   ├── monitor.py
│   ├── conditions.py
│   ├── position_manager.py
│   ├── reaper.py
│   └── config.py
│
├── exchanges/
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   ├── hyperliquid.py
│   ├── dydx.py
│   └── deribit.py
│
├── llm/
│   ├── __init__.py
│   ├── base.py
│   ├── claude.py
│   ├── groq.py
│   └── cache.py
│
├── storage/
│   ├── __init__.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── postgres.py
│   │   └── sqlite.py
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── memory.py
│   │   └── redis.py
│   └── object_store/
│       ├── __init__.py
│       ├── base.py
│       ├── local.py
│       └── s3.py
│
├── tracking/
│   ├── __init__.py
│   ├── financial.py
│   ├── decision.py
│   ├── health.py
│   ├── data_moat.py
│   └── audit.py
│
├── distribution/
│   ├── __init__.py
│   ├── base.py
│   ├── discord.py
│   ├── telegram.py
│   └── formatter.py
│
├── mcp/
│   ├── __init__.py
│   ├── overnight_quant.py
│   └── macro_regime.py
│
├── api/
│   ├── __init__.py
│   ├── app.py                   # FastAPI app factory + lifespan
│   ├── auth.py                  # X-API-Key auth, user_id derivation
│   ├── schemas.py               # Pydantic request/response models
│   ├── dependencies.py          # DI: repos, health tracker
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── bots.py              # CRUD + /analyze
│   │   ├── trades.py            # Trade listing + detail
│   │   ├── health.py            # System health (no auth)
│   │   ├── positions.py         # Open positions
│   │   ├── rules.py             # Reflection rules
│   │   ├── trading.py           # B2B API (placeholder)
│   │   └── websocket.py         # WebSocket streams (placeholder)
│   └── middleware/
│       ├── __init__.py
│       ├── rate_limit.py
│       └── tenant.py
│
├── dashboard/                   # (empty until Phase 4)
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   └── test_runner.py       # 26 tests: start/stop, add/remove, auto-restart, scheduled, CLI
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_migrations.py   # 30 tests: structure, schema parity, pool, seed, CLI
│   ├── api/
│   │   ├── __init__.py
│   │   └── test_endpoints.py    # 38 tests: auth, CRUD, trades, health, positions, rules
│   ├── adapters/
│   │   └── __init__.py
│   └── fixtures/
│       ├── __init__.py
│       ├── sample_ohlcv.json
│       └── mock_llm_responses.json
│
├── scripts/
│   ├── __init__.py
│   ├── migrate_db.py
│   ├── seed_dev.py              # Seed dev DB: 3 bots + 2 rules
│   ├── collect_metrics.py
│   └── health_check.py
│
└── config/
    ├── features.yaml
    ├── sentinel.yaml
    └── profiles.yaml
```

> Claude Code: update this tree after creating files.

---

## 4. Event Catalog

Event dataclasses and EventBus (InProcessBus) implemented in `engine/events.py`.

| Event | Payload | Emitted By | Consumed By | Status |
|-------|---------|-----------|-------------|--------|
| DataReady | MarketData | DataModule | SignalModule | IMPLEMENTED |
| SignalsReady | list[SignalOutput] | SignalModule | ConvictionModule | IMPLEMENTED |
| ConvictionScored | ConvictionOutput | ConvictionModule | ExecutionModule | IMPLEMENTED |
| TradeOpened | TradeAction, OrderResult | ExecutionModule | TrackingModule, DistributionModule | IMPLEMENTED |
| TradeClosed | symbol, pnl, exit_reason | ExecutionModule | TrackingModule, DistributionModule, ReflectionModule | IMPLEMENTED |
| PositionUpdated | symbol, Position | SentinelModule | TrackingModule | IMPLEMENTED |
| SetupDetected | symbol, readiness, conditions | SentinelModule | Pipeline (spawn TraderBot) | IMPLEMENTED |
| RuleGenerated | rule dict | ReflectionModule | TrackingModule | IMPLEMENTED |
| FactorsUpdated | filepath | MCPModule | (file-based, read at cycle start) | IMPLEMENTED |
| MacroUpdated | filepath | MCPModule | (file-based, read at cycle start) | IMPLEMENTED |
| CycleCompleted | symbol, action, conviction | Pipeline | TrackingModule | IMPLEMENTED |

> Claude Code: mark events as IMPLEMENTED when the Event Bus handles them.

---

## 5. Agent Status

| Agent | Prompt Version | Status | LLM Provider | Notes |
|-------|---------------|--------|-------------|-------|
| IndicatorAgent | v1.0 | IMPLEMENTED | Claude (text) | Grounded with indicator summary, JSON output, parse-safe |
| PatternAgent | v1.0 | IMPLEMENTED | Claude (vision) | 16-pattern library, grounding emphasis, parse-safe |
| TrendAgent | v1.0 | IMPLEMENTED | Claude (vision) | OLS trendlines + BB, trend regime classification, parse-safe |
| ConvictionAgent | v1.0 | IMPLEMENTED | Claude (text) | Fact/subjective labeling, regime classification, 0-1 scoring, parse-safe (SKIP default) |
| DecisionAgent | v1.0 | IMPLEMENTED | Claude (text) | 7 actions, conviction-tier sizing, SL/TP via risk_profiles, safety check enforcement, parse-safe (HOLD/SKIP default) |
| ReflectionAgent | v1.0 | IMPLEMENTED | Claude (text) | Async post-trade, distills ONE rule per trade, saves to repo, emits RuleGenerated, TradeClosed handler |

> Claude Code: update status to IMPLEMENTED when agent is working. Update prompt version when prompts change.

---

## 6. Exchange Status

| Exchange | Adapter File | Status | Capabilities Declared | Notes |
|----------|-------------|--------|----------------------|-------|
| Hyperliquid | `exchanges/hyperliquid.py` | IMPLEMENTED | native_sl_tp, short, funding, OI, 50x lev | Primary. 35 mapped symbols (perp + HIP-3). Ported from v1. |
| dYdX v4 | `exchanges/dydx.py` | NOT BUILT | — | Secondary. IOC orders, 4 CCXT patches needed. |
| Deribit | `exchanges/deribit.py` | NOT BUILT | — | Legacy. Options data source. |

> Claude Code: update status when adapters are implemented.

---

## 7. Database Schema

**Backend:** PostgreSQL (standard) / SQLite (local dev fallback)
**ORM/Driver:** asyncpg for PostgreSQL, aiosqlite for SQLite
**Migrations:** Alembic (async via asyncpg+sqlalchemy). Run: `python -m quantagent migrate`

**Tables:**

| Table | Purpose | Status |
|-------|---------|--------|
| users | User accounts, auth | NOT CREATED |
| bots | Bot configuration per user (id, user_id, symbol, timeframe, exchange, status, config_json, created_at, last_health) | CREATED |
| trades | Complete trade lifecycle (id, user_id, bot_id, symbol, timeframe, direction, entry/exit price, size, pnl, r_multiple, entry/exit time, exit_reason, conviction_score, engine_version, status) | CREATED |
| cycles | Every analysis cycle (id, bot_id, symbol, timeframe, timestamp, indicators_json, signals_json, conviction_json, action, conviction_score) | CREATED |
| cycle_costs | API costs per cycle | NOT CREATED |
| rules | Reflection rulebook (id, symbol, timeframe, rule_text, score, active, created_at) | CREATED |
| cross_bot_signals | Shared intelligence (id, user_id, symbol, direction, conviction, bot_id, timestamp) | CREATED |
| regime_history | Regime ring buffer per bot | NOT CREATED |
| bot_health | Health snapshots (merged into bots.last_health) | MERGED |
| api_health | API latency tracking | NOT CREATED |
| api_usage | B2B customer usage and billing | NOT CREATED |
| training_examples | Golden record linking all 6 data moat layers | NOT CREATED |

> Claude Code: update status when tables are created via migrations.

---

## 8. Feature Flags

Target flags from ARCHITECTURE.md. All default to `false` until the feature is implemented and tested.

```yaml
# config/features.yaml (not yet created)
sentinel_enabled: false
reflection_enabled: false
cross_bot_enabled: false
distribution_discord: false
distribution_telegram: false
ml_regime_model: false
ml_direction_model: false
ml_anomaly_detector: false
external_quanthq: false
external_guavy: false
external_sentimentrader: false
mcp_overnight_quant: false
mcp_macro_manager: false
chat_enabled: false
paper_trading_mode: true          # Default ON for safety
```

> Claude Code: flip flags to `true` when features are implemented and tested.

---

## 9. Configuration

### .env — Secrets Only (gitignored)

```bash
ANTHROPIC_API_KEY=               # Required
GROQ_API_KEY=                    # Optional
HYPERLIQUID_WALLET_ADDRESS=      # Required for trading
HYPERLIQUID_PRIVATE_KEY=         # Required for trading
DATABASE_URL=                    # Required
DATABASE_BACKEND=postgresql      # postgresql | sqlite
REDIS_URL=                       # Optional (memory cache if absent)
CACHE_BACKEND=memory             # memory | redis
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
JWT_SECRET=
CREDENTIAL_ENCRYPTION_KEY=
```

### Trading Config (per-bot, from API or config)

| Variable | Default | Notes |
|----------|---------|-------|
| SYMBOL | BTC-USDC | Any exchange-supported symbol |
| TIMEFRAME | 1h | 15m, 30m, 1h, 4h, 1d |
| EXCHANGE | hyperliquid | Exchange adapter name |
| ACCOUNT_BALANCE | 0 | 0 = fetch from exchange |
| ATR_LENGTH | 14 | ATR lookback period |
| FORECAST_CANDLES | 3 | Dynamic per regime (see ARCHITECTURE.md sec 11) |
| MAX_CONCURRENT_POSITIONS | 1 | Per-symbol per-bot limit |
| MAX_POSITION_PCT | 1.0 | Max % of budget per trade |
| CONVICTION_THRESHOLD | 0.5 | Minimum conviction to trade |

### Timeframe Profiles (base, modified by regime)

| TF | Candles | ATR Mult | RR Min | RR Max |
|----|---------|----------|--------|--------|
| 15m | 100 | 2.5 | 0.8 | 1.2 |
| 30m | 100 | 2.0 | 1.0 | 1.5 |
| 1h | 150 | 1.5 | 1.5 | 2.0 |
| 4h | 150 | 1.0 | 3.0 | 5.0 |
| 1d | 200 | 1.0 | 3.0 | 5.0 |

---

## 10. Dependencies

Target dependencies (not yet installed):

```toml
# pyproject.toml [project.dependencies]
python = ">=3.12"
anthropic = ">=0.40"           # Claude API
langchain-anthropic = "*"      # LangChain + LangSmith tracing
langgraph = "*"                # Agent orchestration (evaluate if still needed with Event Bus)
ccxt = ">=4.0"                 # Exchange abstraction
fastapi = ">=0.110"            # Web API
uvicorn = ">=0.30"             # ASGI server
asyncpg = ">=0.30"             # PostgreSQL async driver
aiosqlite = ">=0.20"           # SQLite async fallback
alembic = ">=1.13"             # Database migrations
redis = ">=5.0"                # Cache + Event Bus (scaled)
httpx = ">=0.27"               # Async HTTP for external APIs
pydantic = ">=2.5"             # Data validation
matplotlib = ">=3.8"           # Chart generation
numpy = ">=1.26"               # Numerical computation
pandas = ">=2.1"               # Data manipulation
cachetools = ">=5.3"           # In-memory caching
pyjwt = ">=2.8"                # JWT authentication
cryptography = ">=42.0"        # AES-256 credential encryption
apscheduler = ">=3.10"         # Scheduled tasks (Sentinel fallback timer)
websockets = ">=12.0"          # WebSocket connections to exchanges
pytest = ">=8.0"               # Testing
pytest-asyncio = ">=0.23"      # Async test support
```

> Claude Code: update this when dependencies change.

---

## 11. Known Issues

No issues yet (no code exists).

> Claude Code: add issues as they are discovered. Remove when fixed.

---

## 12. Decision Log

> Architectural decisions with reasoning. Newest first.

| Date | Decision | Reasoning |
|------|----------|-----------|
| 2026-04-05 | v2 is a complete rebuild, not a refactor of v1 | v1 has fundamental couplings (LangGraph state, SQLite baked in, React dashboard) that would fight every v2 concept. Clean start is faster than incremental migration. |
| 2026-04-05 | Event Bus over direct imports | Modules must be swappable without code changes. Direct imports create hidden dependencies. Events make coupling explicit and breakable. |
| 2026-04-05 | PostgreSQL standard, SQLite dev-only fallback | Multi-tenant, concurrent writes, connection pooling, row-level security required for the platform. SQLite can't do any of these. |
| 2026-04-05 | Engine as pure library, web layer separate | Engine must work identically for consumer platform (FastAPI wrapper) and B2B API (stateless wrapper). No web framework in the core. |
| 2026-04-05 | CalVer+SemVer versioning | CalVer communicates freshness (critical for signal product). SemVer communicates compatibility (critical for B2B API). Both needed. |

> Claude Code: add decisions when non-obvious architectural choices are made.

---

## 13. Version History

| Version | Date | Codename | Key Changes |
|---------|------|----------|-------------|
| 2026.04.2.0.0-alpha.1 | — | Genesis | Target version. Not yet released. |

> Claude Code: add rows when version bumps occur.

---

## 14. Changelog (Last 5 Updates)

> Claude Code: update after every significant task. Keep only last 5. Newest first.

- **2026-04-06:** CI/CD + README — GitHub Actions workflow (unit+API on push, integration on main, import violation checks), README.md with CI badge + setup instructions. 851 tests.
- **2026-04-06:** Alembic migrations — async env.py, 001_initial (5 tables + 14 indexes), PostgreSQL pool lifecycle + health_check(), CLI migrate/seed commands, seed_dev.py. 30 new tests. 851 tests.
- **2026-04-06:** BotRunner + CLI — Production service with sentinel management, scheduled fallbacks, auto-restart with exponential backoff, graceful shutdown. 26 new tests. 821 tests.
- **2026-04-06:** FastAPI web layer — app, auth, schemas, 5 route modules, dependency injection. 38 new tests. 795 tests.
- **2026-04-06:** ARCHITECTURE.md section 7.5 — Execution Cost Model (Zero-Leakage Execution), 8 subsections. 757 tests.

> Full history in CHANGELOG.md.

---

## 15. What's Next

Current sprint: **Phase 1a — Core Engine Redesign (Week 1)**

See `SPRINT.md` for detailed tasks with paste-ready Claude Code instructions.

Priority order:
1. Project skeleton (pyproject.toml, folder structure, empty __init__.py files)
2. Event Bus (`engine/events.py`)
3. Core types (`engine/types.py`)
4. Config system (`engine/config.py` + `config/features.yaml`)
5. LLM provider abstraction (`llm/base.py` + `llm/claude.py`)
6. Indicator calculator (`engine/data/indicators.py` + tests)
7. SignalProducer interface (`engine/signals/base.py`)

> Claude Code: this section should always reflect the immediate next priorities.
