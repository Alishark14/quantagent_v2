# CHANGELOG.md — QuantAgent v2

> All notable changes to this project.
> Format: date, what changed, test count.
> Updated by: Claude Code after each task.

---

## Week 3 — Conviction, Decision, Memory, Pipeline (April 6, 2026)

### 2026-04-06
- **CI/CD + README** — `.github/workflows/ci.yml`: 3 jobs (lint import violations, unit+API tests on every push, integration tests on main merge). `README.md` with CI badge, architecture diagram, setup instructions, API reference. **(851 total)**
- **Alembic Migrations** — Moved from raw CREATE TABLE to proper Alembic migrations. `alembic.ini` (reads DATABASE_URL from env), `alembic/env.py` (async asyncpg support), `alembic/versions/001_initial.py` (5 tables: bots, trades, cycles, rules, cross_bot_signals + 14 indexes). Updated `storage/repositories/postgres.py`: configurable pool sizing (min/max), cached repo instances, `health_check()` (pool stats + PG version). CLI: `python -m quantagent migrate` (Alembic upgrade) + `python -m quantagent seed` (dev data). `scripts/seed_dev.py`: 3 bots + 2 reflection rules, idempotent. 30 new tests. **(851 total)**
- **BotRunner + CLI** — Production service `quantagent/runner.py`: loads bots from DB, starts SentinelMonitor per unique symbol, scheduled fallback analysis loops, auto-restart with exponential backoff (5s→5min), graceful shutdown (stop sentinels, cancel tasks, wait for active bots). Updated `quantagent/main.py`: `python -m quantagent run` starts BotRunner + FastAPI (uvicorn) together, SIGINT/SIGTERM graceful shutdown. 26 new tests. **(821 total)**
- **FastAPI Web Layer** — Full API layer wrapping the engine. 8 new files: `api/app.py` (lifespan), `api/auth.py` (X-API-Key header auth, SHA-256 user_id), `api/schemas.py` (Pydantic models), `api/dependencies.py` (DI), `api/routes/bots.py` (CRUD + /analyze), `api/routes/trades.py` (list + detail with filters), `api/routes/health.py` (health snapshot, no auth), `api/routes/positions.py` (open positions), `api/routes/rules.py` (reflection rules). 38 new tests with in-memory mock repos. **(795 total)**
- **ARCHITECTURE.md section 7.5** — New "Execution Cost Model (Zero-Leakage Execution)" section (7.5.1-7.5.8). Covers exchange-agnostic design, dynamic asset metadata, cost components, iterative position sizing, fee-adjusted R:R, safety check #6, dual execution mode, expected hold duration tables. **(757 total)**
- **ExecutionCostModel** — CRITICAL SAFETY: cost-aware execution. 9 files. ABC + HyperliquidCostModel (HIP-3 deployer fees, orderbook slippage, funding) + GenericCostModel. Safety check #6: cost viability gate. Cost-aware position sizing (iterative solver). COST_AWARE_RR grounding line. 41 new tests. **(757 total)**
- **run_trade.py** — First real trade script. Full TraderBot lifecycle: analyze + execute on exchange. `--testnet` (default), `--live` with confirmation, `--dry-run`, `--verbose`. **(716 total)**
- **Ephemeral Swarm Architecture** — `TraderBot` (pipeline+execute+register, never crashes), `BotManager` (SetupDetected subscription, per-symbol concurrency, finally cleanup), `OrphanReaper` (exchange vs registry cross-check, emergency SL at 2x ATR). 22 new tests. **(716 total)**
- **ARCHITECTURE.md updated** — Section 8.3: timeframe-dependent cooldown. Section 19.4: LangSmith tracing. Section 16.2.1: testnet. Section 26.1: paper trading via testnet. **(692 total)**
- **Sentinel Timeframe-Dependent Cooldown** — Replaced fixed 15-min cooldown with timeframe-scaled values: 15m=900s, 30m=1800s, 1h=3600s, 4h=14400s, 1d=86400s. Daily budgets also scaled: 15m=16, 30m=12, 1h=8, 4h=4, 1d=2. New `sentinel/config.py` with `get_sentinel_cooldown()` and `get_sentinel_daily_budget()`. SentinelMonitor defaults from timeframe, explicit overrides still work. 15 new tests. **(692 total)**
- **PositionManager v2** — Added `adapter.modify_sl()` direct call when SL changes (previously only emitted event). Tests now verify modify_sl is called on tighten and NOT called when adjustment would widen. Added retrace tests (price goes back -> SL stays). 33 tests (was 31). **(694 total)**
- **PositionManager** — Sentinel SL adjustment. Only tightens: trailing (1 ATR trail), break-even (after TP1), funding tighten (0.3 ATR). `_is_tighter()` guard. Emits PositionUpdated. **(677 total)**
- **Sentinel System** — `ReadinessScorer` (5 weighted conditions), `SentinelMonitor` (poll-based, cooldown, daily budget, SetupDetected). 24 new tests. **(646 total)**
- **TrackingModule** — Pure observer subscribing to ALL events. 4 trackers: FinancialTracker, DecisionTracker, HealthTracker, DataMoatCapture (6-layer). _safe() wrappers. 28 new tests. **(622 total)**
- **ReflectionAgent** — Async post-trade analyst. Distills ONE rule per trade, saves to repo, emits RuleGenerated. Null-safe. TradeClosed handler factory. Prompt v1.0. 17 new tests. **(594 total)**
- **Executor** — Bridge between DecisionAgent and exchange orders. LONG/SHORT: market order + SL + TP1 (50%) + TP2 (50%). ADD_LONG/ADD_SHORT: pyramid market order + SL adjustment via modify_sl. CLOSE_ALL: cancel all orders + market close. SL placement failure triggers emergency close (no unprotected positions). USD-to-base size conversion. Emits TradeOpened/TradeClosed events. 24 new tests. **(577 total)**
- **LangSmith Tracing** — ClaudeProvider traces all LLM calls to LangSmith when `LANGCHAIN_TRACING_V2=true`. Traces include agent_name, truncated prompts, response content, token counts, cost, model, temperature, latency. Conditional — silently skips if langsmith not installed or API key missing. Added `langsmith>=0.1` to pyproject.toml, `LANGCHAIN_PROJECT=quantagent-v2` to .env. **(553 total)**
- **Hyperliquid Testnet Support** — HyperliquidAdapter reads `HYPERLIQUID_TESTNET` env var; `testnet=True` via env or constructor triggers `set_sandbox_mode(True)` pointing to `api.hyperliquid-testnet.xyz`. Adapter now auto-reads wallet/key from env. `run_cycle.py --testnet` flag. Verified testnet OHLCV fetch works. **(553 total)**
- **Debug & Fix: Signal Agents Silent Failure** — Root cause: `.env` parser didn't strip inline `#` comments, so API key included trailing text with em-dash (U+2014) causing `UnicodeEncodeError` in httpx headers. Second issue: agent feature flags missing from `features.yaml`. Fixes: (1) `.env` parser strips inline comments, (2) agent flags added as `true`, (3) `SignalRegistry.run_all()` verbose logging. Created `scripts/debug_agents.py`. All 3 agents verified live. **(553 total)**
- **run_cycle.py** — End-to-end manual verification script. Real exchange data, real LLM calls, full pipeline. .env loading, event logging, verbose mode. **(553 total)**
- **AnalysisPipeline** — 4-stage orchestrator (Data→Signal→Conviction→Execution). Event emission at each stage, cycle persistence, cross-bot signal publishing, regime history update. Graceful degradation on agent/parse failures. 14 integration tests. **(553 total)**
- **Memory System** — All 4 loops: CycleMemory (save/retrieve/format), ReflectionRules (self-correcting counters, auto-deactivation), CrossBotSignals (user_id-scoped), RegimeHistory (ring buffer, transition/streak detection). build_memory_context() assembles all for prompt injection. 34 new tests. **(500 total)**
- **DecisionAgent** — LLM action selection: 7 actions (LONG, SHORT, ADD_LONG, ADD_SHORT, CLOSE_ALL, HOLD, SKIP). Conviction-tier sizing (1.0x/1.15x/1.3x), SL/TP via risk_profiles, pyramid at 50% size, safety check enforcement (overrides action), parse-safe (HOLD if position, SKIP if not). Prompt v1.0. 29 new tests. **(466 total)**
- **ConvictionAgent** — LLM meta-evaluator: continuous conviction score 0-1, regime classification (5 types: TRENDING_UP/DOWN, RANGING, HIGH_VOLATILITY, BREAKOUT), fact/subjective weighting, contradiction analysis, parse-safe (score=0.0, SKIP on failure). Prompt v1.0. 27 new tests. **(437 total)**
- **Repository Pattern** — 5 abstract interfaces (Trade, Cycle, Rule, Bot, CrossBot). SQLite backend (aiosqlite) for dev, PostgreSQL backend (asyncpg + connection pool) for production. Factory with env-driven backend selection. Multi-tenant isolation on cross-bot signals. 31 new tests. **(410 total)**

---

## Week 2 — Signal Agents, Exchange, Data Layer (April 5-6, 2026)

### 2026-04-06
- **FlowAgent** — FlowProvider ABC, CryptoFlowProvider (funding rate thresholds + OI), FlowAgent aggregator with multi-provider merge and data_richness classification. 22 new tests. **(418 total)**
- **Risk Profiles + Safety Checks** — compute_sl_tp (ATR-based SL, swing snapping, TP1/TP2), compute_position_size (risk/SL-distance, capped). 5 mechanical safety checks (conviction floor, daily loss, pyramid gate, position limit, SL validation). 51 new tests. **(396 total)**
- **TrendAgent** — Second vision agent. Generates trendline chart (OLS + BB), analyzes trend direction/strength/reversal with trend_regime classification. 22 new tests. **(345 total)**
- **PatternAgent** — First vision agent. Generates candlestick chart PNG, sends to Claude Vision with 16-pattern library + grounding emphasis. 22 new tests. **(323 total)**
- **IndicatorAgent** — First LLM signal agent (text-only). Structured JSON output, grounding header injection, parse-safe (None on failure). Prompt v1.0. 27 new tests. **(301 total)**
- **Chart Generation** — Dark-themed candlestick chart (volume bars, swing levels), trendline chart (full + short OLS, Bollinger Bands shaded), grounding header builder. 23 new tests. **(274 total)**
- **OHLCV Fetcher** — Assembles MarketData from adapter + indicators + swings + parent TF. 9 new tests. **(251 total)**
- **Hyperliquid Adapter** — Full ExchangeAdapter impl with native SL/TP, HIP-3 symbol conversion (35 symbols), positions, orders, flow data. Ported from v1. 36 new tests. **(242 total)**

### 2026-04-05
- **Exchange Adapter Base** — ExchangeAdapter ABC (15 abstract + 2 optional methods), ExchangeFactory (singleton cache). 10 new tests. **(206 total)**

---

## Week 1 — Core Engine Foundation (April 5, 2026)

### 2026-04-05
- **SignalProducer Interface** — SignalProducer ABC, SignalRegistry (parallel run_all, error isolation), MLModelSlot base + 3 placeholder slots (direction, regime, anomaly). 20 new tests. **(196 total)**
- **Indicator Calculator** — 9 indicators + compute_all + volatility_percentile. Swing pivot detection + SL snapping. Parent TF context (SMA trend, ADX, BB width percentile). 61 new tests. **(176 total)**
- **LLM Provider** — LLMProvider ABC, ClaudeProvider (retry, cost calc, prompt caching, vision), PromptCache TTL tracker. 16 new tests. **(115 total)**
- **Config System** — TradingConfig (env-driven), TimeframeProfile + DEFAULT_PROFILES, get_dynamic_profile (regime+volatility), FeatureFlags (YAML+env override), helper functions. 42 new tests. **(99 total)**
- **Event Bus** — EventBus ABC + InProcessBus with asyncio.gather, error isolation, metrics. 15 new tests. **(57 total)**
- **Core Types** — 9 dataclasses in engine/types.py, 11 event dataclasses in engine/events.py. 42 unit tests. **(42 total)**
- **Project Skeleton** — pyproject.toml, version.py, .gitignore, full folder structure (90+ files), config YAMLs, quantagent entry point. **(0 tests)**

---

## Project Init

### 2026-04-05
- Project initialized. ARCHITECTURE.md (36 sections, 7 Parts), CLAUDE.md, PROJECT_CONTEXT.md, SPRINT.md created. No code.
