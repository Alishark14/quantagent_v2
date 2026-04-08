# BACKLOG.md — QuantAgent v2

> Prioritized future tasks beyond the current sprint.
> Sources: Architecture doc roadmap, brainstorming sessions, Research Agent findings.
> Updated by: Architect (Claude Chat), Research Agent (weekly), Sprint Manager (reprioritization).
> Last Updated: 2026-04-07

---

## Priority Legend

- **P0 — Critical:** Blocks other work or required for next sprint
- **P1 — High:** Next sprint candidate
- **P2 — Medium:** Within 2-3 sprints
- **P3 — Low:** Future phase, no urgency
- **Research:** Added by Research Agent, needs founder review

---

## Completed Phases

### Phase 1a: Core Engine (Weeks 1-3) — COMPLETE ✅

- [x] Project skeleton, types, config — Week 1
- [x] Event Bus (InProcessBus) — Week 1
- [x] LLM provider + Claude adapter + LangSmith tracing — Week 1
- [x] Indicator calculator (9 indicators + volatility percentile) — Week 1
- [x] SignalProducer interface + registry — Week 1
- [x] Exchange adapter ABC + Hyperliquid adapter — Week 2
- [x] OHLCV fetcher + swing detection + parent TF — Week 2
- [x] Chart generation (candlestick + trendline) — Week 2
- [x] IndicatorAgent, PatternAgent, TrendAgent — Week 2
- [x] Risk profiles + safety checks — Week 2
- [x] FlowAgent + CryptoFlowProvider — Week 3
- [x] Repository pattern + PostgreSQL + SQLite — Week 3
- [x] ConvictionAgent — Week 3
- [x] DecisionAgent — Week 3
- [x] Memory system (4 loops: cycle, reflection, cross-bot, regime) — Week 3
- [x] Pipeline orchestrator (Data → Signal → Conviction → Execution) — Week 3
- [x] First end-to-end analysis cycle on live Hyperliquid data — Week 3

### Phase 1b: Execution Layer (Week 4) — COMPLETE ✅

- [x] ExecutionCostModel (fees + slippage + spread + funding, HIP-3 deployer scaling) — Week 4
- [x] Executor (market + SL + TP1/TP2, emergency close on SL failure) — Week 4
- [x] ReflectionAgent (post-trade rule distillation, self-correcting scores) — Week 4
- [x] Tracking system (financial + decision + health + data moat capture) — Week 4
- [x] Sentinel monitor + readiness scoring + timeframe-dependent cooldown — Week 4
- [x] Position Manager (trailing stops, break-even, funding tighten — NEVER widens) — Week 4
- [x] TraderBot lifecycle + BotManager + Orphan Reaper — Week 4
- [x] First testnet trade on Hyperliquid — Week 4

### Phase 2: Platform Foundation (Week 5) — COMPLETE ✅

- [x] FastAPI web layer (app, auth, schemas, 5 route modules, DI) — Week 5
- [x] BotRunner + CLI (`python -m quantagent run`, graceful shutdown) — Week 5
- [x] Alembic migrations (async, 5 tables + 14 indexes) — Week 5
- [x] CI/CD (GitHub Actions: unit+API on push, integration on main) — Week 5
- [x] Multi-layer caching (CacheBackend ABC, MemoryCache, CacheMetrics, CacheManager) — Week 5
- [x] Caching layer hardening (thundering herd, epoch-aligned expiry, file-system chart cache, Sentinel L1 bypass / L2+L3 read) — Week 5
- [x] Version registry (version.py, prompt versions, ML model versions) — Week 5

### AI Crew Foundation — COMPLETE ✅

- [x] QA Engineer skill (daily 08:00 → #qa-reports) — Active
- [x] Sprint Manager skill (daily 09:00 → #sprint-updates) — Active
- [x] Documentation Manager skill (daily 23:30 → #sprint-updates) — Active
- [x] Research Agent skill (weekly Sun 21:00 → #sprint-updates) — Active

---

## Phase 3: Validation (Weeks 6-7)

### Week 6 — Backtesting Framework (IN PROGRESS)

- [x] Data downloader (HL API → Parquet, Polars, adapter-agnostic) — **P0** — Task 1 DONE
- [x] SimulatedExchangeAdapter + SimExecutor — **P0** — Task 2 DONE
- [x] BacktestEngine Tier 1 mechanical — **P0** — Task 3 DONE
- [x] Forward Price Path + Tier 2 replay — **P0** — Task 4 DONE
- [x] Backtest metrics + reporter (JSON + HTML) — **P0** — Task 5 DONE
- [X] Shadow mode (--shadow flag, shadow_db, adapter injection) — **P0** — Task 6
- [X] Eval framework scaffold (schema, runner, judge, output contract, 5 scenarios) — **P0** — Task 7

### Week 7 — Eval Framework + MCP Agents

- [ ] Hand-label 10 more eval scenarios across critical categories — **P0**
- [ ] Smoke test + golden master Makefile targets — **P0**
- [ ] Auto-miner scaffold (overconfident disasters + missed opportunities) — **P1**
- [ ] Quant Data Scientist MCP agent (alpha mining, FDR, out-of-sample validation, confidence decay) — **P0**
- [ ] Macro Regime Manager MCP agent (tiered schedule, DVOL, blackout windows, swarm-triggered emergency) — **P0**
- [ ] Prompt cache layering §18.6 (strict Block 1 static / Block 2 dynamic structure) — **P1**
- [ ] Run golden master eval, establish baseline scores — **P1**

---

## Phase 4: Signal Quality + Distribution (Weeks 8-9)

### Week 8 — Prompt Optimization + Signal Distribution

- [ ] Prompt iteration on weakest eval categories (trap setups, edge cases) — **P1**
- [ ] Run parameter sweeps via Tier 2 replay (ATR multiplier, conviction threshold calibration) — **P1**
- [ ] Discord signal bot (read-only — publish analysis results, no trading) — **P2**
- [ ] Telegram signal bot (mirror of Discord) — **P2**
- [ ] Prompt A/B testing framework (two prompt versions on same data, compare eval scores) — **P2**

### Week 9 — Landing Page + First Public Presence

- [ ] Landing page (product positioning, waitlist, live signal feed) — **P2**
- [ ] Blog pipeline activation (Research Agent → Blog Writer → SEO Agent) — **P2**
- [ ] QuantEval benchmark page (public scenario library, model comparison) — **P2**
- [ ] quantagent-blog.skill for Cowork — **P2**
- [ ] quantagent-seo.skill for Cowork — **P2**
- [ ] quantagent-marketing.skill for Cowork — **P2**

---

## Phase 5: Multi-Market + B2B Foundation (Weeks 10-12)

### Week 10 — Multi-Exchange Adapters

- [ ] IBKR adapter (stocks, options — implements ExchangeAdapter ABC) — **P2**
- [ ] Binance adapter (spot + futures) — **P2**
- [ ] Exchange-specific cost models for IBKR and Binance — **P2**
- [ ] Exchange adapter capabilities system (which features each adapter supports) — **P2**
- [ ] Data downloader already adapter-agnostic — just wire new adapters — **P2**

### Week 11 — B2B API + Dashboard

- [ ] B2B API layer (stateless wrappers, API key management, rate limiting, usage tracking, metering) — **P2**
- [ ] SolidJS dashboard rebuild (real-time portfolio, trade history, signal feed, system health) — **P2**
- [ ] WebSocket server for live dashboard updates — **P3**

### Week 12 — ML Model Training (v1)

- [ ] Direction model (first real training on data moat Layer 0-4, XGBoost/LightGBM) — **P3**
- [ ] Regime model (classify market regimes from indicators) — **P3**
- [ ] Anomaly detector (flag unusual market conditions) — **P3**
- [ ] Plug trained models into ML model slots (currently returning None) — **P3**
- [ ] Run QuantAgent Eval comparing ML-enhanced vs LLM-only — **P3**

---

## Phase 6: Intelligence Amplification (Weeks 13-16)

### Weeks 13-14 — Model Distillation

- [ ] Distill LLM pipeline into lightweight model (7B or smaller) — **P3**
- [ ] Run QuantAgent Eval: distilled vs Claude (teacher agreement rate) — **P3**
- [ ] Optimize for latency (target <10ms for HFT path) — **P3**
- [ ] Degradation curve analysis (accuracy vs speed vs model size) — **P3**
- [ ] Output contract already supports distilled models (model_id field) — **P3**

### Weeks 15-16 — RL Optimization + Advanced Alpha

- [ ] Reinforcement learning on historical trade outcomes — **P3**
- [ ] Alpha factor refinement from Quant Data Scientist cumulative findings — **P3**
- [ ] Cross-asset correlation signals — **P3**
- [ ] Sentiment integration (news, social, on-chain) — **P3**

---

## Phase 7: Consumer Product (Weeks 17-24)

### Weeks 17-18 — iOS App

- [ ] iOS app (Swift/SwiftUI) — portfolio view, signal alerts, trade confirmation — **P3**
- [ ] Push notifications for high-conviction setups — **P3**
- [ ] Chat interface with the engine — **P3**
- [ ] TestFlight beta → App Store submission — **P3**

### Weeks 19-20 — Consumer Platform

- [ ] User onboarding flow (7-step, paper trading first) — **P3**
- [ ] Exchange connection wizard (API key management, AES-256 encrypted) — **P3**
- [ ] Risk profile selection (conservative / balanced / aggressive) — **P3**
- [ ] Subscription billing — **P3**

### Weeks 21-24 — Data Products + Scale

- [ ] Labeled dataset marketplace (data moat Layer 0-5) — **P3**
- [ ] Alpha factor feed (subscription product) — **P3**
- [ ] Fine-tuned model API (B2B) — **P3**
- [ ] Multi-tenant scaling (Redis Event Bus, connection pooling) — **P3**
- [ ] GDPR compliance audit + data export/deletion endpoints — **P3**
- [ ] Public launch — **P3**

---

## Infrastructure & Operations

### Immediate (before Week 8)

- [ ] PostgreSQL production deployment (Hetzner or equivalent) — **P1**
- [ ] Exchange credential encryption (AES-256) — **P1**
- [ ] `_months_in_range` inclusive bug fix in ParquetDataLoader — **P2**
- [ ] Historical funding rate Parquet schema (for realistic backtests) — **P2**
- [ ] Margin/liquidation model in SimExchange (catch leverage blow-ups) — **P2**
- [ ] CLI for Tier 2 replay (`scripts/run_tier2_replay.py`) — **P2**
- [ ] CLI for eval reporter (integrate into `scripts/run_backtest.py`) — **P2**

### Later

- [ ] Redis deployment for caching + scaled Event Bus — **P3**
- [ ] JWT authentication system — **P3**
- [ ] quantagent-deploy.skill for automated deployment — **P3**
- [ ] Third-party data providers (Yahoo Finance, Alpha Vantage, Polygon.io) for non-exchange historical data — **P3**
- [ ] Deribit adapter (for DVOL data access in Macro Regime Manager) — **P3**

---

## Cowork AI Crew — Remaining Agents

- [ ] Data Analyst skill (quantagent-metrics.skill) — **P2** — Phase 3 (after first backtest results)
- [ ] Frontend Engineer skill — **P3** — Phase 5 (SolidJS dashboard)
- [ ] DevOps skill — **P3** — Phase 5 (deployment automation)
- [ ] SEO Agent skill — **P3** — Phase 4
- [ ] Marketing Agent skill — **P3** — Phase 4
- [ ] Blog Writer sub-skill — **P3** — Phase 4

---

## Research Agent Additions

> Items below are added by the Research Agent. Founder reviews during Sunday planning.

*(Pending — Research Agent runs weekly Sunday 21:00)*

---

## Known Caveats (from implementation)

> Technical debt and design limitations tracked for future resolution.

- **Flat funding rate in SimExchange** — uses configurable 1 bps default, not historical rates. Realistic backtests on long holds need historical funding Parquet schema.
- **No margin/liquidation model** — SimExchange tracks balance but not margin requirements. Won't catch leverage blow-ups.
- **R-multiple fallback** — uses `initial_balance × risk_per_trade` when per-trade SL data unavailable. Will resolve once live trades populate proper SL data in data moat.
- **Forward path `duration_candles=60` default** — sufficient for 1h trades on 1-min candles. 4h trades need `duration_candles=240+` passed explicitly.
- **Sharpe is daily bucketed** — single-day backtests return Sharpe of 0.0.
- **Drawdown duration in equity-curve points, not wall-clock** — consistent across runs but not directly time-interpretable.
- **Hyperliquid CCXT rate limits** — downloader may need tuning of `_FETCH_CHUNK` / `_RATE_LIMIT_SLEEP` for sustained mainnet downloads.
- **`_months_in_range` inclusive of end month** — `end_date=2026-02-01` tries to load February. Workaround: use `2026-01-31 23:00`.
- **Mechanical backtest bypasses ConvictionAgent/DecisionAgent** — translates mock signals directly to TradeAction. Correct for Tier 1 (validate math), not for Tier 3 (validate reasoning).
- **`result.metrics` is dict, not BacktestMetrics instance** — keeps JSON serialization trivial. Typed access via `result.metrics_obj` can be added later.

---

> **How this file is used:**
> - Sunday planning: Architect pulls P0/P1 items into next SPRINT.md
> - Research Agent: appends findings to "Research Agent Additions" section
> - Sprint Manager: references this to flag upcoming dependencies
> - Items move from here → SPRINT.md when they're ready to build
