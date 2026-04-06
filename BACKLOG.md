# BACKLOG.md — QuantAgent v2

> Prioritized future tasks beyond the current sprint.
> Sources: Architecture doc roadmap, brainstorming sessions, Research Agent findings.
> Updated by: Architect (this chat), Research Agent (weekly), Sprint Manager (reprioritization).
> Last Updated: 2026-04-06

---

## Priority Legend

- **P0 — Critical:** Blocks other work or required for next sprint
- **P1 — High:** Next sprint candidate
- **P2 — Medium:** Within 2-3 sprints
- **P3 — Low:** Future phase, no urgency
- **Research:** Added by Research Agent, needs founder review

---

## Phase 2: Memory + Database + Tracking (Weeks 3-4)

> Week 3 is the current sprint. Items below are what follows.

- [x] FlowAgent with CryptoFlowProvider — **P0** — Week 3 Task 1
- [x] Repository pattern + PostgreSQL/SQLite — **P0** — Week 3 Task 2
- [x] ConvictionAgent — **P0** — Week 3 Task 3
- [x] DecisionAgent — **P0** — Week 3 Task 4
- [x] Memory System (4 loops) — **P0** — Week 3 Task 5
- [x] Pipeline Orchestrator — **P0** — Week 3 Task 6
- [x] First end-to-end test script — **P0** — Week 3 Task 7
- [ ] Executor: actual order placement via exchange adapter — **P0** — Week 4
- [ ] ReflectionAgent: post-trade rule distillation — **P0** — Week 4
- [ ] Ephemeral TraderBot lifecycle manager — **P1** — Week 4
- [ ] First live trade on Hyperliquid testnet — **P1** — Week 4
- [ ] Tracking system: financial + decision + health metrics — **P1** — Week 4

## Phase 2b: AI Crew Foundation

- [ ] Build quantagent-qa.skill — **P1** — Cowork skill file created, needs testing
- [ ] Build quantagent-pm.skill — **P1** — Cowork skill file created, needs testing
- [ ] Build quantagent-docs.skill — **P1** — Cowork skill file created, needs testing
- [ ] Build quantagent-research.skill — **P1** — Cowork skill file created, needs testing
- [ ] Configure Cowork scheduled tasks for all 4 skills — **P1**
- [ ] Test Slack integration for all skill reports — **P1**

## Phase 3: Backtesting + MCP + Signals (Weeks 5-7)

- [ ] Backtesting framework with offline replay — **P2**
- [ ] Overnight Quant MCP agent (nightly cron → alpha_factors.json) — **P2**
- [ ] Macro Regime Manager MCP agent (daily → macro_regime.json) — **P2**
- [ ] Signal distribution: Discord webhook integration — **P2**
- [ ] Signal distribution: Telegram Bot API integration — **P2**
- [ ] quantagent-metrics.skill for automated nightly performance — **P2**
- [ ] Prove alpha: run LLM-only engine across multiple assets/TFs — **P2**
- [ ] Conviction threshold calibration from backtest data — **P2**
- [ ] Prompt A/B testing framework — **P3**

## Phase 4: ML + Multi-Exchange + Dashboard (Weeks 8-12)

- [ ] Train RegimeModel from collected data — **P3**
- [ ] Train DirectionModel (XGBoost/LightGBM) — **P3**
- [ ] Exchange adapter capabilities system — **P3**
- [ ] Interactive Brokers adapter — **P3**
- [ ] Alpaca adapter — **P3**
- [ ] Binance adapter — **P3**
- [ ] B2B API layer: auth, rate limiting, metering — **P3**
- [ ] SolidJS dashboard rebuild — **P3**
- [ ] Landing page with live performance metrics — **P3**
- [ ] quantagent-seo.skill — **P3**
- [ ] quantagent-marketing.skill — **P3**
- [ ] quantagent-blog.skill — **P3**
- [ ] Daily blog content pipeline activation — **P3**

## Phase 5: Product Launch + Scale (Weeks 12-24)

- [ ] Complete SolidJS dashboard migration (4 experiences) — **P3**
- [ ] iOS app (Swift/SwiftUI, TestFlight → App Store) — **P3**
- [ ] Consumer onboarding funnel (7-step, paper trading) — **P3**
- [ ] Consumer product launch with managed hosting — **P3**
- [ ] B2B API public launch — **P3**
- [ ] Data product licensing (labeled datasets, alpha factors) — **P3**
- [ ] AnomalyDetector training — **P3**
- [ ] Distillation: Claude decisions → fast student models — **P3**
- [ ] RL-based parameter optimization — **P3**

## Infrastructure & Operations

- [ ] PostgreSQL production deployment (Hetzner) — **P2**
- [ ] Redis deployment for caching + scaled Event Bus — **P3**
- [ ] CI/CD pipeline (GitHub Actions) — **P2**
- [ ] Alembic database migration setup — **P2**
- [ ] quantagent-deploy.skill for automated deployment — **P3**
- [ ] GDPR compliance: data export + deletion endpoints — **P3**
- [ ] JWT authentication system — **P3**
- [ ] Exchange credential encryption (AES-256) — **P2**

## Research Agent Additions

> Items below are added by the Research Agent. Founder reviews during Sunday planning.

*(No research findings yet — Research Agent hasn't run its first scan)*

---

> **How this file is used:**
> - Sunday planning: Architect pulls P0/P1 items into next SPRINT.md
> - Research Agent: appends findings to "Research Agent Additions" section
> - Sprint Manager: references this to flag upcoming dependencies
> - Items move from here → SPRINT.md when they're ready to build
