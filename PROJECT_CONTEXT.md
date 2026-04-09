# PROJECT_CONTEXT.md — QuantAgent v2

> Current state of the v2 codebase. Claude Code: read this before every task.
> Engine Version: 2026.04.3.9.0-alpha.1 (not yet released)
> Last Updated: 2026-04-10 (auto-synced)

---

## 1. Current State

**Status: ACTIVE DEVELOPMENT.** Core engine, API, backtesting framework, eval framework, MCP agents, and CI/CD are all built and tested. Dashboard (Phase 4) and some exchange adapters remain.

**What exists:**
- Full engine: pipeline, signals (3 LLM + 1 flow), conviction, execution, reflection, memory, sentinel
- FastAPI web layer with auth, bot management, trades, positions, rules endpoints
- Backtesting Tier 1 + Tier 2 replay + eval framework (15 hand-labelled scenarios)
- MCP agents: QuantDataScientist (alpha mining) + MacroRegimeManager (macro overlay)
- Storage: PostgreSQL + SQLite fallback, Redis/memory cache, S3/local object store
- Alembic migrations (002 migrations), CI/CD, deploy scripts
- 1737 tests (as of Sprint Portfolio-Risk-Manager Task 4), pytest collection partially blocked by missing venv packages (see §11)

**What does NOT exist yet:**
- Distribution module (Discord/Telegram channels — stubs only)
- dYdX and Deribit adapters (stubs only)
- Frontend / dashboard (Phase 4)
- ML model implementations (slots exist, return None)
- Some DB tables (see §7)

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
| Pipeline | `engine/pipeline.py` | IMPLEMENTED | Events, all layers, memory, PRM | AnalysisPipeline: 4-stage orchestrator (Data→Signal→Conviction→Execution→PRM-sizing), event emission, cycle persistence. **`is_shadow: bool = False` ctor param (Paper Trading Task 2, 2026-04-09)**: when True, every persisted cycle's `cycle_record["is_shadow"]` is set so paper-bot AND shadow-bot cycles get filtered out of the live data moat (`live_cycles` view + QuantDataScientist mining job). Default False keeps the live production path and existing test fixtures byte-for-byte unchanged. `_make_bot_factory` in main.py resolves `pipeline_is_shadow = shadow_mode or paper_mode` and threads it in. **Sprint Portfolio-Risk-Manager Task 1 (2026-04-10)**: removed `_resolve_account_balance()` helper. DecisionAgent invocation is `decide(conviction=, market_data=, current_position=, memory_context=)` — no balance arg. **Sprint Portfolio-Risk-Manager Task 4 (2026-04-10)**: new optional `portfolio_risk_manager: PortfolioRiskManager \| None = None` ctor param; new `_peak_equity: float` per-bot tracker for the Layer 6 drawdown throttle (in-memory only, resets on bot restart — DB persistence is a future enhancement). New `_apply_prm(action, market_data)` async method runs ONLY for entry actions (LONG/SHORT/ADD_LONG/ADD_SHORT): fetches `balance` + `positions` from `self._ohlcv._adapter` in two separate try/except blocks (any failure → SKIP per CLAUDE.md "SKIP is always safe"), validates `balance > 0` / `current_price > 0` / `sl_price + tp1_price are populated`, normalises adapter Position objects into PRM's dict shape (`{symbol, notional=abs(size*entry_price), direction}`), computes `sl_distance_pct + tp1_distance_pct` as positive fractions, calls `prm.size_trade(...)` with the resolved args, and either mutates `action.position_size` with the PRM-computed dollar size or converts the action to SKIP via `_convert_to_skip()` with the PRM reason appended as `[PRM override: ...]`. Peak equity update happens AFTER PRM runs so this cycle uses the OLD peak. Loud-by-default `WARNING` log at construction when PRM is None — production paths must wire one. `_make_bot_factory` builds one PRM per bot (per-bot scoping is mandatory because the `_halted` hysteresis flag must not cross-contaminate between unrelated portfolios). `scripts/run_testnet_cycle.py` also wires a real PRM and reads `action.position_size` straight from the PRM-stamped result. |
| TraderBot | `engine/trader_bot.py` | IMPLEMENTED | pipeline, executor, position_manager | Ephemeral worker: analyze→execute→register→die. Error-safe (never crashes). |
| BotManager | `engine/bot_manager.py` | IMPLEMENTED | events, TraderBot | Spawns bots on SetupDetected, per-symbol concurrency limit, cleanup in finally. **Task 11**: after every spawned (and manually-spawned) TraderBot finishes, publishes a `SetupResult` event with the canonical TRADE / SKIP classification (`_classify_outcome` — TRADE iff action ∈ {LONG/SHORT/ADD_LONG/ADD_SHORT} AND `order_result.success`). The event closes the feedback loop back to Sentinel for escalation. CLOSE_ALL classifies as SKIP (no new entry rationale to reset escalation). Concurrency-blocked spawns do NOT publish — pipeline never ran. |
| OHLCV Fetcher | `engine/data/ohlcv.py` | IMPLEMENTED | exchanges/, indicators, swings, parent_tf | Assembles complete MarketData |
| Indicators | `engine/data/indicators.py` | IMPLEMENTED | numpy | 9 indicators + compute_all + volatility_percentile |
| Swing Detection | `engine/data/swing_detection.py` | IMPLEMENTED | numpy | Pivot detection, SL structure snapping |
| Charts | `engine/data/charts.py` | IMPLEMENTED | matplotlib, numpy | Candlestick + trendline (OLS + BB) charts, grounding header builder |
| Parent TF | `engine/data/parent_tf.py` | IMPLEMENTED | indicators | SMA trend, ADX, BB width percentile |
| FlowAgent | `engine/data/flow/` | IMPLEMENTED | exchanges/ | FlowProvider ABC, CryptoFlowProvider (funding+OI), FlowAgent aggregator |
| FlowSignalAgent | `engine/data/flow/signal_agent.py` | IMPLEMENTED | engine/signals/base, engine/types | Code-only rules-based SignalProducer (4th signal voice alongside the 3 LLM agents). signal_type="flow" (new third type). Rules pipeline (first match wins): BEARISH divergence (price up + funding ≤0 + OI dropping), BULLISH accumulation (price down + OI building + funding ≤0), extreme crowded long (funding > +0.10% → contrarian BEARISH), extreme crowded short (funding < -0.10% → contrarian BULLISH), default NEUTRAL with explicit "no signal" reasoning. ~microsecond latency. **OI history limitation**: CryptoFlowProvider currently leaves `oi_change_4h=None` (no per-symbol OI history buffer), so divergence/accumulation rules fall through to NEUTRAL in the live engine until backfill lands. Documented inline at module top. Registered via SignalRegistry alongside the 3 LLM agents in `scripts/run_cycle.py`, `scripts/run_trade.py`, `backtesting/evals/pipeline_adapter.py`. Flag: `flow_signal_agent: true` in `config/features.yaml`. |
| SignalProducer Base | `engine/signals/base.py` | IMPLEMENTED | types | ABC with name/signal_type/is_enabled/analyze/requires_vision |
| Signal Registry | `engine/signals/registry.py` | IMPLEMENTED | base | register/unregister/get_enabled/get_by_type/run_all (parallel, error-safe) |
| IndicatorAgent | `engine/signals/indicator_agent.py` | IMPLEMENTED | llm/, data, charts | LLM text agent (JSON output, grounding header) |
| PatternAgent | `engine/signals/pattern_agent.py` | IMPLEMENTED | llm/, charts | LLM vision agent (16-pattern library, JSON output) |
| TrendAgent | `engine/signals/trend_agent.py` | IMPLEMENTED | llm/, charts | LLM vision agent (OLS trendlines + BB, trend regime) |
| ML Model Slots | `engine/signals/ml/` | IMPLEMENTED | base, config | MLModelSlot + DirectionModel, RegimeModel, AnomalyDetector (all return None) |
| ConvictionAgent | `engine/conviction/agent.py` | IMPLEMENTED | llm/, signals, charts, mcp.macro_regime | LLM meta-evaluator: conviction 0-1, regime, fact/subj weighting, parse-safe. Macro overlay (§13.2.4): loads `macro_regime.json` at cycle start, forces conviction=0.0 + `macro_blackout_reason` when in a blackout window, injects `MACRO REGIME OVERLAY` context block into the LLM prompt for non-NEUTRAL regimes, stamps `macro_threshold_boost` + `macro_position_size_multiplier` on the output for downstream DecisionAgent. Missing/expired/corrupt macro file → safe default (no overlay). **Prompt v1.2 (N-signal aware)**: USER_PROMPT replaced 3 hardcoded `Indicator/Pattern/Trend` blocks with a single `{signals_block}` placeholder. `_build_signals_block(signal_map)` renders all known agents in fixed order via module-level `_AGENT_DISPLAY_NAMES` dict (indicator → pattern → trend → flow), plus any unknown agents alphabetically; missing slots render "Agent did not produce a signal" so the prompt structure stays predictable. Per-agent rendering format mirrors v1.0/v1.1 EXACTLY for the first 3 agents (display names + line layout match byte-for-byte) so prompt-cache hits and eval calibration don't drift on existing voices. Adding a 5th, 6th, ... signal in the future requires zero changes — just register a new SignalProducer and add an entry to `_AGENT_DISPLAY_NAMES`. **SYSTEM_PROMPT (v1.1 carry-over)**: UNCERTAINTY ANCHOR (genuine uncertainty must anchor 0.10-0.30; the 0.40-0.49 band is reserved for "near-miss" setups; never default to 0.40 as the "uncertain" answer) and STRUCTURAL VETO / Signals Are Not A Democracy (parent-TF bearish + analysis-TF bullish forces ≥0.25 drop; 2/3 agents agreeing with a structural-risk dissent forces ≥0.15 drop; price-vs-flow contradiction forces ≥0.15 drop; conviction > 0.75 only when ALL signals agree without any veto). |
| DecisionAgent | `engine/execution/agent.py` | IMPLEMENTED | llm/, conviction, risk_profiles, safety_checks | LLM action selection: 7 actions, SL/TP, safety overrides, parse-safe. **Sprint Portfolio-Risk-Manager Task 1 (2026-04-10)**: stripped of dollar sizing — `decide()` no longer takes `account_balance`, never calls `compute_position_size`, and always emits `position_size=None`. The agent attaches a deterministic `risk_weight: float \| None` (0.75 / 1.0 / 1.15 / 1.3) computed in plain Python by the new module-level `risk_weight_from_conviction(score)` helper — the LLM never does sizing math. Conviction bands: `[0.50, 0.60) → 0.75`, `[0.60, 0.70) → 1.0`, `[0.70, 0.85) → 1.15`, `[0.85, 1.0] → 1.30`. risk_weight is set ONLY for entry actions (LONG/SHORT/ADD_LONG/ADD_SHORT); HOLD/SKIP/CLOSE_ALL leave it None so PRM can't accidentally size a non-trade. Safety-override branch clears risk_weight alongside SL/TP/RR when an entry gets converted to HOLD/SKIP. Prompt v1.1: removed the `Balance:` line, removed all sizing instructions from SYSTEM_PROMPT, added an explicit "Position sizing is OUT OF SCOPE" preamble so a re-prompted run can't pollute the JSON with sizing fields. JSON output schema unchanged (`action`, `reasoning`, `suggested_rr`). PortfolioRiskManager (Tasks 2-4) will own dollar sizing downstream and consume `risk_weight` as its multiplier on top of the bot's base `risk_per_trade`. |
| Executor | `engine/execution/executor.py` | IMPLEMENTED | exchanges/, events | Market orders + SL + TP1/TP2, pyramid, close_all, SL failure emergency close, event emission |
| Cost Model ABC | `engine/execution/cost_model.py` | IMPLEMENTED | None | ExecutionCost, PositionSizeResult, abstract ExecutionCostModel with compute_total_cost, fee_adjusted_rr, cost_aware_position_size, is_trade_viable |
| HL Cost Model | `engine/execution/cost_models/hyperliquid.py` | IMPLEMENTED | adapter | HyperliquidCostModel: fee tiers, HIP-3 deployer scaling, growth mode, orderbook slippage, funding costs |
| Generic Cost Model | `engine/execution/cost_models/generic.py` | IMPLEMENTED | None | Conservative defaults (0.1% taker, 0.05% slippage) for unknown exchanges |
| Risk Profiles | `engine/execution/risk_profiles.py` | IMPLEMENTED | config | SL/TP (ATR + swing snap), position sizing |
| Portfolio Risk Manager | `engine/execution/portfolio_risk_manager.py` | IMPLEMENTED + WIRED | None (pure stdlib) | **Sprint Portfolio-Risk-Manager Tasks 2 + 3 + 4 (2026-04-10)**: deterministic six-layer risk pipeline that owns ALL position-sizing math now that DecisionAgent only emits trade INTENT. **Wired into the production pipeline (Task 4)**: every bot built by `_make_bot_factory` constructs its own `PortfolioRiskManager(PortfolioRiskConfig())` instance and threads it into the AnalysisPipeline. The pipeline calls `prm.size_trade(...)` after DecisionAgent returns for entry actions only. Public surface: `PortfolioRiskConfig` (8 tunables), `SizingResult` (position_size_usd, risk_dollars, effective_risk_pct, drawdown_multiplier, skipped, skip_reason), `PortfolioRiskManager.size_trade(equity, peak_equity, sl_distance_pct, tp1_distance_pct, risk_weight, symbol, open_positions=None) -> SizingResult`. **Layers shipped (1, 2, 3, 5, 6)** — every layer except the reserved Layer 4 (future cost-aware sizing) is now live: Layer 1 Fixed Fractional (`risk_dollars = equity * risk_pct * dd_mult * weight`, `position = risk_dollars / sl_pct`); Layer 2 Per-Asset Cap (sums same-symbol notionals, clamps to `equity * per_asset_cap_pct - existing` or SKIPs when existing ≥ cap); Layer 3 Portfolio Cap (sums all notionals, clamps to `equity * portfolio_cap_pct - total` or SKIPs); Layer 5 LLM Cost Floor (`expected_profit ≥ cycle_cost * 20` else SKIP — strict `<` boundary); Layer 6 Drawdown Throttle (1.0/0.5/0.0 multiplier across [0, 5%, 10%) bands with hysteresis: halt at 10% drawdown, resume only below 8% — the 2-pp gap is the explicit thrash-prevention zone). **Layer execution order is 6 → 1 → 5 → 2 → 3** (the spec said "Layer 5 first" but Layer 5 needs Layer 1's size; resolved by running Layer 1 first while Layer 5 still functions as a fail-fast gate). **Tightest-cap-wins** between Layers 2 and 3 is automatic from the ordering — Layer 3 receives the post-Layer-2 size as input and either clamps further or passes it through. **Layer-attributed SKIP reasons**: `Per-asset cap: BTC-USDC existing exposure $X ≥ cap $Y (15% of equity)` vs `Portfolio cap: total exposure $X ≥ cap $Y (30% of equity)` so an operator reading PRM logs can attribute the skip to the right layer without checking the source. **Defensive position-dict parsing**: `_symbol_exposure` and `_total_exposure` static helpers skip entries with missing or non-numeric `notional` fields so a partially populated `adapter.get_positions()` result in Task 4 wiring can't crash sizing. **Statefulness**: only `_halted: bool` for the drawdown hysteresis state machine — exposed via the `is_halted` property for diagnostics + tests. Everything else is a pure function of inputs. One PRM instance per bot is the recommended scope (matches Task 4's plan to thread it through `_make_bot_factory`). **Defensive input validation**: `equity ≤ 0`, `sl_distance_pct ≤ 0`, `tp1_distance_pct ≤ 0`, `risk_weight ≤ 0` all return SKIP with structured reasons before sizing math runs. **Logging**: halt-trigger logs at WARNING (single line per state change, not per cycle), happy-path SIZE / Layer 5 SKIP / Layer 2 SKIP / Layer 3 SKIP all log at INFO. Skip reasons are self-contained and include dollar amounts, percentages, and the relevant config thresholds. **Wired in production (Task 4)**: `_make_bot_factory` constructs one PRM per bot (per-bot scoping is MANDATORY because the `_halted` hysteresis flag must not cross-contaminate between unrelated portfolios — pinned by `test_factory_constructs_one_prm_per_bot_not_shared`); the AnalysisPipeline's new `_apply_prm` method fetches balance + positions from `self._ohlcv._adapter`, normalises Position objects into PRM's dict shape with `abs(size * entry_price)` as the notional, and calls `size_trade(...)` with all the resolved arguments. PRM result either populates `action.position_size` or converts the action to SKIP with the PRM reason appended as `[PRM override: ...]`. |
| Safety Checks | `engine/execution/safety_checks.py` | IMPLEMENTED | types | 5 mechanical checks + SafetyCheckResult |
| ReflectionAgent | `engine/reflection/agent.py` | IMPLEMENTED | llm/, ReflectionRules, events | Post-trade rule distillation, TradeClosed handler, RuleGenerated event |
| Cycle Memory | `engine/memory/cycle_memory.py` | IMPLEMENTED | CycleRepository | Loop 1: save/retrieve recent cycles, prompt formatting |
| Reflection Rules | `engine/memory/reflection_rules.py` | IMPLEMENTED | RuleRepository | Loop 2: rules with self-correcting counters, auto-deactivation |
| Cross-Bot | `engine/memory/cross_bot.py` | IMPLEMENTED | CrossBotRepository | Loop 3: user_id-scoped signal sharing, prompt formatting |
| Regime History | `engine/memory/regime_history.py` | IMPLEMENTED | None (in-memory) | Loop 4: ring buffer, transition detection, streak counting |
| Memory Context | `engine/memory/__init__.py` | IMPLEMENTED | all 4 loops | build_memory_context() assembles all loops for prompt injection |
| Sentinel Monitor | `sentinel/monitor.py` | IMPLEMENTED | exchanges/, events, conditions, mcp.macro_regime | Poll-based monitor, cooldown, daily budget, SetupDetected emission. Macro blackout suppression (§13.2.4): before emitting SetupDetected, calls `_active_blackout_reason()` which loads `macro_regime.json`, honours `expires`, and walks `blackout_windows`. Suppression happens AFTER cooldown so blackouts don't burn the daily budget; missing/expired/corrupt file → no suppression. **Escalating readiness threshold (Task 11)**: per-symbol `_current_threshold: dict[str, float]` ratchets up by `ESCALATION_STEP=0.10` after every SKIP outcome (capped at base + `MAX_ESCALATION=0.25`), with cooldown switched to `SKIP_COOLDOWN_SECONDS=900` (15 min) instead of the full candle period. TRADE outcomes reset threshold + cooldown to defaults. Every new candle close (detected via `candles[-1].timestamp` advancing) wipes all escalation state. The trigger gate at the top of `_check_once` reads `current_threshold()` instead of a static value, so escalation flows through one site. Subscribes to `SetupResult` events via `subscribe_results()` to receive feedback from BotManager — without this loop, every readiness spike that the pipeline rejected as SKIP would re-fire on the next tick at the same threshold, wasting LLM budget. Tunables (`base_threshold`, `escalation_step`, `max_escalation`, `skip_cooldown_seconds`) are constructor params with config-driven defaults from `sentinel/config.py`. **Production wiring** (2026-04-09): `quantagent/runner.py::BotRunner._register_bot` calls `sentinel.subscribe_results()` immediately after constructing each new SentinelMonitor and BEFORE kicking off the run loop, so the escalation feedback loop is active by default in live runs. Tests that build a Sentinel directly still need to opt in (preserves the test isolation contract). |
| Macro Event Aggregator | `sentinel/macro_aggregator.py` | IMPLEMENTED | events, sentinel | Swarm-consensus pipeline (§13.2.5). Subscribes to `VolumeAnomaly` + `ExtremeMove`, holds them in a 60-second sliding `OrderedDict[(symbol, anomaly_type), _PendingEvent]`, emits `MacroReassessmentRequired` when 5+ UNIQUE symbols accumulate within the window. Same symbol firing both event types counts once. Window resets after emission, 10-minute cooldown blocks immediate re-trigger (cooldown SUPPRESSES emission AFTER ingest so the window stays warm). Emission payload picks the highest-severity observation per symbol. `subscribe()`/`unsubscribe()` register/remove the handler on the bus. `AggregatorMetrics` exposes counters for observability. |
| Sentinel Conditions | `sentinel/conditions.py` | IMPLEMENTED | indicators | ReadinessScorer: 5 weighted conditions (RSI/level/volume/flow/MACD), 0-1 score |
| Position Manager | `sentinel/position_manager.py` | IMPLEMENTED | adapter, events | Only tightens SL: trailing, BE after TP1, funding tighten. Calls adapter.modify_sl() + emits PositionUpdated. |
| Orphan Reaper | `sentinel/reaper.py` | IMPLEMENTED | adapter, position_manager | Detects orphans, emergency SL at 2x ATR for unprotected positions |
| Exchange Base | `exchanges/base.py` | IMPLEMENTED | None | ExchangeAdapter ABC (15 abstract + 2 optional methods) |
| Exchange Factory | `exchanges/factory.py` | IMPLEMENTED | base | Singleton-cached factory with `register`/`get_adapter`/`reset`. **Per-bot mode parameter**: `get_adapter(name, mode="live", **kwargs)` accepts an explicit mode argument — `"live"` returns the cached real adapter (existing behaviour), `"shadow"` constructs the real adapter, scrubs its signing key via `_scrub_signing_keys()`, and wraps it in a `SimulatedExchangeAdapter` whose `data_adapter=` is the scrubbed live adapter so Sentinel + signal agents see real OHLCV / orderbook / funding / open-interest data while every order method stays on the virtual portfolio. **Paper Trading Task 2 (2026-04-09)**: new third mode `"paper"` returns the real registered adapter constructed with `testnet=True` forwarded into the ctor — real signing capability, real order routing, but the venue's testnet endpoint, so fills come from the real testnet orderbook with fake money. Paper mode is the simplest of the three: NO key scrubbing (we sign real testnet orders) and NO `SimulatedExchangeAdapter` wrapping (we want real fills). `testnet=True` is FORCED — `kwargs.pop("testnet", None)` runs before the constructor call so even an explicit caller `testnet=False` is overridden, paper can never accidentally hit mainnet. Unknown exchanges raise `ValueError("Unknown exchange: ... (paper mode requires a registered adapter — testnet has no usable fallback)")` rather than falling back like shadow does. New `_paper_instances` cache kept distinct from `_instances` and `_shadow_instances` so all three modes can hold the same exchange name simultaneously without interference; `reset_paper_cache()` mirrors `reset_shadow_cache()` for symmetric test isolation. The old global `is_shadow_mode()` env-var check inside the factory is GONE — the swap is per-call now, which is what lets a single BotRunner manage live, shadow, and paper bots side-by-side. `_get_shadow_adapter` was renamed to `_build_shadow_adapter`; the new paper-mode entry point is `_build_paper_adapter`. **Key scrubbing (defense-in-depth)**: `_scrub_signing_keys(adapter)` walks two layers: (1) inner `adapter._exchange` (the ccxt object) — sets `privateKey` and `secret` to `None` (`_CCXT_SIGNING_ATTRS`), (2) wrapper-level `_private_key` / `private_key` / `_secret` on the adapter itself (`_ADAPTER_SIGNING_ATTRS`). `walletAddress` and `apiKey` are intentionally PRESERVED — wallet address is needed for `fetch_user_fees` and authenticated metadata; apiKey is a public identifier on most venues and signing requires the paired secret which IS scrubbed. If `setattr` fails (slots / properties), the failure is logged at DEBUG and skipped — the sim's order layer is the primary defense, scrubbing is hardening. **Failure modes**: in shadow mode, if the real adapter ctor raises (missing credentials etc), the factory logs at ERROR and returns a sim with `data_adapter=None`; the operator hits an explicit `RuntimeError` on the first `fetch_ohlcv` call which is the loud failure mode. Unknown `mode` values raise `ValueError("Unknown adapter mode: ... (expected 'live', 'shadow', or 'paper')")` rather than silently falling through to live. |
| Hyperliquid Adapter | `exchanges/hyperliquid.py` | IMPLEMENTED | CCXT | Native SL/TP, HIP-3, symbol conversion, flow data, testnet support (env-driven) |
| Binance Adapter | `exchanges/binance.py` | DATA-ONLY | CCXT | USDT-M perp futures (`options.defaultType="future"`). `fetch_ohlcv` only — every trading method raises `NotImplementedError` (Phase 5 Week 10). Programmatic symbol mapping: `BASE-USDC` → `BASE/USDT:USDT` (Binance perps settle in USDT, not USDC); reverse canonicalises back to `BASE-USDC` so consumers never see USDT. Parquet path stays `data/parquet/binance/BASE-USDC/`. No API key needed for OHLCV. Registered via `ExchangeFactory.register("binance", BinanceAdapter)` at module bottom; `scripts/download_history.py` imports the module to wire registration. |
| LLM Base | `llm/base.py` | IMPLEMENTED | None | LLMProvider ABC + LLMResponse dataclass |
| Claude Provider | `llm/claude.py` | IMPLEMENTED | anthropic SDK, langsmith | Retry, cost calc, prompt caching, vision, LangSmith tracing (conditional) |
| Repository Base | `storage/repositories/base.py` | IMPLEMENTED | None | 5 ABCs: Trade, Cycle, Rule, Bot, CrossBot. **Shadow-mode contract (Task 2)**: every list-returning read method takes `include_shadow: bool = False` (kw-only) and defaults to live-only — production callers leave the default and never see shadow rows; analytics paths opt in explicitly. Writes (`save_trade`, `save_cycle`, `save_bot`) read `is_shadow` from the input dict and persist it; `save_bot` also derives `is_shadow=True` when `mode == "shadow"`. New abstract method `BotRepository.get_active_bots_by_mode(mode: str)` returns active bots filtered by exact mode match — used by the shadow-redesign main.py so a single BotRunner can be booted in either mode without inverse filtering. `get_trade(trade_id)` does NOT filter by shadow flag (single-row primary-key lookup — caller already knows the row's identity). `is_shadow` is NOT inferred from the bot at the repo layer — caller passes it explicitly to avoid coupling TradeRepository to BotRepository and a second round-trip per write. |
| SQLite Repo | `storage/repositories/sqlite.py` | IMPLEMENTED | aiosqlite | Local dev fallback, all 5 repos + container. Shadow-mode read filtering applies `AND is_shadow = 0` when `include_shadow=False`. `get_active_bots_by_mode` filters by exact mode match. DDL includes the shadow columns from Task 1. |
| PostgreSQL Repo | `storage/repositories/postgres.py` | IMPLEMENTED | asyncpg | Standard backend, all 5 repos + pool container. Shadow-mode read filtering applies `AND is_shadow = FALSE` when `include_shadow=False`. `get_active_bots_by_mode` filters by exact mode match. DDL fallback (used in dev/test paths where Alembic isn't run) includes the shadow columns added by revision 003. Production paths use `alembic upgrade head`. |
| Repo Factory | `storage/repositories/__init__.py` | IMPLEMENTED | base, sqlite, postgres | get_repositories() factory, env-driven backend |
| Alembic Migrations | `alembic/` | IMPLEMENTED | alembic, sqlalchemy, asyncpg | async env.py, 001_initial (5 tables + 14 indexes), 002_forward_max_r (adds nullable forward_max_r FLOAT to trades), 003_shadow_mode_columns (adds `is_shadow BOOL NOT NULL DEFAULT false` to bots/trades/cycles, `mode VARCHAR(10) NOT NULL DEFAULT 'live'` to bots, plus `live_trades` / `live_cycles` PostgreSQL views as ergonomic shadow-free reads), script.py.mako |
| Seed Script | `scripts/seed_dev.py` | IMPLEMENTED | repos | 3 dev bots + 2 reflection rules, idempotent |
| Shadow Bot Migration | `scripts/migrate_shadow_bots.py` | IMPLEMENTED | asyncpg | One-shot script for the shadow-redesign cutover. Reads every row from `quantagent_shadow.bots`, inserts into `quantagent.bots` with `is_shadow=true, mode='shadow'`, preserves IDs (so existing trades/cycles/Sentinel state stay joinable), skips duplicates by primary key. Reads `DATABASE_URL` (prod) and `SHADOW_DATABASE_URL` (legacy shadow DB; defaults to prod URL with `/quantagent` rewritten to `/quantagent_shadow`). Run once after `alembic upgrade head` lands revision 003 on the server, then drop the old shadow DB. |
| Cache Backend | `storage/cache/base.py` | IMPLEMENTED | None | CacheBackend ABC (get/set/delete/clear/has) |
| Memory Cache | `storage/cache/memory.py` | IMPLEMENTED | cachetools | MemoryCacheBackend: TTLCache buckets per TTL, single-server |
| Cache Metrics | `storage/cache/metrics.py` | IMPLEMENTED | None | CacheMetrics: hits, misses, hit_rate, summary |
| Cache Manager | `storage/cache/__init__.py` | IMPLEMENTED | base, memory, metrics, file_cache, ttl | CacheManager: thundering herd (per-key locks), dual backend routing (memory + file for charts), epoch-aligned TTL |
| File Cache | `storage/cache/file_cache.py` | IMPLEMENTED | None | Filesystem cache for chart images (zlib-compressed). TTL via mtime. Not Redis-suitable (200-500KB blobs). |
| TTL Utilities | `storage/cache/ttl.py` | IMPLEMENTED | None | Epoch-aligned TTL computation (avoids drift at candle boundaries), TIMEFRAME_SECONDS constants |
| LLM Prompt Cache | `llm/cache.py` | IMPLEMENTED | None | Anthropic prompt-cache TTL tracking (5-min window sentinel) |
| Tracking | `tracking/` | IMPLEMENTED | events | TrackingModule + FinancialTracker, DecisionTracker, HealthTracker. _safe() wrappers (now async-aware: detects coroutine handlers, schedules them on the running loop, propagates nothing). Optional `forward_max_r_stamper` constructor param wires `ForwardMaxRStamper` as a TradeClosed handler. |
| Data Moat | `tracking/data_moat.py` | IMPLEMENTED | events | DataMoatCapture: 6-layer capture (L0-L5), links cycles and trades |
| Forward Max-R Stamper | `tracking/forward_r.py` | IMPLEMENTED | TradeRepository, ForwardPathLoader | `compute_forward_max_r(direction, entry_price, risk, forward_path)` pure helper (LONG: max((high−entry)/risk); SHORT: max((entry−low)/risk); clamped at 0; accepts polars DF or list-of-dicts). `ForwardMaxRStamper` orchestration class: `stamp_trade(trade_id)` loads via repo, picks 1m/5m forward path resolution, computes, persists via `repo.update_trade({"forward_max_r": v})`. `on_trade_closed(event)` is the event-bus handler — skips events without `trade_id` with a debug log (graceful no-op until executor threads id through). Risk derivation order: explicit `risk` > `sl_price` distance > 1% of entry default. All failure modes silent (missing parquet/trade/fields → warn + None). |
| FastAPI App | `api/app.py` | IMPLEMENTED | FastAPI | Lifespan (startup/shutdown), router includes |
| API Auth | `api/auth.py` | IMPLEMENTED | None | X-API-Key header auth, SHA-256 user_id derivation |
| API Schemas | `api/schemas.py` | IMPLEMENTED | Pydantic | Request/response models for all endpoints |
| API Dependencies | `api/dependencies.py` | IMPLEMENTED | FastAPI | DI providers for repos + health tracker |
| Bot Routes | `api/routes/bots.py` | IMPLEMENTED | FastAPI | CRUD + /analyze (manual cycle trigger) |
| Trade Routes | `api/routes/trades.py` | IMPLEMENTED | FastAPI | List + detail with filters |
| Health Route | `api/routes/health.py` | IMPLEMENTED | FastAPI | System health snapshot (no auth required) |
| Position Routes | `api/routes/positions.py` | IMPLEMENTED | FastAPI | Open positions across all bots |
| Rule Routes | `api/routes/rules.py` | IMPLEMENTED | FastAPI | Reflection rules by symbol/timeframe |
| BotRunner | `quantagent/runner.py` | IMPLEMENTED | sentinel, bot_manager, repos | Production service: sentinel per symbol, scheduled fallbacks, auto-restart with backoff, graceful shutdown. **Per-bot mode (Task 4)**: constructor accepts `shadow_mode: bool = False` (process-wide marker, exposed via `runner.shadow_mode` property). `_register_bot(bot)` reads `mode = bot.get("mode", "live")` from each bot dict and threads it through to `self._adapter_factory(exchange, mode=mode)` so the per-bot adapter type matches the bot's mode. In a single-mode runner (the production happy path) this echoes the process mode; in a mixed-mode runner (the integration test) each sentinel gets the right adapter regardless. |
| CLI | `quantagent/main.py` | IMPLEMENTED | runner, api, uvicorn | `python -m quantagent.main run` starts BotRunner + FastAPI together, signal-based graceful shutdown. **Shadow mode (Task 4)**: `--shadow` CLI flag is now a SIMPLE PROCESS MARKER — sets `os.environ["QUANTAGENT_SHADOW"]="1"`, no DB URL swap, no `configure_shadow` call, no `ensure_shadow_db` pre-create. `_run_server` reads the env var into `shadow_mode: bool`, logs the shadow banner if set, passes `shadow_mode` to both `_make_bot_factory` (which threads it through to `adapter_factory(exchange, mode=...)`) and `BotRunner(...)`. Loads bots filtered by mode via `repos.bots.get_active_bots_by_mode("shadow" if shadow_mode else "live")` so a shadow boot never sees live bots and vice versa. `_setup_logging()` now passes `force=True` to `logging.basicConfig` so handlers installed by uvicorn / anthropic / ccxt at import time don't silently no-op the engine's startup log lines. |
| Version Module | `quantagent/version.py` | IMPLEMENTED | None | Single source of truth: ENGINE_VERSION, API_VERSION, PROMPT_VERSIONS, ML_MODEL_VERSIONS |
| CI/CD | `.github/workflows/ci.yml` | IMPLEMENTED | GitHub Actions | Unit+API tests on push, integration on main, import violation checks |
| Deploy | `deploy/` | IMPLEMENTED | systemd, nginx | quantagent.service, nginx.conf, .env template, deploy.sh, health_check.sh, README |
| Backtest Downloader | `backtesting/data_downloader.py` | IMPLEMENTED | adapter ABC, polars | Adapter-agnostic. Paginates `fetch_ohlcv(since=...)` via the ExchangeAdapter ABC. Month-partitioned Parquet at `data/parquet/{exchange}/{SYMBOL}/{TF}_{YYYY-MM}.parquet`. Zero exchange-specific imports (enforced by AST test). |
| Backtest Loader | `backtesting/data_loader.py` | IMPLEMENTED | polars | Per-exchange instance. Reads `data/parquet/{exchange}/{SYMBOL}/...`. Cross-month stitching, dedup, `load_as_market_data()` matches OHLCVFetcher shape. |
| Sim Exchange | `backtesting/sim_exchange.py` | IMPLEMENTED | ExchangeAdapter ABC, types | SimulatedExchangeAdapter: full ABC implementation, virtual balance, market/limit/SL/TP fills, slippage, fee_model integration, funding via apply_funding(), candle-driven SL/TP triggering (SL > TP priority), netting positions with avg-entry, internal trade history + equity curve, fetch_ohlcv delegates to ParquetDataLoader. |
| Sim Executor | `backtesting/sim_executor.py` | IMPLEMENTED | sim_exchange | SimExecutor: thin read facade. get_trade_history(), get_equity_curve(), aggregate metrics (total_pnl, total_fees, win_rate, num_trades), tick driver pass-throughs (on_candle, on_prices, apply_funding). |
| Mock Signals | `backtesting/mock_signals.py` | IMPLEMENTED | SignalProducer ABC | MockSignalProducer: deterministic SignalProducer for backtests. Modes: always_long/short/skip, random_seed:N (reproducible RNG), from_file:PATH (replay JSON-recorded directional decisions). Implements full SignalProducer ABC. |
| Backtest Engine | `backtesting/engine.py` | IMPLEMENTED | sim_exchange, sim_executor, mock_signals, ParquetDataLoader, ReadinessScorer, risk_profiles, indicators, swing_detection, InProcessBus | BacktestEngine for Tier 1 mechanical mode. BacktestConfig validates inputs. Loads merged candle stream from Parquet, ticks each candle through sim_adapter (drives SL/TP), applies funding at intervals, runs ReadinessScorer gate, asks MockSignalProducer for direction, mechanically translates to TradeAction via risk_profiles + compute_position_size, places orders on sim_adapter. Emits SetupDetected/CycleCompleted/TradeOpened on EventBus. Full mode raises NotImplementedError (Tier 3). BacktestResult dataclass: trade_history, equity_curve, metrics (win_rate, total_pnl, max_drawdown), JSON-serialisable. |
| Forward Path Loader | `backtesting/forward_path.py` | IMPLEMENTED | data_loader, polars | ForwardPathLoader: thin window-based wrapper over ParquetDataLoader. `load(symbol, entry_timestamp_ms, duration_candles, resolution)` returns N forward bars at 1m or 5m resolution. Auto-fallback 1m → 5m with warning when high-res file missing. `recommended_resolution(tf)` → 1m for ≤1h, 5m for 4h/1d. |
| Tier 2 Replay | `backtesting/tier2_replay.py` | IMPLEMENTED | forward_path, polars | Tier2ReplayEngine: counterfactual replay of recorded trades against modified mechanical params. `replay_trade(trade, params, fp)`: walks the forward path bar-by-bar, models 50/50 TP1+TP2 partial close, SL > TP priority, optional break-even and Chandelier-style trailing, conviction-threshold filter (skips trades below threshold). Returns ReplayResult with delta_pnl + delta_r vs original. `replay_batch()` + `parameter_sweep()` cache forward paths and aggregate results into SweepResult rows (param_value, num_trades, num_skipped, total_pnl, win_rate, avg_r, max_drawdown). Zero LLM calls. |
| Backtest Metrics | `backtesting/metrics.py` | IMPLEMENTED | None | `BacktestMetrics` (21-field dataclass) + `calculate_metrics(history, equity, config, setups_detected, setups_taken)`. Sharpe via end-of-day equity buckets × √365 (crypto). Calmar via annualised return / max DD. Profit factor capped at 999.9. Max drawdown + duration via single-pass equity walk with running peak. R-multiple per-trade if `sl_price` present, else fallback to `initial_balance × risk_per_trade`. Zero-trade / single-trade / all-winner / all-loser edges all return safe values. Wired into `BacktestEngine._build_result`. |
| Backtest Reporter | `backtesting/reporter.py` | IMPLEMENTED | matplotlib (Agg backend), backtesting.metrics | `generate_json_report()` + `generate_html_report()`. Files saved as `{output_dir}/{YYYY-MM-DD}_{mode}_backtest.{ext}`. JSON: full payload (config, metrics, equity curve, trades, engine_versions). HTML: self-contained doc — header + config grid, metrics summary cards, base64-embedded equity + drawdown PNGs, trade table with html.escape on every cell, footer with engine + prompt versions. No external CSS or JS. |
| Shadow Mode | DELETED (`backtesting/shadow.py` removed in Sprint Shadow-Redesign Task 4) | n/a | n/a | The old env-var-driven shadow infrastructure (`is_shadow_mode`, `configure_shadow`, `ensure_shadow_db`, `get_shadow_db_url`, `ShadowConfig`, `enable_shadow_mode`, `disable_shadow_mode`) is gone. Shadow mode is now a per-row property of `bots` / `trades` / `cycles` (Alembic 003) plus an explicit `mode` argument on `ExchangeFactory.get_adapter` (Task 3) plus a `shadow_mode: bool` constructor param on `BotRunner` (Task 4). The `--shadow` CLI flag is now a simple `os.environ["QUANTAGENT_SHADOW"] = "1"` marker that `_run_server` reads to pick which mode to filter bots by (`get_active_bots_by_mode("shadow")`). Same DB, same engine code path, mode-tagged data. |
| Eval Schema | `backtesting/evals/scenario_schema.py` | IMPLEMENTED | pydantic | `Scenario` / `ScenarioInput` / `ExpectedBehavior` Pydantic models. Schema enforced at load time so malformed scenarios fail loudly. `extra="allow"` so future fields don't break old loaders. |
| Eval Output Contract | `backtesting/evals/output_contract.py` | IMPLEMENTED | None | `EvalOutput` dataclass — model-agnostic decision shape (direction, conviction, sl/tp, latency, model_id + teacher_agreement / conviction_calibration populated by framework). |
| Eval Runner | `backtesting/evals/framework.py` | IMPLEMENTED | scenario_schema, output_contract, statistics | `EvalRunner` loads scenarios from `manifest.json`, runs them through any duck-typed `pipeline` (callable or object with `analyze()`), grades against expected behaviour (action match, direction match, conviction range), measures consistency stdev across runs. Tiered modes: `run_smoke()` (2 per category × 1 run), `run_category(cat)` (one category × N runs), `run_full(N)` (all × N runs). Per-scenario crashes are recorded as FAIL, never propagate. `EvalReport` is JSON-serialisable with model + prompt versions stamped. |
| Eval Judge | `backtesting/evals/judge.py` + `judge_rubrics.py` + `prompts/judge_system_prompt.txt` | IMPLEMENTED | LLMProvider ABC, judge_rubrics | LLM-as-Judge: separate Claude call with strict system prompt + category-specific rubric. Scores 4 dimensions 1-5 with explanations. Defensive JSON parsing (handles markdown fences, preambles, missing fields, out-of-range scores). Falls back to all-1s `_failure_score` on parse error so the framework can flag for review without crashing. Rubrics for clear_setups, clear_avoids, conflicting_signals, regime_transitions, trap_setups, high_impact_events, edge_cases, cross_tf_conflicts, flow_divergence. |
| Eval Auto-Miner | `backtesting/evals/auto_miner.py` | IMPLEMENTED | None (loose-coupled trade fetcher) + RepositoryTradeFetcher uses storage.repositories | `AutoMiner.mine(days)` scans recent trades for two failure modes: (1) overconfident disasters (conviction ≥ 0.85 AND pnl ≤ 0) → categorised as `trap_setups`; (2) missed opportunities (low conviction or SKIP action AND `forward_max_r` ≥ 3.0) → categorised as `clear_setups`. Writes pre-filled scenario JSON drafts to `scenarios/auto_mined/pending_review/` with the inputs frozen and `expected` blank for human labelling. Trade fetcher is sync-or-async, duck-typed. **`RepositoryTradeFetcher`** is the production implementation that satisfies the TradeFetcher protocol: pulls `get_trades_by_bot()` across explicit `bot_ids` (or discovers via `BotRepository.get_bots_by_user("dev-user")` for single-tenant dev), normalises field names (`conviction_score → conviction`, `direction → action`, `id → trade_id`, `entry_time` ISO → `entry_timestamp` ms), swallows per-bot fetch failures with a warning. |
| Auto-Mine CLI | `scripts/mine_eval_scenarios.py` | IMPLEMENTED | get_repositories, AutoMiner, RepositoryTradeFetcher | `python scripts/mine_eval_scenarios.py --days 7 [--bot-id A] [--backend sqlite] [--per-bot-limit 200] [--output-dir DIR]`. Connects to trade repo via `get_repositories()`, runs AutoMiner end-to-end, prints scan summary + draft list. Has sys.path bootstrap so it works without `python -m`. Promotion to `manifest.json` is intentionally manual. |
| Eval Reporter | `backtesting/evals/reporter.py` | IMPLEMENTED | EvalReport | `generate_eval_report(report, output_dir, run_date, previous_report)` writes JSON + self-contained HTML to `backtesting/evals/reports/{YYYY-MM-DD}_eval.{ext}`. HTML: hero pass-rate card with good/warn/bad colour bands, per-category table, regressions section (when previous_report supplied; 5pp threshold per ARCHITECTURE §31.4.9), top failures list, footer with model + prompt versions. Every cell `html.escape`'d. |
| Eval Scenarios | `backtesting/evals/scenarios/` | IMPLEMENTED | — | **15 hand-labelled scenarios across 9 categories** (expanded from 5). `clear_setups` (2), `clear_avoids` (2), `conflicting_signals` (1), `regime_transitions` (2: trending→ranging exhaustion, quiet→volatile expansion), `trap_setups` (2: distribution top with RSI divergence, Wyckoff bear-trap spring), `high_impact_events` (2: pre-FOMC blackout, post-CPI whipsaw), `edge_cases` (1: extreme funding + low liquidity), `cross_tf_conflicts` (1: 1h bull / 4h bear), `flow_divergence` (2: smart-money distribution, smart-money accumulation). Each scenario has 49–60 synthetic OHLCV candles, plausible indicator + flow values that match the narrative, and graded `ExpectedBehavior` with notes explaining the labelling rationale. Indexed via `manifest.json`. `auto_mined/{pending_review,promoted}/` reserved for AutoMiner output. |
| Eval Framework | `backtesting/evals/framework.py` | IMPLEMENTED | None | EvalRunner + EvalReport + ScenarioResult; model-agnostic scenario grading — same scenarios run against live Claude pipeline, fine-tuned models, or deterministic mocks. (ARCHITECTURE.md §31.4) |
| Eval Schema | `backtesting/evals/scenario_schema.py` | IMPLEMENTED | None | Scenario, ScenarioInput, ExpectedBehavior dataclasses |
| Eval Output Contract | `backtesting/evals/output_contract.py` | IMPLEMENTED | None | EvalOutput standard model output contract |
| Eval Judge | `backtesting/evals/judge.py` | IMPLEMENTED | None | LLM-backed JudgeScore + judge_output + parse_judge_response; grades EvalOutput against ExpectedBehavior |
| Eval Judge Rubrics | `backtesting/evals/judge_rubrics.py` | IMPLEMENTED | None | CATEGORY_RUBRICS dict + get_rubric() — per-category grading prompts |
| Eval Auto-Miner | `backtesting/evals/auto_miner.py` | IMPLEMENTED | tracking/ | AutoMiner: scans live trades for overconfident disasters (conviction > 0.85, trade lost) and missed opportunities (conviction < 0.5, > 3R forward move). Packages as pending scenario drafts for human review. |
| Eval Reporter | `backtesting/evals/reporter.py` | IMPLEMENTED | matplotlib | generate_eval_report() → JSON + HTML evaluation reports |
| Eval Pipeline Adapter | `backtesting/evals/pipeline_adapter.py` | IMPLEMENTED | mock_signals, engine.signals.*, engine.conviction.*, engine.execution.* | `PipelineAdapter` bridges `EvalRunner` callable contract to the engine. Two modes: `mock` (deterministic per-category answer, exercises MockSignalProducer ABC, $0) and `live` (lazy-builds ClaudeProvider + IndicatorAgent/PatternAgent/TrendAgent + ConvictionAgent + DecisionAgent and runs Signal→Conviction→Decision; collapses 7-action engine output to 3-state eval contract via `_collapse_action`). `_scenario_to_market_data` recomputes indicators then layers author-provided values on top, builds FlowOutput, runs swing detection. All errors caught at `__call__` boundary → graceful SKIP EvalOutput with `Pipeline error: ...` reason. **Sprint Portfolio-Risk-Manager Task 1 (2026-04-10)**: removed the `account_balance` constructor parameter (no more synthetic $10k); `_live_decision()` calls `decide()` without `account_balance=`; `_trade_action_to_eval_output()` sets `position_size_pct=None` (DecisionAgent never sizes anymore) and stamps `risk_weight=action.risk_weight` so eval reports surface DecisionAgent's intent. `EvalOutput` gained an optional `risk_weight: float \| None = None` field at the end of the dataclass (back-compat: existing recordings load fine via `from_dict`'s drop-unknown-keys path). Mock-mode `position_size_pct=0.10` is preserved for back-compat with the existing CI smoke assertion. |
| Eval CLI Harnesses | `backtesting/evals/run_smoke.py` + `run_eval.py` + `run_eval_full.py` + `_cli.py` | IMPLEMENTED | EvalRunner, PipelineAdapter, reporter | Three module-runnable harnesses (`python -m backtesting.evals.run_*`) that wrap `runner.run_smoke/run_category/run_full`, write JSON+HTML report, print summary, return CI exit code (>50% pass-rate → 0, ≤50% → 1, per ARCHITECTURE §31.4.7). Shared `_cli.py` factors out adapter construction, summary formatting, exit-code logic. Lazy imports keep `--help` from pulling in anthropic SDK. Makefile `test-smoke`/`test-eval`/`test-eval-full` targets call these. |
| Distribution | `distribution/` | NOT BUILT | — | Discord, Telegram |
| MCP Quant Data Scientist | `mcp/quant_scientist/` | IMPLEMENTED | LLMProvider, ParquetDataLoader, TradeRepository (optional), pandas/scipy/statsmodels (sandbox runtime) | Offline alpha-mining agent (ARCHITECTURE.md §13.1). 8 modules: `factor.py` (AlphaFactor frozen dataclass + nested-JSON conversion), `decay.py` (`apply_decay` + `merge_factors` + `decay_weight_for_age` formula = max(0, 1−days/30), prune threshold 0.1), `prompts.py` (SYSTEM_PROMPT + `build_analysis_prompt` mandating BH-FDR + 4mo discovery / 2mo validation + n≥15 + avg_r≥1.5), `sandbox.py` (`screen_code` + `run_analysis` — substring-blocklisted forbidden patterns + minimal `__builtins__` whitelist + lazy pandas/scipy/statsmodels binding), `agent.py` (`QuantDataScientist.run(dry_run=False)` orchestrates trade fetch → OHLCV load → LLM call → sandbox exec → result parse → decay-aware merge → atomic tmp+rename write to `alpha_factors.json`; every step wrapped so failures become an error-stamped `AlphaFactorsReport` instead of a half-written file), `runner.py` (cron entry point: `python -m mcp.quant_scientist.runner [--dry-run] [--bot-id ID] [--no-ohlcv] ...`, refuses to run without ANTHROPIC_API_KEY, lazy LLM/loader builders so --help is fast). The agent ONLY ever writes to `self._output_path` per §13.1.6. `quant_data_scientist` registered in PROMPT_VERSIONS at v1.0. |
| MCP Macro Regime | `mcp/macro_regime/` | IMPLEMENTED | LLMProvider, httpx (optional), data_fetcher.MacroDataFetcher | Offline 3-tier macro assessment (ARCHITECTURE.md §13.2). 5 modules: `data_fetcher.py` (`MacroDataFetcher.fetch()` pulls VIX/DXY via Yahoo, DVOL via Deribit, F&G via alternative.me, BTC dominance via CoinGecko, Hyperliquid total OI + avg funding via /info, plus a hardcoded FOMC/CPI/NFP calendar — every fetch wrapped in try/except, partial failures keep the rest, total failure returns calendar-only with errors dict; rate-limited 1s between calls; injectable http_client+sleep+clock for tests), `lightweight_check.py` (`LightweightCheck.run(current, previous)` — pure-code delta gate, NO LLM: triggers on VIX>5%, DXY>1%, DVOL>10%, HL_OI>10%, F&G category change, or HIGH-impact event within 24h; weekend special: when Sat/Sun AND VIX/DXY timestamps >24h stale → skip TradFi rules and use 15% DVOL threshold instead; persists current snapshot atomically to `macro_regime_snapshot.json` for next tick to diff against), `agent.py` (`MacroRegimeManager.run_deep(snapshot, urgency, triggering_symbols, reasons, dry_run)` — builds the assessment prompt, calls `LLMProvider.generate_text(agent_name="macro_regime_manager")`, parses the LLM JSON (tolerates markdown fences + prose prefix), validates regime ∈ {RISK_ON, RISK_OFF, NEUTRAL} + confidence ∈ [0,1] + clamps adjustments to safe ranges, then deterministically derives `blackout_windows` from the calendar — LLM is NEVER trusted to invent dates; only HIGH-impact events with default 60min pre / 30min post buffers; atomic tmp+rename write to `macro_regime.json`; sets `expires = generated_at + 24h`; every error path returns an error-stamped `MacroRegime` instead of crashing). Also exports `load_macro_regime(path)` helper for ConvictionAgent / Sentinel to consume in Task 6. `runner.py` cron entry point: `python -m mcp.macro_regime.runner --mode [check|deep|emergency] [--trigger-symbols ...] [--dry-run]`. `check` runs lightweight only (no LLM, ~$0); if a trigger fires, escalates inline to deep on the SAME snapshot. `deep` runs full LLM assessment. `emergency` accepts triggering symbols + sets urgency context. Lazy LLM/fetcher builders so `--help` is fast; tests monkeypatch `_build_fetcher` + `_build_llm_provider`. The agent ONLY ever writes to `self._output_path`. `macro_regime_manager` registered in PROMPT_VERSIONS at v1.0. |

---

## 3. Project Structure (Actual Files)

```
quantagent-v2/
├── ARCHITECTURE.md
├── PROJECT_CONTEXT.md           ← this file
├── CLAUDE.md
├── SPRINT.md
├── BACKLOG.md
├── CHANGELOG.md
├── README.md
├── pyproject.toml               # Project dependencies and metadata
├── version.py                   # CalVer+SemVer, model costs, prompt versions
├── alembic.ini                  # Alembic config (reads DATABASE_URL from env)
├── .gitignore
│
├── alembic/                     # Database migrations (Alembic)
│   ├── env.py                   # Async migration env (asyncpg)
│   ├── script.py.mako           # Migration template
│   └── versions/
│       ├── 001_initial.py       # Initial schema: 5 tables + 14 indexes
│       ├── 002_forward_max_r.py # Adds nullable forward_max_r FLOAT to trades
│       └── 003_shadow_mode_columns.py # is_shadow/mode columns to bots/trades/cycles + live_* views
│
├── quantagent/                  # CLI entry point package
│   ├── __init__.py
│   ├── main.py                  # CLI: run, migrate, seed commands
│   ├── runner.py                # BotRunner: sentinels, scheduled loops, auto-restart
│   └── version.py               # Re-exports ENGINE_VERSION, PROMPT_VERSIONS (source of truth)
│
├── engine/                      # THE CORE — pure library
│   ├── __init__.py
│   ├── pipeline.py
│   ├── events.py
│   ├── types.py
│   ├── config.py
│   ├── trader_bot.py            # Ephemeral worker: analyze → execute → register → die
│   ├── bot_manager.py           # Spawns bots on SetupDetected, concurrency limit
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
│   │       ├── forex.py
│   │       └── signal_agent.py  # FlowSignalAgent: code-only rules-based SignalProducer
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
│   │   ├── cost_model.py        # ExecutionCost, PositionSizeResult, abstract ExecutionCostModel
│   │   ├── portfolio_risk_manager.py  # PortfolioRiskManager: 6-layer sizing pipeline (Layers 1-3, 5-6)
│   │   ├── risk_profiles.py
│   │   ├── safety_checks.py
│   │   ├── cost_models/
│   │   │   ├── __init__.py
│   │   │   ├── hyperliquid.py   # HyperliquidCostModel: fee tiers, HIP-3, slippage, funding
│   │   │   └── generic.py       # GenericCostModel: conservative defaults for unknown exchanges
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
│   ├── monitor.py                # SentinelMonitor (now blackout-aware)
│   ├── conditions.py
│   ├── position_manager.py
│   ├── reaper.py
│   ├── macro_aggregator.py       # MacroEventAggregator (§13.2.5 swarm consensus)
│   └── config.py
│
├── exchanges/
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   ├── hyperliquid.py
│   ├── binance.py                # USDT-M perp futures, data-download only (Task 7)
│   ├── dydx.py
│   └── deribit.py
│
├── llm/
│   ├── __init__.py
│   ├── base.py
│   ├── claude.py
│   ├── groq.py                  # GroqProvider stub (not yet implemented)
│   └── cache.py                 # Prompt-cache TTL tracking (Anthropic 5-min window)
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
│   │   ├── file_cache.py        # Filesystem cache for chart images (zlib-compressed, TTL via mtime)
│   │   ├── ttl.py               # Epoch-aligned TTL computation, TIMEFRAME_SECONDS constants
│   │   ├── metrics.py
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
│   ├── forward_r.py             # ForwardMaxRStamper: post-trade forward R computation
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
│   ├── overnight_quant.py       # Stub placeholder (implementation in quant_scientist/)
│   ├── macro_regime.py          # Stub placeholder (implementation in macro_regime/)
│   ├── quant_scientist/         # Offline alpha-mining (§13.1)
│   │   ├── agent.py             # QuantDataScientist orchestrator
│   │   ├── decay.py             # Confidence decay + merge
│   │   ├── factor.py            # AlphaFactor + nested-JSON conversion
│   │   ├── prompts.py           # Analysis prompt builder (BH-FDR)
│   │   ├── runner.py            # Cron entry: --dry-run, --bot-id ...
│   │   └── sandbox.py           # screen_code + run_analysis (exec sandbox)
│   └── macro_regime/            # Offline 3-tier macro assessment (§13.2)
│       ├── agent.py             # MacroRegimeManager + MacroRegime/MacroAdjustments/BlackoutWindow
│       ├── data_fetcher.py      # MacroDataFetcher (VIX/DXY/DVOL/F&G/HL/calendar)
│       ├── lightweight_check.py # LightweightCheck delta gate (no LLM)
│       └── runner.py            # Cron entry: --mode [check|deep|emergency]
│
├── backtesting/                 # BACKTEST FRAMEWORK (Tier 1)
│   ├── __init__.py
│   ├── data_downloader.py       # HistoricalDataDownloader → Parquet (month-partitioned)
│   ├── data_loader.py           # ParquetDataLoader (Polars, cross-month stitching)
│   ├── sim_exchange.py          # SimulatedExchangeAdapter (full ABC, candle-driven SL/TP)
│   ├── sim_executor.py          # SimExecutor (thin read facade + tick driver)
│   ├── mock_signals.py          # MockSignalProducer (deterministic backtest signals)
│   ├── engine.py                # BacktestEngine + BacktestConfig + BacktestResult
│   ├── forward_path.py          # ForwardPathLoader (1m/5m forward window, w/ fallback)
│   ├── tier2_replay.py          # Tier2ReplayEngine + ReplayResult + SweepResult
│   ├── metrics.py               # BacktestMetrics + calculate_metrics()
│   ├── reporter.py              # generate_json_report() + generate_html_report()
│   # shadow.py removed in Sprint Shadow-Redesign Task 4 — see ExchangeFactory mode parameter + bots.mode column
│   ├── evals/                   # Eval framework (ARCHITECTURE.md §31.4)
│   │   ├── __init__.py          # Public API re-exports
│   │   ├── framework.py         # EvalRunner + EvalReport + ScenarioResult (model-agnostic grading)
│   │   ├── scenario_schema.py   # Scenario, ScenarioInput, ExpectedBehavior (Pydantic)
│   │   ├── output_contract.py   # EvalOutput standard model output contract
│   │   ├── judge.py             # JudgeScore, judge_output, parse_judge_response (LLM grading)
│   │   ├── judge_rubrics.py     # CATEGORY_RUBRICS, get_rubric() (per-category prompts)
│   │   ├── auto_miner.py        # AutoMiner: overconfident disasters + missed opportunities
│   │   ├── reporter.py          # generate_eval_report() → JSON + HTML
│   │   ├── pipeline_adapter.py  # PipelineAdapter: bridges EvalRunner ↔ engine (mock + live modes)
│   │   ├── run_smoke.py         # CLI harness: python -m backtesting.evals.run_smoke
│   │   ├── run_eval.py          # CLI harness: python -m backtesting.evals.run_eval
│   │   ├── run_eval_full.py     # CLI harness: python -m backtesting.evals.run_eval_full
│   │   ├── _cli.py              # Shared CLI helpers (adapter construction, summary, exit code)
│   │   ├── scenarios/           # 15 hand-labelled scenarios across 9 categories + manifest.json + auto_mined/
│   │   └── reports/             # Eval report outputs (gitignored placeholder)
│   └── results/                 # CLI run outputs (gitignored)
│
├── prompts/
│   └── judge_system_prompt.txt  # Strict LLM-as-Judge system prompt
│
├── data/                        # Local Parquet datasets (gitignored)
│   └── parquet/{exchange}/{SYMBOL}/{TIMEFRAME}_{YYYY-MM}.parquet
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
│   │   ├── portfolio.py         # Portfolio summary (placeholder)
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
│   │   ├── test_bot_manager.py
│   │   ├── test_cache.py
│   │   ├── test_charts.py
│   │   ├── test_config.py
│   │   ├── test_conviction_agent.py
│   │   ├── test_cost_model.py
│   │   ├── test_decision_agent.py
│   │   ├── test_event_bus.py
│   │   ├── test_exchange_base.py
│   │   ├── test_executor.py
│   │   ├── test_flow_agent.py
│   │   ├── test_flow_signal_agent.py # 28 tests: FlowSignalAgent rules + interface
│   │   ├── test_forward_r.py         # ForwardMaxRStamper tests
│   │   ├── test_indicator_agent.py
│   │   ├── test_indicators.py
│   │   ├── test_llm.py
│   │   ├── test_main_bot_factory.py  # Factory wires PRM + per-bot scoping
│   │   ├── test_main_cli.py          # CLI flags: --shadow, --paper, LangSmith routing
│   │   ├── test_memory.py
│   │   ├── test_ohlcv.py
│   │   ├── test_pattern_agent.py
│   │   ├── test_portfolio_risk_manager.py  # 31 tests: Layers 1/2/3/5/6 + integration
│   │   ├── test_position_manager.py
│   │   ├── test_reaper.py
│   │   ├── test_reflection_agent.py
│   │   ├── test_repositories.py
│   │   ├── test_risk_profiles.py
│   │   ├── test_runner.py
│   │   ├── test_safety_checks.py
│   │   ├── test_sentinel.py
│   │   ├── test_signal_registry.py
│   │   ├── test_swing_detection.py
│   │   ├── test_tracking.py
│   │   ├── test_trader_bot.py
│   │   ├── test_trend_agent.py
│   │   ├── test_types.py
│   │   └── test_version.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_migrations.py   # 30 tests: structure, schema parity, pool, seed, CLI
│   │   ├── test_pipeline.py
│   │   └── test_signal_registry.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── test_endpoints.py    # 38 tests: auth, CRUD, trades, health, positions, rules
│   ├── test_backtesting/        # 264 tests: downloader, loader, sim exchange/executor, mock signals, engine, forward path, tier2 replay, metrics, reporter, shadow + evals/
│   │   ├── __init__.py
│   │   ├── test_data_downloader.py
│   │   ├── test_data_loader.py
│   │   ├── test_sim_exchange.py
│   │   ├── test_sim_executor.py
│   │   ├── test_mock_signals.py
│   │   ├── test_engine.py
│   │   ├── test_forward_path.py
│   │   ├── test_tier2_replay.py
│   │   ├── test_metrics.py
│   │   ├── test_reporter.py
│   │   ├── test_shadow.py
│   │   └── test_evals/          # 104 tests: schema, output_contract, framework, judge, auto_miner, reporter, scenario_expansion
│   │       ├── __init__.py
│   │       ├── test_scenario_schema.py
│   │       ├── test_output_contract.py
│   │       ├── test_framework.py
│   │       ├── test_judge.py
│   │       ├── test_auto_miner.py
│   │       ├── test_auto_miner_wiring.py   # AutoMiner + RepositoryTradeFetcher wiring
│   │       ├── test_pipeline_adapter.py    # PipelineAdapter mock + live mode tests
│   │       ├── test_reporter.py
│   │       └── test_scenario_expansion.py  # 31 tests: 5→15 expansion contract
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── test_hyperliquid.py
│   │   └── test_binance.py
│   ├── test_mcp/                # MCP agent tests (quant_scientist + macro_regime)
│   │   ├── __init__.py
│   │   ├── test_blackout.py
│   │   ├── test_macro_aggregator.py
│   │   ├── test_macro_regime/
│   │   │   ├── __init__.py
│   │   │   ├── test_agent.py
│   │   │   ├── test_data_fetcher.py
│   │   │   ├── test_lightweight_check.py
│   │   │   └── test_runner.py
│   │   └── test_quant_scientist/
│   │       ├── __init__.py
│   │       ├── test_agent.py
│   │       ├── test_decay.py
│   │       ├── test_factor.py
│   │       ├── test_prompts.py
│   │       ├── test_runner.py
│   │       └── test_sandbox.py
│   └── fixtures/
│       ├── __init__.py
│       ├── sample_ohlcv.json
│       └── mock_llm_responses.json
│
├── scripts/
│   ├── __init__.py
│   ├── migrate_db.py
│   ├── migrate_shadow_bots.py   # One-shot: merge shadow DB → main DB with is_shadow=true
│   ├── seed_dev.py              # Seed dev DB: 3 bots + 2 rules
│   ├── download_history.py      # CLI: download HL OHLCV → data/parquet/
│   ├── run_backtest.py          # CLI: run Tier 1 mechanical backtest from Parquet
│   ├── collect_metrics.py
│   ├── health_check.py
│   ├── run_cycle.py             # Manual end-to-end cycle: real exchange + real LLM
│   ├── run_trade.py             # First real trade script: full TraderBot lifecycle
│   ├── run_testnet_cycle.py     # Testnet full TraderBot cycle (paper mode, real HL testnet + PRM)
│   ├── mine_eval_scenarios.py   # CLI: AutoMiner → pending scenario drafts
│   └── debug_agents.py         # Verbose agent debugging: feature flags + signal tracing
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
| VolumeAnomaly | symbol, severity, detail | SentinelModule | MacroEventAggregator | IMPLEMENTED |
| ExtremeMove | symbol, severity, move_pct, detail | SentinelModule | MacroEventAggregator | IMPLEMENTED |
| MacroReassessmentRequired | triggering_symbols, anomaly_types, severity_scores, triggered_at | MacroEventAggregator | (handler triggers MCP emergency mode — wiring deferred) | IMPLEMENTED |
| SetupResult | symbol, outcome ("TRADE"/"SKIP"), action, bot_id, conviction_score | BotManager | SentinelMonitor (escalation feedback) | IMPLEMENTED |
| CycleCompleted | symbol, action, conviction | Pipeline | TrackingModule | IMPLEMENTED |

> Claude Code: mark events as IMPLEMENTED when the Event Bus handles them.

---

## 5. Agent Status

| Agent | Prompt Version | Status | LLM Provider | Notes |
|-------|---------------|--------|-------------|-------|
| IndicatorAgent | v1.0 | IMPLEMENTED | Claude (text) | Grounded with indicator summary, JSON output, parse-safe |
| PatternAgent | v1.0 | IMPLEMENTED | Claude (vision) | 16-pattern library, grounding emphasis, parse-safe |
| TrendAgent | v1.0 | IMPLEMENTED | Claude (vision) | OLS trendlines + BB, trend regime classification, parse-safe |
| ConvictionAgent | v1.2 | IMPLEMENTED | Claude (text) | Fact/subjective labeling, regime classification, 0-1 scoring, parse-safe (SKIP default). **v1.2 (N-signal aware)**: USER_PROMPT now uses single `{signals_block}` placeholder rendered dynamically from all SignalProducers via `_AGENT_DISPLAY_NAMES` map (indicator → pattern → trend → flow → unknown alphabetically). Per-agent format for first 3 agents matches v1.1 byte-for-byte to avoid calibration drift. **v1.1 carry-over**: SYSTEM_PROMPT carries UNCERTAINTY ANCHOR (uncertain → 0.10-0.30, never 0.40) + STRUCTURAL VETO (multi-TF + flow contradictions force minimum conviction drops) rules. |
| DecisionAgent | v1.0 | IMPLEMENTED | Claude (text) | 7 actions, conviction-tier sizing, SL/TP via risk_profiles, safety check enforcement, parse-safe (HOLD/SKIP default) |
| ReflectionAgent | v1.0 | IMPLEMENTED | Claude (text) | Async post-trade, distills ONE rule per trade, saves to repo, emits RuleGenerated, TradeClosed handler |
| FlowSignalAgent | v1.0 | IMPLEMENTED | None (code-only) | 4th signal voice. Code-only rules-based flow interpreter — zero LLM cost, ~microsecond latency. signal_type="flow". Rules: BEARISH divergence, BULLISH accumulation, extreme crowded long/short (contrarian), default NEUTRAL. OI history limitation noted in §11. |

> Claude Code: update status to IMPLEMENTED when agent is working. Update prompt version when prompts change.

---

## 6. Exchange Status

| Exchange | Adapter File | Status | Capabilities Declared | Notes |
|----------|-------------|--------|----------------------|-------|
| Hyperliquid | `exchanges/hyperliquid.py` | IMPLEMENTED | native_sl_tp, short, funding, OI, 50x lev | Primary. 35 mapped symbols (perp + HIP-3). Ported from v1. **Testnet support**: `__init__(testnet: bool = False)` resolves the testnet flag FIRST (kwarg or `HYPERLIQUID_TESTNET` env var) so credential lookup can branch on it. In testnet mode the constructor reads `HYPERLIQUID_TESTNET_WALLET_ADDRESS` + `HYPERLIQUID_TESTNET_PRIVATE_KEY` first, falling back to the mainnet vars (`HYPERLIQUID_WALLET_ADDRESS` / `HYPERLIQUID_PRIVATE_KEY`) for backward compatibility — operators can keep both pairs in `.env` without manual swapping. Live mode reads ONLY the mainnet vars, so a leftover `HYPERLIQUID_TESTNET_*` value cannot leak into a mainnet run. `ExchangeFactory.get_adapter("hyperliquid", mode="paper")` constructs this adapter with `testnet=True` forced. |
| Binance | `exchanges/binance.py` | DATA-ONLY | native_sl_tp, short, funding, OI, 125x lev (declared) | USDT-M perp futures. `fetch_ohlcv` only — every trading method raises `NotImplementedError("BinanceAdapter is data-download only. ... Phase 5 Week 10")`. Programmatic symbol mapping `BASE-USDC ↔ BASE/USDT:USDT`. Used today for multi-venue Tier 1 backtests via `scripts/download_history.py --exchange binance`. |
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
| trades | Complete trade lifecycle (id, user_id, bot_id, symbol, timeframe, direction, entry/exit price, size, pnl, r_multiple, entry/exit time, exit_reason, conviction_score, engine_version, status, forward_max_r) | CREATED |
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
# config/features.yaml
indicator_agent: true            # 3 LLM signal agents (implemented)
pattern_agent: true
trend_agent: true
flow_signal_agent: true          # Code-only rules-based flow interpreter
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
python-dotenv = ">=1.0"        # .env loader for CLI harnesses (eval scripts)
pytest = ">=8.0"               # Testing
pytest-asyncio = ">=0.23"      # Async test support
```

> Claude Code: update this when dependencies change.

---

## 11. Known Issues

- **`CryptoFlowProvider` doesn't track OI history (limits FlowSignalAgent in production).** `engine/data/flow/crypto.py:50-57` populates `funding_rate` and `open_interest` but leaves `oi_change_4h=None` and sets `oi_trend="STABLE"` because the provider has no per-symbol OI history buffer. As a result, `FlowSignalAgent`'s BEARISH-divergence and BULLISH-accumulation rules (which require `oi_change_4h ≤ -2%` or `≥ +2%`) fall through to NEUTRAL in the live engine. The extreme-funding contrarian rules still fire normally. Unit tests use synthetic FlowOutputs with `oi_change_4h` populated to exercise the divergence/accumulation paths, so the rules are validated — they just don't fire against real exchange data yet. **Fix when convenient**: add a per-symbol rolling OI snapshot buffer to `CryptoFlowProvider` (or a new `CryptoFlowOIHistoryProvider`) that records OI on each fetch and computes the 4h delta from the buffer. Once `oi_change_4h` is populated, the divergence rules start firing automatically with no FlowSignalAgent change.
- **Project venv pip is broken (pre-existing, not in scope for any active task).** `.venv/bin/pip` shebang points at `/Users/alireza/Documents/QuantGOD/QuantAgent v2/.venv/bin/python3.12` (the OLD path, with a literal space) — the directory was renamed `QuantAgent v2` → `QuantAgent_v2` after the venv was created, so the shim can't even start. `python -m pip` also fails with `ModuleNotFoundError: No module named 'pip._vendor.rich'` because the vendored bundle is incomplete. **Workaround**: install packages via the base miniconda pip with `--python` targeting the venv binary, e.g. `/opt/homebrew/Caskroom/miniconda/base/bin/pip --python /Users/alireza/Documents/QuantGOD/QuantAgent_v2/.venv/bin/python install <pkg>`. **Fix when convenient**: recreate the venv (`python3.12 -m venv .venv --clear && pip install -e ".[dev]"`) or migrate to `uv venv`. The venv runs scripts and tests fine — only `pip install` is impacted.

> Claude Code: add issues as they are discovered. Remove when fixed.

---

## 12. Decision Log

> Architectural decisions with reasoning. Newest first.

| Date | Decision | Reasoning |
|------|----------|-----------|
| 2026-04-10 | DecisionAgent outputs trade INTENT only; PortfolioRiskManager owns dollar sizing | Asking the LLM to do sizing math couples model accuracy to capital allocation — every prompt drift, model swap, or hallucination directly moves real money. Splitting intent (LLM) from sizing (deterministic Python) lets us version + regression-test sizing rules in isolation, swap risk profiles per bot without touching prompts, and add fixed-fractional / cost-floor / drawdown layers without re-prompting the LLM. The conviction → risk_weight mapping is the single bridge between the two: a pure-Python helper grading the LLM's confidence into a multiplier PRM consumes. (Sprint Portfolio-Risk-Manager Task 1.) |
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
| 2026.04.3.9.0-alpha.1 | 2026-04-10 | Portfolio-Risk-Manager | PRM wired into AnalysisPipeline + _make_bot_factory; run_testnet_cycle updated (Tasks 1–4 complete) |
| 2026.04.3.8.0-alpha.1 | 2026-04-10 | Portfolio-Risk-Manager | PRM Layers 2 + 3: per-asset cap + portfolio cap with tightest-wins ordering |
| 2026.04.3.7.0-alpha.1 | 2026-04-10 | Portfolio-Risk-Manager | New portfolio_risk_manager.py — Layers 1 (fixed fractional), 5 (cost floor), 6 (drawdown throttle + hysteresis) |
| 2026.04.3.6.0-alpha.1 | 2026-04-10 | Portfolio-Risk-Manager | DecisionAgent outputs trade intent only; risk_weight replaces dollar sizing |
| 2026.04.3.5.0-alpha.1 | 2026-04-09 | Paper-Trading | LangSmith per-mode project routing; shadow trace separation bug fixed |
| 2026.04.2.0.0-alpha.1 | — | Genesis | Initial scaffold. |

> Claude Code: add rows when version bumps occur.

---

## 14. Changelog (Last 5 Updates)

> Claude Code: update after every significant task. Keep only last 5. Newest first.

- **2026-04-10:** Sprint Portfolio-Risk-Manager Task 4 — Wire PRM into the analysis pipeline + production bot factory. Capstone task that turns the PRM module from "tested in isolation" into "live in every bot the runner spawns". **`engine/pipeline.py`**: new optional `portfolio_risk_manager: PortfolioRiskManager \| None = None` ctor param + new `_peak_equity: float` per-bot tracker for the Layer 6 drawdown throttle (in-memory only, future DB persistence flagged in the spec). New `_apply_prm(action, market_data)` async method runs ONLY for entry actions (LONG/SHORT/ADD_LONG/ADD_SHORT): fetches balance + positions from `self._ohlcv._adapter` in two separate try/except blocks, validates everything, normalises Position objects into PRM's dict shape (`abs(size*entry_price)` as direction-agnostic notional), computes SL/TP distance percentages, calls `prm.size_trade(...)`, and either mutates `action.position_size` with the PRM-computed dollar size or converts to SKIP via `_convert_to_skip()` with the PRM reason appended as `[PRM override: ...]`. Peak equity update happens AFTER PRM runs so this cycle uses the OLD peak. Loud-by-default WARNING log at construction when PRM is None — production paths must wire one. **`quantagent/main.py::_make_bot_factory`**: imports + constructs one `PortfolioRiskManager(PortfolioRiskConfig())` per bot inside the closure (per-bot scoping is MANDATORY because the `_halted` hysteresis flag must not cross-contaminate between unrelated portfolios), threads it into the AnalysisPipeline. **`scripts/run_testnet_cycle.py`**: wires a real PRM into the pipeline ctor and removes the local fixed-fractional stand-in from the order block — the script now reads `action.position_size` straight from the PRM-stamped result. **17 new tests**: `TestPipelinePortfolioRiskManagerWiring` (9 tests including the canonical Layer 1 math pin via the actual SL price, the `[PRM override: ...]` SKIP reason format, non-entry actions bypass PRM via call-counter assertions on a new `_PortfolioStateAdapter`, balance + positions fetch failure paths, the load-bearing position-shape test that proves `abs(size*entry_price)` normalisation works end-to-end via a real Position(0.1 BTC @ $50k) → SKIP with `Per-asset cap: BTC-USDC` reason); `TestPipelinePeakEquityTracking` (4 tests pinning fresh-start, seed-from-first-balance, increase-on-growth, NEVER-decrease — the load-bearing drawdown-depth contract); `TestPipelinePrmOptional` (2 tests for back-compat — pipeline runs without PRM, construction logs a WARNING). 2 new tests in `tests/unit/test_main_bot_factory.py`: `test_factory_wires_portfolio_risk_manager_into_pipeline` (asserts the constructed bot has a `PortfolioRiskManager` of the right type with spec-default config); `test_factory_constructs_one_prm_per_bot_not_shared` (the per-bot scoping pin — `bot1._pipeline._prm is not bot2._pipeline._prm`). **Spec deviations**: PRM is OPTIONAL on the ctor for back-compat with 30+ existing test fixtures; the WARNING log makes a missing PRM loud. Peak equity is in-memory only (the spec explicitly defers DB persistence). The `_apply_prm` method SKIPs on adapter failure rather than falling back — running PRM on stale state is materially worse than skipping. **Testnet validation deferred**: spec's last criterion ("Run testnet cycle and paste output showing PRM working") needs Hyperliquid testnet credentials + funded testnet wallet + operator env setup the test harness doesn't have. Script wiring is complete and all behaviors are unit/integration tested. Engine version bumped MINOR `2026.04.3.8.0-alpha.1 → 2026.04.3.9.0-alpha.1` (Sprint Portfolio-Risk-Manager COMPLETE end-to-end). Full `pytest -q` → **1737 passed in 15.12s** (1720 → 1737, +17 net), zero regressions. **Sprint Portfolio-Risk-Manager COMPLETE — all 4 tasks landed. (1737 total)**
- **2026-04-10:** Sprint Portfolio-Risk-Manager Task 3 — Layers 2, 3 (exposure caps). Replaced the inert Layer 2 / Layer 3 placeholders that Task 2 stubbed in with the real per-asset and portfolio-wide exposure caps. Body-only change to `engine/execution/portfolio_risk_manager.py` (Task 2 already locked the public surface). **Layer 2 Per-Asset Cap**: sums same-symbol notionals via the new `_symbol_exposure` helper, computes `cap = equity * per_asset_cap_pct`, returns `0.0` (existing ≥ cap → SKIP), `min(position_size, remaining)` (clamp), or `position_size` unchanged. Other symbols don't count — Layer 2 is per-symbol scoped. **Layer 3 Portfolio Cap**: same shape, sums notional across EVERY open position via `_total_exposure`, against `equity * portfolio_cap_pct`. Both helpers skip malformed dicts (missing or non-numeric `notional`) so a partially populated `adapter.get_positions()` result in Task 4 wiring can't crash sizing — defensive against the future production wiring path. **`size_trade` integration**: replaced the catch-all `if position_size <= 0: return SKIP("Exposure caps clamped...")` block from Task 2 with TWO layer-attributed early-return blocks BETWEEN the layer calls. SKIP reasons are now `Per-asset cap: BTC-USDC existing exposure $X ≥ cap $Y (15% of equity)` vs `Portfolio cap: total exposure $X ≥ cap $Y (30% of equity)` so operators can attribute the skip to the right layer without checking the source. Both branches log at INFO. **Tightest-cap-wins**: ordering (Layer 2 first, then Layer 3) gives this automatically — Layer 3 receives the post-Layer-2 size and either clamps further or passes it through. **18 new tests**: `TestLayer2PerAssetCap` (8 tests including no-existing-positions clamps to cap, existing position math, at/above-cap SKIP, other-symbol doesn't count, multiple-same-symbol sums, skip-reason format, malformed-dict safety), `TestLayer3PortfolioCap` (5 tests including the spec scenario where 25% exposure leaves only 5% remaining → clamp to $500, at/above SKIP, skip-reason format), `TestExposureCapsCombined` (5 tests including the load-bearing tightest-wins test from the spec — Layer 2 clamps to $150, Layer 3 would allow $200, result $150; the inverse where Layer 3 is tighter; mixed-direction no-netting; Layer 2 zero short-circuits before Layer 3's reason appears; intermediate state preservation on SKIP). **Test helper change**: `_prm()` now defaults both caps to `100.0` (10000% of equity → effectively disabled) so Layer 1/5/6 tests can isolate their layer; new `_capped_prm()` helper opts back into the spec defaults for Layer 2/3 tests. The relax-by-default decision was forced because the natural Layer 1 baseline ($5000 on $10k equity with 2% SL) is 50% of equity, way over the 15% per-asset cap — without the relax, every existing Task 2 test would have started failing the moment Layer 2 went live. The `TestPortfolioRiskConfigDefaults` smoke tests still pin the spec defaults via `PortfolioRiskConfig()` directly so the dataclass contract has separate coverage. **Updated existing test**: `test_layers_2_and_3_placeholders_dont_change_size` renamed to `test_small_existing_positions_below_caps_dont_clamp` and reworked to use $50 BTC + $50 ETH positions (well below both caps) so it now serves as a NEGATIVE-CONTROL regression test for the no-op path. **Not yet wired**: Task 4 will instantiate one PRM per bot in `_make_bot_factory` and call `size_trade` from `pipeline.py` after DecisionAgent returns, passing `open_positions` from `adapter.get_positions()` as the dependency-injected exposure source. Engine version bumped MINOR `2026.04.3.7.0-alpha.1 → 2026.04.3.8.0-alpha.1` (new public behavior — Layers 2/3 actually do something now). Full `pytest -q` → **1720 passed in 14.60s** (1702 → 1720, +18 net), zero regressions. **(1720 total)**
- **2026-04-10:** Sprint Portfolio-Risk-Manager Task 2 — PortfolioRiskManager (Layers 1, 5, 6). New file `engine/execution/portfolio_risk_manager.py` ships the deterministic six-layer risk pipeline that owns ALL position sizing now that DecisionAgent only emits trade INTENT (Task 1). Public surface: `PortfolioRiskConfig` (8 tunables with spec-mandated defaults — 1% risk per trade, 15% per-asset cap, 30% portfolio cap, 20x cost-floor multiplier, 10/5/8% drawdown halt/reduce/resume), `SizingResult` (position + diagnostics + skip reason), `PortfolioRiskManager.size_trade(equity, peak_equity, sl_distance_pct, tp1_distance_pct, risk_weight, symbol, open_positions)`. **Layer 1 (Fixed Fractional)**: `risk_dollars = equity * risk_pct * dd_mult * risk_weight`, `position = risk_dollars / sl_pct`. **Layer 5 (LLM Cost Floor)**: `expected_profit ≥ cycle_cost * 20` else SKIP — strict `<` boundary, skip reason includes both dollar amounts. **Layer 6 (Drawdown Throttle)**: 1.0 / 0.5 / 0.0 multiplier with hysteresis — once halted at 10% drawdown, stays halted until equity recovers below 8% (the 2-pp gap is the explicit thrash-prevention zone). State machine exposed via `is_halted` property. `peak_equity ≤ 0` is treated as "no history" → full risk + clears stale halt. **Layer execution order**: 6 → 1 → 5 → 2 → 3. The spec said "Layer 5 first" but Layer 5 needs Layer 1's size; resolved by running Layer 1 first while Layer 5 still functions as a fail-fast gate. **Layers 2 and 3 placeholders**: signature-locked methods that return position unchanged in Task 2; Task 3 replaces the bodies without touching `size_trade`. The post-cap sanity-check `if position_size <= 0: return SKIP("Exposure caps clamped...")` is wired up in advance. **Defensive input validation**: equity / SL / TP1 / risk_weight ≤ 0 all return SKIP before any math. **31 new tests** in `tests/unit/test_portfolio_risk_manager.py`: `TestLayer1FixedFractional` (8 tests covering basic math, SL scaling invariant, weight scaling, all three zero-input safety paths), `TestLayer5LLMCostFloor` (5 tests including the at-boundary edge case + one-cent-below sanity check + skip reason format validation), `TestLayer6DrawdownThrottle` (9 tests pinning every state-machine transition: halt at 10%, halt persists at 9% drawdown — the load-bearing thrash-prevention test, resumes at 7%, full recovery to 1.0, drawdown EXACTLY at resume threshold STAYS halted because of the strict `<` release, peak_equity=0 safe), `TestSizeTradeIntegration` (5 tests including drawdown-then-Layer-5 layer interaction, intermediate-state preservation on SKIP, `open_positions=None` vs `[]` equivalence, Layer 2/3 placeholder no-op contract pin), `TestPortfolioRiskConfigDefaults` + `TestSizingResultDataclass` (4 smoke tests). Test helper `_size` defaults `peak_equity` to whatever `equity` is set to so overriding only `equity` doesn't accidentally trip Layer 6 — this gotcha was caught on the second test run and the docstring documents the trap. **Not yet wired**: Task 4 will instantiate one PRM per bot in `_make_bot_factory` and call `size_trade` from `pipeline.py` after DecisionAgent returns. Engine version bumped MINOR `2026.04.3.6.0-alpha.1 → 2026.04.3.7.0-alpha.1` (new public surface — entire `portfolio_risk_manager.py` module). Full `pytest -q` → **1702 passed in 14.87s** (1671 → 1702, +31 net), zero regressions. **(1702 total)**
- **2026-04-10:** Sprint Portfolio-Risk-Manager Task 1 — Strip DecisionAgent of dollar sizing. `DecisionAgent.decide()` no longer takes `account_balance`, never calls `compute_position_size`, and always emits `position_size=None`. New module-level `risk_weight_from_conviction(score)` helper attaches a deterministic conviction-band weight (0.75 / 1.0 / 1.15 / 1.3) AFTER the LLM call — the LLM never does sizing math. Bands: `[0.50, 0.60) → 0.75`, `[0.60, 0.70) → 1.0`, `[0.70, 0.85) → 1.15`, `[0.85, 1.0] → 1.30`. risk_weight is set ONLY for entry actions; HOLD/SKIP/CLOSE_ALL leave it None. Safety-override clears risk_weight alongside SL/TP/RR. **`engine/types.py`**: added `risk_weight: float \| None = None` to TradeAction (default-None field at the end so existing constructions still work). **`engine/execution/agent.py`**: removed `compute_position_size` import + the entire sizing branch + `_CONVICTION_SIZE_MULTIPLIER` dict + `_conviction_size_multiplier()`. **`engine/execution/prompts/decision_v1.py`**: removed `Balance: $...` line and `## ACCOUNT` section, rewrote SYSTEM_PROMPT preamble to forbid sizing output, cut size columns from the conviction-tier table, cut "ADD size = 50% of base" from pyramid rules. JSON schema unchanged. PROMPT_VERSION 1.0 → 1.1. **`engine/pipeline.py`**: deleted `_resolve_account_balance()` + `_FALLBACK_BALANCE_USD`, dropped balance fetch from `run_cycle()`, `_skip_action()` passes `risk_weight=None`. **`backtesting/evals/pipeline_adapter.py`**: dropped `account_balance` ctor param, `_live_decision()` calls `decide()` without it, `_trade_action_to_eval_output()` sets `position_size_pct=None` and stamps `risk_weight=action.risk_weight`. **`backtesting/evals/output_contract.py`**: added optional `risk_weight: float \| None = None` field to EvalOutput (back-compat preserved via `from_dict`'s drop-unknown-keys). **Tests**: 34 unit tests in `test_decision_agent.py` rewritten — replaced `TestDecisionAgentSizing` with `TestDecisionAgentRiskWeight` (7 tests covering each conviction band, the None-for-non-entry contract, the always-None position_size contract, and direct boundary testing of the helper); replaced `test_account_balance_in_prompt` with `test_account_balance_not_in_prompt` (asserts `Balance`/`$`/`account_balance` are NOT in the prompt). 17 integration tests in `test_pipeline.py` — deleted `TestPipelineAccountBalanceResolution` (4 tests), replaced with `TestPipelineDoesNotFetchBalanceForDecisionAgent` (1 test asserting `balance_call_count == 0`). **`scripts/run_testnet_cycle.py`**: prints `Risk weight` line; order placement uses local fixed-fractional sizer as PRM stand-in until Task 4 wires the real PRM. Engine version bumped MINOR `2026.04.3.5.0-alpha.1 → 2026.04.3.6.0-alpha.1`. PROMPT_VERSIONS["decision_agent"] bumped 2.0 → 2.1. Full `pytest -q` → **1671 passed in 15.13s** (1669 → 1671, +2 net), zero regressions. **(1671 total)**
- **2026-04-09:** Sprint Paper-Trading Task 4 — LangSmith per-mode project routing (process-level) + two real bug fixes. **Decision deferred**: per-call `langsmith_project` API parameter would touch 10 files (LLM ABC + Claude impl + 8 agents) — exceeds the spec's ">5 files = skip per-call routing" threshold. Audit of `llm/claude.py::_trace_call` showed it ALREADY reads `LANGCHAIN_PROJECT` per-call from `os.environ`, so the process-level env var the CLI dispatcher sets is already sufficient for the single-mode-per-process workflow Tasks 1-3 set up. Per-call routing flagged as a follow-up for when we run mixed modes within one process. **Two real bugs fixed**: (1) **Shadow mode never set `LANGCHAIN_PROJECT`** before this task — Task 3 added the paper-mode env var setup but the shadow branch only set `QUANTAGENT_SHADOW=1`, so shadow runs were inheriting `quantagent-live` from `.env` and silently leaking shadow LLM traces into the live observability dashboard since Task 3 landed. Fixed in `quantagent/main.py:436-447` by adding `os.environ["LANGCHAIN_PROJECT"] = "quantagent-shadow"` for symmetry with the paper branch. (2) **The default fallback in `_trace_call` was the stale `quantagent-v2`** — a legacy bucket name from before the live/shadow/paper convention. Bare `python -m quantagent run` without `LANGCHAIN_PROJECT` in `.env` was landing in the legacy bucket. Fixed in `llm/claude.py:261-275` by changing the default to `quantagent-live` and adding a multi-line comment documenting the routing chain. **6 new tests**: `TestLangSmithProjectRouting` (5 tests) pins the per-call read contract on `_trace_call` directly: shadow project read from env, paper project read from env, default fallback is `quantagent-live` (and specifically NOT `quantagent-v2`), env read happens per-call NOT cached (mutate env between two `_trace_call` invocations on the same provider, both calls capture the right project), silent when `_TRACING_ENABLED=False`. `TestShadowFlagSetsLangSmithProject` (1 test) calls `main()` with `--shadow` mocked into argv and asserts the env var was set to `quantagent-shadow`. Both classes use force-patched module globals (`_TRACING_ENABLED`, `_traceable`) instead of monkeypatching `LANGCHAIN_TRACING_V2` because that boolean is resolved at module load time. `langsmith.Client` is patched at the source module (not at `llm.claude.Client`) because the import is lazy inside `_trace_call`'s try block. Updated existing `test_main_cli.py::test_shadow_alone_does_not_set_paper_env_vars` — its `LANGCHAIN_PROJECT is None` assertion was no longer correct, replaced with `== "quantagent-shadow"` and added a `!= "quantagent-paper"` cross-check. Engine version bumped MINOR `2026.04.3.4.0-alpha.1 → 2026.04.3.5.0-alpha.1`. Full `pytest -q` → **1669 passed in 14.45s** (1663 → 1669, +6 net), zero regressions. **Sprint Paper-Trading COMPLETE — all 4 tasks landed. (1669 total)**
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
