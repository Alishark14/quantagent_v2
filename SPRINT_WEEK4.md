# SPRINT.md — Week 4 (Engine Completion: First Trade)

> Theme: Make the engine trade. Executor places real orders, ReflectionAgent learns from outcomes, Sentinel watches markets proactively.
> Start: April 28, 2026
> Target: May 2, 2026
> Tasks: 7 (ordered by dependency)
> Foundation: 553 tests. Full analysis pipeline running on real Hyperliquid data + Claude API. ConvictionAgent scoring, DecisionAgent deciding, Memory system storing.

---

## Task 1: Executor — Order Placement [L, High]

**Status:** [ ] Not Started
**Depends on:** Exchange Adapter (Week 2), Risk Profiles (Week 2), DecisionAgent (Week 3)
**Estimated time:** 90 minutes

**Acceptance criteria:**
- [ ] Executor takes TradeAction and places orders via exchange adapter
- [ ] LONG/SHORT: market order + SL + TP1 (50%) + TP2 (50%)
- [ ] ADD_LONG/ADD_SHORT: pyramid at 50% size + adjust SL
- [ ] CLOSE_ALL: cancel all orders + market close
- [ ] HOLD/SKIP: no action, returns immediately
- [ ] SL failure triggers emergency close (non-negotiable safety)
- [ ] Emits TradeOpened/TradeClosed events via Event Bus
- [ ] Position verified on exchange after order
- [ ] Unit tests with mock adapter
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to HIGH for this task.

Implement the Executor — bridge between DecisionAgent output and exchange orders.

FILE: engine/execution/executor.py

class Executor:
    def __init__(self, adapter: ExchangeAdapter, event_bus: EventBus, config: TradingConfig):
        self._adapter = adapter
        self._bus = event_bus
        self._config = config

    async def execute(self, action: TradeAction, symbol: str) -> OrderResult:
        # SKIP/HOLD: return immediately, no adapter calls
        # LONG/SHORT: _open_position()
        # ADD_LONG/ADD_SHORT: _add_to_position()
        # CLOSE_ALL: _close_position()

    async def _open_position(self, action, symbol):
        # 1. Place market order (entry)
        # 2. Place SL order — CRITICAL: if SL fails, emergency close position immediately
        # 3. Place TP1 at 50% of position size
        # 4. Place TP2 at remaining 50%
        # 5. Verify position exists on exchange via get_positions()
        # 6. Emit TradeOpened event
        # All steps wrapped in try/except with structured logging

    async def _add_to_position(self, action, symbol):
        # Pyramid at 50% of original size
        # Adjust SL to new level if provided
        # Log pyramid details

    async def _close_position(self, symbol):
        # 1. Cancel all open orders for symbol
        # 2. Market close position
        # 3. Emit TradeClosed event

Write tests in tests/unit/test_executor.py:
- Mock adapter that tracks all method calls in order
- Test LONG: market order + SL + TP1 + TP2 placed in correct sequence
- Test SHORT: same but opposite sides
- Test SL placement failure -> emergency close_position called
- Test CLOSE_ALL: cancel_all_orders then close_position
- Test SKIP and HOLD: zero adapter method calls
- Test ADD_LONG: pyramid at 50% size
- Test TradeOpened event emitted on successful entry
- Test TradeClosed event emitted on close

Update PROJECT_CONTEXT.md sections 2, 14. Update CHANGELOG.md.
```

---

## Task 2: ReflectionAgent [M, Medium-High]

**Status:** [ ] Not Started
**Depends on:** LLM Provider (Week 1), Repository Pattern (Week 3)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] Runs asynchronously after trade closes (subscribes to TradeClosed event)
- [ ] Receives trade entry/exit data + cycle signals/conviction + market conditions
- [ ] Produces ONE concise, actionable rule (1 sentence) with reasoning
- [ ] Saves rule to RuleRepository with initial score of 0
- [ ] Emits RuleGenerated event
- [ ] Parse failure: no rule saved, no crash, log warning
- [ ] Prompt in engine/reflection/prompts/reflection_v1.py
- [ ] Unit tests with mock LLM
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to MEDIUM-HIGH for this task.

ReflectionAgent distills trade outcomes into reusable rules.
Rules with positive scores help future trades. Rules scoring below -2 get auto-deactivated.

FILE 1: engine/reflection/prompts/reflection_v1.py

System prompt instructs Claude to:
- Analyze a completed trade (entry/exit, P&L, signals at entry, indicators)
- Produce ONE actionable rule that could improve future decisions
- Good rules: specific, conditional, testable
  ("When RSI > 75 and funding positive, SHORT signals have 70% win rate")
- Bad rules: generic ("Be careful"), mechanical ("Use stop losses")
- JSON response: {"rule": str, "reasoning": str, "applies_to": str, "confidence": float}

User prompt provides: symbol, timeframe, direction, entry/exit prices, P&L,
R-multiple, duration, exit reason, conviction at entry, regime, all 3 agent
signals with confidence, key indicators (RSI, MACD, ADX, funding).

FILE 2: engine/reflection/agent.py

class ReflectionAgent:
    def __init__(self, llm_provider: LLMProvider, rule_repo: RuleRepository,
                 event_bus: EventBus):

    async def reflect(self, trade_data: dict, cycle_data: dict) -> dict | None:
        # 1. Format prompts with trade + cycle data
        # 2. Call LLM (text, not vision)
        # 3. Parse JSON response
        # 4. Save rule to repository with score=0
        # 5. Emit RuleGenerated event
        # On parse failure: log warning, return None

Set up event subscription: bus.subscribe(TradeClosed, reflection_handler)

Write tests in tests/unit/test_reflection_agent.py:
- Mock LLM returning valid rule JSON
- Test rule saved to repository with score 0
- Test RuleGenerated event emitted
- Test parse failure returns None (no crash, no rule saved)
- Test rule text is specific (not empty or generic)

Update PROJECT_CONTEXT.md sections 2, 5, 14. Update CHANGELOG.md.
```

---

## Task 3: Tracking System [M, Medium-High]

**Status:** [ ] Not Started
**Depends on:** Event Bus (Week 1), Repository Pattern (Week 3)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] TrackingModule subscribes to ALL pipeline events
- [ ] Financial: records trade open/close with P&L, R-multiple
- [ ] Decision: records per-cycle signals, conviction, action
- [ ] Health: counts events, API calls, errors, uptime
- [ ] Data moat: links all 6 layers per trade into training_examples
- [ ] All tracking is fire-and-forget (never blocks the pipeline)
- [ ] Unit tests
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to MEDIUM-HIGH for this task.

TrackingModule is a pure observer — subscribes to events, records everything.
If tracking fails, the trade still executes. Tracking NEVER blocks the pipeline.

FILES:
- tracking/financial.py: FinancialTracker (on_trade_opened, on_trade_closed)
- tracking/decision.py: DecisionTracker (on_cycle_completed)
- tracking/health.py: HealthTracker (in-memory counters, get_health_snapshot)
- tracking/data_moat.py: DataMoatCapture (links 6 layers per trade on close)
- tracking/__init__.py: TrackingModule (subscribes all, _safe wrapper for fire-and-forget)

Write tests in tests/unit/test_tracking.py:
- Test financial tracker records trade open/close
- Test health tracker increments counters on events
- Test TrackingModule subscribes to all event types
- Test _safe wrapper catches errors without propagating
- Test data moat capture links layers

Update PROJECT_CONTEXT.md sections 2, 14. Update CHANGELOG.md.
```

---

## Task 4: Sentinel — Monitor + Readiness Scoring [L, High]

**Status:** [ ] Not Started
**Depends on:** Exchange Adapter (Week 2), Indicators (Week 1), Event Bus (Week 1)
**Estimated time:** 90 minutes

**Acceptance criteria:**
- [ ] SentinelMonitor polls price data every 15-30 seconds
- [ ] ReadinessScorer evaluates 5 conditions (indicator cross, level touch, volume spike, flow shift, MACD cross)
- [ ] Readiness > 0.7 emits SetupDetected event
- [ ] Cooldown is TIMEFRAME-DEPENDENT: 15m=15min, 30m=30min, 1h=60min, 4h=4h, 1d=24h
- [ ] Daily budget is TIMEFRAME-DEPENDENT: 15m=16, 30m=12, 1h=8, 4h=4, 1d=2
- [ ] get_sentinel_cooldown() and get_sentinel_daily_budget() helper functions
- [ ] Graceful error handling and reconnection
- [ ] Unit tests including timeframe-specific cooldown and budget tests
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to HIGH for this task.

Sentinel is persistent, code-only (zero LLM cost), proactive.
IMPORTANT: Cooldown and daily budget are TIMEFRAME-DEPENDENT, not fixed values.
Cooldown equals one candle period — don't re-analyze the same candle.

FILE 1: sentinel/config.py

SENTINEL_COOLDOWNS = {
    "15m": 900,      # 15 minutes — one candle period
    "30m": 1800,     # 30 minutes
    "1h": 3600,      # 60 minutes
    "4h": 14400,     # 4 hours
    "1d": 86400,     # 24 hours
}

SENTINEL_DAILY_BUDGETS = {
    "15m": 16,       # ~25% of candles can trigger analysis
    "30m": 12,
    "1h": 8,         # ~33% of candles
    "4h": 4,         # ~66% of candles
    "1d": 2,         # both daily candle closes
}

def get_sentinel_cooldown(timeframe: str) -> int:
    return SENTINEL_COOLDOWNS.get(timeframe, 3600)

def get_sentinel_daily_budget(timeframe: str) -> int:
    return SENTINEL_DAILY_BUDGETS.get(timeframe, 8)

FILE 2: sentinel/conditions.py
ReadinessScorer with 5 weighted conditions:
- Indicator cross (RSI 30/70): weight 0.25
- Level touch (price near BB/swing within 0.3 ATR): weight 0.30
- Volume anomaly (>3x avg): weight 0.20
- Flow shift (|funding| > 0.01): weight 0.15
- MACD cross: weight 0.10
Returns (score, list[ReadinessCondition])

FILE 3: sentinel/monitor.py
SentinelMonitor:
- __init__ accepts timeframe, uses get_sentinel_cooldown(timeframe) and get_sentinel_daily_budget(timeframe)
- Polls adapter.fetch_ohlcv every check_interval seconds
- Computes fast indicators on 30-candle window
- Runs ReadinessScorer
- Emits SetupDetected when score >= threshold
- Enforces timeframe-dependent cooldown and budget

Write tests in tests/unit/test_sentinel.py:
- Test cooldown varies by timeframe (15m=900s, 1h=3600s, 4h=14400s)
- Test daily budget varies (15m=16, 1h=8, 4h=4)
- Test default fallback for unknown timeframe
- Test readiness scoring with known values (all triggered = 1.0, none = 0.0)
- Test cooldown prevents rapid re-triggers
- Test daily budget blocks after limit
- Test daily counter resets at midnight
- Test SetupDetected emitted when threshold met
- Test no event when below threshold

Update PROJECT_CONTEXT.md sections 2, 14. Update CHANGELOG.md.
```

---

## Task 5: Sentinel Position Manager [M, Medium-High]

**Status:** [ ] Not Started
**Depends on:** Task 4 (Sentinel Monitor), Exchange Adapter (Week 2)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] PositionManager monitors open positions and adjusts SL/TP
- [ ] Only TIGHTENS stops — NEVER widens (critical safety rule)
- [ ] Trailing stop: moves SL by 0.5 ATR when price moves 1 ATR in favor
- [ ] Break-even: moves SL to entry price when TP1 fills
- [ ] Flow tighten: tightens SL by 0.3 ATR when funding flips against position
- [ ] Does NOT close positions — only TraderBot can CLOSE_ALL
- [ ] Emits PositionUpdated event when SL changes
- [ ] Unit tests
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to MEDIUM-HIGH for this task.

Critical rule: Sentinel only TIGHTENS stops — never widens.
Only a TraderBot with full LLM analysis can CLOSE_ALL.

FILE: sentinel/position_manager.py

class PositionManager:
    - register_position(symbol, direction, entry_price, sl_price, atr)
    - check_adjustments(symbol, current_price, funding_rate=None)
      -> trailing stop, break-even after TP1, funding tighten
    - _is_tighter(state, new_sl) -> True only if new SL is closer to price
    - mark_tp1_filled(symbol)
    - remove_position(symbol)
    - Emits PositionUpdated when SL changes

Write tests in tests/unit/test_position_manager.py:
- Test trailing triggers after 1 ATR move
- Test trailing NEVER widens (price retraces -> SL stays)
- Test break-even after TP1 filled
- Test funding tighten when crowded against
- Test _is_tighter for LONG and SHORT
- Test modify_sl called when SL changes
- Test no modify_sl when adjustment would widen

Update PROJECT_CONTEXT.md sections 2, 14. Update CHANGELOG.md.
```

---

## Task 6: Ephemeral TraderBot Lifecycle [L, High]

**Status:** [ ] Not Started
**Depends on:** Pipeline (Week 3), Executor (Task 1), Sentinel (Task 4)
**Estimated time:** 90 minutes

**Acceptance criteria:**
- [ ] TraderBot wraps pipeline + executor in short-lived lifecycle
- [ ] BotManager spawns TraderBots on SetupDetected events
- [ ] Concurrent bot limit per symbol (default 1)
- [ ] OrphanReaper checks for positions without active bot/sentinel
- [ ] Orphans without SL get emergency SL at 2x ATR
- [ ] Unit tests for all three components
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to HIGH for this task.

TraderBots are ephemeral: spawn on trigger, run one analysis, execute, die.

FILE 1: engine/trader_bot.py
class TraderBot:
    async def run(self) -> dict:
        # Run pipeline.run_cycle() -> TradeAction
        # If action requires execution -> executor.execute()
        # Return result dict: bot_id, action, conviction, order_result, duration
        # Handle errors gracefully (return ERROR, don't crash)

FILE 2: engine/bot_manager.py
class BotManager:
    # Subscribes to SetupDetected
    # Spawns TraderBot, enforces per-symbol concurrent limit
    # Cleans up after bot completes or crashes (finally block)

FILE 3: sentinel/reaper.py
class OrphanReaper:
    # Periodically checks exchange positions vs PositionManager
    # Orphans with SL: log as safe, track
    # Orphans without SL: emergency SL at 2x ATR, critical alert

Write tests in tests/unit/test_trader_bot.py, test_bot_manager.py, test_reaper.py

Update PROJECT_CONTEXT.md sections 2, 14. Update CHANGELOG.md.
```

---

## Task 7: First Testnet Trade Script [S, Medium]

**Status:** [ ] Not Started
**Depends on:** All previous tasks
**Estimated time:** 30 minutes

**Acceptance criteria:**
- [ ] Script runs one full cycle with REAL execution on Hyperliquid TESTNET
- [ ] --testnet flag uses sandbox mode (default: True)
- [ ] --dry-run flag shows intended action without executing
- [ ] Live mode requires typing "yes" to confirm
- [ ] Prints: analysis result, orders placed, SL/TP levels, position verified
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to MEDIUM for this task.

First real trade script — on Hyperliquid testnet with fake money.

FILE: scripts/run_trade.py

Usage:
    python scripts/run_trade.py --symbol BTC-USDC --timeframe 1h --testnet
    python scripts/run_trade.py --symbol BTC-USDC --timeframe 1h --dry-run
    python scripts/run_trade.py --symbol ETH-USDC --timeframe 4h --testnet --verbose

- Load .env (with inline comment stripping)
- Validate API keys
- Initialize full stack with Executor (adapter uses testnet flag)
- Create TraderBot wrapping pipeline + executor
- --dry-run: pipeline only, print action, no execution
- --testnet (default True): adapter sandbox mode
- No --testnet: WARNING + require "yes" confirmation for live trading
- Print: action, conviction, order ID, fill price, SL/TP, position verification
- --verbose: each agent signal, conviction breakdown, memory context

No automated tests — manual verification script.
Update PROJECT_CONTEXT.md section 14. Update CHANGELOG.md.
```

---

## End of Week 4 Checklist

- [ ] pytest passes all tests (target: ~650+)
- [ ] Executor places market + SL + TP orders on mock adapter
- [ ] SL failure triggers emergency close
- [ ] ReflectionAgent generates actionable rules from trade data
- [ ] Rules saved to database with initial score 0
- [ ] Tracking system records all events (fire-and-forget)
- [ ] Sentinel readiness scoring works with 5 weighted conditions
- [ ] Sentinel cooldown is timeframe-dependent (15m=15min, 1h=60min, 4h=4h)
- [ ] Sentinel daily budget is timeframe-dependent (15m=16, 1h=8, 4h=4)
- [ ] Position Manager only tightens stops (never widens)
- [ ] TraderBot spawns, runs one cycle, executes, dies cleanly
- [ ] BotManager enforces per-symbol concurrent limit
- [ ] Orphan Reaper detects untracked positions
- [ ] run_trade.py --dry-run shows full pipeline + intended action
- [ ] run_trade.py --testnet executes on Hyperliquid testnet (manual)
- [ ] No SQL imports in engine/
- [ ] No FastAPI imports in engine/

**If all pass:** The core engine is COMPLETE. It can analyze, trade, learn, and monitor proactively.

---

## Week 5 Preview

- FastAPI web layer (bot management API, health endpoints, WebSocket streams)
- Continuous bot runner (Sentinel + BotManager running as a long-lived service)
- PostgreSQL production deployment on Hetzner
- CI/CD pipeline (GitHub Actions)
- Begin Phase 3: backtesting framework
