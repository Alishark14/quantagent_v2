# Architecture Updates — April 2026

> Updates to the QuantAgent v2 Architecture Document.
> Apply these to ARCHITECTURE.md or use the Claude Code prompt below to implement in code.

---

## Update 1: Timeframe-Dependent Sentinel Cooldown

### Previous Design
Fixed 15-minute cooldown between Sentinel triggers regardless of timeframe.

### New Design
Cooldown equals one candle period for the bot's trading timeframe. This ensures the Sentinel doesn't re-analyze the same candle multiple times while still being responsive to new setups.

| Timeframe | Cooldown | Rationale |
|-----------|----------|-----------|
| 15m | 15 minutes | One full candle — new data available |
| 30m | 30 minutes | One full candle |
| 1h | 60 minutes | One full candle |
| 4h | 4 hours | One full candle |
| 1d | 24 hours | One full candle |

The Sentinel still monitors continuously during cooldown — readiness scores still compute, they just don't fire SetupDetected. If a genuine new setup forms after the cooldown expires, it triggers immediately.

Daily budget also scales with timeframe:

| Timeframe | Max Daily Triggers | Rationale |
|-----------|-------------------|-----------|
| 15m | 16 | Up to ~25% of candles can trigger analysis |
| 30m | 12 | ~25% of candles |
| 1h | 8 | ~33% of candles |
| 4h | 4 | ~66% of candles |
| 1d | 2 | Both candle closes could trigger |

### Architecture Doc Section to Update
**Section 8.3 (Sentinel Cost Control)** — replace the fixed 15-minute cooldown with:

> **Cooldown:** One candle period for the trading timeframe (15m bot = 15 min cooldown, 1h bot = 60 min, 4h bot = 4 hours). The Sentinel does not re-analyze the same candle.
>
> **Daily LLM budget:** Scales with timeframe — 16 triggers for 15m, 12 for 30m, 8 for 1h, 4 for 4h, 2 for 1d. Higher-readiness events are prioritized when budget is limited.

---

## Update 2: LangSmith Integration for LLM Tracing

### What
All LLM calls are traced via LangSmith for debugging, cost analysis, and prompt performance tracking. Every agent call (IndicatorAgent, PatternAgent, TrendAgent, ConvictionAgent, DecisionAgent, ReflectionAgent) is logged with full input/output, latency, token usage, and cost.

### Architecture Doc Section to Update
**Section 28 (Tracking & Observability)** — add subsection:

> **28.5 LLM Tracing (LangSmith)**
>
> All LLM calls are traced via LangSmith (LangChain's observability platform). Each trace includes: agent name, prompt version, full input (system + user prompt), full output (raw LLM response), input/output token counts, estimated cost, latency, and cache hit status.
>
> Traces are organized by: run (single LLM call), cycle (all 5 agent calls in one analysis), and bot (all cycles for a TraderBot's lifecycle).
>
> LangSmith enables: debugging agent behavior on specific inputs, comparing prompt versions (A/B testing), tracking cost per agent over time, identifying slow or expensive calls, and reproducing any historical decision for the data moat.
>
> Configuration: `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env`.

### Implementation Status
Already configured in `.env` and ClaudeProvider. Tracing is active on all LLM calls.

---

## Update 3: Hyperliquid Testnet for Development

### What
The Hyperliquid adapter supports testnet (sandbox) mode for development and testing. Testnet uses fake money on a separate environment, allowing full trade execution testing without financial risk.

### Architecture Doc Section to Update
**Section 14 (Exchange Adapter System)** — add to section 14.2:

> **Testnet Support:** The Hyperliquid adapter supports sandbox mode via `testnet=True` parameter. Testnet uses Hyperliquid's separate test environment with simulated balances. All adapter methods work identically on testnet — orders, positions, SL/TP, funding rates. This enables: end-to-end trade execution testing, paper trading mode for consumer onboarding, and CI/CD integration tests against a live API without financial risk.
>
> Usage: `ExchangeFactory.get_adapter("hyperliquid", testnet=True)` or set via config/environment.

**Section 24 (Consumer Onboarding Funnel)** — update paper trading section:

> Paper trading uses the Hyperliquid testnet adapter. The engine runs the full pipeline with real market data analysis but executes trades on testnet with simulated balances. This is not a mock — it's a real exchange environment with real order matching, just fake money.

### Implementation Status
Already implemented in `exchanges/hyperliquid.py` via `self._exchange.setSandboxMode(True)`.

---

## Claude Code Prompt: Implement Timeframe-Dependent Cooldown

Paste this into Claude Code:

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Update the Sentinel cooldown to be timeframe-dependent instead of a fixed 15 minutes.

1. In sentinel/config.py (or wherever sentinel config lives), add a SENTINEL_COOLDOWNS dict:

SENTINEL_COOLDOWNS = {
    "15m": 900,      # 15 minutes in seconds
    "30m": 1800,     # 30 minutes
    "1h": 3600,      # 60 minutes
    "4h": 14400,     # 4 hours
    "1d": 86400,     # 24 hours
}

SENTINEL_DAILY_BUDGETS = {
    "15m": 16,
    "30m": 12,
    "1h": 8,
    "4h": 4,
    "1d": 2,
}

def get_sentinel_cooldown(timeframe: str) -> int:
    """Get cooldown in seconds for a timeframe. Default: 3600 (1h)."""
    return SENTINEL_COOLDOWNS.get(timeframe, 3600)

def get_sentinel_daily_budget(timeframe: str) -> int:
    """Get max daily triggers for a timeframe. Default: 8."""
    return SENTINEL_DAILY_BUDGETS.get(timeframe, 8)

2. In sentinel/monitor.py, update SentinelMonitor.__init__() to accept timeframe and use dynamic cooldown:

- Replace fixed cooldown_seconds=900 with: cooldown_seconds=get_sentinel_cooldown(timeframe)
- Replace fixed max_daily_triggers=8 with: max_daily_triggers=get_sentinel_daily_budget(timeframe)
- Store self._timeframe for logging

3. Update any tests in tests/unit/test_sentinel.py:
- Test that cooldown varies by timeframe (15m bot = 900s, 1h bot = 3600s, 4h bot = 14400s)
- Test that daily budget varies (15m = 16, 1h = 8, 4h = 4)
- Test default fallback for unknown timeframe

4. Update PROJECT_CONTEXT.md and CHANGELOG.md.
```

---

## SPRINT.md Task 4 Update

Replace the current Task 4 acceptance criteria with:

```markdown
## Task 4: Sentinel — WebSocket Monitor + Readiness [L, High]

**Acceptance criteria:**
- [ ] SentinelMonitor polls price data every 15-30 seconds
- [ ] ReadinessScorer evaluates 5 conditions (indicator cross, level touch, volume spike, flow shift, MACD cross)
- [ ] Readiness > 0.7 emits SetupDetected
- [ ] Cooldown is TIMEFRAME-DEPENDENT: 15m=15min, 30m=30min, 1h=60min, 4h=4h, 1d=24h
- [ ] Daily budget is TIMEFRAME-DEPENDENT: 15m=16, 30m=12, 1h=8, 4h=4, 1d=2
- [ ] get_sentinel_cooldown() and get_sentinel_daily_budget() helper functions
- [ ] Graceful error handling and recovery
- [ ] Unit tests including timeframe-specific cooldown and budget tests
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated
```

---

## Summary of All Changes

| What | Where | Status |
|------|-------|--------|
| Sentinel cooldown: fixed → timeframe-dependent | Architecture sec 8.3, sentinel/config.py, sentinel/monitor.py | Needs code update |
| LangSmith LLM tracing documented | Architecture sec 28.5 | Already implemented, needs doc |
| Hyperliquid testnet documented | Architecture sec 14.2, sec 24 | Already implemented, needs doc |
| SPRINT.md Task 4 updated | SPRINT.md | Needs file update |
| Effort levels on all tasks | SPRINT.md, CLAUDE.md | Starting from Week 5 |
