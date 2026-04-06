# SPRINT.md — Week 3 (Pipeline Completion: First End-to-End Cycle)

> Theme: Complete the 5-layer pipeline and run the first full analysis cycle on live data.
> Start: April 21, 2026
> Target: April 25, 2026
> Tasks: 7 (ordered by dependency)
> Foundation: 396 tests. Data Layer, Signal Layer (3 agents), Exchange Adapter, Risk Profiles, Safety Checks all working.

---

## Task 1: FlowAgent — CryptoFlowProvider [M]

**Status:** [ ] Not Started
**Depends on:** Exchange Adapter (Week 2), Types (Week 1)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] Abstract `FlowProvider` base class
- [ ] `CryptoFlowProvider` fetches funding rate + OI from Hyperliquid adapter
- [ ] Returns `FlowOutput` dataclass (already defined in types.py)
- [ ] `FlowAgent` aggregates all enabled providers into single FlowOutput
- [ ] Graceful fallback: if funding/OI unavailable, returns FlowOutput with data_richness="MINIMAL"
- [ ] Unit tests with mock adapter
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Implement the FlowAgent — code-only (no LLM), fetches market positioning data.

FILE 1: engine/data/flow/base.py

class FlowProvider(ABC):
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    async def fetch(self, symbol: str, adapter: ExchangeAdapter) -> dict:
        """Returns partial flow data dict. FlowAgent merges all providers."""
        ...

FILE 2: engine/data/flow/crypto.py

class CryptoFlowProvider(FlowProvider):
    def name(self) -> str:
        return "crypto"

    async def fetch(self, symbol: str, adapter: ExchangeAdapter) -> dict:
        result = {}
        # Funding rate
        try:
            rate = await adapter.get_funding_rate(symbol)
            if rate is not None:
                result['funding_rate'] = rate
                if rate > 0.01:
                    result['funding_signal'] = 'CROWDED_LONG'
                elif rate < -0.01:
                    result['funding_signal'] = 'CROWDED_SHORT'
                else:
                    result['funding_signal'] = 'NEUTRAL'
        except Exception as e:
            logger.warning(f"Funding rate unavailable: {e}")

        # Open Interest
        try:
            oi = await adapter.get_open_interest(symbol)
            if oi is not None:
                result['open_interest'] = oi
                # OI delta requires previous value — store/compare later
                result['oi_trend'] = 'STABLE'  # placeholder until we track history
        except Exception as e:
            logger.warning(f"OI unavailable: {e}")

        return result

FILE 3: engine/data/flow/__init__.py (or a new flow_agent.py at engine/data/ level)

class FlowAgent:
    """Aggregates flow data from all enabled providers. Code-only, zero LLM cost."""

    def __init__(self, providers: list[FlowProvider] | None = None):
        self.providers = providers or []

    def add_provider(self, provider: FlowProvider) -> None:
        self.providers.append(provider)

    async def fetch_flow(self, symbol: str, adapter: ExchangeAdapter) -> FlowOutput:
        merged = {}
        for provider in self.providers:
            try:
                data = await provider.fetch(symbol, adapter)
                merged.update(data)
            except Exception as e:
                logger.warning(f"FlowProvider {provider.name()} failed: {e}")

        # Determine data richness
        has_funding = 'funding_rate' in merged
        has_oi = 'open_interest' in merged
        if has_funding and has_oi:
            richness = 'FULL'
        elif has_funding or has_oi:
            richness = 'PARTIAL'
        else:
            richness = 'MINIMAL'

        return FlowOutput(
            funding_rate=merged.get('funding_rate'),
            funding_signal=merged.get('funding_signal', 'NEUTRAL'),
            oi_change_4h=merged.get('oi_change_4h'),
            oi_trend=merged.get('oi_trend', 'STABLE'),
            nearest_liquidation_above=merged.get('nearest_liquidation_above'),
            nearest_liquidation_below=merged.get('nearest_liquidation_below'),
            gex_regime=merged.get('gex_regime'),
            gex_flip_level=merged.get('gex_flip_level'),
            data_richness=richness,
        )

Write tests in tests/unit/test_flow_agent.py:
- Mock adapter with funding_rate and OI
- Test CryptoFlowProvider returns correct funding signal thresholds
- Test FlowAgent merges multiple providers
- Test graceful fallback when adapter methods fail
- Test data_richness classification (FULL, PARTIAL, MINIMAL)

Update PROJECT_CONTEXT.md sections 2, 14.
```

---

## Task 2: Repository Pattern + PostgreSQL [L]

**Status:** [ ] Not Started
**Depends on:** Types (Week 1)
**Estimated time:** 90 minutes

**Acceptance criteria:**
- [ ] Abstract repository interfaces: TradeRepository, CycleRepository, BotRepository, RuleRepository
- [ ] PostgreSQL implementation using asyncpg
- [ ] SQLite fallback implementation using aiosqlite (dev only)
- [ ] Database initialization (create tables if not exist)
- [ ] Factory function: `get_repository(backend="postgresql")` → returns correct impl
- [ ] Unit tests with SQLite (no PostgreSQL needed for CI)
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Implement the repository pattern. Engine code NEVER touches SQL directly.

FILE 1: storage/repositories/base.py

class TradeRepository(ABC):
    @abstractmethod
    async def save_trade(self, trade: dict) -> str: ...
    @abstractmethod
    async def get_trade(self, trade_id: str) -> dict | None: ...
    @abstractmethod
    async def get_open_positions(self, user_id: str, bot_id: str) -> list[dict]: ...
    @abstractmethod
    async def get_trades_by_bot(self, bot_id: str, limit: int = 50) -> list[dict]: ...
    @abstractmethod
    async def update_trade(self, trade_id: str, updates: dict) -> bool: ...

class CycleRepository(ABC):
    @abstractmethod
    async def save_cycle(self, cycle: dict) -> str: ...
    @abstractmethod
    async def get_recent_cycles(self, bot_id: str, limit: int = 5) -> list[dict]: ...

class RuleRepository(ABC):
    @abstractmethod
    async def save_rule(self, rule: dict) -> str: ...
    @abstractmethod
    async def get_rules(self, symbol: str, timeframe: str) -> list[dict]: ...
    @abstractmethod
    async def update_rule_score(self, rule_id: str, delta: int) -> bool: ...
    @abstractmethod
    async def deactivate_rule(self, rule_id: str) -> bool: ...

class BotRepository(ABC):
    @abstractmethod
    async def save_bot(self, bot: dict) -> str: ...
    @abstractmethod
    async def get_bot(self, bot_id: str) -> dict | None: ...
    @abstractmethod
    async def get_bots_by_user(self, user_id: str) -> list[dict]: ...
    @abstractmethod
    async def update_bot_health(self, bot_id: str, health: dict) -> bool: ...

class CrossBotRepository(ABC):
    @abstractmethod
    async def save_signal(self, signal: dict) -> None: ...
    @abstractmethod
    async def get_recent_signals(self, symbol: str, user_id: str,
                                  limit: int = 10) -> list[dict]: ...

FILE 2: storage/repositories/sqlite.py

Implement all repository interfaces using aiosqlite.
- __init__ takes db_path: str (default "quantagent_dev.db")
- init_db() creates all tables if not exist
- Use parameterized queries (never string interpolation)
- All IDs generated with uuid4

Tables to create:
- trades: id, user_id, bot_id, symbol, timeframe, direction, entry_price,
          exit_price, size, pnl, r_multiple, entry_time, exit_time,
          exit_reason, conviction_score, engine_version, status
- cycles: id, bot_id, symbol, timeframe, timestamp, indicators_json,
          signals_json, conviction_json, action, conviction_score
- rules: id, symbol, timeframe, rule_text, score, active, created_at
- bots: id, user_id, symbol, timeframe, exchange, status, config_json,
        created_at, last_health
- cross_bot_signals: id, user_id, symbol, direction, conviction, bot_id,
                      timestamp

FILE 3: storage/repositories/postgres.py

Implement all interfaces using asyncpg.
- __init__ takes dsn: str (from DATABASE_URL env var)
- init_db() creates tables (same schema as SQLite)
- Use asyncpg connection pool
- Parameterized queries with $1, $2 syntax

FILE 4: storage/repositories/__init__.py

async def get_repositories(backend: str = None) -> dict:
    backend = backend or os.getenv("DATABASE_BACKEND", "sqlite")
    if backend == "sqlite":
        repo = SQLiteRepositories(db_path="quantagent_dev.db")
    elif backend == "postgresql":
        repo = PostgresRepositories(dsn=os.getenv("DATABASE_URL"))
    await repo.init_db()
    return repo

class SQLiteRepositories / PostgresRepositories:
    Properties: trades, cycles, rules, bots, cross_bot
    Each returns the corresponding repository implementation.

Write tests in tests/unit/test_repositories.py (using SQLite backend):
- Test save_trade and get_trade roundtrip
- Test get_open_positions filters by user_id and bot_id
- Test save_cycle and get_recent_cycles
- Test save_rule, get_rules, update_rule_score, deactivate
- Test cross_bot save_signal filters by user_id (multi-tenant isolation)

Update PROJECT_CONTEXT.md sections 2, 7, 14.
```

---

## Task 3: ConvictionAgent [L]

**Status:** [ ] Not Started
**Depends on:** LLM Provider (Week 1), Signal agents (Week 2), FlowAgent (Task 1)
**Estimated time:** 90 minutes

**Acceptance criteria:**
- [ ] Receives all signal outputs + flow data + parent TF + memory context
- [ ] Labels every input as FACTUAL or SUBJECTIVE
- [ ] Produces ConvictionOutput with 0-1 score, regime, direction
- [ ] Regime classification: TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, BREAKOUT
- [ ] Structured JSON output with fact/subjective weight reporting
- [ ] Parse failure → ConvictionOutput with score=0.0 and direction="SKIP"
- [ ] Prompt in engine/conviction/prompts/conviction_v1.py
- [ ] Unit tests with mock LLM
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

The most critical agent in the system. ConvictionAgent does NOT generate
a signal — it EVALUATES signal quality and coherence.

FILE 1: engine/conviction/prompts/conviction_v1.py

CONVICTION_SYSTEM_PROMPT = """You are a conviction evaluator for {symbol} on {timeframe}.

You receive signals from multiple agents and market data. Your job is NOT
to generate a new signal — it is to evaluate the quality and coherence
of existing signals and produce a conviction score.

INPUTS LABELED AS FACTUAL (computed, deterministic — trust these):
- Indicator values (RSI, MACD, ADX, Stochastic, etc.)
- Flow data (funding rate, open interest)
- Parent timeframe trend
- Swing levels (support/resistance)
- Volume metrics

INPUTS LABELED AS SUBJECTIVE (LLM-interpreted — weigh with context):
- IndicatorAgent signal and reasoning
- PatternAgent signal, pattern detected, reasoning
- TrendAgent signal and reasoning

REGIME CLASSIFICATION (choose one):
- TRENDING_UP: Strong uptrend, ADX > 25, +DI > -DI
- TRENDING_DOWN: Strong downtrend, ADX > 25, -DI > +DI
- RANGING: ADX < 20, price between BB bands, no clear direction
- HIGH_VOLATILITY: ATR percentile > 80, rapid price swings
- BREAKOUT: Price at/beyond key level with volume confirmation

REGIME-BASED WEIGHTING:
- TRENDING: factual_weight=0.4, subjective_weight=0.6
- RANGING: factual_weight=0.7, subjective_weight=0.3
- HIGH_VOLATILITY: factual_weight=0.6, subjective_weight=0.4
- BREAKOUT: factual_weight=0.3, subjective_weight=0.7

{grounding_header}

RESPOND IN EXACTLY THIS JSON FORMAT:
{{
  "conviction_score": 0.0 to 1.0,
  "direction": "LONG" | "SHORT" | "SKIP",
  "regime": "TRENDING_UP" | "TRENDING_DOWN" | "RANGING" | "HIGH_VOLATILITY" | "BREAKOUT",
  "regime_confidence": 0.0 to 1.0,
  "signal_quality": "HIGH" | "MEDIUM" | "LOW" | "CONFLICTING",
  "contradictions": ["list of noted contradictions"],
  "reasoning": "2-3 sentences explaining your conviction assessment",
  "factual_weight": actual weight used (0-1),
  "subjective_weight": actual weight used (0-1)
}}

CONVICTION SCORING RULES:
- All agents agree + flow confirms: 0.7-0.9
- Majority agree, minor contradictions: 0.5-0.7
- Mixed signals, no clear consensus: 0.3-0.5
- Agents disagree significantly: 0.1-0.3
- Critical contradictions (e.g., bullish signals but bearish flow + bearish parent TF): cap at 0.4
- ADX < 15: cap conviction at 0.5 regardless (no trend = no confidence)
"""

CONVICTION_USER_PROMPT = """Evaluate these signals for {symbol} ({timeframe}):

## FACTUAL DATA
{grounding_header}

## SUBJECTIVE SIGNALS
IndicatorAgent: direction={ind_direction}, confidence={ind_confidence}
  Reasoning: {ind_reasoning}
  Contradictions noted: {ind_contradictions}

PatternAgent: direction={pat_direction}, confidence={pat_confidence}
  Pattern: {pat_pattern}
  Reasoning: {pat_reasoning}
  Contradictions noted: {pat_contradictions}

TrendAgent: direction={trend_direction}, confidence={trend_confidence}
  Reasoning: {trend_reasoning}
  Contradictions noted: {trend_contradictions}

## MEMORY CONTEXT
{memory_context}

What is your conviction score for this setup?"""

FILE 2: engine/conviction/agent.py

class ConvictionAgent:
    def __init__(self, llm_provider: LLMProvider):
        self._llm = llm_provider

    async def evaluate(
        self,
        signals: list[SignalOutput],
        market_data: MarketData,
        memory_context: str = "No prior history.",
    ) -> ConvictionOutput:
        # 1. Build grounding header from market_data
        # 2. Extract individual signal details
        # 3. Format prompts with all signal data
        # 4. Call LLM
        # 5. Parse response into ConvictionOutput
        # On parse failure: return ConvictionOutput with score=0.0, direction="SKIP"

    def _format_signal(self, signal: SignalOutput) -> dict:
        """Extract display fields from a signal."""
        return {
            'direction': signal.direction or 'N/A',
            'confidence': signal.confidence,
            'reasoning': signal.reasoning,
            'contradictions': signal.contradictions,
            'pattern': signal.pattern_detected,
        }

    def _parse_response(self, response: LLMResponse) -> ConvictionOutput:
        # Same JSON extraction as signal agents
        # On failure: return safe default (score=0, direction=SKIP)

Write tests in tests/unit/test_conviction_agent.py:
- Mock LLM returning high conviction JSON
- Mock LLM returning low conviction with contradictions
- Test parse failure returns safe default (0.0, SKIP)
- Test all signal data appears in formatted prompt
- Test grounding header is included
- Test memory context is included

Update PROJECT_CONTEXT.md sections 2, 5, 14.
```

---

## Task 4: DecisionAgent [M]

**Status:** [ ] Not Started
**Depends on:** ConvictionAgent (Task 3), Risk Profiles + Safety Checks (Week 2)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] Receives ConvictionOutput + current position + memory
- [ ] Produces TradeAction with sizing, SL/TP, reasoning
- [ ] Mechanical safety checks run AFTER LLM decision
- [ ] Parse failure → TradeAction with action="SKIP"
- [ ] Low conviction (< threshold) → immediate SKIP without LLM call
- [ ] Prompt in engine/execution/prompts/decision_v1.py
- [ ] Unit tests with mock LLM
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

DecisionAgent focuses on trade mechanics. It receives a PRE-FILTERED
conviction score — the hard work is done. It decides sizing and risk.

FILE 1: engine/execution/prompts/decision_v1.py

System prompt covers: 7 action types (LONG, SHORT, ADD_LONG, ADD_SHORT,
CLOSE_ALL, HOLD, SKIP), position context, pyramid rules (50% size,
0.5 ATR distance requirement), conviction tier behavior.

User prompt provides: conviction score + direction + regime,
current position (if any), recent cycles (memory), account balance.

JSON response: action, reasoning, suggested_rr (optional override).

FILE 2: engine/execution/agent.py

class DecisionAgent:
    def __init__(self, llm_provider: LLMProvider, config: TradingConfig):
        self._llm = llm_provider
        self._config = config

    async def decide(
        self,
        conviction: ConvictionOutput,
        market_data: MarketData,
        current_position: Position | None,
        account_balance: float,
        memory_context: str = "",
    ) -> TradeAction:
        # 1. Quick exit: conviction below threshold → SKIP without LLM call
        if conviction.conviction_score < self._config.conviction_threshold:
            return TradeAction(action="SKIP", conviction_score=conviction.conviction_score,
                              reasoning=f"Conviction {conviction.conviction_score} below threshold")

        # 2. If position open and conviction direction matches → check for ADD
        # 3. If position open and conviction direction opposes → consider CLOSE_ALL
        # 4. Format prompt with conviction + position + memory
        # 5. Call LLM
        # 6. Parse response into TradeAction
        # 7. Compute SL/TP using risk_profiles.compute_sl_tp()
        # 8. Compute position size using risk_profiles.compute_position_size()
        # 9. Run safety_checks.run_safety_checks() — may override action
        # 10. Return final TradeAction with all fields populated

Write tests in tests/unit/test_decision_agent.py:
- Test low conviction skips without LLM call
- Test LLM decision with mock response
- Test safety check overrides LLM action (e.g., pyramid gate blocks ADD)
- Test SL/TP computed correctly for the action
- Test parse failure returns SKIP

Update PROJECT_CONTEXT.md sections 2, 5, 14.
```

---

## Task 5: Memory System (Loops 1-4) [M]

**Status:** [ ] Not Started
**Depends on:** Repository Pattern (Task 2)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] CycleMemory: stores/retrieves last 5 cycles per bot (Loop 1)
- [ ] ReflectionRules: stores/retrieves rules with self-correcting scores (Loop 2)
- [ ] CrossBotSignals: stores/retrieves signals filtered by user_id (Loop 3)
- [ ] RegimeHistory: ring buffer of last 20 regime classifications (Loop 4)
- [ ] Helper: `build_memory_context()` assembles context string for agents
- [ ] Unit tests with SQLite backend
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Implement the 4 memory loops. Each reads/writes through the repository pattern.

FILE 1: engine/memory/cycle_memory.py

class CycleMemory:
    def __init__(self, cycle_repo: CycleRepository):
        self._repo = cycle_repo

    async def save_cycle(self, bot_id: str, cycle_data: dict) -> None:
        await self._repo.save_cycle({**cycle_data, "bot_id": bot_id})

    async def get_recent(self, bot_id: str, limit: int = 5) -> list[dict]:
        return await self._repo.get_recent_cycles(bot_id, limit)

    def format_for_prompt(self, cycles: list[dict]) -> str:
        """Format recent cycles as context string for agent prompts."""
        if not cycles:
            return "No prior cycles."
        lines = []
        for c in cycles:
            lines.append(f"- {c.get('timestamp','?')}: {c.get('action','?')} "
                         f"(conviction={c.get('conviction_score','?')})")
        return "Recent cycles:\n" + "\n".join(lines)

FILE 2: engine/memory/reflection_rules.py

class ReflectionRules:
    def __init__(self, rule_repo: RuleRepository):
        self._repo = rule_repo

    async def get_active_rules(self, symbol: str, timeframe: str) -> list[dict]:
        rules = await self._repo.get_rules(symbol, timeframe)
        return [r for r in rules if r.get('active', True) and r.get('score', 0) > -2]

    async def increment_score(self, rule_id: str) -> None:
        await self._repo.update_rule_score(rule_id, +1)

    async def decrement_score(self, rule_id: str) -> None:
        result = await self._repo.update_rule_score(rule_id, -1)
        rule = await self._repo.get_rule(rule_id) if hasattr(self._repo, 'get_rule') else None
        if rule and rule.get('score', 0) <= -2:
            await self._repo.deactivate_rule(rule_id)

    def format_for_prompt(self, rules: list[dict]) -> str:
        if not rules:
            return "No learned rules for this asset."
        lines = []
        for r in rules:
            lines.append(f"- [score={r.get('score',0)}] {r.get('rule_text','?')}")
        return "Learned rules:\n" + "\n".join(lines)

FILE 3: engine/memory/cross_bot.py

class CrossBotSignals:
    def __init__(self, cross_bot_repo: CrossBotRepository):
        self._repo = cross_bot_repo

    async def publish_signal(self, user_id: str, bot_id: str,
                              symbol: str, direction: str,
                              conviction: float) -> None:
        await self._repo.save_signal({
            "user_id": user_id, "bot_id": bot_id, "symbol": symbol,
            "direction": direction, "conviction": conviction,
            "timestamp": datetime.utcnow().isoformat(),
        })

    async def get_other_bot_signals(self, symbol: str, user_id: str,
                                     limit: int = 10) -> list[dict]:
        """Get signals from OTHER bots for same symbol/user."""
        return await self._repo.get_recent_signals(symbol, user_id, limit)

    def format_for_prompt(self, signals: list[dict]) -> str:
        if not signals:
            return "No signals from other bots."
        lines = []
        for s in signals:
            lines.append(f"- Bot {s.get('bot_id','?')}: {s.get('direction','?')} "
                         f"(conviction={s.get('conviction','?')}) at {s.get('timestamp','?')}")
        return "Cross-bot signals:\n" + "\n".join(lines)

FILE 4: engine/memory/regime_history.py

class RegimeHistory:
    def __init__(self, max_size: int = 20):
        self._buffer: list[dict] = []
        self._max = max_size

    def add(self, regime: str, confidence: float) -> None:
        self._buffer.append({
            "regime": regime, "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat(),
        })
        if len(self._buffer) > self._max:
            self._buffer.pop(0)

    def get_history(self) -> list[dict]:
        return list(self._buffer)

    def detect_transition(self) -> str | None:
        """Detect if regime just changed."""
        if len(self._buffer) < 2:
            return None
        prev = self._buffer[-2]["regime"]
        curr = self._buffer[-1]["regime"]
        if prev != curr:
            return f"{prev} -> {curr}"
        return None

    def format_for_prompt(self) -> str:
        if not self._buffer:
            return "No regime history."
        transition = self.detect_transition()
        recent = self._buffer[-5:] if len(self._buffer) >= 5 else self._buffer
        lines = [f"- {r['regime']} (conf={r['confidence']})" for r in recent]
        header = "Regime history (last 5):\n" + "\n".join(lines)
        if transition:
            header += f"\n⚠️ REGIME TRANSITION: {transition}"
        return header

FILE 5: engine/memory/__init__.py

async def build_memory_context(
    cycle_mem: CycleMemory,
    rules: ReflectionRules,
    cross_bot: CrossBotSignals,
    regime: RegimeHistory,
    bot_id: str,
    symbol: str,
    timeframe: str,
    user_id: str,
) -> str:
    """Assemble full memory context string for ConvictionAgent/DecisionAgent."""
    recent_cycles = await cycle_mem.get_recent(bot_id)
    active_rules = await rules.get_active_rules(symbol, timeframe)
    other_signals = await cross_bot.get_other_bot_signals(symbol, user_id)

    parts = [
        cycle_mem.format_for_prompt(recent_cycles),
        rules.format_for_prompt(active_rules),
        cross_bot.format_for_prompt(other_signals),
        regime.format_for_prompt(),
    ]
    return "\n\n".join(parts)

Write tests in tests/unit/test_memory.py:
- Test CycleMemory save and retrieve
- Test ReflectionRules score increment/decrement/deactivation
- Test CrossBotSignals user_id filtering
- Test RegimeHistory ring buffer overflow
- Test RegimeHistory transition detection
- Test build_memory_context assembles all 4 parts
- Test format_for_prompt outputs readable strings

Update PROJECT_CONTEXT.md sections 2, 14.
```

---

## Task 6: Pipeline Orchestrator [L]

**Status:** [ ] Not Started
**Depends on:** All previous tasks this week + Week 1-2
**Estimated time:** 90 minutes

**Acceptance criteria:**
- [ ] `AnalysisPipeline` orchestrates: Data → Signal → Conviction → Execution
- [ ] Uses Event Bus to emit events at each stage
- [ ] All signal producers run in parallel via SignalRegistry
- [ ] Memory context assembled and injected into ConvictionAgent + DecisionAgent
- [ ] Full cycle data captured for data moat (cycle record saved to repository)
- [ ] Error handling: any stage failure → SKIP, log error, emit CycleCompleted
- [ ] Integration test with mock LLM + mock exchange
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

This is the central orchestrator — the heart of the engine.

FILE: engine/pipeline.py

class AnalysisPipeline:
    def __init__(
        self,
        ohlcv_fetcher: OHLCVFetcher,
        flow_agent: FlowAgent,
        signal_registry: SignalRegistry,
        conviction_agent: ConvictionAgent,
        decision_agent: DecisionAgent,
        event_bus: EventBus,
        cycle_memory: CycleMemory,
        reflection_rules: ReflectionRules,
        cross_bot: CrossBotSignals,
        regime_history: RegimeHistory,
        cycle_repo: CycleRepository,
        config: TradingConfig,
        bot_id: str,
        user_id: str,
    ):
        # Store all dependencies

    async def run_cycle(self) -> TradeAction:
        """Run one complete analysis cycle."""
        symbol = self._config.symbol
        timeframe = self._config.timeframe

        try:
            # ── STAGE 1: DATA ──
            logger.info(f"[{symbol}/{timeframe}] Stage 1: Fetching data")
            market_data = await self._ohlcv.fetch(symbol, timeframe)

            # Enrich with flow data
            flow = await self._flow_agent.fetch_flow(symbol, self._ohlcv.adapter)
            market_data.flow = flow

            await self._bus.publish(DataReady(
                timestamp=datetime.utcnow(), source="pipeline",
                market_data=market_data
            ))

            # ── STAGE 2: SIGNALS ──
            logger.info(f"[{symbol}/{timeframe}] Stage 2: Running signal agents")
            signals = await self._signal_registry.run_all(market_data)

            if not signals:
                logger.warning("No signals produced — all agents failed")
                return self._skip_action("No signals produced")

            await self._bus.publish(SignalsReady(
                timestamp=datetime.utcnow(), source="pipeline",
                signals=signals
            ))

            # ── STAGE 3: CONVICTION ──
            logger.info(f"[{symbol}/{timeframe}] Stage 3: Evaluating conviction")
            memory_context = await build_memory_context(
                self._cycle_mem, self._rules, self._cross_bot,
                self._regime, self._bot_id, symbol, timeframe, self._user_id
            )

            conviction = await self._conviction.evaluate(
                signals=signals,
                market_data=market_data,
                memory_context=memory_context,
            )

            # Update regime history
            self._regime.add(conviction.regime, conviction.regime_confidence)

            await self._bus.publish(ConvictionScored(
                timestamp=datetime.utcnow(), source="pipeline",
                conviction=conviction
            ))

            # ── STAGE 4: EXECUTION DECISION ──
            logger.info(f"[{symbol}/{timeframe}] Stage 4: Making decision "
                        f"(conviction={conviction.conviction_score:.2f})")

            # Get current position from exchange (via adapter)
            current_position = None  # TODO: fetch from adapter
            balance = 10000.0  # TODO: fetch from adapter

            action = await self._decision.decide(
                conviction=conviction,
                market_data=market_data,
                current_position=current_position,
                account_balance=balance,
                memory_context=memory_context,
            )

            # ── RECORD CYCLE ──
            cycle_record = {
                "bot_id": self._bot_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.utcnow().isoformat(),
                "indicators_json": str(market_data.indicators),
                "signals_json": str([s.to_dict() for s in signals]),
                "conviction_json": str(conviction.to_dict()),
                "action": action.action,
                "conviction_score": conviction.conviction_score,
            }
            await self._cycle_repo.save_cycle(cycle_record)
            await self._cycle_mem.save_cycle(self._bot_id, cycle_record)

            # Publish cross-bot signal
            if conviction.direction in ("LONG", "SHORT"):
                await self._cross_bot.publish_signal(
                    self._user_id, self._bot_id, symbol,
                    conviction.direction, conviction.conviction_score
                )

            await self._bus.publish(CycleCompleted(
                timestamp=datetime.utcnow(), source="pipeline",
                symbol=symbol, action=action.action,
                conviction=conviction.conviction_score
            ))

            logger.info(f"[{symbol}/{timeframe}] Cycle complete: "
                        f"{action.action} (conviction={conviction.conviction_score:.2f})")
            return action

        except Exception as e:
            logger.error(f"[{symbol}/{timeframe}] Pipeline error: {e}", exc_info=True)
            return self._skip_action(f"Pipeline error: {e}")

    def _skip_action(self, reason: str) -> TradeAction:
        return TradeAction(
            action="SKIP", conviction_score=0.0,
            position_size=None, sl_price=None, tp1_price=None,
            tp2_price=None, rr_ratio=None, atr_multiplier=None,
            reasoning=reason, raw_output=""
        )

Write tests in tests/integration/test_pipeline.py:
- Create full pipeline with mock LLM, mock adapter, SQLite repos
- Test successful cycle: data fetched → signals produced → conviction scored → action decided
- Test pipeline handles agent failure gracefully (one agent fails, others run)
- Test pipeline handles conviction parse failure (returns SKIP)
- Test cycle record saved to repository
- Test events emitted at each stage (subscribe mock handlers)
- Test cross-bot signal published when conviction is directional

Update PROJECT_CONTEXT.md sections 2, 14.
```

---

## Task 7: First End-to-End Test Script [S]

**Status:** [ ] Not Started
**Depends on:** All previous tasks
**Estimated time:** 30 minutes

**Acceptance criteria:**
- [ ] Script that runs one full cycle with REAL Hyperliquid data + REAL Claude API
- [ ] Prints: market data summary, each agent's signal, conviction score, final action
- [ ] Uses SQLite for storage (no PostgreSQL needed to test)
- [ ] Can run with: `python scripts/run_cycle.py --symbol BTC-USDC --timeframe 1h`
- [ ] Clearly formatted output showing the full pipeline flow
- [ ] PROJECT_CONTEXT.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Create a script that runs one complete analysis cycle end-to-end.
This is the moment of truth — real data, real LLM calls, real signals.

FILE: scripts/run_cycle.py

#!/usr/bin/env python3
"""Run a single analysis cycle end-to-end.

Usage:
    python scripts/run_cycle.py --symbol BTC-USDC --timeframe 1h
    python scripts/run_cycle.py --symbol ETH-USDC --timeframe 4h
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

async def main(symbol: str, timeframe: str):
    print(f"\n{'='*60}")
    print(f"  QuantAgent v2 — Analysis Cycle")
    print(f"  Symbol: {symbol} | TF: {timeframe}")
    print(f"  Time: {datetime.utcnow().isoformat()}")
    print(f"{'='*60}\n")

    # 1. Initialize components
    from engine.config import TradingConfig, DEFAULT_PROFILES
    from engine.events import InProcessBus
    from engine.data.ohlcv import OHLCVFetcher
    from engine.data.flow import FlowAgent
    from engine.data.flow.crypto import CryptoFlowProvider
    from engine.signals.registry import SignalRegistry
    from engine.signals.indicator_agent import IndicatorAgent
    from engine.signals.pattern_agent import PatternAgent
    from engine.signals.trend_agent import TrendAgent
    from engine.conviction.agent import ConvictionAgent
    from engine.execution.agent import DecisionAgent
    from engine.memory.cycle_memory import CycleMemory
    from engine.memory.reflection_rules import ReflectionRules
    from engine.memory.cross_bot import CrossBotSignals
    from engine.memory.regime_history import RegimeHistory
    from engine.pipeline import AnalysisPipeline
    from exchanges.factory import ExchangeFactory
    from llm.claude import ClaudeProvider
    from storage.repositories import get_repositories

    config = TradingConfig(symbol=symbol, timeframe=timeframe)
    bus = InProcessBus()
    adapter = ExchangeFactory.get_adapter("hyperliquid")
    llm = ClaudeProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Repositories (SQLite for dev)
    repos = await get_repositories("sqlite")

    # Data layer
    fetcher = OHLCVFetcher(adapter, config)
    flow_agent = FlowAgent([CryptoFlowProvider()])

    # Signal layer
    registry = SignalRegistry()
    from engine.config import FeatureFlags
    flags = FeatureFlags()
    registry.register(IndicatorAgent(llm, flags))
    registry.register(PatternAgent(llm, flags))
    registry.register(TrendAgent(llm, flags))

    # Conviction + Decision
    conviction_agent = ConvictionAgent(llm)
    decision_agent = DecisionAgent(llm, config)

    # Memory
    cycle_mem = CycleMemory(repos.cycles)
    rules = ReflectionRules(repos.rules)
    cross_bot = CrossBotSignals(repos.cross_bot)
    regime = RegimeHistory()

    # Pipeline
    pipeline = AnalysisPipeline(
        ohlcv_fetcher=fetcher,
        flow_agent=flow_agent,
        signal_registry=registry,
        conviction_agent=conviction_agent,
        decision_agent=decision_agent,
        event_bus=bus,
        cycle_memory=cycle_mem,
        reflection_rules=rules,
        cross_bot=cross_bot,
        regime_history=regime,
        cycle_repo=repos.cycles,
        config=config,
        bot_id="test-bot-001",
        user_id="dev-user",
    )

    # 2. Run cycle
    print("Running analysis cycle...\n")
    action = await pipeline.run_cycle()

    # 3. Print results
    print(f"\n{'='*60}")
    print(f"  RESULT: {action.action}")
    print(f"  Conviction: {action.conviction_score:.2f}")
    if action.sl_price:
        print(f"  SL: {action.sl_price}")
        print(f"  TP1: {action.tp1_price}")
        print(f"  TP2: {action.tp2_price}")
        print(f"  RR: {action.rr_ratio}")
    print(f"  Reasoning: {action.reasoning}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC-USDC")
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()
    asyncio.run(main(args.symbol, args.timeframe))

This script is NOT a test — it runs real API calls. You need:
- ANTHROPIC_API_KEY in .env
- HYPERLIQUID_WALLET_ADDRESS in .env (read-only is fine, no trading)

No automated tests for this task — it's a manual verification script.
Update PROJECT_CONTEXT.md section 14 changelog.
```

---

## End of Week 3 Checklist

- [ ] pytest passes all tests (target: ~500+)
- [ ] FlowAgent fetches funding rate + OI from Hyperliquid
- [ ] Repository pattern saves/retrieves from SQLite
- [ ] ConvictionAgent produces conviction scores from real signals
- [ ] DecisionAgent produces trade actions with SL/TP
- [ ] Memory system stores cycles, rules, cross-bot signals, regime history
- [ ] Pipeline orchestrates full Data → Signal → Conviction → Execution flow
- [ ] `scripts/run_cycle.py` runs end-to-end with real data + real LLM
- [ ] Events emitted at every pipeline stage
- [ ] Cycle records saved to database
- [ ] No SQL imports in engine/ (all through repositories)
- [ ] PROJECT_CONTEXT.md fully updated

**If all pass:** The engine works. It can analyze any market and produce a conviction-scored trade decision. Week 4 adds the Executor (actual trade placement), ReflectionAgent, and Sentinel.

---

## Week 4 Preview

- Executor: actual order placement via exchange adapter
- ReflectionAgent: post-trade rule distillation
- Sentinel system: WebSocket monitor + readiness scoring
- Ephemeral TraderBot lifecycle manager
- First live trade on Hyperliquid testnet
