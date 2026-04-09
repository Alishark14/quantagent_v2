"""Orchestrates the Data -> Signal -> Conviction -> Execution pipeline.

The central orchestrator — the heart of the engine. Each run_cycle() call
executes one complete analysis cycle through all 4 stages, recording the
result and emitting events at each stage boundary.

On any error, returns SKIP (never LONG/SHORT on failure).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from engine.config import TradingConfig
from engine.conviction.agent import ConvictionAgent
from engine.data.flow import FlowAgent
from engine.data.ohlcv import OHLCVFetcher
from engine.events import (
    ConvictionScored,
    CycleCompleted,
    DataReady,
    EventBus,
    SignalsReady,
)
from engine.execution.agent import DecisionAgent
from engine.memory import build_memory_context
from engine.memory.cross_bot import CrossBotSignals
from engine.memory.cycle_memory import CycleMemory
from engine.memory.reflection_rules import ReflectionRules
from engine.memory.regime_history import RegimeHistory
from engine.signals.registry import SignalRegistry
from engine.types import TradeAction
from storage.repositories.base import CycleRepository

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Orchestrates one complete Data -> Signal -> Conviction -> Execution cycle."""

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
    ) -> None:
        self._ohlcv = ohlcv_fetcher
        self._flow_agent = flow_agent
        self._signal_registry = signal_registry
        self._conviction = conviction_agent
        self._decision = decision_agent
        self._bus = event_bus
        self._cycle_mem = cycle_memory
        self._rules = reflection_rules
        self._cross_bot = cross_bot
        self._regime = regime_history
        self._cycle_repo = cycle_repo
        self._config = config
        self._bot_id = bot_id
        self._user_id = user_id

    async def run_cycle(self) -> TradeAction:
        """Run one complete analysis cycle through all 4 stages.

        Stages:
        1. DATA — Fetch OHLCV + flow, emit DataReady
        2. SIGNALS — Run all signal producers in parallel, emit SignalsReady
        3. CONVICTION — Evaluate signal consensus, emit ConvictionScored
        4. EXECUTION — Decide trade action based on conviction

        Returns:
            TradeAction with the decided action. On any failure, returns SKIP.
        """
        symbol = self._config.symbol
        timeframe = self._config.timeframe

        try:
            # ── STAGE 1: DATA ──
            logger.info(f"[{symbol}/{timeframe}] Stage 1: Fetching data")
            market_data = await self._ohlcv.fetch(symbol, timeframe)

            # Enrich with flow data
            try:
                flow = await self._flow_agent.fetch_flow(symbol, self._ohlcv._adapter)
                market_data.flow = flow
            except Exception:
                logger.warning(f"[{symbol}/{timeframe}] Flow data fetch failed, continuing without")

            await self._bus.publish(DataReady(
                source="pipeline",
                market_data=market_data,
            ))

            # ── STAGE 2: SIGNALS ──
            logger.info(f"[{symbol}/{timeframe}] Stage 2: Running signal agents")
            signals = await self._signal_registry.run_all(market_data)

            if not signals:
                logger.warning(f"[{symbol}/{timeframe}] No signals produced — all agents failed")
                return self._skip_action("No signals produced")

            await self._bus.publish(SignalsReady(
                source="pipeline",
                signals=signals,
            ))

            # ── STAGE 3: CONVICTION ──
            logger.info(f"[{symbol}/{timeframe}] Stage 3: Evaluating conviction")
            memory_context = await build_memory_context(
                self._cycle_mem, self._rules, self._cross_bot,
                self._regime, self._bot_id, symbol, timeframe, self._user_id,
            )

            conviction = await self._conviction.evaluate(
                signals=signals,
                market_data=market_data,
                memory_context=memory_context,
            )

            # Update regime history
            self._regime.add(conviction.regime, conviction.regime_confidence)

            await self._bus.publish(ConvictionScored(
                source="pipeline",
                conviction=conviction,
            ))

            # ── STAGE 4: EXECUTION DECISION ──
            logger.info(
                f"[{symbol}/{timeframe}] Stage 4: Making decision "
                f"(conviction={conviction.conviction_score:.2f})"
            )

            # TODO: fetch from exchange adapter in production
            current_position = None
            balance = self._config.account_balance or 10000.0

            action = await self._decision.decide(
                conviction=conviction,
                market_data=market_data,
                current_position=current_position,
                account_balance=balance,
                memory_context=memory_context,
            )

            # ── RECORD CYCLE ──
            # NOTE: timestamp is a raw datetime object (not isoformat string).
            # PostgreSQL TIMESTAMPTZ via asyncpg requires a datetime; SQLite
            # accepts both. Stringifying here breaks PG with `asyncpg.exceptions.DataError`.
            cycle_record = {
                "bot_id": self._bot_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now(timezone.utc),
                "indicators": market_data.indicators,
                "signals": [s.to_dict() for s in signals],
                "conviction": conviction.to_dict(),
                "action": action.action,
                "conviction_score": conviction.conviction_score,
            }
            await self._cycle_mem.save_cycle(self._bot_id, cycle_record)

            # Publish cross-bot signal for directional convictions
            if conviction.direction in ("LONG", "SHORT"):
                try:
                    await self._cross_bot.publish_signal(
                        self._user_id, self._bot_id, symbol,
                        conviction.direction, conviction.conviction_score,
                    )
                except Exception:
                    logger.warning("Failed to publish cross-bot signal", exc_info=True)

            await self._bus.publish(CycleCompleted(
                source="pipeline",
                symbol=symbol,
                action=action.action,
                conviction=conviction.conviction_score,
            ))

            logger.info(
                f"[{symbol}/{timeframe}] Cycle complete: "
                f"{action.action} (conviction={conviction.conviction_score:.2f})"
            )
            return action

        except Exception as e:
            logger.error(f"[{symbol}/{timeframe}] Pipeline error: {e}", exc_info=True)
            return self._skip_action(f"Pipeline error: {e}")

    def _skip_action(self, reason: str) -> TradeAction:
        """Return a safe SKIP action. SKIP is always safe."""
        return TradeAction(
            action="SKIP",
            conviction_score=0.0,
            position_size=None,
            sl_price=None,
            tp1_price=None,
            tp2_price=None,
            rr_ratio=None,
            atr_multiplier=None,
            reasoning=reason,
            raw_output="",
        )
