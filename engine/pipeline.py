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
from engine.execution.portfolio_risk_manager import PortfolioRiskManager
from engine.memory import build_memory_context
from engine.memory.cross_bot import CrossBotSignals
from engine.memory.cycle_memory import CycleMemory
from engine.memory.reflection_rules import ReflectionRules
from engine.memory.regime_history import RegimeHistory
from engine.signals.registry import SignalRegistry
from engine.types import MarketData, OrderResult, TradeAction
from storage.repositories.base import CycleRepository, TradeRepository

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
        is_shadow: bool = False,
        portfolio_risk_manager: PortfolioRiskManager | None = None,
        shadow_fixed_size_usd: float | None = None,
        trade_repo: TradeRepository | None = None,
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
        # ``is_shadow`` flags every cycle this pipeline persists as
        # belonging to the data moat's shadow partition. Set to True
        # for shadow-mode AND paper-mode bots — both trade with fake
        # money (virtual portfolio for shadow, testnet orderbook for
        # paper) and must be excluded from the live alpha-mining
        # views. Defaults to False so existing test fixtures and the
        # live production path remain byte-for-byte unchanged.
        self._is_shadow = is_shadow
        # Shadow signal-quality data collection: when set, entry actions
        # bypass the PortfolioRiskManager entirely and use this fixed
        # dollar size. The PRM's per-asset / portfolio exposure caps
        # would otherwise block every shadow trade once a few simulated
        # positions accumulated (they never close in pure shadow until
        # the Sentinel SL/TP monitor lands), starving the data moat of
        # signal samples. Pure shadow mode (sim adapter, fake money)
        # opts in; paper mode (real testnet orders) does NOT — its PRM
        # validation is the whole point of paper trading.
        self._shadow_fixed_size_usd = shadow_fixed_size_usd
        # Trade repository for shadow trade lifecycle persistence. Live
        # trades flow through native exchange SL/TP orders and don't
        # need DB-backed monitoring; shadow trades do (positions live
        # in SimulatedExchangeAdapter memory, never close on their own,
        # don't survive restarts). When this repo is wired AND
        # ``shadow_fixed_size_usd`` is set, ``record_trade_open`` is
        # the path TraderBot calls after a successful execution.
        self._trade_repo = trade_repo

        # Sprint Portfolio-Risk-Manager Task 4: PortfolioRiskManager
        # owns ALL position sizing for entry actions. Optional in the
        # constructor so existing test fixtures don't have to thread
        # one through; production paths (`_make_bot_factory` in
        # quantagent/main.py) construct one PRM per bot and pass it
        # explicitly. When None, the pipeline never sizes positions
        # — ``action.position_size`` stays as DecisionAgent emitted
        # it (always None now that Task 1 stripped sizing), so the
        # executor will refuse to place orders. We log a WARNING at
        # construction so a misconfigured production path is loud.
        self._prm = portfolio_risk_manager
        if self._prm is None:
            logger.warning(
                f"AnalysisPipeline[{bot_id}/{config.symbol}] constructed "
                "WITHOUT PortfolioRiskManager — entry actions will have "
                "position_size=None and the executor will refuse to place "
                "orders. Wire a PRM via the constructor for production runs."
            )

        # Per-bot peak-equity tracker for the drawdown throttle.
        # Initialised to 0; the first cycle's balance becomes the
        # peak via the post-PRM update path. In-memory only — resets
        # on bot restart. Future enhancement: persist via bot
        # config_json so a restart doesn't lose the high-water mark.
        self._peak_equity: float = 0.0

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

            # Feed the current price to the adapter so simulated
            # adapters (SimulatedExchangeAdapter) can price fills.
            # No-op on real adapters — they don't have this method.
            if market_data.candles:
                _feed_price = float(market_data.candles[-1]["close"])
                adapter = self._ohlcv._adapter
                if hasattr(adapter, "set_current_prices"):
                    adapter.set_current_prices({symbol: _feed_price})

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

            # TODO: fetch open positions from exchange adapter in production
            current_position = None

            # DecisionAgent outputs trade INTENT only — direction, SL/TP,
            # and a deterministic risk_weight derived from conviction. It
            # does NOT see the account balance and never computes dollar
            # sizing (Sprint Portfolio-Risk-Manager Task 1). Sizing is owned
            # downstream by PortfolioRiskManager (Task 4 wires it in below).
            action = await self._decision.decide(
                conviction=conviction,
                market_data=market_data,
                current_position=current_position,
                memory_context=memory_context,
            )

            # ── PORTFOLIO RISK MANAGER ──
            # Sprint Portfolio-Risk-Manager Task 4: for entry actions,
            # PRM owns the dollar sizing. It fetches the live balance
            # and open positions, runs the six-layer pipeline (drawdown
            # throttle → fixed fractional → cost floor → exposure caps),
            # and either populates ``action.position_size`` with the
            # final dollar size or converts the action to SKIP with a
            # PRM-attributed reason. Non-entry actions (HOLD / SKIP /
            # CLOSE_ALL) bypass PRM entirely — there's nothing to size.
            if (
                self._shadow_fixed_size_usd is not None
                and action.action in ("LONG", "SHORT", "ADD_LONG", "ADD_SHORT")
            ):
                # Shadow data-collection bypass: skip PRM, stamp fixed size.
                action.position_size = float(self._shadow_fixed_size_usd)
            elif (
                self._prm is not None
                and action.action in ("LONG", "SHORT", "ADD_LONG", "ADD_SHORT")
            ):
                action = await self._apply_prm(action, market_data)

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
                # Mark this cycle as belonging to the shadow data
                # partition for shadow-mode and paper-mode bots so the
                # live_cycles view (and the QuantDataScientist mining
                # job that consumes it) never see fake-money fills.
                "is_shadow": self._is_shadow,
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
            risk_weight=None,
        )

    async def _apply_prm(
        self, action: TradeAction, market_data: MarketData
    ) -> TradeAction:
        """Run PortfolioRiskManager for an entry action.

        Fetches live balance + open positions from the exchange adapter,
        normalises the positions into the dict shape PRM expects, runs
        the six-layer sizing pipeline, and either:

        * Mutates ``action.position_size`` with the PRM-computed dollar
          size and returns the action (happy path), or
        * Returns a SKIP action with the PRM skip reason appended to
          the original reasoning (PRM-rejected path), preserving the
          conviction_score so cycle records still capture it.

        Errors fetching portfolio state convert to SKIP with a clear
        reason. SKIP-on-error is preferred here over the legacy "fall
        back to $10k constant" pattern because PRM needs accurate
        state to make safe sizing decisions — running PRM on stale or
        missing portfolio state is worse than skipping the cycle.
        """
        symbol = self._config.symbol

        # Fetch portfolio state from the exchange. Both calls go via
        # the OHLCVFetcher's adapter handle (the engine has zero direct
        # exchange couplings — every external call flows through one
        # of the data-layer modules per CLAUDE.md Rule 3). Wrapped in
        # try/except so a transient API hiccup doesn't crash the cycle.
        try:
            adapter = self._ohlcv._adapter
            balance = await adapter.get_balance()
        except Exception as e:
            logger.warning(
                f"[{symbol}] PRM SKIP: failed to fetch balance: {e}"
            )
            return self._convert_to_skip(action, f"PRM balance fetch failed: {e}")

        if not balance or balance <= 0:
            logger.warning(
                f"[{symbol}] PRM SKIP: non-positive balance {balance!r}"
            )
            return self._convert_to_skip(
                action, f"PRM saw non-positive balance ({balance!r})"
            )

        try:
            positions = await adapter.get_positions()
        except Exception as e:
            logger.warning(
                f"[{symbol}] PRM SKIP: failed to fetch positions: {e}"
            )
            return self._convert_to_skip(action, f"PRM positions fetch failed: {e}")

        # Normalise the adapter's Position objects into the dict
        # shape PRM expects. Use abs(size * entry_price) as the
        # notional — direction-agnostic because PRM treats long and
        # short capacity equally (no netting at the cap layer).
        open_positions: list[dict] = []
        for p in positions or []:
            try:
                open_positions.append({
                    "symbol": p.symbol,
                    "notional": abs(p.size * p.entry_price),
                    "direction": p.direction,
                })
            except (AttributeError, TypeError):
                # Defensive: a malformed Position from the adapter
                # shouldn't crash the whole cycle. Skip + log so
                # operators can spot the bad row in the audit trail.
                logger.warning(
                    f"[{symbol}] PRM: skipping malformed position {p!r}"
                )

        # Compute SL/TP distances as fractions of the current price.
        # PRM consumes them as positive percentages so it works the
        # same way for LONG and SHORT trades.
        if not market_data.candles:
            return self._convert_to_skip(action, "PRM: no candles for current price")
        try:
            current_price = float(market_data.candles[-1]["close"])
        except (KeyError, TypeError, ValueError) as e:
            return self._convert_to_skip(action, f"PRM: bad current price ({e})")

        if current_price <= 0:
            return self._convert_to_skip(
                action, f"PRM: non-positive current price ({current_price!r})"
            )

        if action.sl_price is None or action.tp1_price is None:
            return self._convert_to_skip(
                action,
                f"PRM: missing SL/TP (sl={action.sl_price}, tp1={action.tp1_price})",
            )

        sl_distance_pct = abs(current_price - action.sl_price) / current_price
        tp1_distance_pct = abs(action.tp1_price - current_price) / current_price

        # On the very first cycle ``_peak_equity`` is 0, so feed PRM
        # the current balance as the peak — drawdown will be 0 and
        # the multiplier will be 1.0. Subsequent cycles use the
        # accumulated peak so drawdowns are computed against the
        # all-time high. The peak update happens AFTER PRM runs (see
        # below) so the same cycle uses the OLD peak — though when
        # balance ≥ peak the result is identical either way.
        peak_for_prm = self._peak_equity if self._peak_equity > 0 else balance

        # risk_weight is None on the SKIP/HOLD/CLOSE_ALL paths but
        # we already gated on entry actions above, so DecisionAgent
        # should have set it to one of the four conviction-band
        # values. Defensive fallback to 1.0 if it's somehow missing.
        risk_weight = action.risk_weight if action.risk_weight else 1.0

        sizing = self._prm.size_trade(  # type: ignore[union-attr]
            equity=float(balance),
            peak_equity=peak_for_prm,
            sl_distance_pct=sl_distance_pct,
            tp1_distance_pct=tp1_distance_pct,
            risk_weight=risk_weight,
            symbol=symbol,
            open_positions=open_positions,
        )

        # Update peak equity AFTER PRM has computed its multiplier so
        # this cycle uses the OLD peak. (When balance ≥ old peak the
        # result is identical either way; when balance < old peak we
        # don't update so the order doesn't matter; the only case
        # where it matters is when balance just hit a new high, and
        # we want this cycle to size against the OLD peak so the
        # update can't accidentally erase a recent drawdown.)
        if balance > self._peak_equity:
            self._peak_equity = float(balance)

        if sizing.skipped:
            logger.info(f"[{symbol}] PRM SKIP: {sizing.skip_reason}")
            return self._convert_to_skip(action, f"PRM: {sizing.skip_reason}")

        # Happy path: stamp the PRM-computed dollar size onto the
        # action so the executor can convert it to base units.
        action.position_size = sizing.position_size_usd
        logger.info(
            f"[{symbol}] PRM: size=${sizing.position_size_usd:,.2f} "
            f"(risk=${sizing.risk_dollars:,.2f}, weight={risk_weight:.2f}, "
            f"DD_mult={sizing.drawdown_multiplier:.2f})"
        )
        return action

    async def record_trade_open(
        self, action: TradeAction, order_result: OrderResult
    ) -> str | None:
        """Persist a freshly-opened shadow trade to the trade repository.

        Called by TraderBot after the executor returns a successful
        OrderResult for a LONG / SHORT entry. The pipeline owns the
        write because it has the full context (bot_id, user_id,
        is_shadow, conviction, engine_version) that TraderBot does not.

        Returns the new trade row ID, or ``None`` when the call is a
        no-op (no trade_repo wired, not shadow data-collection mode,
        unsuccessful order, non-entry action). Errors are logged and
        swallowed — a persistence failure must NEVER bubble up and
        crash the trading loop, per CLAUDE.md fire-and-forget rules.
        """
        if self._trade_repo is None:
            return None
        if self._shadow_fixed_size_usd is None:
            # Only the shadow data-collection path persists trades
            # via this code path right now. Live trades flow through
            # the exchange's native SL/TP orders and are persisted
            # by the existing tracking pipeline (when wired).
            return None
        if action.action not in ("LONG", "SHORT"):
            return None
        if not order_result.success:
            return None

        from uuid import uuid4

        from quantagent.version import ENGINE_VERSION

        symbol = self._config.symbol
        timeframe = self._config.timeframe
        direction = action.action  # "LONG" | "SHORT"
        entry_price = order_result.fill_price
        if entry_price is None or entry_price <= 0:
            logger.warning(
                f"[{symbol}] record_trade_open: missing/invalid fill_price "
                f"({entry_price!r}) — skipping trade persist"
            )
            return None
        # tp_price column collapses tp1/tp2 into a single primary level.
        # Prefer tp1 (the conservative target the executor sells half at).
        tp_price = action.tp1_price

        trade_id = str(uuid4())
        trade_row = {
            "id": trade_id,
            "user_id": self._user_id,
            "bot_id": self._bot_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "entry_price": float(entry_price),
            "size": float(action.position_size or 0.0),
            "sl_price": float(action.sl_price) if action.sl_price else None,
            "tp_price": float(tp_price) if tp_price else None,
            "conviction_score": action.conviction_score,
            "entry_time": datetime.now(timezone.utc),
            "status": "open",
            "is_shadow": True,
            "engine_version": ENGINE_VERSION,
        }

        try:
            await self._trade_repo.save_trade(trade_row)
        except Exception:
            logger.exception(
                f"[{symbol}] record_trade_open: failed to persist shadow trade"
            )
            return None

        logger.info(
            f"Shadow trade OPENED: {symbol} {direction} @ {entry_price}, "
            f"SL={action.sl_price}, TP={tp_price}, size=${action.position_size}"
        )
        return trade_id

    @staticmethod
    def _convert_to_skip(action: TradeAction, reason: str) -> TradeAction:
        """Convert a (typically entry) TradeAction to a safe SKIP.

        Preserves ``conviction_score`` and ``raw_output`` so cycle
        records still capture what DecisionAgent originally decided,
        but zeroes out every sizing field so the executor can't act
        on stale SL/TP/risk_weight from the pre-conversion action.
        The PRM rejection reason is appended to the original
        reasoning so audit logs show BOTH what the LLM intended AND
        why PRM blocked it.
        """
        return TradeAction(
            action="SKIP",
            conviction_score=action.conviction_score,
            position_size=None,
            sl_price=None,
            tp1_price=None,
            tp2_price=None,
            rr_ratio=None,
            atr_multiplier=None,
            reasoning=f"{action.reasoning} [PRM override: {reason}]",
            raw_output=action.raw_output,
            risk_weight=None,
        )
