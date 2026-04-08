"""BacktestEngine — Tier 1 mechanical backtest orchestrator.

Feeds historical candles through the mechanical layers of the engine
(indicators, swing detection, ReadinessScorer, risk profiles, position
sizing, sim execution) candle-by-candle, with LLM agents replaced by
``MockSignalProducer`` so the run is deterministic, free, and fast.

The Tier 1 contract per ARCHITECTURE.md §31.3.1: this validates the
**math**, not the agents. ConvictionAgent and DecisionAgent are NOT
invoked in mechanical mode — the mock signal direction maps directly to
a TradeAction with SL/TP / size computed by the existing
``risk_profiles`` and ``compute_position_size`` helpers.

Full LLM mode (Tier 3) is reserved for a later sprint and currently
raises ``NotImplementedError`` from ``run()``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from backtesting.data_loader import ParquetDataLoader
from backtesting.mock_signals import MockSignalProducer
from backtesting.sim_exchange import SimulatedExchangeAdapter
from engine.execution.cost_model import ExecutionCostModel
from engine.execution.cost_models import get_cost_model
from backtesting.sim_executor import SimExecutor
from engine.config import DEFAULT_PROFILES
from engine.data.indicators import compute_all_indicators
from engine.data.swing_detection import find_swing_highs, find_swing_lows
from engine.events import (
    CycleCompleted,
    EventBus,
    InProcessBus,
    SetupDetected,
    TradeOpened,
)
from engine.execution.risk_profiles import compute_position_size, compute_sl_tp
from engine.types import MarketData, TradeAction
from sentinel.conditions import ReadinessScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """All inputs to a backtest run."""

    symbols: list[str]
    timeframes: list[str]
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10_000.0
    mode: str = "mechanical"  # "mechanical" | "full"
    slippage_pct: float = 0.0005
    funding_interval_hours: int = 8
    funding_rate: float = 0.0001  # flat 0.01% per period (longs pay)
    risk_per_trade: float = 0.01  # 1% account risk per trade
    max_position_pct: float = 0.5
    readiness_threshold: float = 0.30  # min readiness score to consider entry
    indicator_lookback: int = 100  # rolling window for indicator computation
    min_warmup_candles: int = 50  # skip first N (MACD needs slow=26 + signal=9)
    exchange: str = "hyperliquid"  # for ParquetDataLoader path resolution

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("symbols must be non-empty")
        if not self.timeframes:
            raise ValueError("timeframes must be non-empty")
        if self.end_date <= self.start_date:
            raise ValueError(
                f"end_date ({self.end_date}) must be after start_date ({self.start_date})"
            )
        if self.initial_balance <= 0:
            raise ValueError(f"initial_balance must be > 0, got {self.initial_balance}")
        if self.mode not in ("mechanical", "full"):
            raise ValueError(f"mode must be 'mechanical' or 'full', got {self.mode!r}")


@dataclass
class BacktestResult:
    """Output of a backtest run. Serialisable to JSON via ``to_dict``."""

    config: BacktestConfig
    trade_history: list[dict]
    equity_curve: list[tuple[int, float]]
    duration_seconds: float
    candles_processed: int
    final_balance: float
    initial_balance: float
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config": {
                "symbols": self.config.symbols,
                "timeframes": self.config.timeframes,
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "initial_balance": self.config.initial_balance,
                "mode": self.config.mode,
                "slippage_pct": self.config.slippage_pct,
                "funding_interval_hours": self.config.funding_interval_hours,
                "exchange": self.config.exchange,
            },
            "duration_seconds": self.duration_seconds,
            "candles_processed": self.candles_processed,
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "trade_history": self.trade_history,
            "equity_curve": [(int(t), float(e)) for t, e in self.equity_curve],
            "metrics": self.metrics,
        }


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Drives a single backtest run from config to result."""

    def __init__(
        self,
        config: BacktestConfig,
        data_loader: ParquetDataLoader | None = None,
        signal_producer: MockSignalProducer | None = None,
        event_bus: EventBus | None = None,
        adapter: SimulatedExchangeAdapter | None = None,
        cost_model: ExecutionCostModel | None = None,
    ) -> None:
        self._config = config
        self._loader = data_loader or ParquetDataLoader(exchange=config.exchange)
        self._signal_producer = signal_producer or MockSignalProducer("always_skip")
        self._bus = event_bus or InProcessBus()

        # Pick a cost model from the config exchange so simulated fills are
        # billed at realistic rates. Callers may inject an explicit model
        # (tests, scenario sweeps, custom fee tiers); None falls through
        # to the per-exchange default. If a pre-built `adapter` is also
        # passed in we honour the adapter's existing fee model — never
        # silently overwrite caller intent.
        self._cost_model = cost_model or get_cost_model(config.exchange)

        # Inject loader into adapter so fetch_ohlcv works for any code path
        # that asks the sim adapter for historical data.
        self._adapter = adapter or SimulatedExchangeAdapter(
            initial_balance=config.initial_balance,
            slippage_pct=config.slippage_pct,
            data_loader=self._loader,
            fee_model=self._cost_model,
        )
        self._executor = SimExecutor(self._adapter)
        self._scorer = ReadinessScorer()

        # Per-(symbol, tf) rolling state
        self._windows: dict[tuple[str, str], deque[dict]] = defaultdict(
            lambda: deque(maxlen=self._config.indicator_lookback)
        )
        self._prev_macd_hist: dict[tuple[str, str], float] = {}
        self._last_funding_ms: dict[str, int] = {}
        self._candles_processed: int = 0
        self._setups_detected: int = 0
        self._setups_taken: int = 0

    # ------------------------------------------------------------------
    # Public read surface (handy for tests)
    # ------------------------------------------------------------------

    @property
    def adapter(self) -> SimulatedExchangeAdapter:
        return self._adapter

    @property
    def executor(self) -> SimExecutor:
        return self._executor

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self) -> BacktestResult:
        if self._config.mode == "full":
            raise NotImplementedError(
                "Full LLM backtest mode (Tier 3) is not yet implemented. "
                "Use mode='mechanical' for the mechanical Tier 1 path."
            )

        start_wall = time.perf_counter()
        merged = self._load_merged_stream()
        logger.info(
            f"Backtest: {len(merged)} candles across "
            f"{len(self._config.symbols)} symbols × "
            f"{len(self._config.timeframes)} timeframes"
        )

        for symbol, timeframe, candle in merged:
            await self._tick(symbol, timeframe, candle)
            self._candles_processed += 1

        duration = time.perf_counter() - start_wall
        return self._build_result(duration)

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    async def _tick(self, symbol: str, timeframe: str, candle: dict) -> None:
        # 1. Hand the candle to the sim adapter — drives SL/TP fills + equity.
        self._adapter.set_current_candle(symbol, candle)

        # 2. Funding (per-symbol cadence).
        self._maybe_apply_funding(symbol, candle)

        # 3. Maintain the rolling indicator window.
        key = (symbol, timeframe)
        window = self._windows[key]
        window.append(candle)

        if len(window) < self._config.min_warmup_candles:
            await self._publish(
                CycleCompleted(
                    source="backtest_engine",
                    symbol=symbol,
                    action="WARMUP",
                    conviction=0.0,
                )
            )
            return

        # 4. Compute indicators + swings on the rolling window.
        candles_list = list(window)
        indicators = compute_all_indicators(candles_list)
        highs = np.array([c["high"] for c in candles_list], dtype=float)
        lows = np.array([c["low"] for c in candles_list], dtype=float)
        swing_highs = find_swing_highs(highs)
        swing_lows = find_swing_lows(lows)

        # 5. Sentinel readiness gate.
        readiness, conditions = self._scorer.score(
            indicators=indicators,
            current_price=float(candle["close"]),
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            funding_rate=self._config.funding_rate,
            prev_macd_histogram=self._prev_macd_hist.get(key),
        )
        # Track MACD histogram for next-tick cross detection
        self._prev_macd_hist[key] = indicators.get("macd", {}).get("histogram", 0.0)

        action = "SKIP"
        if readiness >= self._config.readiness_threshold:
            self._setups_detected += 1
            await self._publish(
                SetupDetected(
                    source="backtest_engine",
                    symbol=symbol,
                    readiness=readiness,
                    conditions=[c.name for c in conditions if c.triggered],
                )
            )
            action = await self._maybe_enter(
                symbol=symbol,
                timeframe=timeframe,
                candle=candle,
                indicators=indicators,
                swing_highs=swing_highs,
                swing_lows=swing_lows,
            )
            if action in ("LONG", "SHORT"):
                self._setups_taken += 1

        await self._publish(
            CycleCompleted(
                source="backtest_engine",
                symbol=symbol,
                action=action,
                conviction=readiness,
            )
        )

    # ------------------------------------------------------------------
    # Decision (mechanical translation of mock signal → trade)
    # ------------------------------------------------------------------

    async def _maybe_enter(
        self,
        symbol: str,
        timeframe: str,
        candle: dict,
        indicators: dict,
        swing_highs: list[float],
        swing_lows: list[float],
    ) -> str:
        """Ask the mock producer for a direction and (maybe) open a trade.

        Returns the action label (LONG/SHORT/SKIP/HOLD) for telemetry.
        """
        # Don't stack — one position per symbol in mechanical Tier 1.
        existing = await self._adapter.get_positions(symbol)
        if existing:
            return "HOLD"

        market_data = self._build_market_data(
            symbol=symbol,
            timeframe=timeframe,
            window=list(self._windows[(symbol, timeframe)]),
            indicators=indicators,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
        )

        signal = await self._signal_producer.analyze(market_data)
        if signal is None or signal.direction not in ("BULLISH", "BEARISH"):
            return "SKIP"

        direction = "LONG" if signal.direction == "BULLISH" else "SHORT"
        entry_price = float(candle["close"])
        atr = indicators.get("atr") or 0.0
        if atr <= 0:
            return "SKIP"

        profile = DEFAULT_PROFILES.get(timeframe)
        if profile is None:
            logger.warning(f"No DEFAULT_PROFILE for {timeframe}; skipping entry")
            return "SKIP"

        sltp = compute_sl_tp(
            entry_price=entry_price,
            direction=direction,
            atr=atr,
            profile=profile,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
        )
        if sltp["sl_price"] <= 0 or sltp["risk_distance"] <= 0:
            return "SKIP"

        size_usd = compute_position_size(
            account_balance=self._adapter.balance,
            risk_per_trade=self._config.risk_per_trade,
            entry_price=entry_price,
            sl_price=sltp["sl_price"],
            max_position_pct=self._config.max_position_pct,
        )
        if size_usd <= 0:
            return "SKIP"
        size_units = size_usd / entry_price

        # Place the orders on the sim adapter.
        side = "buy" if direction == "LONG" else "sell"
        market_res = await self._adapter.place_market_order(symbol, side, size_units)
        if not market_res.success:
            logger.warning(f"Market entry failed: {market_res.error}")
            return "SKIP"

        close_side = "sell" if direction == "LONG" else "buy"
        await self._adapter.place_sl_order(
            symbol, close_side, size_units, sltp["sl_price"]
        )
        await self._adapter.place_tp_order(
            symbol, close_side, size_units, sltp["tp2_price"]
        )

        await self._publish(
            TradeOpened(
                source="backtest_engine",
                trade_action=self._build_trade_action(
                    direction=direction,
                    size_usd=size_usd,
                    sltp=sltp,
                    profile_atr_mult=profile.atr_multiplier,
                    confidence=signal.confidence,
                ),
                order_result=market_res,
            )
        )
        return direction

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_merged_stream(self) -> list[tuple[str, str, dict]]:
        """Load every (symbol, timeframe) candle stream and merge by timestamp."""
        merged: list[tuple[int, str, str, dict]] = []
        for symbol in self._config.symbols:
            for tf in self._config.timeframes:
                candles = self._loader.load_as_market_data(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=self._config.start_date,
                    end_date=self._config.end_date,
                )
                for c in candles:
                    merged.append((int(c["timestamp"]), symbol, tf, c))
        # Stable sort by timestamp; ties broken by insertion order (deterministic).
        merged.sort(key=lambda row: row[0])
        return [(s, t, c) for _, s, t, c in merged]

    def _maybe_apply_funding(self, symbol: str, candle: dict) -> None:
        ts = int(candle["timestamp"])
        last = self._last_funding_ms.get(symbol)
        interval_ms = self._config.funding_interval_hours * 3600 * 1000
        if last is None:
            # First candle for this symbol — start the clock without charging
            self._last_funding_ms[symbol] = ts
            return
        if ts - last >= interval_ms:
            self._adapter.apply_funding(symbol, self._config.funding_rate)
            self._last_funding_ms[symbol] = ts

    def _build_market_data(
        self,
        symbol: str,
        timeframe: str,
        window: list[dict],
        indicators: dict,
        swing_highs: list[float],
        swing_lows: list[float],
    ) -> MarketData:
        return MarketData(
            symbol=symbol,
            timeframe=timeframe,
            candles=window,
            num_candles=len(window),
            lookback_description=f"{len(window)} candles",
            forecast_candles=3,
            forecast_description="3 candles",
            indicators=indicators,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
        )

    def _build_trade_action(
        self,
        direction: str,
        size_usd: float,
        sltp: dict,
        profile_atr_mult: float,
        confidence: float,
    ) -> TradeAction:
        return TradeAction(
            action=direction,
            conviction_score=confidence,
            position_size=size_usd,
            sl_price=sltp["sl_price"],
            tp1_price=sltp["tp1_price"],
            tp2_price=sltp["tp2_price"],
            rr_ratio=sltp["rr_ratio"],
            atr_multiplier=profile_atr_mult,
            reasoning="backtest mechanical mode",
            raw_output="",
        )

    async def _publish(self, event) -> None:
        """Fire-and-forget publish that never propagates handler errors."""
        try:
            await self._bus.publish(event)
        except Exception:  # pragma: no cover - bus is in-process and isolates
            logger.exception(f"Failed to publish {type(event).__name__}")

    def _build_result(self, duration: float) -> BacktestResult:
        from backtesting.metrics import calculate_metrics

        history = self._adapter.get_trade_history()
        equity = self._adapter.get_equity_curve()
        final = self._adapter.equity()
        metrics_obj = calculate_metrics(
            trade_history=history,
            equity_curve=equity,
            config=self._config,
            setups_detected=self._setups_detected,
            setups_taken=self._setups_taken,
        )
        return BacktestResult(
            config=self._config,
            trade_history=history,
            equity_curve=equity,
            duration_seconds=duration,
            candles_processed=self._candles_processed,
            final_balance=final,
            initial_balance=self._config.initial_balance,
            metrics=metrics_obj.to_dict(),
        )
