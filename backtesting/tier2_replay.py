"""Tier 2 backtest: counterfactual replay against recorded trades.

ARCHITECTURE.md §31.3.2 — Tier 2 replays *recorded* analysis cycles
(captured by the live data moat) with modified mechanical parameters,
without re-running any LLM agents. The expensive part of a real cycle
(the conviction reasoning, the agent calls) is reused as-is. Only the
mechanical layers — SL/TP, trailing, break-even, conviction-threshold
filtering — are re-evaluated against the high-resolution Forward Price
Path that followed each entry.

Counterfactuals this engine answers:

- "If we'd used a tighter ATR multiplier, would the stops have survived?"
- "If we'd held longer for TP2, would the avg R have improved?"
- "If we'd raised the conviction threshold to 0.6, what % of trades would
  we have skipped — and were they winners or losers?"

Cost: zero LLM calls, zero exchange API calls. Just walks the recorded
forward paths in memory.

Recorded trade dict schema (the engine reads only these keys; everything
else is preserved verbatim in ``ReplayResult.original_outcome``):

    {
        "trade_id": str,
        "symbol": str,
        "timeframe": str,            # used for forward-path resolution
        "direction": "LONG" | "SHORT",
        "entry_timestamp": int,      # ms — anchors the forward path
        "entry_price": float,
        "size": float,               # size in units (not USD)
        "sl_price": float,
        "tp1_price": float,
        "tp2_price": float,
        "atr_at_entry": float,
        "conviction": float,         # 0.0 - 1.0
        "exit_price": float,
        "exit_timestamp": int,
        "exit_reason": str,
        "pnl": float,
    }

Modified-params dict (every key optional; missing keys mean "use the
recorded trade's value"):

    {
        "sl_price": float,            # absolute new SL
        "tp1_price": float,
        "tp2_price": float,
        "atr_multiplier": float,      # re-derive SL from atr_at_entry
        "trailing_atr_mult": float,   # enable Chandelier-style trailing
        "breakeven_after_tp1": bool,  # snap SL to entry once TP1 fires
        "conviction_threshold": float,  # filter: skip if conviction < threshold
    }
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import polars as pl

from backtesting.forward_path import ForwardPathLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """Outcome of replaying one recorded trade with modified params."""

    trade_id: str
    original_outcome: dict
    counterfactual_outcome: dict
    delta_pnl: float
    delta_r: float
    skipped: bool = False

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "original_outcome": self.original_outcome,
            "counterfactual_outcome": self.counterfactual_outcome,
            "delta_pnl": self.delta_pnl,
            "delta_r": self.delta_r,
            "skipped": self.skipped,
        }


@dataclass
class SweepRow:
    """One row of a parameter sweep table."""

    param_value: float
    num_trades: int
    num_skipped: int
    total_pnl: float
    win_rate: float
    avg_r: float
    max_drawdown: float


@dataclass
class SweepResult:
    """Result of a parameter sweep across recorded trades."""

    param_name: str
    rows: list[SweepRow] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "param_name": self.param_name,
            "rows": [asdict(r) for r in self.rows],
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Tier2ReplayEngine:
    """Replay recorded trades with modified mechanical parameters."""

    def __init__(self, forward_loader: ForwardPathLoader) -> None:
        self._forward_loader = forward_loader

    # ------------------------------------------------------------------
    # Single-trade replay
    # ------------------------------------------------------------------

    def replay_trade(
        self,
        recorded_trade: dict,
        modified_params: dict,
        forward_path: pl.DataFrame,
    ) -> ReplayResult:
        """Walk the forward path and produce the counterfactual outcome."""
        original = self._original_outcome(recorded_trade)

        # 1. Conviction filter — recorded trade falls below the new threshold.
        threshold = modified_params.get("conviction_threshold")
        if threshold is not None and recorded_trade.get("conviction", 1.0) < threshold:
            cf = {
                "exit_price": recorded_trade["entry_price"],
                "exit_reason": "SKIPPED_BY_CONVICTION_FILTER",
                "exit_index": -1,
                "pnl": 0.0,
            }
            return ReplayResult(
                trade_id=str(recorded_trade.get("trade_id", "")),
                original_outcome=original,
                counterfactual_outcome=cf,
                delta_pnl=0.0 - original["pnl"],
                delta_r=self._delta_r(original["pnl"], 0.0, recorded_trade),
                skipped=True,
            )

        # 2. Resolve new SL / TP from params (with fallback to recorded values).
        sl = self._resolve_sl(recorded_trade, modified_params)
        tp1 = modified_params.get("tp1_price", recorded_trade["tp1_price"])
        tp2 = modified_params.get("tp2_price", recorded_trade["tp2_price"])
        breakeven = bool(modified_params.get("breakeven_after_tp1", False))
        trail_mult = modified_params.get("trailing_atr_mult")

        # 3. Walk the forward path bar by bar.
        cf = self._walk_forward(
            trade=recorded_trade,
            initial_sl=sl,
            tp1=tp1,
            tp2=tp2,
            breakeven_after_tp1=breakeven,
            trailing_atr_mult=trail_mult,
            forward_path=forward_path,
        )

        return ReplayResult(
            trade_id=str(recorded_trade.get("trade_id", "")),
            original_outcome=original,
            counterfactual_outcome=cf,
            delta_pnl=cf["pnl"] - original["pnl"],
            delta_r=self._delta_r(original["pnl"], cf["pnl"], recorded_trade),
            skipped=False,
        )

    # ------------------------------------------------------------------
    # Batch + sweep
    # ------------------------------------------------------------------

    def replay_batch(
        self,
        trades: list[dict],
        modified_params: dict,
    ) -> list[ReplayResult]:
        """Replay every trade. Trades whose forward path is missing are
        logged and skipped (no result row)."""
        results: list[ReplayResult] = []
        for trade in trades:
            try:
                fp = self._load_forward_path_for(trade)
            except FileNotFoundError as e:
                logger.warning(
                    f"Skipping trade {trade.get('trade_id')}: forward path missing ({e})"
                )
                continue
            results.append(self.replay_trade(trade, modified_params, fp))
        return results

    def parameter_sweep(
        self,
        trades: list[dict],
        param_name: str,
        param_values: list[float],
    ) -> SweepResult:
        """Run ``replay_batch`` once per ``param_value`` and aggregate.

        Forward paths are loaded once up-front (per trade) and cached so
        the inner loop only walks the path, never hits disk again.
        """
        # Cache forward paths once — sweeps are O(values × trades) and
        # disk IO would dominate.
        cached: dict[str, pl.DataFrame] = {}
        valid_trades: list[dict] = []
        for trade in trades:
            tid = str(trade.get("trade_id", ""))
            try:
                cached[tid] = self._load_forward_path_for(trade)
                valid_trades.append(trade)
            except FileNotFoundError as e:
                logger.warning(
                    f"Skipping trade {tid} from sweep: forward path missing ({e})"
                )

        sweep = SweepResult(param_name=param_name)
        for value in param_values:
            params = {param_name: value}
            results = [
                self.replay_trade(t, params, cached[str(t.get("trade_id", ""))])
                for t in valid_trades
            ]
            sweep.rows.append(self._aggregate(value, results, valid_trades))
        return sweep

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_forward_path_for(self, trade: dict) -> pl.DataFrame:
        tf = trade.get("timeframe", "1h")
        resolution = self._forward_loader.recommended_resolution(tf)
        return self._forward_loader.load(
            symbol=trade["symbol"],
            entry_timestamp=int(trade["entry_timestamp"]),
            duration_candles=60,
            resolution=resolution,
        )

    @staticmethod
    def _original_outcome(trade: dict) -> dict:
        return {
            "exit_price": trade["exit_price"],
            "exit_reason": trade.get("exit_reason", ""),
            "exit_timestamp": trade.get("exit_timestamp", 0),
            "pnl": float(trade["pnl"]),
        }

    @staticmethod
    def _resolve_sl(trade: dict, params: dict) -> float:
        if "sl_price" in params:
            return float(params["sl_price"])
        if "atr_multiplier" in params and trade.get("atr_at_entry", 0) > 0:
            atr = float(trade["atr_at_entry"])
            mult = float(params["atr_multiplier"])
            if trade["direction"] == "LONG":
                return float(trade["entry_price"]) - atr * mult
            return float(trade["entry_price"]) + atr * mult
        return float(trade["sl_price"])

    @staticmethod
    def _delta_r(original_pnl: float, cf_pnl: float, trade: dict) -> float:
        """Counterfactual PnL minus original PnL, expressed in R-multiples
        of the original recorded risk."""
        risk = abs(float(trade["entry_price"]) - float(trade["sl_price"])) * float(
            trade["size"]
        )
        if risk <= 0:
            return 0.0
        return (cf_pnl - original_pnl) / risk

    def _walk_forward(
        self,
        trade: dict,
        initial_sl: float,
        tp1: float,
        tp2: float,
        breakeven_after_tp1: bool,
        trailing_atr_mult: float | None,
        forward_path: pl.DataFrame,
    ) -> dict:
        """Bar-by-bar simulation of the modified mechanical layer.

        Models the production exit policy: 50% closes at TP1, the
        remaining 50% rides until TP2 or SL. SL is checked first each
        bar (conservative — pessimistic on bars where both could fire).
        Trailing stop and break-even both *only tighten* the SL —
        consistent with the Sentinel position manager rules.
        """
        direction = trade["direction"]
        entry = float(trade["entry_price"])
        size = float(trade["size"])
        atr = float(trade.get("atr_at_entry", 0.0))
        half = size / 2.0

        sl = float(initial_sl)
        remaining = size
        realized_pnl = 0.0
        tp1_hit = False
        exit_index = -1
        exit_reason = "still_open"
        last_close = entry  # fallback if path is empty

        for i, row in enumerate(forward_path.iter_rows(named=True)):
            high = float(row["high"])
            low = float(row["low"])
            last_close = float(row["close"])

            # ----- 1. Stop-loss check (always first, on remaining size) -----
            if direction == "LONG" and low <= sl:
                realized_pnl += (sl - entry) * remaining
                exit_reason = "stop_hit"
                exit_index = i
                remaining = 0.0
                break
            if direction == "SHORT" and high >= sl:
                realized_pnl += (entry - sl) * remaining
                exit_reason = "stop_hit"
                exit_index = i
                remaining = 0.0
                break

            # ----- 2. TP1 (50% close) — only fires once -----
            if not tp1_hit:
                tp1_triggered = (
                    (direction == "LONG" and high >= tp1)
                    or (direction == "SHORT" and low <= tp1)
                )
                if tp1_triggered:
                    if direction == "LONG":
                        realized_pnl += (tp1 - entry) * half
                    else:
                        realized_pnl += (entry - tp1) * half
                    remaining -= half
                    tp1_hit = True
                    if breakeven_after_tp1:
                        # Tighten SL to break-even (entry). Only if it
                        # actually moves the stop in the trader's favour.
                        if direction == "LONG" and entry > sl:
                            sl = entry
                        elif direction == "SHORT" and entry < sl:
                            sl = entry

            # ----- 3. TP2 — closes the remaining half -----
            tp2_triggered = (
                (direction == "LONG" and high >= tp2)
                or (direction == "SHORT" and low <= tp2)
            )
            if tp2_triggered and remaining > 0:
                if direction == "LONG":
                    realized_pnl += (tp2 - entry) * remaining
                else:
                    realized_pnl += (entry - tp2) * remaining
                exit_reason = "tp2_hit"
                exit_index = i
                remaining = 0.0
                break

            # ----- 4. Trailing stop (Chandelier-style on close) -----
            if trailing_atr_mult is not None and atr > 0:
                trail_dist = atr * trailing_atr_mult
                if direction == "LONG":
                    candidate = last_close - trail_dist
                    if candidate > sl:
                        sl = candidate
                else:
                    candidate = last_close + trail_dist
                    if candidate < sl:
                        sl = candidate

        # If we ran off the end of the forward path with size remaining,
        # mark-to-market at the last close: trade was still open at the
        # end of the available data.
        if remaining > 0:
            if direction == "LONG":
                realized_pnl += (last_close - entry) * remaining
            else:
                realized_pnl += (entry - last_close) * remaining
            exit_reason = "still_open" if exit_index < 0 else exit_reason
            if exit_index < 0:
                exit_index = forward_path.height - 1
            cf_exit_price = last_close
        else:
            # exit_price is the price at which the *final* close happened.
            if exit_reason == "stop_hit":
                cf_exit_price = sl
            elif exit_reason == "tp2_hit":
                cf_exit_price = tp2
            else:
                cf_exit_price = last_close

        return {
            "exit_price": cf_exit_price,
            "exit_reason": exit_reason,
            "exit_index": exit_index,
            "pnl": realized_pnl,
            "tp1_hit": tp1_hit,
        }

    @staticmethod
    def _aggregate(
        param_value: float,
        results: list[ReplayResult],
        trades: list[dict],
    ) -> SweepRow:
        active = [r for r in results if not r.skipped]
        skipped = [r for r in results if r.skipped]

        cf_pnls = [r.counterfactual_outcome["pnl"] for r in active]
        wins = sum(1 for p in cf_pnls if p > 0)
        win_rate = wins / len(cf_pnls) if cf_pnls else 0.0
        total_pnl = sum(cf_pnls)

        # Avg R based on the original recorded risk for each active trade.
        rs: list[float] = []
        # Build a quick lookup so the active list and trade list stay aligned.
        trade_by_id = {str(t.get("trade_id", "")): t for t in trades}
        for r in active:
            t = trade_by_id.get(r.trade_id)
            if t is None:
                continue
            risk = abs(float(t["entry_price"]) - float(t["sl_price"])) * float(t["size"])
            if risk > 0:
                rs.append(r.counterfactual_outcome["pnl"] / risk)
        avg_r = sum(rs) / len(rs) if rs else 0.0

        # Equity curve drawdown over the active trades' cumulative pnl.
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in cf_pnls:
            equity += p
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        return SweepRow(
            param_value=float(param_value),
            num_trades=len(active),
            num_skipped=len(skipped),
            total_pnl=round(total_pnl, 6),
            win_rate=round(win_rate, 4),
            avg_r=round(avg_r, 4),
            max_drawdown=round(max_dd, 6),
        )
