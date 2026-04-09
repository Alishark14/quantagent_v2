"""PortfolioRiskManager: deterministic, six-layer risk pipeline.

Owns ALL position-sizing math. DecisionAgent outputs trade INTENT only
(action + SL/TP + risk_weight); PRM consumes the conviction-band weight
and turns it into a dollar size after running every safety layer.

Layers (Sprint Portfolio-Risk-Manager):
    1. Fixed Fractional         — risk_dollars = equity * risk_pct * weight
    2. Per-Asset Cap            — clamp/skip if symbol exposure > 15% of equity
    3. Portfolio Cap            — clamp/skip if total exposure > 30% of equity
    4. (Reserved)               — future cost-aware sizing layer
    5. LLM Cost Floor           — expected_profit must clear N x cycle cost
    6. Drawdown Throttle        — equity-curve aware multiplier with hysteresis

This module implements layers 1, 2, 3, 5, and 6 (Tasks 2 + 3). Layer 4 is
reserved for the future cost-aware sizing layer; the public ``size_trade``
shape is stable so Task 4 (pipeline wiring) can be written against the
final signature without churn.

PRM is stateful only with respect to the drawdown hysteresis flag —
once equity drops by ``drawdown_halt_pct`` the manager stays halted
until equity recovers above ``drawdown_resume_pct``. Everything else
is a pure function of the inputs (no per-symbol caches, no rolling
windows). One PRM instance per bot is the recommended scope, mirroring
how peak_equity is tracked per-bot in the live runner.

See SPRINT_portfolio_risk_manager.md and ARCHITECTURE.md for the
full layer rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskConfig:
    """All configurable PRM parameters with sensible defaults.

    Defaults are chosen for a small ($1k-$10k) crypto-perp account
    running the 4-LLM cycle on Hyperliquid:

    - 1% risk per trade is the conventional fixed-fractional baseline
    - 15% per-asset cap means a single symbol can hit 15x leverage at
      most before getting clamped (Task 3)
    - 30% portfolio cap leaves headroom for ~2 simultaneous full-size
      positions before Layer 3 starts shrinking new entries
    - cost_floor_multiplier=20 is the "LLM costs ≤ 5% of TP1 profit"
      gate — protects the bot from spending more on inference than the
      trade can plausibly clear
    - drawdown_halt 10% / reduce 5% / resume 8% gives one full
      reduce-zone of breathing room between halve-size and full-halt,
      and a 2-percentage-point hysteresis gap so the manager doesn't
      thrash on/off at the halt boundary
    - llm_cycle_cost defaults to $0.025 — matches CONTEXT.md's "5 LLM
      calls per cycle ≈ $0.045" budget after prompt-cache hits
    """

    risk_per_trade_pct: float = 0.01
    per_asset_cap_pct: float = 0.15
    portfolio_cap_pct: float = 0.30
    cost_floor_multiplier: float = 20.0
    drawdown_halt_pct: float = 0.10
    drawdown_reduce_pct: float = 0.05
    drawdown_resume_pct: float = 0.08
    llm_cycle_cost: float = 0.025


@dataclass
class SizingResult:
    """Output of one full ``size_trade`` pipeline run.

    On the happy path: ``skipped=False`` and ``position_size_usd > 0``.
    On any safety-layer rejection: ``skipped=True`` with ``skip_reason``
    populated and ``position_size_usd=0.0``. The intermediate
    ``risk_dollars`` / ``effective_risk_pct`` / ``drawdown_multiplier``
    fields are filled in as far as the pipeline got — if Layer 6 halts
    we have multiplier=0.0 but no risk dollars; if Layer 5 catches a
    too-small trade we have both multiplier and risk dollars but a
    zeroed-out final position. This makes the result self-documenting
    for operators reading PRM logs.
    """

    position_size_usd: float
    risk_dollars: float
    effective_risk_pct: float
    drawdown_multiplier: float
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "position_size_usd": self.position_size_usd,
            "risk_dollars": self.risk_dollars,
            "effective_risk_pct": self.effective_risk_pct,
            "drawdown_multiplier": self.drawdown_multiplier,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


class PortfolioRiskManager:
    """Six-layer deterministic risk pipeline.

    Stateful only with respect to the drawdown halt flag — every other
    decision is a pure function of the inputs to ``size_trade``. The
    state machine for the halt flag is:

        not halted → halted          when drawdown ≥ halt_pct
        halted     → not halted      when drawdown < resume_pct
        halted     → halted          (any other drawdown level)

    Note the asymmetry: triggering uses ``drawdown_halt_pct`` (10% by
    default), recovering uses ``drawdown_resume_pct`` (8%). The 2-pp
    gap is the hysteresis band — without it, equity bouncing between
    9.9% and 10.1% drawdown would flip the manager on and off every
    cycle and the bot would be uselessly stuttering.

    Recommended scope: one instance per bot. PRM does not coordinate
    across bots; cross-bot exposure is the exchange's job (the same
    account funds every bot in production).
    """

    def __init__(self, config: PortfolioRiskConfig | None = None) -> None:
        self._config = config or PortfolioRiskConfig()
        # Hysteresis state — once True, stays True until equity recovers
        # above the resume threshold. Initialised False so a fresh PRM
        # instance always grants full risk on its first call.
        self._halted: bool = False

    @property
    def config(self) -> PortfolioRiskConfig:
        """Read-only access to the active config."""
        return self._config

    @property
    def is_halted(self) -> bool:
        """Whether the drawdown throttle is currently in halt state.

        Useful for diagnostics + the eventual /health endpoint. Test
        suite asserts on this directly to verify the hysteresis state
        transitions without having to inspect a private attribute.
        """
        return self._halted

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def size_trade(
        self,
        equity: float,
        peak_equity: float,
        sl_distance_pct: float,
        tp1_distance_pct: float,
        risk_weight: float,
        symbol: str,
        open_positions: list | None = None,
    ) -> SizingResult:
        """Run the full sizing pipeline for one trade intent.

        Layer execution order: 6 → 1 → 5 → 2 → 3.

        Layer 5 is documented in the spec as "first" because conceptually
        it's a fail-fast cost gate, but in practice it needs the
        position_size from Layer 1 to compute expected profit — the math
        is identical to a fast-check on equity * risk_pct *
        tp1_distance_pct / sl_distance_pct, so doing Layer 1 first costs
        nothing but lets Layer 5 use the canonical size value.

        Args:
            equity: Current account equity in USD (live balance).
            peak_equity: All-time-high equity for drawdown calculation.
            sl_distance_pct: |entry - SL| / entry, as a positive fraction.
            tp1_distance_pct: |TP1 - entry| / entry, as a positive fraction.
            risk_weight: Conviction-band weight from DecisionAgent
                (0.75 / 1.0 / 1.15 / 1.3).
            symbol: Trading symbol (Layer 2 will use this in Task 3).
            open_positions: List of {"symbol", "notional", "direction"}
                dicts (Layers 2 and 3 will use this in Task 3). None
                or empty list means no exposure caps apply.

        Returns:
            SizingResult with the final position size or a skip reason.
        """
        # Defensive input validation. PRM is invoked from the pipeline
        # which has its own SKIP-on-error pattern, but we double-check
        # here so a misuse from a script or test surfaces a clear
        # SizingResult instead of a divide-by-zero or negative size.
        if equity <= 0:
            return self._skip_result(
                drawdown_multiplier=1.0,
                reason=f"Invalid equity: {equity!r} (must be > 0)",
            )
        if sl_distance_pct <= 0:
            return self._skip_result(
                drawdown_multiplier=1.0,
                reason=f"Invalid sl_distance_pct: {sl_distance_pct!r} (must be > 0)",
            )
        if tp1_distance_pct <= 0:
            return self._skip_result(
                drawdown_multiplier=1.0,
                reason=f"Invalid tp1_distance_pct: {tp1_distance_pct!r} (must be > 0)",
            )
        if risk_weight is None or risk_weight <= 0:
            # DecisionAgent should never emit a non-positive weight for
            # an entry action — but if it does, we don't want to size a
            # zero or negative position. Treat as a safety skip.
            return self._skip_result(
                drawdown_multiplier=1.0,
                reason=f"Invalid risk_weight: {risk_weight!r} (must be > 0)",
            )

        # ── Layer 6: Drawdown Throttle ──
        # Resolved first because it gates all the math below — a
        # halted manager returns SKIP without doing any sizing work.
        dd_mult = self._get_drawdown_multiplier(equity, peak_equity)
        if dd_mult == 0.0:
            drawdown_pct = (
                (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            )
            return self._skip_result(
                drawdown_multiplier=0.0,
                reason=(
                    f"Drawdown halt: equity ${equity:,.2f} is "
                    f"{drawdown_pct * 100:.1f}% below peak ${peak_equity:,.2f} "
                    f"(halt threshold {self._config.drawdown_halt_pct * 100:.1f}%)"
                ),
            )

        # ── Layer 1: Fixed Fractional ──
        effective_risk_pct = self._config.risk_per_trade_pct * dd_mult
        risk_dollars = equity * effective_risk_pct * risk_weight
        position_size = risk_dollars / sl_distance_pct

        # ── Layer 5: LLM Cost Floor ──
        # Reject trades whose TP1 profit can't even cover the LLM cost
        # of the cycle that produced them by a healthy margin
        # (default: TP1 profit must be ≥ 20x the cycle cost, i.e. LLM
        # spend ≤ 5% of expected profit).
        expected_profit = position_size * tp1_distance_pct
        min_profit = self._config.llm_cycle_cost * self._config.cost_floor_multiplier
        if expected_profit < min_profit:
            result = SizingResult(
                position_size_usd=0.0,
                risk_dollars=risk_dollars,
                effective_risk_pct=effective_risk_pct,
                drawdown_multiplier=dd_mult,
                skipped=True,
                skip_reason=(
                    f"LLM cost floor: expected profit ${expected_profit:.2f} "
                    f"< min ${min_profit:.2f} "
                    f"({self._config.cost_floor_multiplier:.0f}x "
                    f"cycle cost ${self._config.llm_cycle_cost:.4f})"
                ),
            )
            logger.info(f"PRM SKIP [{symbol}]: {result.skip_reason}")
            return result

        positions = open_positions or []

        # ── Layer 2: Per-Asset Cap ──
        # Clamp to whatever capacity remains for THIS symbol's bucket.
        # If the symbol's existing exposure already meets or exceeds
        # the per-asset cap (15% of equity by default) the layer
        # returns 0.0 and we convert to a layer-attributed SKIP so
        # operator logs say "Per-asset cap" not just "exposure clamped".
        position_size_after_layer2 = self._apply_per_asset_cap(
            position_size=position_size,
            equity=equity,
            symbol=symbol,
            open_positions=positions,
        )
        if position_size_after_layer2 <= 0:
            existing = self._symbol_exposure(symbol, positions)
            cap = equity * self._config.per_asset_cap_pct
            result = SizingResult(
                position_size_usd=0.0,
                risk_dollars=risk_dollars,
                effective_risk_pct=effective_risk_pct,
                drawdown_multiplier=dd_mult,
                skipped=True,
                skip_reason=(
                    f"Per-asset cap: {symbol} existing exposure "
                    f"${existing:,.2f} ≥ cap ${cap:,.2f} "
                    f"({self._config.per_asset_cap_pct * 100:.0f}% of equity)"
                ),
            )
            logger.info(f"PRM SKIP [{symbol}]: {result.skip_reason}")
            return result
        position_size = position_size_after_layer2

        # ── Layer 3: Portfolio Cap ──
        # Same shape as Layer 2 but against TOTAL exposure across
        # every open position. Tightest cap wins — Layer 3 only fires
        # when Layer 2 left enough headroom but the portfolio is
        # already at or near the 30% portfolio-wide cap.
        position_size_after_layer3 = self._apply_portfolio_cap(
            position_size=position_size,
            equity=equity,
            open_positions=positions,
        )
        if position_size_after_layer3 <= 0:
            total = self._total_exposure(positions)
            cap = equity * self._config.portfolio_cap_pct
            result = SizingResult(
                position_size_usd=0.0,
                risk_dollars=risk_dollars,
                effective_risk_pct=effective_risk_pct,
                drawdown_multiplier=dd_mult,
                skipped=True,
                skip_reason=(
                    f"Portfolio cap: total exposure ${total:,.2f} ≥ "
                    f"cap ${cap:,.2f} "
                    f"({self._config.portfolio_cap_pct * 100:.0f}% of equity)"
                ),
            )
            logger.info(f"PRM SKIP [{symbol}]: {result.skip_reason}")
            return result
        position_size = position_size_after_layer3

        result = SizingResult(
            position_size_usd=round(position_size, 2),
            risk_dollars=round(risk_dollars, 2),
            effective_risk_pct=effective_risk_pct,
            drawdown_multiplier=dd_mult,
            skipped=False,
            skip_reason="",
        )
        logger.info(
            f"PRM SIZE [{symbol}]: ${result.position_size_usd:.2f} "
            f"(risk ${result.risk_dollars:.2f}, weight {risk_weight:.2f}, "
            f"DD_mult {dd_mult:.2f})"
        )
        return result

    # ------------------------------------------------------------------
    # Layer helpers
    # ------------------------------------------------------------------

    def _get_drawdown_multiplier(self, equity: float, peak_equity: float) -> float:
        """Layer 6: equity-curve drawdown throttle with hysteresis.

        Returns:
            1.0  — full risk (drawdown < reduce_pct)
            0.5  — half risk (reduce_pct ≤ drawdown < halt_pct)
            0.0  — halted    (drawdown ≥ halt_pct)

        Hysteresis: once halted, the manager only resumes when
        drawdown improves to BELOW ``drawdown_resume_pct``. Between
        the resume threshold and the halt threshold the manager stays
        in the halted state — this is the explicit asymmetry that
        prevents thrashing at the boundary.

        ``peak_equity <= 0`` is treated as "no history yet" and
        returns 1.0 (full risk). This is the path a brand-new bot
        takes on its very first cycle before peak_equity has been
        seeded; we'd rather size a fresh trade than skip every cycle
        until peak gets populated.
        """
        if peak_equity <= 0:
            # No equity history → no drawdown → full risk.
            # Also clear any stale halt flag from a previous bot run.
            self._halted = False
            return 1.0

        drawdown = (peak_equity - equity) / peak_equity

        # Hysteresis: if currently halted, only release the halt when
        # drawdown improves below the resume threshold. Between the
        # resume threshold and the halt threshold the manager stays
        # halted — that's the explicit thrash-prevention zone.
        if self._halted:
            if drawdown < self._config.drawdown_resume_pct:
                # Recovery — drop the halt and fall through to compute
                # the non-halted multiplier (might be 0.5 if we're
                # still in the reduce zone, or 1.0 if we're fully
                # recovered).
                logger.info(
                    f"PRM drawdown throttle: halt released "
                    f"(drawdown {drawdown * 100:.2f}% < resume "
                    f"{self._config.drawdown_resume_pct * 100:.2f}%)"
                )
                self._halted = False
            else:
                # Still halted, stay halted. Don't log every cycle —
                # the halt was already logged when it triggered.
                return 0.0

        # Not currently halted: classify drawdown into one of three bands.
        if drawdown >= self._config.drawdown_halt_pct:
            self._halted = True
            logger.warning(
                f"PRM drawdown throttle: HALT triggered "
                f"(drawdown {drawdown * 100:.2f}% ≥ "
                f"{self._config.drawdown_halt_pct * 100:.2f}%)"
            )
            return 0.0
        if drawdown >= self._config.drawdown_reduce_pct:
            return 0.5
        return 1.0

    def _apply_per_asset_cap(
        self,
        position_size: float,
        equity: float,
        symbol: str,
        open_positions: list,
    ) -> float:
        """Layer 2: per-asset exposure cap.

        Sums the notional of every existing position in the SAME symbol,
        compares against ``equity * per_asset_cap_pct``, and returns:

        * ``0.0`` — when the symbol's existing exposure already meets
          or exceeds the cap (caller converts to SKIP with a per-asset
          attribution in the reason).
        * ``min(position_size, remaining)`` — when there's still room
          but the proposed trade would exceed it (clamping is silent;
          the trade still goes through, just smaller).
        * ``position_size`` unchanged — when the symbol's existing
          exposure plus the proposed trade fits within the cap.

        Other symbols' positions DON'T count here — that's Layer 3's
        job (portfolio-wide cap).

        ``open_positions`` is a list of dicts shaped like
        ``{"symbol": str, "notional": float, "direction": str}``;
        only ``symbol`` and ``notional`` are read in this layer.
        ``direction`` is informational for now (long vs short both
        consume capacity equally — there's no netting at this layer
        because the SL/TP risk is symmetric across direction).
        """
        existing = self._symbol_exposure(symbol, open_positions)
        cap = equity * self._config.per_asset_cap_pct
        remaining = cap - existing
        if remaining <= 0:
            return 0.0
        return min(position_size, remaining)

    def _apply_portfolio_cap(
        self,
        position_size: float,
        equity: float,
        open_positions: list,
    ) -> float:
        """Layer 3: portfolio-wide exposure cap.

        Same shape as Layer 2 but sums notional across EVERY open
        position regardless of symbol or direction, and compares against
        ``equity * portfolio_cap_pct``. Returns ``0.0`` when there's
        no remaining capacity (caller converts to SKIP), the clamped
        size when the proposed trade would push total exposure over
        the cap, or the position unchanged when it fits.

        Layer 3 runs AFTER Layer 2 in ``size_trade``, so a tightest-
        wins ordering is automatic — if Layer 2 already clamped the
        size to $150 and Layer 3 would have allowed up to $200, the
        result is $150. If Layer 2 allowed $500 but Layer 3 only has
        $200 of headroom, the result is $200.
        """
        total = self._total_exposure(open_positions)
        cap = equity * self._config.portfolio_cap_pct
        remaining = cap - total
        if remaining <= 0:
            return 0.0
        return min(position_size, remaining)

    @staticmethod
    def _symbol_exposure(symbol: str, open_positions: list) -> float:
        """Sum the notional of every existing position in the given symbol.

        Skips entries with malformed shape so a stale or partially
        populated position dict can't crash sizing — defensive against
        the future Task 4 wiring where positions come straight from
        ``adapter.get_positions()`` and might miss fields on a venue
        that doesn't expose them.
        """
        total = 0.0
        for p in open_positions:
            if p.get("symbol") != symbol:
                continue
            try:
                total += float(p.get("notional", 0.0))
            except (TypeError, ValueError):
                continue
        return total

    @staticmethod
    def _total_exposure(open_positions: list) -> float:
        """Sum the notional of every open position regardless of symbol."""
        total = 0.0
        for p in open_positions:
            try:
                total += float(p.get("notional", 0.0))
            except (TypeError, ValueError):
                continue
        return total

    @staticmethod
    def _skip_result(drawdown_multiplier: float, reason: str) -> SizingResult:
        """Build a SKIP SizingResult with zeroed sizing fields.

        Used by every early-return path so the shape stays consistent
        regardless of which layer fired.
        """
        return SizingResult(
            position_size_usd=0.0,
            risk_dollars=0.0,
            effective_risk_pct=0.0,
            drawdown_multiplier=drawdown_multiplier,
            skipped=True,
            skip_reason=reason,
        )
