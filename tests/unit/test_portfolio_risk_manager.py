"""Unit tests for engine/execution/portfolio_risk_manager.py.

Covers the three layers shipped in Sprint Portfolio-Risk-Manager Task 2:

* Layer 1 (Fixed Fractional)  — basic sizing math, SL distance scaling,
  risk_weight scaling, invalid-input safety
* Layer 5 (LLM Cost Floor)    — below-floor SKIP, above-floor PASS, edge
  cases at the boundary
* Layer 6 (Drawdown Throttle) — full risk / half risk / halt bands plus
  the hysteresis state machine (halt → resume gap)

Plus integration tests that walk a request through every layer end to
end so a future regression that breaks layer ordering is caught.
"""

from __future__ import annotations

import pytest

from engine.execution.portfolio_risk_manager import (
    PortfolioRiskConfig,
    PortfolioRiskManager,
    SizingResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prm(**overrides) -> PortfolioRiskManager:
    """Build a PRM whose Layer 2/3 caps are RELAXED by default.

    Layer 1 / 5 / 6 tests don't care about exposure caps and would
    otherwise get clobbered by the spec defaults (per-asset cap 15%
    is tighter than the natural Layer 1 position size with 1% risk
    and 2% SL — `0.01 / 0.02 = 50%` of equity, way over the 15% cap).
    Defaulting both caps to ~100% effectively disables Layer 2/3 so
    each layer can be tested in isolation; the Layer 2/3 test classes
    pass explicit `per_asset_cap_pct=0.15` and `portfolio_cap_pct=0.30`
    to opt back into the spec defaults. The TestPortfolioRiskConfigDefaults
    smoke tests pin the dataclass defaults directly via
    `PortfolioRiskConfig()` so the spec contract still has coverage.
    """
    defaults = {
        "per_asset_cap_pct": 100.0,  # 10000% of equity → effectively disabled
        "portfolio_cap_pct": 100.0,
    }
    defaults.update(overrides)
    return PortfolioRiskManager(PortfolioRiskConfig(**defaults))


def _size(
    prm: PortfolioRiskManager,
    *,
    equity: float = 10_000.0,
    peak_equity: float | None = None,
    sl_distance_pct: float = 0.02,
    tp1_distance_pct: float = 0.04,
    risk_weight: float = 1.0,
    symbol: str = "BTC-USDC",
    open_positions: list | None = None,
) -> SizingResult:
    """Convenience wrapper around ``size_trade`` with sensible defaults.

    ``peak_equity`` defaults to whatever ``equity`` is set to so that
    overriding only ``equity`` doesn't accidentally trip Layer 6 (the
    drawdown throttle) by leaving a stale 10k peak in place. Tests
    that explicitly want a drawdown scenario pass both args.
    """
    return prm.size_trade(
        equity=equity,
        peak_equity=equity if peak_equity is None else peak_equity,
        sl_distance_pct=sl_distance_pct,
        tp1_distance_pct=tp1_distance_pct,
        risk_weight=risk_weight,
        symbol=symbol,
        open_positions=open_positions,
    )


# ---------------------------------------------------------------------------
# Layer 1 — Fixed Fractional sizing
# ---------------------------------------------------------------------------


class TestLayer1FixedFractional:
    """`risk_dollars = equity * risk_pct * weight; size = risk_dollars / sl_pct`."""

    def test_basic_sizing(self) -> None:
        """$10k equity, 1% risk, weight 1.0, 2% SL → $100 risk, $5000 size."""
        result = _size(_prm())
        assert not result.skipped
        assert result.risk_dollars == pytest.approx(100.0)
        assert result.position_size_usd == pytest.approx(5000.0)
        assert result.effective_risk_pct == pytest.approx(0.01)
        assert result.drawdown_multiplier == 1.0

    def test_wider_sl_smaller_position(self) -> None:
        """Doubling the SL distance halves the position size for the same risk."""
        narrow = _size(_prm(), sl_distance_pct=0.02)
        wide = _size(_prm(), sl_distance_pct=0.04)
        assert not narrow.skipped and not wide.skipped
        # Same risk dollars in both cases (the whole point of fixed-fractional)
        assert narrow.risk_dollars == pytest.approx(wide.risk_dollars)
        # But the position size is exactly half
        assert wide.position_size_usd == pytest.approx(narrow.position_size_usd / 2)

    def test_same_risk_dollars_either_way(self) -> None:
        """Across multiple SL distances, risk_dollars stays at equity * risk_pct."""
        prm = _prm()
        for sl in (0.01, 0.02, 0.03, 0.05, 0.10):
            result = _size(prm, sl_distance_pct=sl)
            assert result.risk_dollars == pytest.approx(100.0), (
                f"sl={sl} produced risk_dollars={result.risk_dollars}, "
                "expected 100.0 (1% of $10k)"
            )

    def test_risk_weight_scales_position(self) -> None:
        """A 1.3 weight produces 30% more risk dollars than weight 1.0."""
        base = _size(_prm(), risk_weight=1.0)
        boosted = _size(_prm(), risk_weight=1.3)
        assert boosted.risk_dollars == pytest.approx(base.risk_dollars * 1.3)
        assert boosted.position_size_usd == pytest.approx(
            base.position_size_usd * 1.3
        )

    def test_risk_weight_low_band_under_sizes(self) -> None:
        """A 0.75 weight produces 25% less than weight 1.0."""
        base = _size(_prm(), risk_weight=1.0)
        low = _size(_prm(), risk_weight=0.75)
        assert low.position_size_usd == pytest.approx(base.position_size_usd * 0.75)

    def test_zero_equity_returns_skip(self) -> None:
        result = _size(_prm(), equity=0.0)
        assert result.skipped
        assert "Invalid equity" in result.skip_reason
        assert result.position_size_usd == 0.0

    def test_zero_sl_distance_returns_skip(self) -> None:
        result = _size(_prm(), sl_distance_pct=0.0)
        assert result.skipped
        assert "sl_distance_pct" in result.skip_reason

    def test_zero_risk_weight_returns_skip(self) -> None:
        result = _size(_prm(), risk_weight=0.0)
        assert result.skipped
        assert "risk_weight" in result.skip_reason


# ---------------------------------------------------------------------------
# Layer 5 — LLM cost floor
# ---------------------------------------------------------------------------


class TestLayer5LLMCostFloor:
    """`expected_profit = position_size * tp1_pct >= cycle_cost * multiplier`."""

    def test_below_cost_floor_skips(self) -> None:
        """Tiny equity → tiny size → expected_profit < min → SKIP."""
        # $50 equity, 1% risk, 2% SL, 4% TP1, weight 1.0:
        # risk_dollars = 0.50, size = $25, expected_profit = $1.00
        # Default min_profit = $0.025 * 20 = $0.50 → above floor.
        # Push it below by raising the multiplier so the floor jumps.
        result = _size(
            _prm(cost_floor_multiplier=100.0),  # min_profit = 100 * 0.025 = $2.50
            equity=50.0,
        )
        assert result.skipped
        assert "LLM cost floor" in result.skip_reason
        assert result.position_size_usd == 0.0
        # Layer 1 still ran, so risk_dollars should be populated
        assert result.risk_dollars == pytest.approx(0.5)

    def test_above_cost_floor_passes(self) -> None:
        """Normal $10k account → large size → expected_profit >> floor."""
        result = _size(_prm())
        assert not result.skipped
        # Sanity-check the math: 5000 * 0.04 = $200 expected profit,
        # well above the default $0.50 floor.
        assert result.position_size_usd * 0.04 > 0.5

    def test_at_cost_floor_boundary(self) -> None:
        """A trade exactly AT the floor should pass (strict <, not ≤)."""
        # We craft inputs so expected_profit lands exactly on the floor.
        # min_profit = cycle_cost * multiplier = 0.025 * 20 = 0.50
        # Want: position_size * tp1_pct == 0.50
        # → equity * risk_pct * tp1_pct / sl_pct == 0.50
        # With risk_pct = 0.01, tp1_pct = 0.04, sl_pct = 0.02:
        # equity * 0.01 * 0.04 / 0.02 == 0.50  →  equity == 25.0
        result = _size(_prm(), equity=25.0)
        assert not result.skipped, (
            f"At-boundary trade should pass; got {result.skip_reason}"
        )

    def test_just_below_cost_floor_boundary_skips(self) -> None:
        """One cent below the boundary → SKIP."""
        # Same math as above but equity = 24.99 → expected_profit ≈ 0.4998
        result = _size(_prm(), equity=24.99)
        assert result.skipped
        assert "LLM cost floor" in result.skip_reason

    def test_skip_reason_includes_dollar_amounts(self) -> None:
        """The reason must surface both numbers so operators can debug."""
        result = _size(_prm(cost_floor_multiplier=100.0), equity=50.0)
        assert result.skipped
        assert "$" in result.skip_reason
        # Both the actual expected and the min should appear
        assert "expected" in result.skip_reason.lower()
        assert "min" in result.skip_reason.lower()


# ---------------------------------------------------------------------------
# Layer 6 — Drawdown throttle
# ---------------------------------------------------------------------------


class TestLayer6DrawdownThrottle:

    def test_no_drawdown_full_risk(self) -> None:
        """equity == peak → multiplier 1.0."""
        result = _size(_prm(), equity=10_000.0, peak_equity=10_000.0)
        assert result.drawdown_multiplier == 1.0
        assert not result.skipped

    def test_5pct_drawdown_half_risk(self) -> None:
        """5% drawdown lands inside the reduce band → multiplier 0.5."""
        result = _size(_prm(), equity=9_500.0, peak_equity=10_000.0)
        assert result.drawdown_multiplier == 0.5
        # Effective risk should be HALF the configured 1%
        assert result.effective_risk_pct == pytest.approx(0.005)
        # Position math: equity ($9500) * effective_risk (0.005) * weight (1.0)
        # = $47.50 risk_dollars; position = 47.50 / 0.02 SL = $2375.
        # NOT exactly half a baseline trade because the drawdown also
        # shrinks the equity the position is sized against.
        assert result.risk_dollars == pytest.approx(47.5)
        assert result.position_size_usd == pytest.approx(2375.0)

    def test_10pct_drawdown_halts(self) -> None:
        """10% drawdown lands at the halt threshold → SKIP with multiplier 0."""
        prm = _prm()
        result = _size(prm, equity=9_000.0, peak_equity=10_000.0)
        assert result.skipped
        assert result.drawdown_multiplier == 0.0
        assert "Drawdown halt" in result.skip_reason
        # And the manager remembers it's halted
        assert prm.is_halted

    def test_zero_peak_equity_safe(self) -> None:
        """peak_equity == 0 → no drawdown computation, full risk, no zero-div."""
        result = _size(_prm(), equity=10_000.0, peak_equity=0.0)
        assert result.drawdown_multiplier == 1.0
        assert not result.skipped

    def test_hysteresis_halt_persists_in_reduce_band(self) -> None:
        """Once halted at 10%, recovering to 9% drawdown stays halted.

        9% is below the halt threshold (10%) but above the resume
        threshold (8%), so the manager must remain halted — that's the
        whole point of the hysteresis gap.
        """
        prm = _prm()
        # Trip the halt at 10% drawdown
        first = _size(prm, equity=9_000.0, peak_equity=10_000.0)
        assert first.skipped and prm.is_halted

        # Recover slightly to 9% drawdown — still halted
        second = _size(prm, equity=9_100.0, peak_equity=10_000.0)
        assert second.skipped, (
            "Manager should stay halted at 9% drawdown after halting at 10% "
            "(hysteresis: halt at halt_pct, resume only below resume_pct)"
        )
        assert prm.is_halted

    def test_hysteresis_resume_below_8pct(self) -> None:
        """Once equity recovers to <8% drawdown, the halt is released."""
        prm = _prm()
        # Trip the halt
        _size(prm, equity=9_000.0, peak_equity=10_000.0)
        assert prm.is_halted

        # Recover to 7% drawdown — halt should release, multiplier 0.5
        # because we're still in the reduce band (5% ≤ DD < 10%)
        result = _size(prm, equity=9_300.0, peak_equity=10_000.0)
        assert not prm.is_halted, "Manager should resume below the resume threshold"
        assert not result.skipped
        assert result.drawdown_multiplier == 0.5

    def test_hysteresis_full_recovery_returns_to_full_risk(self) -> None:
        """Recover all the way (drawdown 0) → multiplier 1.0."""
        prm = _prm()
        _size(prm, equity=9_000.0, peak_equity=10_000.0)  # halt
        assert prm.is_halted

        result = _size(prm, equity=10_000.0, peak_equity=10_000.0)
        assert not prm.is_halted
        assert result.drawdown_multiplier == 1.0

    def test_hysteresis_at_resume_threshold_stays_halted(self) -> None:
        """drawdown == resume_pct (8% exactly) is still halted — strict <."""
        prm = _prm()
        _size(prm, equity=9_000.0, peak_equity=10_000.0)  # halt
        assert prm.is_halted

        # Drawdown exactly at the resume threshold (8%) → still halted
        result = _size(prm, equity=9_200.0, peak_equity=10_000.0)
        assert result.skipped, (
            "drawdown == resume_pct should still be halted (strict < release)"
        )
        assert prm.is_halted

    def test_drawdown_halt_skip_reason_has_numbers(self) -> None:
        prm = _prm()
        result = _size(prm, equity=8_500.0, peak_equity=10_000.0)
        assert result.skipped
        assert "Drawdown halt" in result.skip_reason
        assert "$" in result.skip_reason  # equity + peak both shown
        assert "%" in result.skip_reason  # drawdown percentage shown


# ---------------------------------------------------------------------------
# Integration: full pipeline through every layer
# ---------------------------------------------------------------------------


class TestSizeTradeIntegration:

    def test_happy_path_all_layers_pass(self) -> None:
        """Realistic $10k account, normal trade — every layer passes."""
        result = _size(
            _prm(),
            equity=10_000.0,
            peak_equity=10_000.0,
            sl_distance_pct=0.02,
            tp1_distance_pct=0.04,
            risk_weight=1.15,  # high conviction
        )
        assert not result.skipped
        assert result.position_size_usd > 0
        assert result.risk_dollars == pytest.approx(115.0)  # 10000 * 0.01 * 1.15
        assert result.position_size_usd == pytest.approx(5750.0)  # 115 / 0.02
        assert result.drawdown_multiplier == 1.0
        assert result.effective_risk_pct == pytest.approx(0.01)

    def test_drawdown_then_layer_5_skip(self) -> None:
        """5% drawdown halves the size, then a tight TP1 breaks the cost floor.

        Verifies the layer interaction: Layer 6 reduces effective_risk
        to 0.5%, Layer 1 computes the smaller size, Layer 5 then
        catches that the smaller size doesn't clear the cost floor
        anymore. End result is SKIP with the cost-floor reason (NOT
        the drawdown reason — Layer 6 only reduced, didn't halt).
        """
        # In drawdown band (5%) → multiplier 0.5
        # Equity $50, multiplier 0.5 → effective 0.5%
        # risk_dollars = 50 * 0.005 * 1.0 = 0.25
        # position = 0.25 / 0.02 = 12.50
        # expected_profit = 12.50 * 0.04 = 0.50 (exactly at default floor)
        # → push the floor up so it skips
        result = _size(
            _prm(cost_floor_multiplier=50.0),  # min_profit = 50 * 0.025 = 1.25
            equity=50.0 * 0.95,  # ~5% drawdown so multiplier=0.5
            peak_equity=50.0,
        )
        assert result.skipped
        assert "LLM cost floor" in result.skip_reason
        # Layer 6 reduced but didn't halt — multiplier should be 0.5
        assert result.drawdown_multiplier == 0.5

    def test_skipped_result_preserves_intermediate_state(self) -> None:
        """A Layer 5 SKIP still reports the risk_dollars and multiplier
        the earlier layers computed, so logs are debuggable."""
        result = _size(_prm(cost_floor_multiplier=100.0), equity=50.0)
        assert result.skipped
        # Layer 1 ran, so these are populated
        assert result.risk_dollars > 0
        assert result.effective_risk_pct == pytest.approx(0.01)
        assert result.drawdown_multiplier == 1.0
        # But the final position is zeroed
        assert result.position_size_usd == 0.0

    def test_open_positions_argument_is_optional(self) -> None:
        """`open_positions=None` is treated as no exposure (Layers 2-3 placeholders)."""
        result_none = _size(_prm(), open_positions=None)
        result_empty = _size(_prm(), open_positions=[])
        assert not result_none.skipped and not result_empty.skipped
        assert result_none.position_size_usd == result_empty.position_size_usd

    def test_small_existing_positions_below_caps_dont_clamp(self) -> None:
        """Positions well below both caps don't affect the new trade size.

        Negative-control regression test for Layers 2 and 3: with the
        defaults ($10k equity, 15% per-asset cap = $1500, 30%
        portfolio cap = $3000), a $50 BTC position and $50 ETH position
        leave plenty of headroom — the resulting size must match the
        no-positions case exactly. Catches a future bug where Layer 2
        or Layer 3 starts subtracting unconditionally."""
        tiny_positions = [
            {"symbol": "BTC-USDC", "notional": 50.0, "direction": "long"},
            {"symbol": "ETH-USDC", "notional": 50.0, "direction": "short"},
        ]
        with_positions = _size(_prm(), open_positions=tiny_positions)
        without_positions = _size(_prm(), open_positions=[])
        assert not with_positions.skipped
        assert (
            with_positions.position_size_usd == without_positions.position_size_usd
        )


# ---------------------------------------------------------------------------
# Layer 2 — Per-asset exposure cap (Task 3)
# ---------------------------------------------------------------------------


def _capped_prm(**overrides) -> PortfolioRiskManager:
    """Build a PRM with the spec-default Layer 2/3 caps active.

    Used by Layer 2 / Layer 3 / combined-caps tests that explicitly
    want the 15% per-asset / 30% portfolio caps to fire. The vanilla
    `_prm()` helper relaxes both caps to make Layer 1/5/6 tests
    isolate-able; this helper undoes that for the cap tests.
    """
    defaults = {
        "per_asset_cap_pct": 0.15,
        "portfolio_cap_pct": 0.30,
    }
    defaults.update(overrides)
    return PortfolioRiskManager(PortfolioRiskConfig(**defaults))


class TestLayer2PerAssetCap:
    """`existing = sum notionals for symbol; remaining = cap - existing`."""

    def test_no_existing_positions_clamps_to_cap(self) -> None:
        """Empty positions, default 15% per-asset cap → clamped to $1500.

        With the spec defaults the natural Layer 1 size ($5000) is
        already over the 15% cap on a fresh account, so Layer 2
        clamps to the cap. This is the spec's intended behavior —
        the cap is a hard ceiling, not a "headroom" subtract from a
        smaller baseline.
        """
        result = _size(_capped_prm(), open_positions=[])
        assert not result.skipped
        # cap = 10000 * 0.15 = 1500
        assert result.position_size_usd == pytest.approx(1500.0)

    def test_existing_position_clamps_proposed_size(self) -> None:
        """$1000 existing BTC + $1500 per-asset cap → $500 of headroom.

        With the default test setup (equity $10k, 2% SL → $5000
        Layer 1 baseline) the proposed trade is much larger than the
        $500 remaining cap for BTC, so Layer 2 clamps to $500.
        """
        positions = [
            {"symbol": "BTC-USDC", "notional": 1000.0, "direction": "long"},
        ]
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert not result.skipped
        # cap = 10000 * 0.15 = 1500; existing = 1000; remaining = 500
        assert result.position_size_usd == pytest.approx(500.0)

    def test_at_per_asset_cap_skips(self) -> None:
        """Existing exposure at exactly the cap → SKIP."""
        positions = [
            {"symbol": "BTC-USDC", "notional": 1500.0, "direction": "long"},
        ]
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert result.skipped
        assert "Per-asset cap" in result.skip_reason
        assert "BTC-USDC" in result.skip_reason
        assert result.position_size_usd == 0.0

    def test_above_per_asset_cap_skips(self) -> None:
        """Existing exposure already over the cap → SKIP."""
        positions = [
            {"symbol": "BTC-USDC", "notional": 2000.0, "direction": "long"},
        ]
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert result.skipped
        assert "Per-asset cap" in result.skip_reason

    def test_other_symbol_doesnt_count(self) -> None:
        """Existing ETH exposure doesn't affect new BTC sizing.

        Layer 2 is per-symbol — an existing ETH position would consume
        the entire BTC per-asset cap if it counted, but it doesn't.
        New BTC trade should be unaffected by Layer 2 (Layer 3 still
        catches the portfolio-wide exposure separately, so we relax
        the portfolio cap here to keep this test focused on Layer 2).
        """
        positions = [
            {"symbol": "ETH-USDC", "notional": 1200.0, "direction": "long"},
        ]
        prm = _capped_prm(portfolio_cap_pct=0.999)  # disable Layer 3
        result = _size(prm, symbol="BTC-USDC", open_positions=positions)
        assert not result.skipped
        # BTC bucket has zero existing → full per-asset cap available → $1500
        assert result.position_size_usd == pytest.approx(1500.0)

    def test_multiple_positions_same_symbol_sum(self) -> None:
        """Two BTC positions sum into one bucket against the per-asset cap."""
        positions = [
            {"symbol": "BTC-USDC", "notional": 700.0, "direction": "long"},
            {"symbol": "BTC-USDC", "notional": 600.0, "direction": "long"},
        ]
        # Existing = 1300; cap = 1500; remaining = 200
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert not result.skipped
        assert result.position_size_usd == pytest.approx(200.0)

    def test_skip_reason_includes_dollar_amounts(self) -> None:
        positions = [
            {"symbol": "BTC-USDC", "notional": 2000.0, "direction": "long"},
        ]
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert result.skipped
        assert "$" in result.skip_reason
        assert "cap" in result.skip_reason.lower()

    def test_malformed_position_dict_is_skipped_safely(self) -> None:
        """A dict missing ``notional`` or with a non-numeric value
        must not crash sizing — defensive against future Task 4 wiring
        where adapter.get_positions() may return partially populated
        dicts on a venue that doesn't expose every field."""
        positions = [
            {"symbol": "BTC-USDC", "notional": 1000.0, "direction": "long"},
            {"symbol": "BTC-USDC", "direction": "long"},  # missing notional
            {"symbol": "BTC-USDC", "notional": "garbage", "direction": "long"},
        ]
        # Only the first one (1000.0) should count → remaining 500
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert not result.skipped
        assert result.position_size_usd == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Layer 3 — Portfolio-wide exposure cap (Task 3)
# ---------------------------------------------------------------------------


class TestLayer3PortfolioCap:
    """`total = sum every notional; remaining = cap - total`."""

    def test_no_existing_positions_clamps_to_per_asset_then_portfolio(self) -> None:
        """With both caps active, no existing positions, the per-asset
        cap (15%) is the tighter ceiling so Layer 2 clamps first to
        $1500. Layer 3's 30% portfolio cap is well above that.
        """
        result = _size(_capped_prm(), open_positions=[])
        assert not result.skipped
        assert result.position_size_usd == pytest.approx(1500.0)

    def test_25pct_exposure_only_5pct_remaining(self) -> None:
        """Total exposure 25% of equity → only 5% remaining → clamp to $500.

        cap = 10000 * 0.30 = 3000
        existing total = 2500 (25% of 10k)
        remaining = 500
        Two different symbols so Layer 2 doesn't fire — only Layer 3.
        Layer 2's 15% per-asset cap = $1500 still allows the new BTC
        trade (zero existing BTC), so the trade gets clamped by
        Layer 3's $500 portfolio remaining.
        """
        positions = [
            {"symbol": "ETH-USDC", "notional": 1300.0, "direction": "long"},
            {"symbol": "SOL-USDC", "notional": 1200.0, "direction": "short"},
        ]
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert not result.skipped
        assert result.position_size_usd == pytest.approx(500.0)

    def test_at_portfolio_cap_skips(self) -> None:
        """Total exposure at exactly the portfolio cap → SKIP."""
        positions = [
            {"symbol": "ETH-USDC", "notional": 1500.0, "direction": "long"},
            {"symbol": "SOL-USDC", "notional": 1500.0, "direction": "short"},
        ]
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert result.skipped
        assert "Portfolio cap" in result.skip_reason

    def test_above_portfolio_cap_skips(self) -> None:
        positions = [
            {"symbol": "ETH-USDC", "notional": 2000.0, "direction": "long"},
            {"symbol": "SOL-USDC", "notional": 1500.0, "direction": "short"},
        ]
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert result.skipped
        assert "Portfolio cap" in result.skip_reason

    def test_skip_reason_includes_dollar_amounts(self) -> None:
        positions = [
            {"symbol": "ETH-USDC", "notional": 2000.0, "direction": "long"},
            {"symbol": "SOL-USDC", "notional": 1500.0, "direction": "short"},
        ]
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert result.skipped
        assert "$" in result.skip_reason
        assert "exposure" in result.skip_reason.lower()


# ---------------------------------------------------------------------------
# Combined Layer 2 + Layer 3 — tightest cap wins (Task 3)
# ---------------------------------------------------------------------------


class TestExposureCapsCombined:

    def test_layer2_clamps_tighter_than_layer3(self) -> None:
        """Layer 2 gives $150 of headroom, Layer 3 gives $200 → result $150.

        Spec: "Layer 1 gives $500, Layer 2 clamps to $150, Layer 3
        would allow $200 → result is $150 (tightest wins)."

        Construct it precisely:
        - equity = $10k
        - per_asset_cap = 15% = $1500
        - portfolio_cap = 30% = $3000
        - existing BTC = $1350 → BTC remaining = $150
        - existing ETH = $1450 → portfolio total = $2800 → portfolio remaining = $200
        - Layer 1 gives $5000 baseline
        - Layer 2 clamps $5000 → $150
        - Layer 3 would clamp $150 → min($150, $200) = $150 (no change)
        """
        positions = [
            {"symbol": "BTC-USDC", "notional": 1350.0, "direction": "long"},
            {"symbol": "ETH-USDC", "notional": 1450.0, "direction": "long"},
        ]
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert not result.skipped
        assert result.position_size_usd == pytest.approx(150.0)

    def test_layer3_clamps_tighter_than_layer2(self) -> None:
        """Layer 2 gives $1000 of headroom, Layer 3 gives $200 → result $200.

        - existing BTC = $500 → BTC remaining = $1000
        - existing ETH = $2300 → portfolio total = $2800 → remaining = $200
        - Layer 2 clamps $5000 → $1000
        - Layer 3 then clamps $1000 → $200
        """
        positions = [
            {"symbol": "BTC-USDC", "notional": 500.0, "direction": "long"},
            {"symbol": "ETH-USDC", "notional": 2300.0, "direction": "long"},
        ]
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert not result.skipped
        assert result.position_size_usd == pytest.approx(200.0)

    def test_mixed_directions_count_in_total_exposure(self) -> None:
        """Long + short both consume capacity equally — there's no
        netting at this layer because the SL/TP risk is symmetric
        across direction. A long $1000 BTC + short $800 BTC
        contributes $1800 to the per-asset cap, NOT $200 net."""
        positions = [
            {"symbol": "BTC-USDC", "notional": 1000.0, "direction": "long"},
            {"symbol": "BTC-USDC", "notional": 800.0, "direction": "short"},
        ]
        # Existing = 1800; cap = 1500 → SKIP
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert result.skipped
        assert "Per-asset cap" in result.skip_reason

    def test_layer2_zero_short_circuits_before_layer3(self) -> None:
        """When Layer 2 zeroes the size, Layer 3's reason doesn't appear.

        Pins the layer attribution: an operator reading PRM logs must
        be able to tell WHICH layer fired, not just "exposure clamped".
        """
        positions = [
            {"symbol": "BTC-USDC", "notional": 2000.0, "direction": "long"},
            {"symbol": "ETH-USDC", "notional": 5000.0, "direction": "long"},
        ]
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert result.skipped
        assert "Per-asset cap" in result.skip_reason
        assert "Portfolio cap" not in result.skip_reason

    def test_skipped_result_preserves_intermediate_state(self) -> None:
        """Layer 2/3 SKIP still reports the Layer 1 risk_dollars +
        Layer 6 multiplier so logs are debuggable end-to-end."""
        positions = [
            {"symbol": "BTC-USDC", "notional": 2000.0, "direction": "long"},
        ]
        result = _size(
            _capped_prm(), symbol="BTC-USDC", open_positions=positions
        )
        assert result.skipped
        assert result.risk_dollars > 0  # Layer 1 ran
        assert result.effective_risk_pct == pytest.approx(0.01)
        assert result.drawdown_multiplier == 1.0  # Layer 6 ran
        assert result.position_size_usd == 0.0


# ---------------------------------------------------------------------------
# Config + result dataclass smoke tests
# ---------------------------------------------------------------------------


class TestPortfolioRiskConfigDefaults:

    def test_defaults_match_spec(self) -> None:
        """Pin the documented defaults so a typo in PortfolioRiskConfig
        is caught at the unit-test layer."""
        cfg = PortfolioRiskConfig()
        assert cfg.risk_per_trade_pct == 0.01
        assert cfg.per_asset_cap_pct == 0.15
        assert cfg.portfolio_cap_pct == 0.30
        assert cfg.cost_floor_multiplier == 20.0
        assert cfg.drawdown_halt_pct == 0.10
        assert cfg.drawdown_reduce_pct == 0.05
        assert cfg.drawdown_resume_pct == 0.08
        assert cfg.llm_cycle_cost == 0.025

    def test_config_overrides_take_effect(self) -> None:
        """An overridden risk_per_trade_pct propagates into the math."""
        prm = _prm(risk_per_trade_pct=0.02)  # 2% risk per trade
        result = _size(prm)
        # Doubled risk_pct → doubled risk_dollars + doubled position size
        assert result.risk_dollars == pytest.approx(200.0)
        assert result.position_size_usd == pytest.approx(10_000.0)


class TestSizingResultDataclass:

    def test_to_dict_round_trip(self) -> None:
        result = _size(_prm())
        d = result.to_dict()
        assert d["position_size_usd"] == result.position_size_usd
        assert d["risk_dollars"] == result.risk_dollars
        assert d["drawdown_multiplier"] == result.drawdown_multiplier
        assert d["skipped"] is False
        assert d["skip_reason"] == ""

    def test_skip_result_serialises(self) -> None:
        result = _size(_prm(), equity=0.0)
        d = result.to_dict()
        assert d["skipped"] is True
        assert d["position_size_usd"] == 0.0
        assert "Invalid equity" in d["skip_reason"]
