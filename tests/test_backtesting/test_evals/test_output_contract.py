"""Tests for the EvalOutput contract dataclass."""

from __future__ import annotations

from backtesting.evals.output_contract import EvalOutput


def _make(**overrides) -> EvalOutput:
    base = dict(
        direction="LONG",
        conviction=0.75,
        sl_price=64500.0,
        tp1_price=65300.0,
        tp2_price=66100.0,
        position_size_pct=0.1,
        reasoning="bull flag with rising volume",
        latency_ms=2300.0,
        model_id="claude-sonnet-4-6",
    )
    base.update(overrides)
    return EvalOutput(**base)


def test_construction_minimal():
    out = _make()
    assert out.direction == "LONG"
    assert out.conviction == 0.75
    assert out.teacher_agreement is None
    assert out.conviction_calibration is None


def test_to_dict_round_trip():
    out = _make()
    d = out.to_dict()
    assert d["direction"] == "LONG"
    assert d["conviction"] == 0.75
    assert d["model_id"] == "claude-sonnet-4-6"
    out2 = EvalOutput.from_dict(d)
    assert out2 == out


def test_from_dict_drops_unknown_keys():
    """A future schema with extra fields shouldn't break old loaders."""
    payload = _make().to_dict()
    payload["unknown_future_field"] = "ignored"
    out = EvalOutput.from_dict(payload)
    assert out.direction == "LONG"


def test_distilled_model_can_have_null_reasoning():
    out = _make(reasoning=None, model_id="distilled-7b-v1")
    assert out.reasoning is None
    assert out.model_id == "distilled-7b-v1"


def test_optional_price_fields_can_be_null():
    out = _make(sl_price=None, tp1_price=None, tp2_price=None, position_size_pct=None)
    assert out.sl_price is None
    assert out.tp1_price is None
    assert out.tp2_price is None
    assert out.position_size_pct is None


def test_teacher_agreement_can_be_set_after_construction():
    out = _make()
    out.teacher_agreement = 0.92
    out.conviction_calibration = 0.85
    assert out.to_dict()["teacher_agreement"] == 0.92
    assert out.to_dict()["conviction_calibration"] == 0.85
