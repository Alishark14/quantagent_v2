"""Tests for AlphaFactor + AlphaFactorsReport + nested-JSON conversion."""

from __future__ import annotations

import pytest

from mcp.quant_scientist.factor import (
    AlphaFactor,
    AlphaFactorsReport,
    factors_to_nested_json,
    nested_json_to_factors,
)


def _factor(
    pattern: str = "ascending_triangle",
    symbol: str = "BTC-USDC",
    timeframe: str = "1h",
    win_rate: float = 0.65,
    avg_r: float = 1.8,
    n: int = 22,
    confidence: str = "high",
    decay_weight: float = 1.0,
    note: str | None = None,
    discovered_at: str = "2026-04-08T00:00:00+00:00",
    last_confirmed: str = "2026-04-08T00:00:00+00:00",
) -> AlphaFactor:
    return AlphaFactor(
        pattern=pattern,
        symbol=symbol,
        timeframe=timeframe,
        win_rate=win_rate,
        avg_r=avg_r,
        n=n,
        confidence=confidence,
        discovered_at=discovered_at,
        last_confirmed=last_confirmed,
        decay_weight=decay_weight,
        note=note,
    )


# ---------------------------------------------------------------------------
# AlphaFactor — basic invariants
# ---------------------------------------------------------------------------


def test_alpha_factor_key_is_three_tuple():
    f = _factor()
    assert f.key == ("BTC-USDC", "1h", "ascending_triangle")


def test_alpha_factor_with_updates_returns_new_instance():
    f = _factor(decay_weight=0.5)
    updated = f.with_updates(decay_weight=1.0)
    assert f.decay_weight == 0.5  # original unchanged
    assert updated.decay_weight == 1.0
    assert f is not updated


def test_alpha_factor_validate_rejects_bad_win_rate():
    with pytest.raises(ValueError, match="win_rate"):
        _factor(win_rate=1.5).validate()
    with pytest.raises(ValueError, match="win_rate"):
        _factor(win_rate=-0.1).validate()


def test_alpha_factor_validate_rejects_bad_decay_weight():
    with pytest.raises(ValueError, match="decay_weight"):
        _factor(decay_weight=2.0).validate()


def test_alpha_factor_validate_rejects_unknown_confidence():
    with pytest.raises(ValueError, match="confidence"):
        _factor(confidence="MAYBE").validate()


def test_alpha_factor_validate_rejects_negative_n():
    with pytest.raises(ValueError, match="n"):
        _factor(n=-1).validate()


def test_alpha_factor_validate_rejects_empty_pattern():
    with pytest.raises(ValueError, match="pattern"):
        _factor(pattern="").validate()


def test_alpha_factor_validate_passes_for_well_formed():
    _factor().validate()  # should not raise


# ---------------------------------------------------------------------------
# from_payload — schema parser
# ---------------------------------------------------------------------------


def test_from_payload_accepts_avg_R_capital():
    """The on-disk JSON uses ``avg_R`` (capital R) per the spec."""
    payload = {
        "win_rate": 0.68,
        "avg_R": 1.9,
        "n": 23,
        "confidence": "high",
        "discovered_at": "2026-04-05T02:30:00Z",
        "last_confirmed": "2026-04-05T02:30:00Z",
        "decay_weight": 1.0,
    }
    factor = AlphaFactor.from_payload("BTC-USDC", "1h", "asc_triangle", payload)
    assert factor.avg_r == pytest.approx(1.9)
    assert factor.symbol == "BTC-USDC"


def test_from_payload_accepts_avg_r_lowercase():
    payload = {
        "win_rate": 0.68,
        "avg_r": 1.9,
        "n": 23,
        "confidence": "high",
        "discovered_at": "2026-04-05T02:30:00Z",
        "last_confirmed": "2026-04-05T02:30:00Z",
        "decay_weight": 1.0,
    }
    factor = AlphaFactor.from_payload("BTC-USDC", "1h", "asc_triangle", payload)
    assert factor.avg_r == pytest.approx(1.9)


def test_from_payload_preserves_optional_note():
    payload = {
        "win_rate": 0.31,
        "avg_R": -0.4,
        "n": 16,
        "confidence": "high",
        "discovered_at": "2026-04-01T02:30:00Z",
        "last_confirmed": "2026-04-05T02:30:00Z",
        "decay_weight": 1.0,
        "note": "AVOID",
    }
    factor = AlphaFactor.from_payload("ETH-USDC", "1h", "bearish_flag", payload)
    assert factor.note == "AVOID"


def test_from_payload_rejects_missing_avg_r():
    payload = {
        "win_rate": 0.65,
        "n": 20,
        "confidence": "high",
        "discovered_at": "2026-04-08T00:00:00Z",
        "last_confirmed": "2026-04-08T00:00:00Z",
        "decay_weight": 1.0,
    }
    with pytest.raises(ValueError, match="avg_r"):
        AlphaFactor.from_payload("BTC-USDC", "1h", "x", payload)


# ---------------------------------------------------------------------------
# Nested JSON round-trip
# ---------------------------------------------------------------------------


def test_factors_to_nested_json_groups_by_symbol_and_timeframe():
    factors = [
        _factor(symbol="BTC-USDC", timeframe="1h", pattern="A"),
        _factor(symbol="BTC-USDC", timeframe="1h", pattern="B"),
        _factor(symbol="BTC-USDC", timeframe="4h", pattern="C"),
        _factor(symbol="ETH-USDC", timeframe="1h", pattern="D"),
    ]
    nested = factors_to_nested_json(factors)
    assert set(nested.keys()) == {"BTC-USDC", "ETH-USDC"}
    assert set(nested["BTC-USDC"].keys()) == {"1h", "4h"}
    assert set(nested["BTC-USDC"]["1h"].keys()) == {"A", "B"}
    assert nested["BTC-USDC"]["4h"]["C"]["win_rate"] == pytest.approx(0.65)
    assert "avg_R" in nested["BTC-USDC"]["1h"]["A"]  # capital R per spec


def test_factors_to_nested_json_emits_note_only_when_present():
    factors = [
        _factor(pattern="positive"),
        _factor(pattern="negative", avg_r=-0.4, win_rate=0.31, note="AVOID"),
    ]
    nested = factors_to_nested_json(factors)
    assert "note" not in nested["BTC-USDC"]["1h"]["positive"]
    assert nested["BTC-USDC"]["1h"]["negative"]["note"] == "AVOID"


def test_nested_json_to_factors_round_trips():
    originals = [
        _factor(pattern="A", win_rate=0.65, avg_r=1.8),
        _factor(pattern="B", symbol="ETH-USDC", note="AVOID", avg_r=-0.5, win_rate=0.30),
    ]
    nested = factors_to_nested_json(originals)
    parsed = nested_json_to_factors(nested)
    assert len(parsed) == 2
    parsed_by_pattern = {f.pattern: f for f in parsed}
    assert parsed_by_pattern["A"].avg_r == pytest.approx(1.8)
    assert parsed_by_pattern["B"].note == "AVOID"
    assert parsed_by_pattern["B"].avg_r == pytest.approx(-0.5)


def test_nested_json_to_factors_drops_malformed_entries():
    nested = {
        "BTC-USDC": {
            "1h": {
                "good": {
                    "win_rate": 0.6,
                    "avg_R": 1.7,
                    "n": 20,
                    "confidence": "high",
                    "discovered_at": "2026-04-08T00:00:00Z",
                    "last_confirmed": "2026-04-08T00:00:00Z",
                    "decay_weight": 1.0,
                },
                "bad_no_n": {
                    "win_rate": 0.6,
                    "avg_R": 1.7,
                    "confidence": "high",
                    "discovered_at": "2026-04-08T00:00:00Z",
                    "last_confirmed": "2026-04-08T00:00:00Z",
                    "decay_weight": 1.0,
                },
                "bad_win_rate": {
                    "win_rate": 9.0,  # invalid
                    "avg_R": 1.7,
                    "n": 20,
                    "confidence": "high",
                    "discovered_at": "2026-04-08T00:00:00Z",
                    "last_confirmed": "2026-04-08T00:00:00Z",
                    "decay_weight": 1.0,
                },
            }
        }
    }
    parsed = nested_json_to_factors(nested)
    assert [f.pattern for f in parsed] == ["good"]


def test_nested_json_to_factors_handles_empty_input():
    assert nested_json_to_factors({}) == []
    assert nested_json_to_factors(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AlphaFactorsReport
# ---------------------------------------------------------------------------


def test_report_factor_count_matches_list_length():
    report = AlphaFactorsReport(factors=[_factor("a"), _factor("b")])
    assert report.factor_count == 2


def test_report_to_dict_round_trips():
    report = AlphaFactorsReport(
        factors=[_factor()],
        new_count=1,
        confirmed_count=0,
        pruned_count=0,
        trades_analyzed=42,
        symbols_analyzed=3,
        output_path="alpha_factors.json",
        dry_run=False,
    )
    d = report.to_dict()
    assert d["factor_count"] == 1
    assert d["trades_analyzed"] == 42
    assert d["factors"][0]["pattern"] == "ascending_triangle"
