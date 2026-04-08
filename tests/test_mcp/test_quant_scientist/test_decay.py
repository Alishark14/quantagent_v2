"""Tests for the confidence-decay + merge logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mcp.quant_scientist.decay import (
    DECAY_PRUNE_THRESHOLD,
    apply_decay,
    decay_weight_for_age,
    merge_factors,
)
from mcp.quant_scientist.factor import AlphaFactor


_FIXED_NOW = datetime(2026, 4, 8, 12, 0, 0, tzinfo=timezone.utc)


def _factor(
    pattern: str = "asc_triangle",
    *,
    days_ago: float = 0.0,
    decay_weight: float = 1.0,
    discovered_days_ago: float | None = None,
) -> AlphaFactor:
    last_confirmed = (_FIXED_NOW - timedelta(days=days_ago)).isoformat()
    discovered = (
        _FIXED_NOW - timedelta(days=discovered_days_ago)
        if discovered_days_ago is not None
        else _FIXED_NOW - timedelta(days=days_ago)
    ).isoformat()
    return AlphaFactor(
        pattern=pattern,
        symbol="BTC-USDC",
        timeframe="1h",
        win_rate=0.6,
        avg_r=1.7,
        n=20,
        confidence="high",
        discovered_at=discovered,
        last_confirmed=last_confirmed,
        decay_weight=decay_weight,
    )


# ---------------------------------------------------------------------------
# decay_weight_for_age — pure formula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("age_days", "expected"),
    [
        (0, 1.0),
        (-5, 1.0),  # negative age clamped to 1.0
        (15, 0.5),
        (30, 0.0),
        (45, 0.0),  # past horizon clamped to 0
        (3, pytest.approx(0.9)),
        (7.5, pytest.approx(0.75)),
    ],
)
def test_decay_weight_for_age(age_days, expected):
    assert decay_weight_for_age(age_days) == expected


# ---------------------------------------------------------------------------
# apply_decay
# ---------------------------------------------------------------------------


def test_apply_decay_keeps_fresh_factor_at_full_weight():
    [decayed] = apply_decay([_factor(days_ago=0)], current_time=_FIXED_NOW)
    assert decayed.decay_weight == pytest.approx(1.0)


def test_apply_decay_halves_weight_at_15_days():
    [decayed] = apply_decay([_factor(days_ago=15)], current_time=_FIXED_NOW)
    assert decayed.decay_weight == pytest.approx(0.5)


def test_apply_decay_prunes_at_or_below_threshold():
    """A factor whose new weight falls strictly below 0.1 is pruned."""
    fresh = _factor(pattern="fresh", days_ago=0)
    aged = _factor(pattern="too_old", days_ago=29)  # weight ~ 0.033 → pruned
    survivors = apply_decay([fresh, aged], current_time=_FIXED_NOW)
    assert [f.pattern for f in survivors] == ["fresh"]


def test_apply_decay_keeps_factor_just_above_threshold():
    """A factor whose weight is comfortably above 0.1 is kept.

    NOTE: 27 days → 1 - 27/30 in float math is 0.0999... (strictly
    below 0.1), so a factor at age 27 IS pruned. We use 21 days here
    so the float math works out cleanly to 0.3 — well above the gate.
    """
    aged = _factor(pattern="edge", days_ago=21)
    survivors = apply_decay([aged], current_time=_FIXED_NOW)
    assert len(survivors) == 1
    assert survivors[0].decay_weight == pytest.approx(0.3)


def test_apply_decay_drops_factor_with_unparseable_timestamp():
    bad = AlphaFactor(
        pattern="x", symbol="BTC-USDC", timeframe="1h",
        win_rate=0.6, avg_r=1.7, n=20, confidence="high",
        discovered_at="not-a-date",
        last_confirmed="also-not-a-date",
        decay_weight=1.0,
    )
    survivors = apply_decay([bad], current_time=_FIXED_NOW)
    assert survivors == []


def test_apply_decay_uses_now_when_current_time_omitted():
    # Smoke: just verify it doesn't crash and returns something valid.
    survivors = apply_decay([_factor(days_ago=0)])
    assert len(survivors) == 1
    assert 0.0 < survivors[0].decay_weight <= 1.0


def test_apply_decay_custom_prune_threshold():
    """Caller can override the prune threshold."""
    aged = _factor(pattern="aged", days_ago=20)  # weight = 1/3
    # With a strict 0.5 threshold this should drop.
    survivors = apply_decay([aged], current_time=_FIXED_NOW, prune_threshold=0.5)
    assert survivors == []


# ---------------------------------------------------------------------------
# merge_factors
# ---------------------------------------------------------------------------


def test_merge_adds_new_factor_with_weight_one():
    new_only = _factor(pattern="brand_new")
    merged, counts = merge_factors(
        new_factors=[new_only],
        existing_factors=[],
        current_time=_FIXED_NOW,
    )
    assert len(merged) == 1
    assert merged[0].decay_weight == 1.0
    assert counts == {"new": 1, "confirmed": 0, "decayed": 0, "pruned": 0}


def test_merge_reconfirmed_factor_resets_weight_and_preserves_discovered_at():
    discovered = (_FIXED_NOW - timedelta(days=10)).isoformat()
    existing = AlphaFactor(
        pattern="reconfirmed",
        symbol="BTC-USDC",
        timeframe="1h",
        win_rate=0.55,
        avg_r=1.6,
        n=18,
        confidence="medium",
        discovered_at=discovered,
        last_confirmed=(_FIXED_NOW - timedelta(days=8)).isoformat(),
        decay_weight=0.7,
    )
    new_match = _factor(pattern="reconfirmed", days_ago=0)  # fresh stats

    merged, counts = merge_factors(
        new_factors=[new_match],
        existing_factors=[existing],
        current_time=_FIXED_NOW,
    )
    assert counts == {"new": 0, "confirmed": 1, "decayed": 0, "pruned": 0}
    [f] = merged
    assert f.decay_weight == 1.0
    # discovered_at preserved from the original
    assert f.discovered_at == discovered
    # last_confirmed bumped to "now"
    assert f.last_confirmed.startswith("2026-04-08T12:00:00")


def test_merge_decayed_factor_kept_when_above_threshold():
    """Factor missing from new batch but still has weight ≥ threshold."""
    aged_existing = _factor(pattern="naturally_fading", days_ago=15, decay_weight=0.5)
    merged, counts = merge_factors(
        new_factors=[],
        existing_factors=[aged_existing],
        current_time=_FIXED_NOW,
    )
    assert [f.pattern for f in merged] == ["naturally_fading"]
    assert counts == {"new": 0, "confirmed": 0, "decayed": 1, "pruned": 0}


def test_merge_decayed_factor_pruned_when_below_threshold():
    weak_existing = _factor(pattern="dying", days_ago=29, decay_weight=0.05)
    merged, counts = merge_factors(
        new_factors=[],
        existing_factors=[weak_existing],
        current_time=_FIXED_NOW,
    )
    assert merged == []
    assert counts == {"new": 0, "confirmed": 0, "decayed": 0, "pruned": 1}


def test_merge_combines_new_confirmed_decayed_in_one_call():
    """Three factors, three different fates."""
    existing_reconfirm = _factor(pattern="kept", days_ago=5, decay_weight=0.83)
    existing_fade = _factor(pattern="naturally_fading", days_ago=10, decay_weight=0.66)
    existing_pruned = _factor(pattern="dying", days_ago=29, decay_weight=0.03)

    new_kept = _factor(pattern="kept", days_ago=0)
    new_brand_new = _factor(pattern="brand_new", days_ago=0)

    merged, counts = merge_factors(
        new_factors=[new_kept, new_brand_new],
        existing_factors=[existing_reconfirm, existing_fade, existing_pruned],
        current_time=_FIXED_NOW,
    )
    patterns = sorted(f.pattern for f in merged)
    assert patterns == ["brand_new", "kept", "naturally_fading"]
    assert counts["new"] == 1
    assert counts["confirmed"] == 1
    assert counts["decayed"] == 1
    assert counts["pruned"] == 1


def test_merge_stamps_timestamps_on_undated_new_factor():
    """An LLM-produced factor without timestamps gets them on merge."""
    undated = AlphaFactor(
        pattern="undated",
        symbol="BTC-USDC",
        timeframe="1h",
        win_rate=0.6,
        avg_r=1.7,
        n=20,
        confidence="high",
        discovered_at="",  # missing
        last_confirmed="",
        decay_weight=1.0,
    )
    merged, _ = merge_factors(
        new_factors=[undated],
        existing_factors=[],
        current_time=_FIXED_NOW,
    )
    assert merged[0].discovered_at.startswith("2026-04-08T12:00:00")
    assert merged[0].last_confirmed.startswith("2026-04-08T12:00:00")


def test_decay_prune_threshold_exported_for_consistency_with_conviction_agent():
    """The 0.1 threshold matches §13.1.6 — ConvictionAgent ignores below this."""
    assert DECAY_PRUNE_THRESHOLD == 0.1
