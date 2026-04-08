"""Confidence decay + merge for alpha factors.

ARCHITECTURE.md §13.1.4 specifies the decay formula:

    decay_weight = max(0.0, 1.0 - (days_since_last_confirmed / 30))

A factor confirmed today has weight 1.0; one not confirmed for 30+
days has weight 0.0. ConvictionAgent already ignores factors below
``DECAY_PRUNE_THRESHOLD`` (0.1), so the next agent run drops them
from the JSON to keep the file lean.

The merge logic keeps the system continuously self-correcting:

* A new factor that matches an existing one (same key tuple) bumps
  ``last_confirmed`` to "now" and resets ``decay_weight`` to 1.0.
  ``discovered_at`` is preserved — that's a permanent record.
* A new factor with no match is added fresh.
* An existing factor with no match in the new run is kept but
  with its decayed weight; if that weight falls below the prune
  threshold it's dropped.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from mcp.quant_scientist.factor import AlphaFactor

# ConvictionAgent ignores anything below this. Pruning at the same
# threshold keeps the JSON file in sync with what the consumer reads.
DECAY_PRUNE_THRESHOLD: float = 0.1

# Number of days before decay_weight reaches 0.0. Per §13.1.4.
DECAY_HORIZON_DAYS: float = 30.0


# ---------------------------------------------------------------------------
# Decay helpers
# ---------------------------------------------------------------------------


def decay_weight_for_age(days_since_confirmed: float) -> float:
    """Compute the decay weight for a factor of the given age in days.

    Pure function — same age in, same weight out. Easier to test in
    isolation than the dataclass-mutating wrapper.
    """
    if days_since_confirmed <= 0:
        return 1.0
    return max(0.0, 1.0 - (days_since_confirmed / DECAY_HORIZON_DAYS))


def apply_decay(
    existing_factors: Iterable[AlphaFactor],
    current_time: datetime | None = None,
    prune_threshold: float = DECAY_PRUNE_THRESHOLD,
) -> list[AlphaFactor]:
    """Apply the decay formula to every factor and prune the weakest.

    Args:
        existing_factors: Factors loaded from the previous run's
            ``alpha_factors.json`` (or any iterable).
        current_time: Override the "now" used for age computation.
            Defaults to UTC now. Tests pass a fixed datetime.
        prune_threshold: Factors whose recomputed ``decay_weight``
            falls strictly below this are dropped from the result.

    Returns:
        A new list of :class:`AlphaFactor` with updated weights.
        Frozen dataclass instances are replaced via ``with_updates``.
    """
    now = _ensure_utc(current_time or datetime.now(tz=timezone.utc))
    out: list[AlphaFactor] = []
    for factor in existing_factors:
        try:
            confirmed_at = _parse_iso(factor.last_confirmed)
        except ValueError:
            # Bad timestamp on disk — be conservative and drop it.
            continue
        age_days = (now - confirmed_at).total_seconds() / 86_400.0
        new_weight = decay_weight_for_age(age_days)
        if new_weight < prune_threshold:
            continue
        out.append(factor.with_updates(decay_weight=new_weight))
    return out


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_factors(
    new_factors: Iterable[AlphaFactor],
    existing_factors: Iterable[AlphaFactor],
    current_time: datetime | None = None,
    prune_threshold: float = DECAY_PRUNE_THRESHOLD,
) -> tuple[list[AlphaFactor], dict[str, int]]:
    """Merge a fresh discovery batch with the previously persisted set.

    Rules (per §13.1.4):

    * **Re-confirmed** — a new factor whose key matches an existing
      factor: ``last_confirmed`` updated to "now", ``decay_weight``
      reset to 1.0, but ``discovered_at`` preserved.
    * **New** — a new factor with no match: added with
      ``decay_weight=1.0`` and ``discovered_at=last_confirmed=now``
      if it doesn't already carry timestamps.
    * **Decayed** — an existing factor with no match in the new
      batch: kept with its decayed weight; pruned if below threshold.

    Args:
        new_factors: Output from the latest LLM analysis run.
        existing_factors: Result of :func:`apply_decay` on the
            previous run's persisted factors.
        current_time: Override "now" for the timestamp updates.
        prune_threshold: Threshold for the decayed-and-not-re-confirmed
            keep/drop decision.

    Returns:
        ``(merged, counts)`` where ``counts`` is a dict with keys
        ``new``, ``confirmed``, ``decayed``, ``pruned``. The runner
        CLI uses this to build its summary line.
    """
    now = _ensure_utc(current_time or datetime.now(tz=timezone.utc))
    now_iso = now.isoformat().replace("+00:00", "Z")

    existing_by_key: dict[tuple[str, str, str], AlphaFactor] = {
        f.key: f for f in existing_factors
    }
    new_factors_list = list(new_factors)
    new_keys = {f.key for f in new_factors_list}

    merged: list[AlphaFactor] = []
    counts = {"new": 0, "confirmed": 0, "decayed": 0, "pruned": 0}

    # 1. Process the fresh batch first (new + re-confirmed).
    for new_f in new_factors_list:
        existing = existing_by_key.get(new_f.key)
        if existing is None:
            counts["new"] += 1
            # Stamp timestamps if the new factor came in without them
            # (the LLM may not include them).
            stamped = new_f.with_updates(
                discovered_at=new_f.discovered_at or now_iso,
                last_confirmed=new_f.last_confirmed or now_iso,
                decay_weight=1.0,
            )
            merged.append(stamped)
        else:
            counts["confirmed"] += 1
            # Re-confirmed: bump last_confirmed + decay_weight, but
            # PRESERVE the original discovered_at.
            merged.append(
                new_f.with_updates(
                    discovered_at=existing.discovered_at,
                    last_confirmed=now_iso,
                    decay_weight=1.0,
                )
            )

    # 2. Process existing factors that the new batch did NOT re-confirm.
    for existing_f in existing_by_key.values():
        if existing_f.key in new_keys:
            continue
        if existing_f.decay_weight < prune_threshold:
            counts["pruned"] += 1
            continue
        counts["decayed"] += 1
        merged.append(existing_f)

    return merged, counts


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_iso(value: str) -> datetime:
    """Parse an ISO 8601 timestamp tolerating a trailing 'Z'."""
    if value is None:
        raise ValueError("None ISO timestamp")
    cleaned = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    return _ensure_utc(dt)
