"""Hourly delta check for the Macro Regime Manager.

Per ARCHITECTURE.md §13.2.1 the Lightweight Check is the cheap tier:
no LLM call, just compares the current snapshot against the previous
one and decides whether to escalate to a deep analysis.

Trigger rules (any one suffices):

  * VIX moved > 5% since last check
  * DXY moved > 1% since last check
  * DVOL moved > 10% since last check
  * Fear & Greed *category* changed (e.g. "fear" → "extreme fear")
  * Hyperliquid total OI changed > 10%
  * Any HIGH-impact economic event within the next 24 hours
  * Weekend special: VIX/DXY are stale (> 24h) → DVOL spike of 15%
    or more triggers regardless

The previous snapshot is persisted to ``macro_regime_snapshot.json``
beside the ``macro_regime.json`` output. Each call writes the new
snapshot back so the next tick has something to diff against.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mcp.macro_regime.data_fetcher import (
    MacroSnapshot,
    filter_calendar_within,
    parse_iso,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — pulled out so tests can verify the literal §13.2.1 thresholds
# ---------------------------------------------------------------------------

VIX_DELTA_THRESHOLD_PCT = 5.0
DXY_DELTA_THRESHOLD_PCT = 1.0
DVOL_DELTA_THRESHOLD_PCT = 10.0
HL_OI_DELTA_THRESHOLD_PCT = 10.0
WEEKEND_DVOL_DELTA_THRESHOLD_PCT = 15.0
ECONOMIC_EVENT_LOOKAHEAD_HOURS = 24.0
TRADFI_STALENESS_HOURS = 24.0


@dataclass
class CheckResult:
    """Result of one Lightweight Check.

    ``should_trigger_deep`` tells the runner whether to escalate to a
    Deep Analysis. ``reasons`` is the human-readable list of triggers
    (one entry per rule that fired) — surfaced in the runner summary
    AND fed into the deep-analysis prompt as urgency context.
    """

    should_trigger_deep: bool
    reasons: list[str] = field(default_factory=list)
    snapshot: MacroSnapshot | None = None
    previous: MacroSnapshot | None = None

    def to_dict(self) -> dict:
        return {
            "should_trigger_deep": self.should_trigger_deep,
            "reasons": list(self.reasons),
            "snapshot": self.snapshot.to_dict() if self.snapshot else None,
        }


# ---------------------------------------------------------------------------
# LightweightCheck
# ---------------------------------------------------------------------------


class LightweightCheck:
    """Pure-code (no LLM) delta gate for the Macro Regime Manager.

    Construction is cheap. Call :meth:`run` once per snapshot — it
    persists the snapshot to disk as a side effect so the next tick
    has a baseline.
    """

    def __init__(
        self,
        snapshot_path: Path | str = "macro_regime_snapshot.json",
        *,
        clock=None,
    ) -> None:
        self._snapshot_path = Path(snapshot_path)
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_previous(self) -> MacroSnapshot | None:
        """Load the persisted snapshot from the previous tick."""
        if not self._snapshot_path.exists():
            return None
        try:
            payload = json.loads(self._snapshot_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                f"LightweightCheck: could not read previous snapshot: {e}"
            )
            return None
        try:
            return MacroSnapshot.from_dict(payload)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"LightweightCheck: previous snapshot malformed: {e}")
            return None

    def save_snapshot(self, snapshot: MacroSnapshot) -> None:
        """Persist the current snapshot for the next tick to diff against."""
        try:
            self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._snapshot_path.with_suffix(self._snapshot_path.suffix + ".tmp")
            tmp.write_text(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True))
            tmp.replace(self._snapshot_path)
        except OSError as e:
            logger.warning(f"LightweightCheck: snapshot write failed: {e}")

    # ------------------------------------------------------------------
    # The check itself
    # ------------------------------------------------------------------

    def run(
        self,
        current: MacroSnapshot,
        previous: MacroSnapshot | None = None,
    ) -> CheckResult:
        """Compare ``current`` against ``previous`` and decide.

        Always persists ``current`` to disk before returning so the
        next tick has a baseline — even if no trigger fired.
        """
        if previous is None:
            previous = self.load_previous()

        reasons: list[str] = []
        now = self._clock()

        weekend_mode = self._is_weekend_with_stale_tradfi(current, now)

        # ---- Rule 1: VIX delta (skipped during weekend mode)
        if not weekend_mode:
            r = _pct_change_reason(
                "VIX", previous.vix if previous else None, current.vix,
                VIX_DELTA_THRESHOLD_PCT,
            )
            if r:
                reasons.append(r)

        # ---- Rule 2: DXY delta (skipped during weekend mode)
        if not weekend_mode:
            r = _pct_change_reason(
                "DXY", previous.dxy if previous else None, current.dxy,
                DXY_DELTA_THRESHOLD_PCT,
            )
            if r:
                reasons.append(r)

        # ---- Rule 3: DVOL delta — looser threshold on weekends
        dvol_threshold = (
            WEEKEND_DVOL_DELTA_THRESHOLD_PCT if weekend_mode else DVOL_DELTA_THRESHOLD_PCT
        )
        r = _pct_change_reason(
            "DVOL",
            previous.dvol if previous else None,
            current.dvol,
            dvol_threshold,
        )
        if r:
            reasons.append(r)

        # ---- Rule 4: Fear & Greed category change
        if previous and previous.fear_greed_classification and current.fear_greed_classification:
            prev_cat = _normalise_category(previous.fear_greed_classification)
            curr_cat = _normalise_category(current.fear_greed_classification)
            if prev_cat != curr_cat:
                reasons.append(
                    f"Fear&Greed category changed: {prev_cat} → {curr_cat}"
                )

        # ---- Rule 5: HL OI delta
        r = _pct_change_reason(
            "HL_OI",
            previous.hl_total_oi if previous else None,
            current.hl_total_oi,
            HL_OI_DELTA_THRESHOLD_PCT,
        )
        if r:
            reasons.append(r)

        # ---- Rule 6: Economic event within 24 hours
        upcoming = filter_calendar_within(
            current.economic_calendar, now, ECONOMIC_EVENT_LOOKAHEAD_HOURS
        )
        if upcoming:
            event_names = sorted({e.name for e in upcoming if e.impact == "HIGH"})
            if event_names:
                reasons.append(
                    f"HIGH-impact economic event within "
                    f"{int(ECONOMIC_EVENT_LOOKAHEAD_HOURS)}h: "
                    f"{', '.join(event_names)}"
                )

        result = CheckResult(
            should_trigger_deep=bool(reasons),
            reasons=reasons,
            snapshot=current,
            previous=previous,
        )

        self.save_snapshot(current)
        return result

    # ------------------------------------------------------------------
    # Weekend / staleness detection
    # ------------------------------------------------------------------

    def _is_weekend_with_stale_tradfi(
        self, snapshot: MacroSnapshot, now: datetime
    ) -> bool:
        """Saturday/Sunday + VIX or DXY data older than 24h."""
        weekday = now.weekday()  # Mon=0 ... Sun=6
        if weekday not in (5, 6):
            return False

        for ts in (snapshot.vix_timestamp, snapshot.dxy_timestamp):
            ts_dt = parse_iso(ts)
            if ts_dt is None:
                # No timestamp = treat as stale
                continue
            if (now - ts_dt) <= timedelta(hours=TRADFI_STALENESS_HOURS):
                return False
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pct_change_reason(
    label: str,
    previous: float | None,
    current: float | None,
    threshold_pct: float,
) -> str | None:
    """Return a reason string if the move exceeds threshold."""
    if previous is None or current is None:
        return None
    if previous == 0:
        return None
    delta_pct = (current - previous) / abs(previous) * 100.0
    if abs(delta_pct) >= threshold_pct:
        return (
            f"{label} moved {delta_pct:+.2f}% "
            f"(threshold ±{threshold_pct:.1f}%)"
        )
    return None


def _normalise_category(value: str) -> str:
    """Lower-case + strip so 'Extreme Fear' == 'extreme fear'."""
    return (value or "").strip().lower()
