"""MacroRegimeManager — deep / emergency LLM-powered macro assessment.

Per ARCHITECTURE §13.2: this is the offline agent that takes a
:class:`MacroSnapshot` (built by :class:`MacroDataFetcher`), runs an
LLM assessment via Claude, and writes ``macro_regime.json`` consumed
by ConvictionAgent + Sentinel for global risk-parameter overlay and
blackout-window enforcement.

Safety contract (§13.2 / §13.1.6 — same rule, both agents):

  * The agent reads from public APIs and writes to exactly ONE file
    (``self._output_path``). It is a read-only analyst over the
    trading system.
  * The LLM never directly writes the file. The agent parses the
    LLM JSON, builds the dataclass, and writes through its own code.
  * Parse failures, LLM errors, and write errors all log + return
    a ``MacroRegime`` with ``error`` set rather than crashing.

The blackout-window list is built deterministically from the
economic calendar — the LLM is NOT trusted to invent dates. The LLM's
job is to interpret the data and choose the regime + adjustments.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.macro_regime.data_fetcher import EconomicEvent, MacroSnapshot, parse_iso

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from llm.base import LLMProvider


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


VALID_REGIMES = {"RISK_ON", "RISK_OFF", "NEUTRAL"}
HIGH_IMPACT_EVENT_NAMES = {"FOMC_ANNOUNCEMENT", "CPI", "NFP"}
BLACKOUT_LOOKAHEAD_HOURS = 48.0
BLACKOUT_PRE_BUFFER_MINUTES = 60
BLACKOUT_POST_BUFFER_MINUTES = 30
DEFAULT_OUTPUT_PATH = Path("macro_regime.json")
REGIME_VALIDITY_HOURS = 24

_LLM_MAX_TOKENS = 1024
_LLM_TEMPERATURE = 0.2


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MacroAdjustments:
    """Risk-parameter overlay applied by ConvictionAgent / DecisionAgent."""

    conviction_threshold_boost: float = 0.0
    max_concurrent_positions_override: int | None = None
    position_size_multiplier: float = 1.0
    avoid_assets: list[str] = field(default_factory=list)
    prefer_assets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "conviction_threshold_boost": self.conviction_threshold_boost,
            "max_concurrent_positions_override": self.max_concurrent_positions_override,
            "position_size_multiplier": self.position_size_multiplier,
            "avoid_assets": list(self.avoid_assets),
            "prefer_assets": list(self.prefer_assets),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "MacroAdjustments":
        payload = payload or {}
        return cls(
            conviction_threshold_boost=float(
                payload.get("conviction_threshold_boost", 0.0) or 0.0
            ),
            max_concurrent_positions_override=_opt_int(
                payload.get("max_concurrent_positions_override")
            ),
            position_size_multiplier=float(
                payload.get("position_size_multiplier", 1.0) or 1.0
            ),
            avoid_assets=list(payload.get("avoid_assets") or []),
            prefer_assets=list(payload.get("prefer_assets") or []),
        )


@dataclass
class BlackoutWindow:
    """A scheduled period during which no new entries are permitted."""

    start: str  # ISO 8601 UTC
    end: str
    reason: str  # e.g. "FOMC_ANNOUNCEMENT"
    action: str = "execution_block"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "BlackoutWindow":
        return cls(
            start=str(payload.get("start", "")),
            end=str(payload.get("end", "")),
            reason=str(payload.get("reason", "")),
            action=str(payload.get("action", "execution_block")),
        )

    def contains(self, when: datetime) -> bool:
        start_dt = parse_iso(self.start)
        end_dt = parse_iso(self.end)
        if start_dt is None or end_dt is None:
            return False
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return start_dt <= when <= end_dt


@dataclass
class MacroRegime:
    """Full §13.2.3 output payload."""

    regime: str = "NEUTRAL"
    confidence: float = 0.0
    reasoning: str = ""
    adjustments: MacroAdjustments = field(default_factory=MacroAdjustments)
    blackout_windows: list[BlackoutWindow] = field(default_factory=list)
    generated_at: str = ""
    expires: str = ""
    error: str | None = None
    output_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "adjustments": self.adjustments.to_dict(),
            "blackout_windows": [b.to_dict() for b in self.blackout_windows],
            "generated_at": self.generated_at,
            "expires": self.expires,
        }

    def to_disk_dict(self) -> dict:
        """Schema written to disk — excludes runtime-only fields."""
        return self.to_dict()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are the Macro Regime Manager for an autonomous trading system.

Your job is to interpret a snapshot of macro-economic conditions and
classify the current regime as RISK_ON, RISK_OFF, or NEUTRAL, then
suggest concrete adjustments to the trading system's risk parameters.

You MUST respond with a single JSON object and NOTHING ELSE — no
prose, no markdown fences, no commentary. The JSON must contain
exactly these top-level keys:

  - regime: one of "RISK_ON", "RISK_OFF", "NEUTRAL"
  - confidence: float in [0.0, 1.0]
  - reasoning: 1-3 sentences explaining the classification
  - adjustments: object with the following keys:
      - conviction_threshold_boost: float (0.0 to 0.2)
      - max_concurrent_positions_override: integer or null
      - position_size_multiplier: float (0.3 to 1.5)
      - avoid_assets: array of internal symbol strings (BASE-QUOTE)
      - prefer_assets: array of internal symbol strings (BASE-QUOTE)

Rules:
  * RISK_OFF regimes get conviction_threshold_boost > 0 and
    position_size_multiplier < 1.0.
  * RISK_ON regimes get conviction_threshold_boost = 0 and
    position_size_multiplier >= 1.0 (cap at 1.5).
  * NEUTRAL regimes leave both at the default (0.0 and 1.0).
  * NEVER suggest blackout_windows in your output — they are
    derived deterministically from the economic calendar by the
    enclosing system. You only choose the regime + adjustments.
  * Be conservative when in doubt. SKIP-equivalent (NEUTRAL with
    a small boost) is always safer than a wrong RISK_ON call.
"""


def build_assessment_prompt(
    snapshot: MacroSnapshot,
    *,
    urgency: str = "normal",
    triggering_symbols: list[str] | None = None,
    reasons: list[str] | None = None,
) -> str:
    """Assemble the user-prompt body for the deep / emergency call."""
    lines: list[str] = []
    lines.append(f"## Macro snapshot fetched at {snapshot.fetched_at}")
    lines.append("")
    lines.append("### Sources available")
    if snapshot.available_sources:
        lines.append(", ".join(sorted(snapshot.available_sources)))
    else:
        lines.append("(none — all data sources failed to fetch)")
    lines.append("")
    lines.append("### Data values")
    lines.append(_fmt_value("VIX (CBOE volatility)", snapshot.vix))
    lines.append(_fmt_value("DXY (US dollar index)", snapshot.dxy))
    lines.append(_fmt_value("DVOL (Deribit BTC implied vol)", snapshot.dvol))
    lines.append(
        _fmt_value(
            "Crypto Fear & Greed",
            snapshot.fear_greed_value,
            suffix=(
                f" ({snapshot.fear_greed_classification})"
                if snapshot.fear_greed_classification
                else ""
            ),
        )
    )
    lines.append(_fmt_value("BTC Dominance %", snapshot.btc_dominance))
    lines.append(_fmt_value("Hyperliquid total OI (USD)", snapshot.hl_total_oi))
    lines.append(_fmt_value("Hyperliquid avg funding", snapshot.hl_avg_funding))
    lines.append("")
    lines.append("### Upcoming economic calendar")
    if snapshot.economic_calendar:
        for ev in snapshot.economic_calendar[:8]:
            lines.append(f"  - {ev.timestamp} {ev.name} ({ev.impact})")
    else:
        lines.append("  (no upcoming events on file)")
    lines.append("")
    lines.append(f"### Urgency: {urgency}")
    if reasons:
        lines.append("### Triggering reasons")
        for r in reasons:
            lines.append(f"  - {r}")
    if triggering_symbols:
        lines.append("### Triggering symbols (swarm consensus)")
        lines.append(", ".join(triggering_symbols))
    lines.append("")
    lines.append("Respond with the JSON object now. NO other text.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MacroRegimeManager
# ---------------------------------------------------------------------------


class MacroRegimeManager:
    """Offline LLM-powered macro assessment.

    Construction is cheap — no LLM call until :meth:`run_deep` is
    called. The runner calls this class once per cron tick (or once
    per swarm-triggered emergency).
    """

    def __init__(
        self,
        llm_provider: "LLMProvider",
        *,
        output_path: Path | str = DEFAULT_OUTPUT_PATH,
        clock=None,
    ) -> None:
        self._llm = llm_provider
        self._output_path = Path(output_path)
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def run_deep(
        self,
        snapshot: MacroSnapshot,
        *,
        urgency: str = "normal",
        triggering_symbols: list[str] | None = None,
        reasons: list[str] | None = None,
        dry_run: bool = False,
    ) -> MacroRegime:
        """One full deep / emergency cycle.

        Returns a :class:`MacroRegime` even on failure — callers
        should check ``regime.error`` rather than catching.
        """
        regime = MacroRegime(output_path=str(self._output_path))

        user_prompt = build_assessment_prompt(
            snapshot,
            urgency=urgency,
            triggering_symbols=triggering_symbols,
            reasons=reasons,
        )

        try:
            response = await self._llm.generate_text(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                agent_name="macro_regime_manager",
                max_tokens=_LLM_MAX_TOKENS,
                temperature=_LLM_TEMPERATURE,
                cache_system_prompt=True,
            )
        except Exception as e:
            logger.exception("MacroRegimeManager: LLM call failed")
            regime.error = f"llm_call_failed: {e}"
            return self._finalise(regime, snapshot, dry_run=dry_run)

        try:
            payload = self._parse_llm_json(response.content)
        except ValueError as e:
            logger.warning(f"MacroRegimeManager: LLM JSON parse failed: {e}")
            regime.error = f"llm_parse_failed: {e}"
            return self._finalise(regime, snapshot, dry_run=dry_run)

        try:
            regime = self._payload_to_regime(payload, regime)
        except (ValueError, TypeError) as e:
            logger.warning(f"MacroRegimeManager: payload validation failed: {e}")
            regime.error = f"payload_invalid: {e}"
            return self._finalise(regime, snapshot, dry_run=dry_run)

        return self._finalise(regime, snapshot, dry_run=dry_run)

    # ------------------------------------------------------------------
    # Finalisation: blackout windows + timestamps + write
    # ------------------------------------------------------------------

    def _finalise(
        self,
        regime: MacroRegime,
        snapshot: MacroSnapshot,
        *,
        dry_run: bool,
    ) -> MacroRegime:
        """Stamp timestamps, build blackout windows, write file."""
        now = self._clock()
        regime.generated_at = _iso(now)
        regime.expires = _iso(now + timedelta(hours=REGIME_VALIDITY_HOURS))
        regime.blackout_windows = build_blackout_windows(
            snapshot.economic_calendar, reference=now
        )
        if regime.error is None and not dry_run:
            self._write(regime)
        return regime

    def _write(self, regime: MacroRegime) -> None:
        """Atomically write the §13.2.3 JSON. Refuses any other path."""
        try:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._output_path.with_suffix(self._output_path.suffix + ".tmp")
            tmp.write_text(json.dumps(regime.to_disk_dict(), indent=2, sort_keys=True))
            tmp.replace(self._output_path)
            logger.info(
                f"MacroRegimeManager: wrote {regime.regime} "
                f"(confidence={regime.confidence:.2f}, "
                f"{len(regime.blackout_windows)} blackout window(s)) "
                f"to {self._output_path}"
            )
        except OSError as e:
            logger.exception("MacroRegimeManager: write failed")
            regime.error = f"write_failed: {e}"

    # ------------------------------------------------------------------
    # LLM response parsing
    # ------------------------------------------------------------------

    _JSON_BLOCK_RE = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL)
    _BARE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

    @classmethod
    def _parse_llm_json(cls, content: str) -> dict:
        if not content:
            raise ValueError("empty LLM response")
        # Tolerate the LLM wrapping the response in ```json ... ```.
        match = cls._JSON_BLOCK_RE.search(content)
        candidate = match.group(1).strip() if match else None
        if candidate is None:
            # Or wrapping in prose — pull the first {...} block.
            match2 = cls._BARE_JSON_RE.search(content)
            candidate = match2.group(0) if match2 else content.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ValueError(f"not valid JSON: {e}") from e

    @staticmethod
    def _payload_to_regime(payload: dict, base: MacroRegime) -> MacroRegime:
        regime_str = str(payload.get("regime", "")).upper().strip()
        if regime_str not in VALID_REGIMES:
            raise ValueError(
                f"regime must be one of {sorted(VALID_REGIMES)}, got {regime_str!r}"
            )
        try:
            confidence = float(payload.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError) as e:
            raise ValueError(f"confidence must be a number: {e}") from e
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence out of range [0,1]: {confidence}")

        adjustments = MacroAdjustments.from_dict(payload.get("adjustments"))
        # Clamp adjustments to sensible ranges so a hallucinated 5x
        # multiplier can't blow up the live system.
        adjustments.position_size_multiplier = _clamp(
            adjustments.position_size_multiplier, 0.1, 2.0
        )
        adjustments.conviction_threshold_boost = _clamp(
            adjustments.conviction_threshold_boost, 0.0, 0.5
        )

        base.regime = regime_str
        base.confidence = confidence
        base.reasoning = str(payload.get("reasoning", "")).strip()
        base.adjustments = adjustments
        return base


# ---------------------------------------------------------------------------
# Blackout window construction
# ---------------------------------------------------------------------------


def build_blackout_windows(
    events: list[EconomicEvent],
    *,
    reference: datetime,
    pre_buffer_minutes: int = BLACKOUT_PRE_BUFFER_MINUTES,
    post_buffer_minutes: int = BLACKOUT_POST_BUFFER_MINUTES,
    lookahead_hours: float = BLACKOUT_LOOKAHEAD_HOURS,
) -> list[BlackoutWindow]:
    """Deterministically derive blackout windows from the calendar.

    Per §13.2.4: only HIGH-impact events get a window. The window is
    ``event_time - pre_buffer`` to ``event_time + post_buffer``.
    Events outside the lookahead horizon are ignored.
    """
    windows: list[BlackoutWindow] = []
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    horizon = reference + timedelta(hours=lookahead_hours)

    for ev in events:
        if ev.impact != "HIGH":
            continue
        ts = ev.parsed_time()
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        # Allow events that just passed (post-event buffer still applies).
        if ts < reference - timedelta(minutes=post_buffer_minutes):
            continue
        if ts > horizon:
            continue
        start = ts - timedelta(minutes=pre_buffer_minutes)
        end = ts + timedelta(minutes=post_buffer_minutes)
        windows.append(
            BlackoutWindow(
                start=_iso(start),
                end=_iso(end),
                reason=ev.name,
                action="execution_block",
            )
        )
    return windows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _fmt_value(label: str, value: Any, *, suffix: str = "") -> str:
    if value is None:
        return f"  - {label}: (unavailable)"
    if isinstance(value, float):
        return f"  - {label}: {value:.4f}{suffix}"
    return f"  - {label}: {value}{suffix}"


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def load_macro_regime(path: Path | str) -> MacroRegime | None:
    """Load a previously-written ``macro_regime.json`` from disk.

    Used by ConvictionAgent / Sentinel in Task 6. Returns ``None``
    if the file doesn't exist or is malformed — callers must
    handle the missing-file case gracefully.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"load_macro_regime: read failed: {e}")
        return None
    try:
        return MacroRegime(
            regime=str(payload.get("regime", "NEUTRAL")),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            reasoning=str(payload.get("reasoning", "")),
            adjustments=MacroAdjustments.from_dict(payload.get("adjustments")),
            blackout_windows=[
                BlackoutWindow.from_dict(b)
                for b in (payload.get("blackout_windows") or [])
            ],
            generated_at=str(payload.get("generated_at", "")),
            expires=str(payload.get("expires", "")),
            output_path=str(p),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"load_macro_regime: parse failed: {e}")
        return None
