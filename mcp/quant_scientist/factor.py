"""AlphaFactor + AlphaFactorsReport dataclasses for the Quant Data Scientist.

The on-disk format (`alpha_factors.json`) is a nested dict, keyed by
``symbol → timeframe → pattern_combo → fields``, per ARCHITECTURE.md
§13.1.2. The agent and decay layer prefer to operate on a flat list
of :class:`AlphaFactor` records — easier to filter, dedupe, and merge.
This module is the bridge: pure conversion functions between the flat
list and the nested JSON shape.

Schema (from §13.1.2):

```json
{
  "BTC-USDC": {
    "1h": {
      "ascending_triangle + RSI_below_50 + MACD_bullish": {
        "win_rate": 0.68,
        "avg_R": 1.9,
        "n": 23,
        "confidence": "high",
        "discovered_at": "2026-04-05T02:30:00Z",
        "last_confirmed": "2026-04-05T02:30:00Z",
        "decay_weight": 1.0,
        "note": "AVOID"  // optional, only on negative factors
      }
    }
  }
}
```
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

_VALID_CONFIDENCE = {"high", "medium", "low"}


@dataclass(frozen=True)
class AlphaFactor:
    """One discovered (or persisted) alpha factor.

    A factor is the combination of (symbol, timeframe, pattern) plus
    its measured statistics. Two factors with the same key triple are
    considered the same factor for merge / decay purposes.
    """

    pattern: str
    symbol: str
    timeframe: str
    win_rate: float
    avg_r: float
    n: int
    confidence: str  # "high" | "medium" | "low"
    discovered_at: str  # ISO 8601 UTC
    last_confirmed: str  # ISO 8601 UTC
    decay_weight: float
    note: str | None = None  # e.g. "AVOID" for negative-edge factors

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def key(self) -> tuple[str, str, str]:
        """The (symbol, timeframe, pattern) tuple used for dedupe / merge."""
        return (self.symbol, self.timeframe, self.pattern)

    def to_dict(self) -> dict:
        return asdict(self)

    def with_updates(self, **changes) -> "AlphaFactor":
        """Return a copy with the given fields overridden.

        Frozen dataclasses don't allow in-place mutation; this is the
        substitute the decay layer uses to bump ``last_confirmed`` and
        reset ``decay_weight`` on a re-confirmed factor.
        """
        current = asdict(self)
        current.update(changes)
        return AlphaFactor(**current)

    # ------------------------------------------------------------------
    # Validation (called from from_payload + agent parsing)
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ValueError if any field violates schema invariants."""
        if not self.pattern:
            raise ValueError("AlphaFactor.pattern must be non-empty")
        if not self.symbol:
            raise ValueError("AlphaFactor.symbol must be non-empty")
        if not self.timeframe:
            raise ValueError("AlphaFactor.timeframe must be non-empty")
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError(
                f"AlphaFactor.win_rate must be in [0,1], got {self.win_rate}"
            )
        if not isinstance(self.n, int) or self.n < 0:
            raise ValueError(f"AlphaFactor.n must be a non-negative int, got {self.n!r}")
        if self.confidence not in _VALID_CONFIDENCE:
            raise ValueError(
                f"AlphaFactor.confidence must be one of {sorted(_VALID_CONFIDENCE)}, "
                f"got {self.confidence!r}"
            )
        if not 0.0 <= self.decay_weight <= 1.0:
            raise ValueError(
                f"AlphaFactor.decay_weight must be in [0,1], got {self.decay_weight}"
            )
        if not self.discovered_at or not self.last_confirmed:
            raise ValueError("AlphaFactor.discovered_at / last_confirmed must be set")

    @classmethod
    def from_payload(cls, symbol: str, timeframe: str, pattern: str, payload: dict) -> "AlphaFactor":
        """Build an AlphaFactor from one nested-JSON leaf.

        ``payload`` is the inner dict ({"win_rate": ..., "avg_R": ...}).
        Tolerates the JSON's ``avg_R`` casing AND the dataclass's
        ``avg_r`` casing — the agent's analysis code can use either.
        """
        avg_r = payload.get("avg_r")
        if avg_r is None:
            avg_r = payload.get("avg_R")
        if avg_r is None:
            raise ValueError(
                f"AlphaFactor payload missing avg_r/avg_R: {payload}"
            )

        factor = cls(
            pattern=pattern,
            symbol=symbol,
            timeframe=timeframe,
            win_rate=float(payload["win_rate"]),
            avg_r=float(avg_r),
            n=int(payload["n"]),
            confidence=str(payload.get("confidence", "low")).lower(),
            discovered_at=str(payload["discovered_at"]),
            last_confirmed=str(payload["last_confirmed"]),
            decay_weight=float(payload.get("decay_weight", 1.0)),
            note=payload.get("note"),
        )
        factor.validate()
        return factor


@dataclass
class AlphaFactorsReport:
    """Result of one Quant Data Scientist run.

    Returned by :meth:`QuantDataScientist.run` and used by the runner
    CLI to build its summary line. The factor lists are *flat* — the
    nested JSON shape is only built at write time via
    :func:`factors_to_nested_json`.
    """

    factors: list[AlphaFactor] = field(default_factory=list)
    new_count: int = 0
    confirmed_count: int = 0
    pruned_count: int = 0
    trades_analyzed: int = 0
    symbols_analyzed: int = 0
    output_path: str | None = None
    dry_run: bool = False
    error: str | None = None

    @property
    def factor_count(self) -> int:
        return len(self.factors)

    def to_dict(self) -> dict:
        return {
            "factor_count": self.factor_count,
            "new_count": self.new_count,
            "confirmed_count": self.confirmed_count,
            "pruned_count": self.pruned_count,
            "trades_analyzed": self.trades_analyzed,
            "symbols_analyzed": self.symbols_analyzed,
            "output_path": self.output_path,
            "dry_run": self.dry_run,
            "error": self.error,
            "factors": [f.to_dict() for f in self.factors],
        }


# ---------------------------------------------------------------------------
# Nested JSON ⇄ flat list conversion
# ---------------------------------------------------------------------------


def factors_to_nested_json(factors: Iterable[AlphaFactor]) -> dict:
    """Convert a flat list of AlphaFactor → the §13.1.2 nested shape.

    Output uses ``avg_R`` (capital R) per the spec example. Optional
    ``note`` field is only emitted when present.
    """
    out: dict = {}
    for f in factors:
        symbol_block = out.setdefault(f.symbol, {})
        tf_block = symbol_block.setdefault(f.timeframe, {})
        leaf: dict = {
            "win_rate": round(f.win_rate, 4),
            "avg_R": round(f.avg_r, 4),
            "n": f.n,
            "confidence": f.confidence,
            "discovered_at": f.discovered_at,
            "last_confirmed": f.last_confirmed,
            "decay_weight": round(f.decay_weight, 4),
        }
        if f.note:
            leaf["note"] = f.note
        tf_block[f.pattern] = leaf
    return out


def nested_json_to_factors(nested: dict) -> list[AlphaFactor]:
    """Inverse of :func:`factors_to_nested_json`.

    Used by ``apply_decay`` to load the previous run's
    ``alpha_factors.json`` from disk so it can decay them and merge
    them with newly discovered factors.
    """
    out: list[AlphaFactor] = []
    for symbol, by_timeframe in (nested or {}).items():
        if not isinstance(by_timeframe, dict):
            continue
        for timeframe, by_pattern in by_timeframe.items():
            if not isinstance(by_pattern, dict):
                continue
            for pattern, payload in by_pattern.items():
                if not isinstance(payload, dict):
                    continue
                try:
                    out.append(
                        AlphaFactor.from_payload(symbol, timeframe, pattern, payload)
                    )
                except (KeyError, ValueError, TypeError):
                    # Skip malformed entries quietly — never crash on a
                    # corrupt previous-run file. The bad entry just
                    # gets dropped and replaced on the next run.
                    continue
    return out
