"""MockSignalProducer — deterministic SignalProducer for backtests.

Replaces real LLM agents during Tier 1 mechanical backtests so the engine
can run candle-by-candle in seconds for $0. Returns configurable
``SignalOutput`` objects driven by one of several modes:

- ``always_long``  — every call returns BULLISH
- ``always_short`` — every call returns BEARISH
- ``always_skip``  — every call returns NEUTRAL
- ``random_seed:N`` — seeded RNG produces reproducible LONG/SHORT/SKIP
- ``from_file:PATH`` — replay a JSON file of pre-recorded directional decisions

The "from_file" mode is intended for two scenarios:
1. Replaying real agent outputs captured from a live run for regression
   testing of mechanical changes (risk profiles, safety checks).
2. Property-based / curated scenario tests where the desired sequence of
   directions is hand-authored.

JSON file format::

    [
        {"timestamp": 1700000000000, "direction": "BULLISH", "confidence": 0.8},
        {"timestamp": 1700003600000, "direction": "NEUTRAL", "confidence": 0.5},
        ...
    ]

Records are matched against the ``MarketData.candles[-1]["timestamp"]`` of
the analysis call; if no record matches the current timestamp the
producer returns NEUTRAL (skip).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from engine.signals.base import SignalProducer
from engine.types import MarketData, SignalOutput

logger = logging.getLogger(__name__)


_VALID_MODES = {"always_long", "always_short", "always_skip"}
_DIRECTIONS = ("BULLISH", "BEARISH", "NEUTRAL")


class MockSignalProducer(SignalProducer):
    """Deterministic signal producer for backtests."""

    def __init__(
        self,
        mode: str = "always_skip",
        confidence: float = 0.7,
        name: str = "mock_signal",
    ) -> None:
        """
        Args:
            mode: One of ``always_long`` / ``always_short`` / ``always_skip`` /
                ``random_seed:N`` / ``from_file:PATH``.
            confidence: Confidence value to attach to every emitted signal.
            name: Producer identifier (shown in SignalOutput.agent_name).
        """
        self._mode_raw = mode
        self._confidence = float(confidence)
        self._name = name
        self._enabled = True

        # Parsed mode state
        self._mode_kind: str = ""
        self._rng: random.Random | None = None
        self._file_records: dict[int, dict] = {}

        self._parse_mode(mode)

    # ------------------------------------------------------------------
    # SignalProducer ABC
    # ------------------------------------------------------------------

    def name(self) -> str:
        return self._name

    def signal_type(self) -> str:
        return "ml"  # ML-style fast deterministic producer (zero LLM cost)

    def is_enabled(self) -> bool:
        return self._enabled

    async def analyze(self, data: MarketData) -> SignalOutput | None:
        direction = self._next_direction(data)
        return SignalOutput(
            agent_name=self._name,
            signal_type="ml",
            direction=direction,
            confidence=self._confidence,
            reasoning=f"MockSignalProducer mode={self._mode_raw}",
            signal_category="directional",
            data_richness="full",
            contradictions="",
            key_levels={},
            pattern_detected=None,
            raw_output="",
        )

    # ------------------------------------------------------------------
    # Convenience for non-async callers (BacktestEngine inner loop)
    # ------------------------------------------------------------------

    def next_direction(self, data: MarketData) -> str:
        """Synchronous direction lookup. Used by the BacktestEngine inner loop
        which doesn't need the full SignalOutput envelope.
        """
        return self._next_direction(data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_mode(self, mode: str) -> None:
        if mode in _VALID_MODES:
            self._mode_kind = mode
            return
        if mode.startswith("random_seed:"):
            seed_str = mode.split(":", 1)[1]
            try:
                seed = int(seed_str)
            except ValueError as e:
                raise ValueError(
                    f"random_seed mode requires int seed, got {seed_str!r}"
                ) from e
            self._mode_kind = "random_seed"
            self._rng = random.Random(seed)
            return
        if mode.startswith("from_file:"):
            path_str = mode.split(":", 1)[1]
            path = Path(path_str)
            if not path.exists():
                raise FileNotFoundError(
                    f"MockSignalProducer from_file path does not exist: {path}"
                )
            self._mode_kind = "from_file"
            self._load_file(path)
            return
        raise ValueError(
            f"Unknown MockSignalProducer mode {mode!r}. Valid: "
            f"{sorted(_VALID_MODES)} or random_seed:N or from_file:PATH"
        )

    def _load_file(self, path: Path) -> None:
        with path.open() as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(
                f"from_file expects a JSON list, got {type(records).__name__}"
            )
        indexed: dict[int, dict] = {}
        for rec in records:
            if "timestamp" not in rec or "direction" not in rec:
                raise ValueError(
                    f"from_file record missing 'timestamp' or 'direction': {rec}"
                )
            if rec["direction"] not in _DIRECTIONS:
                raise ValueError(
                    f"from_file direction must be one of {_DIRECTIONS}, "
                    f"got {rec['direction']!r}"
                )
            indexed[int(rec["timestamp"])] = rec
        self._file_records = indexed

    def _next_direction(self, data: MarketData) -> str:
        if self._mode_kind == "always_long":
            return "BULLISH"
        if self._mode_kind == "always_short":
            return "BEARISH"
        if self._mode_kind == "always_skip":
            return "NEUTRAL"
        if self._mode_kind == "random_seed":
            assert self._rng is not None
            return self._rng.choice(_DIRECTIONS)
        if self._mode_kind == "from_file":
            ts = self._current_timestamp(data)
            rec = self._file_records.get(ts)
            return rec["direction"] if rec else "NEUTRAL"
        # Should be unreachable due to _parse_mode validation
        return "NEUTRAL"  # pragma: no cover

    @staticmethod
    def _current_timestamp(data: MarketData) -> int:
        if not data.candles:
            return 0
        return int(data.candles[-1].get("timestamp", 0))
