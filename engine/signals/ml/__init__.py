"""ML model slots — return None until a trained model is loaded."""

from __future__ import annotations

from engine.config import FeatureFlags
from engine.signals.base import SignalProducer
from engine.types import MarketData, SignalOutput


class MLModelSlot(SignalProducer):
    """Base class for ML model slots.

    Returns None until a trained model is loaded via load_model().
    Subclasses override _predict() when trained.
    """

    def __init__(self, slot_name: str, feature_flag: str, flags: FeatureFlags | None = None) -> None:
        self._name = slot_name
        self._flag = feature_flag
        self._flags = flags
        self._model: object | None = None

    def name(self) -> str:
        return self._name

    def signal_type(self) -> str:
        return "ml"

    def is_enabled(self) -> bool:
        if self._model is None:
            return False
        if self._flags is not None:
            return self._flags.is_enabled(self._flag)
        return False

    async def analyze(self, data: MarketData) -> SignalOutput | None:
        if not self.is_enabled():
            return None
        return self._predict(data)

    def _predict(self, data: MarketData) -> SignalOutput:
        raise NotImplementedError("Model not trained yet")

    def load_model(self, model_path: str) -> None:
        """Load a trained model from disk."""
        raise NotImplementedError
