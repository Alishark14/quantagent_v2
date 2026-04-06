"""DirectionModel: ML direction prediction slot (returns null until trained)."""

from __future__ import annotations

from engine.config import FeatureFlags
from engine.signals.ml import MLModelSlot


class DirectionModel(MLModelSlot):
    """Predicts trade direction from market data. Inactive until trained."""

    def __init__(self, flags: FeatureFlags | None = None) -> None:
        super().__init__(
            slot_name="direction_model",
            feature_flag="ml_direction_model",
            flags=flags,
        )
