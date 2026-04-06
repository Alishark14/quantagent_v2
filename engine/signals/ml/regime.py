"""RegimeModel: ML regime classification slot."""

from __future__ import annotations

from engine.config import FeatureFlags
from engine.signals.ml import MLModelSlot


class RegimeModel(MLModelSlot):
    """Classifies market regime from market data. Inactive until trained."""

    def __init__(self, flags: FeatureFlags | None = None) -> None:
        super().__init__(
            slot_name="regime_model",
            feature_flag="ml_regime_model",
            flags=flags,
        )
