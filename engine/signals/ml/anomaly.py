"""AnomalyDetector: ML anomaly detection slot."""

from __future__ import annotations

from engine.config import FeatureFlags
from engine.signals.ml import MLModelSlot


class AnomalyDetector(MLModelSlot):
    """Detects anomalous market conditions. Inactive until trained."""

    def __init__(self, flags: FeatureFlags | None = None) -> None:
        super().__init__(
            slot_name="anomaly_detector",
            feature_flag="ml_anomaly_detector",
            flags=flags,
        )
