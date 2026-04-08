"""Standardised eval output contract.

Every model under evaluation — Claude pipeline, fine-tuned 7B,
ONNX HFT model — produces an ``EvalOutput``. This contract is what
makes the framework model-agnostic. Distillation comparisons just
diff two ``EvalOutput`` instances on identical inputs.

See ARCHITECTURE.md §31.4.1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class EvalOutput:
    """One model decision on one scenario."""

    direction: str  # "LONG" | "SHORT" | "SKIP"
    conviction: float  # 0.0 to 1.0
    sl_price: float | None
    tp1_price: float | None
    tp2_price: float | None
    position_size_pct: float | None
    reasoning: str | None  # null for distilled models that don't reason in text
    latency_ms: float
    model_id: str

    # Populated by the eval framework after the run completes
    teacher_agreement: float | None = None
    conviction_calibration: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "EvalOutput":
        # Drop unknown keys defensively — old recordings shouldn't break new code.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})
