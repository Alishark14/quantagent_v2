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
    """One model decision on one scenario.

    ``position_size_pct`` is retained as an optional field for back-compat
    with historical eval recordings, but live runs now leave it ``None``:
    Sprint Portfolio-Risk-Manager Task 1 stripped DecisionAgent of dollar
    sizing — sizing is owned by ``PortfolioRiskManager`` downstream and
    isn't part of what the eval framework grades. ``risk_weight`` carries
    DecisionAgent's deterministic conviction-band weight (0.75 / 1.0 / 1.15
    / 1.3) so eval reports can still see the sizing INTENT without dollars.
    """

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

    # DecisionAgent's conviction-band weight (0.75 / 1.0 / 1.15 / 1.3),
    # None for SKIP / HOLD / CLOSE_ALL, and None on mock-mode runs that
    # don't go through the real DecisionAgent.
    risk_weight: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "EvalOutput":
        # Drop unknown keys defensively — old recordings shouldn't break new code.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})
