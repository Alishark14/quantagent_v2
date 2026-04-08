"""Quant Data Scientist — offline alpha-mining MCP agent.

Public API:

- :class:`QuantDataScientist` — the agent class itself
- :class:`AlphaFactor` / :class:`AlphaFactorsReport` — output dataclasses
- :func:`build_analysis_prompt` — Claude prompt builder
- :func:`apply_decay` / :func:`merge_factors` — confidence-decay helpers

See ARCHITECTURE.md §13.1.
"""

from mcp.quant_scientist.agent import (
    AnalysisCodeError,
    QuantDataScientist,
    QuantScientistError,
)
from mcp.quant_scientist.decay import (
    DECAY_PRUNE_THRESHOLD,
    apply_decay,
    decay_weight_for_age,
    merge_factors,
)
from mcp.quant_scientist.factor import (
    AlphaFactor,
    AlphaFactorsReport,
    factors_to_nested_json,
    nested_json_to_factors,
)
from mcp.quant_scientist.prompts import build_analysis_prompt

__all__ = [
    "AlphaFactor",
    "AlphaFactorsReport",
    "AnalysisCodeError",
    "DECAY_PRUNE_THRESHOLD",
    "QuantDataScientist",
    "QuantScientistError",
    "apply_decay",
    "build_analysis_prompt",
    "decay_weight_for_age",
    "factors_to_nested_json",
    "merge_factors",
    "nested_json_to_factors",
]
