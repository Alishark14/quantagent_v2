"""Backtesting framework — historical data, replay, simulation, evals.

See ARCHITECTURE.md §31.3 for the 4-tier backtesting design.
"""

from backtesting.data_downloader import HistoricalDataDownloader
from backtesting.data_loader import ParquetDataLoader
from backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult
from backtesting.forward_path import ForwardPathLoader
from backtesting.metrics import BacktestMetrics, calculate_metrics
from backtesting.mock_signals import MockSignalProducer
from backtesting.reporter import generate_html_report, generate_json_report
from backtesting.sim_exchange import AssetMeta, SimulatedExchangeAdapter
from backtesting.sim_executor import SimExecutor
from backtesting.tier2_replay import (
    ReplayResult,
    SweepResult,
    SweepRow,
    Tier2ReplayEngine,
)

# Eval framework — re-export the public surface so callers can do
# `from backtesting import EvalRunner` if they want a flat namespace.
from backtesting.evals import (
    AutoMiner,
    EvalOutput,
    EvalReport,
    EvalRunner,
    JudgeScore,
    Scenario,
    generate_eval_report,
    judge_output,
)

__all__ = [
    "HistoricalDataDownloader",
    "ParquetDataLoader",
    "SimulatedExchangeAdapter",
    "SimExecutor",
    "AssetMeta",
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "BacktestMetrics",
    "calculate_metrics",
    "generate_json_report",
    "generate_html_report",
    "MockSignalProducer",
    "ForwardPathLoader",
    "Tier2ReplayEngine",
    "ReplayResult",
    "SweepResult",
    "SweepRow",
    # Eval framework
    "AutoMiner",
    "EvalOutput",
    "EvalReport",
    "EvalRunner",
    "JudgeScore",
    "Scenario",
    "generate_eval_report",
    "judge_output",
]
