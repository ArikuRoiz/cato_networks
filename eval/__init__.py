"""Replay harness, return vs SPY metrics, and process metrics reporting."""

from eval.metrics import EvalMetrics, compute_metrics
from eval.replay import CycleRecord, EvalConfig, EvalResult, run_eval
from eval.report import generate_report

__all__ = [
    "CycleRecord",
    "EvalConfig",
    "EvalMetrics",
    "EvalResult",
    "compute_metrics",
    "generate_report",
    "run_eval",
]
