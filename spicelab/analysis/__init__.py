"""Public analysis helpers for the unified API."""

from __future__ import annotations

from .measure import GainSpec, OvershootSpec, SettlingTimeSpec, measure
from .montecarlo import (
    Dist,
    LogNormalPct,
    MonteCarloResult,
    NormalPct,
    TriangularPct,
    UniformAbs,
    UniformPct,
    monte_carlo,
)
from .result import AnalysisResult
from .sweep_grid import (
    GridResult,
    GridRun,
    SweepResult,
    SweepRun,
    run_param_grid,
    run_value_sweep,
)

__all__ = [
    "AnalysisResult",
    "measure",
    "GainSpec",
    "OvershootSpec",
    "SettlingTimeSpec",
    "Dist",
    "NormalPct",
    "UniformPct",
    "UniformAbs",
    "LogNormalPct",
    "TriangularPct",
    "MonteCarloResult",
    "monte_carlo",
    "SweepRun",
    "SweepResult",
    "GridRun",
    "GridResult",
    "run_value_sweep",
    "run_param_grid",
]
