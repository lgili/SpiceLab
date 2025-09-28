"""Public analysis helpers for the unified API."""

from __future__ import annotations

from .measure import (
    ENOBSpec,
    GainBandwidthSpec,
    GainMarginSpec,
    GainSpec,
    OvershootSpec,
    PhaseMarginSpec,
    RiseTimeSpec,
    SettlingTimeSpec,
    THDSpec,
    measure,
)
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
from .pipeline import measure_job_result, run_and_measure
from .result import AnalysisResult
from .signal import FFTResult, amplitude_spectrum, power_spectral_density, rfft_coherent, window
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
    "PhaseMarginSpec",
    "GainBandwidthSpec",
    "GainMarginSpec",
    "OvershootSpec",
    "RiseTimeSpec",
    "SettlingTimeSpec",
    "THDSpec",
    "ENOBSpec",
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
    "window",
    "rfft_coherent",
    "amplitude_spectrum",
    "power_spectral_density",
    "FFTResult",
    "measure_job_result",
    "run_and_measure",
]
