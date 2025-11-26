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
from .stats import (
    Statistics,
    compute_stats,
    create_metric_extractor,
    extract_from_analysis,
    extract_trace_value,
    mc_summary,
)
from .sweep_grid import (
    GridResult,
    GridRun,
    SweepResult,
    SweepRun,
    run_param_grid,
    run_value_sweep,
)
from .wca import (
    WcaCorner,
    WcaResult,
    run_wca,
    tolerance_to_normal,
    tolerance_to_uniform,
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
    # WCA
    "WcaCorner",
    "WcaResult",
    "run_wca",
    "tolerance_to_normal",
    "tolerance_to_uniform",
    # Statistics
    "Statistics",
    "compute_stats",
    "extract_trace_value",
    "extract_from_analysis",
    "mc_summary",
    "create_metric_extractor",
]
