from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .core import AC, DC, OP, TRAN, AnalysisResult
from .metrics_ac import (
    Bode,
    ac_gain_phase,
    bandwidth_3db,
    crossover_freq_0db,
    gain_at,
    gain_margin_db,
    loop_gain_bode,
    phase_crossover_freq,
    phase_margin,
)
from .metrics_basic import (
    gain_db_from_traces,
    overshoot_pct,
    peak,
)
from .metrics_tran import (
    OvershootResult,
    RiseFall,
    SettlingResult,
    fall_time,
    overshoot,
    rise_time,
    settling_time,
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
from .post import stack_runs_to_df, stack_step_to_df
from .step import ParamGrid, StepResult, step_grid, step_param
from .step_native import StepNativeResult, run_step_native
from .sweep import SweepResult, sweep_component
from .worstcase import WorstCaseResult, worst_case

if TYPE_CHECKING:  # pragma: no cover
    from ..core.circuit import Circuit

__all__ = [
    "OP",
    "TRAN",
    "AC",
    "DC",
    "AnalysisResult",
    "SweepResult",
    "sweep_component",
    "ParamGrid",
    "StepResult",
    "step_param",
    "step_grid",
    "stack_step_to_df",
    "stack_runs_to_df",
    "peak",
    "settling_time",
    "overshoot_pct",
    "gain_db_from_traces",
    "bandwidth_3db",
    "ac_gain_phase",
    "crossover_freq_0db",
    "phase_margin",
    "phase_crossover_freq",
    "gain_margin_db",
    # Monte Carlo
    "Dist",
    "NormalPct",
    "UniformPct",
    "MonteCarloResult",
    "monte_carlo",
    "WorstCaseResult",
    "worst_case",
    "StepNativeResult",
    "run_step_native",
    "LogNormalPct",
    "TriangularPct",
    "UniformAbs",
    # metrics_ac
    "Bode",
    "gain_at",
    "loop_gain_bode",
    # metrics_tran
    "OvershootResult",
    "RiseFall",
    "SettlingResult",
    "overshoot",
    "rise_time",
    "fall_time",
]


# -------- Convenience, high-level helpers --------
def run_op(circuit: Circuit) -> AnalysisResult:
    """Run a simple .OP analysis and return the AnalysisResult.

    Example:
        >>> from spicelab.core.circuit import Circuit
        >>> from spicelab.core.components import Vdc, Resistor
        >>> from spicelab.core.net import GND
        >>> c = Circuit("rc")
        >>> V1, R1 = Vdc("1", 5.0), Resistor("1", "1k")
        >>> c.add(V1, R1)
        >>> c.connect(V1.ports[0], R1.ports[0])
        >>> c.connect(R1.ports[1], GND)
        >>> c.connect(V1.ports[1], GND)
        >>> _ = run_op(c)  # doctest: +SKIP
    """
    return OP().run(circuit)


def run_tran(
    circuit: Circuit,
    tstep: str,
    tstop: str,
    tstart: str | None = None,
    *,
    return_df: bool = False,
) -> AnalysisResult | Any:
    """Run a transient (.TRAN) analysis and optionally return a Pandas DataFrame.

    - return_df=False returns AnalysisResult
    - return_df=True returns a DataFrame via TraceSet.to_dataframe()
    """
    res = TRAN(tstep, tstop, tstart).run(circuit)
    if return_df:
        return res.traces.to_dataframe()
    return res


def run_ac(
    circuit: Circuit,
    sweep_type: str,
    n: int,
    fstart: float,
    fstop: float,
    *,
    return_df: bool = False,
) -> AnalysisResult | Any:
    """Run an AC analysis and optionally return a DataFrame of traces."""
    res = AC(sweep_type, n, fstart, fstop).run(circuit)
    if return_df:
        return res.traces.to_dataframe()
    return res


def bode(
    circuit: Circuit,
    y_out: str,
    y_in: str | None = None,
    *,
    sweep_type: str = "dec",
    n: int = 201,
    fstart: float = 10.0,
    fstop: float = 1e6,
) -> tuple[Any, Any, Any]:
    """Run AC and return (f, |G|_dB, phase_deg) using spicelab.analysis.metrics_ac.

    Note: The circuit must include appropriate small-signal sources for AC analysis.
    """
    res = AC(sweep_type, n, fstart, fstop).run(circuit)
    return ac_gain_phase(res.traces, y_out=y_out, y_in=y_in)


__all__ += [
    "run_op",
    "run_tran",
    "run_ac",
    "bode",
]
