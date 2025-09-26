from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..core.components import Component
from ..core.types import AnalysisSpec, ResultHandle, SweepSpec
from ..engines import EngineName, run_simulation
from ..orchestrator import JobResult


@dataclass(frozen=True)
class SweepRun:
    value: str | float
    handle: ResultHandle


@dataclass(frozen=True)
class SweepResult:
    component_ref: str
    values: list[str | float]
    runs: list[SweepRun]

    def handles(self) -> list[ResultHandle]:
        return [r.handle for r in self.runs]


def run_value_sweep(
    circuit: object,
    component: Component,
    values: Sequence[str | float],
    analyses: Sequence[AnalysisSpec],
    *,
    engine: EngineName = "ngspice",
    progress: bool | Callable[[int, int], None] | None = None,
    cache_dir: str | Path | None = None,
    workers: int = 1,
    reuse_cache: bool = True,
) -> SweepResult:
    """Run multiple simulations varying a single component value.

    - Mutates component.value for each run; restores the original value at the end.
    - Uses the unified engine API (get_simulator().run(...)).
    - Returns lightweight handles; you can pull xarray datasets from each when needed.
    """

    sweep_spec = SweepSpec(variables={str(component.ref): list(values)})
    result = run_simulation(
        circuit,
        analyses,
        sweep=sweep_spec,
        engine=engine,
        progress=progress,
        cache_dir=cache_dir,
        workers=workers,
        reuse_cache=reuse_cache,
    )
    if not isinstance(result, JobResult):
        raise RuntimeError("Expected JobResult from run_simulation when sweep is provided")
    runs: list[SweepRun] = []
    for job_run in result.runs:
        value = job_run.combo.get(str(component.ref))
        runs.append(SweepRun(value=value if value is not None else "", handle=job_run.handle))
    return SweepResult(component_ref=str(component.ref), values=list(values), runs=runs)


__all__ = ["run_value_sweep", "SweepResult", "SweepRun"]


# ------------------------------ grid (multi-parameter) ------------------------------


@dataclass(frozen=True)
class GridRun:
    combo: Mapping[str, str | float]
    handle: ResultHandle


@dataclass(frozen=True)
class GridResult:
    runs: list[GridRun]

    def handles(self) -> list[ResultHandle]:
        return [r.handle for r in self.runs]


def run_param_grid(
    circuit: object,
    variables: Sequence[tuple[Component, Sequence[str | float]]],
    analyses: Sequence[AnalysisSpec],
    *,
    engine: EngineName = "ngspice",
    progress: bool | Callable[[int, int], None] | None = None,
    cache_dir: str | Path | None = None,
    workers: int = 1,
    reuse_cache: bool = True,
) -> GridResult:
    """Run a Cartesian product of component.value assignments.

    variables: sequence of (component, values) pairs.
    """

    # Prepare original values to restore later
    var_map: dict[str, list[str | float]] = {}
    for comp, vals in variables:
        var_map[str(comp.ref)] = list(vals)

    sweep_spec = SweepSpec(variables=var_map)
    job = run_simulation(
        circuit,
        analyses,
        sweep=sweep_spec,
        engine=engine,
        progress=progress,
        cache_dir=cache_dir,
        workers=workers,
        reuse_cache=reuse_cache,
    )
    if not isinstance(job, JobResult):
        raise RuntimeError("Expected JobResult from run_simulation when sweep is provided")
    grid_runs: list[GridRun] = []
    for job_run in job.runs:
        grid_runs.append(GridRun(combo=dict(job_run.combo), handle=job_run.handle))

    return GridResult(runs=grid_runs)


__all__ += ["run_param_grid", "GridResult", "GridRun"]
