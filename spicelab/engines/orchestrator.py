from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from ..core.types import (
    AnalysisSpec,
    Probe,
    ResultHandle,
    SweepSpec,
    ensure_analysis_spec,
    ensure_sweep_spec,
)
from ..orchestrator import Job, JobResult, run_job
from .base import Simulator
from .factory import create_simulator

EngineName = Literal[
    "ngspice",
    "ngspice-cli",
    "ngspice-shared",
    "ltspice",
    "ltspice-cli",
    "xyce",
    "xyce-cli",
]


def get_simulator(name: EngineName = "ngspice") -> Simulator:
    return create_simulator(name)


def run_simulation(
    circuit: object,
    analyses: Sequence[AnalysisSpec],
    sweep: SweepSpec | None = None,
    probes: Sequence[Probe] | None = None,
    *,
    engine: EngineName = "ngspice",
    cache_dir: str | Path | None = None,
    workers: int = 1,
    reuse_cache: bool = True,
    progress: bool | Callable[[int, int], None] | None = None,
) -> ResultHandle | JobResult:
    sim = get_simulator(engine)
    # normalize inputs (allows dict / pydantic during migration)
    analyses = [ensure_analysis_spec(a) for a in analyses]
    sweep = ensure_sweep_spec(sweep)
    if sweep is not None and sweep.variables:
        job = Job(
            circuit=circuit,
            analyses=analyses,
            sweep=sweep,
            probes=list(probes) if probes else None,
            engine=engine,
        )
        return run_job(
            job,
            cache_dir=cache_dir,
            workers=workers,
            progress=progress,
            reuse_cache=reuse_cache,
        )
    probes_list = list(probes) if probes else None
    return sim.run(circuit, analyses, sweep, probes_list)


__all__ = [
    "get_simulator",
    "run_simulation",
    "EngineName",
    "Job",
    "JobResult",
    "run_job",
]
