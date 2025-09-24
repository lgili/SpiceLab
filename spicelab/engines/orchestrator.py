from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ..core.types import (
    AnalysisSpec,
    Probe,
    ResultHandle,
    SweepSpec,
    ensure_analysis_spec,
    ensure_sweep_spec,
)
from .base import Simulator
from .ltspice import LtSpiceSimulator
from .ngspice import NgSpiceSimulator
from .xyce import XyceSimulator

EngineName = Literal["ngspice", "ltspice", "xyce"]


def get_simulator(name: EngineName = "ngspice") -> Simulator:
    if name == "ngspice":
        return NgSpiceSimulator()
    if name == "ltspice":
        return LtSpiceSimulator()
    if name == "xyce":
        return XyceSimulator()
    raise NotImplementedError(f"Engine '{name}' not supported yet")


def run_simulation(
    circuit: object,
    analyses: Sequence[AnalysisSpec],
    sweep: SweepSpec | None = None,
    probes: Sequence[Probe] | None = None,
    *,
    engine: EngineName = "ngspice",
) -> ResultHandle:
    sim = get_simulator(engine)
    # normalize inputs (allows dict / pydantic during migration)
    analyses = [ensure_analysis_spec(a) for a in analyses]
    sweep = ensure_sweep_spec(sweep)
    if sweep is not None and sweep.variables:
        # Next iteration: orchestrate parameter sweeps (produce multi-run handle)
        raise NotImplementedError("SweepSpec orchestration not implemented yet")
    probes_list = list(probes) if probes else None
    return sim.run(circuit, analyses, sweep, probes_list)


__all__ = ["get_simulator", "run_simulation", "EngineName"]
