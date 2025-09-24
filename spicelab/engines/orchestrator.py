from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ..core.types import AnalysisSpec, ResultHandle, SweepSpec
from .base import Simulator
from .ngspice import NgSpiceSimulator

EngineName = Literal["ngspice"]


def get_simulator(name: EngineName = "ngspice") -> Simulator:
    if name == "ngspice":
        return NgSpiceSimulator()
    raise NotImplementedError(f"Engine '{name}' not supported yet")


def run_simulation(
    circuit: object,
    analyses: Sequence[AnalysisSpec],
    sweep: SweepSpec | None = None,
    *,
    engine: EngineName = "ngspice",
) -> ResultHandle:
    sim = get_simulator(engine)
    if sweep is not None and sweep.variables:
        # Next iteration: orchestrate parameter sweeps (produce multi-run handle)
        raise NotImplementedError("SweepSpec orchestration not implemented yet")
    return sim.run(circuit, analyses, sweep)


__all__ = ["get_simulator", "run_simulation", "EngineName"]
