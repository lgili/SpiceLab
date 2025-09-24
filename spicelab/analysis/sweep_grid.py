from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from ..core.components import Component
from ..core.types import AnalysisSpec, ResultHandle
from ..engines import EngineName, get_simulator


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
) -> SweepResult:
    """Run multiple simulations varying a single component value.

    - Mutates component.value for each run; restores the original value at the end.
    - Uses the unified engine API (get_simulator().run(...)).
    - Returns lightweight handles; you can pull xarray datasets from each when needed.
    """

    original = component.value

    def _notify(done: int) -> None:
        if not progress:
            return
        if callable(progress):
            try:
                progress(done, len(values))
            except Exception:
                pass
            return
        # simple textual progress to stderr
        try:
            import sys

            pct = int(round(100.0 * done / max(len(values), 1)))
            sys.stderr.write(f"\rSWEEP[{component.ref}]: {done}/{len(values)} ({pct}%)")
            sys.stderr.flush()
        except Exception:
            pass

    sim = get_simulator(engine)
    runs: list[SweepRun] = []
    try:
        for i, v in enumerate(values, start=1):
            component.value = v
            handle = sim.run(circuit, analyses, None)
            runs.append(SweepRun(value=v, handle=handle))
            _notify(i)
    finally:
        component.value = original

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
) -> GridResult:
    """Run a Cartesian product of component.value assignments.

    variables: sequence of (component, values) pairs.
    """

    # Prepare original values to restore later
    originals: dict[str, str | float] = {comp.ref: comp.value for comp, _ in variables}

    def _notify(done: int, total: int) -> None:
        if not progress:
            return
        if callable(progress):
            try:
                progress(done, total)
            except Exception:
                pass
            return
        try:
            import sys

            pct = int(round(100.0 * done / max(total, 1)))
            sys.stderr.write(f"\rGRID: {done}/{total} ({pct}%)")
            sys.stderr.flush()
        except Exception:
            pass

    sim = get_simulator(engine)
    runs: list[GridRun] = []

    # Build product of values and iterate
    value_lists = [vals for _, vals in variables]
    total = 1
    for vals in value_lists:
        total *= max(len(vals), 1)

    try:
        done = 0
        for combo_vals in product(*value_lists):
            # Assign
            combo_map: dict[str, str | float] = {}
            for (comp, _), value in zip(variables, combo_vals, strict=False):
                comp.value = value
                combo_map[comp.ref] = value
            handle = sim.run(circuit, analyses, None)
            runs.append(GridRun(combo=combo_map, handle=handle))
            done += 1
            _notify(done, total)
    finally:
        # restore
        for comp, _ in variables:
            try:
                comp.value = originals[comp.ref]
            except Exception:
                pass

    return GridResult(runs=runs)


__all__ += ["run_param_grid", "GridResult", "GridRun"]
