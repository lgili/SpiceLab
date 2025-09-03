from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from typing import TypeVar, cast

from ..core.circuit import Circuit
from ..io.raw_reader import parse_ngspice_ascii_raw
from ..spice import ngspice_cli
from ..spice.base import RunResult as BaseRunResult
from .core import AnalysisResult

A = TypeVar("A")  # instância de análise com método _directives()


@dataclass(frozen=True)
class StepResult:
    params: dict[str, str | float]
    grid: list[dict[str, str | float]]
    runs: list[AnalysisResult]


ParamGrid = Mapping[str, Sequence[str | float]]


def _directives_with_params(
    base_directives: list[str],
    param_values: Mapping[str, str | float],
) -> list[str]:
    return [
        *(f".param {k}={v}" for k, v in param_values.items()),
        *base_directives,
    ]


def _run_once_with_params_text(netlist: str, lines_with_params: list[str]) -> AnalysisResult:
    res = ngspice_cli.run_directives(netlist, lines_with_params)
    if res.returncode != 0:
        raise RuntimeError(f"NGSpice exited with code {res.returncode}")
    if not res.artifacts.raw_path:
        raise RuntimeError("NGSpice produced no RAW path")
    traces = parse_ngspice_ascii_raw(res.artifacts.raw_path)
    return AnalysisResult(run=cast(BaseRunResult, res), traces=traces)


def step_param(
    circuit: Circuit,
    name: str,
    values: Sequence[str | float],
    analysis_factory: Callable[[], A],
    workers: int = 1,
) -> StepResult:
    grid_list: list[dict[str, str | float]] = [{name: v} for v in values]
    net = circuit.build_netlist()

    runs: list[AnalysisResult] = []
    if workers <= 1:
        for point in grid_list:
            base_dirs_one: list[str] = analysis_factory()._directives()  # type: ignore[attr-defined]
            lines_with_params = _directives_with_params(base_dirs_one, point)
            runs.append(_run_once_with_params_text(net, lines_with_params))
    else:
        futs = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for point in grid_list:
                base_dirs_two: list[str] = analysis_factory()._directives()  # type: ignore[attr-defined]
                lines_with_params2 = _directives_with_params(base_dirs_two, point)
                futs.append(ex.submit(_run_once_with_params_text, net, lines_with_params2))
            for f in as_completed(futs):
                runs.append(f.result())

    last = grid_list[-1] if grid_list else {}
    return StepResult(params=last, grid=grid_list, runs=runs)


def step_grid(
    circuit: Circuit,
    grid: ParamGrid,
    analysis_factory: Callable[[], A],
    order: Sequence[str] | None = None,
    workers: int = 1,
) -> StepResult:
    keys = list(order) if order else list(grid.keys())
    values_lists: list[Sequence[str | float]] = [grid[k] for k in keys]

    points: list[dict[str, str | float]] = [
        {k: v for k, v in zip(keys, combo, strict=False)} for combo in product(*values_lists)
    ]
    net = circuit.build_netlist()

    runs: list[AnalysisResult] = []
    if workers <= 1:
        for point in points:
            base_dirs_one: list[str] = analysis_factory()._directives()  # type: ignore[attr-defined]
            lines_with_params = _directives_with_params(base_dirs_one, point)
            runs.append(_run_once_with_params_text(net, lines_with_params))
    else:
        futs = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for point in points:
                base_dirs_two: list[str] = analysis_factory()._directives()  # type: ignore[attr-defined]
                lines_with_params2 = _directives_with_params(base_dirs_two, point)
                futs.append(ex.submit(_run_once_with_params_text, net, lines_with_params2))
            for f in as_completed(futs):
                runs.append(f.result())

    last = points[-1] if points else {}
    return StepResult(params=last, grid=points, runs=runs)
