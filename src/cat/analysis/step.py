from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from typing import TypeVar

from ..core.circuit import Circuit
from ..io.raw_reader import parse_ngspice_ascii_raw
from ..spice.registry import get_run_directives
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
    run_directives = get_run_directives()
    res = run_directives(netlist, lines_with_params)
    if res.returncode != 0:
        raise RuntimeError(f"NGSpice exited with code {res.returncode}")
    if not res.artifacts.raw_path:
        raise RuntimeError("NGSpice produced no RAW path")
    traces = parse_ngspice_ascii_raw(res.artifacts.raw_path)
    return AnalysisResult(run=res, traces=traces)


def step_param(
    circuit: Circuit,
    name: str,
    values: Sequence[str | float],
    analysis_factory: Callable[[], A],
    workers: int = 1,
    progress: bool | Callable[[int, int], None] | None = None,
) -> StepResult:
    grid_list: list[dict[str, str | float]] = [{name: v} for v in values]
    net = circuit.build_netlist()

    # progress helper
    def _notify(done: int) -> None:
        if not progress:
            return
        if callable(progress):
            try:
                progress(done, len(grid_list))
            except Exception:
                pass
            return
        pct = int(round(100.0 * done / max(len(grid_list), 1)))
        sys.stderr.write(f"\rSTEP[{name}]: {done}/{len(grid_list)} ({pct}%)")
        sys.stderr.flush()

    runs: list[AnalysisResult] = []
    if workers <= 1:
        for i, point in enumerate(grid_list, start=1):
            base_dirs_one: list[str] = analysis_factory()._directives()  # type: ignore[attr-defined]
            lines_with_params = _directives_with_params(base_dirs_one, point)
            runs.append(_run_once_with_params_text(net, lines_with_params))
            _notify(i)
    else:
        # Preserva ordem dos pontos mesmo com execução paralela
        runs_buf: list[AnalysisResult | None] = [None] * len(grid_list)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_idx = {}
            for idx, point in enumerate(grid_list):
                base_dirs_two: list[str] = analysis_factory()._directives()  # type: ignore[attr-defined]
                lines_with_params2 = _directives_with_params(base_dirs_two, point)
                fut = ex.submit(_run_once_with_params_text, net, lines_with_params2)
                fut_to_idx[fut] = idx
            done = 0
            for f in as_completed(list(fut_to_idx.keys())):
                idx = fut_to_idx[f]
                runs_buf[idx] = f.result()
                done += 1
                _notify(done)
        runs = [r for r in runs_buf if r is not None]

    last = grid_list[-1] if grid_list else {}
    return StepResult(params=last, grid=grid_list, runs=runs)


def step_grid(
    circuit: Circuit,
    grid: ParamGrid,
    analysis_factory: Callable[[], A],
    order: Sequence[str] | None = None,
    workers: int = 1,
    progress: bool | Callable[[int, int], None] | None = None,
) -> StepResult:
    keys = list(order) if order else list(grid.keys())
    values_lists: list[Sequence[str | float]] = [grid[k] for k in keys]

    points: list[dict[str, str | float]] = [
        {k: v for k, v in zip(keys, combo, strict=False)} for combo in product(*values_lists)
    ]
    net = circuit.build_netlist()

    def _notify(done: int) -> None:
        if not progress:
            return
        if callable(progress):
            try:
                progress(done, len(points))
            except Exception:
                pass
            return
        pct = int(round(100.0 * done / max(len(points), 1)))
        sys.stderr.write(f"\rSTEP[grid]: {done}/{len(points)} ({pct}%)")
        sys.stderr.flush()

    runs: list[AnalysisResult] = []
    if workers <= 1:
        for i, point in enumerate(points, start=1):
            base_dirs_one: list[str] = analysis_factory()._directives()  # type: ignore[attr-defined]
            lines_with_params = _directives_with_params(base_dirs_one, point)
            runs.append(_run_once_with_params_text(net, lines_with_params))
            _notify(i)
    else:
        # Preserva ordem do grid em execução paralela
        runs_buf: list[AnalysisResult | None] = [None] * len(points)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_idx = {}
            for idx, point in enumerate(points):
                base_dirs_two: list[str] = analysis_factory()._directives()  # type: ignore[attr-defined]
                lines_with_params2 = _directives_with_params(base_dirs_two, point)
                fut = ex.submit(_run_once_with_params_text, net, lines_with_params2)
                fut_to_idx[fut] = idx
            done = 0
            for f in as_completed(list(fut_to_idx.keys())):
                idx = fut_to_idx[f]
                runs_buf[idx] = f.result()
                done += 1
                _notify(done)
        runs = [r for r in runs_buf if r is not None]

    last = points[-1] if points else {}
    return StepResult(params=last, grid=points, runs=runs)
