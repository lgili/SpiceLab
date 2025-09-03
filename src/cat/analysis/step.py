from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import TypeVar

from ..core.circuit import Circuit
from ..spice import ngspice_cli
from .core import AnalysisResult

A = TypeVar("A")  # instância concreta de análise (OP/TRAN/AC/DC)


@dataclass(frozen=True)
class StepResult:
    """Resultado de uma varredura paramétrica Python-level."""

    params: dict[str, str | float]  # último conjunto varrido (para compat)
    grid: list[dict[str, str | float]]  # todos os pontos simulados
    runs: list[AnalysisResult]  # um resultado por ponto


# Tipo amigável para grade: {"R": ["1k","2k"], "C": ["100n","220n"]}
ParamGrid = Mapping[str, Sequence[str | float]]


def _directives_with_params(
    base_directives: list[str],
    param_values: Mapping[str, str | float],
) -> list[str]:
    """Prepara diretivas: insere .param no começo, conserva resto."""
    prefix = [f".param {k}={v}" for k, v in param_values.items()]
    return [*prefix, *base_directives]


def _run_once_with_params(
    circuit: Circuit,
    analysis_factory: Callable[[], A],
    param_values: Mapping[str, str | float],
) -> AnalysisResult:
    """Executa uma análise inserindo .param e reaproveitando o runner genérico."""
    # Obter diretivas da análise concreta pela mesma rota do _BaseAnalysis.run
    # sem expor _BaseAnalysis — pedimos que a instância tenha _directives()
    analysis = analysis_factory()
    directives: list[str] = analysis._directives()  # type: ignore[attr-defined]
    directives = _directives_with_params(directives, param_values)
    net = circuit.build_netlist()
    res = ngspice_cli.run_directives(net, directives)
    if res.returncode != 0:
        raise RuntimeError(f"NGSpice exited with code {res.returncode}")
    if not res.artifacts.raw_path:
        raise RuntimeError("NGSpice produced no RAW path")
    from ..io.raw_reader import parse_ngspice_ascii_raw

    traces = parse_ngspice_ascii_raw(res.artifacts.raw_path)
    return AnalysisResult(run=res, traces=traces)


def step_param(
    circuit: Circuit,
    name: str,
    values: Sequence[str | float],
    analysis_factory: Callable[[], A],
) -> StepResult:
    """Varre um único parâmetro (`.param name=value`) executando N simulações.

    Use o nome do parâmetro nas *values* de componentes como `"{name}"`.
    Ex.: R1.value = "{R}" e chamar step_param(..., name="R", values=["1k","2k"])
    """
    grid_list: list[dict[str, str | float]] = [{name: v} for v in values]
    runs: list[AnalysisResult] = []
    for point in grid_list:
        runs.append(_run_once_with_params(circuit, analysis_factory, point))
    last = grid_list[-1] if grid_list else {}
    return StepResult(params=last, grid=grid_list, runs=runs)


def step_grid(
    circuit: Circuit,
    grid: ParamGrid,
    analysis_factory: Callable[[], A],
    order: Sequence[str] | None = None,
) -> StepResult:
    """Varre múltiplos parâmetros pelo produto cartesiano das listas.

    `grid` é um mapeamento {"R": [...], "C": [...], ...}.
    A ordem dos parâmetros no produto pode ser fixada via `order`.
    """
    keys = list(order) if order else list(grid.keys())
    values_lists: list[Sequence[str | float]] = [grid[k] for k in keys]

    runs: list[AnalysisResult] = []
    grid_points: list[dict[str, str | float]] = []
    for combo in product(*values_lists):
        point = {k: v for k, v in zip(keys, combo)}
        grid_points.append(point)
        runs.append(_run_once_with_params(circuit, analysis_factory, point))

    last = grid_points[-1] if grid_points else {}
    return StepResult(params=last, grid=grid_points, runs=runs)
