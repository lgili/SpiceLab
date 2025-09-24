from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from ..core.circuit import Circuit
from ..core.components import Component
from .core import AnalysisResult

A = TypeVar("A")  # tipo da análise (OP/TRAN/AC/DC), mas receberemos uma fábrica callable


@dataclass(frozen=True)
class SweepResult:
    param_name: str
    values: list[str | float]
    runs: list[AnalysisResult]


def sweep_component(
    circuit: Circuit,
    component: Component,
    values: Sequence[str | float],
    analysis_factory: Callable[[], A],
    param_name: str | None = None,
    *,
    progress: bool | Callable[[int, int], None] | None = None,
) -> SweepResult:
    """Executa várias simulações alterando `component.value` em Python.

    - `values`: lista de valores a aplicar no componente (ex.: ["1k","2k","5k"])
    - `analysis_factory`: callable que cria uma instância da análise a cada iteração,
      por ex.: `lambda: TRAN("100us","1ms")` ou `lambda: OP()`
    - `param_name`: opcional, nome amigável para registrar no resultado
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
        pct = int(round(100.0 * done / max(len(values), 1)))
        sys.stderr.write(f"\rSWEEP[{component.ref}]: {done}/{len(values)} ({pct}%)")
        sys.stderr.flush()

    runs: list[AnalysisResult] = []
    try:
        for i, v in enumerate(values, start=1):
            component.value = v
            analysis = analysis_factory()
            res = analysis.run(circuit)  # type: ignore[attr-defined]
            runs.append(res)
            _notify(i)
    finally:
        component.value = original  # restore
    return SweepResult(
        param_name or f"{type(component).__name__}.{component.ref}", list(values), runs
    )
