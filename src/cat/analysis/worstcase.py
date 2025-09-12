from __future__ import annotations

import math
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ..core.circuit import Circuit
from ..io.raw_reader import parse_ngspice_ascii_raw
from ..spice.registry import get_run_directives
from .core import AnalysisResult


@dataclass(frozen=True)
class WorstCaseResult:
    best_params: dict[str, float | str]
    best_value: float
    history: list[tuple[dict[str, float | str], float]]  # (params, metric)


def _directives_with_params(base: list[str], p: Mapping[str, float | str]) -> list[str]:
    return [*(f".param {k}={v}" for k, v in p.items()), *base]


def _run_with_params(net: str, lines_with_params: list[str]) -> AnalysisResult:
    run_directives = get_run_directives()
    res = run_directives(net, lines_with_params)
    if res.returncode != 0 or not res.artifacts.raw_path:
        raise RuntimeError("NGSpice failed in worst-case run")
    traces = parse_ngspice_ascii_raw(res.artifacts.raw_path)
    return AnalysisResult(run=res, traces=traces)


def worst_case(
    circuit: Circuit,
    analysis_factory: Callable[[], object],
    metric: Callable[[AnalysisResult], float],
    space: Mapping[str, Sequence[float | str]],
    mode: str = "min",  # "min" or "max"
    n_random: int = 64,
    n_refine: int = 3,
    progress: bool | Callable[[int, int], None] | None = None,
) -> WorstCaseResult:
    """
    Busca pior caso sobre parâmetros .param discretizados em 'space'.
    Estratégia: amostragem aleatória inicial -> refinamento coordenado local.
    """
    net = circuit.build_netlist()
    base = analysis_factory()._directives()  # type: ignore[attr-defined]

    # 1) random
    total = n_random + sum(len(space[k]) for k in space) * n_refine

    def _notify(done: int) -> None:
        if not progress:
            return
        if callable(progress):
            try:
                progress(done, total)
            except Exception:
                pass
            return
        pct = int(round(100.0 * done / max(total, 1)))
        sys.stderr.write(f"\rWORST: {done}/{total} ({pct}%)")
        sys.stderr.flush()

    hist: list[tuple[dict[str, float | str], float]] = []
    best_p: dict[str, float | str] = {}
    best_v = math.inf if mode == "min" else -math.inf

    keys = list(space.keys())
    choices = [list(space[k]) for k in keys]

    done = 0
    for _ in range(n_random):
        p = {k: random.choice(choices[i]) for i, k in enumerate(keys)}
        res = _run_with_params(net, _directives_with_params(base, p))
        val = metric(res)
        hist.append((p, val))
        if (mode == "min" and val < best_v) or (mode == "max" and val > best_v):
            best_p, best_v = p, val
        done += 1
        _notify(done)

    # 2) refinamento coordenado
    for _ in range(n_refine):
        improved = False
        for i, k in enumerate(keys):
            cand: list[tuple[float, dict[str, float | str]]] = []
            for v in choices[i]:
                p2 = dict(best_p)
                p2[k] = v
                res = _run_with_params(net, _directives_with_params(base, p2))
                val = metric(res)
                hist.append((p2, val))
                cand.append((val, p2))
                done += 1
                _notify(done)
            if mode == "min":
                val, p_sel = min(cand, key=lambda x: x[0])
                if val < best_v:
                    best_v, best_p, improved = val, p_sel, True
            else:
                val, p_sel = max(cand, key=lambda x: x[0])
                if val > best_v:
                    best_v, best_p, improved = val, p_sel, True
        if not improved:
            break

    return WorstCaseResult(best_params=best_p, best_value=best_v, history=hist)
