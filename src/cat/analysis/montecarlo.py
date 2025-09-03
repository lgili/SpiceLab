from __future__ import annotations

import random as _random
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..core.circuit import Circuit
from ..core.components import Component
from .core import AnalysisResult

# ---------- Distribuições (percentual sobre um nominal numérico) ----------


class Dist:
    """Interface de distribuição. Implementações devem sobrescrever sample(nominal)."""

    def sample(self, nominal: float, rnd: _random.Random) -> float:  # pragma: no cover (interface)
        raise NotImplementedError


class NormalPct(Dist):
    """
    Normal centrada no nominal, com desvio-padrão relativo (ex.: sigma_pct=0.05 -> 5%).
    """

    def __init__(self, sigma_pct: float) -> None:
        if sigma_pct < 0:
            raise ValueError("sigma_pct must be >= 0")
        self.sigma_pct = sigma_pct

    def sample(self, nominal: float, rnd: _random.Random) -> float:
        sigma = abs(nominal) * self.sigma_pct
        return float(rnd.gauss(nominal, sigma))


class UniformPct(Dist):
    """
    Uniforme no intervalo [nominal*(1-pct), nominal*(1+pct)].
    """

    def __init__(self, pct: float) -> None:
        if pct < 0:
            raise ValueError("pct must be >= 0")
        self.pct = pct

    def sample(self, nominal: float, rnd: _random.Random) -> float:
        lo = nominal * (1.0 - self.pct)
        hi = nominal * (1.0 + self.pct)
        return float(rnd.uniform(lo, hi))


# ---------- Execução ----------


@dataclass(frozen=True)
class MonteCarloResult:
    """Resultados de Monte Carlo: amostras por run e AnalysisResult por run."""

    samples: list[dict[str, float]]  # lista de dicts { comp_key: value_sampled }
    runs: list[AnalysisResult]  # resultados de simulação


def _as_float(value: str | float) -> float:
    """
    Converte valores simples para float. Aceita float direto ou string numérica "1000", "1e3".
    (Não converte unidades SPICE '1k', '100n' nesta primeira versão.)
    """
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except Exception as e:
        raise ValueError(
            f"Monte Carlo currently needs numeric literals (got {value!r}). "
            "Future versions will support unit suffixes like '1k', '100n'."
        ) from e


def monte_carlo(
    circuit: Circuit,
    mapping: Mapping[Component, Dist],
    n: int,
    analysis_factory: Callable[[], object],
    seed: int | None = None,
    label_fn: Callable[[Component], str] | None = None,
) -> MonteCarloResult:
    """
    Executa Monte Carlo variando valores de componentes conforme distribuições.

    - `mapping`: { componente: Dist(...) }  (não precisa ser hashável; só iteramos)
    - `n`: número de execuções
    - `analysis_factory`: callable que cria a análise (ex.: `lambda: TRAN("100us","1ms")`)
    - `seed`: opcional, para reprodutibilidade
    - `label_fn`: como nomear cada componente nos samples (default: Class.ref, ex. "Resistor.1")

    Limitação inicial: os `component.value` precisam ser numéricos simples (float ou "123.0").
    """
    rnd = _random.Random(seed)

    def _label(c: Component) -> str:
        if label_fn:
            return label_fn(c)
        return f"{type(c).__name__}.{c.ref}"

    # snapshot dos valores originais e pré-cálculo dos nominais
    # (evita usar dict com Component como chave)
    originals: list[tuple[Component, str | float]] = [(comp, comp.value) for comp in mapping.keys()]
    items: list[tuple[Component, Dist, float]] = [
        (comp, dist, _as_float(comp.value)) for comp, dist in mapping.items()
    ]

    samples: list[dict[str, float]] = []
    runs: list[AnalysisResult] = []
    try:
        for _ in range(n):
            current_sample: dict[str, float] = {}
            # sampleia e aplica
            for comp, dist, nominal in items:
                new_val = dist.sample(nominal, rnd)
                comp.value = new_val
                current_sample[_label(comp)] = new_val
            # roda análise
            analysis = analysis_factory()
            res = analysis.run(circuit)  # type: ignore[attr-defined]
            runs.append(res)
            samples.append(current_sample)
    finally:
        # restaura originais
        for comp, val in originals:
            comp.value = val

    return MonteCarloResult(samples=samples, runs=runs)
