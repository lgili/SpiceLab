from __future__ import annotations

import copy
import importlib
import math
import random as _random
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Mapping as TMapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol, cast

from ..core.circuit import Circuit
from ..core.components import Component
from ..utils.units import to_float
from .core import AnalysisResult


class _RunsAnalysis(Protocol):
    def run(self, circuit: Circuit) -> AnalysisResult: ...


# ---------- Distribuições ----------


class Dist:
    def sample(self, nominal: float, rnd: _random.Random) -> float:  # pragma: no cover
        raise NotImplementedError


class NormalPct(Dist):
    def __init__(self, sigma_pct: float) -> None:
        if sigma_pct < 0:
            raise ValueError("sigma_pct must be >= 0")
        self.sigma_pct = sigma_pct

    def sample(self, nominal: float, rnd: _random.Random) -> float:
        sigma = abs(nominal) * self.sigma_pct
        return float(rnd.gauss(nominal, sigma))


class LogNormalPct(Dist):
    def __init__(self, sigma_pct: float) -> None:
        if sigma_pct < 0:
            raise ValueError("sigma_pct must be >= 0")
        self.sigma_pct = sigma_pct

    def sample(self, nominal: float, rnd: _random.Random) -> float:
        if nominal <= 0:
            return nominal
        sigma = abs(nominal) * self.sigma_pct
        sigma_ln = sigma / max(abs(nominal), 1e-30)
        mu_ln = math.log(nominal) - 0.5 * (sigma_ln**2)
        return float(math.exp(rnd.gauss(mu_ln, sigma_ln)))


class UniformPct(Dist):
    def __init__(self, pct: float) -> None:
        if pct < 0:
            raise ValueError("pct must be >= 0")
        self.pct = pct

    def sample(self, nominal: float, rnd: _random.Random) -> float:
        lo = nominal * (1.0 - self.pct)
        hi = nominal * (1.0 + self.pct)
        return float(rnd.uniform(lo, hi))


class UniformAbs(Dist):
    def __init__(self, delta: float) -> None:
        if delta < 0:
            raise ValueError("delta must be >= 0")
        self.delta = delta

    def sample(self, nominal: float, rnd: _random.Random) -> float:
        return float(rnd.uniform(nominal - self.delta, nominal + self.delta))


class TriangularPct(Dist):
    def __init__(self, pct: float) -> None:
        if pct < 0:
            raise ValueError("pct must be >= 0")
        self.pct = pct

    def sample(self, nominal: float, rnd: _random.Random) -> float:
        lo = nominal * (1.0 - self.pct)
        hi = nominal * (1.0 + self.pct)
        return float(rnd.triangular(lo, hi, nominal))


# ---------- Execução ----------


@dataclass(frozen=True)
class MonteCarloResult:
    samples: list[dict[str, float]]
    runs: list[AnalysisResult]

    def to_dataframe(
        self,
        metric: (
            Callable[[AnalysisResult], float | dict[str, Any]]
            | TMapping[str, Callable[[AnalysisResult], Any]]
            | None
        ) = None,
        *,
        trial_name: str = "trial",
        param_prefix: str = "",
        y: Sequence[str] | None = None,
        sample_at: float | None = None,
    ) -> Any:
        """
        Returns a per-trial DataFrame with columns:
          - trial (index within this Monte Carlo run)
          - one column per sampled parameter (from `samples`), optionally prefixed
          - optional metric columns computed from each AnalysisResult
          - optional raw trace columns (final value or sampled at `sample_at` seconds)

        metric:
          - callable → result stored in column 'metric' (float or scalar)
          - mapping name->callable → adds one column per metric name
        y: list of trace names to extract values for each run. If `sample_at` is given,
           the value is linearly interpolated at t=sample_at using the run's time axis;
           otherwise, the last value in the trace is used.
        """
        try:
            pd: Any = importlib.import_module("pandas")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas is required for MonteCarloResult.to_dataframe()") from exc

        rows: list[dict[str, Any]] = []
        for i, (s, run) in enumerate(zip(self.samples, self.runs, strict=False)):
            # copy sampled params; optionally add prefix
            if param_prefix:
                row = {f"{param_prefix}{k}": v for k, v in s.items()}
            else:
                row = dict(s)
            row[trial_name] = i
            if metric is not None:
                if hasattr(metric, "items"):
                    for name, fn in cast(
                        TMapping[str, Callable[[AnalysisResult], Any]], metric
                    ).items():
                        row[name] = fn(run)
                else:
                    m = cast(Callable[[AnalysisResult], Any], metric)(run)
                    if isinstance(m, dict):
                        row.update(m)
                    else:
                        row["metric"] = m

            if y:
                try:
                    import numpy as _np  # local import to avoid hard dep at module import
                except Exception:  # pragma: no cover
                    _np = None  # type: ignore[assignment]

                ts = run.traces
                # pick x axis name
                xname = getattr(ts.x, "name", "time")
                for name in y:
                    vals = ts[name].values
                    if sample_at is not None and _np is not None and xname.lower() == "time":
                        t = ts[xname].values
                        row[name] = float(_np.interp(sample_at, t, vals))
                    else:
                        row[name] = (
                            float(vals[-1]) if len(vals) else _np.nan if _np is not None else 0.0
                        )
            rows.append(row)
        return pd.DataFrame(rows)


def _as_float(value: str | float) -> float:
    return to_float(value)


def monte_carlo(
    circuit: Circuit,
    mapping: Mapping[Component, Dist],
    n: int,
    analysis_factory: Callable[[], _RunsAnalysis],
    seed: int | None = None,
    label_fn: Callable[[Component], str] | None = None,
    workers: int = 1,
) -> MonteCarloResult:
    """
    Executa Monte Carlo variando valores dos componentes conforme distribuições.
    """
    rnd = _random.Random(seed)

    def _label(c: Component) -> str:
        if label_fn:
            return label_fn(c)
        return f"{type(c).__name__}.{c.ref}"

    comps: list[Component] = list(mapping.keys())
    nominals: list[float] = [_as_float(c.value) for c in comps]
    dists: list[Dist] = [mapping[c] for c in comps]

    samples: list[dict[str, float]] = []
    for _ in range(n):
        s: dict[str, float] = {}
        for comp, nominal, dist in zip(comps, nominals, dists, strict=False):
            s[_label(comp)] = dist.sample(nominal, rnd)
        samples.append(s)

    def _run_one(sample: dict[str, float]) -> AnalysisResult:
        c_copy: Circuit = copy.deepcopy(circuit)
        comp_list = getattr(c_copy, "components", None)
        if comp_list is None:
            comp_list = getattr(c_copy, "_components", [])
        by_label: dict[str, Component] = {_label(c): c for c in comp_list}
        for k, v in sample.items():
            by_label[k].value = v
        analysis = analysis_factory()
        return analysis.run(c_copy)

    runs: list[AnalysisResult] = []
    if workers <= 1:
        for s in samples:
            runs.append(_run_one(s))
    else:
        # Executa em paralelo preservando a ordem dos samples
        runs_buf: list[AnalysisResult | None] = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_idx = {}
            for idx, s in enumerate(samples):
                fut = ex.submit(_run_one, s)
                fut_to_idx[fut] = idx
            for f in as_completed(list(fut_to_idx.keys())):
                idx = fut_to_idx[f]
                runs_buf[idx] = f.result()
        runs = [r for r in runs_buf if r is not None]

    return MonteCarloResult(samples=samples, runs=runs)
