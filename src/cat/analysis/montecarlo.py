from __future__ import annotations

import copy
import importlib
import math
import random as _random
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
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


class _MiniDataFrame:
    """Tiny, pandas-free table used when pandas isn't available."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = [dict(r) for r in rows]
        columns: list[str] = []
        for row in self._rows:
            for key in row.keys():
                if key not in columns:
                    columns.append(key)
        self.columns: list[str] = columns
        self._data: dict[str, list[Any]] = {
            col: [r.get(col) for r in self._rows] for col in columns
        }

    def __getitem__(self, key: str) -> list[Any]:
        return self._data[key]

    def __len__(self) -> int:  # pragma: no cover - defensive helper
        return len(self._rows)

    def __iter__(self) -> Iterator[dict[str, Any]]:  # pragma: no cover - rarely used in tests
        return iter(self._rows)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MiniDataFrame(columns={self.columns!r}, rows={len(self._rows)})"


@dataclass(frozen=True)
class MonteCarloResult:
    samples: list[dict[str, float]]
    runs: list[AnalysisResult]
    # optional metadata about the varied parameters: list of (label, nominal, dist_repr)
    mapping_manifest: list[tuple[str, float, str]] | None = None

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
        except Exception:  # pragma: no cover
            pd = None

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
        if pd is None:
            return _MiniDataFrame(rows)
        return pd.DataFrame(rows)

    def to_csv(
        self,
        path: str,
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
        columns: Sequence[str] | None = None,
        index: bool = False,
        **to_csv_kwargs: Any,
    ) -> None:
        """Write the Monte Carlo per-trial table to CSV.

        - `path`: output file path (passed to pandas.DataFrame.to_csv).
        - `metric`, `trial_name`, `param_prefix`, `y`, `sample_at` are forwarded
          to :meth:`to_dataframe` and behave the same.
        - `columns`: optional sequence of column names to keep (order preserved).
        - `index`: whether to write the DataFrame index (default False).
        - `to_csv_kwargs`: additional keyword args passed to pandas.DataFrame.to_csv.

        Raises RuntimeError if pandas is not available.
        """
        try:
            importlib.import_module("pandas")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas is required for MonteCarloResult.to_csv()") from exc

        df = self.to_dataframe(
            metric=metric,
            trial_name=trial_name,
            param_prefix=param_prefix,
            y=y,
            sample_at=sample_at,
        )
        if columns is not None:
            df = df.loc[:, list(columns)]
        df.to_csv(path, index=index, **to_csv_kwargs)

    def save_samples_csv(
        self, path: str, *, param_prefix: str = "", index: bool = False, **to_csv_kwargs: Any
    ) -> None:
        """Write only the sampled parameters (and trial index) to CSV.

        This is a convenience helper that writes the per-trial sampled parameters
        (the entries produced when generating the Monte Carlo `samples`) to a CSV
        file. Columns are the sampled parameter names (optionally prefixed) and
        the trial column named 'trial'.
        """
        try:
            importlib.import_module("pandas")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "pandas is required for MonteCarloResult.save_samples_csv()"
            ) from exc

        df = self.to_dataframe(metric=None, trial_name="trial", param_prefix=param_prefix, y=None)
        df.to_csv(path, index=index, **to_csv_kwargs)

    def save_manifest_csv(self, path: str, *, index: bool = False, **to_csv_kwargs: Any) -> None:
        """Write a small manifest describing the varied parameters to CSV.

        The manifest columns are: label, nominal, dist. The manifest is taken from
        `mapping_manifest` populated by the `monte_carlo` helper when available.
        """
        try:
            importlib.import_module("pandas")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "pandas is required for MonteCarloResult.save_manifest_csv()"
            ) from exc

        if not self.mapping_manifest:
            # nothing to write
            return

        import pandas as pd  # type: ignore[import-untyped]  # local import; optional runtime dependency

        df = pd.DataFrame(self.mapping_manifest, columns=["label", "nominal", "dist"])
        df.to_csv(path, index=index, **to_csv_kwargs)


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
    progress: bool | Callable[[int, int], None] | None = None,
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

    # Progress handler (optional)
    printer = None

    def _notify(done: int, total: int) -> None:
        if progress is None:
            return
        if callable(progress):
            try:
                progress(done, total)
            except Exception:
                pass
            return
        # simple stderr bar
        nonlocal printer
        if progress is True:
            # lazy-init
            class _Bar:
                def __init__(self, total: int) -> None:
                    self.total = total
                    self.last = -1

                def update(self, done: int) -> None:
                    if done == self.last:
                        return
                    pct = int(round(100.0 * done / max(self.total, 1)))
                    sys.stderr.write(f"\rMC: {done}/{self.total} ({pct}%)")
                    sys.stderr.flush()
                    self.last = done

                def close(self) -> None:
                    sys.stderr.write("\n")

            if printer is None:
                printer = _Bar(total)
            printer.update(done)

    runs: list[AnalysisResult] = []
    if workers <= 1:
        for i, s in enumerate(samples, start=1):
            runs.append(_run_one(s))
            _notify(i, len(samples))
    else:
        # Executa em paralelo preservando a ordem dos samples
        runs_buf: list[AnalysisResult | None] = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_idx = {}
            for idx, s in enumerate(samples):
                fut = ex.submit(_run_one, s)
                fut_to_idx[fut] = idx
            done = 0
            for f in as_completed(list(fut_to_idx.keys())):
                idx = fut_to_idx[f]
                runs_buf[idx] = f.result()
                done += 1
                _notify(done, len(samples))
        runs = [r for r in runs_buf if r is not None]

    if isinstance(progress, bool) and progress and printer is not None:
        try:
            printer.close()
        except Exception:
            pass

    # build optional manifest: list of (label, nominal, dist_repr)
    manifest: list[tuple[str, float, str]] = []
    for c, nom, d in zip(comps, nominals, dists, strict=False):
        try:
            d_repr = repr(d)
        except Exception:
            d_repr = type(d).__name__
        manifest.append((_label(c), nom, d_repr))

    return MonteCarloResult(samples=samples, runs=runs, mapping_manifest=manifest)
