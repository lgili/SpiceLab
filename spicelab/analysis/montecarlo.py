from __future__ import annotations

import importlib
import math
import random as _random
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from collections.abc import Mapping as TMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from ..core.circuit import Circuit
from ..core.components import Component
from ..core.types import AnalysisSpec, ResultHandle
from ..engines import EngineName
from ..io.raw_reader import TraceSet
from ..orchestrator import Job, JobResult, run_job
from ..spice.base import RunArtifacts, RunResult
from ..utils.units import to_float
from .result import AnalysisResult
from .stats import Statistics


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


class NormalAbs(Dist):
    """Normal distribution with absolute sigma (not percentage).

    Use for parameters like op-amp offset voltage where tolerance is
    specified as an absolute value (e.g., ±2mV) rather than percentage.

    Args:
        sigma: Absolute sigma value (e.g., 0.002 for 2mV at 3-sigma)
    """

    def __init__(self, sigma: float) -> None:
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        self.sigma = sigma

    def sample(self, nominal: float, rnd: _random.Random) -> float:
        return float(rnd.gauss(nominal, self.sigma))

    def __repr__(self) -> str:
        return f"NormalAbs({self.sigma})"


class TriangularAbs(Dist):
    """Triangular distribution with absolute half-range (not percentage).

    Args:
        delta: Absolute half-range (e.g., 0.005 for ±5mV range)
    """

    def __init__(self, delta: float) -> None:
        if delta < 0:
            raise ValueError("delta must be >= 0")
        self.delta = delta

    def sample(self, nominal: float, rnd: _random.Random) -> float:
        return float(rnd.triangular(nominal - self.delta, nominal + self.delta, nominal))

    def __repr__(self) -> str:
        return f"TriangularAbs({self.delta})"


# ---------- Correlated Groups ----------


class CorrelatedGroup:
    """Group of components that vary together (same-lot correlation).

    Components in a CorrelatedGroup share the same random factor in each
    Monte Carlo iteration, modeling the behavior of components from the
    same manufacturing batch.

    Args:
        components: List of Component objects that vary together
        dist: Distribution to apply (the same random draw is used for all)

    Example:
        >>> # Resistors from same batch vary together
        >>> monte_carlo(circuit, {
        ...     CorrelatedGroup([R1, R2, R3], NormalPct(0.001)): None,
        ...     Voff1: NormalAbs(0.002),  # Independent
        ... }, n=1000, ...)
    """

    def __init__(self, components: Sequence[Component], dist: Dist) -> None:
        if not components:
            raise ValueError("CorrelatedGroup must have at least one component")
        self.components = list(components)
        self.dist = dist

    def __repr__(self) -> str:
        refs = [getattr(c, "ref", str(c)) for c in self.components]
        return f"CorrelatedGroup({refs}, {self.dist!r})"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


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
    handles: list[ResultHandle] | None = None
    job: JobResult | None = None

    def result_handles(self) -> list[ResultHandle]:
        """Return the list of ResultHandle objects backing each run (if available)."""

        return list(self.handles or [])

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

        import pandas as pd  # local import; optional runtime dependency

        df = pd.DataFrame(self.mapping_manifest, columns=["label", "nominal", "dist"])
        df.to_csv(path, index=index, **to_csv_kwargs)

    def extract_values(
        self,
        metric: Callable[[AnalysisResult], float],
    ) -> list[float]:
        """Extract a metric value from each Monte Carlo run.

        Args:
            metric: Function that extracts a scalar from an AnalysisResult.

        Returns:
            List of extracted values, one per run.

        Example:
            >>> values = mc_result.extract_values(
            ...     lambda r: r.traces['V(vout)'].values[-1]
            ... )
        """
        return [metric(run) for run in self.runs]

    def statistics(
        self,
        metric: Callable[[AnalysisResult], float],
    ) -> Statistics:
        """Compute statistics for a metric across all Monte Carlo runs.

        Args:
            metric: Function that extracts a scalar from an AnalysisResult.

        Returns:
            Statistics object with mean, std, percentiles, etc.

        Example:
            >>> stats = mc_result.statistics(
            ...     lambda r: r.traces['V(vout)'].values[-1]
            ... )
            >>> print(f"Mean: {stats.mean}, Std: {stats.std}")
        """
        from .stats import compute_stats

        values = self.extract_values(metric)
        return compute_stats(values)

    def cpk(
        self,
        metric: Callable[[AnalysisResult], float],
        lsl: float,
        usl: float,
    ) -> float:
        """Calculate process capability index Cpk for a metric.

        Cpk measures how well a process fits within specification limits.
        Higher Cpk indicates better process capability:
        - Cpk >= 1.33: Acceptable for most applications
        - Cpk >= 1.67: Good capability
        - Cpk >= 2.00: Excellent (6-sigma quality)

        Args:
            metric: Function that extracts a scalar from an AnalysisResult.
            lsl: Lower specification limit.
            usl: Upper specification limit.

        Returns:
            Cpk value.

        Example:
            >>> cpk = mc_result.cpk(
            ...     lambda r: r.traces['V(vout)'].values[-1],
            ...     lsl=2.4, usl=2.6,
            ... )
            >>> print(f"Cpk: {cpk:.2f}")
        """
        stats = self.statistics(metric)
        return stats.cpk(lsl, usl)

    def yield_estimate(
        self,
        metric: Callable[[AnalysisResult], float],
        lsl: float,
        usl: float,
    ) -> float:
        """Estimate yield (fraction within spec limits).

        Uses normal distribution assumption based on computed statistics.

        Args:
            metric: Function that extracts a scalar from an AnalysisResult.
            lsl: Lower specification limit.
            usl: Upper specification limit.

        Returns:
            Estimated yield as fraction (0.0 to 1.0).

        Example:
            >>> yield_pct = mc_result.yield_estimate(
            ...     lambda r: r.traces['V(vout)'].values[-1],
            ...     lsl=2.4, usl=2.6,
            ... ) * 100
            >>> print(f"Estimated yield: {yield_pct:.2f}%")
        """
        stats = self.statistics(metric)
        return stats.yield_estimate(lsl, usl)

    def actual_yield(
        self,
        metric: Callable[[AnalysisResult], float],
        lsl: float,
        usl: float,
    ) -> float:
        """Calculate actual yield from Monte Carlo samples.

        This counts how many samples fall within spec limits, which may
        differ from the normal distribution estimate if the distribution
        is non-normal.

        Args:
            metric: Function that extracts a scalar from an AnalysisResult.
            lsl: Lower specification limit.
            usl: Upper specification limit.

        Returns:
            Actual yield as fraction (0.0 to 1.0).

        Example:
            >>> actual = mc_result.actual_yield(
            ...     lambda r: r.traces['V(vout)'].values[-1],
            ...     lsl=2.4, usl=2.6,
            ... )
            >>> print(f"Actual yield: {actual * 100:.2f}%")
        """
        values = self.extract_values(metric)
        in_spec = sum(1 for v in values if lsl <= v <= usl)
        return in_spec / len(values) if values else 0.0

    def sigma_level(
        self,
        metric: Callable[[AnalysisResult], float],
        lsl: float,
        usl: float,
    ) -> float:
        """Convert Cpk to equivalent sigma level.

        Args:
            metric: Function that extracts a scalar from an AnalysisResult.
            lsl: Lower specification limit.
            usl: Upper specification limit.

        Returns:
            Equivalent sigma level (e.g., 6.0 for 6-sigma).

        Example:
            >>> sigma = mc_result.sigma_level(
            ...     lambda r: r.traces['V(vout)'].values[-1],
            ...     lsl=2.4, usl=2.6,
            ... )
            >>> print(f"Sigma level: {sigma:.1f}")
        """
        return self.cpk(metric, lsl, usl) * 3.0

    def process_summary(
        self,
        metric: Callable[[AnalysisResult], float],
        lsl: float,
        usl: float,
        *,
        metric_name: str = "metric",
    ) -> dict[str, Any]:
        """Generate a comprehensive process capability summary.

        Args:
            metric: Function that extracts a scalar from an AnalysisResult.
            lsl: Lower specification limit.
            usl: Upper specification limit.
            metric_name: Name of the metric for display.

        Returns:
            Dictionary with all process capability metrics.

        Example:
            >>> summary = mc_result.process_summary(
            ...     lambda r: r.traces['V(vout)'].values[-1],
            ...     lsl=2.4, usl=2.6,
            ...     metric_name="Vout",
            ... )
            >>> print(f"Cpk: {summary['cpk']:.2f}")
            >>> print(f"Yield: {summary['yield_pct']:.1f}%")
        """
        stats = self.statistics(metric)
        values = self.extract_values(metric)

        # Count actual failures
        failures = sum(1 for v in values if v < lsl or v > usl)

        return {
            "metric_name": metric_name,
            "n": len(values),
            "mean": stats.mean,
            "std": stats.std,
            "min": stats.min,
            "max": stats.max,
            "median": stats.median,
            "p1": stats.p1,
            "p5": stats.p5,
            "p95": stats.p95,
            "p99": stats.p99,
            "sigma3_low": stats.sigma3_low,
            "sigma3_high": stats.sigma3_high,
            "lsl": lsl,
            "usl": usl,
            "cpk": stats.cpk(lsl, usl),
            "sigma_level": stats.cpk(lsl, usl) * 3.0,
            "yield_estimate": stats.yield_estimate(lsl, usl),
            "yield_pct": stats.yield_estimate(lsl, usl) * 100,
            "actual_yield": (len(values) - failures) / len(values) if values else 0.0,
            "actual_yield_pct": (len(values) - failures) / len(values) * 100 if values else 0.0,
            "failures": failures,
        }


def _as_float(value: str | float) -> float:
    return to_float(value)


def _handle_to_analysis_result(handle: ResultHandle) -> AnalysisResult:
    ds = handle.dataset()
    traces = TraceSet.from_dataset(ds)
    attrs = handle.attrs()
    artifacts = RunArtifacts(
        netlist_path=str(attrs.get("netlist_path") or ""),
        log_path=str(attrs.get("log_path") or ""),
        raw_path=str(attrs.get("raw_path")) if attrs.get("raw_path") else None,
        workdir=str(attrs.get("workdir")) if attrs.get("workdir") else None,
    )
    run_result = RunResult(
        artifacts=artifacts,
        returncode=int(attrs.get("returncode", 0)),
        stdout=str(attrs.get("stdout", "")),
        stderr=str(attrs.get("stderr", "")),
    )
    return AnalysisResult(run=run_result, traces=traces)


def _dataset_to_traceset(dataset: Any) -> TraceSet:
    """Backward-compatible helper retained for tests and external callers."""

    warnings.warn(
        "_dataset_to_traceset is deprecated; use TraceSet.from_dataset instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return TraceSet.from_dataset(dataset)


def monte_carlo(
    circuit: Circuit,
    mapping: Mapping[Component | CorrelatedGroup, Dist | None],
    n: int,
    seed: int | None = None,
    label_fn: Callable[[Component], str] | None = None,
    workers: int = 1,
    progress: bool | Callable[[int, int], None] | None = None,
    *,
    analyses: Sequence[AnalysisSpec] | None = None,
    engine: EngineName = "ngspice",
    cache_dir: str | Path | None = ".spicelab_cache",
    reuse_cache: bool = True,
) -> MonteCarloResult:
    """
    Executa Monte Carlo variando valores dos componentes conforme distribuições.

    Supports both independent components and correlated groups:
        - Component -> Dist: Each component varies independently
        - CorrelatedGroup -> None: Components in group share same random factor

    Example:
        >>> monte_carlo(circuit, {
        ...     CorrelatedGroup([R1, R2, R3], NormalPct(0.001)): None,  # Same batch
        ...     Voff1: NormalAbs(0.002),  # Independent offset
        ...     Voff2: NormalAbs(0.002),  # Independent offset
        ... }, n=1000, analyses=[...])
    """
    if analyses is None:
        raise ValueError("Provide 'analyses' when running monte_carlo")

    rnd = _random.Random(seed)

    def _label(c: Component) -> str:
        if label_fn:
            return label_fn(c)
        return f"{type(c).__name__}.{c.ref}"

    # Separate correlated groups from individual components
    correlated_groups: list[CorrelatedGroup] = []
    individual_comps: list[Component] = []
    individual_dists: list[Dist] = []

    for key, dist in mapping.items():
        if isinstance(key, CorrelatedGroup):
            correlated_groups.append(key)
        else:
            individual_comps.append(key)
            if dist is None:
                raise ValueError(f"Individual component {key} must have a distribution, not None")
            individual_dists.append(dist)

    # Build complete component list for ref lookup
    all_comps: list[Component] = list(individual_comps)
    for group in correlated_groups:
        all_comps.extend(group.components)

    ref_lookup: dict[Component, str] = {}
    for comp in all_comps:
        ref = getattr(comp, "ref", None)
        if ref is None:
            raise ValueError("All components in mapping must have .ref for Monte Carlo jobs")
        ref_lookup[comp] = str(ref)

    # Precompute nominals
    individual_nominals: list[float] = [_as_float(c.value) for c in individual_comps]
    group_nominals: list[list[float]] = [
        [_as_float(c.value) for c in g.components] for g in correlated_groups
    ]

    samples: list[dict[str, float]] = []
    combos: list[dict[str, float]] = []
    for _ in range(n):
        s: dict[str, float] = {}
        combo: dict[str, float] = {}

        # Sample individual components (independent)
        for comp, nominal, dist in zip(
            individual_comps, individual_nominals, individual_dists, strict=False
        ):
            sampled = dist.sample(nominal, rnd)
            s[_label(comp)] = sampled
            combo[ref_lookup[comp]] = sampled

        # Sample correlated groups (shared random factor per group)
        for group, nominals in zip(correlated_groups, group_nominals, strict=False):
            # Generate ONE random factor for the entire group
            # We sample at nominal=1.0 to get a multiplier, then apply to each component
            if isinstance(group.dist, NormalPct | LogNormalPct | UniformPct | TriangularPct):
                # Percentage-based: sample multiplier at nominal=1.0
                multiplier = group.dist.sample(1.0, rnd)
                for comp, nominal in zip(group.components, nominals, strict=False):
                    sampled = nominal * multiplier
                    s[_label(comp)] = sampled
                    combo[ref_lookup[comp]] = sampled
            else:
                # Absolute-based: sample offset once, apply to all
                # For absolute distributions, we get an offset from nominal=0
                offset = group.dist.sample(0.0, rnd)
                for comp, nominal in zip(group.components, nominals, strict=False):
                    sampled = nominal + offset
                    s[_label(comp)] = sampled
                    combo[ref_lookup[comp]] = sampled

        samples.append(s)
        combos.append(combo)

    # Build manifest: list of (label, nominal, dist_repr)
    manifest: list[tuple[str, float, str]] = []

    # Individual components
    for c, nom, d in zip(individual_comps, individual_nominals, individual_dists, strict=False):
        try:
            d_repr = repr(d)
        except Exception:
            d_repr = type(d).__name__
        manifest.append((_label(c), nom, d_repr))

    # Correlated groups
    for group, nominals in zip(correlated_groups, group_nominals, strict=False):
        try:
            d_repr = f"Correlated({repr(group.dist)})"
        except Exception:
            d_repr = f"Correlated({type(group.dist).__name__})"
        for c, nom in zip(group.components, nominals, strict=False):
            manifest.append((_label(c), nom, d_repr))

    if n <= 0:
        return MonteCarloResult(
            samples=samples,
            runs=[],
            mapping_manifest=manifest,
            handles=[],
            job=None,
        )

    job = Job(
        circuit=circuit,
        analyses=list(analyses),
        engine=engine,
        combos=tuple(dict(combo) for combo in combos),
    )
    job_result = run_job(
        job,
        cache_dir=cache_dir,
        workers=workers,
        progress=progress,
        reuse_cache=reuse_cache,
    )
    handles: list[ResultHandle] = []
    analysis_runs: list[AnalysisResult] = []
    for job_run in job_result.runs:
        handles.append(job_run.handle)
        analysis_runs.append(_handle_to_analysis_result(job_run.handle))
    if len(analysis_runs) != len(samples):
        raise RuntimeError("Mismatch between Monte Carlo samples and job results")
    return MonteCarloResult(
        samples=samples,
        runs=analysis_runs,
        mapping_manifest=manifest,
        handles=handles,
        job=job_result,
    )
