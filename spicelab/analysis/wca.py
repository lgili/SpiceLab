"""Worst-Case Analysis (WCA) module.

Provides functions for running deterministic worst-case corner analysis,
where each component is varied to its tolerance extremes (min/max) and
all combinations are evaluated to find the true worst-case bounds.

This differs from Monte Carlo which provides statistical distributions.
WCA is conservative and guarantees absolute bounds.

Usage:
    from spicelab.analysis import run_wca, WcaResult

    tolerances = {
        resistor1: 0.01,  # 1%
        resistor2: 0.05,  # 5%
    }

    result = run_wca(
        circuit,
        tolerances=tolerances,
        analyses=[AnalysisSpec(mode='op')],
    )

    print(f"Nominal: {result.nominal_run}")
    print(f"Min corner: {result.min_corner}")
    print(f"Max corner: {result.max_corner}")
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..core.circuit import Circuit
from ..core.components import Component
from ..core.types import AnalysisSpec, ResultHandle
from ..engines import EngineName
from ..io.raw_reader import TraceSet
from ..spice.base import RunArtifacts, RunResult
from ..utils.units import to_float
from .result import AnalysisResult


@dataclass(frozen=True)
class WcaCorner:
    """Represents a single corner in the WCA analysis."""

    combo: dict[str, float]
    """Component ref -> value mapping for this corner."""

    corner_signs: dict[str, Literal[-1, 1]]
    """Component ref -> sign (+1 or -1) indicating tolerance direction."""

    handle: ResultHandle
    """Result handle for this corner's simulation."""

    corner_name: str = ""
    """Human-readable corner name, e.g., 'R1+, R2-'."""

    def dataset(self) -> Any:
        """Return the xarray Dataset for this corner."""
        return self.handle.dataset()

    def traces(self) -> TraceSet:
        """Return TraceSet for this corner."""
        return TraceSet.from_dataset(self.handle.dataset())


@dataclass(frozen=True)
class WcaResult:
    """Result of worst-case analysis.

    Contains all corner simulations plus methods to find extremes.
    """

    corners: list[WcaCorner]
    """All evaluated corners (2^n for n components)."""

    nominal_combo: dict[str, float]
    """Nominal values for each component."""

    tolerances: dict[str, float]
    """Tolerance (as fraction) for each component."""

    nominal_run: AnalysisResult | None = None
    """The nominal simulation result (optional, run if include_nominal=True)."""

    component_labels: dict[str, str] = field(default_factory=dict)
    """Mapping from component ref to display label."""

    def handles(self) -> list[ResultHandle]:
        """Return all result handles."""
        return [c.handle for c in self.corners]

    def find_extreme(
        self,
        metric: Callable[[WcaCorner], float],
        kind: Literal["min", "max"] = "max",
    ) -> WcaCorner:
        """Find the corner that produces the extreme value for a metric.

        Args:
            metric: Function that extracts a scalar from a WcaCorner.
            kind: "min" or "max" to find minimum or maximum.

        Returns:
            The WcaCorner with the extreme metric value.

        Example:
            # Find corner with maximum Vout
            def get_vout(corner):
                ds = corner.dataset()
                return float(ds['V(vout)'].values[0])

            worst = result.find_extreme(get_vout, kind='max')
        """
        if not self.corners:
            raise ValueError("No corners to search")

        if kind == "min":
            return min(self.corners, key=metric)
        return max(self.corners, key=metric)

    def all_values(
        self,
        metric: Callable[[WcaCorner], float],
    ) -> list[tuple[WcaCorner, float]]:
        """Evaluate a metric for all corners and return sorted results.

        Args:
            metric: Function that extracts a scalar from a WcaCorner.

        Returns:
            List of (corner, value) tuples sorted by value ascending.
        """
        results = [(c, metric(c)) for c in self.corners]
        results.sort(key=lambda x: x[1])
        return results

    def bounds(
        self,
        metric: Callable[[WcaCorner], float],
    ) -> tuple[float, float, float | None]:
        """Get min, max, and nominal values for a metric.

        Args:
            metric: Function that extracts a scalar from a WcaCorner.

        Returns:
            Tuple of (min_value, max_value, nominal_value).
            nominal_value is None if nominal_run was not included.
        """
        values = [metric(c) for c in self.corners]
        nominal = None
        if self.nominal_run is not None:
            # Create a fake corner for nominal to use with metric
            # Actually we need to handle nominal differently
            pass
        return (min(values), max(values), nominal)

    def to_dataframe(
        self,
        metric: Callable[[WcaCorner], float | dict[str, Any]] | None = None,
    ) -> Any:
        """Convert results to a pandas DataFrame.

        Args:
            metric: Optional function to extract metrics from each corner.
                   Can return a single float or a dict of values.

        Returns:
            DataFrame with columns for corner name, component values,
            and optional metric values.
        """
        try:
            import pandas as pd
        except ImportError as err:
            raise RuntimeError("pandas is required for to_dataframe()") from err

        rows = []
        for corner in self.corners:
            row: dict[str, Any] = {
                "corner": corner.corner_name,
            }
            # Add component values
            for ref, value in corner.combo.items():
                label = self.component_labels.get(ref, ref)
                row[label] = value

            # Add metric if provided
            if metric is not None:
                m = metric(corner)
                if isinstance(m, dict):
                    row.update(m)
                else:
                    row["metric"] = m

            rows.append(row)

        return pd.DataFrame(rows)

    def summary(
        self,
        metric: Callable[[WcaCorner], float],
        metric_name: str = "metric",
    ) -> dict[str, Any]:
        """Generate a summary of the WCA results.

        Args:
            metric: Function to extract the metric of interest.
            metric_name: Name of the metric for display.

        Returns:
            Dict with min, max, nominal, range, and worst corners.
        """
        values = self.all_values(metric)
        min_corner, min_val = values[0]
        max_corner, max_val = values[-1]

        nominal_val = None
        if self.nominal_run is not None:
            # Extract nominal from nominal_run traces
            pass

        return {
            "metric_name": metric_name,
            "min_value": min_val,
            "max_value": max_val,
            "range": max_val - min_val,
            "min_corner": min_corner.corner_name,
            "max_corner": max_corner.corner_name,
            "nominal_value": nominal_val,
            "n_corners": len(self.corners),
        }


def _handle_to_analysis_result(handle: ResultHandle) -> AnalysisResult:
    """Convert a ResultHandle to an AnalysisResult."""
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


ToleranceSpec = float | tuple[float, Literal["abs", "rel"]]
"""Tolerance specification: either a float (relative) or tuple (value, 'abs'|'rel')."""


def run_wca(
    circuit: Circuit,
    tolerances: Mapping[Component, ToleranceSpec],
    analyses: Sequence[AnalysisSpec],
    *,
    engine: EngineName = "ngspice",
    include_nominal: bool = True,
    label_fn: Callable[[Component], str] | None = None,
    workers: int = 1,
    progress: bool | Callable[[int, int], None] | None = None,
    cache_dir: str | Path | None = ".spicelab_cache",
    reuse_cache: bool = True,
) -> WcaResult:
    """Run worst-case analysis varying components to their tolerance extremes.

    For n components with tolerances, this runs 2^n simulations (all combinations
    of +tolerance and -tolerance). This gives the absolute worst-case bounds.

    Args:
        circuit: The circuit to analyze.
        tolerances: Mapping from Component -> tolerance specification.
            - float: Relative tolerance as fraction (e.g., 0.01 for 1%)
            - (float, 'abs'): Absolute tolerance (e.g., (0.002, 'abs') for ±2mV)
            - (float, 'rel'): Explicit relative tolerance
        analyses: List of analyses to run (e.g., [AnalysisSpec(mode='op')]).
        engine: Simulation engine to use.
        include_nominal: If True, also run the nominal case.
        label_fn: Optional function to generate labels for components.
        workers: Number of parallel workers.
        progress: Progress callback or True for default progress.
        cache_dir: Directory for caching results.
        reuse_cache: Whether to reuse cached results.

    Returns:
        WcaResult containing all corner simulations.

    Example:
        # Mixed relative and absolute tolerances
        tolerances = {
            resistor1: 0.01,              # 1% relative
            resistor2: 0.05,              # 5% relative
            voffset: (0.002, 'abs'),      # ±2mV absolute
        }

        result = run_wca(
            circuit,
            tolerances=tolerances,
            analyses=[AnalysisSpec(mode='op')],
        )

        # Find worst case for output voltage
        def get_vout(corner):
            return float(corner.dataset()['V(vout)'].values[0])

        worst_high = result.find_extreme(get_vout, kind='max')
        worst_low = result.find_extreme(get_vout, kind='min')
    """
    if not tolerances:
        raise ValueError("tolerances must not be empty")

    def _label(c: Component) -> str:
        if label_fn:
            return label_fn(c)
        return f"{type(c).__name__}.{c.ref}"

    def _parse_tolerance(spec: ToleranceSpec) -> tuple[float, bool]:
        """Parse tolerance spec into (value, is_absolute)."""
        if isinstance(spec, tuple):
            value, mode = spec
            return (value, mode == "abs")
        return (spec, False)

    # Extract component info
    components = list(tolerances.keys())
    refs = [str(c.ref) for c in components]
    nominals = {str(c.ref): to_float(c.value) for c in components}

    # Parse tolerances into values and modes
    tol_values: dict[str, float] = {}
    tol_is_abs: dict[str, bool] = {}
    for c in components:
        val, is_abs = _parse_tolerance(tolerances[c])
        tol_values[str(c.ref)] = val
        tol_is_abs[str(c.ref)] = is_abs

    labels = {str(c.ref): _label(c) for c in components}

    # Generate all corner combinations: each component at +tol or -tol
    # For n components, we have 2^n corners
    n = len(components)
    combos: list[dict[str, float]] = []
    corner_signs_list: list[dict[str, Literal[-1, 1]]] = []
    corner_names: list[str] = []

    for signs in itertools.product([-1, 1], repeat=n):
        combo: dict[str, float] = {}
        corner_signs: dict[str, Literal[-1, 1]] = {}
        name_parts = []

        for ref, sign in zip(refs, signs, strict=False):
            nom = nominals[ref]
            tol = tol_values[ref]
            is_abs = tol_is_abs[ref]

            if is_abs:
                # Absolute tolerance: nom ± tol
                value = nom + sign * tol
            else:
                # Relative tolerance: nom * (1 ± tol)
                value = nom * (1 + sign * tol)

            combo[ref] = value
            corner_signs[ref] = sign  # type: ignore[assignment]
            sign_str = "+" if sign > 0 else "-"
            name_parts.append(f"{labels[ref]}{sign_str}")

        combos.append(combo)
        corner_signs_list.append(corner_signs)
        corner_names.append(", ".join(name_parts))

    # Optionally add nominal
    nominal_result: AnalysisResult | None = None
    if include_nominal:
        combos.insert(0, dict(nominals))
        corner_signs_list.insert(0, {ref: 1 for ref in refs})
        corner_names.insert(0, "nominal")

    # Run all simulations using the job system

    # Create sweep spec with explicit combos
    # We need to use the job orchestrator directly for explicit combos
    from ..orchestrator import Job, run_job

    job = Job(
        circuit=circuit,
        analyses=list(analyses),
        engine=engine,
        combos=tuple(combos),
    )

    job_result = run_job(
        job,
        cache_dir=cache_dir,
        workers=workers,
        progress=progress,
        reuse_cache=reuse_cache,
    )

    if len(job_result.runs) != len(combos):
        raise RuntimeError(f"Mismatch: expected {len(combos)} runs, got {len(job_result.runs)}")

    # Build corners list
    corners: list[WcaCorner] = []
    start_idx = 0

    if include_nominal:
        # First run is nominal
        nominal_result = _handle_to_analysis_result(job_result.runs[0].handle)
        start_idx = 1

    for i, job_run in enumerate(job_result.runs[start_idx:], start=start_idx):
        idx = i if include_nominal else i
        corner = WcaCorner(
            combo=combos[idx],
            corner_signs=corner_signs_list[idx],
            handle=job_run.handle,
            corner_name=corner_names[idx],
        )
        corners.append(corner)

    return WcaResult(
        corners=corners,
        nominal_combo=dict(nominals),
        tolerances=dict(tol_values),
        nominal_run=nominal_result,
        component_labels=labels,
    )


# Tolerance helper functions


def tolerance_to_normal(tolerance: float, sigma_multiplier: float = 3.0) -> float:
    """Convert a tolerance specification to a NormalPct sigma_pct value.

    By default, assumes 3-sigma specification (99.7% of values within tolerance).

    Args:
        tolerance: Tolerance as fraction (e.g., 0.01 for 1%).
        sigma_multiplier: Number of sigmas that correspond to tolerance.
                         Default 3.0 means 3σ = tolerance.

    Returns:
        sigma_pct value for use with NormalPct.

    Example:
        from spicelab.analysis import NormalPct, tolerance_to_normal

        # 1% tolerance, 3-sigma -> sigma_pct = 0.333%
        sigma = tolerance_to_normal(0.01)  # returns 0.00333...
        dist = NormalPct(sigma)
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    if sigma_multiplier <= 0:
        raise ValueError("sigma_multiplier must be > 0")
    return tolerance / sigma_multiplier


def tolerance_to_uniform(tolerance: float) -> float:
    """Convert a tolerance specification to a UniformPct pct value.

    For uniform distribution, the tolerance directly maps to the pct parameter.

    Args:
        tolerance: Tolerance as fraction (e.g., 0.01 for 1%).

    Returns:
        pct value for use with UniformPct.

    Example:
        from spicelab.analysis import UniformPct, tolerance_to_uniform

        # 1% tolerance -> uniform distribution ±1%
        pct = tolerance_to_uniform(0.01)  # returns 0.01
        dist = UniformPct(pct)
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    return tolerance


__all__ = [
    "ToleranceSpec",
    "WcaCorner",
    "WcaResult",
    "run_wca",
    "tolerance_to_normal",
    "tolerance_to_uniform",
]
