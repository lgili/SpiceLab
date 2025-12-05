"""Temperature and condition sweep integration for Monte Carlo and WCA.

Provides functions for running tolerance analyses across multiple conditions
(e.g., temperature sweep) and collecting results in a structured format.

Usage:
    from spicelab.analysis import monte_carlo_sweep, wca_sweep

    # Monte Carlo across temperatures
    result = monte_carlo_sweep(
        circuit,
        mapping={R1: NormalPct(0.01), R2: NormalPct(0.01)},
        conditions={'temp': [-40, 25, 85]},
        n=1000,
        analyses=[AnalysisSpec(mode='op')],
    )

    # Access results by condition
    for cond_result in result.condition_results:
        print(f"Temp {cond_result.condition}: mean={cond_result.stats.mean}")

    # Export to DataFrame with multi-index
    df = result.to_dataframe(metric=lambda r: r.traces['V(vout)'].values[-1])
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.circuit import Circuit
from ..core.components import Component
from ..core.types import AnalysisSpec
from ..engines import EngineName
from .montecarlo import (
    CorrelatedGroup,
    Dist,
    MonteCarloResult,
    monte_carlo,
)
from .result import AnalysisResult
from .stats import Statistics, compute_stats
from .wca import ToleranceSpec, WcaResult, run_wca


@dataclass(frozen=True)
class ConditionResult:
    """Result for a single condition in a sweep."""

    condition: dict[str, Any]
    """The condition values (e.g., {'temp': 25})."""

    condition_label: str
    """Human-readable label for this condition."""


@dataclass(frozen=True)
class MonteCarloConditionResult(ConditionResult):
    """Monte Carlo result for a single condition."""

    mc_result: MonteCarloResult
    """The full Monte Carlo result for this condition."""

    stats: Statistics | None = None
    """Optional statistics computed for a metric."""


@dataclass(frozen=True)
class WcaConditionResult(ConditionResult):
    """WCA result for a single condition."""

    wca_result: WcaResult
    """The full WCA result for this condition."""


@dataclass(frozen=True)
class MonteCarloSweepResult:
    """Result of Monte Carlo analysis across multiple conditions.

    Contains all condition results plus methods for aggregation and export.
    """

    condition_results: list[MonteCarloConditionResult]
    """Results for each condition."""

    conditions: dict[str, Sequence[Any]]
    """The conditions that were swept."""

    n: int
    """Number of Monte Carlo iterations per condition."""

    def __len__(self) -> int:
        return len(self.condition_results)

    def __getitem__(self, idx: int) -> MonteCarloConditionResult:
        return self.condition_results[idx]

    def __iter__(self):
        return iter(self.condition_results)

    def get_by_condition(self, **kwargs: Any) -> MonteCarloConditionResult | None:
        """Find a result by condition values.

        Args:
            **kwargs: Condition key-value pairs to match.

        Returns:
            The matching ConditionResult or None if not found.
        """
        for cr in self.condition_results:
            if all(cr.condition.get(k) == v for k, v in kwargs.items()):
                return cr
        return None

    def to_dataframe(
        self,
        metric: Callable[[AnalysisResult], float] | None = None,
        *,
        include_params: bool = True,
    ) -> Any:
        """Convert all results to a DataFrame with multi-index.

        Args:
            metric: Optional function to extract a metric from each run.
            include_params: Whether to include sampled parameter columns.

        Returns:
            DataFrame with (condition, trial) as index and metric/param columns.
        """
        try:
            import pandas as pd
        except ImportError as err:
            raise RuntimeError("pandas is required for to_dataframe()") from err

        rows = []
        for cr in self.condition_results:
            mc = cr.mc_result
            for trial_idx, (sample, run) in enumerate(zip(mc.samples, mc.runs, strict=False)):
                row: dict[str, Any] = {
                    **cr.condition,
                    "trial": trial_idx,
                }
                if include_params:
                    row.update(sample)
                if metric is not None:
                    row["metric"] = metric(run)
                rows.append(row)

        df = pd.DataFrame(rows)

        # Create multi-index from condition columns
        condition_cols = list(self.conditions.keys())
        if condition_cols:
            df = df.set_index(condition_cols + ["trial"])

        return df

    def to_csv(
        self,
        path: str | Path,
        metric: Callable[[AnalysisResult], float] | None = None,
        *,
        include_params: bool = True,
        **kwargs: Any,
    ) -> None:
        """Export results to CSV.

        Args:
            path: Output file path.
            metric: Optional metric extractor.
            include_params: Whether to include parameter columns.
            **kwargs: Additional arguments passed to DataFrame.to_csv().
        """
        df = self.to_dataframe(metric=metric, include_params=include_params)
        df.to_csv(path, **kwargs)

    def summary_dataframe(
        self,
        metric: Callable[[AnalysisResult], float],
        *,
        lsl: float | None = None,
        usl: float | None = None,
    ) -> Any:
        """Create a summary DataFrame with statistics per condition.

        Args:
            metric: Function to extract the metric of interest.
            lsl: Lower specification limit (optional).
            usl: Upper specification limit (optional).

        Returns:
            DataFrame with one row per condition containing statistics.
        """
        try:
            import pandas as pd
        except ImportError as err:
            raise RuntimeError("pandas is required for summary_dataframe()") from err

        rows = []
        for cr in self.condition_results:
            values = [metric(run) for run in cr.mc_result.runs]
            stats = compute_stats(values)

            row: dict[str, Any] = {
                **cr.condition,
                "n": stats.n,
                "mean": stats.mean,
                "std": stats.std,
                "min": stats.min,
                "max": stats.max,
                "median": stats.median,
                "p5": stats.p5,
                "p95": stats.p95,
                "sigma3_low": stats.sigma3_low,
                "sigma3_high": stats.sigma3_high,
            }

            if lsl is not None and usl is not None:
                row["cpk"] = stats.cpk(lsl, usl)
                row["yield_pct"] = stats.yield_estimate(lsl, usl) * 100

            rows.append(row)

        return pd.DataFrame(rows)


@dataclass(frozen=True)
class WcaSweepResult:
    """Result of WCA across multiple conditions."""

    condition_results: list[WcaConditionResult]
    """Results for each condition."""

    conditions: dict[str, Sequence[Any]]
    """The conditions that were swept."""

    def __len__(self) -> int:
        return len(self.condition_results)

    def __getitem__(self, idx: int) -> WcaConditionResult:
        return self.condition_results[idx]

    def __iter__(self):
        return iter(self.condition_results)

    def get_by_condition(self, **kwargs: Any) -> WcaConditionResult | None:
        """Find a result by condition values."""
        for cr in self.condition_results:
            if all(cr.condition.get(k) == v for k, v in kwargs.items()):
                return cr
        return None

    def bounds_dataframe(
        self,
        metric: Callable[[Any], float],
        metric_name: str = "metric",
    ) -> Any:
        """Create a DataFrame showing min/max bounds per condition.

        Args:
            metric: Function to extract metric from WcaCorner.
            metric_name: Name for the metric column.

        Returns:
            DataFrame with min/max bounds for each condition.
        """
        try:
            import pandas as pd
        except ImportError as err:
            raise RuntimeError("pandas is required for bounds_dataframe()") from err

        rows = []
        for cr in self.condition_results:
            wca = cr.wca_result
            values = wca.all_values(metric)
            min_corner, min_val = values[0]
            max_corner, max_val = values[-1]

            row: dict[str, Any] = {
                **cr.condition,
                f"{metric_name}_min": min_val,
                f"{metric_name}_max": max_val,
                f"{metric_name}_range": max_val - min_val,
                "min_corner": min_corner.corner_name,
                "max_corner": max_corner.corner_name,
            }
            rows.append(row)

        return pd.DataFrame(rows)


def monte_carlo_sweep(
    circuit: Circuit,
    mapping: Mapping[Component | CorrelatedGroup, Dist | None],
    conditions: dict[str, Sequence[Any]],
    n: int,
    analyses: Sequence[AnalysisSpec],
    *,
    condition_param: str | None = None,
    engine: EngineName = "ngspice",
    seed: int | None = None,
    workers: int = 1,
    progress: bool | Callable[[int, int], None] | None = None,
    cache_dir: str | Path | None = ".spicelab_cache",
    reuse_cache: bool = True,
    label_fn: Callable[[Component], str] | None = None,
) -> MonteCarloSweepResult:
    """Run Monte Carlo analysis across multiple conditions.

    This is useful for analyzing circuit behavior across temperature,
    supply voltage, or other environmental conditions.

    Args:
        circuit: The circuit to analyze.
        mapping: Component to distribution mapping (same as monte_carlo).
        conditions: Dict of condition name to list of values.
            Example: {'temp': [-40, 25, 85], 'vcc': [4.5, 5.0, 5.5]}
        n: Number of Monte Carlo iterations per condition.
        analyses: List of analyses to run.
        condition_param: If specified, the circuit parameter to modify.
            If None, conditions are stored as metadata only.
        engine: Simulation engine.
        seed: Random seed for reproducibility.
        workers: Number of parallel workers.
        progress: Progress callback.
        cache_dir: Cache directory.
        reuse_cache: Whether to use cached results.
        label_fn: Custom label function for components.

    Returns:
        MonteCarloSweepResult containing results for all conditions.

    Example:
        result = monte_carlo_sweep(
            circuit,
            mapping={R1: NormalPct(0.01)},
            conditions={'temp': [-40, 25, 85]},
            n=1000,
            analyses=[AnalysisSpec(mode='op')],
        )

        # Get summary across temperatures
        summary = result.summary_dataframe(
            metric=lambda r: r.traces['V(vout)'].values[-1],
            lsl=2.4, usl=2.6,
        )
    """
    import itertools

    # Generate all condition combinations
    condition_keys = list(conditions.keys())
    condition_values = [conditions[k] for k in condition_keys]
    all_combos = list(itertools.product(*condition_values))

    results: list[MonteCarloConditionResult] = []

    # Use different seed for each condition to avoid correlation
    base_seed = seed if seed is not None else 42

    for combo_idx, combo in enumerate(all_combos):
        condition = dict(zip(condition_keys, combo, strict=False))
        condition_label = ", ".join(f"{k}={v}" for k, v in condition.items())

        # Modify circuit for this condition if condition_param is specified
        # For now, we assume conditions are handled via .PARAM or similar
        # The circuit modification would need to be done by the user
        # or through a more sophisticated mechanism

        # Update analyses with condition (e.g., .OPTIONS TEMP=...)
        modified_analyses = _apply_conditions_to_analyses(analyses, condition)

        # Run Monte Carlo with condition-specific seed
        condition_seed = base_seed + combo_idx * n if seed is not None else None

        mc_result = monte_carlo(
            circuit,
            mapping,
            n=n,
            seed=condition_seed,
            label_fn=label_fn,
            workers=workers,
            progress=progress,
            analyses=modified_analyses,
            engine=engine,
            cache_dir=cache_dir,
            reuse_cache=reuse_cache,
        )

        results.append(
            MonteCarloConditionResult(
                condition=condition,
                condition_label=condition_label,
                mc_result=mc_result,
            )
        )

    return MonteCarloSweepResult(
        condition_results=results,
        conditions=conditions,
        n=n,
    )


def wca_sweep(
    circuit: Circuit,
    tolerances: Mapping[Component, ToleranceSpec],
    conditions: dict[str, Sequence[Any]],
    analyses: Sequence[AnalysisSpec],
    *,
    engine: EngineName = "ngspice",
    include_nominal: bool = True,
    label_fn: Callable[[Component], str] | None = None,
    workers: int = 1,
    progress: bool | Callable[[int, int], None] | None = None,
    cache_dir: str | Path | None = ".spicelab_cache",
    reuse_cache: bool = True,
) -> WcaSweepResult:
    """Run Worst-Case Analysis across multiple conditions.

    Args:
        circuit: The circuit to analyze.
        tolerances: Component to tolerance mapping (same as run_wca).
        conditions: Dict of condition name to list of values.
        analyses: List of analyses to run.
        engine: Simulation engine.
        include_nominal: Whether to include nominal simulation.
        label_fn: Custom label function for components.
        workers: Number of parallel workers.
        progress: Progress callback.
        cache_dir: Cache directory.
        reuse_cache: Whether to use cached results.

    Returns:
        WcaSweepResult containing WCA results for all conditions.

    Example:
        result = wca_sweep(
            circuit,
            tolerances={R1: 0.01, Voff: (0.002, 'abs')},
            conditions={'temp': [-40, 25, 85]},
            analyses=[AnalysisSpec(mode='op')],
        )

        # Get bounds across temperatures
        bounds = result.bounds_dataframe(
            metric=lambda corner: float(corner.dataset()['V(vout)'].values[-1]),
            metric_name='Vout',
        )
    """
    import itertools

    condition_keys = list(conditions.keys())
    condition_values = [conditions[k] for k in condition_keys]
    all_combos = list(itertools.product(*condition_values))

    results: list[WcaConditionResult] = []

    for combo in all_combos:
        condition = dict(zip(condition_keys, combo, strict=False))
        condition_label = ", ".join(f"{k}={v}" for k, v in condition.items())

        modified_analyses = _apply_conditions_to_analyses(analyses, condition)

        wca_result = run_wca(
            circuit,
            tolerances,
            modified_analyses,
            engine=engine,
            include_nominal=include_nominal,
            label_fn=label_fn,
            workers=workers,
            progress=progress,
            cache_dir=cache_dir,
            reuse_cache=reuse_cache,
        )

        results.append(
            WcaConditionResult(
                condition=condition,
                condition_label=condition_label,
                wca_result=wca_result,
            )
        )

    return WcaSweepResult(
        condition_results=results,
        conditions=conditions,
    )


def _apply_conditions_to_analyses(
    analyses: Sequence[AnalysisSpec],
    conditions: dict[str, Any],
) -> list[AnalysisSpec]:
    """Apply condition values to analyses.

    For temperature, this sets the TEMP option in SPICE via the args dict.
    Other conditions may require circuit modification.
    """
    modified = []
    for spec in analyses:
        # Create a copy of the args dict
        new_args = dict(spec.args) if spec.args else {}

        # Handle temperature condition - add to args
        if "temp" in conditions:
            new_args["temp"] = conditions["temp"]

        # Create new AnalysisSpec with modified args
        new_spec = AnalysisSpec(
            mode=spec.mode,
            args=new_args,
        )

        modified.append(new_spec)

    return modified


__all__ = [
    "ConditionResult",
    "MonteCarloConditionResult",
    "WcaConditionResult",
    "MonteCarloSweepResult",
    "WcaSweepResult",
    "monte_carlo_sweep",
    "wca_sweep",
]
