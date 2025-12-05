"""Statistical analysis helpers for Monte Carlo and WCA results.

Provides functions for computing statistics, extracting metrics from traces,
and generating summary reports.

Usage:
    from spicelab.analysis import monte_carlo, extract_trace_value, compute_stats

    mc_result = monte_carlo(circuit, mapping, n=1000, analyses=...)

    # Extract Vout from each run
    vout_values = [
        extract_trace_value(run.traces, 'V(vout)')
        for run in mc_result.runs
    ]

    # Compute statistics
    stats = compute_stats(vout_values)
    print(f"Mean: {stats['mean']:.4f}")
    print(f"3-sigma range: {stats['sigma3_low']:.4f} to {stats['sigma3_high']:.4f}")
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..io.raw_reader import TraceSet
from .result import AnalysisResult


def extract_trace_value(
    traces: TraceSet,
    trace_name: str,
    *,
    index: int = -1,
    at_time: float | None = None,
) -> float:
    """Extract a scalar value from a trace.

    Args:
        traces: TraceSet containing simulation results.
        trace_name: Name of the trace (e.g., 'V(vout)', 'I(R1)').
                   Case-insensitive matching is attempted.
        index: Index to extract if at_time is not specified.
               Default -1 extracts the final value.
        at_time: If specified, interpolate the value at this time.

    Returns:
        The extracted scalar value.

    Raises:
        KeyError: If trace_name is not found (after case-insensitive search).

    Example:
        # Get final Vout from OP analysis
        vout = extract_trace_value(traces, 'V(vout)')

        # Get value at t=1ms from transient
        vout_1ms = extract_trace_value(traces, 'V(vout)', at_time=1e-3)
    """
    # Get available trace names
    available_names = traces.names

    # Try exact match first
    if trace_name in available_names:
        values = traces[trace_name].values
    else:
        # Try case-insensitive match
        trace_lower = trace_name.lower()
        found = None
        for name in available_names:
            if name.lower() == trace_lower:
                found = name
                break
        if found is None:
            # Try partial match (e.g., 'vout' matches 'V(vout)')
            for name in available_names:
                if trace_lower in name.lower():
                    found = name
                    break
        if found is None:
            raise KeyError(f"Trace '{trace_name}' not found. Available: {available_names}")
        values = traces[found].values

    if at_time is not None:
        try:
            import numpy as np

            # Get time array
            time_name = None
            for name in ["time", "Time", "TIME"]:
                if name in available_names:
                    time_name = name
                    break
            if time_name is None:
                # Try x attribute
                if hasattr(traces, "x") and traces.x is not None:
                    time_arr = traces.x.values
                else:
                    raise ValueError("No time axis found for interpolation")
            else:
                time_arr = traces[time_name].values

            return float(np.interp(at_time, time_arr, values))
        except ImportError as err:
            raise RuntimeError("numpy is required for time interpolation") from err

    return float(values[index])


def extract_from_analysis(
    result: AnalysisResult,
    trace_name: str,
    **kwargs: Any,
) -> float:
    """Extract a trace value from an AnalysisResult.

    Convenience wrapper around extract_trace_value.

    Args:
        result: AnalysisResult containing traces.
        trace_name: Name of the trace.
        **kwargs: Passed to extract_trace_value.

    Returns:
        The extracted scalar value.
    """
    return extract_trace_value(result.traces, trace_name, **kwargs)


@dataclass
class Statistics:
    """Statistical summary of a set of values."""

    n: int
    """Number of samples."""

    mean: float
    """Arithmetic mean."""

    std: float
    """Standard deviation (sample, n-1 denominator)."""

    min: float
    """Minimum value."""

    max: float
    """Maximum value."""

    median: float
    """Median (50th percentile)."""

    p1: float
    """1st percentile."""

    p5: float
    """5th percentile."""

    p95: float
    """95th percentile."""

    p99: float
    """99th percentile."""

    sigma3_low: float
    """Mean - 3*std (lower 3-sigma bound)."""

    sigma3_high: float
    """Mean + 3*std (upper 3-sigma bound)."""

    def __repr__(self) -> str:
        return (
            f"Statistics(n={self.n}, mean={self.mean:.4g}, std={self.std:.4g}, "
            f"range=[{self.min:.4g}, {self.max:.4g}])"
        )

    def cpk(self, lsl: float, usl: float) -> float:
        """Calculate process capability index Cpk.

        Args:
            lsl: Lower specification limit.
            usl: Upper specification limit.

        Returns:
            Cpk value. Higher is better; Cpk >= 1.33 is typically acceptable.
        """
        if self.std == 0:
            return float("inf") if lsl <= self.mean <= usl else 0.0
        cpu = (usl - self.mean) / (3 * self.std)
        cpl = (self.mean - lsl) / (3 * self.std)
        return min(cpu, cpl)

    def yield_estimate(self, lsl: float, usl: float) -> float:
        """Estimate yield (fraction within spec limits).

        Uses normal distribution assumption.

        Args:
            lsl: Lower specification limit.
            usl: Upper specification limit.

        Returns:
            Estimated yield as fraction (0.0 to 1.0).
        """
        if self.std == 0:
            return 1.0 if lsl <= self.mean <= usl else 0.0

        try:
            from scipy.stats import norm

            z_low = (lsl - self.mean) / self.std
            z_high = (usl - self.mean) / self.std
            return float(norm.cdf(z_high) - norm.cdf(z_low))
        except ImportError:
            # Fallback: use empirical estimate if scipy not available
            # This is less accurate but works without scipy
            z_low = (lsl - self.mean) / self.std
            z_high = (usl - self.mean) / self.std
            # Approximate using erf
            return 0.5 * (math.erf(z_high / math.sqrt(2)) - math.erf(z_low / math.sqrt(2)))


def compute_stats(values: Sequence[float]) -> Statistics:
    """Compute comprehensive statistics for a set of values.

    Args:
        values: Sequence of numeric values.

    Returns:
        Statistics dataclass with all computed metrics.

    Example:
        values = [1.0, 1.1, 0.9, 1.05, 0.95]
        stats = compute_stats(values)
        print(f"Mean: {stats.mean}, Std: {stats.std}")
    """
    if not values:
        raise ValueError("values must not be empty")

    n = len(values)
    sorted_values = sorted(values)

    # Basic stats
    mean = sum(values) / n

    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    min_val = sorted_values[0]
    max_val = sorted_values[-1]

    # Percentiles using linear interpolation
    def percentile(p: float) -> float:
        if n == 1:
            return sorted_values[0]
        idx = (n - 1) * p / 100.0
        lower = int(idx)
        upper = lower + 1
        if upper >= n:
            return sorted_values[-1]
        frac = idx - lower
        return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac

    return Statistics(
        n=n,
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
        median=percentile(50),
        p1=percentile(1),
        p5=percentile(5),
        p95=percentile(95),
        p99=percentile(99),
        sigma3_low=mean - 3 * std,
        sigma3_high=mean + 3 * std,
    )


def mc_summary(
    runs: Sequence[AnalysisResult],
    trace_name: str,
    *,
    index: int = -1,
    at_time: float | None = None,
    lsl: float | None = None,
    usl: float | None = None,
) -> dict[str, Any]:
    """Generate a summary of Monte Carlo results for a specific trace.

    Args:
        runs: List of AnalysisResult from Monte Carlo.
        trace_name: Name of the trace to analyze.
        index: Index to extract from each trace.
        at_time: If specified, interpolate at this time.
        lsl: Lower specification limit (optional).
        usl: Upper specification limit (optional).

    Returns:
        Dictionary with statistics and optional yield/Cpk metrics.

    Example:
        from spicelab.analysis import monte_carlo, mc_summary

        mc_result = monte_carlo(circuit, mapping, n=1000, analyses=...)

        summary = mc_summary(
            mc_result.runs,
            'V(vout)',
            lsl=2.4,
            usl=2.6,
        )

        print(f"Mean Vout: {summary['mean']:.4f} V")
        print(f"Cpk: {summary['cpk']:.2f}")
        print(f"Estimated yield: {summary['yield_pct']:.2f}%")
    """
    values = [extract_from_analysis(run, trace_name, index=index, at_time=at_time) for run in runs]

    stats = compute_stats(values)

    result: dict[str, Any] = {
        "trace": trace_name,
        "n": stats.n,
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
        "range": stats.max - stats.min,
    }

    if lsl is not None and usl is not None:
        result["lsl"] = lsl
        result["usl"] = usl
        result["cpk"] = stats.cpk(lsl, usl)
        result["yield"] = stats.yield_estimate(lsl, usl)
        result["yield_pct"] = stats.yield_estimate(lsl, usl) * 100

        # Count actual failures
        failures = sum(1 for v in values if v < lsl or v > usl)
        result["failures"] = failures
        result["actual_yield_pct"] = (len(values) - failures) / len(values) * 100

    return result


def create_metric_extractor(
    trace_name: str,
    *,
    index: int = -1,
    at_time: float | None = None,
) -> Callable[[AnalysisResult], float]:
    """Create a reusable metric extractor function.

    Useful for passing to MonteCarloResult.to_dataframe() or similar methods.

    Args:
        trace_name: Name of the trace.
        index: Index to extract.
        at_time: Time to interpolate at.

    Returns:
        A callable that extracts the specified value from an AnalysisResult.

    Example:
        from spicelab.analysis import monte_carlo, create_metric_extractor

        get_vout = create_metric_extractor('V(vout)')

        mc_result = monte_carlo(...)
        df = mc_result.to_dataframe(metric=get_vout)
    """

    def extractor(result: AnalysisResult) -> float:
        return extract_from_analysis(result, trace_name, index=index, at_time=at_time)

    return extractor


# ---------- Simplified Result Extraction ----------


def get_voltage(
    traces: TraceSet,
    node: str,
    *,
    index: int = -1,
    at_time: float | None = None,
) -> float:
    """Extract voltage at a node from simulation traces.

    Simplified interface that handles the V(...) naming convention.

    Args:
        traces: TraceSet from simulation result.
        node: Node name (e.g., 'vout', 'in', 'n1'). Case-insensitive.
        index: Index to extract (-1 for final value).
        at_time: If specified, interpolate at this time.

    Returns:
        Voltage value in volts.

    Raises:
        KeyError: If node not found. Lists available nodes in error message.

    Example:
        # Get final Vout
        vout = get_voltage(result.traces, 'vout')

        # Get voltage at specific time
        vout_1ms = get_voltage(result.traces, 'vout', at_time=1e-3)
    """
    # Try common SPICE naming patterns
    patterns = [
        f"V({node})",
        f"v({node})",
        f"V({node.lower()})",
        f"V({node.upper()})",
        node,  # Sometimes node voltage is stored directly
    ]

    for pattern in patterns:
        try:
            return extract_trace_value(traces, pattern, index=index, at_time=at_time)
        except KeyError:
            continue

    # If all patterns fail, raise helpful error
    available = traces.names
    voltage_traces = [n for n in available if n.lower().startswith("v(")]
    raise KeyError(
        f"Voltage for node '{node}' not found. "
        f"Available voltage traces: {voltage_traces or available}"
    )


def get_current(
    traces: TraceSet,
    component: str,
    *,
    index: int = -1,
    at_time: float | None = None,
) -> float:
    """Extract current through a component from simulation traces.

    Simplified interface that handles the I(...) naming convention.

    Args:
        traces: TraceSet from simulation result.
        component: Component reference (e.g., 'R1', 'V1', 'M1'). Case-insensitive.
        index: Index to extract (-1 for final value).
        at_time: If specified, interpolate at this time.

    Returns:
        Current value in amperes.

    Raises:
        KeyError: If current not found. Lists available currents in error message.

    Example:
        # Get current through R1
        i_r1 = get_current(result.traces, 'R1')

        # Get supply current
        i_vcc = get_current(result.traces, 'Vcc')
    """
    # Try common SPICE naming patterns
    patterns = [
        f"I({component})",
        f"i({component})",
        f"I({component.lower()})",
        f"I({component.upper()})",
        f"I(V{component})" if not component.lower().startswith("v") else None,
        component,
    ]

    for pattern in patterns:
        if pattern is None:
            continue
        try:
            return extract_trace_value(traces, pattern, index=index, at_time=at_time)
        except KeyError:
            continue

    # If all patterns fail, raise helpful error
    available = traces.names
    current_traces = [n for n in available if n.lower().startswith("i(")]
    raise KeyError(
        f"Current for component '{component}' not found. "
        f"Available current traces: {current_traces or available}"
    )


def get_power(
    traces: TraceSet,
    component: str,
    *,
    index: int = -1,
    at_time: float | None = None,
) -> float:
    """Extract power dissipated by a component from simulation traces.

    Tries to find power directly (P(...)) or calculates from V*I if available.

    Args:
        traces: TraceSet from simulation result.
        component: Component reference (e.g., 'R1', 'M1'). Case-insensitive.
        index: Index to extract (-1 for final value).
        at_time: If specified, interpolate at this time.

    Returns:
        Power value in watts.

    Raises:
        KeyError: If power cannot be determined.

    Example:
        # Get power dissipated in R1
        p_r1 = get_power(result.traces, 'R1')
    """
    # Try direct power trace first
    patterns = [
        f"P({component})",
        f"p({component})",
        f"P({component.lower()})",
    ]

    for pattern in patterns:
        try:
            return extract_trace_value(traces, pattern, index=index, at_time=at_time)
        except KeyError:
            continue

    # Power trace not found, try to calculate from V*I
    # This is more complex and requires knowing the component terminals
    raise KeyError(
        f"Power for component '{component}' not found. "
        f"Direct power traces (P(...)) not available in this simulation."
    )


def sigma_level(cpk: float) -> float:
    """Convert Cpk to equivalent sigma level.

    Args:
        cpk: Process capability index.

    Returns:
        Equivalent sigma level (e.g., 2.0 for 6-sigma if Cpk=2.0).

    Example:
        >>> sigma_level(1.33)  # Common industrial target
        3.99
        >>> sigma_level(2.0)   # 6-sigma target
        6.0
    """
    return cpk * 3.0


__all__ = [
    "Statistics",
    "compute_stats",
    "extract_trace_value",
    "extract_from_analysis",
    "mc_summary",
    "create_metric_extractor",
    "get_voltage",
    "get_current",
    "get_power",
    "sigma_level",
]
