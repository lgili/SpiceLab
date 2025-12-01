"""E-series standard values for resistors and capacitors.

This module provides utilities for working with standard electronic component
values (E12, E24, E48, E96, E192) and custom series. It includes functions for:

- Finding nearest standard values
- Finding series/parallel combinations to match target values
- Generating value arrays for different series
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

# Standard E-series types
ESeries = Literal["E12", "E24", "E48", "E96", "E192"]

# Custom series support (e.g., manufacturer-specific values)
CustomSeriesValues = Sequence[float]


# -----------------------------------------------------------------------------
# Standard E-series base values
# -----------------------------------------------------------------------------

# E12 and E24 have well-defined standard values
E12_BASE = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
E24_BASE = [
    1.0,
    1.1,
    1.2,
    1.3,
    1.5,
    1.6,
    1.8,
    2.0,
    2.2,
    2.4,
    2.7,
    3.0,
    3.3,
    3.6,
    3.9,
    4.3,
    4.7,
    5.1,
    5.6,
    6.2,
    6.8,
    7.5,
    8.2,
    9.1,
]


def _base(series: ESeries) -> list[float]:
    """Get base values for a standard E-series."""
    if series == "E12":
        return E12_BASE.copy()
    if series == "E24":
        return E24_BASE.copy()
    # For E48, E96, E192 use geometric progression
    n = int(series[1:])
    return [round(10 ** (k / n), 3) for k in range(n)]


@lru_cache(maxsize=8)
def get_series_array(
    series: ESeries | str = "E24",
    decade_min: int = -3,
    decade_max: int = 6,
    custom_base: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Generate a numpy array of all values in a series across decades.

    Args:
        series: E-series name ("E12", "E24", "E48", "E96", "E192") or custom name
        decade_min: Starting decade (10^decade_min), default -3 (1mΩ/1pF)
        decade_max: Ending decade (10^decade_max), default 6 (1MΩ/1µF)
        custom_base: Optional tuple of custom base values (for custom series)

    Returns:
        Sorted numpy array of all values in the series
    """
    if custom_base is not None:
        base = list(custom_base)
    else:
        base = _base(series)  # type: ignore[arg-type]

    values: list[float] = []
    for d in range(decade_min, decade_max + 1):
        multiplier = 10.0**d
        values.extend(v * multiplier for v in base)

    return np.array(sorted(set(values)))


def enumerate_values(
    series: ESeries = "E24",
    decade_min: int = -3,
    decade_max: int = 6,
) -> Iterable[float]:
    """Iterate over all values in a standard E-series across decades.

    Args:
        series: E-series name
        decade_min: Starting decade
        decade_max: Ending decade

    Yields:
        Values from the series in ascending order
    """
    arr = get_series_array(series, decade_min, decade_max)
    yield from arr.tolist()


# -----------------------------------------------------------------------------
# Nearest value finding
# -----------------------------------------------------------------------------


def round_to_series(x: float, series: ESeries = "E24") -> float:
    """Find the nearest standard value in an E-series.

    Args:
        x: Target value
        series: E-series to use

    Returns:
        Nearest standard value
    """
    arr = get_series_array(series, -6, 9)
    idx = np.abs(arr - x).argmin()
    return float(arr[idx])


def nearest_value(
    target: float,
    series: ESeries | str = "E24",
    custom_values: Sequence[float] | None = None,
) -> float:
    """Find the nearest standard value to a target.

    Args:
        target: Desired value
        series: E-series name (ignored if custom_values provided)
        custom_values: Optional sequence of custom values to search

    Returns:
        Nearest value from the series
    """
    if custom_values is not None:
        arr = np.array(sorted(custom_values))
    else:
        arr = get_series_array(series, -6, 9)

    idx = np.abs(arr - target).argmin()
    return float(arr[idx])


# -----------------------------------------------------------------------------
# Combination finding
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentMatch:
    """Result of a component matching operation."""

    match_type: Literal["single", "series", "parallel"]
    value: float
    components: tuple[float, ...]
    error: float  # Relative error

    @property
    def error_pct(self) -> float:
        """Error as a percentage."""
        return self.error * 100


def _parallel_resistance(*values: float) -> float:
    """Calculate parallel combination of resistances."""
    if not values:
        return 0.0
    return 1.0 / sum(1.0 / v for v in values)


def _series_resistance(*values: float) -> float:
    """Calculate series combination of resistances."""
    return sum(values)


def find_parallel_combination(
    target: float,
    series: ESeries | str = "E24",
    max_components: int = 2,
    custom_values: Sequence[float] | None = None,
) -> tuple[float, tuple[float, ...]]:
    """Find a parallel combination of standard resistors matching target.

    For capacitors, parallel combinations add (C_total = C1 + C2 + ...).
    For resistors, 1/R_total = 1/R1 + 1/R2 + ...

    Args:
        target: Desired resistance/impedance value
        series: E-series to use
        max_components: Maximum number of components in combination
        custom_values: Optional custom values to use

    Returns:
        Tuple of (achieved_value, component_values)
    """
    if custom_values is not None:
        arr = np.array(sorted(custom_values))
    else:
        arr = get_series_array(series, -6, 9)

    # Filter to reasonable range
    arr = arr[(arr >= target * 0.5) & (arr <= target * 10)]

    best_combo: tuple[float, ...] = ()
    best_diff = float("inf")
    best_value = 0.0

    def search(depth: int, current: tuple[float, ...], start_idx: int) -> None:
        nonlocal best_combo, best_diff, best_value

        if depth == max_components:
            if current:
                combined = _parallel_resistance(*current)
                diff = abs(combined - target)
                if diff < best_diff:
                    best_diff = diff
                    best_combo = current
                    best_value = combined
            return

        for i in range(start_idx, len(arr)):
            v = float(arr[i])
            search(depth + 1, (*current, v), i)

    search(0, (), 0)
    return best_value, best_combo


def find_series_combination(
    target: float,
    series: ESeries | str = "E24",
    max_components: int = 2,
    custom_values: Sequence[float] | None = None,
) -> tuple[float, tuple[float, ...]]:
    """Find a series combination of standard resistors matching target.

    For resistors, R_total = R1 + R2 + ...
    For capacitors in series, 1/C_total = 1/C1 + 1/C2 + ...

    Args:
        target: Desired resistance value
        series: E-series to use
        max_components: Maximum number of components in combination
        custom_values: Optional custom values to use

    Returns:
        Tuple of (achieved_value, component_values)
    """
    if custom_values is not None:
        arr = np.array(sorted(custom_values))
    else:
        arr = get_series_array(series, -6, 9)

    # Filter to reasonable range
    arr = arr[(arr > 0) & (arr <= target)]

    best_combo: tuple[float, ...] = ()
    best_diff = float("inf")
    best_value = 0.0

    def search(depth: int, current: tuple[float, ...], start_idx: int) -> None:
        nonlocal best_combo, best_diff, best_value

        if depth == max_components:
            if current:
                combined = _series_resistance(*current)
                diff = abs(combined - target)
                if diff < best_diff:
                    best_diff = diff
                    best_combo = current
                    best_value = combined
            return

        current_sum = sum(current)
        for i in range(start_idx, len(arr)):
            v = float(arr[i])
            if current_sum + v > target * 1.1:  # Early termination
                break
            search(depth + 1, (*current, v), i)

    search(0, (), 0)
    return best_value, best_combo


def find_best_match(
    target: float,
    series: ESeries | str = "E24",
    max_error: float = 0.01,
    max_components: int = 2,
    custom_values: Sequence[float] | None = None,
) -> ComponentMatch:
    """Find the best matching component(s) for a target value.

    Tries single value first, then parallel and series combinations if needed.

    Args:
        target: Desired value
        series: E-series to use
        max_error: Maximum acceptable relative error (default 1%)
        max_components: Maximum components in combination
        custom_values: Optional custom values to use

    Returns:
        ComponentMatch with the best solution found
    """
    # Try single value first
    single = nearest_value(target, series, custom_values)
    single_error = abs(single - target) / target if target != 0 else 0

    result = ComponentMatch(
        match_type="single",
        value=single,
        components=(single,),
        error=single_error,
    )

    if single_error <= max_error:
        return result

    # Try parallel combination
    par_val, par_combo = find_parallel_combination(target, series, max_components, custom_values)
    if par_combo:
        par_error = abs(par_val - target) / target if target != 0 else 0
        if par_error < result.error:
            result = ComponentMatch(
                match_type="parallel",
                value=par_val,
                components=par_combo,
                error=par_error,
            )

    # Try series combination
    ser_val, ser_combo = find_series_combination(target, series, max_components, custom_values)
    if ser_combo:
        ser_error = abs(ser_val - target) / target if target != 0 else 0
        if ser_error < result.error:
            result = ComponentMatch(
                match_type="series",
                value=ser_val,
                components=ser_combo,
                error=ser_error,
            )

    return result


# -----------------------------------------------------------------------------
# Capacitor-specific helpers
# -----------------------------------------------------------------------------

# Standard capacitor series (typically E12 or E24 based)
E12_CAP_BASE = E12_BASE.copy()
E24_CAP_BASE = E24_BASE.copy()


def get_capacitor_series(
    series: Literal["E12", "E24"] = "E12",
    decade_min: int = -12,  # 1pF
    decade_max: int = -3,  # 1mF
) -> np.ndarray:
    """Generate standard capacitor values.

    Args:
        series: E12 or E24
        decade_min: Starting decade (default -12 for pF)
        decade_max: Ending decade (default -3 for mF)

    Returns:
        Array of capacitor values in Farads
    """
    base = E12_CAP_BASE if series == "E12" else E24_CAP_BASE
    return get_series_array(series, decade_min, decade_max, tuple(base))


def nearest_capacitor(
    target: float,
    series: Literal["E12", "E24"] = "E12",
) -> float:
    """Find nearest standard capacitor value.

    Args:
        target: Desired capacitance in Farads
        series: E12 or E24

    Returns:
        Nearest standard value in Farads
    """
    arr = get_capacitor_series(series)
    idx = np.abs(arr - target).argmin()
    return float(arr[idx])


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def tolerance_for_series(series: ESeries) -> float:
    """Get typical tolerance for an E-series.

    Args:
        series: E-series name

    Returns:
        Typical tolerance as a fraction (e.g., 0.05 for 5%)
    """
    tolerances = {
        "E12": 0.10,  # 10%
        "E24": 0.05,  # 5%
        "E48": 0.02,  # 2%
        "E96": 0.01,  # 1%
        "E192": 0.005,  # 0.5%
    }
    return tolerances.get(series, 0.05)


def format_value(
    value: float,
    unit: str = "Ω",
    precision: int = 3,
) -> str:
    """Format a component value with appropriate SI prefix.

    Args:
        value: Value to format
        unit: Unit symbol (default Ω for ohms)
        precision: Number of significant figures

    Returns:
        Formatted string like "4.7kΩ" or "100nF"
    """
    prefixes = [
        (1e-15, "f"),
        (1e-12, "p"),
        (1e-9, "n"),
        (1e-6, "µ"),
        (1e-3, "m"),
        (1, ""),
        (1e3, "k"),
        (1e6, "M"),
        (1e9, "G"),
    ]

    for mult, prefix in reversed(prefixes):
        if value >= mult:
            scaled = value / mult
            if scaled == int(scaled):
                return f"{int(scaled)}{prefix}{unit}"
            return f"{scaled:.{precision}g}{prefix}{unit}"

    return f"{value:.{precision}g}{unit}"
