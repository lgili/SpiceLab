"""Corner Analysis & PVT for SpiceLab.

This module provides corner and PVT (Process-Voltage-Temperature) analysis:
- Corner generation (all combinations of parameter extremes)
- PVT sweep framework
- Statistical corner analysis (Monte Carlo)
- Worst-case analysis
- Yield estimation

Example:
    >>> from spicelab.optimization.corner import (
    ...     CornerDefinition, CornerAnalysis, PVTSweep
    ... )
    >>>
    >>> # Define corners
    >>> corners = CornerDefinition()
    >>> corners.add_parameter("process", ["slow", "typical", "fast"])
    >>> corners.add_parameter("voltage", [3.0, 3.3, 3.6])
    >>> corners.add_parameter("temperature", [-40, 25, 85])
    >>>
    >>> # Generate all corners
    >>> all_corners = corners.generate_all()  # 27 combinations
    >>>
    >>> # Run analysis
    >>> analysis = CornerAnalysis(circuit, simulate_func)
    >>> results = analysis.run(corners, objective_func)
    >>> print(results.worst_case)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Sequence

import numpy as np

from .base import ParameterBounds


@dataclass
class Corner:
    """A single corner configuration.

    Attributes:
        name: Human-readable corner name (e.g., "slow_3.0V_-40C")
        parameters: Dictionary of parameter name -> value
        metadata: Optional metadata about this corner
    """

    name: str
    parameters: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"Corner({self.name}: {params})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "parameters": self.parameters.copy(),
            "metadata": self.metadata.copy(),
        }


@dataclass
class CornerResult:
    """Result from evaluating a single corner.

    Attributes:
        corner: The corner that was evaluated
        value: The objective function value
        success: Whether simulation succeeded
        dataset: Optional simulation dataset
        metrics: Optional additional metrics
    """

    corner: Corner
    value: float
    success: bool = True
    dataset: Any = None
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "corner": self.corner.to_dict(),
            "value": self.value,
            "success": self.success,
            "metrics": self.metrics.copy(),
        }


class CornerDefinition:
    """Defines the parameter space for corner analysis.

    This class allows defining discrete values for each parameter
    and generates all combinations (corners).

    Example:
        >>> corners = CornerDefinition()
        >>> corners.add_parameter("process", ["slow", "typical", "fast"])
        >>> corners.add_parameter("vdd", [3.0, 3.3, 3.6])
        >>> all_corners = corners.generate_all()  # 9 corners
    """

    def __init__(self) -> None:
        """Initialize empty corner definition."""
        self._parameters: dict[str, list[Any]] = {}
        self._labels: dict[str, dict[Any, str]] = {}

    def add_parameter(
        self,
        name: str,
        values: Sequence[Any],
        labels: dict[Any, str] | None = None,
    ) -> "CornerDefinition":
        """Add a parameter with discrete values.

        Args:
            name: Parameter name
            values: List of discrete values
            labels: Optional labels for values (used in corner names)

        Returns:
            Self for method chaining
        """
        self._parameters[name] = list(values)
        if labels:
            self._labels[name] = labels
        return self

    def add_pvt(
        self,
        process: Sequence[str] = ("slow", "typical", "fast"),
        voltage: Sequence[float] | None = None,
        temperature: Sequence[float] = (-40, 25, 85),
    ) -> "CornerDefinition":
        """Add standard PVT parameters.

        Args:
            process: Process corners (default: slow/typical/fast)
            voltage: Voltage values (default: None, must be specified)
            temperature: Temperature values in Celsius (default: -40/25/85)

        Returns:
            Self for method chaining
        """
        self.add_parameter("process", process)
        if voltage is not None:
            self.add_parameter("voltage", voltage)
        self.add_parameter("temperature", temperature)
        return self

    @property
    def parameter_names(self) -> list[str]:
        """Get list of parameter names."""
        return list(self._parameters.keys())

    @property
    def n_parameters(self) -> int:
        """Get number of parameters."""
        return len(self._parameters)

    @property
    def n_corners(self) -> int:
        """Get total number of corners (product of all values)."""
        if not self._parameters:
            return 0
        n = 1
        for values in self._parameters.values():
            n *= len(values)
        return n

    def _make_corner_name(self, params: dict[str, Any]) -> str:
        """Generate corner name from parameters."""
        parts = []
        for name, value in params.items():
            if name in self._labels and value in self._labels[name]:
                parts.append(self._labels[name][value])
            elif isinstance(value, float):
                parts.append(f"{value:.2g}")
            else:
                parts.append(str(value))
        return "_".join(parts)

    def generate_all(self) -> list[Corner]:
        """Generate all corner combinations.

        Returns:
            List of all corners (Cartesian product of all parameter values)
        """
        if not self._parameters:
            return []

        names = list(self._parameters.keys())
        value_lists = [self._parameters[n] for n in names]

        corners = []
        for combo in product(*value_lists):
            params = dict(zip(names, combo))
            name = self._make_corner_name(params)
            corners.append(Corner(name=name, parameters=params))

        return corners

    def generate_extremes(self) -> list[Corner]:
        """Generate only extreme corners (min/max of each parameter).

        For n parameters with min/max values, generates 2^n corners.

        Returns:
            List of extreme corners
        """
        if not self._parameters:
            return []

        names = list(self._parameters.keys())
        extremes = []
        for name in names:
            values = self._parameters[name]
            if len(values) >= 2:
                extremes.append([values[0], values[-1]])
            else:
                extremes.append(values)

        corners = []
        for combo in product(*extremes):
            params = dict(zip(names, combo))
            name = self._make_corner_name(params)
            corners.append(Corner(name=name, parameters=params))

        return corners

    def generate_typical(self) -> Corner:
        """Generate the typical corner (middle value of each parameter).

        Returns:
            The typical corner
        """
        params = {}
        for name, values in self._parameters.items():
            mid_idx = len(values) // 2
            params[name] = values[mid_idx]

        name = self._make_corner_name(params)
        return Corner(name=name, parameters=params, metadata={"type": "typical"})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameters": {k: list(v) for k, v in self._parameters.items()},
            "labels": self._labels.copy(),
            "n_corners": self.n_corners,
        }


@dataclass
class CornerAnalysisResult:
    """Results from corner analysis.

    Attributes:
        results: List of individual corner results
        definition: The corner definition used
        n_corners: Total number of corners evaluated
        n_success: Number of successful evaluations
    """

    results: list[CornerResult]
    definition: CornerDefinition
    n_corners: int = 0
    n_success: int = 0

    def __post_init__(self) -> None:
        """Calculate statistics."""
        self.n_corners = len(self.results)
        self.n_success = sum(1 for r in self.results if r.success)

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.n_corners == 0:
            return 0.0
        return self.n_success / self.n_corners

    @property
    def values(self) -> list[float]:
        """Get all objective values."""
        return [r.value for r in self.results if r.success]

    @property
    def worst_case(self) -> CornerResult | None:
        """Get worst-case corner (maximum value).

        Returns:
            Corner with maximum objective value, or None if no results
        """
        successful = [r for r in self.results if r.success]
        if not successful:
            return None
        return max(successful, key=lambda r: r.value)

    @property
    def best_case(self) -> CornerResult | None:
        """Get best-case corner (minimum value).

        Returns:
            Corner with minimum objective value, or None if no results
        """
        successful = [r for r in self.results if r.success]
        if not successful:
            return None
        return min(successful, key=lambda r: r.value)

    def get_worst_case(self, maximize: bool = False) -> CornerResult | None:
        """Get worst-case corner.

        Args:
            maximize: If True, worst = maximum; if False, worst = minimum

        Returns:
            Worst-case corner result
        """
        if maximize:
            return self.best_case
        return self.worst_case

    def get_statistics(self) -> dict[str, float]:
        """Get statistical summary of results.

        Returns:
            Dictionary with min, max, mean, std, range
        """
        if not self.values:
            return {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "range": float("nan"),
            }

        arr = np.array(self.values)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "range": float(np.max(arr) - np.min(arr)),
        }

    def get_by_parameter(
        self, param_name: str, param_value: Any
    ) -> list[CornerResult]:
        """Get all results with a specific parameter value.

        Args:
            param_name: Parameter name
            param_value: Parameter value to filter by

        Returns:
            List of matching corner results
        """
        return [
            r
            for r in self.results
            if r.corner.parameters.get(param_name) == param_value
        ]

    def sensitivity_to_parameter(self, param_name: str) -> dict[Any, float]:
        """Calculate average value for each level of a parameter.

        Args:
            param_name: Parameter to analyze

        Returns:
            Dictionary mapping parameter value to average objective
        """
        if param_name not in self.definition.parameter_names:
            raise ValueError(f"Unknown parameter: {param_name}")

        result = {}
        for value in self.definition._parameters[param_name]:
            matching = self.get_by_parameter(param_name, value)
            values = [r.value for r in matching if r.success]
            if values:
                result[value] = float(np.mean(values))

        return result

    def to_dataframe(self) -> Any:
        """Convert results to pandas DataFrame (if available).

        Returns:
            DataFrame with corner parameters and values
        """
        try:
            import pandas as pd

            rows = []
            for r in self.results:
                row = r.corner.parameters.copy()
                row["corner_name"] = r.corner.name
                row["value"] = r.value
                row["success"] = r.success
                row.update(r.metrics)
                rows.append(row)

            return pd.DataFrame(rows)
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "n_corners": self.n_corners,
            "n_success": self.n_success,
            "statistics": self.get_statistics(),
        }


class CornerAnalysis:
    """Run corner analysis on a circuit.

    This class evaluates a circuit across multiple corners and
    collects the results.

    Example:
        >>> analysis = CornerAnalysis(circuit, simulate_func)
        >>> results = analysis.run(corners, objective_func)
        >>> print(f"Worst case: {results.worst_case}")
    """

    def __init__(
        self,
        circuit: Any,
        simulate: Callable[[Any], Any],
        apply_corner: Callable[[Any, Corner], Any] | None = None,
    ) -> None:
        """Initialize corner analysis.

        Args:
            circuit: The circuit to analyze
            simulate: Function to simulate the circuit
            apply_corner: Optional function to apply corner to circuit.
                         Signature: apply_corner(circuit, corner) -> modified_circuit
                         If None, corner parameters are set as circuit parameters.
        """
        self.circuit = circuit
        self.simulate = simulate
        self.apply_corner = apply_corner
        self._results: list[CornerResult] = []

    def _default_apply_corner(self, circuit: Any, corner: Corner) -> Any:
        """Default corner application - set circuit parameters."""
        modified = circuit.copy() if hasattr(circuit, "copy") else circuit
        for name, value in corner.parameters.items():
            if hasattr(modified, "set_parameter"):
                modified.set_parameter(name, value)
        return modified

    def run(
        self,
        definition: CornerDefinition,
        objective: Callable[[dict[str, Any], Any], float],
        corners: list[Corner] | None = None,
        verbose: bool = False,
    ) -> CornerAnalysisResult:
        """Run corner analysis.

        Args:
            definition: Corner definition
            objective: Function to evaluate results. Signature:
                      objective(params, dataset) -> float
            corners: Optional specific corners to run. If None, runs all.
            verbose: Print progress

        Returns:
            CornerAnalysisResult with all results
        """
        if corners is None:
            corners = definition.generate_all()

        self._results = []
        apply_fn = self.apply_corner or self._default_apply_corner

        for i, corner in enumerate(corners):
            if verbose:
                print(f"[{i + 1}/{len(corners)}] {corner.name}")

            try:
                modified_circuit = apply_fn(self.circuit, corner)
                dataset = self.simulate(modified_circuit)
                value = objective(corner.parameters, dataset)
                result = CornerResult(
                    corner=corner,
                    value=value,
                    success=True,
                    dataset=dataset,
                )
            except Exception as e:
                result = CornerResult(
                    corner=corner,
                    value=float("nan"),
                    success=False,
                    metrics={"error": str(e)},
                )

            self._results.append(result)

        return CornerAnalysisResult(
            results=self._results,
            definition=definition,
        )

    def run_parallel(
        self,
        definition: CornerDefinition,
        objective: Callable[[dict[str, Any], Any], float],
        corners: list[Corner] | None = None,
        n_workers: int = 4,
    ) -> CornerAnalysisResult:
        """Run corner analysis in parallel.

        Note: This is a placeholder. Actual parallel implementation
        would require careful handling of circuit copies.

        Args:
            definition: Corner definition
            objective: Objective function
            corners: Optional specific corners
            n_workers: Number of parallel workers

        Returns:
            CornerAnalysisResult
        """
        # For now, fall back to sequential
        return self.run(definition, objective, corners)


@dataclass
class PVTCondition:
    """A single PVT condition.

    Attributes:
        process: Process corner (e.g., "slow", "typical", "fast")
        voltage: Supply voltage
        temperature: Temperature in Celsius
        name: Optional name override
    """

    process: str
    voltage: float
    temperature: float
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = f"{self.process}_{self.voltage}V_{self.temperature}C"

    def to_corner(self) -> Corner:
        """Convert to Corner object."""
        return Corner(
            name=self.name or f"{self.process}_{self.voltage}V_{self.temperature}C",
            parameters={
                "process": self.process,
                "voltage": self.voltage,
                "temperature": self.temperature,
            },
        )


class PVTSweep:
    """PVT (Process-Voltage-Temperature) sweep analysis.

    Provides convenient methods for common PVT analysis patterns.

    Example:
        >>> pvt = PVTSweep()
        >>> pvt.set_process(["slow", "typical", "fast"])
        >>> pvt.set_voltage([3.0, 3.3, 3.6])
        >>> pvt.set_temperature([-40, 25, 85])
        >>> all_pvt = pvt.generate_all()  # 27 PVT conditions
    """

    # Standard corner names
    PROCESS_CORNERS = ["slow", "typical", "fast"]
    TEMP_CORNERS = [-40, 25, 85]  # Military grade
    TEMP_CORNERS_COMMERCIAL = [0, 25, 70]
    TEMP_CORNERS_INDUSTRIAL = [-40, 25, 105]

    def __init__(self) -> None:
        """Initialize PVT sweep."""
        self._process: list[str] = ["typical"]
        self._voltage: list[float] = []
        self._temperature: list[float] = [25]

    def set_process(self, corners: Sequence[str]) -> "PVTSweep":
        """Set process corners.

        Args:
            corners: List of process corner names

        Returns:
            Self for chaining
        """
        self._process = list(corners)
        return self

    def set_voltage(
        self,
        nominal: float,
        tolerance: float = 0.1,
        n_points: int = 3,
    ) -> "PVTSweep":
        """Set voltage range.

        Args:
            nominal: Nominal voltage
            tolerance: Fractional tolerance (e.g., 0.1 for ±10%)
            n_points: Number of voltage points

        Returns:
            Self for chaining
        """
        v_min = nominal * (1 - tolerance)
        v_max = nominal * (1 + tolerance)
        self._voltage = list(np.linspace(v_min, v_max, n_points))
        return self

    def set_voltage_values(self, voltages: Sequence[float]) -> "PVTSweep":
        """Set explicit voltage values.

        Args:
            voltages: List of voltage values

        Returns:
            Self for chaining
        """
        self._voltage = list(voltages)
        return self

    def set_temperature(
        self,
        corners: Sequence[float] | str = "military",
    ) -> "PVTSweep":
        """Set temperature corners.

        Args:
            corners: Temperature values or preset name
                    ("military", "commercial", "industrial")

        Returns:
            Self for chaining
        """
        if isinstance(corners, str):
            presets = {
                "military": self.TEMP_CORNERS,
                "commercial": self.TEMP_CORNERS_COMMERCIAL,
                "industrial": self.TEMP_CORNERS_INDUSTRIAL,
            }
            if corners not in presets:
                raise ValueError(f"Unknown preset: {corners}")
            self._temperature = list(presets[corners])
        else:
            self._temperature = list(corners)
        return self

    @property
    def n_conditions(self) -> int:
        """Get total number of PVT conditions."""
        n_v = len(self._voltage) if self._voltage else 1
        return len(self._process) * n_v * len(self._temperature)

    def generate_all(self) -> list[PVTCondition]:
        """Generate all PVT conditions.

        Returns:
            List of all PVT conditions
        """
        voltages = self._voltage if self._voltage else [None]
        conditions = []

        for p in self._process:
            for v in voltages:
                for t in self._temperature:
                    if v is not None:
                        conditions.append(PVTCondition(p, v, t))
                    else:
                        # No voltage specified - use dummy
                        conditions.append(
                            PVTCondition(
                                p,
                                0.0,
                                t,
                                name=f"{p}_{t}C",
                            )
                        )

        return conditions

    def generate_extremes(self) -> list[PVTCondition]:
        """Generate extreme PVT conditions (corners only).

        Returns:
            List of extreme PVT conditions
        """
        process_ext = (
            [self._process[0], self._process[-1]]
            if len(self._process) >= 2
            else self._process
        )
        voltage_ext = (
            [self._voltage[0], self._voltage[-1]]
            if len(self._voltage) >= 2
            else self._voltage
        )
        temp_ext = (
            [self._temperature[0], self._temperature[-1]]
            if len(self._temperature) >= 2
            else self._temperature
        )

        voltages = voltage_ext if voltage_ext else [None]
        conditions = []

        for p in process_ext:
            for v in voltages:
                for t in temp_ext:
                    if v is not None:
                        conditions.append(PVTCondition(p, v, t))
                    else:
                        conditions.append(
                            PVTCondition(p, 0.0, t, name=f"{p}_{t}C")
                        )

        return conditions

    def to_corner_definition(self) -> CornerDefinition:
        """Convert to CornerDefinition.

        Returns:
            CornerDefinition for use with CornerAnalysis
        """
        definition = CornerDefinition()
        definition.add_parameter("process", self._process)
        if self._voltage:
            definition.add_parameter("voltage", self._voltage)
        definition.add_parameter("temperature", self._temperature)
        return definition


@dataclass
class StatisticalCornerResult:
    """Results from statistical corner analysis.

    Attributes:
        samples: Array of sampled values
        n_samples: Number of samples
        mean: Mean value
        std: Standard deviation
        percentiles: Dictionary of percentile values
    """

    samples: np.ndarray
    n_samples: int
    mean: float
    std: float
    percentiles: dict[float, float]

    @property
    def min(self) -> float:
        """Get minimum value."""
        return float(np.min(self.samples))

    @property
    def max(self) -> float:
        """Get maximum value."""
        return float(np.max(self.samples))

    def get_yield(self, spec_min: float | None = None, spec_max: float | None = None) -> float:
        """Calculate yield (fraction within spec).

        Args:
            spec_min: Minimum specification (None = no lower limit)
            spec_max: Maximum specification (None = no upper limit)

        Returns:
            Yield as fraction (0-1)
        """
        in_spec = np.ones(len(self.samples), dtype=bool)
        if spec_min is not None:
            in_spec &= self.samples >= spec_min
        if spec_max is not None:
            in_spec &= self.samples <= spec_max
        return float(np.mean(in_spec))

    def get_cpk(
        self, spec_min: float | None = None, spec_max: float | None = None
    ) -> float:
        """Calculate process capability index (Cpk).

        Args:
            spec_min: Lower specification limit
            spec_max: Upper specification limit

        Returns:
            Cpk value
        """
        if self.std == 0:
            return float("inf")

        cpk_values = []
        if spec_min is not None:
            cpu = (self.mean - spec_min) / (3 * self.std)
            cpk_values.append(cpu)
        if spec_max is not None:
            cpl = (spec_max - self.mean) / (3 * self.std)
            cpk_values.append(cpl)

        if not cpk_values:
            return float("nan")

        return min(cpk_values)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "percentiles": self.percentiles,
        }


def statistical_corner_analysis(
    objective: Callable[[dict[str, float]], float],
    bounds: list[ParameterBounds],
    n_samples: int = 1000,
    distribution: str = "uniform",
    seed: int | None = None,
    percentiles: Sequence[float] = (1, 5, 50, 95, 99),
) -> StatisticalCornerResult:
    """Run statistical (Monte Carlo) corner analysis.

    Args:
        objective: Function to evaluate. Signature: objective(params) -> float
        bounds: Parameter bounds for sampling
        n_samples: Number of Monte Carlo samples
        distribution: Distribution type ("uniform" or "normal")
        seed: Random seed
        percentiles: Percentiles to calculate

    Returns:
        StatisticalCornerResult with distribution statistics
    """
    rng = np.random.default_rng(seed)

    samples = []
    for _ in range(n_samples):
        params = {}
        for bound in bounds:
            if distribution == "uniform":
                value = rng.uniform(bound.lower, bound.upper)
            elif distribution == "normal":
                # Normal distribution with mean at center, 3-sigma at bounds
                center = (bound.lower + bound.upper) / 2
                sigma = (bound.upper - bound.lower) / 6
                value = rng.normal(center, sigma)
                value = np.clip(value, bound.lower, bound.upper)
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
            params[bound.name] = value

        try:
            result = objective(params)
            samples.append(result)
        except Exception:
            pass  # Skip failed samples

    arr = np.array(samples)
    pct_values = {p: float(np.percentile(arr, p)) for p in percentiles}

    return StatisticalCornerResult(
        samples=arr,
        n_samples=len(samples),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        percentiles=pct_values,
    )


@dataclass
class WorstCaseResult:
    """Results from worst-case analysis.

    Attributes:
        worst_value: The worst-case objective value
        worst_params: Parameters at worst case
        sensitivity: Sensitivity of each parameter to worst case
        n_evaluations: Number of function evaluations
    """

    worst_value: float
    worst_params: dict[str, float]
    sensitivity: dict[str, float]
    n_evaluations: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "worst_value": self.worst_value,
            "worst_params": self.worst_params.copy(),
            "sensitivity": self.sensitivity.copy(),
            "n_evaluations": self.n_evaluations,
        }


def worst_case_analysis(
    objective: Callable[[dict[str, float]], float],
    bounds: list[ParameterBounds],
    maximize: bool = True,
    method: str = "extremes",
    n_samples: int = 1000,
    seed: int | None = None,
) -> WorstCaseResult:
    """Find worst-case parameter combination.

    Args:
        objective: Function to evaluate
        bounds: Parameter bounds
        maximize: If True, find maximum (worst = highest)
        method: Search method:
               - "extremes": Check all extreme combinations (2^n)
               - "statistical": Monte Carlo sampling
               - "gradient": Gradient-based search from extremes
        n_samples: Number of samples for statistical method
        seed: Random seed

    Returns:
        WorstCaseResult with worst-case information
    """
    rng = np.random.default_rng(seed)

    if method == "extremes":
        # Check all corner combinations
        n_params = len(bounds)
        n_corners = 2**n_params

        best_value = float("-inf") if maximize else float("inf")
        best_params: dict[str, float] = {}
        n_evals = 0

        for i in range(n_corners):
            params = {}
            for j, bound in enumerate(bounds):
                # Use bit j of i to select min/max
                if (i >> j) & 1:
                    params[bound.name] = bound.upper
                else:
                    params[bound.name] = bound.lower

            try:
                value = objective(params)
                n_evals += 1

                if maximize:
                    if value > best_value:
                        best_value = value
                        best_params = params.copy()
                else:
                    if value < best_value:
                        best_value = value
                        best_params = params.copy()
            except Exception:
                n_evals += 1

    elif method == "statistical":
        # Monte Carlo sampling
        best_value = float("-inf") if maximize else float("inf")
        best_params = {}

        for _ in range(n_samples):
            params = {}
            for bound in bounds:
                params[bound.name] = rng.uniform(bound.lower, bound.upper)

            try:
                value = objective(params)

                if maximize:
                    if value > best_value:
                        best_value = value
                        best_params = params.copy()
                else:
                    if value < best_value:
                        best_value = value
                        best_params = params.copy()
            except Exception:
                pass

        n_evals = n_samples

    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate sensitivity at worst case
    sensitivity = {}
    base_value = objective(best_params)
    for bound in bounds:
        # Perturb each parameter
        delta = (bound.upper - bound.lower) * 0.01
        perturbed = best_params.copy()
        perturbed[bound.name] = min(best_params[bound.name] + delta, bound.upper)
        try:
            perturbed_value = objective(perturbed)
            sensitivity[bound.name] = (perturbed_value - base_value) / delta
            n_evals += 1
        except Exception:
            sensitivity[bound.name] = 0.0
            n_evals += 1

    return WorstCaseResult(
        worst_value=best_value,
        worst_params=best_params,
        sensitivity=sensitivity,
        n_evaluations=n_evals,
    )


def print_corner_report(result: CornerAnalysisResult) -> str:
    """Generate a text report for corner analysis results.

    Args:
        result: Corner analysis result

    Returns:
        Formatted report string
    """
    lines = ["=" * 60]
    lines.append("CORNER ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    stats = result.get_statistics()
    lines.append(f"Corners evaluated: {result.n_corners}")
    lines.append(f"Successful: {result.n_success} ({result.success_rate:.1%})")
    lines.append("")

    # Statistics
    lines.append("Statistics:")
    lines.append(f"  Min:   {stats['min']:.6g}")
    lines.append(f"  Max:   {stats['max']:.6g}")
    lines.append(f"  Mean:  {stats['mean']:.6g}")
    lines.append(f"  Std:   {stats['std']:.6g}")
    lines.append(f"  Range: {stats['range']:.6g}")
    lines.append("")

    # Worst/Best case
    if result.worst_case:
        lines.append(f"Worst case: {result.worst_case.corner.name}")
        lines.append(f"  Value: {result.worst_case.value:.6g}")
        for k, v in result.worst_case.corner.parameters.items():
            lines.append(f"  {k}: {v}")
    lines.append("")

    if result.best_case:
        lines.append(f"Best case: {result.best_case.corner.name}")
        lines.append(f"  Value: {result.best_case.value:.6g}")
    lines.append("")

    # Parameter sensitivity
    lines.append("Parameter sensitivity (average value per level):")
    for param in result.definition.parameter_names:
        sens = result.sensitivity_to_parameter(param)
        lines.append(f"  {param}:")
        for level, avg in sens.items():
            lines.append(f"    {level}: {avg:.6g}")
    lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
