"""Base classes and protocols for circuit optimization.

This module provides the foundation for optimization in SpiceLab:
- OptimizationResult: Result container with value, parameters, history
- ObjectiveFunction: Protocol for objective functions
- Optimizer: Protocol for optimization algorithms
- CircuitOptimizer: Base class connecting circuits to optimizers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Protocol, TypeVar

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from spicelab.core.circuit import Circuit

T = TypeVar("T")


@dataclass
class ParameterBounds:
    """Bounds for an optimization parameter.

    Attributes:
        name: Parameter name (e.g., "R1", "C1.value")
        lower: Lower bound
        upper: Upper bound
        initial: Initial value (optional, defaults to midpoint)
        scale: Scale type ("linear", "log") for better convergence
    """

    name: str
    lower: float
    upper: float
    initial: float | None = None
    scale: str = "linear"

    def __post_init__(self) -> None:
        if self.lower >= self.upper:
            raise ValueError(f"Lower bound must be < upper bound for {self.name}")
        if self.initial is None:
            if self.scale == "log":
                self.initial = np.sqrt(self.lower * self.upper)
            else:
                self.initial = (self.lower + self.upper) / 2
        if not (self.lower <= self.initial <= self.upper):
            raise ValueError(f"Initial value must be within bounds for {self.name}")

    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range."""
        if self.scale == "log":
            return (np.log10(value) - np.log10(self.lower)) / (
                np.log10(self.upper) - np.log10(self.lower)
            )
        return (value - self.lower) / (self.upper - self.lower)

    def denormalize(self, normalized: float) -> float:
        """Convert normalized [0, 1] value back to actual value."""
        if self.scale == "log":
            log_value = (
                normalized * (np.log10(self.upper) - np.log10(self.lower)) + np.log10(self.lower)
            )
            return float(10**log_value)
        return normalized * (self.upper - self.lower) + self.lower


@dataclass
class OptimizationResult(Generic[T]):
    """Result of an optimization run.

    Attributes:
        success: Whether optimization converged
        value: Final objective function value
        parameters: Optimal parameter values
        n_iterations: Number of iterations/generations
        n_evaluations: Number of objective function evaluations
        history: History of (parameters, value) tuples
        message: Status message
        metadata: Additional algorithm-specific data
    """

    success: bool
    value: float
    parameters: dict[str, float]
    n_iterations: int
    n_evaluations: int
    history: list[tuple[dict[str, float], float]] = field(default_factory=list)
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "value": self.value,
            "parameters": self.parameters,
            "n_iterations": self.n_iterations,
            "n_evaluations": self.n_evaluations,
            "history_length": len(self.history),
            "message": self.message,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        params_str = ", ".join(f"{k}={v:.4g}" for k, v in self.parameters.items())
        return f"OptimizationResult({status}, value={self.value:.6g}, params=[{params_str}])"


class ObjectiveFunction(Protocol):
    """Protocol for objective functions.

    An objective function takes parameter values and returns a scalar cost.
    Lower values are better (minimization).
    """

    def __call__(self, parameters: dict[str, float]) -> float:
        """Evaluate objective function.

        Args:
            parameters: Dictionary mapping parameter names to values

        Returns:
            Scalar objective value (lower is better)
        """
        ...


class Constraint(Protocol):
    """Protocol for optimization constraints.

    A constraint function returns a value that should be >= 0 for feasibility.
    """

    def __call__(self, parameters: dict[str, float]) -> float:
        """Evaluate constraint.

        Args:
            parameters: Dictionary mapping parameter names to values

        Returns:
            Value >= 0 if constraint is satisfied
        """
        ...


@dataclass
class OptimizationConfig:
    """Configuration for optimization runs.

    Attributes:
        max_iterations: Maximum number of iterations
        max_evaluations: Maximum objective function evaluations
        tolerance: Convergence tolerance
        verbose: Print progress during optimization
        seed: Random seed for reproducibility
        parallel: Enable parallel evaluation
        n_workers: Number of parallel workers
    """

    max_iterations: int = 100
    max_evaluations: int = 1000
    tolerance: float = 1e-6
    verbose: bool = False
    seed: int | None = None
    parallel: bool = False
    n_workers: int = 4


class Optimizer(ABC):
    """Abstract base class for optimization algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Optimizer name."""
        ...

    @abstractmethod
    def optimize(
        self,
        objective: ObjectiveFunction,
        bounds: list[ParameterBounds],
        constraints: list[Constraint] | None = None,
        config: OptimizationConfig | None = None,
    ) -> OptimizationResult:
        """Run optimization.

        Args:
            objective: Objective function to minimize
            bounds: Parameter bounds
            constraints: Optional constraints (g(x) >= 0)
            config: Optimization configuration

        Returns:
            OptimizationResult with optimal parameters
        """
        ...


class CircuitObjective:
    """Objective function that evaluates a circuit.

    This class connects circuit simulation to optimization by:
    1. Updating circuit parameters
    2. Running simulation
    3. Extracting measurements
    4. Computing objective value

    Example:
        >>> def my_objective(params, result):
        ...     gain = measure(result.dataset, "gain", output_node="vout")
        ...     # Minimize negative gain (maximize gain)
        ...     return -gain.value
        >>>
        >>> obj = CircuitObjective(circuit, simulation_func, my_objective)
        >>> value = obj({"R1": 1000, "R2": 10000})
    """

    def __init__(
        self,
        circuit: Circuit,
        simulate: Callable[[Circuit], xr.Dataset],
        evaluate: Callable[[dict[str, float], xr.Dataset], float],
        cache_results: bool = True,
    ):
        """Initialize circuit objective.

        Args:
            circuit: Base circuit to modify
            simulate: Function that simulates the circuit and returns dataset
            evaluate: Function that computes objective from params and dataset
            cache_results: Cache simulation results for repeated evaluations
        """
        self.circuit = circuit
        self.simulate = simulate
        self.evaluate = evaluate
        self.cache_results = cache_results
        self._cache: dict[tuple[tuple[str, float], ...], float] = {}
        self._n_evaluations = 0
        self._n_cache_hits = 0

    def __call__(self, parameters: dict[str, float]) -> float:
        """Evaluate objective function.

        Args:
            parameters: Parameter values to test

        Returns:
            Objective value (lower is better)
        """
        # Check cache
        cache_key = tuple(sorted(parameters.items()))
        if self.cache_results and cache_key in self._cache:
            self._n_cache_hits += 1
            return self._cache[cache_key]

        self._n_evaluations += 1

        # Update circuit parameters
        circuit_copy = self._update_circuit(parameters)

        # Run simulation
        try:
            dataset = self.simulate(circuit_copy)
        except Exception as e:
            # Return large penalty for failed simulations
            return 1e12 + hash(str(e)) % 1000

        # Evaluate objective
        try:
            value = self.evaluate(parameters, dataset)
        except Exception:
            # Return large penalty for evaluation failures
            return 1e12

        # Cache result
        if self.cache_results:
            self._cache[cache_key] = value

        return value

    def _update_circuit(self, parameters: dict[str, float]) -> Circuit:
        """Create a copy of the circuit with updated parameters."""
        # Create a shallow copy - we'll modify parameter values
        circuit_copy = self.circuit.copy()

        for name, value in parameters.items():
            # Handle nested parameter references (e.g., "R1.value")
            if "." in name:
                component_name, param_name = name.split(".", 1)
                component = circuit_copy.get_component(component_name)
                if component is not None:
                    setattr(component, param_name, value)
            else:
                # Direct component value update
                component = circuit_copy.get_component(name)
                if component is not None:
                    component.value = value

        return circuit_copy

    @property
    def n_evaluations(self) -> int:
        """Number of objective function evaluations."""
        return self._n_evaluations

    @property
    def n_cache_hits(self) -> int:
        """Number of cache hits."""
        return self._n_cache_hits

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()


class CircuitOptimizer:
    """High-level interface for circuit optimization.

    Combines circuit, objective, and optimizer into a unified interface.

    Example:
        >>> optimizer = CircuitOptimizer(circuit, simulate_func)
        >>> optimizer.add_parameter("R1", 1e3, 100e3, scale="log")
        >>> optimizer.add_parameter("C1", 1e-12, 1e-9, scale="log")
        >>>
        >>> def objective(params, dataset):
        ...     bw = measure(dataset, "bandwidth", output_node="vout")
        ...     return -bw.value  # Maximize bandwidth
        >>>
        >>> result = optimizer.optimize(objective, method="nelder-mead")
    """

    def __init__(
        self,
        circuit: Circuit,
        simulate: Callable[[Circuit], xr.Dataset],
    ):
        """Initialize circuit optimizer.

        Args:
            circuit: Circuit to optimize
            simulate: Simulation function
        """
        self.circuit = circuit
        self.simulate = simulate
        self._bounds: list[ParameterBounds] = []
        self._constraints: list[Constraint] = []

    def add_parameter(
        self,
        name: str,
        lower: float,
        upper: float,
        initial: float | None = None,
        scale: str = "linear",
    ) -> CircuitOptimizer:
        """Add a parameter to optimize.

        Args:
            name: Parameter name (component name or "component.attribute")
            lower: Lower bound
            upper: Upper bound
            initial: Initial value (optional)
            scale: "linear" or "log"

        Returns:
            self for method chaining
        """
        self._bounds.append(ParameterBounds(name, lower, upper, initial, scale))
        return self

    def add_constraint(self, constraint: Constraint) -> CircuitOptimizer:
        """Add an optimization constraint.

        Args:
            constraint: Constraint function (must return >= 0 when satisfied)

        Returns:
            self for method chaining
        """
        self._constraints.append(constraint)
        return self

    def optimize(
        self,
        objective: Callable[[dict[str, float], xr.Dataset], float],
        method: str = "nelder-mead",
        config: OptimizationConfig | None = None,
    ) -> OptimizationResult:
        """Run optimization.

        Args:
            objective: Objective function (params, dataset) -> scalar
            method: Optimization method name
            config: Optimization configuration

        Returns:
            OptimizationResult with optimal parameters
        """
        from .scipy_optimizers import get_scipy_optimizer

        # Create circuit objective
        circuit_obj = CircuitObjective(self.circuit, self.simulate, objective)

        # Get optimizer
        optimizer = get_scipy_optimizer(method)

        # Run optimization
        result = optimizer.optimize(
            circuit_obj,
            self._bounds,
            self._constraints if self._constraints else None,
            config,
        )

        # Add evaluation stats to metadata
        result.metadata["n_simulations"] = circuit_obj.n_evaluations
        result.metadata["n_cache_hits"] = circuit_obj.n_cache_hits

        return result

    @property
    def bounds(self) -> list[ParameterBounds]:
        """Current parameter bounds."""
        return self._bounds.copy()

    @property
    def constraints(self) -> list[Constraint]:
        """Current constraints."""
        return self._constraints.copy()


__all__ = [
    "ParameterBounds",
    "OptimizationResult",
    "ObjectiveFunction",
    "Constraint",
    "OptimizationConfig",
    "Optimizer",
    "CircuitObjective",
    "CircuitOptimizer",
]
