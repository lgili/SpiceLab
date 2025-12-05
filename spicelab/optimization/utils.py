"""Optimization utilities and helpers.

This module provides additional optimization capabilities:
- MultiStartOptimizer: Run multiple optimization from different starting points
- ConvergenceTracker: Track and visualize optimization convergence
- Constraint builders for common constraint types
- Result analysis utilities
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import (
    Constraint,
    ObjectiveFunction,
    OptimizationConfig,
    OptimizationResult,
    Optimizer,
    ParameterBounds,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Constraint Builders
# =============================================================================


def make_inequality_constraint(
    func: Callable[[dict[str, float]], float],
    name: str = "constraint",
) -> Constraint:
    """Create an inequality constraint g(x) >= 0.

    Args:
        func: Function that returns value >= 0 when satisfied
        name: Constraint name for debugging

    Returns:
        Constraint object

    Example:
        >>> # R1 must be at least twice R2
        >>> constraint = make_inequality_constraint(
        ...     lambda p: p["R1"] - 2 * p["R2"],
        ...     "R1 >= 2*R2"
        ... )
    """

    class InequalityConstraint:
        def __init__(self, f: Callable[[dict[str, float]], float], n: str):
            self._func = f
            self._name = n

        def __call__(self, parameters: dict[str, float]) -> float:
            return self._func(parameters)

        def __repr__(self) -> str:
            return f"InequalityConstraint({self._name})"

    return InequalityConstraint(func, name)


def make_equality_constraint(
    func: Callable[[dict[str, float]], float],
    tolerance: float = 1e-6,
    name: str = "equality",
) -> Constraint:
    """Create an equality constraint |g(x)| <= tolerance.

    Converted to inequality: tolerance - |g(x)| >= 0

    Args:
        func: Function that should equal zero
        tolerance: Allowed tolerance around zero
        name: Constraint name for debugging

    Returns:
        Constraint object

    Example:
        >>> # R1 * C1 must equal 1e-6
        >>> constraint = make_equality_constraint(
        ...     lambda p: p["R1"] * p["C1"] - 1e-6,
        ...     tolerance=1e-9,
        ...     name="RC = 1us"
        ... )
    """

    class EqualityConstraint:
        def __init__(self, f: Callable[[dict[str, float]], float], tol: float, n: str):
            self._func = f
            self._tolerance = tol
            self._name = n

        def __call__(self, parameters: dict[str, float]) -> float:
            return self._tolerance - abs(self._func(parameters))

        def __repr__(self) -> str:
            return f"EqualityConstraint({self._name}, tol={self._tolerance})"

    return EqualityConstraint(func, tolerance, name)


def make_bound_constraint(
    param_name: str,
    lower: float | None = None,
    upper: float | None = None,
) -> list[Constraint]:
    """Create bound constraints for a parameter.

    Args:
        param_name: Parameter name
        lower: Lower bound (optional)
        upper: Upper bound (optional)

    Returns:
        List of constraints (one for each bound specified)
    """
    constraints: list[Constraint] = []

    if lower is not None:
        lb_val: float = lower

        def lower_constraint(
            p: dict[str, float], lb: float = lb_val, pn: str = param_name
        ) -> float:
            return p[pn] - lb

        constraints.append(
            make_inequality_constraint(
                lower_constraint,
                f"{param_name} >= {lower}",
            )
        )

    if upper is not None:
        ub_val: float = upper

        def upper_constraint(
            p: dict[str, float], ub: float = ub_val, pn: str = param_name
        ) -> float:
            return ub - p[pn]

        constraints.append(
            make_inequality_constraint(
                upper_constraint,
                f"{param_name} <= {upper}",
            )
        )

    return constraints


# =============================================================================
# Convergence Tracking
# =============================================================================


@dataclass
class ConvergenceData:
    """Data from convergence tracking.

    Attributes:
        iterations: List of iteration numbers
        values: Objective values at each iteration
        best_values: Best value found up to each iteration
        parameters: Parameters at each iteration
        gradients: Gradient norms (if available)
        step_sizes: Step sizes (if available)
    """

    iterations: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    best_values: list[float] = field(default_factory=list)
    parameters: list[dict[str, float]] = field(default_factory=list)
    gradients: list[float] = field(default_factory=list)
    step_sizes: list[float] = field(default_factory=list)

    def add_point(
        self,
        iteration: int,
        value: float,
        params: dict[str, float],
        gradient: float | None = None,
        step_size: float | None = None,
    ) -> None:
        """Add a convergence point."""
        self.iterations.append(iteration)
        self.values.append(value)

        if not self.best_values:
            self.best_values.append(value)
        else:
            self.best_values.append(min(self.best_values[-1], value))

        self.parameters.append(params.copy())

        if gradient is not None:
            self.gradients.append(gradient)
        if step_size is not None:
            self.step_sizes.append(step_size)

    @property
    def n_iterations(self) -> int:
        """Number of recorded iterations."""
        return len(self.iterations)

    @property
    def final_value(self) -> float:
        """Final objective value."""
        return self.values[-1] if self.values else float("nan")

    @property
    def best_value(self) -> float:
        """Best objective value found."""
        return self.best_values[-1] if self.best_values else float("nan")

    @property
    def improvement(self) -> float:
        """Improvement from first to best value."""
        if len(self.values) < 2:
            return 0.0
        return self.values[0] - self.best_value

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        """Convert to dictionary for serialization."""
        return {
            "iterations": self.iterations,
            "values": self.values,
            "best_values": self.best_values,
        }


class ConvergenceTracker:
    """Track optimization convergence for analysis and visualization.

    Example:
        >>> tracker = ConvergenceTracker()
        >>> # Use with optimization
        >>> for i, (params, value) in enumerate(optimization_steps):
        ...     tracker.record(i, value, params)
        >>> print(f"Converged: {tracker.is_converged()}")
        >>> print(f"Stagnant: {tracker.is_stagnant()}")
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        patience: int = 10,
        min_improvement: float = 1e-8,
    ):
        """Initialize convergence tracker.

        Args:
            tolerance: Absolute tolerance for convergence
            patience: Number of iterations without improvement before stagnation
            min_improvement: Minimum relative improvement to count as progress
        """
        self.tolerance = tolerance
        self.patience = patience
        self.min_improvement = min_improvement
        self.data = ConvergenceData()
        self._stagnant_count = 0

    def record(
        self,
        iteration: int,
        value: float,
        params: dict[str, float],
        gradient: float | None = None,
    ) -> None:
        """Record an optimization iteration.

        Args:
            iteration: Iteration number
            value: Objective value
            params: Parameter values
            gradient: Gradient norm (optional)
        """
        # Check for improvement
        if self.data.best_values:
            prev_best = self.data.best_values[-1]
            rel_improvement = (prev_best - value) / (abs(prev_best) + 1e-10)
            if rel_improvement < self.min_improvement:
                self._stagnant_count += 1
            else:
                self._stagnant_count = 0

        self.data.add_point(iteration, value, params, gradient)

    def is_converged(self, target: float | None = None) -> bool:
        """Check if optimization has converged.

        Args:
            target: Optional target value to converge to

        Returns:
            True if converged
        """
        if len(self.data.values) < 2:
            return False

        if target is not None:
            return abs(self.data.best_value - target) < self.tolerance

        # Check if recent changes are small
        recent_values = self.data.values[-5:]
        if len(recent_values) < 2:
            return False

        value_range = max(recent_values) - min(recent_values)
        return value_range < self.tolerance

    def is_stagnant(self) -> bool:
        """Check if optimization is stagnant (no improvement).

        Returns:
            True if no improvement for `patience` iterations
        """
        return self._stagnant_count >= self.patience

    def reset(self) -> None:
        """Reset tracker state."""
        self.data = ConvergenceData()
        self._stagnant_count = 0


# =============================================================================
# Multi-Start Optimization
# =============================================================================


@dataclass
class MultiStartResult:
    """Result from multi-start optimization.

    Attributes:
        best_result: Best result across all starts
        all_results: Results from all starts
        start_points: Initial points used
        n_successful: Number of successful optimizations
    """

    best_result: OptimizationResult[Any]
    all_results: list[OptimizationResult[Any]]
    start_points: list[dict[str, float]]
    n_successful: int

    @property
    def n_starts(self) -> int:
        """Number of optimization starts."""
        return len(self.all_results)

    @property
    def success_rate(self) -> float:
        """Fraction of successful optimizations."""
        return self.n_successful / self.n_starts if self.n_starts > 0 else 0.0

    def get_unique_optima(self, tolerance: float = 0.01) -> list[OptimizationResult[Any]]:
        """Get unique optima (different by more than tolerance).

        Args:
            tolerance: Relative tolerance for considering optima different

        Returns:
            List of unique optimization results
        """
        if not self.all_results:
            return []

        unique: list[OptimizationResult[Any]] = []
        for result in sorted(self.all_results, key=lambda r: r.value):
            is_unique = True
            for u in unique:
                # Check if parameters are similar
                param_diff = 0.0
                for key in result.parameters:
                    if key in u.parameters:
                        ref = abs(u.parameters[key]) + 1e-10
                        diff = abs(result.parameters[key] - u.parameters[key]) / ref
                        param_diff = max(param_diff, diff)
                if param_diff < tolerance:
                    is_unique = False
                    break
            if is_unique:
                unique.append(result)

        return unique


class MultiStartOptimizer:
    """Run optimization from multiple starting points.

    This helps avoid local minima by exploring different regions
    of the parameter space.

    Example:
        >>> base_optimizer = NelderMeadOptimizer()
        >>> multi = MultiStartOptimizer(base_optimizer, n_starts=10)
        >>> result = multi.optimize(objective, bounds)
        >>> print(f"Found {len(result.get_unique_optima())} unique optima")
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        n_starts: int = 10,
        sampling: str = "random",
        seed: int | None = None,
    ):
        """Initialize multi-start optimizer.

        Args:
            base_optimizer: Optimizer to use for each start
            n_starts: Number of starting points
            sampling: Sampling method ("random", "lhs", "grid")
            seed: Random seed for reproducibility
        """
        self.base_optimizer = base_optimizer
        self.n_starts = n_starts
        self.sampling = sampling
        self.seed = seed

    def _generate_start_points(
        self,
        bounds: list[ParameterBounds],
        n_points: int,
    ) -> list[dict[str, float]]:
        """Generate starting points.

        Args:
            bounds: Parameter bounds
            n_points: Number of points to generate

        Returns:
            List of parameter dictionaries
        """
        rng = np.random.default_rng(self.seed)

        if self.sampling == "lhs":
            # Latin Hypercube Sampling
            points = self._lhs_sample(len(bounds), n_points, rng)
        elif self.sampling == "grid":
            # Grid sampling (approximate)
            points = self._grid_sample(len(bounds), n_points)
        else:
            # Random sampling
            points = rng.random((n_points, len(bounds)))

        # Convert to parameter dictionaries with proper scaling
        result = []
        for point in points:
            params = {}
            for i, b in enumerate(bounds):
                params[b.name] = b.denormalize(point[i])
            result.append(params)

        return result

    def _lhs_sample(
        self,
        n_dims: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate Latin Hypercube samples.

        Args:
            n_dims: Number of dimensions
            n_samples: Number of samples

        Returns:
            Array of shape (n_samples, n_dims) with values in [0, 1]
        """
        result = np.zeros((n_samples, n_dims))
        for i in range(n_dims):
            # Create evenly spaced intervals
            intervals = np.linspace(0, 1, n_samples + 1)
            # Sample within each interval
            for j in range(n_samples):
                result[j, i] = rng.uniform(intervals[j], intervals[j + 1])
            # Shuffle the column
            rng.shuffle(result[:, i])
        return result

    def _grid_sample(self, n_dims: int, n_samples: int) -> np.ndarray:
        """Generate approximate grid samples.

        Args:
            n_dims: Number of dimensions
            n_samples: Approximate number of samples

        Returns:
            Array with values in [0, 1]
        """
        # Calculate points per dimension
        points_per_dim = max(2, int(np.ceil(n_samples ** (1.0 / n_dims))))

        # Generate grid
        grids = [np.linspace(0, 1, points_per_dim) for _ in range(n_dims)]
        mesh = np.meshgrid(*grids)
        points = np.column_stack([g.ravel() for g in mesh])

        # Limit to n_samples
        if len(points) > n_samples:
            indices = np.linspace(0, len(points) - 1, n_samples, dtype=int)
            points = points[indices]

        return points

    def optimize(
        self,
        objective: ObjectiveFunction,
        bounds: list[ParameterBounds],
        constraints: list[Constraint] | None = None,
        config: OptimizationConfig | None = None,
    ) -> MultiStartResult:
        """Run multi-start optimization.

        Args:
            objective: Objective function
            bounds: Parameter bounds
            constraints: Optional constraints
            config: Optimization configuration

        Returns:
            MultiStartResult with best and all results
        """
        if config is None:
            config = OptimizationConfig()

        # Generate starting points
        start_points = self._generate_start_points(bounds, self.n_starts)

        # Run optimization from each start
        all_results: list[OptimizationResult[Any]] = []
        best_result: OptimizationResult[Any] | None = None
        n_successful = 0

        for i, start in enumerate(start_points):
            # Update bounds with new initial point
            modified_bounds = []
            for b in bounds:
                new_bound = ParameterBounds(
                    name=b.name,
                    lower=b.lower,
                    upper=b.upper,
                    initial=start[b.name],
                    scale=b.scale,
                )
                modified_bounds.append(new_bound)

            # Run optimization
            if config.verbose:
                print(f"  Start {i + 1}/{self.n_starts}")

            result = self.base_optimizer.optimize(
                objective,
                modified_bounds,
                constraints,
                config,
            )

            all_results.append(result)

            if result.success:
                n_successful += 1

            # Track best
            if best_result is None or result.value < best_result.value:
                best_result = result

        # Ensure we have a best result
        if best_result is None:
            best_result = (
                all_results[0]
                if all_results
                else OptimizationResult(
                    success=False,
                    value=float("inf"),
                    parameters={b.name: b.initial or 0 for b in bounds},
                    n_iterations=0,
                    n_evaluations=0,
                    message="No optimization runs completed",
                )
            )

        return MultiStartResult(
            best_result=best_result,
            all_results=all_results,
            start_points=start_points,
            n_successful=n_successful,
        )


# =============================================================================
# Result Analysis
# =============================================================================


def analyze_sensitivity(
    result: OptimizationResult[Any],
    objective: ObjectiveFunction,
    bounds: list[ParameterBounds],
    perturbation: float = 0.01,
) -> dict[str, float]:
    """Analyze parameter sensitivity at the optimum.

    Computes the relative change in objective for small parameter changes.

    Args:
        result: Optimization result
        objective: Objective function
        bounds: Parameter bounds
        perturbation: Relative perturbation size

    Returns:
        Dictionary mapping parameter names to sensitivity values
    """
    sensitivities: dict[str, float] = {}

    for bound in bounds:
        param_name = bound.name
        base_param = result.parameters[param_name]

        # Perturb parameter
        delta = abs(base_param * perturbation) if base_param != 0 else perturbation

        # Positive perturbation
        perturbed_params = result.parameters.copy()
        perturbed_params[param_name] = base_param + delta
        value_plus = objective(perturbed_params)

        # Negative perturbation
        perturbed_params[param_name] = base_param - delta
        value_minus = objective(perturbed_params)

        # Central difference sensitivity
        sensitivity = (value_plus - value_minus) / (2 * delta)

        # Normalize by parameter scale
        sensitivities[param_name] = (
            sensitivity * abs(base_param) if base_param != 0 else sensitivity
        )

    return sensitivities


def compute_hessian_diagonal(
    result: OptimizationResult[Any],
    objective: ObjectiveFunction,
    bounds: list[ParameterBounds],
    perturbation: float = 0.01,
) -> dict[str, float]:
    """Compute diagonal of Hessian at the optimum.

    Useful for estimating parameter uncertainty.

    Args:
        result: Optimization result
        objective: Objective function
        bounds: Parameter bounds
        perturbation: Relative perturbation size

    Returns:
        Dictionary mapping parameter names to Hessian diagonal values
    """
    base_value = result.value
    hessian_diag = {}

    for bound in bounds:
        param_name = bound.name
        base_param = result.parameters[param_name]

        delta = abs(base_param * perturbation) if base_param != 0 else perturbation

        # f(x + h)
        params_plus = result.parameters.copy()
        params_plus[param_name] = base_param + delta
        value_plus = objective(params_plus)

        # f(x - h)
        params_minus = result.parameters.copy()
        params_minus[param_name] = base_param - delta
        value_minus = objective(params_minus)

        # Second derivative: (f(x+h) - 2f(x) + f(x-h)) / h^2
        hessian_diag[param_name] = (value_plus - 2 * base_value + value_minus) / (delta**2)

    return hessian_diag


__all__ = [
    # Constraint builders
    "make_inequality_constraint",
    "make_equality_constraint",
    "make_bound_constraint",
    # Convergence tracking
    "ConvergenceData",
    "ConvergenceTracker",
    # Multi-start
    "MultiStartResult",
    "MultiStartOptimizer",
    # Analysis
    "analyze_sensitivity",
    "compute_hessian_diagonal",
]
