"""Scipy-based optimization algorithms.

This module wraps scipy.optimize functions for circuit optimization:
- NelderMeadOptimizer: Simplex method (gradient-free)
- BFGSOptimizer: Quasi-Newton method
- LBFGSBOptimizer: Limited-memory BFGS with bounds
- PowellOptimizer: Powell's conjugate direction method
- DifferentialEvolutionOptimizer: Global optimizer
- DualAnnealingOptimizer: Simulated annealing variant
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    from scipy.optimize import OptimizeResult


def _bounds_to_scipy(bounds: list[ParameterBounds]) -> list[tuple[float, float]]:
    """Convert ParameterBounds to scipy bounds format."""
    return [(b.lower, b.upper) for b in bounds]


def _get_initial_point(bounds: list[ParameterBounds]) -> np.ndarray:
    """Get initial point from bounds."""
    return np.array([b.initial for b in bounds])


def _array_to_params(x: np.ndarray, bounds: list[ParameterBounds]) -> dict[str, float]:
    """Convert numpy array to parameter dictionary."""
    return {b.name: float(x[i]) for i, b in enumerate(bounds)}


def _params_to_array(params: dict[str, float], bounds: list[ParameterBounds]) -> np.ndarray:
    """Convert parameter dictionary to numpy array."""
    return np.array([params[b.name] for b in bounds])


class ScipyOptimizer(Optimizer):
    """Base class for scipy-based optimizers."""

    def __init__(self, method: str):
        self._method = method
        self._history: list[tuple[dict[str, float], float]] = []

    @property
    def name(self) -> str:
        return f"scipy.{self._method}"

    def _create_callback(
        self, bounds: list[ParameterBounds]
    ) -> tuple[list[tuple[dict[str, float], float]], object]:
        """Create callback for tracking optimization history."""
        history: list[tuple[dict[str, float], float]] = []

        def callback(xk: np.ndarray, *args: object) -> None:
            # Note: callback signature varies by method
            params = _array_to_params(xk, bounds)
            # We don't have the function value in standard callback
            history.append((params, float("nan")))

        return history, callback

    def optimize(
        self,
        objective: ObjectiveFunction,
        bounds: list[ParameterBounds],
        constraints: list[Constraint] | None = None,
        config: OptimizationConfig | None = None,
    ) -> OptimizationResult:
        """Run scipy optimization."""
        from scipy import optimize

        if config is None:
            config = OptimizationConfig()

        if config.seed is not None:
            np.random.seed(config.seed)

        # Wrap objective to track evaluations
        n_evals = [0]
        best_value = [float("inf")]
        history: list[tuple[dict[str, float], float]] = []

        def wrapped_objective(x: np.ndarray) -> float:
            params = _array_to_params(x, bounds)
            value = objective(params)
            n_evals[0] += 1
            if value < best_value[0]:
                best_value[0] = value
                history.append((params.copy(), value))
            if config.verbose and n_evals[0] % 10 == 0:
                print(f"  Eval {n_evals[0]}: best={best_value[0]:.6g}")
            return value

        # Get initial point and bounds
        x0 = _get_initial_point(bounds)
        scipy_bounds = _bounds_to_scipy(bounds)

        # Build options
        options: dict[str, object] = {
            "maxiter": config.max_iterations,
            "disp": config.verbose,
        }
        if self._method in ("nelder-mead", "powell"):
            options["xatol"] = config.tolerance
            options["fatol"] = config.tolerance

        # Run optimization
        try:
            result: OptimizeResult = optimize.minimize(
                wrapped_objective,
                x0,
                method=self._method,
                bounds=scipy_bounds if self._method in ("l-bfgs-b", "slsqp", "trust-constr") else None,
                options=options,
            )

            return OptimizationResult(
                success=bool(result.success),
                value=float(result.fun),
                parameters=_array_to_params(result.x, bounds),
                n_iterations=int(result.nit) if hasattr(result, "nit") else 0,
                n_evaluations=n_evals[0],
                history=history,
                message=str(result.message) if hasattr(result, "message") else "",
                metadata={"scipy_result": str(result)},
            )
        except Exception as e:
            # Return failure result
            return OptimizationResult(
                success=False,
                value=best_value[0] if best_value[0] < float("inf") else float("nan"),
                parameters=_array_to_params(x0, bounds),
                n_iterations=0,
                n_evaluations=n_evals[0],
                history=history,
                message=f"Optimization failed: {e}",
            )


class NelderMeadOptimizer(ScipyOptimizer):
    """Nelder-Mead simplex optimizer.

    Good for:
    - Gradient-free optimization
    - Noisy objective functions
    - Low-dimensional problems (< 10 parameters)

    Not recommended for:
    - High-dimensional problems
    - Constrained optimization
    """

    def __init__(self) -> None:
        super().__init__("nelder-mead")


class PowellOptimizer(ScipyOptimizer):
    """Powell's conjugate direction method.

    Good for:
    - Gradient-free optimization
    - Smooth objective functions
    - Medium-dimensional problems

    Not recommended for:
    - Noisy objectives
    - Constrained problems
    """

    def __init__(self) -> None:
        super().__init__("powell")


class LBFGSBOptimizer(ScipyOptimizer):
    """L-BFGS-B optimizer with bound constraints.

    Good for:
    - Large-scale problems
    - Bound-constrained optimization
    - Smooth objective functions

    Note: Uses numerical gradients if not provided.
    """

    def __init__(self) -> None:
        super().__init__("l-bfgs-b")


class DifferentialEvolutionOptimizer(Optimizer):
    """Differential evolution global optimizer.

    Good for:
    - Global optimization
    - Non-convex problems
    - Multi-modal objective functions
    - Robust to local minima

    Parameters:
        popsize: Population size multiplier (default 15)
        mutation: Mutation constant (default (0.5, 1))
        recombination: Crossover probability (default 0.7)
    """

    def __init__(
        self,
        popsize: int = 15,
        mutation: tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
    ):
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination

    @property
    def name(self) -> str:
        return "scipy.differential_evolution"

    def optimize(
        self,
        objective: ObjectiveFunction,
        bounds: list[ParameterBounds],
        constraints: list[Constraint] | None = None,
        config: OptimizationConfig | None = None,
    ) -> OptimizationResult:
        """Run differential evolution optimization."""
        from scipy import optimize

        if config is None:
            config = OptimizationConfig()

        if config.seed is not None:
            np.random.seed(config.seed)

        # Track evaluations
        n_evals = [0]
        best_value = [float("inf")]
        history: list[tuple[dict[str, float], float]] = []

        def wrapped_objective(x: np.ndarray) -> float:
            params = _array_to_params(x, bounds)
            value = objective(params)
            n_evals[0] += 1
            if value < best_value[0]:
                best_value[0] = value
                history.append((params.copy(), value))
            return value

        # Callback for progress
        def callback(xk: np.ndarray, convergence: float) -> bool:
            if config.verbose:
                print(f"  Gen: convergence={convergence:.4f}, best={best_value[0]:.6g}")
            return False  # Don't stop early

        scipy_bounds = _bounds_to_scipy(bounds)

        try:
            result = optimize.differential_evolution(
                wrapped_objective,
                scipy_bounds,
                maxiter=config.max_iterations,
                popsize=self.popsize,
                mutation=self.mutation,
                recombination=self.recombination,
                tol=config.tolerance,
                seed=config.seed,
                callback=callback if config.verbose else None,
                workers=config.n_workers if config.parallel else 1,
                updating="deferred" if config.parallel else "immediate",
            )

            return OptimizationResult(
                success=bool(result.success),
                value=float(result.fun),
                parameters=_array_to_params(result.x, bounds),
                n_iterations=int(result.nit),
                n_evaluations=n_evals[0],
                history=history,
                message=str(result.message),
                metadata={
                    "population_size": self.popsize * len(bounds),
                    "mutation": self.mutation,
                    "recombination": self.recombination,
                },
            )
        except Exception as e:
            return OptimizationResult(
                success=False,
                value=best_value[0] if best_value[0] < float("inf") else float("nan"),
                parameters=_array_to_params(_get_initial_point(bounds), bounds),
                n_iterations=0,
                n_evaluations=n_evals[0],
                history=history,
                message=f"Optimization failed: {e}",
            )


class DualAnnealingOptimizer(Optimizer):
    """Dual annealing global optimizer.

    Combines classical simulated annealing with local optimization.
    Good for:
    - Global optimization with many local minima
    - Continuous optimization
    - Bound-constrained problems
    """

    def __init__(self, initial_temp: float = 5230.0, restart_temp_ratio: float = 2e-5):
        self.initial_temp = initial_temp
        self.restart_temp_ratio = restart_temp_ratio

    @property
    def name(self) -> str:
        return "scipy.dual_annealing"

    def optimize(
        self,
        objective: ObjectiveFunction,
        bounds: list[ParameterBounds],
        constraints: list[Constraint] | None = None,
        config: OptimizationConfig | None = None,
    ) -> OptimizationResult:
        """Run dual annealing optimization."""
        from scipy import optimize

        if config is None:
            config = OptimizationConfig()

        # Track evaluations
        n_evals = [0]
        best_value = [float("inf")]
        history: list[tuple[dict[str, float], float]] = []

        def wrapped_objective(x: np.ndarray) -> float:
            params = _array_to_params(x, bounds)
            value = objective(params)
            n_evals[0] += 1
            if value < best_value[0]:
                best_value[0] = value
                history.append((params.copy(), value))
            return value

        scipy_bounds = _bounds_to_scipy(bounds)
        x0 = _get_initial_point(bounds)

        try:
            result = optimize.dual_annealing(
                wrapped_objective,
                scipy_bounds,
                maxiter=config.max_iterations,
                initial_temp=self.initial_temp,
                restart_temp_ratio=self.restart_temp_ratio,
                seed=config.seed,
                x0=x0,
            )

            return OptimizationResult(
                success=bool(result.success),
                value=float(result.fun),
                parameters=_array_to_params(result.x, bounds),
                n_iterations=int(result.nit) if hasattr(result, "nit") else 0,
                n_evaluations=n_evals[0],
                history=history,
                message=str(result.message) if hasattr(result, "message") else "",
                metadata={
                    "initial_temp": self.initial_temp,
                    "restart_temp_ratio": self.restart_temp_ratio,
                },
            )
        except Exception as e:
            return OptimizationResult(
                success=False,
                value=best_value[0] if best_value[0] < float("inf") else float("nan"),
                parameters=_array_to_params(x0, bounds),
                n_iterations=0,
                n_evaluations=n_evals[0],
                history=history,
                message=f"Optimization failed: {e}",
            )


# Registry of available optimizers
_SCIPY_OPTIMIZERS: dict[str, type[Optimizer]] = {
    "nelder-mead": NelderMeadOptimizer,
    "powell": PowellOptimizer,
    "l-bfgs-b": LBFGSBOptimizer,
    "differential-evolution": DifferentialEvolutionOptimizer,
    "dual-annealing": DualAnnealingOptimizer,
}


def get_scipy_optimizer(method: str) -> Optimizer:
    """Get a scipy optimizer by name.

    Args:
        method: Optimizer name (e.g., "nelder-mead", "differential-evolution")

    Returns:
        Optimizer instance

    Raises:
        ValueError: If method is not recognized
    """
    method_lower = method.lower().replace("_", "-")
    if method_lower not in _SCIPY_OPTIMIZERS:
        available = ", ".join(_SCIPY_OPTIMIZERS.keys())
        raise ValueError(f"Unknown optimizer '{method}'. Available: {available}")
    return _SCIPY_OPTIMIZERS[method_lower]()


def list_scipy_optimizers() -> list[str]:
    """List available scipy optimizer names."""
    return list(_SCIPY_OPTIMIZERS.keys())


__all__ = [
    "NelderMeadOptimizer",
    "PowellOptimizer",
    "LBFGSBOptimizer",
    "DifferentialEvolutionOptimizer",
    "DualAnnealingOptimizer",
    "get_scipy_optimizer",
    "list_scipy_optimizers",
]
