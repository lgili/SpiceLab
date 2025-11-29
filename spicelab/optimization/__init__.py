"""Optimization & Design of Experiments for SpiceLab.

This module provides circuit optimization capabilities:
- Parameter optimization with scipy (Nelder-Mead, L-BFGS-B, DE, etc.)
- Multi-objective optimization with genetic algorithms
- Design of experiments (DoE) for sensitivity analysis
- Corner and PVT analysis

Quick Start:
    >>> from spicelab.optimization import CircuitOptimizer
    >>> from spicelab.measurements import measure
    >>>
    >>> # Create optimizer
    >>> opt = CircuitOptimizer(circuit, simulate_func)
    >>> opt.add_parameter("R1", 1e3, 100e3, scale="log")
    >>> opt.add_parameter("C1", 1e-12, 1e-9, scale="log")
    >>>
    >>> # Define objective
    >>> def objective(params, dataset):
    ...     bw = measure(dataset, "bandwidth", output_node="vout")
    ...     return -bw.value  # Maximize bandwidth
    >>>
    >>> # Run optimization
    >>> result = opt.optimize(objective, method="nelder-mead")
    >>> print(f"Optimal: {result.parameters}")

Available Optimizers:
    Local (scipy):
        - nelder-mead: Simplex method (gradient-free)
        - powell: Conjugate direction method
        - l-bfgs-b: Limited-memory BFGS with bounds

    Global (scipy):
        - differential-evolution: Evolutionary algorithm
        - dual-annealing: Simulated annealing variant

Classes:
    CircuitOptimizer: High-level interface for circuit optimization
    CircuitObjective: Connects circuit simulation to objective function
    ParameterBounds: Defines optimization parameter bounds and scale
    OptimizationResult: Container for optimization results
    OptimizationConfig: Configuration for optimization runs
"""

from __future__ import annotations

from .base import (
    CircuitObjective,
    CircuitOptimizer,
    Constraint,
    ObjectiveFunction,
    OptimizationConfig,
    OptimizationResult,
    Optimizer,
    ParameterBounds,
)
from .scipy_optimizers import (
    DifferentialEvolutionOptimizer,
    DualAnnealingOptimizer,
    LBFGSBOptimizer,
    NelderMeadOptimizer,
    PowellOptimizer,
    get_scipy_optimizer,
    list_scipy_optimizers,
)

__all__ = [
    # Core classes
    "CircuitOptimizer",
    "CircuitObjective",
    "ParameterBounds",
    "OptimizationResult",
    "OptimizationConfig",
    "Optimizer",
    "ObjectiveFunction",
    "Constraint",
    # Scipy optimizers
    "NelderMeadOptimizer",
    "PowellOptimizer",
    "LBFGSBOptimizer",
    "DifferentialEvolutionOptimizer",
    "DualAnnealingOptimizer",
    "get_scipy_optimizer",
    "list_scipy_optimizers",
]
