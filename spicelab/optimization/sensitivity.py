"""Sensitivity analysis methods for circuit optimization.

This module provides sensitivity analysis capabilities:
- Morris method (one-at-a-time screening)
- Sobol variance-based sensitivity indices
- Local sensitivity (gradient-based)
- Tornado plot generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from .base import ParameterBounds

if TYPE_CHECKING:
    pass


# =============================================================================
# Sensitivity Analysis Results
# =============================================================================


@dataclass
class MorrisResult:
    """Results from Morris sensitivity analysis.

    Attributes:
        mu: Mean of elementary effects (measure of importance)
        mu_star: Mean of absolute elementary effects (robust importance)
        sigma: Standard deviation of effects (measure of interaction/nonlinearity)
        factor_names: Names of factors analyzed
        n_trajectories: Number of trajectories used
    """

    mu: dict[str, float]
    mu_star: dict[str, float]
    sigma: dict[str, float]
    factor_names: list[str]
    n_trajectories: int
    elementary_effects: dict[str, list[float]] = field(default_factory=dict)

    def get_ranking(self, by: str = "mu_star") -> list[str]:
        """Get factors ranked by sensitivity.

        Args:
            by: Metric to rank by ("mu", "mu_star", "sigma")

        Returns:
            Factor names sorted by decreasing sensitivity
        """
        metric = getattr(self, by)
        return sorted(self.factor_names, key=lambda x: abs(metric[x]), reverse=True)

    def is_influential(self, factor: str, threshold: float = 0.1) -> bool:
        """Check if a factor is influential.

        Args:
            factor: Factor name
            threshold: Relative threshold (fraction of max mu_star)

        Returns:
            True if factor is influential
        """
        max_mu_star = max(abs(v) for v in self.mu_star.values())
        if max_mu_star == 0:
            return False
        return abs(self.mu_star[factor]) / max_mu_star >= threshold

    def has_interactions(self, factor: str, threshold: float = 0.5) -> bool:
        """Check if factor has significant interactions or nonlinearities.

        A high sigma/mu_star ratio indicates interactions or nonlinearity.

        Args:
            factor: Factor name
            threshold: Sigma/mu_star ratio threshold

        Returns:
            True if interactions/nonlinearity detected
        """
        if abs(self.mu_star[factor]) < 1e-10:
            return False
        return self.sigma[factor] / abs(self.mu_star[factor]) > threshold

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Convert to dictionary."""
        return {
            "mu": self.mu,
            "mu_star": self.mu_star,
            "sigma": self.sigma,
        }


@dataclass
class SobolResult:
    """Results from Sobol sensitivity analysis.

    Attributes:
        S1: First-order sensitivity indices (main effects)
        ST: Total-order sensitivity indices (main + interactions)
        S2: Second-order indices (pairwise interactions), optional
        factor_names: Names of factors
        n_samples: Number of samples used
        confidence: Confidence intervals (if computed)
    """

    S1: dict[str, float]
    ST: dict[str, float]
    factor_names: list[str]
    n_samples: int
    S2: dict[tuple[str, str], float] | None = None
    confidence: dict[str, tuple[float, float]] | None = None

    def get_main_effects_ranking(self) -> list[str]:
        """Get factors ranked by first-order effects."""
        return sorted(self.factor_names, key=lambda x: self.S1[x], reverse=True)

    def get_total_effects_ranking(self) -> list[str]:
        """Get factors ranked by total effects."""
        return sorted(self.factor_names, key=lambda x: self.ST[x], reverse=True)

    def get_interaction_strength(self, factor: str) -> float:
        """Get interaction strength for a factor (ST - S1).

        Higher values indicate more influence from interactions.
        """
        return self.ST[factor] - self.S1[factor]

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Convert to dictionary."""
        result = {
            "S1": self.S1,
            "ST": self.ST,
        }
        if self.S2:
            result["S2"] = {f"{k[0]}*{k[1]}": v for k, v in self.S2.items()}
        return result


@dataclass
class LocalSensitivity:
    """Local sensitivity analysis results (gradient-based).

    Attributes:
        gradients: Partial derivatives at the point
        normalized_gradients: Elasticity (% change in output / % change in input)
        point: Parameter values where sensitivity was computed
        factor_names: Names of factors
    """

    gradients: dict[str, float]
    normalized_gradients: dict[str, float]
    point: dict[str, float]
    factor_names: list[str]
    output_value: float = 0.0

    def get_ranking(self, normalized: bool = True) -> list[str]:
        """Get factors ranked by sensitivity."""
        metric = self.normalized_gradients if normalized else self.gradients
        return sorted(self.factor_names, key=lambda x: abs(metric[x]), reverse=True)


# =============================================================================
# Morris Method (Elementary Effects)
# =============================================================================


def morris_analysis(
    objective: Callable[[dict[str, float]], float],
    bounds: list[ParameterBounds],
    n_trajectories: int = 10,
    n_levels: int = 4,
    seed: int | None = None,
) -> MorrisResult:
    """Perform Morris sensitivity analysis (method of elementary effects).

    Morris method is a screening method that identifies which factors are:
    - Negligible (low mu_star)
    - Linear and additive (high mu_star, low sigma)
    - Nonlinear or involved in interactions (high mu_star, high sigma)

    Args:
        objective: Function to analyze
        bounds: Parameter bounds
        n_trajectories: Number of trajectories (more = better estimates)
        n_levels: Number of levels for discretization (typically 4-10)
        seed: Random seed

    Returns:
        MorrisResult with sensitivity measures

    Example:
        >>> def circuit_gain(params):
        ...     return params["R1"] / params["R2"] * params["gain_factor"]
        >>> bounds = [ParameterBounds("R1", 1e3, 10e3), ...]
        >>> result = morris_analysis(circuit_gain, bounds, n_trajectories=20)
        >>> print(f"Most important: {result.get_ranking()[0]}")
    """
    rng = np.random.default_rng(seed)
    n_factors = len(bounds)
    factor_names = [b.name for b in bounds]

    # Grid step size
    delta = n_levels / (2 * (n_levels - 1))

    # Store elementary effects for each factor
    elementary_effects: dict[str, list[float]] = {name: [] for name in factor_names}

    for _ in range(n_trajectories):
        # Generate random trajectory
        trajectory = _generate_morris_trajectory(n_factors, n_levels, delta, rng)

        # Evaluate along trajectory and compute elementary effects
        prev_value = None
        prev_point = None

        for i, point in enumerate(trajectory):
            # Convert to parameter values
            params = {bounds[j].name: bounds[j].denormalize(point[j]) for j in range(n_factors)}
            value = objective(params)

            if prev_value is not None:
                # Find which factor changed
                diff = point - prev_point
                changed_idx = np.argmax(np.abs(diff))
                changed_name = factor_names[changed_idx]

                # Compute elementary effect
                # EE = (f(x + delta) - f(x)) / delta
                ee = (value - prev_value) / delta
                elementary_effects[changed_name].append(ee)

            prev_value = value
            prev_point = point.copy()

    # Compute statistics
    mu = {}
    mu_star = {}
    sigma = {}

    for name in factor_names:
        effects = elementary_effects[name]
        if effects:
            mu[name] = float(np.mean(effects))
            mu_star[name] = float(np.mean(np.abs(effects)))
            sigma[name] = float(np.std(effects))
        else:
            mu[name] = 0.0
            mu_star[name] = 0.0
            sigma[name] = 0.0

    return MorrisResult(
        mu=mu,
        mu_star=mu_star,
        sigma=sigma,
        factor_names=factor_names,
        n_trajectories=n_trajectories,
        elementary_effects=elementary_effects,
    )


def _generate_morris_trajectory(
    n_factors: int,
    n_levels: int,
    delta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a single Morris trajectory.

    Returns array of shape (n_factors + 1, n_factors).
    """
    # Start from random point on grid
    levels = np.linspace(0, 1, n_levels)
    start = rng.choice(levels[: n_levels // 2], size=n_factors)

    # Generate trajectory by moving one factor at a time
    trajectory = [start.copy()]

    # Random order of factors to change
    order = rng.permutation(n_factors)

    current = start.copy()
    for idx in order:
        # Move factor by +/- delta
        if rng.random() < 0.5 and current[idx] + delta <= 1:
            current[idx] += delta
        elif current[idx] - delta >= 0:
            current[idx] -= delta
        else:
            current[idx] += delta

        trajectory.append(current.copy())

    return np.array(trajectory)


# =============================================================================
# Sobol Sensitivity Analysis
# =============================================================================


def sobol_analysis(
    objective: Callable[[dict[str, float]], float],
    bounds: list[ParameterBounds],
    n_samples: int = 1024,
    calc_second_order: bool = False,
    seed: int | None = None,
) -> SobolResult:
    """Perform Sobol variance-based sensitivity analysis.

    Sobol method decomposes output variance into contributions from each
    input factor and their interactions.

    Args:
        objective: Function to analyze
        bounds: Parameter bounds
        n_samples: Base sample size (total evaluations = n_samples * (2*k + 2))
        calc_second_order: Whether to compute second-order indices
        seed: Random seed

    Returns:
        SobolResult with sensitivity indices

    Note:
        - S1 (first-order): Main effect of each factor
        - ST (total-order): Main effect + all interactions involving the factor
        - S2 (second-order): Pairwise interaction effects

    Example:
        >>> result = sobol_analysis(circuit_func, bounds, n_samples=2048)
        >>> print(f"Main effects: {result.S1}")
        >>> print(f"Total effects: {result.ST}")
    """
    rng = np.random.default_rng(seed)
    n_factors = len(bounds)
    factor_names = [b.name for b in bounds]

    # Generate Saltelli sample matrices
    A, B, AB = _generate_saltelli_samples(n_factors, n_samples, rng)

    # Evaluate model
    def evaluate(samples: np.ndarray) -> np.ndarray:
        results = []
        for row in samples:
            params = {bounds[i].name: bounds[i].denormalize(row[i]) for i in range(n_factors)}
            results.append(objective(params))
        return np.array(results)

    y_A = evaluate(A)
    y_B = evaluate(B)
    y_AB = [evaluate(ab) for ab in AB]

    # Compute indices using Saltelli estimator
    S1 = {}
    ST = {}

    total_var = np.var(np.concatenate([y_A, y_B]))
    if total_var < 1e-10:
        # No variance - all indices are zero
        for name in factor_names:
            S1[name] = 0.0
            ST[name] = 0.0
        return SobolResult(
            S1=S1,
            ST=ST,
            factor_names=factor_names,
            n_samples=n_samples,
        )

    for i, name in enumerate(factor_names):
        # First-order index: S1 = V[E[Y|Xi]] / V[Y]
        # Saltelli estimator
        S1[name] = float(np.mean(y_B * (y_AB[i] - y_A)) / total_var)

        # Total-order index: ST = E[V[Y|X~i]] / V[Y]
        ST[name] = float(np.mean((y_A - y_AB[i]) ** 2) / (2 * total_var))

        # Clip to valid range
        S1[name] = max(0, min(1, S1[name]))
        ST[name] = max(0, min(1, ST[name]))

    # Second-order indices (optional)
    S2 = None
    if calc_second_order and n_factors > 1:
        S2 = {}
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                # S2_ij = S_ij - S_i - S_j
                # Approximate using Saltelli
                s2_val = (
                    np.mean(y_AB[j] * (y_AB[i] - y_A)) / total_var
                    - S1[factor_names[i]]
                    - S1[factor_names[j]]
                )
                S2[(factor_names[i], factor_names[j])] = float(max(0, s2_val))

    return SobolResult(
        S1=S1,
        ST=ST,
        factor_names=factor_names,
        n_samples=n_samples,
        S2=S2,
    )


def _generate_saltelli_samples(
    n_factors: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Generate Saltelli sample matrices for Sobol analysis.

    Returns:
        A: Base sample matrix A
        B: Base sample matrix B
        AB: List of matrices where AB[i] has column i from B, rest from A
    """
    # Generate two independent sample matrices
    A = rng.random((n_samples, n_factors))
    B = rng.random((n_samples, n_factors))

    # Create AB matrices
    AB = []
    for i in range(n_factors):
        ab = A.copy()
        ab[:, i] = B[:, i]
        AB.append(ab)

    return A, B, AB


# =============================================================================
# Local Sensitivity Analysis
# =============================================================================


def local_sensitivity(
    objective: Callable[[dict[str, float]], float],
    bounds: list[ParameterBounds],
    point: dict[str, float] | None = None,
    step_size: float = 0.01,
) -> LocalSensitivity:
    """Compute local sensitivity (partial derivatives) at a point.

    Uses central difference approximation for gradients.

    Args:
        objective: Function to analyze
        bounds: Parameter bounds
        point: Point to evaluate at (default: center of bounds)
        step_size: Relative step size for finite difference

    Returns:
        LocalSensitivity with gradients

    Example:
        >>> result = local_sensitivity(circuit_func, bounds, point=optimal_params)
        >>> print(f"Most sensitive: {result.get_ranking()[0]}")
    """
    factor_names = [b.name for b in bounds]

    if point is None:
        point = {b.name: (b.lower + b.upper) / 2 for b in bounds}

    base_value = objective(point)

    gradients = {}
    normalized_gradients = {}

    for b in bounds:
        name = b.name
        x0 = point[name]

        # Step size
        h = abs(x0 * step_size) if x0 != 0 else step_size * (b.upper - b.lower)

        # Central difference
        point_plus = point.copy()
        point_plus[name] = x0 + h
        point_minus = point.copy()
        point_minus[name] = x0 - h

        y_plus = objective(point_plus)
        y_minus = objective(point_minus)

        grad = (y_plus - y_minus) / (2 * h)
        gradients[name] = grad

        # Normalized gradient (elasticity): (dy/y) / (dx/x) = (dy/dx) * (x/y)
        if abs(base_value) > 1e-10 and abs(x0) > 1e-10:
            normalized_gradients[name] = grad * x0 / base_value
        else:
            normalized_gradients[name] = grad

    return LocalSensitivity(
        gradients=gradients,
        normalized_gradients=normalized_gradients,
        point=point,
        factor_names=factor_names,
        output_value=base_value,
    )


# =============================================================================
# One-at-a-Time (OAT) Analysis
# =============================================================================


@dataclass
class OATResult:
    """Results from one-at-a-time sensitivity analysis.

    Attributes:
        sweeps: Dict mapping factor name to (param_values, response_values)
        baseline: Baseline response at center point
        factor_names: Names of factors
    """

    sweeps: dict[str, tuple[np.ndarray, np.ndarray]]
    baseline: float
    baseline_point: dict[str, float]
    factor_names: list[str]

    def get_range(self, factor: str) -> float:
        """Get response range for a factor."""
        _, responses = self.sweeps[factor]
        return float(np.max(responses) - np.min(responses))

    def get_sensitivity(self, factor: str) -> float:
        """Get linear sensitivity estimate for a factor."""
        params, responses = self.sweeps[factor]
        # Linear regression slope
        slope = np.polyfit(params, responses, 1)[0]
        return float(slope)

    def get_ranking(self) -> list[str]:
        """Get factors ranked by response range."""
        return sorted(self.factor_names, key=lambda x: self.get_range(x), reverse=True)


def oat_analysis(
    objective: Callable[[dict[str, float]], float],
    bounds: list[ParameterBounds],
    n_points: int = 11,
    baseline: dict[str, float] | None = None,
) -> OATResult:
    """Perform one-at-a-time sensitivity analysis.

    Sweeps each parameter while holding others at baseline.
    Simple but doesn't capture interactions.

    Args:
        objective: Function to analyze
        bounds: Parameter bounds
        n_points: Number of points per sweep
        baseline: Baseline point (default: center)

    Returns:
        OATResult with sweep data
    """
    factor_names = [b.name for b in bounds]

    if baseline is None:
        baseline = {b.name: (b.lower + b.upper) / 2 for b in bounds}

    baseline_value = objective(baseline)

    sweeps = {}
    for b in bounds:
        param_values = np.linspace(b.lower, b.upper, n_points)
        responses = []

        for val in param_values:
            point = baseline.copy()
            point[b.name] = val
            responses.append(objective(point))

        sweeps[b.name] = (param_values, np.array(responses))

    return OATResult(
        sweeps=sweeps,
        baseline=baseline_value,
        baseline_point=baseline,
        factor_names=factor_names,
    )


# =============================================================================
# Visualization Helpers
# =============================================================================


def generate_tornado_data(
    result: MorrisResult | SobolResult | LocalSensitivity,
) -> dict[str, list[tuple[str, float, float]]]:
    """Generate data for tornado plot.

    Args:
        result: Sensitivity analysis result

    Returns:
        Dict with 'data' key containing list of (name, low_impact, high_impact)
    """
    data = []

    if isinstance(result, MorrisResult):
        for name in result.get_ranking():
            # Use mu_star for magnitude, sigma for spread indication
            value = result.mu_star[name]
            spread = result.sigma[name]
            data.append((name, -value - spread / 2, value + spread / 2))

    elif isinstance(result, SobolResult):
        for name in result.get_total_effects_ranking():
            s1 = result.S1[name]
            st = result.ST[name]
            # S1 is main effect, ST-S1 is interaction contribution
            data.append((name, s1, st))

    elif isinstance(result, LocalSensitivity):
        for name in result.get_ranking():
            value = result.normalized_gradients[name]
            data.append((name, min(0, value), max(0, value)))

    return {"data": data}


def print_sensitivity_report(
    result: MorrisResult | SobolResult,
    top_n: int = 10,
) -> str:
    """Generate text report of sensitivity analysis.

    Args:
        result: Sensitivity analysis result
        top_n: Number of top factors to show

    Returns:
        Formatted report string
    """
    lines = []

    if isinstance(result, MorrisResult):
        lines.append("Morris Sensitivity Analysis Report")
        lines.append("=" * 40)
        lines.append(f"Trajectories: {result.n_trajectories}")
        lines.append("")
        lines.append(f"{'Factor':<15} {'μ*':>10} {'σ':>10} {'μ':>10}")
        lines.append("-" * 45)

        for name in result.get_ranking()[:top_n]:
            lines.append(
                f"{name:<15} {result.mu_star[name]:>10.4f} "
                f"{result.sigma[name]:>10.4f} {result.mu[name]:>10.4f}"
            )

        lines.append("")
        lines.append("Interpretation:")
        for name in result.get_ranking()[:3]:
            if result.has_interactions(name):
                lines.append(f"  - {name}: Important with interactions/nonlinearity")
            elif result.is_influential(name):
                lines.append(f"  - {name}: Important, approximately linear")

    elif isinstance(result, SobolResult):
        lines.append("Sobol Sensitivity Analysis Report")
        lines.append("=" * 40)
        lines.append(f"Samples: {result.n_samples}")
        lines.append("")
        lines.append(f"{'Factor':<15} {'S1':>10} {'ST':>10} {'Interaction':>12}")
        lines.append("-" * 47)

        for name in result.get_total_effects_ranking()[:top_n]:
            interaction = result.get_interaction_strength(name)
            lines.append(
                f"{name:<15} {result.S1[name]:>10.4f} "
                f"{result.ST[name]:>10.4f} {interaction:>12.4f}"
            )

    return "\n".join(lines)


__all__ = [
    # Results
    "MorrisResult",
    "SobolResult",
    "LocalSensitivity",
    "OATResult",
    # Analysis methods
    "morris_analysis",
    "sobol_analysis",
    "local_sensitivity",
    "oat_analysis",
    # Visualization
    "generate_tornado_data",
    "print_sensitivity_report",
]
