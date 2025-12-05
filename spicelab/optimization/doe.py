"""Design of Experiments (DoE) for circuit optimization.

This module provides experimental design methods:
- Factorial designs (full and fractional)
- Latin Hypercube Sampling (LHS)
- Sobol sequences (quasi-random)
- Central Composite Design (CCD)
- Box-Behnken design
- Response Surface Methodology (RSM)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np

from .base import ParameterBounds

if TYPE_CHECKING:
    pass


# =============================================================================
# Experimental Design Results
# =============================================================================


@dataclass
class ExperimentalDesign:
    """Container for experimental design points.

    Attributes:
        points: List of parameter dictionaries for each experiment
        design_type: Type of design (factorial, lhs, sobol, etc.)
        n_factors: Number of factors (parameters)
        n_runs: Number of experimental runs
        bounds: Parameter bounds used
    """

    points: list[dict[str, float]]
    design_type: str
    n_factors: int
    n_runs: int
    bounds: list[ParameterBounds]
    metadata: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return self.n_runs

    def __iter__(self) -> Iterator[dict[str, float]]:
        return iter(self.points)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array (n_runs x n_factors)."""
        return np.array([[p[b.name] for b in self.bounds] for p in self.points])

    def to_normalized(self) -> np.ndarray:
        """Convert to normalized [0, 1] array."""
        return np.array(
            [[b.normalize(p[b.name]) for b in self.bounds] for p in self.points]
        )


@dataclass
class DoEResult:
    """Result from running DoE experiments.

    Attributes:
        design: The experimental design used
        responses: Response values for each run
        factor_names: Names of factors
        response_name: Name of response variable
    """

    design: ExperimentalDesign
    responses: np.ndarray
    factor_names: list[str]
    response_name: str = "response"

    @property
    def n_runs(self) -> int:
        return len(self.responses)

    def get_best(self, minimize: bool = True) -> tuple[dict[str, float], float]:
        """Get best experimental point.

        Args:
            minimize: True to find minimum, False for maximum

        Returns:
            Tuple of (parameters, response_value)
        """
        if minimize:
            idx = int(np.argmin(self.responses))
        else:
            idx = int(np.argmax(self.responses))
        return self.design.points[idx], float(self.responses[idx])

    def to_dataframe(self) -> Any:
        """Convert to pandas DataFrame if available."""
        try:
            import pandas as pd

            data: dict[str, Any] = {
                b.name: [p[b.name] for p in self.design.points] for b in self.design.bounds
            }
            data[self.response_name] = list(self.responses)
            return pd.DataFrame(data)
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")


# =============================================================================
# Factorial Designs
# =============================================================================


def full_factorial(
    bounds: list[ParameterBounds],
    levels: int | list[int] = 2,
) -> ExperimentalDesign:
    """Generate full factorial design.

    Args:
        bounds: Parameter bounds
        levels: Number of levels per factor (int for all same, list for different)

    Returns:
        ExperimentalDesign with all combinations

    Example:
        >>> bounds = [ParameterBounds("R1", 1e3, 10e3), ParameterBounds("C1", 1e-9, 10e-9)]
        >>> design = full_factorial(bounds, levels=3)
        >>> print(f"Generated {len(design)} experiments")
    """
    n_factors = len(bounds)

    if isinstance(levels, int):
        levels_per_factor = [levels] * n_factors
    else:
        levels_per_factor = levels

    # Generate level values for each factor
    factor_levels = []
    for i, b in enumerate(bounds):
        n_levels = levels_per_factor[i]
        if n_levels == 2:
            values = [b.lower, b.upper]
        else:
            values = np.linspace(b.lower, b.upper, n_levels).tolist()
        factor_levels.append(values)

    # Generate all combinations
    points = []
    for combo in product(*factor_levels):
        point = {bounds[i].name: combo[i] for i in range(n_factors)}
        points.append(point)

    return ExperimentalDesign(
        points=points,
        design_type=f"full_factorial_{levels}^{n_factors}",
        n_factors=n_factors,
        n_runs=len(points),
        bounds=bounds,
        metadata={"levels": levels_per_factor},
    )


def fractional_factorial(
    bounds: list[ParameterBounds],
    resolution: int = 3,
) -> ExperimentalDesign:
    """Generate fractional factorial design (2^(k-p)).

    Args:
        bounds: Parameter bounds
        resolution: Design resolution (III, IV, or V)

    Returns:
        ExperimentalDesign with reduced runs

    Note:
        Resolution III: Main effects clear of 2-factor interactions
        Resolution IV: Main effects clear of 2-factor, 2-factor aliased
        Resolution V: Main effects and 2-factor interactions clear
    """
    n_factors = len(bounds)

    # Determine fraction based on resolution and number of factors
    if n_factors <= 3:
        # Use full factorial for small designs
        return full_factorial(bounds, levels=2)

    # Generate base 2^k design
    if resolution == 3:
        # Resolution III: highly fractionated
        n_base = max(3, int(np.ceil(np.log2(n_factors + 1))))
    elif resolution == 4:
        # Resolution IV: moderate fractionation
        n_base = max(4, int(np.ceil(np.log2(n_factors + 1))) + 1)
    else:
        # Resolution V: minimal fractionation
        n_base = min(n_factors, 5)

    n_runs = 2**n_base

    # Generate base design
    base_design = np.zeros((n_runs, n_factors))
    for i in range(min(n_base, n_factors)):
        pattern = np.tile(
            np.repeat([0, 1], 2 ** (n_base - i - 1)), 2**i
        )[:n_runs]
        base_design[:, i] = pattern

    # Generate additional columns using interactions
    if n_factors > n_base:
        col_idx = n_base
        for i in range(n_base):
            for j in range(i + 1, n_base):
                if col_idx >= n_factors:
                    break
                base_design[:, col_idx] = (base_design[:, i] + base_design[:, j]) % 2
                col_idx += 1

    # Convert to parameter values
    points = []
    for row in base_design:
        point = {}
        for i, b in enumerate(bounds):
            point[b.name] = b.lower if row[i] == 0 else b.upper
        points.append(point)

    return ExperimentalDesign(
        points=points,
        design_type=f"fractional_factorial_2^{n_factors}_res{resolution}",
        n_factors=n_factors,
        n_runs=len(points),
        bounds=bounds,
        metadata={"resolution": resolution, "n_base": n_base},
    )


# =============================================================================
# Latin Hypercube Sampling
# =============================================================================


def latin_hypercube(
    bounds: list[ParameterBounds],
    n_samples: int,
    criterion: str = "maximin",
    seed: int | None = None,
) -> ExperimentalDesign:
    """Generate Latin Hypercube Sample design.

    Args:
        bounds: Parameter bounds
        n_samples: Number of samples
        criterion: Optimization criterion ("random", "maximin", "correlation")
        seed: Random seed

    Returns:
        ExperimentalDesign with LHS points

    Example:
        >>> bounds = [ParameterBounds("R1", 1e3, 10e3), ParameterBounds("C1", 1e-9, 10e-9)]
        >>> design = latin_hypercube(bounds, n_samples=20, criterion="maximin")
    """
    rng = np.random.default_rng(seed)
    n_factors = len(bounds)

    if criterion == "random":
        sample = _lhs_random(n_factors, n_samples, rng)
    elif criterion == "maximin":
        sample = _lhs_maximin(n_factors, n_samples, rng)
    elif criterion == "correlation":
        sample = _lhs_correlation(n_factors, n_samples, rng)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    # Convert to parameter values
    points = []
    for row in sample:
        point = {bounds[i].name: bounds[i].denormalize(row[i]) for i in range(n_factors)}
        points.append(point)

    return ExperimentalDesign(
        points=points,
        design_type=f"lhs_{criterion}",
        n_factors=n_factors,
        n_runs=n_samples,
        bounds=bounds,
        metadata={"criterion": criterion, "seed": seed},
    )


def _lhs_random(n_dims: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random LHS."""
    result = np.zeros((n_samples, n_dims))
    for i in range(n_dims):
        intervals = np.linspace(0, 1, n_samples + 1)
        for j in range(n_samples):
            result[j, i] = rng.uniform(intervals[j], intervals[j + 1])
        rng.shuffle(result[:, i])
    return result


def _lhs_maximin(
    n_dims: int, n_samples: int, rng: np.random.Generator, iterations: int = 100
) -> np.ndarray:
    """Generate maximin LHS (maximize minimum distance between points)."""
    best_sample = _lhs_random(n_dims, n_samples, rng)
    best_min_dist = _min_distance(best_sample)

    for _ in range(iterations):
        candidate = _lhs_random(n_dims, n_samples, rng)
        min_dist = _min_distance(candidate)
        if min_dist > best_min_dist:
            best_sample = candidate
            best_min_dist = min_dist

    return best_sample


def _lhs_correlation(
    n_dims: int, n_samples: int, rng: np.random.Generator, iterations: int = 100
) -> np.ndarray:
    """Generate LHS minimizing correlation between columns."""
    best_sample = _lhs_random(n_dims, n_samples, rng)
    best_corr = _max_correlation(best_sample)

    for _ in range(iterations):
        candidate = _lhs_random(n_dims, n_samples, rng)
        max_corr = _max_correlation(candidate)
        if max_corr < best_corr:
            best_sample = candidate
            best_corr = max_corr

    return best_sample


def _min_distance(sample: np.ndarray) -> float:
    """Calculate minimum Euclidean distance between any two points."""
    n = len(sample)
    min_dist = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((sample[i] - sample[j]) ** 2))
            min_dist = min(min_dist, dist)
    return min_dist


def _max_correlation(sample: np.ndarray) -> float:
    """Calculate maximum absolute correlation between columns."""
    if sample.shape[1] < 2:
        return 0.0
    corr_matrix = np.corrcoef(sample.T)
    np.fill_diagonal(corr_matrix, 0)
    return float(np.max(np.abs(corr_matrix)))


# =============================================================================
# Sobol Sequences
# =============================================================================


def sobol_sequence(
    bounds: list[ParameterBounds],
    n_samples: int,
    skip: int = 0,
    seed: int | None = None,
) -> ExperimentalDesign:
    """Generate Sobol quasi-random sequence.

    Args:
        bounds: Parameter bounds
        n_samples: Number of samples
        skip: Number of initial points to skip
        seed: Random seed for scrambling (None for unscrambled)

    Returns:
        ExperimentalDesign with Sobol points

    Note:
        Sobol sequences provide better space-filling than random sampling,
        especially for high-dimensional problems.
    """
    n_factors = len(bounds)

    # Generate Sobol sequence
    sample = _sobol_sample(n_factors, n_samples + skip, seed)

    # Skip initial points if requested
    if skip > 0:
        sample = sample[skip:]

    # Convert to parameter values
    points = []
    for row in sample:
        point = {bounds[i].name: bounds[i].denormalize(row[i]) for i in range(n_factors)}
        points.append(point)

    return ExperimentalDesign(
        points=points,
        design_type="sobol",
        n_factors=n_factors,
        n_runs=n_samples,
        bounds=bounds,
        metadata={"skip": skip, "seed": seed},
    )


def _sobol_sample(n_dims: int, n_samples: int, seed: int | None = None) -> np.ndarray:
    """Generate Sobol sequence using simple implementation.

    For production use, consider scipy.stats.qmc.Sobol.
    """
    # Try to use scipy if available
    try:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=n_dims, scramble=seed is not None, seed=seed)
        result: np.ndarray = sampler.random(n_samples)
        return result
    except ImportError:
        pass

    # Fallback: simple gray code based quasi-random
    rng = np.random.default_rng(seed)
    result = np.zeros((n_samples, n_dims))

    # Direction numbers for Sobol (simplified)
    for i in range(n_samples):
        for j in range(n_dims):
            # Van der Corput sequence in different bases
            base = _get_prime(j)
            result[i, j] = _van_der_corput(i + 1, base)

    # Add small random perturbation if seed provided
    if seed is not None:
        result += rng.uniform(-0.01, 0.01, result.shape)
        result = np.clip(result, 0, 1)

    return result


def _van_der_corput(n: int, base: int) -> float:
    """Van der Corput sequence element."""
    result = 0.0
    f = 1.0 / base
    i = n
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def _get_prime(index: int) -> int:
    """Get the index-th prime number."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    return primes[index % len(primes)]


# =============================================================================
# Central Composite Design
# =============================================================================


def central_composite(
    bounds: list[ParameterBounds],
    alpha: str = "orthogonal",
    center_points: int = 3,
) -> ExperimentalDesign:
    """Generate Central Composite Design (CCD).

    Args:
        bounds: Parameter bounds
        alpha: Axial point distance ("orthogonal", "rotatable", "face")
        center_points: Number of center points

    Returns:
        ExperimentalDesign with CCD points

    Note:
        CCD is commonly used for Response Surface Methodology (RSM).
        It consists of factorial points, axial points, and center points.
    """
    n_factors = len(bounds)

    # Calculate alpha value
    n_factorial = 2**n_factors
    if alpha == "face":
        alpha_val = 1.0
    elif alpha == "rotatable":
        alpha_val = n_factorial**0.25
    else:  # orthogonal
        alpha_val = np.sqrt(n_factors)

    points = []

    # Factorial points (corners of hypercube)
    for combo in product([0, 1], repeat=n_factors):
        point = {}
        for i, b in enumerate(bounds):
            point[b.name] = b.lower if combo[i] == 0 else b.upper
        points.append(point)

    # Axial points (star points)
    centers = [(b.lower + b.upper) / 2 for b in bounds]
    ranges = [(b.upper - b.lower) / 2 for b in bounds]

    for i, b in enumerate(bounds):
        # Low axial point
        point_low = {bounds[j].name: centers[j] for j in range(n_factors)}
        point_low[b.name] = centers[i] - alpha_val * ranges[i]
        point_low[b.name] = max(point_low[b.name], b.lower)  # Clip to bounds
        points.append(point_low)

        # High axial point
        point_high = {bounds[j].name: centers[j] for j in range(n_factors)}
        point_high[b.name] = centers[i] + alpha_val * ranges[i]
        point_high[b.name] = min(point_high[b.name], b.upper)  # Clip to bounds
        points.append(point_high)

    # Center points
    center_point = {b.name: (b.lower + b.upper) / 2 for b in bounds}
    for _ in range(center_points):
        points.append(center_point.copy())

    return ExperimentalDesign(
        points=points,
        design_type=f"ccd_{alpha}",
        n_factors=n_factors,
        n_runs=len(points),
        bounds=bounds,
        metadata={"alpha": alpha, "alpha_value": alpha_val, "center_points": center_points},
    )


def box_behnken(
    bounds: list[ParameterBounds],
    center_points: int = 3,
) -> ExperimentalDesign:
    """Generate Box-Behnken design.

    Args:
        bounds: Parameter bounds (3-7 factors recommended)
        center_points: Number of center points

    Returns:
        ExperimentalDesign with Box-Behnken points

    Note:
        Box-Behnken designs are efficient for 3-7 factors and avoid
        extreme corners of the design space.
    """
    n_factors = len(bounds)

    if n_factors < 3:
        raise ValueError("Box-Behnken requires at least 3 factors")

    points = []
    centers = [(b.lower + b.upper) / 2 for b in bounds]

    # Generate edge midpoint combinations
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            # Each pair of factors at corners, others at center
            for vi in [bounds[i].lower, bounds[i].upper]:
                for vj in [bounds[j].lower, bounds[j].upper]:
                    point = {bounds[k].name: centers[k] for k in range(n_factors)}
                    point[bounds[i].name] = vi
                    point[bounds[j].name] = vj
                    points.append(point)

    # Center points
    center_point = {b.name: centers[idx] for idx, b in enumerate(bounds)}
    for _ in range(center_points):
        points.append(center_point.copy())

    return ExperimentalDesign(
        points=points,
        design_type="box_behnken",
        n_factors=n_factors,
        n_runs=len(points),
        bounds=bounds,
        metadata={"center_points": center_points},
    )


# =============================================================================
# Response Surface Modeling
# =============================================================================


@dataclass
class ResponseSurface:
    """Fitted response surface model.

    Attributes:
        coefficients: Model coefficients
        factor_names: Names of factors
        model_type: Type of model (linear, quadratic, etc.)
        r_squared: Coefficient of determination
    """

    coefficients: dict[str, float]
    factor_names: list[str]
    model_type: str
    r_squared: float
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))

    def predict(self, params: dict[str, float]) -> float:
        """Predict response for given parameters."""
        result = self.coefficients.get("intercept", 0.0)

        for name in self.factor_names:
            # Linear terms
            if name in self.coefficients:
                result += self.coefficients[name] * params[name]

            # Quadratic terms
            quad_key = f"{name}^2"
            if quad_key in self.coefficients:
                result += self.coefficients[quad_key] * params[name] ** 2

        # Interaction terms
        for i, name_i in enumerate(self.factor_names):
            for name_j in self.factor_names[i + 1 :]:
                inter_key = f"{name_i}*{name_j}"
                if inter_key in self.coefficients:
                    result += self.coefficients[inter_key] * params[name_i] * params[name_j]

        return result

    def get_optimum(
        self,
        bounds: list[ParameterBounds],
        minimize: bool = True,
        n_samples: int = 1000,
    ) -> dict[str, float]:
        """Find optimum of response surface by sampling.

        Args:
            bounds: Parameter bounds
            minimize: True to find minimum
            n_samples: Number of samples for search

        Returns:
            Parameters at optimum
        """
        rng = np.random.default_rng()
        best_params: dict[str, float] | None = None
        best_value = float("inf") if minimize else float("-inf")

        for _ in range(n_samples):
            params = {b.name: rng.uniform(b.lower, b.upper) for b in bounds}
            value = self.predict(params)

            if minimize and value < best_value:
                best_value = value
                best_params = params
            elif not minimize and value > best_value:
                best_value = value
                best_params = params

        return best_params if best_params else {b.name: (b.lower + b.upper) / 2 for b in bounds}


def fit_response_surface(
    doe_result: DoEResult,
    model_type: str = "quadratic",
) -> ResponseSurface:
    """Fit response surface model to DoE results.

    Args:
        doe_result: Results from DoE experiments
        model_type: "linear", "quadratic", or "interaction"

    Returns:
        Fitted ResponseSurface model
    """
    X = doe_result.design.to_array()
    y = doe_result.responses
    factor_names = doe_result.factor_names

    # Build design matrix
    if model_type == "linear":
        X_design = np.column_stack([np.ones(len(X)), X])
        term_names = ["intercept"] + factor_names
    elif model_type == "interaction":
        # Linear + interactions
        terms = [np.ones(len(X))]
        term_names = ["intercept"]
        for i, name in enumerate(factor_names):
            terms.append(X[:, i])
            term_names.append(name)
        for i in range(len(factor_names)):
            for j in range(i + 1, len(factor_names)):
                terms.append(X[:, i] * X[:, j])
                term_names.append(f"{factor_names[i]}*{factor_names[j]}")
        X_design = np.column_stack(terms)
    else:  # quadratic
        terms = [np.ones(len(X))]
        term_names = ["intercept"]
        for i, name in enumerate(factor_names):
            terms.append(X[:, i])
            term_names.append(name)
        for i, name in enumerate(factor_names):
            terms.append(X[:, i] ** 2)
            term_names.append(f"{name}^2")
        for i in range(len(factor_names)):
            for j in range(i + 1, len(factor_names)):
                terms.append(X[:, i] * X[:, j])
                term_names.append(f"{factor_names[i]}*{factor_names[j]}")
        X_design = np.column_stack(terms)

    # Solve least squares
    coeffs, residuals, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

    # Calculate R-squared
    y_pred = X_design @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return ResponseSurface(
        coefficients={term_names[i]: float(coeffs[i]) for i in range(len(coeffs))},
        factor_names=factor_names,
        model_type=model_type,
        r_squared=r_squared,
        residuals=y - y_pred,
    )


# =============================================================================
# DoE Runner
# =============================================================================


def run_doe(
    design: ExperimentalDesign,
    objective: Callable[[dict[str, float]], float],
    response_name: str = "response",
) -> DoEResult:
    """Run experiments for a design.

    Args:
        design: Experimental design
        objective: Function to evaluate
        response_name: Name for response variable

    Returns:
        DoEResult with all responses
    """
    responses = np.array([objective(point) for point in design.points])

    return DoEResult(
        design=design,
        responses=responses,
        factor_names=[b.name for b in design.bounds],
        response_name=response_name,
    )


__all__ = [
    # Design containers
    "ExperimentalDesign",
    "DoEResult",
    # Factorial designs
    "full_factorial",
    "fractional_factorial",
    # Space-filling designs
    "latin_hypercube",
    "sobol_sequence",
    # Response surface designs
    "central_composite",
    "box_behnken",
    # Response surface modeling
    "ResponseSurface",
    "fit_response_surface",
    # Runner
    "run_doe",
]
