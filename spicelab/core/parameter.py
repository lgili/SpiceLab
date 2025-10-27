"""First-class parameter system for circuit simulation.

Phase 2 (P2.1) of architecture redesign: replace stringly-typed component values
with typed Parameter objects that include:
- Units (Ω, F, H, V, A)
- Tolerances (for Monte Carlo)
- Expressions (dependent parameters)
- Validation

This eliminates ambiguity between "1k" (string), 1000.0 (float), and "{R1}" (param ref).

Example:
    >>> from spicelab.core.parameter import Parameter, NormalTolerance
    >>> from spicelab.core.units import Unit
    >>>
    >>> R_load = Parameter(
    ...     name="Rload",
    ...     nominal=10_000,
    ...     unit=Unit.OHM,
    ...     tolerance=NormalTolerance(sigma_pct=5.0)
    ... )
    >>> R_load.sample_value(seed=42)
    10234.5  # 10kΩ ± 5% normal distribution
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

from .units import Unit

__all__ = [
    "Parameter",
    "ParameterRef",
    "Tolerance",
    "NormalTolerance",
    "UniformTolerance",
    "LogNormalTolerance",
    "TriangularTolerance",
    "Unit",  # Re-export for convenience
]


# ----------------------------------------------------------------------------------
# Tolerance Abstractions
# ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Tolerance(ABC):
    """Base class for parameter tolerance/variation specifications.

    Tolerances define how parameters vary in Monte Carlo simulations.
    Subclasses implement different probability distributions.
    """

    @abstractmethod
    def sample(self, nominal: float, rng: np.random.Generator) -> float:
        """Sample a value from the distribution.

        Args:
            nominal: Nominal (center) value
            rng: NumPy random generator (for reproducibility)

        Returns:
            Sampled value according to distribution
        """
        ...

    @abstractmethod
    def bounds(self, nominal: float) -> tuple[float, float]:
        """Return (min, max) bounds for the distribution.

        Used for validation and visualization.
        """
        ...


@dataclass(frozen=True)
class NormalTolerance(Tolerance):
    """Gaussian (normal) distribution tolerance.

    Commonly used for component tolerances (resistors, capacitors).

    Args:
        sigma_pct: Standard deviation as percentage of nominal (e.g., 5.0 for ±5%)
        n_sigma_bounds: Number of sigmas for bounds (default 3σ ≈ 99.7%)

    Example:
        >>> tol = NormalTolerance(sigma_pct=5.0)
        >>> # 68% of samples within ±5%, 95% within ±10%, 99.7% within ±15%
    """

    sigma_pct: float
    n_sigma_bounds: float = 3.0

    def sample(self, nominal: float, rng: np.random.Generator) -> float:
        """Sample from normal distribution."""
        sigma_abs = nominal * (self.sigma_pct / 100.0)
        return rng.normal(loc=nominal, scale=sigma_abs)

    def bounds(self, nominal: float) -> tuple[float, float]:
        """Return ±3σ bounds (99.7% coverage)."""
        delta = nominal * (self.sigma_pct / 100.0) * self.n_sigma_bounds
        return (nominal - delta, nominal + delta)


@dataclass(frozen=True)
class UniformTolerance(Tolerance):
    """Uniform (flat) distribution tolerance.

    All values within range equally likely.

    Args:
        plus_minus_pct: Range as ±percentage of nominal (e.g., 10.0 for ±10%)

    Example:
        >>> tol = UniformTolerance(plus_minus_pct=10.0)
        >>> # All values in [0.9×nominal, 1.1×nominal] equally likely
    """

    plus_minus_pct: float

    def sample(self, nominal: float, rng: np.random.Generator) -> float:
        """Sample from uniform distribution."""
        delta = nominal * (self.plus_minus_pct / 100.0)
        return rng.uniform(nominal - delta, nominal + delta)

    def bounds(self, nominal: float) -> tuple[float, float]:
        """Return exact min/max bounds."""
        delta = nominal * (self.plus_minus_pct / 100.0)
        return (nominal - delta, nominal + delta)


@dataclass(frozen=True)
class LogNormalTolerance(Tolerance):
    """Log-normal distribution tolerance.

    Used for parameters that must be positive and have multiplicative errors
    (e.g., gain, bandwidth).

    Args:
        sigma_pct: Approximate standard deviation as percentage

    Note:
        Log-normal is asymmetric: more spread on high side than low side.
    """

    sigma_pct: float

    def sample(self, nominal: float, rng: np.random.Generator) -> float:
        """Sample from log-normal distribution."""
        import numpy as np

        # Convert percentage to log-space sigma
        sigma_log = np.log1p(self.sigma_pct / 100.0)
        mu_log = np.log(nominal) - 0.5 * sigma_log**2  # Adjust mean

        return np.exp(rng.normal(loc=mu_log, scale=sigma_log))

    def bounds(self, nominal: float) -> tuple[float, float]:
        """Return approximate ±3σ bounds (asymmetric)."""
        import numpy as np

        sigma_log = np.log1p(self.sigma_pct / 100.0)
        mu_log = np.log(nominal) - 0.5 * sigma_log**2

        lower = np.exp(mu_log - 3 * sigma_log)
        upper = np.exp(mu_log + 3 * sigma_log)
        return (lower, upper)


@dataclass(frozen=True)
class TriangularTolerance(Tolerance):
    """Triangular distribution tolerance.

    Peak probability at nominal, linearly decreasing to bounds.
    Often used when normal distribution is too pessimistic.

    Args:
        plus_minus_pct: Range as ±percentage of nominal
    """

    plus_minus_pct: float

    def sample(self, nominal: float, rng: np.random.Generator) -> float:
        """Sample from triangular distribution."""
        delta = nominal * (self.plus_minus_pct / 100.0)
        return rng.triangular(nominal - delta, nominal, nominal + delta)

    def bounds(self, nominal: float) -> tuple[float, float]:
        """Return exact min/max bounds."""
        delta = nominal * (self.plus_minus_pct / 100.0)
        return (nominal - delta, nominal + delta)


# ----------------------------------------------------------------------------------
# Parameter Reference (for component values that reference params)
# ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class ParameterRef:
    """Reference to a circuit parameter by name.

    Used in components to indicate value comes from .param statement:
        R("R1", "a", "b", resistance=ParameterRef("Rload"))
        → Netlist: RR1 a b {Rload}

    Attributes:
        name: Parameter name (must exist in circuit's params dict)
    """

    name: str

    def __str__(self) -> str:
        return f"{{{self.name}}}"

    def __repr__(self) -> str:
        return f"ParameterRef({self.name!r})"


# ----------------------------------------------------------------------------------
# Parameter (First-Class)
# ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Parameter:
    """First-class circuit parameter with units, tolerance, and expressions.

    Replaces stringly-typed component values with fully typed parameters.

    Attributes:
        name: Parameter identifier (e.g., "Rload", "Vdd", "tau")
        nominal: Nominal value in SI units (e.g., 10000 for 10kΩ)
        unit: Physical unit (OHM, FARAD, VOLT, etc.)
        tolerance: Optional variation spec for Monte Carlo
        expression: Optional expression (e.g., "2*R1 + C1")
        description: Optional human-readable description

    Example:
        >>> R_load = Parameter(
        ...     name="Rload",
        ...     nominal=10_000,
        ...     unit=Unit.OHM,
        ...     tolerance=NormalTolerance(5.0),
        ...     description="Load resistance"
        ... )
    """

    name: str
    nominal: float
    unit: Unit
    tolerance: Tolerance | None = None
    expression: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate parameter definition."""
        if self.nominal < 0 and self.unit.name not in ("VOLT", "AMPERE", "CELSIUS"):
            # Most units must be positive (except voltage, current, temperature)
            raise ValueError(
                f"Parameter {self.name}: nominal must be >= 0 for unit {self.unit.name}"
            )

        if self.expression and self.tolerance:
            raise ValueError(
                f"Parameter {self.name}: cannot have both expression and tolerance "
                "(dependent params don't have independent variation)"
            )

    def evaluate(self, context: Mapping[str, float]) -> float:
        """Evaluate parameter value in context.

        If expression is set, evaluates it using context.
        Otherwise returns nominal value.

        Args:
            context: Dict mapping parameter names to values

        Returns:
            Evaluated value (in SI units)

        Raises:
            KeyError: If expression references undefined parameter
            ValueError: If expression is invalid
        """
        if self.expression:
            from .expressions import safe_eval_expression

            return safe_eval_expression(self.expression, context)
        return self.nominal

    def sample_value(
        self, rng: np.random.Generator | None = None, seed: int | None = None
    ) -> float:
        """Sample a value according to tolerance distribution.

        Args:
            rng: NumPy random generator (for reproducibility)
            seed: Random seed (alternative to rng)

        Returns:
            Sampled value (in SI units)

        Raises:
            ValueError: If parameter has expression (use evaluate instead)
        """
        if self.expression:
            raise ValueError(f"Parameter {self.name} has expression, use evaluate() instead")

        if not self.tolerance:
            return self.nominal

        # Create RNG if not provided
        if rng is None:
            import numpy as np

            rng = np.random.default_rng(seed)

        return self.tolerance.sample(self.nominal, rng)

    def to_spice(self) -> str:
        """Generate SPICE .param statement.

        Returns:
            SPICE directive like ".param Rload=10k"
        """
        from .units import format_si_value

        if self.expression:
            return f".param {self.name}={self.expression}"

        value_str = format_si_value(self.nominal, self.unit)
        return f".param {self.name}={value_str}"

    def __str__(self) -> str:
        """Human-readable representation."""
        from .units import format_si_value

        value_str = format_si_value(self.nominal, self.unit)

        if self.tolerance:
            tol_str = (
                f" ±{self.tolerance.sigma_pct if hasattr(self.tolerance, 'sigma_pct') else '?'}%"
            )
        else:
            tol_str = ""

        if self.expression:
            return f"{self.name} = {self.expression} ({value_str}){tol_str}"

        return f"{self.name} = {value_str}{tol_str}"

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Parameter(name={self.name!r}, nominal={self.nominal}, "
            f"unit={self.unit.name}, tolerance={self.tolerance!r})"
        )
