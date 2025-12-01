"""Base classes and protocols for the measurement framework.

This module defines the core abstractions:
- MeasurementResult: Structured result with value, unit, pass/fail, and metadata
- Measurement: Protocol that all measurement classes must implement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    import xarray as xr

T = TypeVar("T", bound=float | int | list[float])


@dataclass
class MeasurementResult(Generic[T]):
    """Result of a measurement with metadata.

    Attributes:
        value: The measured value (float, int, or list of floats)
        unit: Unit of measurement (e.g., "dB", "Hz", "V", "%")
        passed: Whether the measurement passed spec check (None if not checked)
        metadata: Additional information about the measurement

    Example:
        >>> result = MeasurementResult(
        ...     value=60.5,
        ...     unit="dB",
        ...     passed=True,
        ...     metadata={"frequency": 1000, "output_node": "vout"}
        ... )
        >>> print(f"Gain: {result.value} {result.unit}")
        Gain: 60.5 dB
    """

    value: T
    unit: str
    passed: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable representation."""
        status = ""
        if self.passed is True:
            status = " [PASS]"
        elif self.passed is False:
            status = " [FAIL]"
        return f"{self.value} {self.unit}{status}"

    def __repr__(self) -> str:
        return (
            f"MeasurementResult(value={self.value!r}, unit={self.unit!r}, "
            f"passed={self.passed!r}, metadata={self.metadata!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "value": self.value,
            "unit": self.unit,
            "passed": self.passed,
            "metadata": self.metadata,
        }


@runtime_checkable
class Measurement(Protocol):
    """Protocol that all measurement classes must implement.

    A measurement extracts a specific metric from simulation results
    and optionally checks it against a specification.

    Attributes:
        name: Unique identifier for the measurement type
        description: Human-readable description
        required_analyses: List of analysis types needed (e.g., ["ac"], ["tran"])

    Example:
        >>> class MyMeasurement:
        ...     name = "my_measurement"
        ...     description = "Measures something useful"
        ...     required_analyses = ["tran"]
        ...
        ...     def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        ...         value = float(dataset["vout"].max())
        ...         return MeasurementResult(value=value, unit="V")
        ...
        ...     def check_spec(self, result: MeasurementResult, spec: dict) -> bool:
        ...         if "min" in spec and result.value < spec["min"]:
        ...             return False
        ...         if "max" in spec and result.value > spec["max"]:
        ...             return False
        ...         return True
    """

    name: str
    description: str
    required_analyses: list[str]

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[Any]:
        """Extract measurement from simulation results.

        Args:
            dataset: xarray Dataset containing simulation results

        Returns:
            MeasurementResult with the extracted value

        Raises:
            ValueError: If required data is missing or invalid
        """
        ...

    def check_spec(self, result: MeasurementResult[Any], spec: dict[str, Any]) -> bool:
        """Check if measurement meets specification.

        Args:
            result: The measurement result to check
            spec: Specification dict with keys like "min", "max", "target", etc.

        Returns:
            True if the measurement passes the spec, False otherwise
        """
        ...


class BaseMeasurement:
    """Base class providing common functionality for measurements.

    Subclasses should override `measure()` and optionally `check_spec()`.
    The default `check_spec()` implementation supports min/max bounds.

    Attributes:
        name: Measurement name (set by subclass or decorator)
        description: Measurement description (set by subclass or decorator)
        required_analyses: Required analysis types (set by subclass)
    """

    name: str = "base"
    description: str = "Base measurement"
    required_analyses: list[str] = []

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[Any]:
        """Extract measurement from dataset. Override in subclass."""
        raise NotImplementedError("Subclasses must implement measure()")

    def check_spec(self, result: MeasurementResult[Any], spec: dict[str, Any]) -> bool:
        """Check result against spec. Default: min/max bounds checking.

        Args:
            result: Measurement result to check
            spec: Dict with optional "min", "max", "target", "tolerance" keys

        Returns:
            True if within bounds, False otherwise
        """
        value = result.value
        if not isinstance(value, (int, float)):
            # For non-numeric results, subclass should override
            return True

        if "min" in spec and value < spec["min"]:
            return False
        if "max" in spec and value > spec["max"]:
            return False
        if "target" in spec and "tolerance" in spec:
            target = spec["target"]
            tol = spec["tolerance"]
            if abs(value - target) > tol:
                return False
        return True

    def measure_and_check(
        self, dataset: xr.Dataset, spec: dict[str, Any] | None = None
    ) -> MeasurementResult[Any]:
        """Measure and optionally check against spec.

        Args:
            dataset: Simulation results
            spec: Optional specification to check against

        Returns:
            MeasurementResult with passed field set if spec provided
        """
        result = self.measure(dataset)
        if spec is not None:
            passed = self.check_spec(result, spec)
            result.passed = passed
        return result


__all__ = [
    "MeasurementResult",
    "Measurement",
    "BaseMeasurement",
]
