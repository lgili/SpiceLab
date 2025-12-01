"""Measurement registry for discovering and instantiating measurements.

The registry provides:
- @measurement decorator for registering measurement classes
- MeasurementRegistry for looking up measurements by name
- measure() convenience function for one-line measurements
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar

from .base import BaseMeasurement, MeasurementResult

if TYPE_CHECKING:
    import xarray as xr

T = TypeVar("T", bound=type[BaseMeasurement])


class MeasurementRegistry:
    """Central registry of available measurements.

    Use the @measurement decorator to register measurements, then look them
    up by name using get() or list_all().

    Example:
        >>> @measurement("my_gain")
        ... class MyGainMeasurement(BaseMeasurement):
        ...     name = "my_gain"
        ...     description = "Custom gain measurement"
        ...     required_analyses = ["ac"]
        ...
        ...     def measure(self, dataset):
        ...         return MeasurementResult(value=42.0, unit="dB")
        >>>
        >>> # Later, look up by name
        >>> cls = MeasurementRegistry.get("my_gain")
        >>> m = cls()
        >>> result = m.measure(dataset)
    """

    _measurements: dict[str, type[BaseMeasurement]] = {}

    @classmethod
    def register(cls, name: str | None = None) -> Callable[[T], T]:
        """Decorator to register a measurement class.

        Args:
            name: Optional name override. If not provided, uses class name
                  in lowercase (e.g., GainMeasurement -> gainmeasurement)

        Returns:
            Decorator function
        """

        def decorator(measurement_cls: T) -> T:
            reg_name = name or measurement_cls.__name__.lower()
            cls._measurements[reg_name] = measurement_cls
            # Also set the name attribute on the class if not already set
            if not hasattr(measurement_cls, "name") or measurement_cls.name == "base":
                measurement_cls.name = reg_name
            return measurement_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseMeasurement]:
        """Get measurement class by name.

        Args:
            name: Registered measurement name

        Returns:
            The measurement class

        Raises:
            ValueError: If measurement not found
        """
        if name not in cls._measurements:
            available = ", ".join(sorted(cls._measurements.keys()))
            raise ValueError(
                f"Measurement '{name}' not found. Available: [{available}]"
            )
        return cls._measurements[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered measurement names.

        Returns:
            Sorted list of measurement names
        """
        return sorted(cls._measurements.keys())

    @classmethod
    def list_by_analysis(cls, analysis_type: str) -> list[str]:
        """List measurements that support a specific analysis type.

        Args:
            analysis_type: Analysis type (e.g., "ac", "tran", "op")

        Returns:
            List of measurement names that require this analysis
        """
        result = []
        for name, measurement_cls in cls._measurements.items():
            required = getattr(measurement_cls, "required_analyses", [])
            if analysis_type in required:
                result.append(name)
        return sorted(result)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered measurements. Mainly for testing."""
        cls._measurements.clear()

    @classmethod
    def info(cls, name: str) -> dict[str, Any]:
        """Get information about a measurement.

        Args:
            name: Measurement name

        Returns:
            Dict with name, description, required_analyses
        """
        measurement_cls = cls.get(name)
        return {
            "name": getattr(measurement_cls, "name", name),
            "description": getattr(measurement_cls, "description", "No description"),
            "required_analyses": getattr(measurement_cls, "required_analyses", []),
            "class": measurement_cls.__name__,
        }


def measurement(name: str | None = None) -> Callable[[T], T]:
    """Decorator to register a measurement class.

    This is a convenience wrapper around MeasurementRegistry.register().

    Args:
        name: Optional name for the measurement. If not provided,
              uses the class name in lowercase.

    Example:
        >>> @measurement("bandwidth")
        ... class BandwidthMeasurement(BaseMeasurement):
        ...     description = "Measure -3dB bandwidth"
        ...     required_analyses = ["ac"]
        ...
        ...     def __init__(self, output_node: str, n_db: float = 3.0):
        ...         self.output_node = output_node
        ...         self.n_db = n_db
        ...
        ...     def measure(self, dataset):
        ...         # ... implementation ...
        ...         return MeasurementResult(value=10000.0, unit="Hz")
    """
    return MeasurementRegistry.register(name)


def measure(
    dataset: xr.Dataset,
    name: str,
    spec: dict[str, Any] | None = None,
    **kwargs: Any,
) -> MeasurementResult[Any]:
    """Convenience function to perform a measurement by name.

    Args:
        dataset: xarray Dataset with simulation results
        name: Registered measurement name
        spec: Optional specification to check against
        **kwargs: Arguments passed to the measurement constructor

    Returns:
        MeasurementResult from the measurement

    Example:
        >>> result = measure(dataset, "bandwidth", output_node="vout")
        >>> print(f"Bandwidth: {result.value} {result.unit}")

        >>> # With spec checking
        >>> result = measure(
        ...     dataset, "gain",
        ...     spec={"min": 40, "max": 80},
        ...     output_node="vout", frequency=1000
        ... )
        >>> if result.passed:
        ...     print("Gain within spec!")
    """
    measurement_cls = MeasurementRegistry.get(name)
    measurement_instance = measurement_cls(**kwargs)
    return measurement_instance.measure_and_check(dataset, spec)


__all__ = [
    "MeasurementRegistry",
    "measurement",
    "measure",
]
