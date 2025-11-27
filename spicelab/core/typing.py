"""Type definitions and validation decorators for SpiceLab.

This module provides:
- Common type aliases for circuit simulation
- Type validation decorators for runtime checking
- Protocol definitions for structural subtyping
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .circuit import Circuit
    from .components import Component
    from .net import Net, Port


# Type variables
T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

# Basic numeric types
NumericValue: TypeAlias = Union[int, float]
PositiveFloat: TypeAlias = float  # Should be > 0
NonNegativeFloat: TypeAlias = float  # Should be >= 0

# Component value types
Resistance: TypeAlias = float  # Ohms
Capacitance: TypeAlias = float  # Farads
Inductance: TypeAlias = float  # Henries
Voltage: TypeAlias = float  # Volts
Current: TypeAlias = float  # Amperes
Frequency: TypeAlias = float  # Hertz
Time: TypeAlias = float  # Seconds

# Engineering notation string (e.g., "1k", "10u", "100n")
EngineeringString: TypeAlias = str

# Component reference designator
RefDesignator: TypeAlias = str

# Net name
NetName: TypeAlias = str

# Analysis mode literals
AnalysisMode: TypeAlias = Literal["op", "dc", "ac", "tran", "noise"]

# Engine literals
EngineName: TypeAlias = Literal["ngspice", "ltspice", "xyce"]

# Sweep types
SweepType: TypeAlias = Literal["lin", "dec", "oct"]


# Protocols for structural typing
@runtime_checkable
class HasPorts(Protocol):
    """Protocol for objects with ports."""

    @property
    def ports(self) -> tuple[Port, ...]:
        """Return sequence of ports."""
        ...


@runtime_checkable
class HasRef(Protocol):
    """Protocol for objects with reference designator."""

    @property
    def ref(self) -> str:
        """Return reference designator."""
        ...


@runtime_checkable
class Simulatable(Protocol):
    """Protocol for objects that can be simulated."""

    def build_netlist(self) -> str:
        """Build SPICE netlist."""
        ...

    def validate(self) -> Any:
        """Validate the circuit."""
        ...


@runtime_checkable
class Connectable(Protocol):
    """Protocol for objects that can be connected."""

    def connect(self, port: Port, net: Net) -> None:
        """Connect a port to a net."""
        ...


# Validation decorators
class ValidationError(TypeError):
    """Error raised when type validation fails."""

    def __init__(self, param_name: str, expected: str, got: Any) -> None:
        self.param_name = param_name
        self.expected = expected
        self.got = got
        super().__init__(
            f"Parameter '{param_name}' expected {expected}, got {type(got).__name__}: {got!r}"
        )


def validate_types(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to validate function argument types at runtime.

    Uses type hints from the function signature to validate arguments.

    Example:
        >>> from spicelab.core.typing import validate_types
        >>>
        >>> @validate_types
        ... def add_resistor(resistance: float, ref: str) -> None:
        ...     print(f"R{ref} = {resistance}")
        ...
        >>> add_resistor(1000.0, "1")  # OK
        >>> add_resistor("1000", "1")  # Raises ValidationError
    """
    sig = inspect.signature(func)
    hints = func.__annotations__

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # Bind arguments to parameters
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Validate each argument
        for param_name, value in bound.arguments.items():
            if param_name in hints and param_name != "return":
                expected_type = hints[param_name]
                if not _check_type(value, expected_type):
                    raise ValidationError(param_name, str(expected_type), value)

        return func(*args, **kwargs)

    return wrapper


def _check_type(value: Any, expected: Any) -> bool:
    """Check if value matches expected type hint."""
    # Handle None
    if value is None:
        # Check if None is allowed (Optional, Union with None, or types.UnionType)
        origin = getattr(expected, "__origin__", None)
        if origin is Union:
            args = getattr(expected, "__args__", ())
            return type(None) in args
        # Python 3.10+ uses types.UnionType for X | Y syntax
        if hasattr(expected, "__args__"):
            args = getattr(expected, "__args__", ())
            return type(None) in args
        return False

    # Get origin for generic types
    origin = getattr(expected, "__origin__", None)

    if origin is None:
        # Simple type check
        if expected is Any:
            return True
        if isinstance(expected, type):
            # Allow int where float is expected (Python convention)
            if expected is float and isinstance(value, int) and not isinstance(value, bool):
                return True
            return isinstance(value, expected)
        return True  # Unknown type hint, allow

    # Handle Union
    if origin is Union:
        args = getattr(expected, "__args__", ())
        return any(_check_type(value, arg) for arg in args)

    # Handle Literal
    if origin is Literal:
        args = getattr(expected, "__args__", ())
        return value in args

    # Handle generic collections
    if origin in (list, set, frozenset):
        if not isinstance(value, origin):
            return False
        args = getattr(expected, "__args__", ())
        if args:
            return all(_check_type(item, args[0]) for item in value)
        return True

    if origin is dict:
        if not isinstance(value, dict):
            return False
        dict_args: tuple[Any, ...] = getattr(expected, "__args__", ())
        if len(dict_args) >= 2:
            key_type = dict_args[0]
            val_type = dict_args[1]
            return all(
                _check_type(k, key_type) and _check_type(v, val_type)
                for k, v in value.items()
            )
        return True

    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        args = getattr(expected, "__args__", ())
        if args:
            if len(args) == 2 and args[1] is ...:
                return all(_check_type(item, args[0]) for item in value)
            if len(args) == len(value):
                return all(_check_type(v, t) for v, t in zip(value, args))
        return True

    # Handle Sequence, Iterable, etc.
    if origin in (Sequence,):
        try:
            iter(value)
            return True
        except TypeError:
            return False

    # For other generic types, just check the origin
    return isinstance(value, origin)


def positive(value: float) -> float:
    """Validate that a value is positive.

    Args:
        value: Value to validate

    Returns:
        The validated value

    Raises:
        ValueError: If value is not positive

    Example:
        >>> from spicelab.core.typing import positive
        >>> positive(1000.0)
        1000.0
        >>> positive(-1.0)  # Raises ValueError
    """
    if value <= 0:
        raise ValueError(f"Value must be positive, got {value}")
    return value


def non_negative(value: float) -> float:
    """Validate that a value is non-negative.

    Args:
        value: Value to validate

    Returns:
        The validated value

    Raises:
        ValueError: If value is negative

    Example:
        >>> from spicelab.core.typing import non_negative
        >>> non_negative(0.0)
        0.0
        >>> non_negative(-1.0)  # Raises ValueError
    """
    if value < 0:
        raise ValueError(f"Value must be non-negative, got {value}")
    return value


def in_range(
    value: float,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Validate that a value is within a range.

    Args:
        value: Value to validate
        min_val: Minimum value (inclusive), or None for no minimum
        max_val: Maximum value (inclusive), or None for no maximum

    Returns:
        The validated value

    Raises:
        ValueError: If value is out of range

    Example:
        >>> from spicelab.core.typing import in_range
        >>> in_range(5.0, 0, 10)
        5.0
        >>> in_range(15.0, 0, 10)  # Raises ValueError
    """
    if min_val is not None and value < min_val:
        raise ValueError(f"Value {value} is below minimum {min_val}")
    if max_val is not None and value > max_val:
        raise ValueError(f"Value {value} is above maximum {max_val}")
    return value


def one_of(value: T, options: Sequence[T]) -> T:
    """Validate that a value is one of the allowed options.

    Args:
        value: Value to validate
        options: Allowed values

    Returns:
        The validated value

    Raises:
        ValueError: If value is not in options

    Example:
        >>> from spicelab.core.typing import one_of
        >>> one_of("ac", ["op", "dc", "ac", "tran"])
        'ac'
        >>> one_of("invalid", ["op", "dc", "ac", "tran"])  # Raises ValueError
    """
    if value not in options:
        raise ValueError(f"Value must be one of {options}, got {value!r}")
    return value


# Type guards
def is_numeric(value: Any) -> bool:
    """Check if value is numeric (int or float)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_positive(value: Any) -> bool:
    """Check if value is a positive number."""
    return is_numeric(value) and value > 0


def is_non_negative(value: Any) -> bool:
    """Check if value is a non-negative number."""
    return is_numeric(value) and value >= 0


def is_valid_ref(value: Any) -> bool:
    """Check if value is a valid reference designator."""
    if not isinstance(value, str):
        return False
    return len(value) > 0 and value[0].isalnum()


__all__ = [
    # Type aliases
    "NumericValue",
    "PositiveFloat",
    "NonNegativeFloat",
    "Resistance",
    "Capacitance",
    "Inductance",
    "Voltage",
    "Current",
    "Frequency",
    "Time",
    "EngineeringString",
    "RefDesignator",
    "NetName",
    "AnalysisMode",
    "EngineName",
    "SweepType",
    # Protocols
    "HasPorts",
    "HasRef",
    "Simulatable",
    "Connectable",
    # Decorators and validators
    "ValidationError",
    "validate_types",
    "positive",
    "non_negative",
    "in_range",
    "one_of",
    # Type guards
    "is_numeric",
    "is_positive",
    "is_non_negative",
    "is_valid_ref",
]
