"""Tests for spicelab.core.typing module."""

import pytest
from spicelab.core.typing import (
    # Type aliases
    NumericValue,
    PositiveFloat,
    Resistance,
    Capacitance,
    AnalysisMode,
    # Protocols
    HasPorts,
    HasRef,
    Simulatable,
    # Validators
    ValidationError,
    validate_types,
    positive,
    non_negative,
    in_range,
    one_of,
    # Type guards
    is_numeric,
    is_positive,
    is_non_negative,
    is_valid_ref,
)
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Capacitor


class TestTypeAliases:
    """Tests for type aliases."""

    def test_numeric_value_accepts_int_and_float(self):
        """Test NumericValue accepts int and float."""
        # These are just type aliases, so test they exist
        value1: NumericValue = 1
        value2: NumericValue = 1.0
        assert value1 == 1
        assert value2 == 1.0

    def test_resistance_is_float(self):
        """Test Resistance type alias."""
        r: Resistance = 1000.0
        assert r == 1000.0

    def test_capacitance_is_float(self):
        """Test Capacitance type alias."""
        c: Capacitance = 1e-6
        assert c == 1e-6


class TestProtocols:
    """Tests for Protocol definitions."""

    def test_has_ports_protocol(self):
        """Test HasPorts protocol matching."""
        r = Resistor(ref="1", resistance=1000)
        # Resistor should match HasPorts protocol
        assert isinstance(r, HasPorts)
        assert hasattr(r, "ports")

    def test_has_ref_protocol(self):
        """Test HasRef protocol matching."""
        r = Resistor(ref="1", resistance=1000)
        assert isinstance(r, HasRef)
        assert r.ref == "1"

    def test_simulatable_protocol(self):
        """Test Simulatable protocol matching."""
        circuit = Circuit("test")
        assert isinstance(circuit, Simulatable)
        assert hasattr(circuit, "build_netlist")
        assert hasattr(circuit, "validate")


class TestValidateTypesDecorator:
    """Tests for validate_types decorator."""

    def test_valid_args_pass(self):
        """Test that valid arguments pass validation."""

        @validate_types
        def add_numbers(a: int, b: int) -> int:
            return a + b

        result = add_numbers(1, 2)
        assert result == 3

    def test_invalid_args_raise(self):
        """Test that invalid arguments raise ValidationError."""

        @validate_types
        def add_numbers(a: int, b: int) -> int:
            return a + b

        with pytest.raises(ValidationError) as exc_info:
            add_numbers("1", 2)  # type: ignore

        assert "a" in str(exc_info.value)
        assert "int" in str(exc_info.value)

    def test_float_validation(self):
        """Test float type validation."""

        @validate_types
        def set_resistance(value: float) -> float:
            return value

        # Float should pass
        assert set_resistance(1000.0) == 1000.0

        # Int should also pass (subtype of float in Python)
        assert set_resistance(1000) == 1000

        # String should fail
        with pytest.raises(ValidationError):
            set_resistance("1000")  # type: ignore

    def test_optional_validation(self):
        """Test Optional type validation."""

        @validate_types
        def maybe_value(value: int | None = None) -> int | None:
            return value

        assert maybe_value(5) == 5
        assert maybe_value(None) is None
        assert maybe_value() is None

    def test_list_validation(self):
        """Test list type validation."""

        @validate_types
        def sum_values(values: list[int]) -> int:
            return sum(values)

        assert sum_values([1, 2, 3]) == 6

        with pytest.raises(ValidationError):
            sum_values("not a list")  # type: ignore

    def test_kwargs_validation(self):
        """Test keyword arguments validation."""

        @validate_types
        def configure(name: str, value: float) -> dict:
            return {"name": name, "value": value}

        result = configure(name="test", value=1.0)
        assert result == {"name": "test", "value": 1.0}

        with pytest.raises(ValidationError):
            configure(name=123, value=1.0)  # type: ignore


class TestPositiveValidator:
    """Tests for positive() validator."""

    def test_positive_value_passes(self):
        """Test that positive values pass."""
        assert positive(1.0) == 1.0
        assert positive(0.001) == 0.001
        assert positive(1e9) == 1e9

    def test_zero_fails(self):
        """Test that zero fails."""
        with pytest.raises(ValueError, match="positive"):
            positive(0.0)

    def test_negative_fails(self):
        """Test that negative values fail."""
        with pytest.raises(ValueError, match="positive"):
            positive(-1.0)


class TestNonNegativeValidator:
    """Tests for non_negative() validator."""

    def test_positive_value_passes(self):
        """Test that positive values pass."""
        assert non_negative(1.0) == 1.0

    def test_zero_passes(self):
        """Test that zero passes."""
        assert non_negative(0.0) == 0.0

    def test_negative_fails(self):
        """Test that negative values fail."""
        with pytest.raises(ValueError, match="non-negative"):
            non_negative(-1.0)


class TestInRangeValidator:
    """Tests for in_range() validator."""

    def test_value_in_range_passes(self):
        """Test that values in range pass."""
        assert in_range(5.0, 0, 10) == 5.0
        assert in_range(0.0, 0, 10) == 0.0
        assert in_range(10.0, 0, 10) == 10.0

    def test_value_below_min_fails(self):
        """Test that values below minimum fail."""
        with pytest.raises(ValueError, match="below minimum"):
            in_range(-1.0, 0, 10)

    def test_value_above_max_fails(self):
        """Test that values above maximum fail."""
        with pytest.raises(ValueError, match="above maximum"):
            in_range(11.0, 0, 10)

    def test_no_min_bound(self):
        """Test with no minimum bound."""
        assert in_range(-1000.0, None, 10) == -1000.0

    def test_no_max_bound(self):
        """Test with no maximum bound."""
        assert in_range(1000.0, 0, None) == 1000.0


class TestOneOfValidator:
    """Tests for one_of() validator."""

    def test_valid_option_passes(self):
        """Test that valid options pass."""
        assert one_of("ac", ["op", "dc", "ac", "tran"]) == "ac"
        assert one_of(1, [1, 2, 3]) == 1

    def test_invalid_option_fails(self):
        """Test that invalid options fail."""
        with pytest.raises(ValueError, match="must be one of"):
            one_of("invalid", ["op", "dc", "ac", "tran"])


class TestTypeGuards:
    """Tests for type guard functions."""

    def test_is_numeric(self):
        """Test is_numeric type guard."""
        assert is_numeric(1) is True
        assert is_numeric(1.0) is True
        assert is_numeric("1") is False
        assert is_numeric(True) is False  # bool is subclass of int, but we exclude it
        assert is_numeric(None) is False

    def test_is_positive(self):
        """Test is_positive type guard."""
        assert is_positive(1) is True
        assert is_positive(0.1) is True
        assert is_positive(0) is False
        assert is_positive(-1) is False
        assert is_positive("1") is False

    def test_is_non_negative(self):
        """Test is_non_negative type guard."""
        assert is_non_negative(1) is True
        assert is_non_negative(0) is True
        assert is_non_negative(-1) is False
        assert is_non_negative("0") is False

    def test_is_valid_ref(self):
        """Test is_valid_ref type guard."""
        assert is_valid_ref("1") is True
        assert is_valid_ref("R1") is True
        assert is_valid_ref("") is False
        assert is_valid_ref(123) is False


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_error_message(self):
        """Test ValidationError message format."""
        error = ValidationError("resistance", "float", "1000")
        assert "resistance" in str(error)
        assert "float" in str(error)
        assert "str" in str(error)

    def test_error_attributes(self):
        """Test ValidationError attributes."""
        error = ValidationError("value", "int", 1.5)
        assert error.param_name == "value"
        assert error.expected == "int"
        assert error.got == 1.5
