"""Tests for usability improvements (templates, validators, shortcuts).

Phase: Usability enhancements from openspec/changes/usability-improvements
"""

from __future__ import annotations

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor
from spicelab.core.net import GND, Net
from spicelab.core.units import Unit, parse_value_flexible
from spicelab.templates import rc_lowpass, voltage_divider
from spicelab.validators import validate_circuit


class TestTemplates:
    """Test circuit template helpers."""

    def test_rc_lowpass_default(self) -> None:
        """RC lowpass with default impedance."""
        circuit = rc_lowpass(fc=1000, impedance=10_000)
        assert circuit.name == "RC_Filter"
        assert len(circuit._components) == 2

        # Check components exist
        components_by_type = {type(c).__name__: c for c in circuit._components}
        assert "Resistor" in components_by_type
        assert "Capacitor" in components_by_type

    def test_rc_lowpass_custom_name(self) -> None:
        """RC lowpass with custom name."""
        circuit = rc_lowpass(fc=5000, name="MyFilter")
        assert circuit.name == "MyFilter"

    def test_voltage_divider_50_percent(self) -> None:
        """Voltage divider with 50% ratio."""
        circuit = voltage_divider(ratio=0.5, total_resistance=10_000)
        assert circuit.name == "Voltage_Divider"
        assert len(circuit._components) == 2

        # Both resistors should be 5kΩ for 50% division
        resistors = [c for c in circuit._components if type(c).__name__ == "Resistor"]
        assert len(resistors) == 2

    def test_voltage_divider_invalid_ratio(self) -> None:
        """Voltage divider rejects invalid ratio."""
        with pytest.raises(ValueError, match="ratio must be in"):
            voltage_divider(ratio=1.5)

        with pytest.raises(ValueError, match="ratio must be in"):
            voltage_divider(ratio=0.0)


class TestUnitParser:
    """Test flexible unit parsing."""

    def test_parse_plain_numbers(self) -> None:
        """Parse plain numeric values."""
        assert parse_value_flexible(1000) == 1000.0
        assert parse_value_flexible(1.5) == 1.5
        assert parse_value_flexible("1000") == 1000.0

    def test_parse_si_prefixes(self) -> None:
        """Parse values with SI prefixes."""
        assert parse_value_flexible("10k") == 10_000.0
        assert parse_value_flexible("1.5Meg") == 1_500_000.0
        assert abs(parse_value_flexible("100n") - 100e-9) < 1e-15
        assert abs(parse_value_flexible("22p") - 22e-12) < 1e-18

    def test_parse_with_unit(self) -> None:
        """Parse values with unit suffix."""
        assert parse_value_flexible("10kOhm", Unit.OHM) == 10_000.0
        assert abs(parse_value_flexible("100nF", Unit.FARAD) - 100e-9) < 1e-15

    def test_parse_scientific_notation(self) -> None:
        """Parse scientific notation."""
        assert parse_value_flexible("1e3") == 1000.0
        assert parse_value_flexible("1.5e-6") == 1.5e-6


class TestCircuitValidation:
    """Test circuit validation."""

    def test_validate_simple_circuit(self) -> None:
        """Validate a simple RC circuit."""
        circuit = Circuit("test")
        vin = Net("vin")
        vout = Net("vout")

        R1 = Resistor(ref="R1", resistance=10_000)
        C1 = Capacitor(ref="C1", capacitance=100e-9)
        circuit.add(R1, C1)
        circuit.connect(R1.ports[0], vin)
        circuit.connect(R1.ports[1], vout)
        circuit.connect(C1.ports[0], vout)
        circuit.connect(C1.ports[1], GND)

        result = validate_circuit(circuit)
        assert not result.is_valid

    def test_validate_unusual_resistance(self) -> None:
        """Detect unusually small or large resistances."""
        circuit = Circuit("test")
        vin = Net("vin")

        # Very small resistance (< 1mΩ)
        R1 = Resistor(ref="R1", resistance=0.0001)
        circuit.add(R1)
        circuit.connect(R1.ports[0], vin)
        circuit.connect(R1.ports[1], GND)

        result = validate_circuit(circuit)
        assert len(result.warnings) > 0
        assert any("small resistance" in w.message.lower() for w in result.warnings)

    def test_validate_unusual_capacitance(self) -> None:
        """Detect unusually small or large capacitances."""
        circuit = Circuit("test")
        vin = Net("vin")

        # Very large capacitance (> 1F)
        C1 = Capacitor(ref="C1", capacitance=10.0)
        circuit.add(C1)
        circuit.connect(C1.ports[0], vin)
        circuit.connect(C1.ports[1], GND)

        result = validate_circuit(circuit)
        assert len(result.warnings) > 0
        assert any("large capacitance" in w.message.lower() for w in result.warnings)

    def test_validate_strict_mode(self) -> None:
        """Strict mode converts warnings to errors."""
        circuit = Circuit("test")
        vin = Net("vin")

        # Unusual value
        R1 = Resistor(ref="R1", resistance=1e12)
        circuit.add(R1)
        circuit.connect(R1.ports[0], vin)
        circuit.connect(R1.ports[1], GND)

        result = validate_circuit(circuit, strict=True)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert len(result.warnings) == 0


class TestShortcuts:
    """Test simulation shortcuts."""

    def test_import_shortcuts(self) -> None:
        """Verify shortcuts can be imported."""
        from spicelab.shortcuts import quick_ac, quick_tran

        assert callable(quick_ac)
        assert callable(quick_tran)

    # Note: Actual simulation tests would require ngspice binary
    # These are integration tests that should be run separately
