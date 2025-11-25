"""Tests for circuit templates (M4 DX improvement).

Tests the circuit template functions for filters and amplifiers.
"""

import math

import pytest
from spicelab.templates import (
    FilterResult,
    inverting_amplifier,
    non_inverting_amplifier,
    rc_highpass,
    rc_lowpass,
    rlc_bandpass,
    sallen_key_lowpass,
    voltage_divider,
)


class TestRCLowpass:
    """Tests for RC lowpass filter template."""

    def test_creates_circuit(self):
        """Should create a valid circuit."""
        circuit = rc_lowpass(fc=1000)
        assert circuit.name == "RC_Filter"
        assert len(circuit._components) == 2

    def test_cutoff_frequency_calculation(self):
        """RC values should give correct cutoff frequency."""
        fc = 1000  # 1kHz
        impedance = 10_000  # 10kΩ
        circuit = rc_lowpass(fc=fc, impedance=impedance)

        # Find R and C components
        r_comp = next(c for c in circuit._components if "Resistor" in type(c).__name__)
        c_comp = next(c for c in circuit._components if "Capacitor" in type(c).__name__)

        # Use typed fields for numeric values
        R = r_comp.resistance
        C = c_comp.capacitance

        # fc = 1/(2π RC)
        calculated_fc = 1 / (2 * math.pi * R * C)
        assert abs(calculated_fc - fc) < 1  # Within 1Hz

    def test_generates_valid_netlist(self):
        """Should generate valid SPICE netlist."""
        circuit = rc_lowpass(fc=1000)
        netlist = circuit.build_netlist()

        assert "RC_Filter" in netlist
        assert "R1" in netlist
        assert "C1" in netlist
        assert ".end" in netlist


class TestRCHighpass:
    """Tests for RC highpass filter template."""

    def test_creates_circuit(self):
        """Should create a valid circuit."""
        circuit = rc_highpass(fc=100)
        assert circuit.name == "RC_Highpass"
        assert len(circuit._components) == 2

    def test_cutoff_frequency_calculation(self):
        """RC values should give correct cutoff frequency."""
        fc = 100  # 100Hz
        circuit = rc_highpass(fc=fc, impedance=10_000)

        r_comp = next(c for c in circuit._components if "Resistor" in type(c).__name__)
        c_comp = next(c for c in circuit._components if "Capacitor" in type(c).__name__)

        R = r_comp.resistance
        C = c_comp.capacitance

        calculated_fc = 1 / (2 * math.pi * R * C)
        assert abs(calculated_fc - fc) < 1

    def test_topology_is_highpass(self):
        """Capacitor should be in series (first in chain)."""
        circuit = rc_highpass(fc=100)
        netlist = circuit.build_netlist()

        # In highpass, C connects vin to vout, R connects vout to GND
        assert "C1 vin vout" in netlist
        assert "R1 vout 0" in netlist


class TestRLCBandpass:
    """Tests for RLC bandpass filter template."""

    def test_returns_filter_result(self):
        """Should return FilterResult with circuit and metadata."""
        result = rlc_bandpass(fc=1000, bandwidth=100)

        assert isinstance(result, FilterResult)
        assert result.cutoff_frequency == 1000
        assert result.q_factor == 10.0  # Q = fc / bandwidth

    def test_q_factor_calculation(self):
        """Q factor should equal fc/bandwidth."""
        fc = 5000
        bandwidth = 500
        result = rlc_bandpass(fc=fc, bandwidth=bandwidth)

        expected_q = fc / bandwidth
        assert result.q_factor == expected_q

    def test_components_dict(self):
        """Should provide access to components."""
        result = rlc_bandpass(fc=1000, bandwidth=100)

        assert "R1" in result.components
        assert "L1" in result.components
        assert "C1" in result.components

    def test_circuit_has_three_components(self):
        """RLC filter should have R, L, and C."""
        result = rlc_bandpass(fc=1000, bandwidth=100)
        assert len(result.circuit._components) == 3


class TestSallenKeyLowpass:
    """Tests for Sallen-Key lowpass filter template."""

    def test_returns_filter_result(self):
        """Should return FilterResult."""
        result = sallen_key_lowpass(fc=1000)
        assert isinstance(result, FilterResult)

    def test_default_butterworth_q(self):
        """Default Q should be 0.707 (Butterworth)."""
        result = sallen_key_lowpass(fc=1000)
        assert abs(result.q_factor - 0.707) < 0.001

    def test_custom_q_factor(self):
        """Should accept custom Q factor."""
        result = sallen_key_lowpass(fc=1000, q=1.0)
        assert result.q_factor == 1.0

    def test_has_four_components(self):
        """Sallen-Key should have 2 resistors and 2 capacitors."""
        result = sallen_key_lowpass(fc=1000)
        assert len(result.circuit._components) == 4

    def test_components_accessible(self):
        """Should provide access to R1, R2, C1, C2."""
        result = sallen_key_lowpass(fc=1000)
        assert "R1" in result.components
        assert "R2" in result.components
        assert "C1" in result.components
        assert "C2" in result.components


class TestVoltageDivider:
    """Tests for voltage divider template."""

    def test_creates_circuit(self):
        """Should create valid circuit."""
        circuit = voltage_divider(ratio=0.5)
        assert len(circuit._components) == 2

    def test_ratio_50_percent(self):
        """50% ratio should create equal resistors."""
        circuit = voltage_divider(ratio=0.5, total_resistance=10_000)

        resistors = [c for c in circuit._components if "Resistor" in type(c).__name__]
        values = [r.resistance for r in resistors]

        # Both should be 5kΩ for 50% division
        assert abs(values[0] - 5000) < 1
        assert abs(values[1] - 5000) < 1

    def test_ratio_25_percent(self):
        """25% ratio: R2/(R1+R2) = 0.25, so R2 = 0.25 * total."""
        circuit = voltage_divider(ratio=0.25, total_resistance=10_000)

        resistors = sorted(
            [c for c in circuit._components if "Resistor" in type(c).__name__],
            key=lambda r: r.resistance,
        )

        # R2 = 2.5kΩ (smaller), R1 = 7.5kΩ (larger)
        assert abs(resistors[0].resistance - 2500) < 1
        assert abs(resistors[1].resistance - 7500) < 1

    def test_invalid_ratio_raises(self):
        """Ratio outside (0,1) should raise ValueError."""
        with pytest.raises(ValueError):
            voltage_divider(ratio=0)

        with pytest.raises(ValueError):
            voltage_divider(ratio=1)

        with pytest.raises(ValueError):
            voltage_divider(ratio=1.5)

        with pytest.raises(ValueError):
            voltage_divider(ratio=-0.5)


class TestInvertingAmplifier:
    """Tests for inverting amplifier template."""

    def test_creates_circuit(self):
        """Should create valid circuit."""
        circuit = inverting_amplifier(gain=10)
        assert len(circuit._components) == 2

    def test_gain_10_resistor_values(self):
        """Gain=10 with Rin=10kΩ should give Rf=100kΩ."""
        circuit = inverting_amplifier(gain=10, input_impedance=10_000)

        resistors = {c.ref: c for c in circuit._components}
        assert abs(resistors["in"].resistance - 10_000) < 1
        assert abs(resistors["f"].resistance - 100_000) < 1

    def test_invalid_gain_raises(self):
        """Non-positive gain should raise ValueError."""
        with pytest.raises(ValueError):
            inverting_amplifier(gain=0)

        with pytest.raises(ValueError):
            inverting_amplifier(gain=-5)

    def test_netlist_has_vin_vminus_vout(self):
        """Netlist should have vin, vminus, vout nodes."""
        circuit = inverting_amplifier(gain=10)
        netlist = circuit.build_netlist()

        assert "vin" in netlist
        assert "vminus" in netlist
        assert "vout" in netlist


class TestNonInvertingAmplifier:
    """Tests for non-inverting amplifier template."""

    def test_creates_circuit(self):
        """Should create valid circuit."""
        circuit = non_inverting_amplifier(gain=11)
        assert len(circuit._components) == 2

    def test_gain_11_resistor_values(self):
        """Gain=11 with Rf=10kΩ: R1 = Rf/(gain-1) = 10k/10 = 1kΩ."""
        circuit = non_inverting_amplifier(gain=11, feedback_resistance=10_000)

        resistors = {c.ref: c for c in circuit._components}
        assert abs(resistors["1"].resistance - 1000) < 1
        assert abs(resistors["f"].resistance - 10_000) < 1

    def test_invalid_gain_raises(self):
        """Gain < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            non_inverting_amplifier(gain=0.5)

    def test_gain_2_creates_equal_resistors(self):
        """Gain=2: R1 = Rf/(2-1) = Rf."""
        circuit = non_inverting_amplifier(gain=2, feedback_resistance=10_000)

        resistors = {c.ref: c for c in circuit._components}
        assert abs(resistors["1"].resistance - 10_000) < 1
        assert abs(resistors["f"].resistance - 10_000) < 1


class TestFilterResultDataclass:
    """Tests for FilterResult dataclass."""

    def test_repr(self):
        """Should have readable repr."""
        result = rlc_bandpass(fc=1000, bandwidth=100)
        repr_str = repr(result)

        assert "FilterResult" in repr_str
        assert "1000" in repr_str

    def test_attributes(self):
        """Should have expected attributes."""
        result = sallen_key_lowpass(fc=1000)

        assert hasattr(result, "circuit")
        assert hasattr(result, "components")
        assert hasattr(result, "cutoff_frequency")
        assert hasattr(result, "q_factor")
