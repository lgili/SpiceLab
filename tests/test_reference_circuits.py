"""Tests for Reference Circuits Library (Sprint 3 - M15)."""

import pytest
from spicelab.core.circuit import Circuit
from spicelab.templates import (
    ReferenceCircuit,
    ReferenceLibrary,
    ac_test_circuit,
    dc_test_circuit,
    decoupling_network,
    led_driver,
    library,
    ohms_law_demo,
    rc_lowpass_filter,
    rc_time_constant_demo,
    rl_time_constant_demo,
    rlc_resonance_demo,
    voltage_divider_demo,
)

# ============================================================================
# ReferenceCircuit Tests
# ============================================================================


class TestReferenceCircuit:
    """Tests for ReferenceCircuit dataclass."""

    def test_basic_structure(self):
        """Test ReferenceCircuit has expected attributes."""
        ref = ohms_law_demo()

        assert isinstance(ref.circuit, Circuit)
        assert isinstance(ref.name, str)
        assert isinstance(ref.category, str)
        assert isinstance(ref.description, str)
        assert isinstance(ref.learning_objectives, list)
        assert isinstance(ref.expected_results, dict)
        assert isinstance(ref.components, dict)
        assert isinstance(ref.nodes, dict)
        assert isinstance(ref.suggested_analyses, list)
        assert isinstance(ref.parameters, dict)

    def test_repr(self):
        """Test string representation."""
        ref = ohms_law_demo()
        repr_str = repr(ref)

        assert "ReferenceCircuit" in repr_str
        assert "Ohm" in repr_str

    def test_summary(self):
        """Test summary method."""
        ref = ohms_law_demo()
        summary = ref.summary()

        assert "Ohm" in summary
        assert "educational" in summary
        assert "Learning Objectives" in summary
        assert "Suggested Analyses" in summary


# ============================================================================
# Educational Circuits Tests
# ============================================================================


class TestOhmsLawDemo:
    """Tests for Ohm's Law demonstration circuit."""

    def test_default_params(self):
        """Test with default parameters."""
        ref = ohms_law_demo()

        assert ref.category == "educational"
        assert ref.parameters["voltage"] == 5.0
        assert ref.parameters["resistance"] == 1000.0
        assert "5.000 mA" in ref.expected_results["current_through_R1"]

    def test_custom_params(self):
        """Test with custom parameters."""
        ref = ohms_law_demo(voltage=12.0, resistance=2000.0)

        assert ref.parameters["voltage"] == 12.0
        assert ref.parameters["resistance"] == 2000.0
        # I = 12V / 2000Ω = 6mA
        assert "6.000 mA" in ref.expected_results["current_through_R1"]

    def test_circuit_valid(self):
        """Test generated circuit is valid."""
        ref = ohms_law_demo()

        assert ref.circuit.name == "ohms_law_demo"
        assert len(ref.components) == 2
        assert "V1" in ref.components
        assert "R1" in ref.components


class TestVoltageDividerDemo:
    """Tests for voltage divider demonstration."""

    def test_equal_resistors(self):
        """Test with equal resistors (50% division)."""
        ref = voltage_divider_demo(vin=10.0, r1=10000.0, r2=10000.0)

        assert "5.000 V" in ref.expected_results["output_voltage"]
        assert "0.500" in ref.expected_results["division_ratio"]

    def test_unequal_resistors(self):
        """Test with unequal resistors."""
        ref = voltage_divider_demo(vin=10.0, r1=30000.0, r2=10000.0)

        # Vout = 10 * 10k / 40k = 2.5V
        assert "2.500 V" in ref.expected_results["output_voltage"]


class TestRCTimeConstant:
    """Tests for RC time constant demonstration."""

    def test_time_constant(self):
        """Test time constant calculation."""
        ref = rc_time_constant_demo(tau_ms=1.0)

        assert ref.parameters["tau_ms"] == 1.0
        assert "1.0 ms" in ref.expected_results["time_constant"]
        assert "C1" in ref.components

    def test_charging_curve(self):
        """Test expected charging curve values."""
        ref = rc_time_constant_demo(tau_ms=1.0, vin=5.0)

        # 63.2% at 1 tau
        assert "3.160 V" in ref.expected_results["voltage_at_1_tau"]


class TestRLTimeConstant:
    """Tests for RL time constant demonstration."""

    def test_time_constant(self):
        """Test time constant calculation."""
        ref = rl_time_constant_demo(tau_us=100.0)

        assert ref.parameters["tau_us"] == 100.0
        assert "L1" in ref.components


class TestRLCResonance:
    """Tests for RLC resonance demonstration."""

    def test_resonance_params(self):
        """Test resonance parameters."""
        ref = rlc_resonance_demo(f0_hz=1000.0, q_factor=10.0)

        assert ref.parameters["f0"] == 1000.0
        assert ref.parameters["Q"] == 10.0
        assert "1000.0 Hz" in ref.expected_results["resonant_frequency"]
        assert "100.0 Hz" in ref.expected_results["bandwidth"]

    def test_high_q(self):
        """Test high Q factor."""
        ref = rlc_resonance_demo(f0_hz=10000.0, q_factor=50.0)

        # BW = f0/Q = 10000/50 = 200 Hz
        assert "200.0 Hz" in ref.expected_results["bandwidth"]


# ============================================================================
# Application Note Circuits Tests
# ============================================================================


class TestLEDDriver:
    """Tests for LED driver circuit."""

    def test_default_params(self):
        """Test with default parameters."""
        ref = led_driver()

        assert ref.category == "appnote"
        assert "20.0 mA" in ref.expected_results["led_current"]
        assert "D1" in ref.components

    def test_custom_current(self):
        """Test with custom LED current."""
        ref = led_driver(vcc=12.0, led_vf=2.0, led_current_ma=10.0)

        # R = (12 - 2) / 0.01 = 1000 Ω
        assert "1000" in ref.expected_results["resistor_value"]


class TestRCLowpassFilter:
    """Tests for RC lowpass filter."""

    def test_cutoff_frequency(self):
        """Test cutoff frequency."""
        ref = rc_lowpass_filter(fc_hz=1000.0)

        assert "1000.0 Hz" in ref.expected_results["cutoff_frequency"]
        assert "-3 dB" in ref.expected_results["gain_at_fc"]

    def test_custom_impedance(self):
        """Test custom impedance."""
        ref = rc_lowpass_filter(fc_hz=1000.0, impedance=50000.0)

        assert ref.parameters["R"] == 50000.0


class TestDecouplingNetwork:
    """Tests for decoupling network."""

    def test_multi_cap_structure(self):
        """Test multi-capacitor structure."""
        ref = decoupling_network()

        assert ref.category == "appnote"
        assert "C_bulk" in ref.components
        assert "C_100n" in ref.components
        assert "C_10n" in ref.components


# ============================================================================
# Test Circuits Tests
# ============================================================================


class TestDCTestCircuit:
    """Tests for DC test circuit."""

    def test_known_solution(self):
        """Test known DC solution."""
        ref = dc_test_circuit()

        assert ref.category == "test"
        assert "10.000 V" in ref.expected_results["V(v1)"]
        # V2 = 10 * 1200 / 2200 = 5.4545V
        assert "5.455 V" in ref.expected_results["V(v2)"]


class TestACTestCircuit:
    """Tests for AC test circuit."""

    def test_known_response(self):
        """Test known AC response."""
        ref = ac_test_circuit(fc_hz=1000.0)

        assert ref.category == "test"
        assert "-3.01 dB" in ref.expected_results["gain_at_fc"]
        assert "-45°" in ref.expected_results["gain_at_fc"]


# ============================================================================
# ReferenceLibrary Tests
# ============================================================================


class TestReferenceLibrary:
    """Tests for ReferenceLibrary class."""

    def test_list_categories(self):
        """Test listing categories."""
        lib = ReferenceLibrary()
        categories = lib.list_categories()

        assert "educational" in categories
        assert "appnote" in categories
        assert "test" in categories

    def test_list_circuits(self):
        """Test listing all circuits."""
        lib = ReferenceLibrary()
        circuits = lib.list_circuits()

        assert len(circuits) == 10
        assert "ohms_law_demo" in circuits
        assert "led_driver" in circuits

    def test_list_circuits_by_category(self):
        """Test listing circuits by category."""
        lib = ReferenceLibrary()

        educational = lib.list_circuits("educational")
        assert len(educational) == 5
        assert "ohms_law_demo" in educational

        appnote = lib.list_circuits("appnote")
        assert len(appnote) == 3
        assert "led_driver" in appnote

        test_circuits = lib.list_circuits("test")
        assert len(test_circuits) == 2

    def test_get_circuit(self):
        """Test getting a circuit by name."""
        lib = ReferenceLibrary()
        ref = lib.get("ohms_law_demo")

        assert isinstance(ref, ReferenceCircuit)
        assert ref.name == "Ohm's Law Demonstration"

    def test_get_circuit_with_params(self):
        """Test getting a circuit with parameters."""
        lib = ReferenceLibrary()
        ref = lib.get("voltage_divider_demo", vin=24.0, r1=20000.0, r2=10000.0)

        assert ref.parameters["vin"] == 24.0
        assert ref.parameters["r1"] == 20000.0

    def test_get_nonexistent_circuit(self):
        """Test error on nonexistent circuit."""
        lib = ReferenceLibrary()

        with pytest.raises(KeyError, match="not found"):
            lib.get("nonexistent_circuit")

    def test_get_all_in_category(self):
        """Test getting all circuits in a category."""
        lib = ReferenceLibrary()
        educational = lib.get_all_in_category("educational")

        assert len(educational) == 5
        assert all(isinstance(ref, ReferenceCircuit) for ref in educational)
        assert all(ref.category == "educational" for ref in educational)


class TestModuleLevelLibrary:
    """Tests for module-level library instance."""

    def test_library_exists(self):
        """Test module-level library exists."""
        assert isinstance(library, ReferenceLibrary)

    def test_library_usage(self):
        """Test using module-level library."""
        ref = library.get("ohms_law_demo")
        assert isinstance(ref, ReferenceCircuit)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for reference circuits."""

    def test_all_circuits_valid(self):
        """Test all circuits can be instantiated."""
        lib = ReferenceLibrary()

        for name in lib.list_circuits():
            ref = lib.get(name)
            assert isinstance(ref.circuit, Circuit)
            assert ref.circuit.name is not None
            assert len(ref.components) > 0

    def test_all_circuits_have_metadata(self):
        """Test all circuits have required metadata."""
        lib = ReferenceLibrary()

        for name in lib.list_circuits():
            ref = lib.get(name)
            assert len(ref.learning_objectives) > 0
            assert len(ref.expected_results) > 0
            assert len(ref.suggested_analyses) > 0

    def test_educational_circuits_have_learning_objectives(self):
        """Test educational circuits have meaningful learning objectives."""
        lib = ReferenceLibrary()

        for ref in lib.get_all_in_category("educational"):
            assert len(ref.learning_objectives) >= 2
            # Check objectives are non-trivial
            for obj in ref.learning_objectives:
                assert len(obj) > 20

    def test_summaries_are_informative(self):
        """Test circuit summaries contain useful information."""
        lib = ReferenceLibrary()

        for name in lib.list_circuits():
            ref = lib.get(name)
            summary = ref.summary()

            assert len(summary) > 100
            assert "Learning Objectives" in summary
            assert "Suggested Analyses" in summary
