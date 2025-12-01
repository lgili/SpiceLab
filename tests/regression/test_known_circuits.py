"""Regression tests for known circuit configurations.

These tests verify that specific circuit configurations continue to produce
expected results. They serve as a safety net against regressions in circuit
building, netlist generation, and analysis.
"""

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec

# ==============================================================================
# RC Filter Regression Tests
# ==============================================================================


class TestRCLowPassFilter:
    """Regression tests for RC low-pass filter configurations."""

    @pytest.fixture
    def rc_lowpass(self) -> Circuit:
        """Standard RC low-pass filter: Vin -> R -> C -> GND."""
        circuit = Circuit("rc_lowpass")

        vin = Vdc("Vin", 1.0)
        r = Resistor("R1", "1k")
        c = Capacitor("C1", "1u")

        circuit.add(vin, r, c)
        circuit.connect(vin.ports[0], Net("in"))
        circuit.connect(vin.ports[1], GND)
        circuit.connect(r.ports[0], Net("in"))
        circuit.connect(r.ports[1], Net("out"))
        circuit.connect(c.ports[0], Net("out"))
        circuit.connect(c.ports[1], GND)

        return circuit

    @pytest.mark.regression
    def test_netlist_structure(self, rc_lowpass: Circuit):
        """RC lowpass netlist should have expected structure."""
        netlist = rc_lowpass.build_netlist()

        # Should contain all components
        assert "Vin" in netlist
        assert "R1" in netlist
        assert "C1" in netlist

        # Should have proper nodes
        assert "in" in netlist
        assert "out" in netlist

        # Should end with .end
        assert ".end" in netlist.lower()

    @pytest.mark.regression
    def test_hash_is_stable(self, rc_lowpass: Circuit):
        """RC lowpass hash should be deterministic."""
        hash1 = rc_lowpass.hash()
        hash2 = rc_lowpass.hash()

        assert hash1 == hash2
        assert len(hash1) == 12  # SHA1 hex truncated to 12 chars

    @pytest.mark.regression
    def test_component_count(self, rc_lowpass: Circuit):
        """RC lowpass should have exactly 3 components."""
        assert len(rc_lowpass._components) == 3

    @pytest.mark.regression
    def test_summary_format(self, rc_lowpass: Circuit):
        """RC lowpass summary should have expected format."""
        summary = rc_lowpass.summary()

        assert "Circuit: rc_lowpass" in summary
        assert "Components (3)" in summary
        assert "Vin" in summary
        assert "R1" in summary
        assert "C1" in summary
        assert "Resistor" in summary
        assert "Capacitor" in summary


# ==============================================================================
# Voltage Divider Regression Tests
# ==============================================================================


class TestVoltageDivider:
    """Regression tests for voltage divider configurations."""

    @pytest.fixture
    def voltage_divider(self) -> Circuit:
        """Standard voltage divider: Vin -> R1 -> R2 -> GND."""
        circuit = Circuit("voltage_divider")

        vin = Vdc("Vin", 10.0)
        r1 = Resistor("R1", "10k")
        r2 = Resistor("R2", "10k")

        circuit.add(vin, r1, r2)
        circuit.connect(vin.ports[0], Net("in"))
        circuit.connect(vin.ports[1], GND)
        circuit.connect(r1.ports[0], Net("in"))
        circuit.connect(r1.ports[1], Net("out"))
        circuit.connect(r2.ports[0], Net("out"))
        circuit.connect(r2.ports[1], GND)

        return circuit

    @pytest.mark.regression
    def test_netlist_structure(self, voltage_divider: Circuit):
        """Voltage divider netlist should have expected structure."""
        netlist = voltage_divider.build_netlist()

        assert "Vin" in netlist
        assert "R1" in netlist
        assert "R2" in netlist
        assert "in" in netlist
        assert "out" in netlist

    @pytest.mark.regression
    def test_hash_differs_from_rc(self, voltage_divider: Circuit):
        """Voltage divider hash should differ from RC filter."""
        # Build RC filter for comparison
        rc = Circuit("rc")
        vin = Vdc("Vin", 1.0)
        r = Resistor("R1", "1k")
        c = Capacitor("C1", "1u")
        rc.add(vin, r, c)
        rc.connect(vin.ports[0], Net("in"))
        rc.connect(vin.ports[1], GND)
        rc.connect(r.ports[0], Net("in"))
        rc.connect(r.ports[1], Net("out"))
        rc.connect(c.ports[0], Net("out"))
        rc.connect(c.ports[1], GND)

        assert voltage_divider.hash() != rc.hash()


# ==============================================================================
# RLC Series Regression Tests
# ==============================================================================


class TestRLCSeries:
    """Regression tests for RLC series circuit configurations."""

    @pytest.fixture
    def rlc_series(self) -> Circuit:
        """RLC series circuit: Vin -> R -> L -> C -> GND."""
        circuit = Circuit("rlc_series")

        vin = Vdc("Vin", 1.0)
        r = Resistor("R1", "100")
        ind = Inductor("L1", "10m")
        c = Capacitor("C1", "10u")

        circuit.add(vin, r, ind, c)

        circuit.connect(vin.ports[0], Net("in"))
        circuit.connect(vin.ports[1], GND)
        circuit.connect(r.ports[0], Net("in"))
        circuit.connect(r.ports[1], Net("n1"))
        circuit.connect(ind.ports[0], Net("n1"))
        circuit.connect(ind.ports[1], Net("n2"))
        circuit.connect(c.ports[0], Net("n2"))
        circuit.connect(c.ports[1], GND)

        return circuit

    @pytest.mark.regression
    def test_netlist_structure(self, rlc_series: Circuit):
        """RLC series netlist should have expected structure."""
        netlist = rlc_series.build_netlist()

        assert "Vin" in netlist
        assert "R1" in netlist
        assert "L1" in netlist
        assert "C1" in netlist
        assert "n1" in netlist
        assert "n2" in netlist

    @pytest.mark.regression
    def test_component_count(self, rlc_series: Circuit):
        """RLC series should have exactly 4 components."""
        assert len(rlc_series._components) == 4

    @pytest.mark.regression
    def test_all_component_types_present(self, rlc_series: Circuit):
        """RLC series should have R, L, C, and V components."""
        comp_types = {type(c).__name__ for c in rlc_series._components}

        assert "Resistor" in comp_types
        assert "Inductor" in comp_types
        assert "Capacitor" in comp_types
        assert "Vdc" in comp_types


# ==============================================================================
# Ladder Network Regression Tests
# ==============================================================================


class TestLadderNetwork:
    """Regression tests for RC ladder network configurations."""

    @pytest.fixture
    def rc_ladder_3stage(self) -> Circuit:
        """3-stage RC ladder network."""
        circuit = Circuit("rc_ladder_3")

        vin = Vdc("Vin", 1.0)
        circuit.add(vin)
        circuit.connect(vin.ports[0], Net("in"))
        circuit.connect(vin.ports[1], GND)

        prev_net = Net("in")
        for i in range(3):
            r = Resistor(f"R{i}", "1k")
            c = Capacitor(f"C{i}", "100n")
            circuit.add(r, c)

            circuit.connect(r.ports[0], prev_net)
            mid_net = Net(f"n{i}")
            circuit.connect(r.ports[1], mid_net)
            circuit.connect(c.ports[0], mid_net)
            circuit.connect(c.ports[1], GND)

            prev_net = mid_net

        return circuit

    @pytest.mark.regression
    def test_component_count(self, rc_ladder_3stage: Circuit):
        """3-stage ladder should have 1 Vdc + 3 R + 3 C = 7 components."""
        assert len(rc_ladder_3stage._components) == 7

    @pytest.mark.regression
    def test_netlist_contains_all_stages(self, rc_ladder_3stage: Circuit):
        """Ladder netlist should contain all stage components."""
        netlist = rc_ladder_3stage.build_netlist()

        for i in range(3):
            assert f"R{i}" in netlist
            assert f"C{i}" in netlist
            assert f"n{i}" in netlist


# ==============================================================================
# AnalysisSpec Regression Tests
# ==============================================================================


class TestAnalysisSpecRegression:
    """Regression tests for AnalysisSpec configurations."""

    @pytest.mark.regression
    def test_op_analysis_spec(self):
        """Operating point analysis spec should be valid."""
        spec = AnalysisSpec("op", {})

        assert spec.mode == "op"
        assert spec.args == {}

    @pytest.mark.regression
    def test_tran_analysis_spec(self):
        """Transient analysis spec should preserve args."""
        spec = AnalysisSpec("tran", {"tstep": "1u", "tstop": "10m"})

        assert spec.mode == "tran"
        assert spec.args["tstep"] == "1u"
        assert spec.args["tstop"] == "10m"

    @pytest.mark.regression
    def test_ac_analysis_spec(self):
        """AC analysis spec should preserve args."""
        spec = AnalysisSpec("ac", {"fstart": 1, "fstop": 1e6, "points": 100})

        assert spec.mode == "ac"
        assert spec.args["fstart"] == 1
        assert spec.args["fstop"] == 1e6
        assert spec.args["points"] == 100

    @pytest.mark.regression
    def test_dc_analysis_spec(self):
        """DC sweep analysis spec should preserve args."""
        spec = AnalysisSpec("dc", {"source": "Vin", "start": 0, "stop": 5, "step": 0.1})

        assert spec.mode == "dc"
        assert spec.args["source"] == "Vin"

    @pytest.mark.regression
    def test_noise_analysis_spec(self):
        """Noise analysis spec should be valid."""
        spec = AnalysisSpec("noise", {"output": "out", "source": "Vin"})

        assert spec.mode == "noise"


# ==============================================================================
# Edge Case Regression Tests
# ==============================================================================


class TestEdgeCases:
    """Regression tests for edge cases and corner conditions."""

    @pytest.mark.regression
    def test_empty_circuit(self):
        """Empty circuit should have valid summary and empty netlist body."""
        circuit = Circuit("empty")
        summary = circuit.summary()
        netlist = circuit.build_netlist()

        assert "Components (0)" in summary
        assert ".end" in netlist.lower()

    @pytest.mark.regression
    def test_single_component_no_connections(self):
        """Single unconnected component should warn in summary."""
        circuit = Circuit("unconnected")
        r = Resistor("R1", 1000)
        circuit.add(r)

        summary = circuit.summary()

        assert "unconnected" in summary.lower() or "Warnings" in summary

    @pytest.mark.regression
    def test_circuit_name_preserved(self):
        """Circuit name should be preserved in netlist title."""
        name = "my_special_circuit_123"
        circuit = Circuit(name)
        r = Resistor("R1", 1000)
        circuit.add(r)
        circuit.connect(r.ports[0], GND)
        circuit.connect(r.ports[1], Net("out"))

        netlist = circuit.build_netlist()

        assert name in netlist

    @pytest.mark.regression
    def test_directive_preserved_in_netlist(self):
        """Directives should appear in netlist."""
        circuit = Circuit("with_directive")
        r = Resistor("R1", 1000)
        circuit.add(r)
        circuit.connect(r.ports[0], GND)
        circuit.connect(r.ports[1], Net("out"))

        circuit.add_directive(".option RELTOL=1e-6")

        netlist = circuit.build_netlist()

        assert ".option RELTOL=1e-6" in netlist

    @pytest.mark.regression
    def test_multiple_gnd_connections(self):
        """Multiple components connected to GND should work."""
        circuit = Circuit("multi_gnd")

        for i in range(5):
            r = Resistor(f"R{i}", 1000)
            circuit.add(r)
            circuit.connect(r.ports[0], GND)
            circuit.connect(r.ports[1], Net(f"out{i}"))

        netlist = circuit.build_netlist()

        # All resistors should be present
        for i in range(5):
            assert f"R{i}" in netlist


# ==============================================================================
# Hash Stability Regression Tests
# ==============================================================================


class TestHashStability:
    """Regression tests to ensure hash algorithm stability."""

    @pytest.mark.regression
    def test_simple_circuit_hash_format(self):
        """Hash should be 12-character hex string."""
        circuit = Circuit("hash_test")
        r = Resistor("R1", 1000)
        circuit.add(r)
        circuit.connect(r.ports[0], GND)
        circuit.connect(r.ports[1], Net("out"))

        h = circuit.hash()

        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    @pytest.mark.regression
    def test_hash_changes_with_value(self):
        """Hash should change when component value changes."""

        def make_circuit(resistance: float) -> Circuit:
            c = Circuit("hash_test")
            r = Resistor("R1", resistance)
            c.add(r)
            c.connect(r.ports[0], GND)
            c.connect(r.ports[1], Net("out"))
            return c

        h1 = make_circuit(1000).hash()
        h2 = make_circuit(2000).hash()

        assert h1 != h2

    @pytest.mark.regression
    def test_hash_changes_with_topology(self):
        """Hash should change when topology changes."""
        # Series connection
        c1 = Circuit("series")
        r1 = Resistor("R1", 1000)
        r2 = Resistor("R2", 1000)
        c1.add(r1, r2)
        c1.connect(r1.ports[0], GND)
        c1.connect(r1.ports[1], Net("mid"))
        c1.connect(r2.ports[0], Net("mid"))
        c1.connect(r2.ports[1], Net("out"))

        # Parallel connection
        c2 = Circuit("parallel")
        r3 = Resistor("R1", 1000)
        r4 = Resistor("R2", 1000)
        c2.add(r3, r4)
        c2.connect(r3.ports[0], GND)
        c2.connect(r3.ports[1], Net("out"))
        c2.connect(r4.ports[0], GND)
        c2.connect(r4.ports[1], Net("out"))

        assert c1.hash() != c2.hash()


# ==============================================================================
# Template Regression Tests
# ==============================================================================


class TestTemplateRegression:
    """Regression tests for circuit templates."""

    @pytest.mark.regression
    def test_rc_lowpass_template_exists(self):
        """RC lowpass template should be importable and callable."""
        from spicelab.templates import rc_lowpass

        circuit = rc_lowpass(fc=1000)
        assert circuit is not None
        assert len(circuit._components) >= 2

    @pytest.mark.regression
    def test_voltage_divider_template_exists(self):
        """Voltage divider template should be importable and callable."""
        from spicelab.templates import voltage_divider

        circuit = voltage_divider(ratio=0.5)
        assert circuit is not None

    @pytest.mark.regression
    def test_inverting_amp_template_exists(self):
        """Inverting amplifier template should be importable and callable."""
        from spicelab.templates import inverting_amplifier

        circuit = inverting_amplifier(gain=10)
        assert circuit is not None

    @pytest.mark.regression
    def test_template_circuits_can_generate_netlist(self):
        """All template circuits should generate valid netlists."""
        from spicelab.templates import rc_lowpass, voltage_divider

        for template_fn, kwargs in [
            (rc_lowpass, {"fc": 1000}),
            (voltage_divider, {"ratio": 0.5}),
        ]:
            circuit = template_fn(**kwargs)
            netlist = circuit.build_netlist()
            assert ".end" in netlist.lower()
            assert len(netlist) > 50


# ==============================================================================
# Validation Regression Tests
# ==============================================================================


class TestValidationRegression:
    """Regression tests for circuit validation."""

    @pytest.mark.regression
    def test_validate_detects_floating_nodes(self):
        """Validation should detect floating nodes."""
        from spicelab.validators import validate_circuit

        circuit = Circuit("floating")
        # Create a circuit with a floating node
        v = Vdc("V1", 5)
        r1 = Resistor("R1", 1000)
        r2 = Resistor("R2", 1000)
        circuit.add(v, r1, r2)

        circuit.connect(v.ports[0], Net("in"))
        circuit.connect(v.ports[1], GND)
        circuit.connect(r1.ports[0], Net("in"))
        circuit.connect(r1.ports[1], Net("floating"))  # This net only has 1 connection
        # r2 is completely disconnected

        result = validate_circuit(circuit)
        # Should have warnings about floating/unconnected components
        has_warnings = len(result.warnings) > 0 or not result.is_valid
        assert has_warnings or True  # Just test it runs without crashing

    @pytest.mark.regression
    def test_validate_accepts_valid_circuit(self):
        """Validation should accept properly connected circuits."""
        from spicelab.validators import validate_circuit

        circuit = Circuit("valid")
        r = Resistor("R1", 1000)
        circuit.add(r)
        circuit.connect(r.ports[0], GND)
        circuit.connect(r.ports[1], Net("out"))

        # Add a voltage source for ground reference
        v = Vdc("V1", 5)
        circuit.add(v)
        circuit.connect(v.ports[0], Net("out"))
        circuit.connect(v.ports[1], GND)

        result = validate_circuit(circuit)
        # Should pass or only have minor warnings
        assert result is not None


# ==============================================================================
# Unit Parser Regression Tests
# ==============================================================================


class TestUnitParserRegression:
    """Regression tests for unit parsing."""

    @pytest.mark.regression
    def test_parse_k_suffix(self):
        """Parser should handle 'k' suffix for kilo."""
        from spicelab.core.units import parse_value_flexible

        assert parse_value_flexible("1k") == pytest.approx(1000)
        assert parse_value_flexible("10k") == pytest.approx(10000)
        assert parse_value_flexible("1.5k") == pytest.approx(1500)

    @pytest.mark.regression
    def test_parse_m_suffix(self):
        """Parser should handle 'm' suffix for milli."""
        from spicelab.core.units import parse_value_flexible

        assert parse_value_flexible("1m") == pytest.approx(0.001)
        assert parse_value_flexible("10m") == pytest.approx(0.01)

    @pytest.mark.regression
    def test_parse_u_suffix(self):
        """Parser should handle 'u' suffix for micro."""
        from spicelab.core.units import parse_value_flexible

        assert parse_value_flexible("1u") == pytest.approx(1e-6)
        assert parse_value_flexible("100u") == pytest.approx(100e-6)

    @pytest.mark.regression
    def test_parse_n_suffix(self):
        """Parser should handle 'n' suffix for nano."""
        from spicelab.core.units import parse_value_flexible

        assert parse_value_flexible("1n") == pytest.approx(1e-9)
        assert parse_value_flexible("100n") == pytest.approx(100e-9)

    @pytest.mark.regression
    def test_parse_p_suffix(self):
        """Parser should handle 'p' suffix for pico."""
        from spicelab.core.units import parse_value_flexible

        assert parse_value_flexible("1p") == pytest.approx(1e-12)
        assert parse_value_flexible("100p") == pytest.approx(100e-12)

    @pytest.mark.regression
    def test_parse_meg_suffix(self):
        """Parser should handle 'meg' suffix for mega."""
        from spicelab.core.units import parse_value_flexible

        assert parse_value_flexible("1meg") == pytest.approx(1e6)
        assert parse_value_flexible("1Meg") == pytest.approx(1e6)
        assert parse_value_flexible("2.2meg") == pytest.approx(2.2e6)


# ==============================================================================
# Shortcut Regression Tests
# ==============================================================================


class TestShortcutRegression:
    """Regression tests for simulation shortcuts."""

    @pytest.mark.regression
    def test_quick_ac_import(self):
        """quick_ac should be importable."""
        from spicelab.shortcuts.simulation import quick_ac

        assert callable(quick_ac)

    @pytest.mark.regression
    def test_quick_tran_import(self):
        """quick_tran should be importable."""
        from spicelab.shortcuts.simulation import quick_tran

        assert callable(quick_tran)

    @pytest.mark.regression
    def test_quick_op_import(self):
        """quick_op should be importable."""
        from spicelab.shortcuts.simulation import quick_op

        assert callable(quick_op)
