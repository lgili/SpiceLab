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
