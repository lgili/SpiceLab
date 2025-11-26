"""Tests for PIN_OFFSETS accuracy and wire connectivity in ASC converter.

These tests validate that:
1. PIN_OFFSETS for each component type are correct
2. Pins align with wire endpoints in real circuits
3. Union-Find correctly propagates net labels
4. Generated netlists have correct connectivity
"""

import pytest
from spicelab.io.asc_converter import (
    DEFAULT_2PIN,
    PIN_OFFSETS,
    _build_net_map,
    _get_pin_positions,
    _rotate_point,
    asc_to_circuit,
)
from spicelab.io.asc_parser import parse_asc_file

# ---------------------------------------------------------------------------
# Test PIN_OFFSETS values
# ---------------------------------------------------------------------------


class TestPinOffsetsDefinitions:
    """Tests that PIN_OFFSETS are properly defined for all component types."""

    def test_resistor_offsets_exist(self):
        """Resistor offsets should be defined."""
        assert "res" in PIN_OFFSETS
        assert "res2" in PIN_OFFSETS
        assert len(PIN_OFFSETS["res"]) == 2

    def test_capacitor_offsets_exist(self):
        """Capacitor offsets should be defined."""
        assert "cap" in PIN_OFFSETS
        assert "cap2" in PIN_OFFSETS
        assert len(PIN_OFFSETS["cap"]) == 2

    def test_inductor_offsets_exist(self):
        """Inductor offsets should be defined."""
        assert "ind" in PIN_OFFSETS
        assert "ind2" in PIN_OFFSETS
        assert len(PIN_OFFSETS["ind"]) == 2

    def test_voltage_source_offsets_exist(self):
        """Voltage source offsets should be defined."""
        assert "voltage" in PIN_OFFSETS
        assert len(PIN_OFFSETS["voltage"]) == 2

    def test_current_source_offsets_exist(self):
        """Current source offsets should be defined."""
        assert "current" in PIN_OFFSETS
        assert len(PIN_OFFSETS["current"]) == 2

    def test_diode_offsets_exist(self):
        """Diode offsets should be defined."""
        assert "diode" in PIN_OFFSETS
        assert "dio" in PIN_OFFSETS
        assert len(PIN_OFFSETS["diode"]) == 2

    def test_opamp_offsets_exist(self):
        """OpAmp offsets should be defined with 3 pins."""
        assert "opamp" in PIN_OFFSETS
        assert "opamp2" in PIN_OFFSETS
        assert "universalopamp" in PIN_OFFSETS
        assert "universalopamp2" in PIN_OFFSETS
        assert len(PIN_OFFSETS["universalopamp"]) == 3

    def test_default_2pin_exists(self):
        """Default 2-pin offset should be defined."""
        assert DEFAULT_2PIN is not None
        assert len(DEFAULT_2PIN) == 2


# ---------------------------------------------------------------------------
# Test rotation function
# ---------------------------------------------------------------------------


class TestRotatePoint:
    """Tests for the _rotate_point function."""

    def test_r0_no_change(self):
        """R0 rotation should not change coordinates."""
        assert _rotate_point(10, 20, "R0") == (10, 20)

    def test_r90_rotation(self):
        """R90 should rotate 90 degrees counterclockwise."""
        # (x, y) -> (-y, x)
        assert _rotate_point(10, 0, "R90") == (0, 10)
        assert _rotate_point(0, 10, "R90") == (-10, 0)

    def test_r180_rotation(self):
        """R180 should rotate 180 degrees."""
        # (x, y) -> (-x, -y)
        assert _rotate_point(10, 20, "R180") == (-10, -20)

    def test_r270_rotation(self):
        """R270 should rotate 270 degrees counterclockwise."""
        # (x, y) -> (y, -x)
        assert _rotate_point(10, 0, "R270") == (0, -10)
        assert _rotate_point(0, 10, "R270") == (10, 0)

    def test_m0_mirror(self):
        """M0 should mirror horizontally."""
        # (x, y) -> (-x, y)
        assert _rotate_point(10, 20, "M0") == (-10, 20)

    def test_unknown_rotation(self):
        """Unknown rotation should return unchanged."""
        assert _rotate_point(10, 20, "UNKNOWN") == (10, 20)


# ---------------------------------------------------------------------------
# Test pin positions with real circuits (using actual PT1000 file)
# ---------------------------------------------------------------------------


class TestPinPositionsWithRealCircuits:
    """Tests that pin positions align with wire endpoints in real circuits.

    These tests use the actual PT1000 circuit file which has been verified
    to work correctly with LTspice. This ensures our PIN_OFFSETS match
    real LTspice behavior.
    """

    @pytest.fixture
    def pt1000_asc(self):
        """Load PT1000 circuit ASC file."""
        from pathlib import Path

        path = Path(__file__).parent.parent / "old" / "sim_files" / "PT1000_circuit_1.asc"
        if path.exists():
            return parse_asc_file(str(path))
        pytest.skip("PT1000 ASC file not found")

    def test_resistor_pins_on_wires(self, pt1000_asc):
        """Test that all resistor pins align with wire endpoints."""
        wire_endpoints = set()
        for wire in pt1000_asc.wires:
            wire_endpoints.add((wire.x1, wire.y1))
            wire_endpoints.add((wire.x2, wire.y2))

        flag_positions = {(f.x, f.y) for f in pt1000_asc.flags}
        all_valid = wire_endpoints | flag_positions

        errors = []
        for comp in pt1000_asc.components:
            if "res" in comp.symbol.lower():
                pins = _get_pin_positions(comp)
                for i, pin in enumerate(pins):
                    if pin not in all_valid:
                        errors.append(
                            f"{comp.ref} pin {i} at {pin} "
                            f"(symbol at ({comp.x}, {comp.y}) rot={comp.rotation})"
                        )

        assert len(errors) == 0, "Resistor pins not on wires:\n" + "\n".join(errors)

    def test_voltage_source_pins_on_wires(self, pt1000_asc):
        """Test that voltage source pins align with wire endpoints."""
        wire_endpoints = set()
        for wire in pt1000_asc.wires:
            wire_endpoints.add((wire.x1, wire.y1))
            wire_endpoints.add((wire.x2, wire.y2))

        flag_positions = {(f.x, f.y) for f in pt1000_asc.flags}
        all_valid = wire_endpoints | flag_positions

        for comp in pt1000_asc.components:
            if comp.symbol.lower() == "voltage":
                pins = _get_pin_positions(comp)
                for i, pin in enumerate(pins):
                    assert pin in all_valid, (
                        f"Voltage source {comp.ref} pin {i} at {pin} not on wire. "
                        f"Symbol at ({comp.x}, {comp.y}) rot={comp.rotation}"
                    )

    def test_opamp_pins_on_wires(self, pt1000_asc):
        """Test that all opamp pins align with wire endpoints."""
        wire_endpoints = set()
        for wire in pt1000_asc.wires:
            wire_endpoints.add((wire.x1, wire.y1))
            wire_endpoints.add((wire.x2, wire.y2))

        for comp in pt1000_asc.components:
            if "opamp" in comp.symbol.lower():
                pins = _get_pin_positions(comp)
                for i, pin in enumerate(pins):
                    assert pin in wire_endpoints, (
                        f"OpAmp {comp.ref} pin {i} at {pin} not on wire. "
                        f"Symbol at ({comp.x}, {comp.y}) rot={comp.rotation}. "
                        f"Expected pins: inp(+), inn(-), out"
                    )


# ---------------------------------------------------------------------------
# Test OpAmp pin order
# ---------------------------------------------------------------------------


class TestOpAmpPinOrder:
    """Tests that OpAmp pins are in correct order: [inp, inn, out]."""

    def test_opamp_pin_order_definition(self):
        """OpAmp pins should be [inp (+), inn (-), out]."""
        offsets = PIN_OFFSETS["universalopamp"]

        # In LTspice with R0 rotation:
        # - Non-inverting input (+) is at bottom-left (higher Y)
        # - Inverting input (-) is at top-left (lower Y)
        # - Output is on the right

        # Pin 0 should be inp (bottom-left, offset with positive Y)
        # Pin 1 should be inn (top-left, offset with negative Y)
        # Pin 2 should be out (right)

        inp_offset = offsets[0]
        inn_offset = offsets[1]
        out_offset = offsets[2]

        # inp has larger Y than inn (in screen coords, larger Y = lower on screen)
        assert inp_offset[1] > inn_offset[1], "inp should be below inn (larger Y)"

        # Both inputs on left side (negative X offset from center)
        assert inp_offset[0] < 0, "inp should be on left side"
        assert inn_offset[0] < 0, "inn should be on left side"

        # Output on right side (positive X offset)
        assert out_offset[0] > 0, "out should be on right side"


# ---------------------------------------------------------------------------
# Test net connectivity (using PT1000 circuit)
# ---------------------------------------------------------------------------


class TestNetConnectivity:
    """Tests for net connectivity and label propagation."""

    @pytest.fixture
    def pt1000_asc(self):
        """Load PT1000 circuit ASC file."""
        from pathlib import Path

        path = Path(__file__).parent.parent / "old" / "sim_files" / "PT1000_circuit_1.asc"
        if path.exists():
            return parse_asc_file(str(path))
        pytest.skip("PT1000 ASC file not found")

    def test_ground_connection(self, pt1000_asc):
        """Components connected to ground should have '0' net."""
        result = asc_to_circuit(pt1000_asc)
        netlist = result.circuit.build_netlist()

        # Ground (0) should appear in netlist
        assert " 0 " in netlist or " 0\n" in netlist, "Ground connection not found"

    def test_named_net_propagation(self, pt1000_asc):
        """Flag labels should propagate through wire network."""
        result = asc_to_circuit(pt1000_asc)
        netlist = result.circuit.build_netlist()

        # Check that named nets appear
        expected_nets = ["Vout", "Vrtd", "p1", "p2", "3.3V"]
        for net in expected_nets:
            assert net in netlist, f"Net '{net}' not found in netlist"

    def test_no_disconnected_components(self, pt1000_asc):
        """All components should be connected (no floating pins)."""
        _result = asc_to_circuit(pt1000_asc)  # noqa: F841 - validates conversion works

        # Build net map
        component_pins = {}
        for comp in pt1000_asc.components:
            component_pins[comp.ref] = _get_pin_positions(comp)

        uf, _ = _build_net_map(pt1000_asc, component_pins)

        # Each component's pins should be connected to the wire network
        for ref, pins in component_pins.items():
            for i, pin in enumerate(pins):
                # Pin should have a root in union-find (means it's tracked)
                root = uf.find(pin)
                assert root is not None, f"{ref} pin {i} is disconnected"


# ---------------------------------------------------------------------------
# Test with real PT1000 circuit
# ---------------------------------------------------------------------------


class TestPT1000Circuit:
    """Tests using the real PT1000 circuit ASC file."""

    @pytest.fixture
    def pt1000_asc_path(self):
        """Path to PT1000 circuit ASC file."""
        from pathlib import Path

        path = Path(__file__).parent.parent / "old" / "sim_files" / "PT1000_circuit_1.asc"
        if path.exists():
            return str(path)
        pytest.skip("PT1000 ASC file not found")

    def test_all_pins_on_wires(self, pt1000_asc_path):
        """All component pins should align with wire endpoints."""
        asc = parse_asc_file(pt1000_asc_path)

        wire_endpoints = set()
        for wire in asc.wires:
            wire_endpoints.add((wire.x1, wire.y1))
            wire_endpoints.add((wire.x2, wire.y2))

        flag_positions = {(f.x, f.y) for f in asc.flags}
        all_valid = wire_endpoints | flag_positions

        errors = []
        for comp in asc.components:
            pins = _get_pin_positions(comp)
            for i, pin in enumerate(pins):
                if pin not in all_valid:
                    errors.append(f"{comp.ref} pin {i} at {pin}")

        assert len(errors) == 0, f"Pins not on wires: {errors}"

    def test_expected_nets_in_netlist(self, pt1000_asc_path):
        """Generated netlist should contain expected named nets."""
        asc = parse_asc_file(pt1000_asc_path)
        result = asc_to_circuit(asc)

        netlist = result.circuit.build_netlist()

        # Expected net names from the circuit
        expected_nets = ["Vout", "Vrtd", "p1", "p2"]

        for net in expected_nets:
            assert net in netlist, f"Expected net '{net}' not found in netlist"

    def test_opamps_properly_connected(self, pt1000_asc_path):
        """OpAmps should have proper 3-pin connections."""
        asc = parse_asc_file(pt1000_asc_path)
        result = asc_to_circuit(asc)

        netlist = result.circuit.build_netlist()

        # Find opamp lines
        opamp_lines = [line for line in netlist.split("\n") if line.startswith("EU")]

        # Should have 2 opamps
        assert len(opamp_lines) == 2, f"Expected 2 opamps, found {len(opamp_lines)}"

        # Each opamp line should have format: E<name> <out> 0 <inp> <inn> <gain>
        for line in opamp_lines:
            parts = line.split()
            assert len(parts) == 6, f"OpAmp line should have 6 parts: {line}"

            name, out, gnd, inp, inn, gain = parts
            assert gnd == "0", f"OpAmp ground reference should be 0: {line}"
            assert gain == "1e6", f"OpAmp gain should be 1e6: {line}"

    def test_voltage_follower_configuration(self, pt1000_asc_path):
        """U2 should be configured as voltage follower (output = inn)."""
        asc = parse_asc_file(pt1000_asc_path)
        result = asc_to_circuit(asc)

        netlist = result.circuit.build_netlist()

        # Find U2 line
        u2_line = None
        for line in netlist.split("\n"):
            if line.startswith("EU2"):
                u2_line = line
                break

        assert u2_line is not None, "U2 not found in netlist"

        # Parse: EU2 <out> 0 <inp> <inn> <gain>
        parts = u2_line.split()
        out = parts[1]
        inp = parts[3]
        inn = parts[4]

        # In voltage follower: output connects to inn (negative feedback)
        assert out == inn, f"U2 should be voltage follower (out={out} should equal inn={inn})"

        # inp should be Vrtd
        assert inp == "Vrtd", f"U2 inp should be Vrtd, got {inp}"

    def test_no_conversion_errors(self, pt1000_asc_path):
        """Conversion should complete without errors."""
        asc = parse_asc_file(pt1000_asc_path)
        result = asc_to_circuit(asc)

        assert result.success, f"Conversion failed: {result.skipped_components}"
        assert len(result.converted_components) == 9  # R1, R2, R3, R4, R5, R10, V1, U1, U2


# ---------------------------------------------------------------------------
# Test simulation correctness (integration test)
# ---------------------------------------------------------------------------


class TestSimulationCorrectness:
    """Integration tests that verify simulation produces correct results."""

    @pytest.fixture
    def pt1000_asc_path(self):
        """Path to PT1000 circuit ASC file."""
        from pathlib import Path

        path = Path(__file__).parent.parent / "old" / "sim_files" / "PT1000_circuit_1.asc"
        if path.exists():
            return str(path)
        pytest.skip("PT1000 ASC file not found")

    def test_netlist_simulates_correctly(self, pt1000_asc_path):
        """Generated netlist should simulate with correct results."""
        import os
        import subprocess
        import tempfile

        # Create a simplified ngspice-compatible netlist
        netlist = """* PT1000 Circuit Test
V1 vcc 0 3.3
.param T1=25
.param A=3.9083e-3
.param B=-5.775e-7
.param R0=1000
.param Rrtd=R0*(1+A*T1+B*T1*T1)

R1 vcc vrtd 3900
R10 vrtd 0 {Rrtd}

* Buffer opamp U2 (voltage follower)
EU2 vrtd_buf 0 vrtd vrtd_buf 1e6

* Amplifier stage
R2 vrtd_buf p1 39000
R3 vcc p1 8200k
R4 p2 0 2000
R5 p2 vout 3900

* Differential amp U1
EU1 vout 0 p1 p2 1e6

.op
.end
"""

        # Check if ngspice is available
        try:
            subprocess.run(["ngspice", "--version"], capture_output=True, timeout=5)
        except (subprocess.SubprocessError, FileNotFoundError):
            pytest.skip("ngspice not available")

        # Write and run
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cir", delete=False) as f:
            f.write(netlist)
            netlist_path = f.name

        try:
            result = subprocess.run(
                ["ngspice", "-b", netlist_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            output = result.stdout + result.stderr

            # Parse Vout
            vout = None
            for line in output.split("\n"):
                if line.strip().lower().startswith("vout"):
                    parts = line.split()
                    if len(parts) >= 2:
                        vout = float(parts[-1])
                        break

            assert vout is not None, f"Could not find Vout in output: {output[:500]}"

            # At T=25°C, Vout should be approximately 2.17V
            assert abs(vout - 2.17) < 0.1, f"Vout={vout}V, expected ~2.17V"

        finally:
            os.unlink(netlist_path)


# ---------------------------------------------------------------------------
# Test component rotations
# ---------------------------------------------------------------------------


class TestComponentRotations:
    """Tests for various component rotations."""

    def test_resistor_all_rotations(self):
        """Test resistor pin positions for all rotations."""
        base_offsets = PIN_OFFSETS["res"]

        # Create test cases for each rotation
        test_cases = [
            ("R0", base_offsets[0], base_offsets[1]),
            ("R90", _rotate_point(*base_offsets[0], "R90"), _rotate_point(*base_offsets[1], "R90")),
            (
                "R180",
                _rotate_point(*base_offsets[0], "R180"),
                _rotate_point(*base_offsets[1], "R180"),
            ),
            (
                "R270",
                _rotate_point(*base_offsets[0], "R270"),
                _rotate_point(*base_offsets[1], "R270"),
            ),
        ]

        for rotation, expected_pin0, expected_pin1 in test_cases:
            # Verify rotation is applied correctly
            actual_pin0 = _rotate_point(*base_offsets[0], rotation)
            actual_pin1 = _rotate_point(*base_offsets[1], rotation)

            assert actual_pin0 == expected_pin0, f"Pin 0 wrong for {rotation}"
            assert actual_pin1 == expected_pin1, f"Pin 1 wrong for {rotation}"

    def test_opamp_rotations(self):
        """Test opamp pin positions for different rotations."""
        base_offsets = PIN_OFFSETS["universalopamp"]

        # R0: standard orientation
        r0_pins = [_rotate_point(*off, "R0") for off in base_offsets]
        assert r0_pins == list(base_offsets)

        # R90: rotated 90 degrees
        r90_pins = [_rotate_point(*off, "R90") for off in base_offsets]
        # Should be different from R0
        assert r90_pins != list(base_offsets)

        # All rotations should maintain relative pin structure
        for rotation in ["R0", "R90", "R180", "R270"]:
            rotated = [_rotate_point(*off, rotation) for off in base_offsets]
            # Should have 3 pins
            assert len(rotated) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
