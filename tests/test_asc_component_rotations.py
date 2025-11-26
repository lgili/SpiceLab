"""Tests for component PIN_OFFSETS with various rotations.

These tests create minimal ASC snippets to verify that PIN_OFFSETS
work correctly for all supported rotations (R0, R90, R180, R270, M0, etc).
"""

import pytest
from spicelab.io.asc_converter import (
    PIN_OFFSETS,
    _get_pin_positions,
    _rotate_point,
)
from spicelab.io.asc_parser import parse_asc_string

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def create_component_asc(symbol: str, x: int, y: int, rotation: str, ref: str) -> str:
    """Create minimal ASC content with a single component."""
    return f"""Version 4
SHEET 1 880 680
SYMBOL {symbol} {x} {y} {rotation}
SYMATTR InstName {ref}
SYMATTR Value 1k
"""


def expected_pins_for_rotation(
    base_offsets: list[tuple[int, int]], comp_x: int, comp_y: int, rotation: str
) -> list[tuple[int, int]]:
    """Calculate expected pin positions for a given rotation."""
    result = []
    for ox, oy in base_offsets:
        rx, ry = _rotate_point(ox, oy, rotation)
        result.append((comp_x + rx, comp_y + ry))
    return result


# ---------------------------------------------------------------------------
# Tests for resistor rotations
# ---------------------------------------------------------------------------


class TestResistorRotations:
    """Test resistor pin positions for all rotations."""

    @pytest.fixture
    def base_offsets(self):
        return PIN_OFFSETS["res"]

    @pytest.mark.parametrize(
        "rotation",
        ["R0", "R90", "R180", "R270"],
    )
    def test_resistor_rotation(self, rotation, base_offsets):
        """Test resistor pin positions for a given rotation."""
        x, y = 200, 200
        asc = parse_asc_string(create_component_asc("res", x, y, rotation, "R1"))

        comp = asc.components[0]
        actual_pins = _get_pin_positions(comp)
        expected_pins = expected_pins_for_rotation(base_offsets, x, y, rotation)

        assert (
            actual_pins == expected_pins
        ), f"Rotation {rotation}: expected {expected_pins}, got {actual_pins}"

    def test_resistor_r0_pin_values(self, base_offsets):
        """Verify R0 resistor pins have correct absolute positions."""
        x, y = 100, 100
        asc = parse_asc_string(create_component_asc("res", x, y, "R0", "R1"))

        pins = _get_pin_positions(asc.components[0])

        # With R0, offsets are applied directly
        # Pin 0: (100+16, 100+16) = (116, 116)
        # Pin 1: (100+16, 100+96) = (116, 196)
        assert pins[0] == (x + base_offsets[0][0], y + base_offsets[0][1])
        assert pins[1] == (x + base_offsets[1][0], y + base_offsets[1][1])

    def test_resistor_r90_pin_values(self, base_offsets):
        """Verify R90 resistor (horizontal) pins have correct positions."""
        x, y = 100, 100
        asc = parse_asc_string(create_component_asc("res", x, y, "R90", "R1"))

        pins = _get_pin_positions(asc.components[0])

        # R90 rotation: (ox, oy) -> (-oy, ox)
        expected_pin0 = (x + (-base_offsets[0][1]), y + base_offsets[0][0])
        expected_pin1 = (x + (-base_offsets[1][1]), y + base_offsets[1][0])

        assert pins[0] == expected_pin0
        assert pins[1] == expected_pin1


# ---------------------------------------------------------------------------
# Tests for voltage source rotations
# ---------------------------------------------------------------------------


class TestVoltageSourceRotations:
    """Test voltage source pin positions for all rotations."""

    @pytest.fixture
    def base_offsets(self):
        return PIN_OFFSETS["voltage"]

    @pytest.mark.parametrize(
        "rotation",
        ["R0", "R90", "R180", "R270"],
    )
    def test_voltage_rotation(self, rotation, base_offsets):
        """Test voltage source pin positions for a given rotation."""
        x, y = 200, 200
        asc = parse_asc_string(create_component_asc("voltage", x, y, rotation, "V1"))

        comp = asc.components[0]
        actual_pins = _get_pin_positions(comp)
        expected_pins = expected_pins_for_rotation(base_offsets, x, y, rotation)

        assert (
            actual_pins == expected_pins
        ), f"Rotation {rotation}: expected {expected_pins}, got {actual_pins}"


# ---------------------------------------------------------------------------
# Tests for opamp rotations
# ---------------------------------------------------------------------------


class TestOpAmpRotations:
    """Test opamp pin positions for all rotations."""

    @pytest.fixture
    def base_offsets(self):
        return PIN_OFFSETS["universalopamp"]

    @pytest.mark.parametrize(
        "rotation",
        ["R0", "R90", "R180", "R270"],
    )
    def test_opamp_rotation(self, rotation, base_offsets):
        """Test opamp pin positions for a given rotation."""
        x, y = 200, 200
        asc = parse_asc_string(
            create_component_asc("OpAmps\\\\UniversalOpAmp", x, y, rotation, "U1")
        )

        comp = asc.components[0]
        actual_pins = _get_pin_positions(comp)
        expected_pins = expected_pins_for_rotation(base_offsets, x, y, rotation)

        assert (
            actual_pins == expected_pins
        ), f"Rotation {rotation}: expected {expected_pins}, got {actual_pins}"

    def test_opamp_r0_pin_semantics(self, base_offsets):
        """Verify R0 opamp pin order: [inp (+), inn (-), out]."""
        x, y = 100, 148
        pins = expected_pins_for_rotation(base_offsets, x, y, "R0")

        # Pin 0 = inp (+) should be below center (positive Y offset)
        # Pin 1 = inn (-) should be above center (negative Y offset)
        # Pin 2 = out should be to the right (positive X offset)

        inp, inn, out = pins

        # inp has larger Y than inn (below = larger Y in screen coords)
        assert inp[1] > inn[1], "inp should be below inn"

        # Both inputs on left side of symbol center
        assert inp[0] < x, "inp should be left of center"
        assert inn[0] < x, "inn should be left of center"

        # Output on right side
        assert out[0] > x, "out should be right of center"

    def test_opamp_has_three_pins(self, base_offsets):
        """OpAmps must have exactly 3 pins."""
        assert len(base_offsets) == 3, "OpAmp should have 3 pins"


# ---------------------------------------------------------------------------
# Tests for mirrored rotations
# ---------------------------------------------------------------------------


class TestMirroredRotations:
    """Test mirrored component rotations (M0, M90, etc)."""

    def test_resistor_m0_mirror(self):
        """Test M0 (horizontal mirror) for resistor."""
        base = PIN_OFFSETS["res"]
        x, y = 100, 100

        asc = parse_asc_string(create_component_asc("res", x, y, "M0", "R1"))
        pins = _get_pin_positions(asc.components[0])

        # M0: (ox, oy) -> (-ox, oy)
        expected = [(x - base[0][0], y + base[0][1]), (x - base[1][0], y + base[1][1])]

        assert pins == expected

    def test_resistor_m180_mirror(self):
        """Test M180 (vertical mirror) for resistor."""
        base = PIN_OFFSETS["res"]
        x, y = 100, 100

        asc = parse_asc_string(create_component_asc("res", x, y, "M180", "R1"))
        pins = _get_pin_positions(asc.components[0])

        # M180: (ox, oy) -> (ox, -oy)
        expected = [(x + base[0][0], y - base[0][1]), (x + base[1][0], y - base[1][1])]

        assert pins == expected


# ---------------------------------------------------------------------------
# Tests for capacitor rotations
# ---------------------------------------------------------------------------


class TestCapacitorRotations:
    """Test capacitor pin positions for rotations."""

    @pytest.fixture
    def base_offsets(self):
        return PIN_OFFSETS["cap"]

    @pytest.mark.parametrize("rotation", ["R0", "R90", "R180", "R270"])
    def test_capacitor_rotation(self, rotation, base_offsets):
        """Test capacitor pin positions for a given rotation."""
        x, y = 200, 200
        asc = parse_asc_string(create_component_asc("cap", x, y, rotation, "C1"))

        comp = asc.components[0]
        actual_pins = _get_pin_positions(comp)
        expected_pins = expected_pins_for_rotation(base_offsets, x, y, rotation)

        assert actual_pins == expected_pins


# ---------------------------------------------------------------------------
# Tests for inductor rotations
# ---------------------------------------------------------------------------


class TestInductorRotations:
    """Test inductor pin positions for rotations."""

    @pytest.fixture
    def base_offsets(self):
        return PIN_OFFSETS["ind"]

    @pytest.mark.parametrize("rotation", ["R0", "R90"])
    def test_inductor_rotation(self, rotation, base_offsets):
        """Test inductor pin positions for a given rotation."""
        x, y = 200, 200
        asc = parse_asc_string(create_component_asc("ind", x, y, rotation, "L1"))

        comp = asc.components[0]
        actual_pins = _get_pin_positions(comp)
        expected_pins = expected_pins_for_rotation(base_offsets, x, y, rotation)

        assert actual_pins == expected_pins


# ---------------------------------------------------------------------------
# Tests for diode rotations
# ---------------------------------------------------------------------------


class TestDiodeRotations:
    """Test diode pin positions for rotations."""

    @pytest.fixture
    def base_offsets(self):
        return PIN_OFFSETS["diode"]

    @pytest.mark.parametrize("rotation", ["R0", "R90", "R180", "R270"])
    def test_diode_rotation(self, rotation, base_offsets):
        """Test diode pin positions for a given rotation."""
        x, y = 200, 200
        asc = parse_asc_string(create_component_asc("diode", x, y, rotation, "D1"))

        comp = asc.components[0]
        actual_pins = _get_pin_positions(comp)
        expected_pins = expected_pins_for_rotation(base_offsets, x, y, rotation)

        assert actual_pins == expected_pins


# ---------------------------------------------------------------------------
# Cross-component consistency tests
# ---------------------------------------------------------------------------


class TestCrossComponentConsistency:
    """Tests to verify consistency across component types."""

    def test_2pin_components_have_same_structure(self):
        """All 2-pin passives should have similar offset structure."""
        two_pin_symbols = ["res", "cap", "ind", "diode"]

        for symbol in two_pin_symbols:
            offsets = PIN_OFFSETS.get(symbol, [])
            assert len(offsets) == 2, f"{symbol} should have 2 pins"

            # Both pins should have same X offset (vertically aligned in R0)
            assert (
                offsets[0][0] == offsets[1][0]
            ), f"{symbol} pins should be vertically aligned in R0"

            # Pin 1 should be below pin 0 (larger Y)
            assert offsets[1][1] > offsets[0][1], f"{symbol} pin 1 should be below pin 0"

    def test_voltage_current_sources_symmetric(self):
        """Voltage and current sources should have same offsets."""
        assert PIN_OFFSETS["voltage"] == PIN_OFFSETS["current"]

    def test_res_res2_equivalent(self):
        """res and res2 should have same offsets."""
        assert PIN_OFFSETS["res"] == PIN_OFFSETS["res2"]

    def test_opamp_variants_equivalent(self):
        """All opamp variants should have same offsets."""
        opamp_variants = ["opamp", "opamp2", "universalopamp", "universalopamp2"]
        base = PIN_OFFSETS["opamp"]

        for variant in opamp_variants:
            assert PIN_OFFSETS[variant] == base, f"{variant} should match opamp offsets"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
