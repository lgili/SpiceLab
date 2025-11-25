"""Tests for the ASC parser module."""

import pytest
from spicelab.io.asc_parser import (
    Flag,
    Point,
    SymbolComponent,
    TextElement,
    Wire,
    parse_asc_string,
)

# ---------------------------------------------------------------------------
# Sample ASC content for testing
# ---------------------------------------------------------------------------

SAMPLE_ASC = """Version 4.1
SHEET 1 880 680
WIRE 100 100 200 100
WIRE 200 100 200 200
FLAG 200 200 0
FLAG 100 100 vin
SYMBOL res 100 100 R0
SYMATTR InstName R1
SYMATTR Value 10k
SYMBOL cap 200 150 R0
SYMATTR InstName C1
SYMATTR Value 100n
TEXT 50 300 Left 2 !.tran 0 1m 0
TEXT 50 350 Left 2 !.param R=10k
TEXT 50 400 Left 2 !.meas Vout MAX V(vout)
TEXT 50 450 Left 2 ;This is a comment
"""

MINIMAL_ASC = """Version 4
SHEET 1 400 300
WIRE 0 0 100 0
FLAG 50 0 GND
"""


# ---------------------------------------------------------------------------
# Tests for data classes
# ---------------------------------------------------------------------------


class TestPoint:
    """Tests for Point dataclass."""

    def test_point_creation(self):
        p = Point(10, 20)
        assert p.x == 10
        assert p.y == 20

    def test_point_iteration(self):
        p = Point(10, 20)
        x, y = p
        assert x == 10
        assert y == 20


class TestWire:
    """Tests for Wire dataclass."""

    def test_wire_creation(self):
        w = Wire(0, 0, 100, 0)
        assert w.x1 == 0
        assert w.y1 == 0
        assert w.x2 == 100
        assert w.y2 == 0

    def test_wire_start_end(self):
        w = Wire(10, 20, 30, 40)
        assert w.start == Point(10, 20)
        assert w.end == Point(30, 40)

    def test_wire_horizontal(self):
        w = Wire(0, 50, 100, 50)
        assert w.is_horizontal is True
        assert w.is_vertical is False

    def test_wire_vertical(self):
        w = Wire(50, 0, 50, 100)
        assert w.is_horizontal is False
        assert w.is_vertical is True

    def test_wire_length(self):
        w = Wire(0, 0, 3, 4)
        assert w.length == 5.0


class TestFlag:
    """Tests for Flag dataclass."""

    def test_flag_creation(self):
        f = Flag("vin", 100, 200)
        assert f.name == "vin"
        assert f.x == 100
        assert f.y == 200

    def test_flag_position(self):
        f = Flag("vin", 100, 200)
        assert f.position == Point(100, 200)

    def test_flag_is_ground(self):
        assert Flag("0", 0, 0).is_ground is True
        assert Flag("GND", 0, 0).is_ground is True
        assert Flag("gnd", 0, 0).is_ground is True
        assert Flag("vin", 0, 0).is_ground is False


class TestSymbolComponent:
    """Tests for SymbolComponent dataclass."""

    def test_component_creation(self):
        c = SymbolComponent(
            symbol="res",
            x=100,
            y=200,
            rotation="R0",
            attributes={"InstName": "R1", "Value": "10k"},
        )
        assert c.symbol == "res"
        assert c.x == 100
        assert c.y == 200
        assert c.rotation == "R0"

    def test_component_ref(self):
        c = SymbolComponent("res", 0, 0, "R0", {"InstName": "R1"})
        assert c.ref == "R1"

    def test_component_value(self):
        c = SymbolComponent("res", 0, 0, "R0", {"Value": "10k"})
        assert c.value == "10k"

    def test_component_symbol_basename(self):
        c = SymbolComponent("OpAmps\\UniversalOpAmp", 0, 0, "R0")
        assert c.symbol_basename == "UniversalOpAmp"

    def test_component_type(self):
        assert SymbolComponent("res", 0, 0, "R0", {"InstName": "R1"}).component_type == "resistor"
        assert SymbolComponent("cap", 0, 0, "R0", {"InstName": "C1"}).component_type == "capacitor"
        assert SymbolComponent("ind", 0, 0, "R0", {"InstName": "L1"}).component_type == "inductor"
        vsrc = SymbolComponent("voltage", 0, 0, "R0", {"InstName": "V1"})
        assert vsrc.component_type == "voltage_source"
        assert SymbolComponent("opamp", 0, 0, "R0", {"InstName": "U1"}).component_type == "ic"


class TestTextElement:
    """Tests for TextElement dataclass."""

    def test_text_element_creation(self):
        t = TextElement(100, 200, "Left", 2, ".tran 0 1m", is_directive=True)
        assert t.x == 100
        assert t.y == 200
        assert t.text == ".tran 0 1m"
        assert t.is_directive is True

    def test_text_is_comment(self):
        t1 = TextElement(0, 0, "Left", 2, ";This is a comment")
        t2 = TextElement(0, 0, "Left", 2, ".tran 0 1m")
        assert t1.is_comment is True
        assert t2.is_comment is False

    def test_text_directive_type(self):
        t1 = TextElement(0, 0, "Left", 2, ".tran 0 1m", is_directive=True)
        t2 = TextElement(0, 0, "Left", 2, ".param R=10k", is_directive=True)
        t3 = TextElement(0, 0, "Left", 2, "comment", is_directive=False)
        assert t1.directive_type == "tran"
        assert t2.directive_type == "param"
        assert t3.directive_type is None


# ---------------------------------------------------------------------------
# Tests for parser
# ---------------------------------------------------------------------------


class TestAscParser:
    """Tests for AscParser class."""

    def test_parse_version(self):
        result = parse_asc_string(SAMPLE_ASC)
        assert result.version == "4.1"

    def test_parse_sheet(self):
        result = parse_asc_string(SAMPLE_ASC)
        assert result.sheet_number == 1
        assert result.sheet_width == 880
        assert result.sheet_height == 680

    def test_parse_wires(self):
        result = parse_asc_string(SAMPLE_ASC)
        assert len(result.wires) == 2
        w = result.wires[0]
        assert w.x1 == 100
        assert w.y1 == 100
        assert w.x2 == 200
        assert w.y2 == 100

    def test_parse_flags(self):
        result = parse_asc_string(SAMPLE_ASC)
        assert len(result.flags) == 2
        # Check ground flag
        ground_flags = [f for f in result.flags if f.is_ground]
        assert len(ground_flags) == 1
        assert ground_flags[0].position == Point(200, 200)

    def test_parse_components(self):
        result = parse_asc_string(SAMPLE_ASC)
        assert len(result.components) == 2
        # Check resistor
        r1 = result.get_component_by_ref("R1")
        assert r1 is not None
        assert r1.symbol == "res"
        assert r1.value == "10k"
        # Check capacitor
        c1 = result.get_component_by_ref("C1")
        assert c1 is not None
        assert c1.symbol == "cap"
        assert c1.value == "100n"

    def test_parse_texts(self):
        result = parse_asc_string(SAMPLE_ASC)
        assert len(result.texts) == 4
        # Check directives
        directives = result.directives
        assert len(directives) == 3
        # Check comment
        comments = result.comments
        assert len(comments) == 1
        assert "comment" in comments[0].text

    def test_parse_parameters(self):
        result = parse_asc_string(SAMPLE_ASC)
        assert "R" in result.parameters
        assert result.parameters["R"].value == "10k"

    def test_parse_measurements(self):
        result = parse_asc_string(SAMPLE_ASC)
        assert len(result.measurements) == 1
        m = result.measurements[0]
        assert m.name == "Vout"
        assert m.measurement_type == "MAX"

    def test_parse_analysis_commands(self):
        result = parse_asc_string(SAMPLE_ASC)
        assert len(result.analysis_commands) == 1
        cmd = result.analysis_commands[0]
        assert cmd.analysis_type == "tran"


class TestAscParseResult:
    """Tests for AscParseResult helper methods."""

    def test_net_names(self):
        result = parse_asc_string(SAMPLE_ASC)
        assert "0" in result.net_names
        assert "vin" in result.net_names

    def test_ground_flags(self):
        result = parse_asc_string(SAMPLE_ASC)
        ground = result.ground_flags
        assert len(ground) == 1

    def test_get_components_by_type(self):
        result = parse_asc_string(SAMPLE_ASC)
        resistors = result.get_components_by_type("R")
        capacitors = result.get_components_by_type("C")
        assert len(resistors) == 1
        assert len(capacitors) == 1

    def test_summary(self):
        result = parse_asc_string(SAMPLE_ASC)
        summary = result.summary()
        assert summary["version"] == "4.1"
        assert summary["components"] == 2
        assert summary["wires"] == 2
        assert summary["flags"] == 2


class TestMinimalAsc:
    """Tests for minimal ASC content."""

    def test_parse_minimal(self):
        result = parse_asc_string(MINIMAL_ASC)
        assert result.version == "4"
        assert len(result.wires) == 1
        assert len(result.flags) == 1
        assert result.flags[0].name == "GND"
        assert result.flags[0].is_ground is True


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_content(self):
        result = parse_asc_string("")
        assert result.version == "4"  # default
        assert len(result.components) == 0

    def test_version_only(self):
        result = parse_asc_string("Version 5\n")
        assert result.version == "5"

    def test_unknown_lines(self):
        content = "Version 4\nUNKNOWN_ELEMENT 1 2 3\n"
        result = parse_asc_string(content)
        assert len(result.unknown_lines) == 1
        assert "UNKNOWN_ELEMENT" in result.unknown_lines[0]

    def test_component_without_attributes(self):
        content = """Version 4
SHEET 1 400 300
SYMBOL res 100 100 R0
"""
        result = parse_asc_string(content)
        assert len(result.components) == 1
        assert result.components[0].ref is None
        assert result.components[0].value is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
