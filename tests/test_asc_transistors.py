"""Tests for transistor/semiconductor ASC parser support."""

import pytest
from spicelab.core.components import JFET, ZenerDiode
from spicelab.io.ltspice_asc import (
    COMPONENT_TO_SYMBOL,
    SYMBOL_LIBRARY,
    AscComponent,
    AscPoint,
    parse_asc,
    schematic_to_circuit,
)
from spicelab.library.transistors import Bjt, Mosfet


class TestSymbolLibraryRegistration:
    """Tests for symbol registration in SYMBOL_LIBRARY."""

    def test_npn_registered(self):
        assert "npn" in SYMBOL_LIBRARY
        assert SYMBOL_LIBRARY["npn"] is not None

    def test_pnp_registered(self):
        assert "pnp" in SYMBOL_LIBRARY
        assert SYMBOL_LIBRARY["pnp"] is not None

    def test_nmos_registered(self):
        assert "nmos" in SYMBOL_LIBRARY
        assert SYMBOL_LIBRARY["nmos"] is not None

    def test_nmos4_registered(self):
        assert "nmos4" in SYMBOL_LIBRARY
        assert SYMBOL_LIBRARY["nmos4"] is not None

    def test_pmos_registered(self):
        assert "pmos" in SYMBOL_LIBRARY
        assert SYMBOL_LIBRARY["pmos"] is not None

    def test_pmos4_registered(self):
        assert "pmos4" in SYMBOL_LIBRARY
        assert SYMBOL_LIBRARY["pmos4"] is not None

    def test_njf_registered(self):
        assert "njf" in SYMBOL_LIBRARY
        assert SYMBOL_LIBRARY["njf"] is not None

    def test_pjf_registered(self):
        assert "pjf" in SYMBOL_LIBRARY
        assert SYMBOL_LIBRARY["pjf"] is not None

    def test_zener_registered(self):
        assert "zener" in SYMBOL_LIBRARY
        assert SYMBOL_LIBRARY["zener"] is not None


class TestComponentToSymbolMapping:
    """Tests for COMPONENT_TO_SYMBOL mapping."""

    def test_bjt_mapped(self):
        assert Bjt in COMPONENT_TO_SYMBOL

    def test_mosfet_mapped(self):
        assert Mosfet in COMPONENT_TO_SYMBOL

    def test_jfet_mapped(self):
        assert JFET in COMPONENT_TO_SYMBOL

    def test_zener_mapped(self):
        assert ZenerDiode in COMPONENT_TO_SYMBOL


class TestBjtBuilders:
    """Tests for BJT component builders."""

    def test_build_npn(self):
        spec = SYMBOL_LIBRARY["npn"]
        comp = AscComponent(
            symbol="npn",
            position=AscPoint(100, 100),
            orientation="R0",
            attributes={"InstName": "Q1", "Value": "2N2222"},
        )
        bjt = spec.build_component(comp)
        assert isinstance(bjt, Bjt)
        assert bjt.ref == "Q1"
        assert bjt.value == "2N2222"

    def test_build_pnp(self):
        spec = SYMBOL_LIBRARY["pnp"]
        comp = AscComponent(
            symbol="pnp",
            position=AscPoint(100, 100),
            orientation="R0",
            attributes={"InstName": "Q2", "Value": "2N2907"},
        )
        bjt = spec.build_component(comp)
        assert isinstance(bjt, Bjt)
        assert bjt.ref == "Q2"
        assert bjt.value == "2N2907"

    def test_bjt_has_three_pins(self):
        spec = SYMBOL_LIBRARY["npn"]
        assert len(spec.pin_offsets) == 3


class TestMosfetBuilders:
    """Tests for MOSFET component builders."""

    def test_build_nmos(self):
        spec = SYMBOL_LIBRARY["nmos"]
        comp = AscComponent(
            symbol="nmos",
            position=AscPoint(100, 100),
            orientation="R0",
            attributes={"InstName": "M1", "Value": "NMOS_BASIC"},
        )
        mos = spec.build_component(comp)
        assert isinstance(mos, Mosfet)
        assert mos.ref == "M1"
        assert mos.value == "NMOS_BASIC"

    def test_build_pmos(self):
        spec = SYMBOL_LIBRARY["pmos"]
        comp = AscComponent(
            symbol="pmos",
            position=AscPoint(100, 100),
            orientation="R0",
            attributes={"InstName": "M2", "Value": "PMOS_BASIC"},
        )
        mos = spec.build_component(comp)
        assert isinstance(mos, Mosfet)
        assert mos.ref == "M2"
        assert mos.value == "PMOS_BASIC"

    def test_mosfet_with_params(self):
        spec = SYMBOL_LIBRARY["nmos"]
        comp = AscComponent(
            symbol="nmos",
            position=AscPoint(100, 100),
            orientation="R0",
            attributes={"InstName": "M3", "Value": "NMOS", "Value2": "W=10u L=1u"},
        )
        mos = spec.build_component(comp)
        assert isinstance(mos, Mosfet)
        assert mos.params == "W=10u L=1u"

    def test_mosfet_has_four_pins(self):
        spec = SYMBOL_LIBRARY["nmos"]
        assert len(spec.pin_offsets) == 4


class TestJfetBuilders:
    """Tests for JFET component builders."""

    def test_build_njf(self):
        spec = SYMBOL_LIBRARY["njf"]
        comp = AscComponent(
            symbol="njf",
            position=AscPoint(100, 100),
            orientation="R0",
            attributes={"InstName": "J1", "Value": "2N5457"},
        )
        jfet = spec.build_component(comp)
        assert isinstance(jfet, JFET)
        assert jfet.ref == "J1"
        assert jfet.value == "2N5457"

    def test_build_pjf(self):
        spec = SYMBOL_LIBRARY["pjf"]
        comp = AscComponent(
            symbol="pjf",
            position=AscPoint(100, 100),
            orientation="R0",
            attributes={"InstName": "J2", "Value": "2N5460"},
        )
        jfet = spec.build_component(comp)
        assert isinstance(jfet, JFET)
        assert jfet.ref == "J2"
        assert jfet.value == "2N5460"

    def test_jfet_has_three_pins(self):
        spec = SYMBOL_LIBRARY["njf"]
        assert len(spec.pin_offsets) == 3


class TestZenerBuilder:
    """Tests for Zener diode component builder."""

    def test_build_zener(self):
        spec = SYMBOL_LIBRARY["zener"]
        comp = AscComponent(
            symbol="zener",
            position=AscPoint(100, 100),
            orientation="R0",
            attributes={"InstName": "D1", "Value": "1N4733"},
        )
        zener = spec.build_component(comp)
        assert isinstance(zener, ZenerDiode)
        assert zener.ref == "D1"
        assert zener.value == "1N4733"

    def test_zener_has_two_pins(self):
        spec = SYMBOL_LIBRARY["zener"]
        assert len(spec.pin_offsets) == 2


class TestAscParsingWithTransistors:
    """Integration tests for parsing ASC files with transistors."""

    def test_parse_npn_transistor(self):
        asc_content = """Version 4
SHEET 1 880 680
SYMBOL npn 200 200 R0
SYMATTR InstName Q1
SYMATTR Value 2N2222
SYMATTR SpiceLine Q1 nc nb ne 2N2222
FLAG 200 200 nc
FLAG 110 290 nb
FLAG 200 380 ne
"""
        schematic = parse_asc(asc_content)
        assert len(schematic.components) == 1
        comp = schematic.components[0]
        assert comp.symbol == "npn"
        assert comp.ref == "Q1"

    def test_parse_nmos_transistor(self):
        asc_content = """Version 4
SHEET 1 880 680
SYMBOL nmos 200 200 R0
SYMATTR InstName M1
SYMATTR Value NMOS_BASIC
SYMATTR SpiceLine M1 nd ng ns nb NMOS_BASIC
FLAG 200 200 nd
FLAG 110 290 ng
FLAG 200 380 ns
FLAG 290 290 nb
"""
        schematic = parse_asc(asc_content)
        assert len(schematic.components) == 1
        comp = schematic.components[0]
        assert comp.symbol == "nmos"
        assert comp.ref == "M1"

    def test_parse_jfet(self):
        asc_content = """Version 4
SHEET 1 880 680
SYMBOL njf 200 200 R0
SYMATTR InstName J1
SYMATTR Value 2N5457
SYMATTR SpiceLine J1 nd ng ns 2N5457
FLAG 200 200 nd
FLAG 110 290 ng
FLAG 200 380 ns
"""
        schematic = parse_asc(asc_content)
        assert len(schematic.components) == 1
        comp = schematic.components[0]
        assert comp.symbol == "njf"
        assert comp.ref == "J1"

    def test_parse_zener(self):
        asc_content = """Version 4
SHEET 1 880 680
SYMBOL zener 200 200 R0
SYMATTR InstName D1
SYMATTR Value 1N4733
SYMATTR SpiceLine D1 na nc 1N4733
FLAG 200 200 na
FLAG 380 200 nc
"""
        schematic = parse_asc(asc_content)
        assert len(schematic.components) == 1
        comp = schematic.components[0]
        assert comp.symbol == "zener"
        assert comp.ref == "D1"


class TestSchematicToCircuitWithTransistors:
    """Tests for converting schematic to circuit with transistors."""

    def test_schematic_to_circuit_npn(self):
        asc_content = """Version 4
SHEET 1 880 680
SYMBOL npn 200 200 R0
SYMATTR InstName Q1
SYMATTR Value 2N2222
SYMATTR SpiceLine Q1 nc nb ne 2N2222
FLAG 200 200 nc
FLAG 110 290 nb
FLAG 200 380 0
"""
        schematic = parse_asc(asc_content)
        circuit = schematic_to_circuit(schematic)
        bjts = [c for c in circuit._components if isinstance(c, Bjt)]
        assert len(bjts) == 1
        assert bjts[0].value == "2N2222"

    def test_schematic_to_circuit_nmos(self):
        asc_content = """Version 4
SHEET 1 880 680
SYMBOL nmos 200 200 R0
SYMATTR InstName M1
SYMATTR Value NMOS
SYMATTR SpiceLine M1 nd ng ns nb NMOS
FLAG 200 200 nd
FLAG 110 290 ng
FLAG 200 380 0
FLAG 290 290 nb
"""
        schematic = parse_asc(asc_content)
        circuit = schematic_to_circuit(schematic)
        mosfets = [c for c in circuit._components if isinstance(c, Mosfet)]
        assert len(mosfets) == 1
        assert mosfets[0].value == "NMOS"

    def test_schematic_to_circuit_jfet(self):
        asc_content = """Version 4
SHEET 1 880 680
SYMBOL njf 200 200 R0
SYMATTR InstName J1
SYMATTR Value 2N5457
SYMATTR SpiceLine J1 nd ng ns 2N5457
FLAG 200 200 nd
FLAG 110 290 ng
FLAG 200 380 0
"""
        schematic = parse_asc(asc_content)
        circuit = schematic_to_circuit(schematic)
        jfets = [c for c in circuit._components if isinstance(c, JFET)]
        assert len(jfets) == 1
        assert jfets[0].value == "2N5457"

    def test_schematic_to_circuit_zener(self):
        asc_content = """Version 4
SHEET 1 880 680
SYMBOL zener 200 200 R0
SYMATTR InstName D1
SYMATTR Value 1N4733
SYMATTR SpiceLine D1 na nc 1N4733
FLAG 200 200 na
FLAG 380 200 0
"""
        schematic = parse_asc(asc_content)
        circuit = schematic_to_circuit(schematic)
        zeners = [c for c in circuit._components if isinstance(c, ZenerDiode)]
        assert len(zeners) == 1
        assert zeners[0].value == "1N4733"


class TestTransistorAttributeExtraction:
    """Tests for attribute extraction from transistor components."""

    def test_bjt_attributes(self):
        spec = SYMBOL_LIBRARY["npn"]
        bjt = Bjt("Q1", "2N2222")
        attrs = spec.extract_attributes(bjt)
        assert attrs["InstName"] == "Q1"
        assert attrs["Value"] == "2N2222"

    def test_mosfet_attributes(self):
        spec = SYMBOL_LIBRARY["nmos"]
        mos = Mosfet("M1", "NMOS", "W=10u L=1u")
        attrs = spec.extract_attributes(mos)
        assert attrs["InstName"] == "M1"
        assert attrs["Value"] == "NMOS"
        assert attrs["Value2"] == "W=10u L=1u"

    def test_mosfet_attributes_no_params(self):
        spec = SYMBOL_LIBRARY["nmos"]
        mos = Mosfet("M1", "NMOS")
        attrs = spec.extract_attributes(mos)
        assert attrs["InstName"] == "M1"
        assert attrs["Value"] == "NMOS"
        assert "Value2" not in attrs

    def test_jfet_attributes(self):
        spec = SYMBOL_LIBRARY["njf"]
        jfet = JFET("J1", "2N5457")
        attrs = spec.extract_attributes(jfet)
        assert attrs["InstName"] == "J1"
        assert attrs["Value"] == "2N5457"

    def test_zener_attributes(self):
        spec = SYMBOL_LIBRARY["zener"]
        zener = ZenerDiode("D1", "1N4733")
        attrs = spec.extract_attributes(zener)
        assert attrs["InstName"] == "D1"
        assert attrs["Value"] == "1N4733"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
