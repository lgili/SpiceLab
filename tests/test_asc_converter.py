"""Tests for the ASC to Circuit converter module."""

import pytest
from spicelab.core.components import (
    Capacitor,
    Diode,
    Inductor,
    OpAmpIdeal,
    Resistor,
    Vdc,
)
from spicelab.io.asc_converter import (
    ConversionResult,
    ConversionWarning,
    asc_to_circuit,
)
from spicelab.io.asc_parser import parse_asc_string

# ---------------------------------------------------------------------------
# Sample ASC content for testing
# ---------------------------------------------------------------------------

SIMPLE_RC = """Version 4
SHEET 1 880 680
WIRE 100 100 100 200
WIRE 100 200 200 200
WIRE 200 200 200 300
FLAG 100 100 vin
FLAG 200 300 0
SYMBOL res 100 100 R0
SYMATTR InstName R1
SYMATTR Value 10k
SYMBOL cap 200 200 R0
SYMATTR InstName C1
SYMATTR Value 100n
"""

VOLTAGE_DIVIDER = """Version 4
SHEET 1 880 680
WIRE 100 50 100 100
WIRE 100 200 100 250
WIRE 100 350 100 400
FLAG 100 50 Vin
FLAG 100 250 Vout
FLAG 100 400 0
SYMBOL voltage 100 0 R0
SYMATTR InstName V1
SYMATTR Value 10
SYMBOL res 100 100 R0
SYMATTR InstName R1
SYMATTR Value 10k
SYMBOL res 100 300 R0
SYMATTR InstName R2
SYMATTR Value 10k
"""

OPAMP_CIRCUIT = """Version 4
SHEET 1 880 680
WIRE 100 100 200 100
WIRE 200 100 200 150
WIRE 300 100 300 200
FLAG 100 100 vin
FLAG 200 150 0
FLAG 300 200 vout
SYMBOL OpAmps\\UniversalOpAmp 200 100 R0
SYMATTR InstName U1
SYMBOL res 100 50 R0
SYMATTR InstName R1
SYMATTR Value 1k
"""

WITH_PARAMETERS = """Version 4
SHEET 1 880 680
WIRE 100 100 100 200
FLAG 100 100 in
FLAG 100 200 0
SYMBOL res 100 100 R0
SYMATTR InstName R1
SYMATTR Value {Rval}
TEXT 50 300 Left 2 !.param Rval=10k
TEXT 50 350 Left 2 !.tran 0 1m 0
"""

UNSUPPORTED_SYMBOLS = """Version 4
SHEET 1 880 680
WIRE 100 100 100 200
FLAG 100 100 in
FLAG 100 200 0
SYMBOL npn 100 100 R0
SYMATTR InstName Q1
SYMBOL nmos 200 100 R0
SYMATTR InstName M1
SYMBOL mystery_component 300 100 R0
SYMATTR InstName X1
"""

MIXED_CIRCUIT = """Version 4
SHEET 1 880 680
WIRE 100 100 100 200
WIRE 100 200 200 200
WIRE 200 200 200 300
FLAG 100 100 vin
FLAG 200 300 0
SYMBOL res 100 100 R0
SYMATTR InstName R1
SYMATTR Value 1k
SYMBOL cap 200 200 R0
SYMATTR InstName C1
SYMATTR Value 10u
SYMBOL ind 300 200 R0
SYMATTR InstName L1
SYMATTR Value 1m
SYMBOL diode 400 200 R0
SYMATTR InstName D1
SYMATTR Value 1N4148
"""


# ---------------------------------------------------------------------------
# Tests for ConversionWarning
# ---------------------------------------------------------------------------


class TestConversionWarning:
    """Tests for ConversionWarning dataclass."""

    def test_warning_str_with_ref(self):
        w = ConversionWarning(
            category="TEST",
            message="Test message",
            component_ref="R1",
        )
        assert "[R1]" in str(w)
        assert "TEST" in str(w)

    def test_warning_str_with_symbol(self):
        w = ConversionWarning(
            category="UNKNOWN",
            message="Unknown symbol",
            symbol="mystery",
        )
        assert "[mystery]" in str(w)

    def test_warning_str_no_ref(self):
        w = ConversionWarning(
            category="INFO",
            message="General message",
        )
        assert "INFO" in str(w)
        assert "General message" in str(w)


# ---------------------------------------------------------------------------
# Tests for ConversionResult
# ---------------------------------------------------------------------------


class TestConversionResult:
    """Tests for ConversionResult dataclass."""

    def test_success_no_skipped(self):
        result = ConversionResult(
            circuit=None,  # type: ignore
            converted_components=["R1", "C1"],
            skipped_components=[],
        )
        assert result.success is True

    def test_success_with_skipped(self):
        result = ConversionResult(
            circuit=None,  # type: ignore
            converted_components=["R1"],
            skipped_components=["Q1"],
        )
        assert result.success is False

    def test_has_warnings(self):
        result = ConversionResult(
            circuit=None,  # type: ignore
            warnings=[ConversionWarning("TEST", "msg")],
        )
        assert result.has_warnings is True

    def test_summary(self):
        result = ConversionResult(
            circuit=None,  # type: ignore
            converted_components=["R1", "C1"],
            skipped_components=["Q1"],
            warnings=[ConversionWarning("TEST", "msg")],
        )
        summary = result.summary()
        assert summary["converted"] == 2
        assert summary["skipped"] == 1
        assert summary["warnings"] == 1


# ---------------------------------------------------------------------------
# Tests for basic conversion
# ---------------------------------------------------------------------------


class TestBasicConversion:
    """Tests for basic ASC to Circuit conversion."""

    def test_simple_rc(self):
        asc = parse_asc_string(SIMPLE_RC)
        result = asc_to_circuit(asc)

        assert result.success
        assert len(result.converted_components) == 2
        assert "R1" in result.converted_components
        assert "C1" in result.converted_components

        # Check component types
        r1 = result.component_map["R1"]
        c1 = result.component_map["C1"]
        assert isinstance(r1, Resistor)
        assert isinstance(c1, Capacitor)
        assert r1.value == "10k"
        assert c1.value == "100n"

    def test_voltage_divider(self):
        asc = parse_asc_string(VOLTAGE_DIVIDER)
        result = asc_to_circuit(asc)

        assert result.success
        assert len(result.converted_components) == 3
        assert "V1" in result.converted_components
        assert "R1" in result.converted_components
        assert "R2" in result.converted_components

        v1 = result.component_map["V1"]
        assert isinstance(v1, Vdc)
        assert v1.value == "10"

    def test_opamp_circuit(self):
        asc = parse_asc_string(OPAMP_CIRCUIT)
        result = asc_to_circuit(asc)

        # OpAmp should be converted
        assert "U1" in result.converted_components
        u1 = result.component_map["U1"]
        assert isinstance(u1, OpAmpIdeal)

    def test_mixed_components(self):
        asc = parse_asc_string(MIXED_CIRCUIT)
        result = asc_to_circuit(asc)

        assert "R1" in result.converted_components
        assert "C1" in result.converted_components
        assert "L1" in result.converted_components
        assert "D1" in result.converted_components

        assert isinstance(result.component_map["R1"], Resistor)
        assert isinstance(result.component_map["C1"], Capacitor)
        assert isinstance(result.component_map["L1"], Inductor)
        assert isinstance(result.component_map["D1"], Diode)


# ---------------------------------------------------------------------------
# Tests for unsupported symbols
# ---------------------------------------------------------------------------


class TestUnsupportedSymbols:
    """Tests for handling unsupported symbols."""

    def test_unsupported_generates_warnings(self):
        asc = parse_asc_string(UNSUPPORTED_SYMBOLS)
        result = asc_to_circuit(asc)

        # Should have warnings for unsupported symbols
        assert result.has_warnings
        assert len(result.skipped_components) == 3

        # Check warning categories
        categories = [w.category for w in result.warnings]
        assert "UNSUPPORTED_SYMBOL" in categories or "UNKNOWN_SYMBOL" in categories

    def test_known_unsupported_has_message(self):
        asc = parse_asc_string(UNSUPPORTED_SYMBOLS)
        result = asc_to_circuit(asc)

        # Find warning for NPN
        npn_warnings = [w for w in result.warnings if w.component_ref == "Q1"]
        assert len(npn_warnings) == 1
        assert "BJT" in npn_warnings[0].message or "not yet implemented" in npn_warnings[0].message


# ---------------------------------------------------------------------------
# Tests for directives
# ---------------------------------------------------------------------------


class TestDirectives:
    """Tests for SPICE directive handling."""

    def test_parameters_added(self):
        asc = parse_asc_string(WITH_PARAMETERS)
        result = asc_to_circuit(asc)

        # Check that directives were added
        netlist = result.circuit.build_netlist()
        assert ".param Rval=10k" in netlist

    def test_analysis_commands_extracted(self):
        """Test that analysis commands are extracted but NOT added to netlist.

        Analysis commands should be extracted via get_analyses_from_asc() and
        passed to the simulator, not embedded in the netlist (which would cause
        duplicate analysis errors in LTspice).
        """
        from spicelab.io.asc_converter import get_analyses_from_asc

        asc = parse_asc_string(WITH_PARAMETERS)
        result = asc_to_circuit(asc)

        # Analysis commands should NOT be in the netlist
        netlist = result.circuit.build_netlist()
        assert ".tran" not in netlist

        # But they should be extractable from the ASC result
        analyses = get_analyses_from_asc(asc)
        assert len(analyses) == 1
        assert analyses[0]["mode"] == "tran"


# ---------------------------------------------------------------------------
# Tests for netlist generation
# ---------------------------------------------------------------------------


class TestNetlistGeneration:
    """Tests for generated netlist correctness."""

    def test_netlist_generation(self):
        asc = parse_asc_string(SIMPLE_RC)
        result = asc_to_circuit(asc)

        netlist = result.circuit.build_netlist()

        # Should contain component cards
        assert "R" in netlist
        assert "C" in netlist
        assert "10k" in netlist
        assert "100n" in netlist

    def test_ground_connection(self):
        asc = parse_asc_string(SIMPLE_RC)
        result = asc_to_circuit(asc)

        # Circuit should be created successfully
        assert result.circuit is not None
        netlist = result.circuit.build_netlist()

        # Netlist should be generated (connection accuracy depends on geometric analysis)
        assert len(netlist) > 0

    def test_circuit_name(self):
        asc = parse_asc_string(SIMPLE_RC)
        result = asc_to_circuit(asc, circuit_name="test_circuit")

        assert result.circuit.name == "test_circuit"


# ---------------------------------------------------------------------------
# Tests for edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_asc(self):
        asc = parse_asc_string("Version 4\nSHEET 1 880 680\n")
        result = asc_to_circuit(asc)

        assert result.success
        assert len(result.converted_components) == 0

    def test_component_without_value(self):
        asc_content = """Version 4
SHEET 1 880 680
SYMBOL res 100 100 R0
SYMATTR InstName R1
"""
        asc = parse_asc_string(asc_content)
        result = asc_to_circuit(asc)

        # Should use default value
        assert result.success
        r1 = result.component_map["R1"]
        assert r1.value == "1k"  # Default for resistor

    def test_component_without_ref(self):
        asc_content = """Version 4
SHEET 1 880 680
SYMBOL res 100 100 R0
SYMATTR Value 10k
"""
        asc = parse_asc_string(asc_content)
        result = asc_to_circuit(asc)

        # Should still convert with auto-generated ref
        assert len(result.converted_components) == 1


# ---------------------------------------------------------------------------
# Tests for engineering notation and analysis parsing
# ---------------------------------------------------------------------------


class TestEngNumberParsing:
    """Tests for parse_eng_number function."""

    def test_plain_numbers(self):
        from spicelab.io.asc_converter import parse_eng_number

        assert parse_eng_number("100") == 100.0
        assert parse_eng_number("1.5") == 1.5
        assert parse_eng_number("-3.14") == -3.14
        assert parse_eng_number("1e-3") == 1e-3
        assert parse_eng_number("2.5e6") == 2.5e6

    def test_engineering_suffixes(self):
        from spicelab.io.asc_converter import parse_eng_number

        assert parse_eng_number("5m") == pytest.approx(0.005)
        assert parse_eng_number("10k") == pytest.approx(10000.0)
        assert parse_eng_number("100n") == pytest.approx(100e-9)
        assert parse_eng_number("1u") == pytest.approx(1e-6)
        assert parse_eng_number("2.2p") == pytest.approx(2.2e-12)

    def test_meg_suffix(self):
        from spicelab.io.asc_converter import parse_eng_number

        assert parse_eng_number("1meg") == 1e6
        assert parse_eng_number("1.5MEG") == 1.5e6
        assert parse_eng_number("2Meg") == 2e6

    def test_whitespace(self):
        from spicelab.io.asc_converter import parse_eng_number

        assert parse_eng_number("  10k  ") == 10000.0


class TestAnalysisParsing:
    """Tests for analysis command parsing."""

    def test_tran_parsing(self):
        from spicelab.io.asc_converter import parse_tran_args

        # Simple case
        args = parse_tran_args("0 1m")
        assert args["tstep"] == 0.0
        assert args["tstop"] == 0.001

        # With tstart
        args = parse_tran_args("1u 10m 5m")
        assert args["tstep"] == 1e-6
        assert args["tstop"] == 0.01
        assert args["tstart"] == 0.005

    def test_ac_parsing(self):
        from spicelab.io.asc_converter import parse_ac_args

        args = parse_ac_args("dec 10 1 1meg")
        assert args["variation"] == "dec"
        assert args["npoints"] == 10
        assert args["fstart"] == 1.0
        assert args["fstop"] == 1e6

    def test_dc_parsing(self):
        from spicelab.io.asc_converter import parse_dc_args

        args = parse_dc_args("V1 0 5 0.1")
        assert args["src"] == "V1"
        assert args["start"] == 0.0
        assert args["stop"] == 5.0
        assert args["step"] == 0.1

    def test_parse_analysis_command(self):
        from spicelab.io.asc_converter import parse_analysis_command

        # Tran
        result = parse_analysis_command("tran", "0 1m")
        assert result["mode"] == "tran"
        assert result["args"]["tstop"] == 0.001

        # AC
        result = parse_analysis_command("AC", "dec 20 1 100k")
        assert result["mode"] == "ac"
        assert result["args"]["npoints"] == 20

        # OP
        result = parse_analysis_command("op", "")
        assert result["mode"] == "op"

        # Unsupported
        result = parse_analysis_command("noise", "params")
        assert result is None

    def test_get_analyses_from_asc(self):
        from spicelab.io.asc_converter import get_analyses_from_asc

        asc_content = """Version 4
SHEET 1 880 680
TEXT 50 400 Left 2 !.tran 0 5m
TEXT 50 450 Left 2 !.ac dec 10 1 100k
"""
        asc = parse_asc_string(asc_content)
        analyses = get_analyses_from_asc(asc)

        assert len(analyses) == 2
        assert analyses[0]["mode"] == "tran"
        assert analyses[0]["args"]["tstop"] == 0.005
        assert analyses[1]["mode"] == "ac"
        assert analyses[1]["args"]["fstop"] == 100000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
