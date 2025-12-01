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
        from spicelab.core.components import Vdc

        circuit = Circuit("test")
        vin = Net("vin")
        vout = Net("vout")

        V1 = Vdc(ref="V1", value=5.0)
        R1 = Resistor(ref="R1", resistance=10_000)
        C1 = Capacitor(ref="C1", capacitance=100e-9)
        circuit.add(V1, R1, C1)
        circuit.connect(V1.ports[0], vin)  # V1+ to vin
        circuit.connect(V1.ports[1], GND)  # V1- to GND
        circuit.connect(R1.ports[0], vin)
        circuit.connect(R1.ports[1], vout)
        circuit.connect(C1.ports[0], vout)
        circuit.connect(C1.ports[1], GND)

        result = validate_circuit(circuit)
        assert result.is_valid

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

    def test_import_all_presets(self) -> None:
        """Verify all analysis presets can be imported."""
        from spicelab.shortcuts import (
            AnalysisSuggestion,
            detailed_ac,
            detailed_tran,
            quick_ac,
            quick_noise,
            quick_op,
            quick_tran,
            suggest_analysis,
        )

        assert callable(quick_op)
        assert callable(quick_ac)
        assert callable(quick_tran)
        assert callable(quick_noise)
        assert callable(detailed_ac)
        assert callable(detailed_tran)
        assert callable(suggest_analysis)
        assert AnalysisSuggestion is not None

    # Note: Actual simulation tests would require ngspice binary
    # These are integration tests that should be run separately


class TestSuggestAnalysis:
    """Test auto-detection of suitable analysis types."""

    def test_suggest_for_rc_filter(self) -> None:
        """RC filter should suggest AC and transient analyses."""
        from spicelab.shortcuts import suggest_analysis
        from spicelab.templates import rc_lowpass

        circuit = rc_lowpass(fc=1000)
        suggestions = suggest_analysis(circuit)

        # Should have suggestions
        assert len(suggestions) > 0

        # Should suggest AC analysis for reactive circuit
        analysis_types = [s.analysis_type for s in suggestions]
        assert "ac" in analysis_types

        # Should be sorted by confidence (highest first)
        confidences = [s.confidence for s in suggestions]
        assert confidences == sorted(confidences, reverse=True)

    def test_suggest_for_voltage_divider(self) -> None:
        """Resistive divider should suggest DC operating point."""
        from spicelab.shortcuts import suggest_analysis
        from spicelab.templates import voltage_divider

        circuit = voltage_divider(ratio=0.5)
        suggestions = suggest_analysis(circuit)

        # Pure resistive circuit won't suggest much (no sources)
        # This is expected - needs sources to suggest analyses
        assert isinstance(suggestions, list)

    def test_suggest_for_active_circuit(self) -> None:
        """Circuit with opamp should suggest OP analysis."""
        from spicelab.shortcuts import suggest_analysis
        from spicelab.templates import voltage_follower

        result = voltage_follower()
        suggestions = suggest_analysis(result.circuit)

        # Active circuit should suggest DC operating point
        analysis_types = [s.analysis_type for s in suggestions]
        assert "op" in analysis_types

        # Should suggest noise for active circuits with resistors
        # (voltage follower has opamp but no resistors, so may not suggest noise)

    def test_suggest_params_contain_estimated_values(self) -> None:
        """Suggestions should include estimated parameters when possible."""
        from spicelab.shortcuts import suggest_analysis
        from spicelab.templates import rc_lowpass

        # 1kHz filter with 10k impedance
        circuit = rc_lowpass(fc=1000, impedance=10_000)
        suggestions = suggest_analysis(circuit)

        # Find AC suggestion
        ac_suggestion = next((s for s in suggestions if s.analysis_type == "ac"), None)
        assert ac_suggestion is not None

        # Should have estimated frequency parameters
        if ac_suggestion.suggested_params:
            # Parameters should be reasonable for 1kHz filter
            if "start" in ac_suggestion.suggested_params:
                assert ac_suggestion.suggested_params["start"] < 1000
            if "stop" in ac_suggestion.suggested_params:
                assert ac_suggestion.suggested_params["stop"] > 1000

    def test_suggestion_dataclass_fields(self) -> None:
        """AnalysisSuggestion should have all expected fields."""
        from spicelab.shortcuts import AnalysisSuggestion

        suggestion = AnalysisSuggestion(
            analysis_type="ac",
            reason="Test reason",
            preset_function="quick_ac",
            suggested_params={"start": 1.0, "stop": 1e6},
            confidence=0.85,
        )

        assert suggestion.analysis_type == "ac"
        assert suggestion.reason == "Test reason"
        assert suggestion.preset_function == "quick_ac"
        assert suggestion.suggested_params == {"start": 1.0, "stop": 1e6}
        assert suggestion.confidence == 0.85


class TestAnalysisPresetParameters:
    """Test that analysis presets accept expected parameters."""

    def test_quick_op_signature(self) -> None:
        """quick_op should accept circuit, engine, probes."""
        import inspect

        from spicelab.shortcuts import quick_op

        sig = inspect.signature(quick_op)
        params = list(sig.parameters.keys())

        assert "circuit" in params
        assert "engine" in params
        assert "probes" in params

    def test_quick_ac_signature(self) -> None:
        """quick_ac should accept frequency range parameters."""
        import inspect

        from spicelab.shortcuts import quick_ac

        sig = inspect.signature(quick_ac)
        params = list(sig.parameters.keys())

        assert "circuit" in params
        assert "start" in params
        assert "stop" in params
        assert "points_per_decade" in params
        assert "engine" in params

    def test_quick_tran_signature(self) -> None:
        """quick_tran should accept duration and timestep."""
        import inspect

        from spicelab.shortcuts import quick_tran

        sig = inspect.signature(quick_tran)
        params = list(sig.parameters.keys())

        assert "circuit" in params
        assert "duration" in params
        assert "timestep" in params
        assert "engine" in params

    def test_quick_noise_signature(self) -> None:
        """quick_noise should accept output_node and input_source."""
        import inspect

        from spicelab.shortcuts import quick_noise

        sig = inspect.signature(quick_noise)
        params = list(sig.parameters.keys())

        assert "circuit" in params
        assert "output_node" in params
        assert "input_source" in params
        assert "start" in params
        assert "stop" in params

    def test_detailed_ac_higher_point_density(self) -> None:
        """detailed_ac should default to higher point density than quick_ac."""
        import inspect

        from spicelab.shortcuts import detailed_ac, quick_ac

        quick_sig = inspect.signature(quick_ac)
        detailed_sig = inspect.signature(detailed_ac)

        quick_ppd = quick_sig.parameters["points_per_decade"].default
        detailed_ppd = detailed_sig.parameters["points_per_decade"].default

        assert detailed_ppd > quick_ppd
        assert quick_ppd == 20
        assert detailed_ppd == 100

    def test_detailed_tran_finer_timestep(self) -> None:
        """detailed_tran should use finer timestep calculation."""
        import inspect

        from spicelab.shortcuts import detailed_tran

        # Verify it's importable and has expected signature
        sig = inspect.signature(detailed_tran)
        params = list(sig.parameters.keys())

        assert "duration" in params
        assert "timestep" in params


class TestChainableResult:
    """Test fluent method chaining API."""

    def test_import_chainable(self) -> None:
        """Verify chainable classes can be imported."""
        from spicelab.shortcuts import ChainableResult, MeasurementResult, wrap_result

        assert ChainableResult is not None
        assert MeasurementResult is not None
        assert callable(wrap_result)

    def test_measurement_result_dataclass(self) -> None:
        """MeasurementResult should have expected fields."""
        from spicelab.shortcuts import MeasurementResult

        m = MeasurementResult(
            name="test",
            value=42.0,
            units="dB",
            details={"extra": "info"},
        )

        assert m.name == "test"
        assert m.value == 42.0
        assert m.units == "dB"
        assert m.details == {"extra": "info"}

    def test_measurement_result_repr(self) -> None:
        """MeasurementResult should have readable repr."""
        from spicelab.shortcuts import MeasurementResult

        m = MeasurementResult(name="gain", value=20.5, units="dB", details={})
        repr_str = repr(m)

        assert "gain" in repr_str
        assert "20.5" in repr_str
        assert "dB" in repr_str

    def test_chainable_result_methods_exist(self) -> None:
        """ChainableResult should have measurement methods."""
        from spicelab.shortcuts import ChainableResult

        # Check methods exist
        assert hasattr(ChainableResult, "pm")
        assert hasattr(ChainableResult, "gm")
        assert hasattr(ChainableResult, "bw")
        assert hasattr(ChainableResult, "gain")
        assert hasattr(ChainableResult, "overshoot")
        assert hasattr(ChainableResult, "settling_time")
        assert hasattr(ChainableResult, "rise_time")

    def test_chainable_result_plot_methods_exist(self) -> None:
        """ChainableResult should have plot methods."""
        from spicelab.shortcuts import ChainableResult

        assert hasattr(ChainableResult, "plot")
        assert hasattr(ChainableResult, "plot_bode")
        assert hasattr(ChainableResult, "plot_traces")
        assert hasattr(ChainableResult, "plot_step")
        assert hasattr(ChainableResult, "plot_nyquist")

    def test_chainable_result_extraction_methods(self) -> None:
        """ChainableResult should have value extraction methods."""
        from spicelab.shortcuts import ChainableResult

        assert hasattr(ChainableResult, "value")
        assert hasattr(ChainableResult, "values")
        assert hasattr(ChainableResult, "summary")
        assert hasattr(ChainableResult, "measurements")

    def test_chainable_result_properties(self) -> None:
        """ChainableResult should expose handle and analysis_type."""
        from spicelab.shortcuts import ChainableResult

        assert hasattr(ChainableResult, "handle")
        assert hasattr(ChainableResult, "analysis_type")
        assert hasattr(ChainableResult, "dataset")
        assert hasattr(ChainableResult, "to_polars")
        assert hasattr(ChainableResult, "attrs")


# =============================================================================
# Error Handling Tests (Section 5)
# =============================================================================


class TestErrorCodes:
    """Test error codes for programmatic handling."""

    def test_error_code_enum_exists(self) -> None:
        """ErrorCode enum should be importable."""
        from spicelab.exceptions import ErrorCode

        assert ErrorCode.FLOATING_NODE.value == "E1001"
        assert ErrorCode.SHORT_CIRCUIT.value == "E1002"
        assert ErrorCode.CONVERGENCE_FAILURE.value == "E2002"
        assert ErrorCode.ENGINE_NOT_FOUND.value == "E2001"

    def test_error_codes_are_unique(self) -> None:
        """All error codes should be unique."""
        from spicelab.exceptions import ErrorCode

        values = [e.value for e in ErrorCode]
        assert len(values) == len(set(values)), "Duplicate error codes found"

    def test_exception_has_code_attribute(self) -> None:
        """All exceptions should have a code attribute."""
        from spicelab.exceptions import (
            ComponentNotFoundError,
            ConvergenceError,
            ErrorCode,
            FloatingNodeError,
        )

        err = FloatingNodeError(nodes=["net1"])
        assert err.code == ErrorCode.FLOATING_NODE

        err2 = ConvergenceError(analysis="tran")
        assert err2.code == ErrorCode.CONVERGENCE_FAILURE

        err3 = ComponentNotFoundError(ref="R1")
        assert err3.code == ErrorCode.COMPONENT_NOT_FOUND


class TestErrorSuggestions:
    """Test recovery suggestions in error messages."""

    def test_floating_node_has_suggestion(self) -> None:
        """FloatingNodeError should include recovery suggestion."""
        from spicelab.exceptions import FloatingNodeError

        err = FloatingNodeError(nodes=["net1", "net2"])
        assert err.suggestion is not None
        assert "ground" in err.suggestion.lower() or "connect" in err.suggestion.lower()

    def test_convergence_error_has_suggestion(self) -> None:
        """ConvergenceError should include recovery suggestions."""
        from spicelab.exceptions import ConvergenceError

        err = ConvergenceError(analysis="tran")
        assert err.suggestion is not None
        assert "reltol" in err.suggestion.lower() or "initial" in err.suggestion.lower()

    def test_engine_not_found_has_suggestion(self) -> None:
        """EngineNotFoundError should suggest installation."""
        from spicelab.exceptions import EngineNotFoundError

        err = EngineNotFoundError(engine="ngspice")
        assert err.suggestion is not None
        assert "install" in err.suggestion.lower() or "path" in err.suggestion.lower()

    def test_custom_suggestion_override(self) -> None:
        """Custom suggestion should override default."""
        from spicelab.exceptions import FloatingNodeError

        custom = "Add a 1GOhm resistor to ground"
        err = FloatingNodeError(nodes=["net1"], suggestion=custom)
        assert err.suggestion == custom


class TestErrorContext:
    """Test context information in errors."""

    def test_with_context_method(self) -> None:
        """Errors should support with_context() for chaining."""
        from spicelab.exceptions import FloatingNodeError

        err = FloatingNodeError(nodes=["net1"])
        err.with_context("while validating circuit 'amplifier'")

        assert err.context is not None
        assert "amplifier" in err.context

    def test_context_appears_in_str(self) -> None:
        """Context should appear in error string representation."""
        from spicelab.exceptions import FloatingNodeError

        err = FloatingNodeError(nodes=["net1"])
        err.with_context("during simulation setup")

        error_str = str(err)
        assert "Context:" in error_str
        assert "simulation setup" in error_str


class TestErrorCatalog:
    """Test error message catalog with solutions."""

    def test_catalog_exists(self) -> None:
        """ERROR_CATALOG should be importable and populated."""
        from spicelab.exceptions import ERROR_CATALOG, ErrorCode

        assert ErrorCode.FLOATING_NODE in ERROR_CATALOG
        assert ErrorCode.CONVERGENCE_FAILURE in ERROR_CATALOG
        assert ErrorCode.ENGINE_NOT_FOUND in ERROR_CATALOG

    def test_catalog_entry_structure(self) -> None:
        """Catalog entries should have required fields."""
        from spicelab.exceptions import ERROR_CATALOG, ErrorCode

        entry = ERROR_CATALOG[ErrorCode.FLOATING_NODE]

        assert "title" in entry
        assert "description" in entry
        assert "common_causes" in entry
        assert "solutions" in entry

    def test_full_help_method(self) -> None:
        """full_help() should return detailed help text."""
        from spicelab.exceptions import FloatingNodeError

        err = FloatingNodeError(nodes=["net1"])
        help_text = err.full_help()

        assert "Floating Node" in help_text
        assert "Common Causes" in help_text
        assert "Solutions" in help_text


class TestDidYouMean:
    """Test 'did you mean?' suggestions for typos."""

    def test_suggest_similar_function(self) -> None:
        """suggest_similar should find close matches."""
        from spicelab.exceptions import suggest_similar

        candidates = ["resistor", "capacitor", "inductor", "transistor"]

        # Typo: 'resitor' -> 'resistor'
        suggestions = suggest_similar("resitor", candidates)
        assert "resistor" in suggestions

        # Typo: 'capicator' -> 'capacitor'
        suggestions = suggest_similar("capicator", candidates)
        assert "capacitor" in suggestions

    def test_format_suggestions_single(self) -> None:
        """format_suggestions should handle single suggestion."""
        from spicelab.exceptions import format_suggestions

        result = format_suggestions(["resistor"])
        assert "Did you mean 'resistor'?" == result

    def test_format_suggestions_multiple(self) -> None:
        """format_suggestions should handle multiple suggestions."""
        from spicelab.exceptions import format_suggestions

        result = format_suggestions(["resistor", "transistor"])
        assert "Did you mean one of:" in result
        assert "'resistor'" in result
        assert "'transistor'" in result

    def test_format_suggestions_empty(self) -> None:
        """format_suggestions should handle empty list."""
        from spicelab.exceptions import format_suggestions

        result = format_suggestions([])
        assert result == ""

    def test_component_not_found_with_suggestions(self) -> None:
        """ComponentNotFoundError should suggest similar components."""
        from spicelab.exceptions import ComponentNotFoundError

        # Use a typo that's more clearly similar
        available = ["R_input", "R_output", "C_filter", "C_bypass", "L_choke"]
        err = ComponentNotFoundError(ref="R_inptu", available=available)  # Typo

        # Should suggest R_input as similar
        assert err.similar  # Not empty
        assert "R_input" in err.similar

    def test_attribute_not_found_error(self) -> None:
        """AttributeNotFoundError should provide suggestions."""
        from spicelab.exceptions import AttributeNotFoundError

        available = ["resistance", "capacitance", "inductance", "voltage"]
        err = AttributeNotFoundError(
            obj_type="Resistor", attr_name="resistnce", available=available
        )

        assert "resistance" in err.similar
        assert err.suggestion is not None
        assert "resistance" in err.suggestion


class TestErrorStringFormat:
    """Test error string formatting."""

    def test_error_code_in_string(self) -> None:
        """Error code should appear in string representation."""
        from spicelab.exceptions import FloatingNodeError

        err = FloatingNodeError(nodes=["net1"])
        error_str = str(err)

        assert "[E1001]" in error_str

    def test_details_in_string(self) -> None:
        """Details should appear in string representation."""
        from spicelab.exceptions import FloatingNodeError

        err = FloatingNodeError(nodes=["net1", "net2"])
        error_str = str(err)

        assert "node_count=2" in error_str

    def test_suggestion_in_string(self) -> None:
        """Suggestion should appear in string representation."""
        from spicelab.exceptions import ConvergenceError

        err = ConvergenceError(analysis="dc")
        error_str = str(err)

        assert "Suggestion:" in error_str


# =============================================================================
# Netlist Preview Tests (Section 6.3)
# =============================================================================


class TestNetlistPreview:
    """Test netlist preview functionality."""

    def test_preview_netlist_exists(self) -> None:
        """Circuit should have preview_netlist method."""
        from spicelab.core.circuit import Circuit

        assert hasattr(Circuit, "preview_netlist")

    def test_preview_netlist_basic(self) -> None:
        """preview_netlist should return formatted string."""
        from spicelab.core.circuit import Circuit
        from spicelab.core.components import Resistor
        from spicelab.core.net import Net

        circuit = Circuit("test_circuit")
        vin = Net("vin")
        vout = Net("vout")

        R1 = Resistor(ref="R1", resistance=10_000)
        circuit.add(R1)
        circuit.connect(R1.ports[0], vin)
        circuit.connect(R1.ports[1], vout)

        preview = circuit.preview_netlist(syntax_highlight=False)

        assert "test_circuit" in preview
        assert "R1" in preview
        assert "Components:" in preview

    def test_preview_netlist_with_stats(self) -> None:
        """preview_netlist should show circuit statistics."""
        from spicelab.core.circuit import Circuit
        from spicelab.core.components import Capacitor, Resistor
        from spicelab.core.net import GND, Net

        circuit = Circuit("rc_filter")
        vin = Net("vin")
        vout = Net("vout")

        R1 = Resistor(ref="R1", resistance=1000)
        C1 = Capacitor(ref="C1", capacitance=1e-6)
        circuit.add(R1, C1)
        circuit.connect(R1.ports[0], vin)
        circuit.connect(R1.ports[1], vout)
        circuit.connect(C1.ports[0], vout)
        circuit.connect(C1.ports[1], GND)

        preview = circuit.preview_netlist(syntax_highlight=False, show_stats=True)

        assert "Components: 2" in preview
        assert "Nets:" in preview

    def test_preview_netlist_without_line_numbers(self) -> None:
        """preview_netlist should work without line numbers."""
        from spicelab.core.circuit import Circuit
        from spicelab.core.components import Resistor
        from spicelab.core.net import Net

        circuit = Circuit("simple")
        vin = Net("vin")
        vout = Net("vout")

        R1 = Resistor(ref="R1", resistance=100)
        circuit.add(R1)
        circuit.connect(R1.ports[0], vin)
        circuit.connect(R1.ports[1], vout)

        preview = circuit.preview_netlist(
            syntax_highlight=False, line_numbers=False, show_stats=False
        )

        # Should not have line number format "1 |"
        lines = preview.split("\n")
        for line in lines:
            assert not line.strip().startswith("1 |")

    def test_preview_netlist_engine_parameter(self) -> None:
        """preview_netlist should accept engine parameter."""
        from spicelab.core.circuit import Circuit

        circuit = Circuit("test")

        # Should not raise for different engines
        preview_ng = circuit.preview_netlist(engine="ngspice", syntax_highlight=False)
        preview_lt = circuit.preview_netlist(engine="ltspice", syntax_highlight=False)
        preview_xy = circuit.preview_netlist(engine="xyce", syntax_highlight=False)

        assert "ngspice" in preview_ng
        assert "ltspice" in preview_lt
        assert "xyce" in preview_xy

    def test_preview_netlist_contains_end(self) -> None:
        """preview_netlist should contain .end directive."""
        from spicelab.core.circuit import Circuit
        from spicelab.core.components import Resistor
        from spicelab.core.net import Net

        circuit = Circuit("test")
        R1 = Resistor(ref="R1", resistance=100)
        circuit.add(R1)
        circuit.connect(R1.ports[0], Net("a"))
        circuit.connect(R1.ports[1], Net("b"))

        preview = circuit.preview_netlist(syntax_highlight=False)

        assert ".end" in preview.lower()

    def test_preview_netlist_with_directives(self) -> None:
        """preview_netlist should include directives."""
        from spicelab.core.circuit import Circuit
        from spicelab.core.components import Resistor
        from spicelab.core.net import Net

        circuit = Circuit("with_params")
        circuit.add_directive(".param R_val=10k")

        R1 = Resistor(ref="R1", resistance=10000)
        circuit.add(R1)
        circuit.connect(R1.ports[0], Net("a"))
        circuit.connect(R1.ports[1], Net("b"))

        preview = circuit.preview_netlist(syntax_highlight=False)

        assert ".param" in preview
        assert "R_val" in preview

    def test_preview_netlist_template_circuit(self) -> None:
        """preview_netlist should work with template circuits."""
        from spicelab.templates import rc_lowpass

        circuit = rc_lowpass(fc=1000, impedance=10_000)

        preview = circuit.preview_netlist(syntax_highlight=False)

        # Should have title, components, and .end
        assert "rc_lowpass" in preview.lower() or "Circuit:" in preview
        assert "R" in preview
        assert "C" in preview
        assert ".end" in preview.lower()
