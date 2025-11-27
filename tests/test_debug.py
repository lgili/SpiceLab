"""Tests for debug module (Section 6 - Interactive and Debugging Features).

Tests verbose mode, dry-run validation, simulation debugger, and interactive mode.
"""

import io

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vac
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec
from spicelab.templates import rc_lowpass


def _make_complete_rc_circuit():
    """Create a complete RC circuit with AC source for testing."""
    circuit = Circuit("Test_RC")

    vin = Net("vin")
    vout = Net("vout")

    # AC source
    V1 = Vac(ref="1", value="1", ac_mag="1")
    R1 = Resistor(ref="1", resistance=10_000)
    C1 = Capacitor(ref="1", capacitance=1e-6)

    circuit.add(V1, R1, C1)

    circuit.connect(V1.ports[0], vin)
    circuit.connect(V1.ports[1], GND)
    circuit.connect(R1.ports[0], vin)
    circuit.connect(R1.ports[1], vout)
    circuit.connect(C1.ports[0], vout)
    circuit.connect(C1.ports[1], GND)

    return circuit


# =============================================================================
# Verbose Mode Tests (6.1)
# =============================================================================


class TestVerboseSimulation:
    """Tests for VerboseSimulation context manager."""

    def test_import(self):
        """Should be importable."""
        from spicelab.debug import VerboseSimulation, get_verbose_context, set_verbose

        assert VerboseSimulation is not None
        assert set_verbose is not None
        assert get_verbose_context is not None

    def test_context_manager_enables_verbose(self):
        """Context manager should enable verbose mode."""
        from spicelab.debug import VerboseSimulation, get_verbose_context

        # Initially not verbose
        assert get_verbose_context() is None

        with VerboseSimulation():
            config = get_verbose_context()
            assert config is not None
            assert config.enabled is True

        # After context, should be disabled again
        assert get_verbose_context() is None

    def test_set_verbose_global(self):
        """set_verbose should enable/disable globally."""
        from spicelab.debug import get_verbose_context, set_verbose

        set_verbose(True)
        assert get_verbose_context() is not None
        assert get_verbose_context().enabled is True

        set_verbose(False)
        assert get_verbose_context() is None

    def test_verbose_log_output(self):
        """Verbose log should write to output stream."""
        from spicelab.debug.verbose import VerboseSimulation, verbose_log

        output = io.StringIO()

        with VerboseSimulation(output=output):
            verbose_log("Test message", "info")

        output.seek(0)
        content = output.read()
        assert "Test message" in content
        assert "[verbose]" in content

    def test_verbose_log_levels(self):
        """Different log levels should have different prefixes."""
        from spicelab.debug.verbose import VerboseSimulation, verbose_log

        output = io.StringIO()

        with VerboseSimulation(output=output):
            verbose_log("Step message", "step")
            verbose_log("Success message", "success")
            verbose_log("Warning message", "warning")

        output.seek(0)
        content = output.read()
        assert "→" in content  # step prefix
        assert "✓" in content  # success prefix
        assert "⚠" in content  # warning prefix

    def test_verbose_circuit_info(self):
        """Should log circuit info."""
        from spicelab.debug.verbose import VerboseSimulation, verbose_circuit_info

        circuit = rc_lowpass(fc=1000)
        output = io.StringIO()

        with VerboseSimulation(output=output):
            verbose_circuit_info(circuit)

        output.seek(0)
        content = output.read()
        assert "RC_Filter" in content
        assert "2 components" in content

    def test_verbose_analysis(self):
        """Should log analysis configuration."""
        from spicelab.debug.verbose import VerboseSimulation, verbose_analysis

        analyses = [AnalysisSpec("ac", {"sweep_type": "dec", "n": 20, "fstart": 1, "fstop": 1e6})]
        output = io.StringIO()

        with VerboseSimulation(output=output, show_analysis=True):
            verbose_analysis(analyses)

        output.seek(0)
        content = output.read()
        assert "AC sweep" in content
        assert "dec" in content

    def test_verbose_netlist_off_by_default(self):
        """show_netlist should be off by default."""
        from spicelab.debug.verbose import VerboseSimulation, verbose_netlist

        output = io.StringIO()

        with VerboseSimulation(output=output, show_netlist=False):
            verbose_netlist("* Test netlist\nR1 a b 10k")

        output.seek(0)
        content = output.read()
        # Should not contain netlist
        assert "R1 a b 10k" not in content

    def test_verbose_netlist_when_enabled(self):
        """show_netlist=True should print netlist."""
        from spicelab.debug.verbose import VerboseSimulation, verbose_netlist

        output = io.StringIO()

        with VerboseSimulation(output=output, show_netlist=True):
            verbose_netlist("* Test netlist\nR1 a b 10k")

        output.seek(0)
        content = output.read()
        assert "R1 a b 10k" in content

    def test_verbose_section_timing(self):
        """verbose_section should time code blocks."""
        import time

        from spicelab.debug.verbose import VerboseSimulation, verbose_section

        output = io.StringIO()

        with VerboseSimulation(output=output, show_timing=True):
            with verbose_section("Test operation"):
                time.sleep(0.01)  # Small delay

        output.seek(0)
        content = output.read()
        assert "Test operation" in content
        assert "completed" in content


# =============================================================================
# Dry-Run Tests (6.2)
# =============================================================================


class TestDryRun:
    """Tests for dry_run validation."""

    def test_import(self):
        """Should be importable."""
        from spicelab.debug import DryRunResult, dry_run

        assert dry_run is not None
        assert DryRunResult is not None

    def test_valid_circuit_passes(self):
        """Valid circuit should pass dry-run (without deep validation)."""
        from spicelab.debug import dry_run

        circuit = rc_lowpass(fc=1000)
        analyses = [AnalysisSpec("ac", {"sweep_type": "dec", "n": 20, "fstart": 1, "fstop": 1e6})]

        # Skip circuit validation (which requires complete circuits with sources)
        result = dry_run(circuit, analyses, validate_circuit=False)

        assert result.valid is True
        assert result.circuit_name == "RC_Filter"
        assert result.component_count == 2
        assert len(result.errors) == 0

    def test_returns_netlist(self):
        """Dry-run should return generated netlist."""
        from spicelab.debug import dry_run

        circuit = rc_lowpass(fc=1000)
        result = dry_run(circuit)

        assert result.netlist != ""
        assert "R1" in result.netlist
        assert "C1" in result.netlist

    def test_returns_analyses(self):
        """Dry-run should return analysis configuration."""
        from spicelab.debug import dry_run

        circuit = rc_lowpass(fc=1000)
        analyses = [AnalysisSpec("ac", {"sweep_type": "dec", "n": 20, "fstart": 1, "fstop": 1e6})]

        result = dry_run(circuit, analyses)

        assert len(result.analyses) == 1
        assert result.analyses[0]["mode"] == "ac"
        assert result.analyses[0]["fstart"] == 1

    def test_warns_no_analyses(self):
        """Should warn when no analyses specified."""
        from spicelab.debug import dry_run

        circuit = rc_lowpass(fc=1000)
        result = dry_run(circuit, analyses=None)

        assert len(result.warnings) >= 1
        warning_msgs = [w.message for w in result.warnings]
        assert any("No analyses" in msg for msg in warning_msgs)

    def test_analysis_modes_validated_by_pydantic(self):
        """Invalid analysis modes are caught by pydantic at creation time."""
        # AnalysisSpec uses pydantic validation, so invalid modes raise
        # ValidationError at creation time, not during dry_run
        with pytest.raises((ValueError, TypeError)):  # pydantic.ValidationError
            AnalysisSpec("invalid_mode", {})

    def test_ac_missing_params(self):
        """AC analysis should require fstart/fstop."""
        from spicelab.debug import dry_run

        circuit = rc_lowpass(fc=1000)
        analyses = [AnalysisSpec("ac", {})]  # Missing fstart, fstop

        result = dry_run(circuit, analyses)

        assert len(result.errors) >= 1

    def test_tran_missing_params(self):
        """Transient analysis should require tstop."""
        from spicelab.debug import dry_run

        circuit = rc_lowpass(fc=1000)
        analyses = [AnalysisSpec("tran", {})]  # Missing tstop

        result = dry_run(circuit, analyses)

        assert len(result.errors) >= 1

    def test_str_representation(self):
        """Should have readable string representation."""
        from spicelab.debug import dry_run

        circuit = rc_lowpass(fc=1000)
        analyses = [AnalysisSpec("ac", {"sweep_type": "dec", "n": 20, "fstart": 1, "fstop": 1e6})]

        # Skip circuit validation for simpler test
        result = dry_run(circuit, analyses, validate_circuit=False)
        result_str = str(result)

        assert "Dry-Run Validation" in result_str
        assert "PASSED" in result_str
        assert "RC_Filter" in result_str

    def test_raise_if_invalid(self):
        """raise_if_invalid should raise on errors."""
        from spicelab.debug import dry_run

        circuit = rc_lowpass(fc=1000)
        # AC analysis with missing parameters
        analyses = [AnalysisSpec("ac", {})]

        result = dry_run(circuit, analyses, validate_circuit=False)

        with pytest.raises(ValueError, match="validation failed"):
            result.raise_if_invalid()

    def test_estimated_runtime(self):
        """Should estimate runtime for complex analyses."""
        from spicelab.debug import dry_run

        circuit = rc_lowpass(fc=1000)
        analyses = [AnalysisSpec("ac", {"sweep_type": "dec", "n": 100, "fstart": 1, "fstop": 1e9})]

        result = dry_run(circuit, analyses)

        # Should have some runtime estimate
        assert result.estimated_runtime is not None or result.estimated_runtime == "< 0.1s (fast)"

    def test_invalid_engine(self):
        """Should warn on invalid engine."""
        from spicelab.debug import dry_run

        circuit = rc_lowpass(fc=1000)
        result = dry_run(circuit, engine="nonexistent_engine", validate_circuit=False)

        assert len(result.errors) >= 1
        error_msgs = [e.message for e in result.errors]
        assert any("Unknown engine" in msg for msg in error_msgs)


# =============================================================================
# Simulation Debugger Tests (6.5)
# =============================================================================


class TestSimulationDebugger:
    """Tests for SimulationDebugger."""

    def test_import(self):
        """Should be importable."""
        from spicelab.debug import DebugStep, SimulationDebugger

        assert SimulationDebugger is not None
        assert DebugStep is not None

    def test_initialization(self):
        """Debugger should initialize at INIT phase."""
        from spicelab.debug import SimulationDebugger
        from spicelab.debug.debugger import DebugPhase

        circuit = _make_complete_rc_circuit()
        analyses = [AnalysisSpec("ac", {"sweep_type": "dec", "n": 20, "fstart": 1, "fstop": 1e6})]

        debugger = SimulationDebugger(circuit, analyses)

        assert debugger.current_phase() == DebugPhase.INIT
        assert not debugger.is_complete()
        assert not debugger.has_error()

    def test_current_step(self):
        """Should return current step info."""
        from spicelab.debug import SimulationDebugger

        circuit = _make_complete_rc_circuit()
        analyses = [AnalysisSpec("ac", {"sweep_type": "dec", "n": 20, "fstart": 1, "fstop": 1e6})]

        debugger = SimulationDebugger(circuit, analyses)
        step = debugger.current_step()

        assert step.description == "Initialize simulation context"
        assert not step.completed

    def test_step_advances(self):
        """step() should advance to next phase."""
        from spicelab.debug import SimulationDebugger
        from spicelab.debug.debugger import DebugPhase

        circuit = _make_complete_rc_circuit()
        analyses = [AnalysisSpec("ac", {"sweep_type": "dec", "n": 20, "fstart": 1, "fstop": 1e6})]

        debugger = SimulationDebugger(circuit, analyses)

        # Initial phase
        assert debugger.current_phase() == DebugPhase.INIT

        # Step forward
        debugger.step()
        assert debugger.current_phase() == DebugPhase.VALIDATE

        debugger.step()
        assert debugger.current_phase() == DebugPhase.NETLIST

    def test_inspect_returns_data(self):
        """inspect() should return phase-specific data."""
        from spicelab.debug import SimulationDebugger

        circuit = _make_complete_rc_circuit()
        analyses = [AnalysisSpec("ac", {"sweep_type": "dec", "n": 20, "fstart": 1, "fstop": 1e6})]

        debugger = SimulationDebugger(circuit, analyses)
        data = debugger.inspect()

        assert "phase" in data
        assert "circuit_name" in data
        assert data["circuit_name"] == "Test_RC"
        assert data["component_count"] == 3  # V1, R1, C1

    def test_get_all_steps(self):
        """Should return all steps."""
        from spicelab.debug import SimulationDebugger

        circuit = _make_complete_rc_circuit()
        analyses = [AnalysisSpec("op", {})]

        debugger = SimulationDebugger(circuit, analyses)
        steps = debugger.get_all_steps()

        assert len(steps) == 8  # INIT through COMPLETE
        phases = [s.phase.name for s in steps]
        assert "INIT" in phases
        assert "COMPLETE" in phases

    def test_str_representation(self):
        """Should have readable string representation."""
        from spicelab.debug import SimulationDebugger

        circuit = _make_complete_rc_circuit()
        analyses = [AnalysisSpec("op", {})]

        debugger = SimulationDebugger(circuit, analyses)
        debugger_str = str(debugger)

        assert "Debugger Status" in debugger_str
        assert "INIT" in debugger_str

    def test_on_step_callback(self):
        """Should call on_step callback."""
        from spicelab.debug import SimulationDebugger

        circuit = _make_complete_rc_circuit()
        analyses = [AnalysisSpec("op", {})]

        steps_seen: list[str] = []

        def on_step(step):
            steps_seen.append(step.phase.name)

        debugger = SimulationDebugger(circuit, analyses, on_step=on_step)

        # Step a few times
        debugger.step()  # INIT -> VALIDATE
        debugger.step()  # VALIDATE -> NETLIST

        assert "INIT" in steps_seen
        assert "VALIDATE" in steps_seen

    def test_run_to_phase(self):
        """run_to_phase should stop at target phase."""
        from spicelab.debug import SimulationDebugger
        from spicelab.debug.debugger import DebugPhase

        circuit = _make_complete_rc_circuit()
        analyses = [AnalysisSpec("op", {})]

        debugger = SimulationDebugger(circuit, analyses)
        debugger.run_to_phase(DebugPhase.NETLIST)

        assert debugger.current_phase() == DebugPhase.NETLIST

    def test_get_netlist_after_phase(self):
        """Should get netlist after NETLIST phase."""
        from spicelab.debug import SimulationDebugger
        from spicelab.debug.debugger import DebugPhase

        circuit = _make_complete_rc_circuit()
        analyses = [AnalysisSpec("op", {})]

        debugger = SimulationDebugger(circuit, analyses)
        debugger.run_to_phase(DebugPhase.ANALYSIS)

        netlist = debugger.get_netlist()
        assert "R1" in netlist
        assert "C1" in netlist

    def test_get_netlist_before_phase_raises(self):
        """get_netlist before NETLIST phase should raise."""
        from spicelab.debug import SimulationDebugger

        circuit = _make_complete_rc_circuit()
        analyses = [AnalysisSpec("op", {})]

        debugger = SimulationDebugger(circuit, analyses)

        with pytest.raises(RuntimeError, match="not yet generated"):
            debugger.get_netlist()


# =============================================================================
# Integration Tests
# =============================================================================


class TestDebugModuleIntegration:
    """Integration tests for debug module."""

    def test_verbose_with_dry_run(self):
        """Verbose mode should work with dry-run."""
        from spicelab.debug import VerboseSimulation, dry_run

        circuit = _make_complete_rc_circuit()
        analyses = [AnalysisSpec("ac", {"sweep_type": "dec", "n": 20, "fstart": 1, "fstop": 1e6})]

        output = io.StringIO()

        with VerboseSimulation(output=output):
            result = dry_run(circuit, analyses)

        assert result.valid is True

    def test_all_exports(self):
        """All public APIs should be exported."""
        from spicelab.debug import (
            DebugStep,
            DryRunResult,
            InteractiveMode,
            InteractiveSession,
            SimulationDebugger,
            VerboseSimulation,
            dry_run,
            get_verbose_context,
            prompt_choice,
            prompt_confirm,
            set_verbose,
        )

        assert VerboseSimulation is not None
        assert set_verbose is not None
        assert get_verbose_context is not None
        assert dry_run is not None
        assert DryRunResult is not None
        assert SimulationDebugger is not None
        assert DebugStep is not None
        assert InteractiveMode is not None
        assert InteractiveSession is not None
        assert prompt_choice is not None
        assert prompt_confirm is not None


# =============================================================================
# Interactive Mode Tests (6.4)
# =============================================================================


class TestInteractiveMode:
    """Tests for interactive mode."""

    def test_import(self):
        """Should be importable."""
        from spicelab.debug import (
            Choice,
            InteractiveMode,
            InteractivePrompt,
            InteractiveSession,
            get_interactive_context,
            prompt_choice,
            prompt_confirm,
            prompt_value,
            set_interactive_mode,
        )

        assert InteractiveMode is not None
        assert InteractiveSession is not None
        assert InteractivePrompt is not None
        assert Choice is not None
        assert set_interactive_mode is not None
        assert get_interactive_context is not None
        assert prompt_choice is not None
        assert prompt_confirm is not None
        assert prompt_value is not None

    def test_context_manager_enables_interactive(self):
        """Context manager should enable interactive mode."""
        from spicelab.debug import InteractiveSession, get_interactive_context

        # Initially not interactive
        assert get_interactive_context() is None

        with InteractiveSession():
            config = get_interactive_context()
            assert config is not None

        # After context, should be disabled again
        assert get_interactive_context() is None

    def test_set_interactive_mode_global(self):
        """set_interactive_mode should enable/disable globally."""
        from spicelab.debug import (
            InteractiveMode,
            get_interactive_context,
            set_interactive_mode,
        )

        set_interactive_mode(InteractiveMode.ALWAYS)
        assert get_interactive_context() is not None

        set_interactive_mode(InteractiveMode.NEVER)
        assert get_interactive_context() is None

    def test_set_interactive_mode_string(self):
        """set_interactive_mode should accept string mode."""
        from spicelab.debug import get_interactive_context, set_interactive_mode

        set_interactive_mode("always")
        assert get_interactive_context() is not None

        set_interactive_mode("never")
        assert get_interactive_context() is None

    def test_choice_dataclass(self):
        """Choice should be a proper dataclass."""
        from spicelab.debug import Choice

        choice = Choice(value="test", label="Test", description="A test choice")
        assert choice.value == "test"
        assert choice.label == "Test"
        assert choice.description == "A test choice"
        assert not choice.is_default

        # String representation
        choice_str = str(choice)
        assert "Test" in choice_str
        assert "A test choice" in choice_str

    def test_choice_default_marker(self):
        """Default choice should show marker in string."""
        from spicelab.debug import Choice

        choice = Choice(value="test", label="Test", is_default=True)
        choice_str = str(choice)
        assert "(default)" in choice_str

    def test_prompt_with_callback(self):
        """Callback should be called for choices."""
        from spicelab.debug import InteractiveSession, prompt_choice

        selected = []

        def callback(prompt):
            selected.append(prompt.question)
            return prompt.choices[1].value  # Pick second option

        with InteractiveSession(callback=callback):
            result = prompt_choice(
                "Pick one",
                [("a", "Option A"), ("b", "Option B")],
                default="a",
            )

        assert result == "b"
        assert "Pick one" in selected

    def test_prompt_choice_outside_context_returns_default(self):
        """prompt_choice outside context should return default."""
        from spicelab.debug import prompt_choice

        result = prompt_choice(
            "Pick one",
            [("a", "Option A"), ("b", "Option B")],
            default="a",
        )

        assert result == "a"

    def test_prompt_choice_outside_context_returns_first_if_no_default(self):
        """prompt_choice outside context should return first if no default."""
        from spicelab.debug import prompt_choice

        result = prompt_choice(
            "Pick one",
            [("a", "Option A"), ("b", "Option B")],
        )

        assert result == "a"

    def test_prompt_confirm_callback(self):
        """prompt_confirm should work with callback."""
        from spicelab.debug import InteractiveSession, prompt_confirm

        def always_yes(prompt):
            return True

        with InteractiveSession(callback=always_yes):
            result = prompt_confirm("Continue?")

        assert result is True

    def test_prompt_confirm_default(self):
        """prompt_confirm should return default outside context."""
        from spicelab.debug import prompt_confirm

        result = prompt_confirm("Continue?", default=True)
        assert result is True

        result = prompt_confirm("Continue?", default=False)
        assert result is False

    def test_prompt_value_with_callback(self):
        """prompt_value should work with callback."""
        from spicelab.debug import InteractiveSession, prompt_value

        def return_1000(prompt):
            return 1000.0

        with InteractiveSession(callback=return_1000):
            result = prompt_value(
                "Enter frequency:",
                default=100,
                suggestions=[100, 1000, 10000],
            )

        assert result == 1000.0

    def test_prompt_value_outside_context_returns_default(self):
        """prompt_value outside context should return default."""
        from spicelab.debug import prompt_value

        result = prompt_value("Enter value:", default=42)
        assert result == 42

    def test_interactive_prompt_get_default(self):
        """InteractivePrompt should return default choice."""
        from spicelab.debug import Choice, InteractivePrompt

        choices = [
            Choice(value="a", label="A"),
            Choice(value="b", label="B", is_default=True),
            Choice(value="c", label="C"),
        ]
        prompt = InteractivePrompt(question="Pick one", choices=choices)

        default = prompt.get_default()
        assert default is not None
        assert default.value == "b"

    def test_interactive_prompt_no_default(self):
        """InteractivePrompt with no default should return first."""
        from spicelab.debug import Choice, InteractivePrompt

        choices = [
            Choice(value="a", label="A"),
            Choice(value="b", label="B"),
        ]
        prompt = InteractivePrompt(question="Pick one", choices=choices)

        default = prompt.get_default()
        assert default is not None
        assert default.value == "a"

    def test_never_mode_uses_defaults(self):
        """NEVER mode should use defaults without prompting."""
        from spicelab.debug import InteractiveMode, InteractiveSession, prompt_choice

        with InteractiveSession(mode=InteractiveMode.NEVER):
            result = prompt_choice(
                "Pick one",
                [("a", "Option A"), ("b", "Option B")],
                default="b",
            )

        assert result == "b"

    def test_prompt_analysis_type(self):
        """prompt_analysis_type should work."""
        from spicelab.debug import InteractiveSession, prompt_analysis_type

        def pick_ac(prompt):
            for choice in prompt.choices:
                if choice.value == "ac":
                    return choice.value
            return prompt.choices[0].value

        with InteractiveSession(callback=pick_ac):
            result = prompt_analysis_type()

        assert result == "ac"

    def test_prompt_frequency_range(self):
        """prompt_frequency_range should work."""
        from spicelab.debug import InteractiveSession, prompt_frequency_range

        def pick_first(prompt):
            return prompt.choices[0].value

        with InteractiveSession(callback=pick_first):
            fstart, fstop = prompt_frequency_range(suggested_fc=1000)

        assert fstart < fstop

    def test_prompt_simulation_engine(self):
        """prompt_simulation_engine should work."""
        from spicelab.debug import InteractiveSession, prompt_simulation_engine

        def pick_ngspice(prompt):
            for choice in prompt.choices:
                if choice.value == "ngspice":
                    return choice.value
            return prompt.choices[0].value

        with InteractiveSession(callback=pick_ngspice):
            result = prompt_simulation_engine()

        assert result == "ngspice"

    def test_nested_contexts(self):
        """Nested contexts should restore previous state."""
        from spicelab.debug import (
            InteractiveSession,
            get_interactive_context,
        )

        def outer_callback(prompt):
            return "outer"

        def inner_callback(prompt):
            return "inner"

        with InteractiveSession(callback=outer_callback):
            outer_config = get_interactive_context()
            assert outer_config.callback is outer_callback

            with InteractiveSession(callback=inner_callback):
                inner_config = get_interactive_context()
                assert inner_config.callback is inner_callback

            # Should restore outer
            restored_config = get_interactive_context()
            assert restored_config.callback is outer_callback

        # Should be None
        assert get_interactive_context() is None


class TestInteractiveIO:
    """Tests for interactive I/O functionality."""

    def test_prompt_writes_to_output(self):
        """Prompts should write to specified output stream."""
        from spicelab.debug import InteractiveMode, InteractiveSession, prompt_choice

        input_stream = io.StringIO("1\n")  # Select first option
        output_stream = io.StringIO()

        with InteractiveSession(
            mode=InteractiveMode.ALWAYS,
            input_stream=input_stream,
            output_stream=output_stream,
        ):
            result = prompt_choice(
                "Pick one",
                [("a", "Option A"), ("b", "Option B")],
                default="a",
            )

        output_stream.seek(0)
        output = output_stream.read()

        assert "Pick one" in output
        assert "Option A" in output
        assert "Option B" in output
        assert result == "a"

    def test_prompt_reads_from_input(self):
        """Prompts should read from specified input stream."""
        from spicelab.debug import InteractiveMode, InteractiveSession, prompt_choice

        input_stream = io.StringIO("2\n")  # Select second option
        output_stream = io.StringIO()

        with InteractiveSession(
            mode=InteractiveMode.ALWAYS,
            input_stream=input_stream,
            output_stream=output_stream,
        ):
            result = prompt_choice(
                "Pick one",
                [("a", "Option A"), ("b", "Option B")],
                default="a",
            )

        assert result == "b"

    def test_empty_input_uses_default(self):
        """Empty input should use default."""
        from spicelab.debug import InteractiveMode, InteractiveSession, prompt_choice

        input_stream = io.StringIO("\n")  # Just press enter
        output_stream = io.StringIO()

        with InteractiveSession(
            mode=InteractiveMode.ALWAYS,
            input_stream=input_stream,
            output_stream=output_stream,
        ):
            result = prompt_choice(
                "Pick one",
                [("a", "Option A"), ("b", "Option B")],
                default="b",
            )

        assert result == "b"

    def test_invalid_number_uses_default(self):
        """Invalid number input should fall back to default."""
        from spicelab.debug import InteractiveMode, InteractiveSession, prompt_choice

        input_stream = io.StringIO("99\n")  # Invalid choice number
        output_stream = io.StringIO()

        with InteractiveSession(
            mode=InteractiveMode.ALWAYS,
            input_stream=input_stream,
            output_stream=output_stream,
        ):
            result = prompt_choice(
                "Pick one",
                [("a", "Option A"), ("b", "Option B")],
                default="b",
            )

        assert result == "b"

    def test_label_matching(self):
        """Should match by label (case-insensitive)."""
        from spicelab.debug import InteractiveMode, InteractiveSession, prompt_choice

        input_stream = io.StringIO("option b\n")  # Type label
        output_stream = io.StringIO()

        with InteractiveSession(
            mode=InteractiveMode.ALWAYS,
            input_stream=input_stream,
            output_stream=output_stream,
        ):
            result = prompt_choice(
                "Pick one",
                [("a", "Option A"), ("b", "Option B")],
                default="a",
            )

        assert result == "b"
