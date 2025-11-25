"""Tests for troubleshooting module (M4 DX improvement).

Tests the diagnostic and interactive troubleshooting features.
"""

from io import StringIO

from rich.console import Console
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.troubleshooting import (
    DiagnosticResult,
    Troubleshooter,
    diagnose_circuit,
    diagnose_convergence,
    diagnose_empty_results,
)
from spicelab.troubleshooting.diagnostics import DiagnosticSeverity, Finding


class TestDiagnosticResult:
    """Tests for DiagnosticResult dataclass."""

    def test_empty_result(self):
        """Empty result should have no issues."""
        result = DiagnosticResult()
        assert not result.has_issues
        assert not result.has_errors
        assert result.error_count == 0
        assert result.warning_count == 0
        assert result.summary() == "No issues found"

    def test_result_with_errors(self):
        """Result with errors should report correctly."""
        result = DiagnosticResult()
        result.add(
            Finding(
                category="test",
                severity=DiagnosticSeverity.ERROR,
                message="Test error",
                suggestion="Fix it",
            )
        )
        assert result.has_issues
        assert result.has_errors
        assert result.error_count == 1
        assert "1 error" in result.summary()

    def test_result_with_warnings(self):
        """Result with warnings should report correctly."""
        result = DiagnosticResult()
        result.add(
            Finding(
                category="test",
                severity=DiagnosticSeverity.WARNING,
                message="Test warning",
                suggestion="Check it",
            )
        )
        assert result.has_issues
        assert not result.has_errors
        assert result.warning_count == 1
        assert "1 warning" in result.summary()

    def test_by_category(self):
        """Should filter findings by category."""
        result = DiagnosticResult()
        result.add(
            Finding(
                category="topology",
                severity=DiagnosticSeverity.ERROR,
                message="Topology error",
                suggestion="Fix",
            )
        )
        result.add(
            Finding(
                category="convergence",
                severity=DiagnosticSeverity.WARNING,
                message="Convergence warning",
                suggestion="Check",
            )
        )

        topology_findings = result.by_category("topology")
        assert len(topology_findings) == 1
        assert topology_findings[0].message == "Topology error"


class TestFinding:
    """Tests for Finding dataclass."""

    def test_str_representation(self):
        """Finding should have readable string representation."""
        finding = Finding(
            category="test",
            severity=DiagnosticSeverity.ERROR,
            message="Test message",
            suggestion="Fix it",
        )
        result = str(finding)
        assert "[test]" in result
        assert "Test message" in result


class TestDiagnoseCircuit:
    """Tests for diagnose_circuit function."""

    def test_valid_circuit_no_errors(self):
        """Valid circuit should have no errors."""
        c = Circuit("valid")
        v = Vdc("1", 5.0)
        r = Resistor("1", resistance=1000)
        c.add(v, r)

        n1 = Net("vcc")
        c.connect(v.ports[0], n1)
        c.connect(r.ports[0], n1)
        c.connect(v.ports[1], GND)
        c.connect(r.ports[1], GND)

        result = diagnose_circuit(c)
        assert not result.has_errors

    def test_floating_node_detected(self):
        """Should detect floating nodes."""
        c = Circuit("floating")
        r = Resistor("1", resistance=1000)
        c.add(r)
        c.connect(r.ports[0], Net("a"))
        c.connect(r.ports[1], Net("b"))

        result = diagnose_circuit(c)
        assert result.has_errors
        floating_errors = [f for f in result.findings if "floating" in f.message.lower()]
        assert len(floating_errors) >= 1

    def test_no_ground_detected(self):
        """Should detect missing ground."""
        c = Circuit("no_gnd")
        r = Resistor("1", resistance=1000)
        c.add(r)
        c.connect(r.ports[0], Net("a"))
        c.connect(r.ports[1], Net("b"))

        result = diagnose_circuit(c)
        gnd_errors = [f for f in result.findings if "ground" in f.message.lower()]
        assert len(gnd_errors) >= 1

    def test_small_resistance_warning(self):
        """Should warn about very small resistance."""
        c = Circuit("small_r")
        v = Vdc("1", 5.0)
        r = Resistor("1", resistance=0.0001)  # 0.1mΩ
        c.add(v, r)

        n1 = Net("vcc")
        c.connect(v.ports[0], n1)
        c.connect(r.ports[0], n1)
        c.connect(v.ports[1], GND)
        c.connect(r.ports[1], GND)

        result = diagnose_circuit(c)
        small_r_warnings = [f for f in result.findings if "small resistance" in f.message.lower()]
        assert len(small_r_warnings) >= 1

    def test_large_resistance_warning(self):
        """Should warn about very large resistance."""
        c = Circuit("large_r")
        v = Vdc("1", 5.0)
        r = Resistor("1", resistance=1e15)  # 1PΩ
        c.add(v, r)

        n1 = Net("vcc")
        c.connect(v.ports[0], n1)
        c.connect(r.ports[0], n1)
        c.connect(v.ports[1], GND)
        c.connect(r.ports[1], GND)

        result = diagnose_circuit(c)
        large_r_warnings = [f for f in result.findings if "large resistance" in f.message.lower()]
        assert len(large_r_warnings) >= 1

    def test_small_capacitance_warning(self):
        """Should warn about very small capacitance."""
        c = Circuit("small_c")
        v = Vdc("1", 5.0)
        cap = Capacitor("1", capacitance=1e-18)  # 1aF
        c.add(v, cap)

        n1 = Net("vcc")
        c.connect(v.ports[0], n1)
        c.connect(cap.ports[0], n1)
        c.connect(v.ports[1], GND)
        c.connect(cap.ports[1], GND)

        result = diagnose_circuit(c)
        small_c_warnings = [f for f in result.findings if "small capacitance" in f.message.lower()]
        assert len(small_c_warnings) >= 1

    def test_empty_circuit_error(self):
        """Should error on empty circuit."""
        c = Circuit("empty")
        result = diagnose_circuit(c)
        empty_errors = [f for f in result.findings if "no components" in f.message.lower()]
        assert len(empty_errors) >= 1


class TestDiagnoseConvergence:
    """Tests for diagnose_convergence function."""

    def test_timestep_error_detection(self):
        """Should detect timestep too small error."""
        c = Circuit("test")
        v = Vdc("1", 5.0)
        r = Resistor("1", resistance=1000)
        c.add(v, r)
        c.connect(v.ports[0], Net("vcc"))
        c.connect(r.ports[0], Net("vcc"))
        c.connect(v.ports[1], GND)
        c.connect(r.ports[1], GND)

        result = diagnose_convergence(c, "timestep too small")
        timestep_findings = [f for f in result.findings if "timestep" in f.message.lower()]
        assert len(timestep_findings) >= 1

    def test_singular_matrix_detection(self):
        """Should detect singular matrix error."""
        c = Circuit("test")
        v = Vdc("1", 5.0)
        c.add(v)
        c.connect(v.ports[0], Net("vcc"))
        c.connect(v.ports[1], GND)

        result = diagnose_convergence(c, "singular matrix encountered")
        singular_findings = [f for f in result.findings if "singular" in f.message.lower()]
        assert len(singular_findings) >= 1

    def test_no_convergence_detection(self):
        """Should detect general no convergence error."""
        c = Circuit("test")
        v = Vdc("1", 5.0)
        c.add(v)
        c.connect(v.ports[0], Net("vcc"))
        c.connect(v.ports[1], GND)

        result = diagnose_convergence(c, "no convergence in DC operating point")
        conv_findings = [f for f in result.findings if "converge" in f.message.lower()]
        assert len(conv_findings) >= 1


class TestDiagnoseEmptyResults:
    """Tests for diagnose_empty_results function."""

    def test_no_probes_error(self):
        """Should error when no probes specified."""
        c = Circuit("test")
        v = Vdc("1", 5.0)
        c.add(v)
        c.connect(v.ports[0], Net("vcc"))
        c.connect(v.ports[1], GND)

        result = diagnose_empty_results(c, probes=[])
        probe_errors = [f for f in result.findings if "no probes" in f.message.lower()]
        assert len(probe_errors) >= 1

    def test_dc_analysis_info(self):
        """Should provide info about DC analysis."""
        c = Circuit("test")
        v = Vdc("1", 5.0)
        c.add(v)
        c.connect(v.ports[0], Net("vcc"))
        c.connect(v.ports[1], GND)

        result = diagnose_empty_results(c, analysis_type="dc")
        dc_info = [f for f in result.findings if "dc" in f.message.lower()]
        assert len(dc_info) >= 1

    def test_ac_analysis_info(self):
        """Should provide info about AC analysis."""
        c = Circuit("test")
        v = Vdc("1", 5.0)
        c.add(v)
        c.connect(v.ports[0], Net("vcc"))
        c.connect(v.ports[1], GND)

        result = diagnose_empty_results(c, analysis_type="ac")
        ac_info = [f for f in result.findings if "ac" in f.message.lower()]
        assert len(ac_info) >= 1


class TestTroubleshooter:
    """Tests for Troubleshooter class."""

    def test_diagnose_returns_result(self):
        """diagnose() should return DiagnosticResult."""
        c = Circuit("test")
        v = Vdc("1", 5.0)
        r = Resistor("1", resistance=1000)
        c.add(v, r)

        n1 = Net("vcc")
        c.connect(v.ports[0], n1)
        c.connect(r.ports[0], n1)
        c.connect(v.ports[1], GND)
        c.connect(r.ports[1], GND)

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        ts = Troubleshooter(c, console)

        result = ts.diagnose(verbose=False)
        assert isinstance(result, DiagnosticResult)

    def test_diagnose_convergence(self):
        """diagnose_convergence() should return DiagnosticResult."""
        c = Circuit("test")
        v = Vdc("1", 5.0)
        c.add(v)
        c.connect(v.ports[0], Net("vcc"))
        c.connect(v.ports[1], GND)

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        ts = Troubleshooter(c, console)

        result = ts.diagnose_convergence(verbose=False)
        assert isinstance(result, DiagnosticResult)

    def test_quick_fix_returns_list(self):
        """quick_fix() should return list of suggestions."""
        c = Circuit("test")
        r = Resistor("1", resistance=1000)
        c.add(r)
        c.connect(r.ports[0], Net("a"))
        c.connect(r.ports[1], Net("b"))

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        ts = Troubleshooter(c, console)

        fixes = ts.quick_fix()
        assert isinstance(fixes, list)
        assert len(fixes) > 0  # Should find issues

    def test_verbose_output(self):
        """Verbose diagnose should produce output."""
        c = Circuit("test")
        r = Resistor("1", resistance=1000)
        c.add(r)
        c.connect(r.ports[0], Net("a"))
        c.connect(r.ports[1], Net("b"))

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)
        ts = Troubleshooter(c, console)

        ts.diagnose(verbose=True)
        result = output.getvalue()
        assert "Diagnostics" in result or "error" in result.lower()
