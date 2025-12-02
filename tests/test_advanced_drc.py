"""Tests for Advanced DRC (Design Rule Checks)."""

import json
import tempfile
from pathlib import Path

import pytest

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.validators import (
    AdvancedDRC,
    ConstraintTemplate,
    DRCContext,
    DRCReport,
    DRCRule,
    Severity,
    run_drc,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_circuit():
    """Create a simple valid circuit."""
    circuit = Circuit("test_circuit")

    v1 = Vdc("V1", 5.0)
    r1 = Resistor("R1", "1k")
    r2 = Resistor("R2", "2k")

    circuit.add(v1, r1, r2)

    circuit.connect(v1.ports[0], Net("vcc"))
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], Net("vcc"))
    circuit.connect(r1.ports[1], Net("out"))
    circuit.connect(r2.ports[0], Net("out"))
    circuit.connect(r2.ports[1], GND)

    return circuit


@pytest.fixture
def low_resistance_circuit():
    """Create circuit with very low resistance."""
    circuit = Circuit("low_r_circuit")

    v1 = Vdc("V1", 12.0)
    r1 = Resistor("R1", 0.1)  # Very low - 0.1 Ohm

    circuit.add(v1, r1)

    circuit.connect(v1.ports[0], Net("in"))
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], Net("in"))
    circuit.connect(r1.ports[1], GND)

    return circuit


@pytest.fixture
def high_power_circuit():
    """Create circuit with high power dissipation."""
    circuit = Circuit("high_power_circuit")

    v1 = Vdc("V1", 48.0)  # High voltage
    r1 = Resistor("R1", 10)  # Will dissipate P = 48²/10 = 230.4W

    circuit.add(v1, r1)

    circuit.connect(v1.ports[0], Net("in"))
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], Net("in"))
    circuit.connect(r1.ports[1], GND)

    return circuit


@pytest.fixture
def high_fanout_circuit():
    """Create circuit with high fanout."""
    circuit = Circuit("high_fanout_circuit")

    v1 = Vdc("V1", 5.0)
    circuit.add(v1)
    circuit.connect(v1.ports[0], Net("bus"))
    circuit.connect(v1.ports[1], GND)

    # Add many resistors to same net (high fanout)
    for i in range(15):
        r = Resistor(f"R{i}", "10k")
        circuit.add(r)
        circuit.connect(r.ports[0], Net("bus"))
        circuit.connect(r.ports[1], GND)

    return circuit


# ============================================================================
# DRCContext Tests
# ============================================================================


class TestDRCContext:
    """Tests for DRCContext dataclass."""

    def test_default_values(self):
        """Test default context values."""
        ctx = DRCContext()

        assert ctx.max_current_ma == 100.0
        assert ctx.max_power_mw == 250.0
        assert ctx.max_voltage == 50.0
        assert ctx.min_resistance == 1.0
        assert ctx.max_fanout == 10

    def test_custom_values(self):
        """Test custom context values."""
        ctx = DRCContext(
            max_current_ma=500,
            max_power_mw=1000,
            max_voltage=24,
            min_resistance=0.1,
            max_fanout=20,
        )

        assert ctx.max_current_ma == 500
        assert ctx.max_power_mw == 1000
        assert ctx.max_voltage == 24
        assert ctx.min_resistance == 0.1
        assert ctx.max_fanout == 20


# ============================================================================
# ConstraintTemplate Tests
# ============================================================================


class TestConstraintTemplate:
    """Tests for ConstraintTemplate presets."""

    def test_low_power_template(self):
        """Test low power template constraints."""
        ctx = ConstraintTemplate.low_power()

        assert ctx.max_current_ma == 10.0
        assert ctx.max_power_mw == 50.0
        assert ctx.max_voltage == 5.0

    def test_high_power_template(self):
        """Test high power template constraints."""
        ctx = ConstraintTemplate.high_power()

        assert ctx.max_current_ma == 10000.0  # 10A
        assert ctx.max_power_mw == 100000.0  # 100W
        assert ctx.max_voltage == 100.0

    def test_precision_analog_template(self):
        """Test precision analog template constraints."""
        ctx = ConstraintTemplate.precision_analog()

        assert ctx.target_impedance == 10000.0
        assert ctx.impedance_tolerance == 5.0

    def test_rf_design_template(self):
        """Test RF design template constraints."""
        ctx = ConstraintTemplate.rf_design()

        assert ctx.target_impedance == 50.0  # Standard RF impedance
        assert ctx.max_fanout == 2

    def test_digital_logic_template(self):
        """Test digital logic template constraints."""
        ctx = ConstraintTemplate.digital_logic()

        assert ctx.max_voltage == 5.0
        assert ctx.max_fanout == 10

    def test_automotive_template(self):
        """Test automotive template constraints."""
        ctx = ConstraintTemplate.automotive()

        assert ctx.max_voltage == 60.0  # Load dump handling
        assert ctx.max_current_ma == 5000.0


# ============================================================================
# AdvancedDRC Tests
# ============================================================================


class TestAdvancedDRC:
    """Tests for AdvancedDRC checker."""

    def test_simple_circuit_passes(self, simple_circuit):
        """Test that a simple valid circuit passes DRC."""
        drc = AdvancedDRC()
        report = drc.check(simple_circuit)

        assert report.passed
        assert report.rules_checked > 0
        assert report.rules_passed > 0

    def test_low_resistance_warning(self, low_resistance_circuit):
        """Test detection of low resistance."""
        drc = AdvancedDRC()
        ctx = DRCContext(min_resistance=1.0)  # R1 is 0.1 Ohm
        report = drc.check(low_resistance_circuit, ctx)

        # Should have a warning about low resistance
        low_r_warnings = [
            v for v in report.violations if "resistance" in v.message.lower()
        ]
        assert len(low_r_warnings) > 0

    def test_high_power_warning(self, high_power_circuit):
        """Test detection of high power dissipation."""
        drc = AdvancedDRC()
        ctx = DRCContext(max_power_mw=100.0)  # Low limit
        report = drc.check(high_power_circuit, ctx)

        # Should have a warning about power dissipation
        power_warnings = [
            v for v in report.violations if "power" in v.message.lower()
        ]
        assert len(power_warnings) > 0

    def test_high_fanout_warning(self, high_fanout_circuit):
        """Test detection of high fanout."""
        drc = AdvancedDRC()
        ctx = DRCContext(max_fanout=10)  # Circuit has 15 loads
        report = drc.check(high_fanout_circuit, ctx)

        # Should have a warning about high fanout
        fanout_warnings = [
            v for v in report.violations if "fanout" in v.message.lower()
        ]
        assert len(fanout_warnings) > 0

    def test_voltage_rating_warning(self):
        """Test detection of high voltage."""
        circuit = Circuit("high_v")
        v1 = Vdc("V1", 100.0)  # 100V
        r1 = Resistor("R1", "10k")
        circuit.add(v1, r1)
        circuit.connect(v1.ports[0], Net("in"))
        circuit.connect(v1.ports[1], GND)
        circuit.connect(r1.ports[0], Net("in"))
        circuit.connect(r1.ports[1], GND)

        drc = AdvancedDRC()
        ctx = DRCContext(max_voltage=50.0)
        report = drc.check(circuit, ctx)

        # Should have a warning about voltage
        v_warnings = [v for v in report.violations if "voltage" in v.message.lower()]
        assert len(v_warnings) > 0

    def test_list_rules(self):
        """Test listing registered rules."""
        drc = AdvancedDRC()
        rules = drc.list_rules()

        assert len(rules) > 0
        assert all("name" in r for r in rules)
        assert all("category" in r for r in rules)

    def test_enable_disable_rule(self):
        """Test enabling and disabling rules."""
        drc = AdvancedDRC()

        # Disable a rule
        drc.enable_rule("min_resistance", False)
        rules = drc.list_rules()
        r = next(r for r in rules if r["name"] == "min_resistance")
        assert r["enabled"] is False

        # Re-enable
        drc.enable_rule("min_resistance", True)
        rules = drc.list_rules()
        r = next(r for r in rules if r["name"] == "min_resistance")
        assert r["enabled"] is True

    def test_add_custom_rule(self, simple_circuit):
        """Test adding a custom rule."""
        drc = AdvancedDRC()

        # Add custom rule that always triggers
        def custom_check(circuit, context):
            from spicelab.validators import ValidationWarning

            return [
                ValidationWarning(
                    severity="info",
                    message="Custom rule triggered",
                )
            ]

        drc.add_rule(
            DRCRule(
                name="custom_test",
                description="Test custom rule",
                category="custom",
                check_fn=custom_check,
                severity=Severity.INFO,
            )
        )

        report = drc.check(simple_circuit)

        # Should have custom rule violation
        custom_violations = [
            v for v in report.violations if "Custom rule" in v.message
        ]
        assert len(custom_violations) == 1

    def test_category_filter(self, simple_circuit):
        """Test filtering checks by category."""
        drc = AdvancedDRC()

        # Only run power category
        report = drc.check(simple_circuit, categories=["power"])

        # Should have checked fewer rules
        full_report = drc.check(simple_circuit)
        assert report.rules_checked < full_report.rules_checked


# ============================================================================
# DRCReport Tests
# ============================================================================


class TestDRCReport:
    """Tests for DRCReport output."""

    def test_report_properties(self, simple_circuit):
        """Test report property methods."""
        drc = AdvancedDRC()
        report = drc.check(simple_circuit)

        assert isinstance(report.passed, bool)
        assert isinstance(report.has_errors, bool)
        assert isinstance(report.has_warnings, bool)

    def test_to_dict(self, simple_circuit):
        """Test report to dictionary conversion."""
        drc = AdvancedDRC()
        report = drc.check(simple_circuit)
        d = report.to_dict()

        assert "circuit_name" in d
        assert "timestamp" in d
        assert "passed" in d
        assert "statistics" in d
        assert "violations" in d

    def test_to_json(self, simple_circuit):
        """Test report to JSON conversion."""
        drc = AdvancedDRC()
        report = drc.check(simple_circuit)
        json_str = report.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["circuit_name"] == "test_circuit"

    def test_to_html(self, simple_circuit):
        """Test report to HTML conversion."""
        drc = AdvancedDRC()
        report = drc.check(simple_circuit)
        html = report.to_html()

        assert "<!DOCTYPE html>" in html
        assert "test_circuit" in html
        assert "DRC" in html or "Design Rule" in html

    def test_save_json(self, simple_circuit):
        """Test saving report to JSON file."""
        drc = AdvancedDRC()
        report = drc.check(simple_circuit)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report"
            saved_path = report.save(path, format="json")

            assert saved_path.exists()
            assert saved_path.suffix == ".json"

            # Verify content
            data = json.loads(saved_path.read_text())
            assert data["circuit_name"] == "test_circuit"

    def test_save_html(self, simple_circuit):
        """Test saving report to HTML file."""
        drc = AdvancedDRC()
        report = drc.check(simple_circuit)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report"
            saved_path = report.save(path, format="html")

            assert saved_path.exists()
            assert saved_path.suffix == ".html"

            content = saved_path.read_text()
            assert "<!DOCTYPE html>" in content


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestRunDRC:
    """Tests for run_drc convenience function."""

    def test_run_drc_basic(self, simple_circuit):
        """Test basic run_drc usage."""
        report = run_drc(simple_circuit)

        assert isinstance(report, DRCReport)
        assert report.circuit_name == "test_circuit"

    def test_run_drc_with_template(self, simple_circuit):
        """Test run_drc with template."""
        report = run_drc(simple_circuit, template="low_power")

        assert isinstance(report, DRCReport)
        assert report.context.max_current_ma == 10.0

    def test_run_drc_with_context(self, simple_circuit):
        """Test run_drc with custom context."""
        ctx = DRCContext(max_voltage=3.3)
        report = run_drc(simple_circuit, context=ctx)

        assert report.context.max_voltage == 3.3

    def test_run_drc_invalid_template(self, simple_circuit):
        """Test run_drc with invalid template."""
        with pytest.raises(ValueError, match="Unknown template"):
            run_drc(simple_circuit, template="nonexistent")

    def test_all_templates(self, simple_circuit):
        """Test all available templates."""
        templates = [
            "low_power",
            "high_power",
            "precision_analog",
            "rf_design",
            "digital_logic",
            "automotive",
        ]

        for template in templates:
            report = run_drc(simple_circuit, template=template)
            assert isinstance(report, DRCReport)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for DRC with various circuit types."""

    def test_rc_lowpass_circuit(self):
        """Test DRC on RC lowpass filter."""
        circuit = Circuit("rc_lowpass")

        v1 = Vdc("V1", 5.0)
        r1 = Resistor("R1", "10k")
        c1 = Capacitor("C1", "100n")

        circuit.add(v1, r1, c1)

        circuit.connect(v1.ports[0], Net("in"))
        circuit.connect(v1.ports[1], GND)
        circuit.connect(r1.ports[0], Net("in"))
        circuit.connect(r1.ports[1], Net("out"))
        circuit.connect(c1.ports[0], Net("out"))
        circuit.connect(c1.ports[1], GND)

        report = run_drc(circuit)
        assert report.passed

    def test_voltage_divider_circuit(self):
        """Test DRC on voltage divider."""
        circuit = Circuit("voltage_divider")

        v1 = Vdc("V1", 12.0)
        r1 = Resistor("R1", "10k")
        r2 = Resistor("R2", "10k")

        circuit.add(v1, r1, r2)

        circuit.connect(v1.ports[0], Net("vin"))
        circuit.connect(v1.ports[1], GND)
        circuit.connect(r1.ports[0], Net("vin"))
        circuit.connect(r1.ports[1], Net("vout"))
        circuit.connect(r2.ports[0], Net("vout"))
        circuit.connect(r2.ports[1], GND)

        report = run_drc(circuit)
        assert report.passed

    def test_decoupling_check(self):
        """Test decoupling capacitor check."""
        # Circuit without decoupling cap
        circuit = Circuit("no_decoupling")

        v1 = Vdc("V1", 5.0)
        v2 = Vdc("V2", 3.3)
        r1 = Resistor("R1", "1k")

        circuit.add(v1, v2, r1)

        circuit.connect(v1.ports[0], Net("vcc5"))
        circuit.connect(v1.ports[1], GND)
        circuit.connect(v2.ports[0], Net("vcc3"))
        circuit.connect(v2.ports[1], GND)
        circuit.connect(r1.ports[0], Net("vcc5"))
        circuit.connect(r1.ports[1], GND)

        drc = AdvancedDRC()
        report = drc.check(circuit)

        # Should suggest decoupling caps
        decoupling_info = [
            v for v in report.violations if "capacitor" in v.message.lower()
        ]
        assert len(decoupling_info) > 0
