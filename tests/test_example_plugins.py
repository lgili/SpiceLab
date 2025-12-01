"""Tests for example plugins."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.plugins.examples import (
    AutoBackupPlugin,
    CircuitTemplatesPlugin,
    DesignRulesPlugin,
    LoggingPlugin,
    ReportGeneratorPlugin,
    SimulationProfilerPlugin,
    TelemetryPlugin,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_circuit() -> Circuit:
    """Create a simple RC circuit for testing."""
    circuit = Circuit("test_rc")
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


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# LoggingPlugin Tests
# =============================================================================


class TestLoggingPlugin:
    """Tests for LoggingPlugin."""

    def test_metadata(self) -> None:
        """Plugin has correct metadata."""
        plugin = LoggingPlugin()
        assert plugin.metadata.name == "logging-plugin"
        assert plugin.metadata.version == "1.0.0"

    def test_configure(self) -> None:
        """Can configure plugin settings."""
        plugin = LoggingPlugin()
        plugin.configure({"log_level": "DEBUG", "console_output": False})

    def test_activate_deactivate(self) -> None:
        """Can activate and deactivate plugin."""
        plugin = LoggingPlugin()
        plugin.activate()
        plugin.deactivate()


# =============================================================================
# TelemetryPlugin Tests
# =============================================================================


class TestTelemetryPlugin:
    """Tests for TelemetryPlugin."""

    def test_metadata(self) -> None:
        """Plugin has correct metadata."""
        plugin = TelemetryPlugin()
        assert plugin.metadata.name == "telemetry-plugin"
        assert plugin.metadata.version == "1.0.0"

    def test_activate_deactivate(self) -> None:
        """Can activate and deactivate plugin."""
        plugin = TelemetryPlugin()
        plugin.activate()
        plugin.deactivate()


# =============================================================================
# ReportGeneratorPlugin Tests
# =============================================================================


class TestReportGeneratorPlugin:
    """Tests for ReportGeneratorPlugin."""

    def test_metadata(self) -> None:
        """Plugin has correct metadata."""
        plugin = ReportGeneratorPlugin()
        assert plugin.metadata.name == "report-generator"
        assert plugin.metadata.version == "1.0.0"

    def test_configure(self, temp_dir: Path) -> None:
        """Can configure plugin settings."""
        plugin = ReportGeneratorPlugin()
        plugin.configure({
            "output_dir": str(temp_dir),
            "format": "html",
            "include_netlist": True,
        })

    def test_activate_deactivate(self, temp_dir: Path) -> None:
        """Can activate and deactivate plugin."""
        plugin = ReportGeneratorPlugin()
        plugin.configure({"output_dir": str(temp_dir)})
        plugin.activate()
        plugin.deactivate()

    def test_generate_report_markdown(
        self, simple_circuit: Circuit, temp_dir: Path
    ) -> None:
        """Can generate markdown report."""
        plugin = ReportGeneratorPlugin()
        plugin.configure({"output_dir": str(temp_dir)})

        # Create mock result
        class MockResult:
            traces: dict[str, Any] = {"v(out)": [1, 2, 3]}

        path = plugin.generate_report(simple_circuit, MockResult(), format="markdown")
        assert path.exists()
        assert path.suffix == ".md"

        content = path.read_text()
        assert "test_rc" in content
        assert "v(out)" in content

    def test_generate_report_html(
        self, simple_circuit: Circuit, temp_dir: Path
    ) -> None:
        """Can generate HTML report."""
        plugin = ReportGeneratorPlugin()
        plugin.configure({"output_dir": str(temp_dir)})

        class MockResult:
            traces: dict[str, Any] = {"v(out)": [1, 2, 3]}

        path = plugin.generate_report(simple_circuit, MockResult(), format="html")
        assert path.exists()
        assert path.suffix == ".html"

        content = path.read_text()
        assert "<html>" in content
        assert "test_rc" in content

    def test_generate_report_json(
        self, simple_circuit: Circuit, temp_dir: Path
    ) -> None:
        """Can generate JSON report."""
        plugin = ReportGeneratorPlugin()
        plugin.configure({"output_dir": str(temp_dir)})

        class MockResult:
            traces: dict[str, Any] = {"v(out)": [1, 2, 3]}

        path = plugin.generate_report(simple_circuit, MockResult(), format="json")
        assert path.exists()
        assert path.suffix == ".json"


# =============================================================================
# DesignRulesPlugin Tests
# =============================================================================


class TestDesignRulesPlugin:
    """Tests for DesignRulesPlugin."""

    def test_metadata(self) -> None:
        """Plugin has correct metadata."""
        plugin = DesignRulesPlugin()
        assert plugin.metadata.name == "design-rules"
        assert plugin.metadata.version == "1.0.0"

    def test_configure(self) -> None:
        """Can configure plugin settings."""
        plugin = DesignRulesPlugin()
        plugin.configure({
            "rules": {"floating_nodes": True, "missing_ground": True},
            "severity": "error",
        })

    def test_activate_deactivate(self) -> None:
        """Can activate and deactivate plugin."""
        plugin = DesignRulesPlugin()
        plugin.activate()
        plugin.deactivate()

    def test_check_passes_valid_circuit(self, simple_circuit: Circuit) -> None:
        """DRC passes for valid circuit."""
        plugin = DesignRulesPlugin()
        result = plugin.check(simple_circuit)
        # Should pass basic checks
        assert result.errors == 0 or result.passed

    def test_check_detects_missing_ground(self) -> None:
        """DRC detects missing ground."""
        plugin = DesignRulesPlugin()

        # Create circuit without ground
        circuit = Circuit("no_ground")
        r = Resistor("R1", "1k")
        circuit.add(r)
        circuit.connect(r.ports[0], Net("a"))
        circuit.connect(r.ports[1], Net("b"))

        result = plugin.check(circuit)
        assert not result.passed
        assert any("ground" in v.message.lower() for v in result.violations)

    def test_check_detects_floating_nodes(self) -> None:
        """DRC detects floating nodes."""
        plugin = DesignRulesPlugin()

        circuit = Circuit("floating")
        r = Resistor("R1", "1k")
        circuit.add(r)
        circuit.connect(r.ports[0], Net("a"))
        circuit.connect(r.ports[1], GND)

        result = plugin.check(circuit)
        # Node "a" has only one connection
        floating_violations = [
            v for v in result.violations if v.rule_name == "floating_node"
        ]
        assert len(floating_violations) > 0


# =============================================================================
# CircuitTemplatesPlugin Tests
# =============================================================================


class TestCircuitTemplatesPlugin:
    """Tests for CircuitTemplatesPlugin."""

    def test_metadata(self) -> None:
        """Plugin has correct metadata."""
        plugin = CircuitTemplatesPlugin()
        assert plugin.metadata.name == "circuit-templates"
        assert plugin.metadata.version == "1.0.0"

    def test_list_templates(self) -> None:
        """Can list available templates."""
        plugin = CircuitTemplatesPlugin()
        templates = plugin.list_templates()
        assert len(templates) > 0
        assert "rc_lowpass" in templates
        assert "voltage_divider" in templates

    def test_list_categories(self) -> None:
        """Can list template categories."""
        plugin = CircuitTemplatesPlugin()
        categories = plugin.list_categories()
        assert "filters" in categories
        assert "basic" in categories
        assert "amplifiers" in categories

    def test_create_rc_lowpass(self) -> None:
        """Can create RC lowpass filter."""
        plugin = CircuitTemplatesPlugin()
        circuit = plugin.create("rc_lowpass", cutoff_freq=1000)
        assert circuit.name == "rc_lowpass"
        assert len(circuit._components) >= 3  # Vin, R, C

    def test_create_voltage_divider(self) -> None:
        """Can create voltage divider."""
        plugin = CircuitTemplatesPlugin()
        circuit = plugin.create("voltage_divider", r1="10k", r2="10k", vin=5.0)
        assert circuit.name == "voltage_divider"

    def test_create_voltage_divider_ratio(self) -> None:
        """Can create voltage divider with ratio."""
        plugin = CircuitTemplatesPlugin()
        circuit = plugin.create("voltage_divider_ratio", ratio=0.5, vin=10.0)
        assert circuit is not None

    def test_create_rc_highpass(self) -> None:
        """Can create RC highpass filter."""
        plugin = CircuitTemplatesPlugin()
        circuit = plugin.create("rc_highpass", cutoff_freq=1000)
        assert circuit.name == "rc_highpass"

    def test_create_invalid_template(self) -> None:
        """Invalid template raises KeyError."""
        plugin = CircuitTemplatesPlugin()
        with pytest.raises(KeyError, match="not found"):
            plugin.create("invalid_template")

    def test_get_template_info(self) -> None:
        """Can get template info."""
        plugin = CircuitTemplatesPlugin()
        info = plugin.get_template_info("rc_lowpass")
        assert info.name == "rc_lowpass"
        assert info.category == "filters"
        assert "cutoff_freq" in info.parameters

    def test_register_custom_template(self) -> None:
        """Can register custom template."""
        plugin = CircuitTemplatesPlugin()

        def create_custom(name: str = "custom") -> Circuit:
            return Circuit(name)

        plugin.register_template(
            name="custom_circuit",
            description="A custom circuit",
            category="custom",
            parameters={"name": "Circuit name"},
            create_fn=create_custom,
        )

        assert "custom_circuit" in plugin.list_templates()
        circuit = plugin.create("custom_circuit", name="my_custom")
        assert circuit.name == "my_custom"


# =============================================================================
# SimulationProfilerPlugin Tests
# =============================================================================


class TestSimulationProfilerPlugin:
    """Tests for SimulationProfilerPlugin."""

    def test_metadata(self) -> None:
        """Plugin has correct metadata."""
        plugin = SimulationProfilerPlugin()
        assert plugin.metadata.name == "simulation-profiler"
        assert plugin.metadata.version == "1.0.0"

    def test_configure(self) -> None:
        """Can configure plugin settings."""
        plugin = SimulationProfilerPlugin()
        plugin.configure({
            "track_memory": True,
            "max_records": 500,
            "auto_report": False,
        })

    def test_activate_deactivate(self) -> None:
        """Can activate and deactivate plugin."""
        plugin = SimulationProfilerPlugin()
        plugin.activate()
        plugin.deactivate()

    def test_get_stats_empty(self) -> None:
        """Stats are empty initially."""
        plugin = SimulationProfilerPlugin()
        stats = plugin.get_stats()
        assert len(stats) == 0

    def test_get_cache_stats(self) -> None:
        """Can get cache stats."""
        plugin = SimulationProfilerPlugin()
        stats = plugin.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_get_report(self) -> None:
        """Can generate report."""
        plugin = SimulationProfilerPlugin()
        plugin.activate()
        report = plugin.get_report()
        assert "PROFILER REPORT" in report
        assert "TIMING STATISTICS" in report

    def test_export_profile(self, temp_dir: Path) -> None:
        """Can export profile to JSON."""
        plugin = SimulationProfilerPlugin()
        plugin.activate()

        export_path = temp_dir / "profile.json"
        plugin.export_profile(export_path)

        assert export_path.exists()

    def test_reset(self) -> None:
        """Can reset profiler data."""
        plugin = SimulationProfilerPlugin()
        plugin.activate()
        plugin.reset()
        assert len(plugin.get_stats()) == 0


# =============================================================================
# AutoBackupPlugin Tests
# =============================================================================


class TestAutoBackupPlugin:
    """Tests for AutoBackupPlugin."""

    def test_metadata(self) -> None:
        """Plugin has correct metadata."""
        plugin = AutoBackupPlugin()
        assert plugin.metadata.name == "auto-backup"
        assert plugin.metadata.version == "1.0.0"

    def test_configure(self, temp_dir: Path) -> None:
        """Can configure plugin settings."""
        plugin = AutoBackupPlugin()
        plugin.configure({
            "backup_dir": str(temp_dir),
            "max_backups": 10,
            "compress": True,
        })

    def test_activate_deactivate(self, temp_dir: Path) -> None:
        """Can activate and deactivate plugin."""
        plugin = AutoBackupPlugin()
        plugin.configure({"backup_dir": str(temp_dir)})
        plugin.activate()
        plugin.deactivate()

    def test_backup_circuit(self, simple_circuit: Circuit, temp_dir: Path) -> None:
        """Can backup a circuit."""
        plugin = AutoBackupPlugin()
        plugin.configure({"backup_dir": str(temp_dir), "compress": False})
        plugin.activate()

        info = plugin.backup(simple_circuit, reason="test")
        assert info is not None
        assert info.circuit_name == "test_rc"
        assert info.path.exists()

    def test_backup_compressed(self, simple_circuit: Circuit, temp_dir: Path) -> None:
        """Can create compressed backup."""
        plugin = AutoBackupPlugin()
        plugin.configure({"backup_dir": str(temp_dir), "compress": True})
        plugin.activate()

        info = plugin.backup(simple_circuit)
        assert info is not None
        assert info.compressed
        assert info.path.suffix == ".gz"

    def test_list_backups(self, temp_dir: Path) -> None:
        """Can list backups."""
        plugin = AutoBackupPlugin()
        plugin.configure({"backup_dir": str(temp_dir), "deduplicate": False})
        plugin.activate()

        # Create two different circuits for distinct backups
        circuit1 = Circuit("circuit1")
        r1 = Resistor("R1", "1k")
        circuit1.add(r1)
        circuit1.connect(r1.ports[0], Net("a"))
        circuit1.connect(r1.ports[1], GND)

        circuit2 = Circuit("circuit2")
        r2 = Resistor("R2", "2k")
        circuit2.add(r2)
        circuit2.connect(r2.ports[0], Net("b"))
        circuit2.connect(r2.ports[1], GND)

        plugin.backup(circuit1)
        plugin.backup(circuit2)

        backups = plugin.list_backups()
        assert len(backups) == 2

    def test_restore_netlist(self, simple_circuit: Circuit, temp_dir: Path) -> None:
        """Can restore netlist from backup."""
        plugin = AutoBackupPlugin()
        plugin.configure({"backup_dir": str(temp_dir), "compress": False})
        plugin.activate()

        info = plugin.backup(simple_circuit)
        assert info is not None

        netlist = plugin.restore_netlist(info.name)
        assert "R1" in netlist
        assert "C1" in netlist

    def test_delete_backup(self, simple_circuit: Circuit, temp_dir: Path) -> None:
        """Can delete backup."""
        plugin = AutoBackupPlugin()
        plugin.configure({"backup_dir": str(temp_dir)})
        plugin.activate()

        info = plugin.backup(simple_circuit)
        assert info is not None

        result = plugin.delete_backup(info.name)
        assert result is True
        assert len(plugin.list_backups()) == 0

    def test_deduplicate(self, simple_circuit: Circuit, temp_dir: Path) -> None:
        """Deduplication prevents duplicate backups."""
        plugin = AutoBackupPlugin()
        plugin.configure({"backup_dir": str(temp_dir), "deduplicate": True})
        plugin.activate()

        info1 = plugin.backup(simple_circuit)
        info2 = plugin.backup(simple_circuit)  # Same content

        assert info1 is not None
        assert info2 is None  # Skipped due to dedup
        assert len(plugin.list_backups()) == 1

    def test_get_stats(self, simple_circuit: Circuit, temp_dir: Path) -> None:
        """Can get backup stats."""
        plugin = AutoBackupPlugin()
        plugin.configure({"backup_dir": str(temp_dir)})
        plugin.activate()

        plugin.backup(simple_circuit)

        stats = plugin.get_stats()
        assert stats["total_backups"] == 1
        assert "test_rc" in stats["circuits"]

    def test_max_backups_cleanup(self, temp_dir: Path) -> None:
        """Old backups are cleaned up when max_backups exceeded."""
        plugin = AutoBackupPlugin()
        plugin.configure({
            "backup_dir": str(temp_dir),
            "max_backups": 2,
            "deduplicate": False,
        })
        plugin.activate()

        # Create 3 different circuits for distinct backups
        for i in range(3):
            circuit = Circuit(f"circuit_{i}")
            r = Resistor(f"R{i}", f"{i + 1}k")
            circuit.add(r)
            circuit.connect(r.ports[0], Net(f"n{i}"))
            circuit.connect(r.ports[1], GND)
            plugin.backup(circuit)

        # Should only have 2 (oldest deleted)
        assert len(plugin.list_backups()) == 2


# =============================================================================
# Import Tests
# =============================================================================


class TestPluginImports:
    """Test that all plugins can be imported."""

    def test_import_all_plugins(self) -> None:
        """All plugins can be imported from examples."""
        from spicelab.plugins.examples import (
            AutoBackupPlugin,
            CircuitTemplatesPlugin,
            DesignRulesPlugin,
            LoggingPlugin,
            ReportGeneratorPlugin,
            SimulationProfilerPlugin,
            TelemetryPlugin,
        )

        assert LoggingPlugin is not None
        assert TelemetryPlugin is not None
        assert ReportGeneratorPlugin is not None
        assert DesignRulesPlugin is not None
        assert CircuitTemplatesPlugin is not None
        assert SimulationProfilerPlugin is not None
        assert AutoBackupPlugin is not None

    def test_all_plugins_have_metadata(self) -> None:
        """All plugins have valid metadata."""
        plugins = [
            LoggingPlugin(),
            TelemetryPlugin(),
            ReportGeneratorPlugin(),
            DesignRulesPlugin(),
            CircuitTemplatesPlugin(),
            SimulationProfilerPlugin(),
            AutoBackupPlugin(),
        ]

        for plugin in plugins:
            assert plugin.metadata.name
            assert plugin.metadata.version
            assert plugin.metadata.description
