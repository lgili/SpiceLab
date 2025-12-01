"""Example plugins demonstrating the SpiceLab plugin system.

This package contains example plugins that serve as templates and
documentation for creating custom plugins.

Available plugins:

**Development & Debug:**
- LoggingPlugin: Comprehensive logging for SpiceLab events
- TelemetryPlugin: Collect simulation telemetry and statistics
- SimulationProfilerPlugin: Profile performance and identify bottlenecks

**Productivity:**
- ReportGeneratorPlugin: Auto-generate simulation reports (MD/HTML/JSON)
- AutoBackupPlugin: Automatic circuit backup with version history
- CircuitTemplatesPlugin: Pre-built circuit templates for common designs

**Validation:**
- DesignRulesPlugin: Design Rule Checks (DRC) for circuit validation

Usage::

    from spicelab.plugins.examples import LoggingPlugin, ReportGeneratorPlugin

    # Register with manager
    manager = PluginManager()
    manager.loader.load_from_class(LoggingPlugin)
    manager.activate_plugin("logging-plugin")

    # Configure a plugin
    manager.set_plugin_settings("report-generator", {
        "output_dir": "./reports",
        "format": "html",
    })
"""

from .backup_plugin import AutoBackupPlugin
from .circuit_templates_plugin import CircuitTemplatesPlugin
from .design_rules_plugin import DesignRulesPlugin
from .logging_plugin import LoggingPlugin
from .profiler_plugin import SimulationProfilerPlugin
from .report_plugin import ReportGeneratorPlugin
from .telemetry_plugin import TelemetryPlugin

__all__ = [
    # Development & Debug
    "LoggingPlugin",
    "TelemetryPlugin",
    "SimulationProfilerPlugin",
    # Productivity
    "ReportGeneratorPlugin",
    "AutoBackupPlugin",
    "CircuitTemplatesPlugin",
    # Validation
    "DesignRulesPlugin",
]
