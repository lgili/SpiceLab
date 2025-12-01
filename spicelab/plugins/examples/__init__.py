"""Example plugins demonstrating the SpiceLab plugin system.

This package contains example plugins that serve as templates and
documentation for creating custom plugins.

Available examples:
- LoggingPlugin: Demonstrates hook system for logging
- TelemetryPlugin: Shows how to collect simulation telemetry
- CustomMeasurementPlugin: Example measurement plugin
- ValidationPlugin: Example validation hook plugin

Usage::

    from spicelab.plugins.examples import LoggingPlugin

    # Register with manager
    manager = PluginManager()
    manager.loader.load_from_class(LoggingPlugin)
    manager.activate_plugin("logging-plugin")
"""

from .logging_plugin import LoggingPlugin
from .telemetry_plugin import TelemetryPlugin

__all__ = [
    "LoggingPlugin",
    "TelemetryPlugin",
]
