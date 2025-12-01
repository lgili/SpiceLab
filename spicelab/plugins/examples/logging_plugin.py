"""Example logging plugin.

This plugin demonstrates how to use the hook system to add logging
functionality to SpiceLab. It logs key events during simulation.

Usage::

    from spicelab.plugins import PluginManager
    from spicelab.plugins.examples import LoggingPlugin

    manager = PluginManager()
    plugin = manager.loader.load_from_class(LoggingPlugin)
    manager.registry.register(plugin)
    manager.activate_plugin("logging-plugin")

    # Now all simulation events will be logged

Configuration::

    # Set log level via plugin settings
    manager.set_plugin_settings("logging-plugin", {
        "log_level": "DEBUG",
        "log_file": "/path/to/spicelab.log",
        "console_output": True,
    })
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..base import Plugin, PluginMetadata, PluginType
from ..hooks import HookManager, HookPriority, HookType


class LoggingPlugin(Plugin):
    """Plugin that adds comprehensive logging to SpiceLab.

    Features:
    - Logs simulation start/end with timing
    - Logs circuit building events
    - Logs validation results
    - Logs cache hits/misses
    - Configurable log level and output

    This plugin demonstrates:
    - Using multiple hook types
    - Plugin configuration via settings
    - Proper activate/deactivate lifecycle
    """

    def __init__(self) -> None:
        self._logger: logging.Logger | None = None
        self._file_handler: logging.FileHandler | None = None
        self._console_handler: logging.StreamHandler | None = None
        self._simulation_start: datetime | None = None
        self._config: dict[str, Any] = {
            "log_level": "INFO",
            "log_file": None,
            "console_output": True,
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        }

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="logging-plugin",
            version="1.0.0",
            description="Comprehensive logging for SpiceLab",
            author="SpiceLab Team",
            plugin_type=PluginType.GENERIC,
            keywords=["logging", "debug", "monitoring"],
        )

    def configure(self, settings: dict[str, Any]) -> None:
        """Configure the logging plugin.

        Args:
            settings: Configuration dictionary with keys:
                - log_level: Log level (DEBUG, INFO, WARNING, ERROR)
                - log_file: Path to log file (optional)
                - console_output: Whether to log to console
                - format: Log format string
        """
        self._config.update(settings)
        self._setup_logger()

    def activate(self) -> None:
        """Activate the plugin and register hooks."""
        self._setup_logger()
        self._register_hooks()
        self._log("Logging plugin activated")

    def deactivate(self) -> None:
        """Deactivate the plugin and cleanup."""
        self._log("Logging plugin deactivating")
        self._cleanup_logger()

    def _setup_logger(self) -> None:
        """Set up the logger with configured handlers."""
        self._logger = logging.getLogger("spicelab.plugins.logging")
        self._logger.setLevel(self._config["log_level"])

        # Remove existing handlers
        self._logger.handlers.clear()

        formatter = logging.Formatter(self._config["format"])

        # Console handler
        if self._config["console_output"]:
            self._console_handler = logging.StreamHandler(sys.stdout)
            self._console_handler.setFormatter(formatter)
            self._logger.addHandler(self._console_handler)

        # File handler
        if self._config["log_file"]:
            log_path = Path(self._config["log_file"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handler = logging.FileHandler(log_path)
            self._file_handler.setFormatter(formatter)
            self._logger.addHandler(self._file_handler)

    def _cleanup_logger(self) -> None:
        """Clean up logger handlers."""
        if self._logger:
            if self._file_handler:
                self._logger.removeHandler(self._file_handler)
                self._file_handler.close()
            if self._console_handler:
                self._logger.removeHandler(self._console_handler)

    def _log(self, message: str, level: str = "INFO") -> None:
        """Log a message."""
        if self._logger:
            log_func = getattr(self._logger, level.lower(), self._logger.info)
            log_func(message)

    def _register_hooks(self) -> None:
        """Register all logging hooks."""
        hook_manager = HookManager.get_instance()

        # Simulation hooks
        hook_manager.register_hook(
            HookType.PRE_SIMULATION,
            self._on_pre_simulation,
            priority=HookPriority.HIGH,
            plugin_name=self.name,
            description="Log simulation start",
        )

        hook_manager.register_hook(
            HookType.POST_SIMULATION,
            self._on_post_simulation,
            priority=HookPriority.LOW,
            plugin_name=self.name,
            description="Log simulation end",
        )

        hook_manager.register_hook(
            HookType.SIMULATION_ERROR,
            self._on_simulation_error,
            plugin_name=self.name,
            description="Log simulation errors",
        )

        # Netlist hooks
        hook_manager.register_hook(
            HookType.PRE_NETLIST_BUILD,
            self._on_pre_netlist,
            plugin_name=self.name,
            description="Log netlist build start",
        )

        hook_manager.register_hook(
            HookType.POST_NETLIST_BUILD,
            self._on_post_netlist,
            plugin_name=self.name,
            description="Log netlist build end",
        )

        # Validation hooks
        hook_manager.register_hook(
            HookType.POST_VALIDATION,
            self._on_validation,
            plugin_name=self.name,
            description="Log validation results",
        )

        # Cache hooks
        hook_manager.register_hook(
            HookType.CACHE_HIT,
            self._on_cache_hit,
            plugin_name=self.name,
            description="Log cache hits",
        )

        hook_manager.register_hook(
            HookType.CACHE_MISS,
            self._on_cache_miss,
            plugin_name=self.name,
            description="Log cache misses",
        )

    # Hook handlers

    def _on_pre_simulation(self, **kwargs: Any) -> None:
        """Handle pre-simulation hook."""
        self._simulation_start = datetime.now()
        circuit = kwargs.get("circuit")
        analyses = kwargs.get("analyses", [])

        circuit_name = getattr(circuit, "name", "unknown") if circuit else "unknown"
        num_analyses = len(analyses) if analyses else 0

        self._log(f"Starting simulation: circuit={circuit_name}, analyses={num_analyses}")

    def _on_post_simulation(self, **kwargs: Any) -> None:
        """Handle post-simulation hook."""
        if self._simulation_start:
            duration = datetime.now() - self._simulation_start
            self._log(f"Simulation completed in {duration.total_seconds():.3f}s")
        else:
            self._log("Simulation completed")

    def _on_simulation_error(self, **kwargs: Any) -> None:
        """Handle simulation error hook."""
        error = kwargs.get("error")
        self._log(f"Simulation error: {error}", level="ERROR")

    def _on_pre_netlist(self, **kwargs: Any) -> None:
        """Handle pre-netlist build hook."""
        self._log("Building netlist...", level="DEBUG")

    def _on_post_netlist(self, **kwargs: Any) -> None:
        """Handle post-netlist build hook."""
        netlist = kwargs.get("netlist", "")
        lines = len(netlist.split("\n")) if netlist else 0
        self._log(f"Netlist built: {lines} lines", level="DEBUG")

    def _on_validation(self, **kwargs: Any) -> None:
        """Handle validation hook."""
        errors = kwargs.get("errors", [])
        warnings = kwargs.get("warnings", [])

        if errors:
            self._log(f"Validation errors: {len(errors)}", level="ERROR")
            for error in errors[:5]:  # Log first 5
                self._log(f"  - {error}", level="ERROR")
        elif warnings:
            self._log(f"Validation warnings: {len(warnings)}", level="WARNING")
        else:
            self._log("Validation passed", level="DEBUG")

    def _on_cache_hit(self, **kwargs: Any) -> None:
        """Handle cache hit hook."""
        key = kwargs.get("key", "unknown")
        self._log(f"Cache hit: {key[:32]}...", level="DEBUG")

    def _on_cache_miss(self, **kwargs: Any) -> None:
        """Handle cache miss hook."""
        key = kwargs.get("key", "unknown")
        self._log(f"Cache miss: {key[:32]}...", level="DEBUG")
