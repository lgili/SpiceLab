"""Example telemetry plugin.

This plugin demonstrates how to collect and report simulation
telemetry data using the hook system.

Usage::

    from spicelab.plugins import PluginManager
    from spicelab.plugins.examples import TelemetryPlugin

    manager = PluginManager()
    plugin = manager.loader.load_from_class(TelemetryPlugin)
    manager.registry.register(plugin)
    manager.activate_plugin("telemetry-plugin")

    # Run simulations...

    # Get telemetry report
    report = plugin.get_report()
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..base import Plugin, PluginMetadata, PluginType
from ..hooks import HookManager, HookPriority, HookType


@dataclass
class SimulationRecord:
    """Record of a single simulation run."""

    circuit_name: str
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    num_analyses: int = 0
    success: bool = True
    error_message: str = ""
    cache_hit: bool = False


@dataclass
class TelemetryStats:
    """Aggregated telemetry statistics."""

    total_simulations: int = 0
    successful_simulations: int = 0
    failed_simulations: int = 0
    total_duration_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    validations_passed: int = 0
    validations_failed: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_simulations == 0:
            return 0.0
        return self.successful_simulations / self.total_simulations

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def average_duration(self) -> float:
        """Calculate average simulation duration."""
        if self.successful_simulations == 0:
            return 0.0
        return self.total_duration_seconds / self.successful_simulations

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_simulations": self.total_simulations,
            "successful_simulations": self.successful_simulations,
            "failed_simulations": self.failed_simulations,
            "success_rate": f"{self.success_rate:.1%}",
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "average_duration_seconds": round(self.average_duration, 3),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{self.cache_hit_rate:.1%}",
            "validations_passed": self.validations_passed,
            "validations_failed": self.validations_failed,
        }


class TelemetryPlugin(Plugin):
    """Plugin that collects simulation telemetry.

    Features:
    - Tracks simulation count, success rate, and duration
    - Records cache hit/miss statistics
    - Tracks validation results
    - Provides aggregated reports

    This plugin demonstrates:
    - Collecting data across multiple hooks
    - Maintaining state between hook calls
    - Generating reports from collected data
    """

    def __init__(self) -> None:
        self._stats = TelemetryStats()
        self._records: list[SimulationRecord] = []
        self._current_record: SimulationRecord | None = None
        self._max_records: int = 1000

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="telemetry-plugin",
            version="1.0.0",
            description="Simulation telemetry collection",
            author="SpiceLab Team",
            plugin_type=PluginType.GENERIC,
            keywords=["telemetry", "metrics", "monitoring"],
        )

    def configure(self, settings: dict[str, Any]) -> None:
        """Configure the telemetry plugin.

        Args:
            settings: Configuration dictionary with keys:
                - max_records: Maximum number of records to keep
        """
        self._max_records = settings.get("max_records", 1000)

    def activate(self) -> None:
        """Activate the plugin and register hooks."""
        self._register_hooks()

    def deactivate(self) -> None:
        """Deactivate the plugin."""
        pass

    def _register_hooks(self) -> None:
        """Register telemetry hooks."""
        hook_manager = HookManager.get_instance()

        # Simulation hooks
        hook_manager.register_hook(
            HookType.PRE_SIMULATION,
            self._on_pre_simulation,
            priority=HookPriority.CRITICAL,
            plugin_name=self.name,
        )

        hook_manager.register_hook(
            HookType.POST_SIMULATION,
            self._on_post_simulation,
            priority=HookPriority.LOWEST,
            plugin_name=self.name,
        )

        hook_manager.register_hook(
            HookType.SIMULATION_ERROR,
            self._on_simulation_error,
            plugin_name=self.name,
        )

        # Cache hooks
        hook_manager.register_hook(
            HookType.CACHE_HIT,
            self._on_cache_hit,
            plugin_name=self.name,
        )

        hook_manager.register_hook(
            HookType.CACHE_MISS,
            self._on_cache_miss,
            plugin_name=self.name,
        )

        # Validation hooks
        hook_manager.register_hook(
            HookType.POST_VALIDATION,
            self._on_validation,
            plugin_name=self.name,
        )

    def _on_pre_simulation(self, **kwargs: Any) -> None:
        """Start tracking a simulation."""
        circuit = kwargs.get("circuit")
        analyses = kwargs.get("analyses", [])

        circuit_name = getattr(circuit, "name", "unknown") if circuit else "unknown"

        self._current_record = SimulationRecord(
            circuit_name=circuit_name,
            start_time=datetime.now(),
            num_analyses=len(analyses) if analyses else 0,
        )

    def _on_post_simulation(self, **kwargs: Any) -> None:
        """Finish tracking a simulation."""
        if self._current_record:
            self._current_record.end_time = datetime.now()
            self._current_record.duration_seconds = (
                self._current_record.end_time - self._current_record.start_time
            ).total_seconds()

            self._stats.total_simulations += 1
            self._stats.successful_simulations += 1
            self._stats.total_duration_seconds += self._current_record.duration_seconds

            self._add_record(self._current_record)
            self._current_record = None

    def _on_simulation_error(self, **kwargs: Any) -> None:
        """Record a simulation error."""
        if self._current_record:
            self._current_record.success = False
            self._current_record.error_message = str(kwargs.get("error", ""))
            self._current_record.end_time = datetime.now()

            self._stats.total_simulations += 1
            self._stats.failed_simulations += 1

            self._add_record(self._current_record)
            self._current_record = None

    def _on_cache_hit(self, **kwargs: Any) -> None:
        """Record a cache hit."""
        self._stats.cache_hits += 1
        if self._current_record:
            self._current_record.cache_hit = True

    def _on_cache_miss(self, **kwargs: Any) -> None:
        """Record a cache miss."""
        self._stats.cache_misses += 1

    def _on_validation(self, **kwargs: Any) -> None:
        """Record validation result."""
        errors = kwargs.get("errors", [])
        if errors:
            self._stats.validations_failed += 1
        else:
            self._stats.validations_passed += 1

    def _add_record(self, record: SimulationRecord) -> None:
        """Add a record, pruning old records if needed."""
        self._records.append(record)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]

    # Public API

    def get_stats(self) -> TelemetryStats:
        """Get current telemetry statistics.

        Returns:
            TelemetryStats object
        """
        return self._stats

    def get_records(self, limit: int = 100) -> list[SimulationRecord]:
        """Get recent simulation records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of SimulationRecord objects
        """
        return self._records[-limit:]

    def get_report(self) -> dict[str, Any]:
        """Get a full telemetry report.

        Returns:
            Dictionary with stats and recent records
        """
        return {
            "stats": self._stats.to_dict(),
            "recent_simulations": [
                {
                    "circuit": r.circuit_name,
                    "duration": round(r.duration_seconds, 3),
                    "success": r.success,
                    "cache_hit": r.cache_hit,
                }
                for r in self._records[-10:]
            ],
        }

    def reset(self) -> None:
        """Reset all telemetry data."""
        self._stats = TelemetryStats()
        self._records.clear()
        self._current_record = None
