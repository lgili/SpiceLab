"""Simulation Profiler Plugin.

This plugin profiles simulation performance and identifies bottlenecks,
helping users optimize their simulation workflows.

Usage::

    from spicelab.plugins import PluginManager
    from spicelab.plugins.examples import SimulationProfilerPlugin

    manager = PluginManager()
    plugin = manager.loader.load_from_class(SimulationProfilerPlugin)
    manager.registry.register(plugin)
    manager.activate_plugin("simulation-profiler")

    # Run simulations normally - profiling happens automatically

    # Get profiling report
    report = plugin.get_report()
    print(report)

    # Or export detailed data
    plugin.export_profile("profile.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..base import Plugin, PluginMetadata, PluginType
from ..hooks import HookManager, HookPriority, HookType


@dataclass
class TimingRecord:
    """Record of a timed operation."""

    operation: str
    start_time: float
    end_time: float | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def complete(self) -> None:
        """Mark the record as complete."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000


@dataclass
class ProfileStats:
    """Aggregated profiling statistics."""

    operation: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    avg_ms: float = 0.0

    def add_sample(self, duration_ms: float) -> None:
        """Add a timing sample."""
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        self.avg_ms = self.total_ms / self.count


class SimulationProfilerPlugin(Plugin):
    """Plugin that profiles simulation performance.

    Features:
    - Tracks simulation time, netlist build time, analysis time
    - Monitors cache hit/miss rates
    - Records memory usage (if psutil available)
    - Identifies performance bottlenecks
    - Generates detailed profiling reports
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {
            "track_memory": True,
            "max_records": 1000,
            "auto_report": False,
            "report_threshold_ms": 100,
        }
        self._records: list[TimingRecord] = []
        self._stats: dict[str, ProfileStats] = {}
        self._active_records: dict[str, TimingRecord] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._session_start: datetime | None = None

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="simulation-profiler",
            version="1.0.0",
            description="Profile simulation performance and identify bottlenecks",
            author="SpiceLab Team",
            plugin_type=PluginType.GENERIC,
            keywords=["profiling", "performance", "optimization", "debug"],
        )

    def configure(self, settings: dict[str, Any]) -> None:
        """Configure the profiler."""
        self._config.update(settings)

    def activate(self) -> None:
        """Activate the profiler."""
        self._session_start = datetime.now()
        self._register_hooks()

    def deactivate(self) -> None:
        """Deactivate and optionally print report."""
        if self._config["auto_report"]:
            print(self.get_report())

    def _register_hooks(self) -> None:
        """Register profiling hooks."""
        hook_manager = HookManager.get_instance()

        # Simulation timing
        hook_manager.register_hook(
            HookType.PRE_SIMULATION,
            self._on_pre_simulation,
            priority=HookPriority.CRITICAL,
            plugin_name=self.name,
            description="Start simulation timer",
        )
        hook_manager.register_hook(
            HookType.POST_SIMULATION,
            self._on_post_simulation,
            priority=HookPriority.LOWEST,
            plugin_name=self.name,
            description="Stop simulation timer",
        )

        # Netlist timing
        hook_manager.register_hook(
            HookType.PRE_NETLIST_BUILD,
            self._on_pre_netlist,
            priority=HookPriority.CRITICAL,
            plugin_name=self.name,
            description="Start netlist build timer",
        )
        hook_manager.register_hook(
            HookType.POST_NETLIST_BUILD,
            self._on_post_netlist,
            priority=HookPriority.LOWEST,
            plugin_name=self.name,
            description="Stop netlist build timer",
        )

        # Analysis timing
        hook_manager.register_hook(
            HookType.PRE_ANALYSIS,
            self._on_pre_analysis,
            priority=HookPriority.CRITICAL,
            plugin_name=self.name,
            description="Start analysis timer",
        )
        hook_manager.register_hook(
            HookType.POST_ANALYSIS,
            self._on_post_analysis,
            priority=HookPriority.LOWEST,
            plugin_name=self.name,
            description="Stop analysis timer",
        )

        # Cache tracking
        hook_manager.register_hook(
            HookType.CACHE_HIT,
            self._on_cache_hit,
            plugin_name=self.name,
            description="Track cache hits",
        )
        hook_manager.register_hook(
            HookType.CACHE_MISS,
            self._on_cache_miss,
            plugin_name=self.name,
            description="Track cache misses",
        )

        # Validation timing
        hook_manager.register_hook(
            HookType.PRE_VALIDATION,
            self._on_pre_validation,
            priority=HookPriority.CRITICAL,
            plugin_name=self.name,
            description="Start validation timer",
        )
        hook_manager.register_hook(
            HookType.POST_VALIDATION,
            self._on_post_validation,
            priority=HookPriority.LOWEST,
            plugin_name=self.name,
            description="Stop validation timer",
        )

    def _start_timing(self, operation: str, **metadata: Any) -> None:
        """Start timing an operation."""
        record = TimingRecord(
            operation=operation,
            start_time=time.perf_counter(),
            metadata=metadata,
        )
        self._active_records[operation] = record

    def _stop_timing(self, operation: str) -> TimingRecord | None:
        """Stop timing an operation and record it."""
        record = self._active_records.pop(operation, None)
        if record:
            record.complete()
            self._add_record(record)
        return record

    def _add_record(self, record: TimingRecord) -> None:
        """Add a timing record and update stats."""
        # Add to records list (with limit)
        if len(self._records) >= self._config["max_records"]:
            self._records.pop(0)
        self._records.append(record)

        # Update stats
        if record.operation not in self._stats:
            self._stats[record.operation] = ProfileStats(operation=record.operation)
        self._stats[record.operation].add_sample(record.duration_ms)

    # Hook handlers

    def _on_pre_simulation(self, **kwargs: Any) -> None:
        circuit = kwargs.get("circuit")
        circuit_name = getattr(circuit, "name", "unknown") if circuit else "unknown"
        self._start_timing("simulation", circuit=circuit_name)

    def _on_post_simulation(self, **kwargs: Any) -> None:
        record = self._stop_timing("simulation")
        if record and self._config["auto_report"]:
            threshold = self._config["report_threshold_ms"]
            if record.duration_ms > threshold:
                print(f"[Profiler] Slow simulation: {record.duration_ms:.2f}ms")

    def _on_pre_netlist(self, **kwargs: Any) -> None:
        self._start_timing("netlist_build")

    def _on_post_netlist(self, **kwargs: Any) -> None:
        self._stop_timing("netlist_build")

    def _on_pre_analysis(self, **kwargs: Any) -> None:
        analysis_type = kwargs.get("analysis_type", "unknown")
        self._start_timing("analysis", analysis_type=analysis_type)

    def _on_post_analysis(self, **kwargs: Any) -> None:
        self._stop_timing("analysis")

    def _on_pre_validation(self, **kwargs: Any) -> None:
        self._start_timing("validation")

    def _on_post_validation(self, **kwargs: Any) -> None:
        self._stop_timing("validation")

    def _on_cache_hit(self, **kwargs: Any) -> None:
        self._cache_hits += 1

    def _on_cache_miss(self, **kwargs: Any) -> None:
        self._cache_misses += 1

    # Public API

    def get_stats(self) -> dict[str, ProfileStats]:
        """Get aggregated statistics."""
        return self._stats.copy()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": total,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
        }

    def get_recent_records(self, n: int = 10) -> list[TimingRecord]:
        """Get most recent timing records."""
        return self._records[-n:]

    def get_slowest(self, operation: str | None = None, n: int = 5) -> list[TimingRecord]:
        """Get slowest operations."""
        records = self._records
        if operation:
            records = [r for r in records if r.operation == operation]
        return sorted(records, key=lambda r: r.duration_ms, reverse=True)[:n]

    def get_report(self) -> str:
        """Generate a profiling report."""
        lines = [
            "=" * 60,
            "SIMULATION PROFILER REPORT",
            "=" * 60,
        ]

        if self._session_start:
            duration = datetime.now() - self._session_start
            lines.append(f"Session duration: {duration}")

        lines.append("")
        lines.append("TIMING STATISTICS")
        lines.append("-" * 60)
        lines.append(
            f"{'Operation':<20} {'Count':<8} {'Total(ms)':<12} {'Avg(ms)':<10} {'Max(ms)':<10}"
        )
        lines.append("-" * 60)

        for op, stats in sorted(self._stats.items()):
            lines.append(
                f"{op:<20} {stats.count:<8} {stats.total_ms:<12.2f} "
                f"{stats.avg_ms:<10.2f} {stats.max_ms:<10.2f}"
            )

        # Cache stats
        cache = self.get_cache_stats()
        if cache["total"] > 0:
            lines.append("")
            lines.append("CACHE STATISTICS")
            lines.append("-" * 60)
            lines.append(f"Hits:     {cache['hits']}")
            lines.append(f"Misses:   {cache['misses']}")
            lines.append(f"Hit Rate: {cache['hit_rate']:.1%}")

        # Slowest operations
        slowest = self.get_slowest(n=5)
        if slowest:
            lines.append("")
            lines.append("SLOWEST OPERATIONS")
            lines.append("-" * 60)
            for record in slowest:
                lines.append(f"{record.operation}: {record.duration_ms:.2f}ms")

        lines.append("=" * 60)

        return "\n".join(lines)

    def export_profile(self, path: str | Path) -> None:
        """Export profiling data to JSON."""
        path = Path(path)

        data = {
            "session_start": self._session_start.isoformat() if self._session_start else None,
            "export_time": datetime.now().isoformat(),
            "stats": {
                op: {
                    "count": stats.count,
                    "total_ms": stats.total_ms,
                    "avg_ms": stats.avg_ms,
                    "min_ms": stats.min_ms if stats.min_ms != float("inf") else None,
                    "max_ms": stats.max_ms,
                }
                for op, stats in self._stats.items()
            },
            "cache": self.get_cache_stats(),
            "records": [
                {
                    "operation": r.operation,
                    "duration_ms": r.duration_ms,
                    "metadata": r.metadata,
                }
                for r in self._records
            ],
        }

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def reset(self) -> None:
        """Reset all profiling data."""
        self._records.clear()
        self._stats.clear()
        self._active_records.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._session_start = datetime.now()

    def get_memory_usage(self) -> dict[str, Any] | None:
        """Get current memory usage (requires psutil)."""
        try:
            import psutil

            process = psutil.Process()
            mem = process.memory_info()
            return {
                "rss_mb": mem.rss / (1024 * 1024),
                "vms_mb": mem.vms / (1024 * 1024),
                "percent": process.memory_percent(),
            }
        except ImportError:
            return None
