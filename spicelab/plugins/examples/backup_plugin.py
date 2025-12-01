"""Auto-Backup Plugin.

This plugin automatically backs up circuits before simulation,
providing version history and recovery capabilities.

Usage::

    from spicelab.plugins import PluginManager
    from spicelab.plugins.examples import AutoBackupPlugin

    manager = PluginManager()
    plugin = manager.loader.load_from_class(AutoBackupPlugin)
    manager.registry.register(plugin)
    manager.activate_plugin("auto-backup")

    # Configure backup settings
    manager.set_plugin_settings("auto-backup", {
        "backup_dir": "./backups",
        "max_backups": 50,
        "compress": True,
        "backup_on_error": True,
    })

    # Run simulations - backups happen automatically

    # List available backups
    backups = plugin.list_backups()

    # Restore a previous version
    circuit = plugin.restore("my_circuit_20240101_120000")
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..base import Plugin, PluginMetadata, PluginType
from ..hooks import HookManager, HookPriority, HookType


@dataclass
class BackupInfo:
    """Information about a backup."""

    name: str
    circuit_name: str
    timestamp: datetime
    path: Path
    size_bytes: int
    compressed: bool
    hash: str
    metadata: dict[str, Any]

    def __str__(self) -> str:
        size_kb = self.size_bytes / 1024
        return f"{self.name} ({self.circuit_name}) - {self.timestamp.isoformat()} - {size_kb:.1f}KB"


class AutoBackupPlugin(Plugin):
    """Plugin that automatically backs up circuits.

    Features:
    - Auto-backup before each simulation
    - Configurable backup directory and retention
    - Optional compression (gzip)
    - Deduplication using content hashing
    - Easy restore from any backup
    - Backup on simulation error
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {
            "backup_dir": "./circuit_backups",
            "max_backups": 50,
            "compress": True,
            "backup_on_error": True,
            "deduplicate": True,
            "include_metadata": True,
        }
        self._backup_index: dict[str, BackupInfo] = {}
        self._hash_cache: dict[str, str] = {}  # hash -> backup_name

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="auto-backup",
            version="1.0.0",
            description="Automatic circuit backup with version history",
            author="SpiceLab Team",
            plugin_type=PluginType.GENERIC,
            keywords=["backup", "version-control", "recovery", "history"],
        )

    def configure(self, settings: dict[str, Any]) -> None:
        """Configure backup settings."""
        self._config.update(settings)
        self._ensure_backup_dir()

    def activate(self) -> None:
        """Activate the plugin."""
        self._ensure_backup_dir()
        self._load_index()
        self._register_hooks()

    def deactivate(self) -> None:
        """Deactivate the plugin."""
        self._save_index()

    def _ensure_backup_dir(self) -> None:
        """Ensure backup directory exists."""
        Path(self._config["backup_dir"]).mkdir(parents=True, exist_ok=True)

    def _get_backup_dir(self) -> Path:
        """Get backup directory path."""
        return Path(self._config["backup_dir"])

    def _index_path(self) -> Path:
        """Get path to backup index file."""
        return self._get_backup_dir() / ".backup_index.json"

    def _load_index(self) -> None:
        """Load backup index from disk."""
        index_path = self._index_path()
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                for name, info in data.get("backups", {}).items():
                    self._backup_index[name] = BackupInfo(
                        name=name,
                        circuit_name=info["circuit_name"],
                        timestamp=datetime.fromisoformat(info["timestamp"]),
                        path=Path(info["path"]),
                        size_bytes=info["size_bytes"],
                        compressed=info["compressed"],
                        hash=info["hash"],
                        metadata=info.get("metadata", {}),
                    )
                self._hash_cache = data.get("hash_cache", {})
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_index(self) -> None:
        """Save backup index to disk."""
        data = {
            "backups": {
                name: {
                    "circuit_name": info.circuit_name,
                    "timestamp": info.timestamp.isoformat(),
                    "path": str(info.path),
                    "size_bytes": info.size_bytes,
                    "compressed": info.compressed,
                    "hash": info.hash,
                    "metadata": info.metadata,
                }
                for name, info in self._backup_index.items()
            },
            "hash_cache": self._hash_cache,
        }
        self._index_path().write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _register_hooks(self) -> None:
        """Register backup hooks."""
        hook_manager = HookManager.get_instance()

        hook_manager.register_hook(
            HookType.PRE_SIMULATION,
            self._on_pre_simulation,
            priority=HookPriority.HIGH,
            plugin_name=self.name,
            description="Backup circuit before simulation",
        )

        if self._config["backup_on_error"]:
            hook_manager.register_hook(
                HookType.SIMULATION_ERROR,
                self._on_simulation_error,
                plugin_name=self.name,
                description="Backup circuit on simulation error",
            )

    def _on_pre_simulation(self, **kwargs: Any) -> None:
        """Backup circuit before simulation."""
        circuit = kwargs.get("circuit")
        if circuit:
            self.backup(circuit, reason="pre_simulation")

    def _on_simulation_error(self, **kwargs: Any) -> None:
        """Backup circuit on error."""
        circuit = kwargs.get("circuit")
        error = kwargs.get("error")
        if circuit:
            self.backup(circuit, reason="error", error_msg=str(error))

    def _compute_hash(self, content: str) -> str:
        """Compute hash of content for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _cleanup_old_backups(self) -> None:
        """Remove old backups exceeding max_backups limit."""
        max_backups = self._config["max_backups"]
        if len(self._backup_index) <= max_backups:
            return

        # Sort by timestamp, oldest first
        sorted_backups = sorted(
            self._backup_index.items(), key=lambda x: x[1].timestamp
        )

        # Remove oldest backups
        to_remove = len(sorted_backups) - max_backups
        for name, info in sorted_backups[:to_remove]:
            try:
                if info.path.exists():
                    info.path.unlink()
                del self._backup_index[name]
                # Remove from hash cache if present
                if info.hash in self._hash_cache:
                    if self._hash_cache[info.hash] == name:
                        del self._hash_cache[info.hash]
            except Exception:
                pass

    # Public API

    def backup(
        self,
        circuit: Any,
        *,
        reason: str = "manual",
        error_msg: str | None = None,
    ) -> BackupInfo | None:
        """Create a backup of a circuit.

        Args:
            circuit: Circuit to backup
            reason: Reason for backup
            error_msg: Error message if backing up due to error

        Returns:
            BackupInfo if backup created, None if skipped (duplicate)
        """
        circuit_name = getattr(circuit, "name", "unknown")

        # Get netlist content
        try:
            netlist = circuit.build_netlist()
        except Exception:
            netlist = str(circuit)

        # Check for duplicate
        content_hash = self._compute_hash(netlist)
        if self._config["deduplicate"] and content_hash in self._hash_cache:
            # Already have this exact circuit backed up
            return None

        # Generate backup name
        timestamp = datetime.now()
        backup_name = f"{circuit_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        # Determine file extension
        ext = ".cir.gz" if self._config["compress"] else ".cir"
        backup_path = self._get_backup_dir() / f"{backup_name}{ext}"

        # Write backup
        if self._config["compress"]:
            with gzip.open(backup_path, "wt", encoding="utf-8") as f:
                f.write(netlist)
        else:
            backup_path.write_text(netlist, encoding="utf-8")

        # Build metadata
        metadata: dict[str, Any] = {
            "reason": reason,
            "component_count": len(getattr(circuit, "_components", [])),
        }
        if error_msg:
            metadata["error"] = error_msg

        # Create backup info
        info = BackupInfo(
            name=backup_name,
            circuit_name=circuit_name,
            timestamp=timestamp,
            path=backup_path,
            size_bytes=backup_path.stat().st_size,
            compressed=self._config["compress"],
            hash=content_hash,
            metadata=metadata,
        )

        # Update index
        self._backup_index[backup_name] = info
        self._hash_cache[content_hash] = backup_name

        # Cleanup old backups
        self._cleanup_old_backups()

        # Save index
        self._save_index()

        return info

    def list_backups(
        self,
        circuit_name: str | None = None,
        limit: int | None = None,
    ) -> list[BackupInfo]:
        """List available backups.

        Args:
            circuit_name: Filter by circuit name
            limit: Maximum number to return

        Returns:
            List of BackupInfo, newest first
        """
        backups = list(self._backup_index.values())

        if circuit_name:
            backups = [b for b in backups if b.circuit_name == circuit_name]

        # Sort by timestamp, newest first
        backups.sort(key=lambda b: b.timestamp, reverse=True)

        if limit:
            backups = backups[:limit]

        return backups

    def get_backup(self, name: str) -> BackupInfo | None:
        """Get a specific backup by name."""
        return self._backup_index.get(name)

    def restore(self, name: str) -> Any:
        """Restore a circuit from backup.

        Args:
            name: Backup name to restore

        Returns:
            Restored Circuit object

        Raises:
            KeyError: If backup not found
            FileNotFoundError: If backup file missing
        """
        info = self._backup_index.get(name)
        if not info:
            raise KeyError(f"Backup '{name}' not found")

        if not info.path.exists():
            raise FileNotFoundError(f"Backup file missing: {info.path}")

        # Read netlist
        if info.compressed:
            with gzip.open(info.path, "rt", encoding="utf-8") as f:
                netlist = f.read()
        else:
            netlist = info.path.read_text(encoding="utf-8")

        # Parse and return circuit
        from spicelab.io.parser import parse_netlist

        return parse_netlist(netlist)

    def restore_netlist(self, name: str) -> str:
        """Get the netlist content from a backup.

        Args:
            name: Backup name

        Returns:
            Netlist string
        """
        info = self._backup_index.get(name)
        if not info:
            raise KeyError(f"Backup '{name}' not found")

        if info.compressed:
            with gzip.open(info.path, "rt", encoding="utf-8") as f:
                return f.read()
        return info.path.read_text(encoding="utf-8")

    def delete_backup(self, name: str) -> bool:
        """Delete a specific backup.

        Args:
            name: Backup name to delete

        Returns:
            True if deleted, False if not found
        """
        info = self._backup_index.get(name)
        if not info:
            return False

        try:
            if info.path.exists():
                info.path.unlink()
            del self._backup_index[name]
            if info.hash in self._hash_cache:
                del self._hash_cache[info.hash]
            self._save_index()
            return True
        except Exception:
            return False

    def clear_backups(self, circuit_name: str | None = None) -> int:
        """Clear all backups.

        Args:
            circuit_name: Only clear backups for this circuit

        Returns:
            Number of backups deleted
        """
        to_delete = []
        for name, info in self._backup_index.items():
            if circuit_name is None or info.circuit_name == circuit_name:
                to_delete.append(name)

        count = 0
        for name in to_delete:
            if self.delete_backup(name):
                count += 1

        return count

    def export_backup(self, name: str, dest_path: str | Path) -> Path:
        """Export a backup to a specific location.

        Args:
            name: Backup name
            dest_path: Destination path

        Returns:
            Path to exported file
        """
        info = self._backup_index.get(name)
        if not info:
            raise KeyError(f"Backup '{name}' not found")

        dest = Path(dest_path)
        shutil.copy2(info.path, dest)
        return dest

    def get_stats(self) -> dict[str, Any]:
        """Get backup statistics."""
        total_size = sum(info.size_bytes for info in self._backup_index.values())
        circuits = set(info.circuit_name for info in self._backup_index.values())

        return {
            "total_backups": len(self._backup_index),
            "total_size_mb": total_size / (1024 * 1024),
            "unique_circuits": len(circuits),
            "circuits": list(circuits),
            "compressed": sum(1 for info in self._backup_index.values() if info.compressed),
            "oldest": (
                min((info.timestamp for info in self._backup_index.values()), default=None)
            ),
            "newest": (
                max((info.timestamp for info in self._backup_index.values()), default=None)
            ),
        }
