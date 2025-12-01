"""Plugin loader using Python entry points.

This module handles discovering and loading plugins using Python's
standard entry points mechanism (setuptools/importlib.metadata).

Plugins are discovered from the 'spicelab.plugins' entry point group.

Example pyproject.toml for a plugin::

    [project.entry-points."spicelab.plugins"]
    memristor = "spicelab_memristor:MemristorPlugin"
    qspice = "spicelab_qspice:QspiceEnginePlugin"

Example::

    from spicelab.plugins import PluginLoader

    loader = PluginLoader()

    # Discover all plugins
    plugins = loader.discover()

    # Load a specific plugin
    plugin = loader.load_plugin("spicelab-memristor")

    # Load from a module path
    plugin = loader.load_from_module("my_plugin.core:MyPlugin")
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .base import Plugin, PluginState

logger = logging.getLogger(__name__)

# Default entry point group for SpiceLab plugins
DEFAULT_ENTRY_POINT_GROUP = "spicelab.plugins"


@dataclass
class PluginEntry:
    """Represents a discovered plugin entry point.

    Attributes:
        name: Entry point name
        group: Entry point group
        value: Entry point value (module:class)
        dist_name: Distribution package name
        dist_version: Distribution version
    """

    name: str
    group: str
    value: str
    dist_name: str = ""
    dist_version: str = ""

    @property
    def module_path(self) -> str:
        """Get the module path (before the colon)."""
        if ":" in self.value:
            return self.value.split(":")[0]
        return self.value

    @property
    def class_name(self) -> str:
        """Get the class name (after the colon)."""
        if ":" in self.value:
            return self.value.split(":")[1]
        return ""


class PluginLoader:
    """Loader for discovering and loading SpiceLab plugins.

    Uses Python's entry points mechanism to discover plugins.

    Example::

        loader = PluginLoader()

        # Discover all available plugins
        entries = loader.discover_entries()

        # Load all discovered plugins
        plugins = loader.load_all()

        # Load a specific plugin by name
        plugin = loader.load_plugin("my-plugin")
    """

    def __init__(
        self,
        entry_point_groups: list[str] | None = None,
    ) -> None:
        """Initialize the plugin loader.

        Args:
            entry_point_groups: List of entry point groups to search.
                Defaults to ["spicelab.plugins"]
        """
        self.entry_point_groups = entry_point_groups or [DEFAULT_ENTRY_POINT_GROUP]
        self._entries: dict[str, PluginEntry] = {}
        self._loaded: dict[str, Plugin] = {}
        self._failed: dict[str, str] = {}  # name -> error message

    def discover_entries(self) -> list[PluginEntry]:
        """Discover all plugin entry points.

        Returns:
            List of discovered plugin entries
        """
        entries = []

        try:
            # Python 3.10+ / importlib.metadata
            from importlib.metadata import entry_points
        except ImportError:
            # Fallback for older Python
            try:
                from importlib_metadata import entry_points  # type: ignore
            except ImportError:
                logger.warning("importlib.metadata not available, plugin discovery disabled")
                return []

        for group in self.entry_point_groups:
            try:
                # Python 3.10+ returns SelectableGroups
                eps = entry_points(group=group)
                if hasattr(eps, "select"):
                    eps = eps.select(group=group)
            except TypeError:
                # Python 3.9 compatibility
                all_eps = entry_points()
                eps = all_eps.get(group, [])

            for ep in eps:
                entry = PluginEntry(
                    name=ep.name,
                    group=group,
                    value=ep.value,
                    dist_name=getattr(ep.dist, "name", "") if hasattr(ep, "dist") else "",
                    dist_version=getattr(ep.dist, "version", "") if hasattr(ep, "dist") else "",
                )
                entries.append(entry)
                self._entries[ep.name] = entry
                logger.debug(f"Discovered plugin entry: {ep.name} = {ep.value}")

        return entries

    def load_plugin(self, name: str) -> Plugin | None:
        """Load a plugin by entry point name.

        Args:
            name: Entry point name

        Returns:
            Loaded plugin instance, or None if failed
        """
        # Check if already loaded
        if name in self._loaded:
            return self._loaded[name]

        # Check if we need to discover
        if name not in self._entries:
            self.discover_entries()

        if name not in self._entries:
            logger.error(f"Plugin '{name}' not found in entry points")
            self._failed[name] = "Not found in entry points"
            return None

        entry = self._entries[name]
        return self._load_entry(entry)

    def _load_entry(self, entry: PluginEntry) -> Plugin | None:
        """Load a plugin from an entry point.

        Args:
            entry: Plugin entry to load

        Returns:
            Loaded plugin instance, or None if failed
        """
        try:
            plugin_class = self._import_plugin_class(entry)
            if plugin_class is None:
                return None

            # Instantiate the plugin
            plugin = plugin_class()

            # Validate it's a Plugin subclass
            if not isinstance(plugin, Plugin):
                error = f"Class {entry.value} is not a Plugin subclass"
                logger.error(error)
                self._failed[entry.name] = error
                return None

            # Mark as loaded
            plugin._state = PluginState.LOADED
            self._loaded[entry.name] = plugin

            logger.info(f"Loaded plugin: {entry.name} v{plugin.version}")
            return plugin

        except Exception as e:
            error = f"Failed to load plugin {entry.name}: {e}"
            logger.error(error)
            self._failed[entry.name] = str(e)
            return None

    def _import_plugin_class(self, entry: PluginEntry) -> type[Plugin] | None:
        """Import a plugin class from an entry point.

        Args:
            entry: Plugin entry

        Returns:
            Plugin class, or None if import failed
        """
        try:
            module = importlib.import_module(entry.module_path)
            if entry.class_name:
                return getattr(module, entry.class_name)
            else:
                # If no class specified, look for a Plugin subclass
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, Plugin) and attr is not Plugin:
                        return attr
                raise AttributeError(f"No Plugin subclass found in {entry.module_path}")
        except ImportError as e:
            logger.error(f"Failed to import {entry.module_path}: {e}")
            self._failed[entry.name] = f"Import error: {e}"
            return None
        except AttributeError as e:
            logger.error(f"Failed to get {entry.class_name} from {entry.module_path}: {e}")
            self._failed[entry.name] = f"Attribute error: {e}"
            return None

    def load_from_module(self, module_path: str) -> Plugin | None:
        """Load a plugin from a module path.

        Args:
            module_path: Path in format "module.path:ClassName"

        Returns:
            Loaded plugin instance, or None if failed

        Example::

            plugin = loader.load_from_module("my_plugins.custom:MyPlugin")
        """
        entry = PluginEntry(
            name=module_path,
            group="direct",
            value=module_path,
        )
        return self._load_entry(entry)

    def load_from_class(self, plugin_class: type[Plugin]) -> Plugin:
        """Load a plugin from a class directly.

        Args:
            plugin_class: Plugin class to instantiate

        Returns:
            Plugin instance
        """
        plugin = plugin_class()
        plugin._state = PluginState.LOADED
        self._loaded[plugin.name] = plugin
        return plugin

    def load_all(self) -> list[Plugin]:
        """Load all discovered plugins.

        Returns:
            List of successfully loaded plugins
        """
        if not self._entries:
            self.discover_entries()

        plugins = []
        for name in self._entries:
            plugin = self.load_plugin(name)
            if plugin:
                plugins.append(plugin)

        return plugins

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin.

        Args:
            name: Plugin name

        Returns:
            True if plugin was unloaded
        """
        plugin = self._loaded.pop(name, None)
        if plugin:
            plugin._state = PluginState.DISCOVERED
            logger.info(f"Unloaded plugin: {name}")
            return True
        return False

    def get_loaded(self, name: str) -> Plugin | None:
        """Get a loaded plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin if loaded, None otherwise
        """
        return self._loaded.get(name)

    def list_loaded(self) -> list[str]:
        """List names of loaded plugins.

        Returns:
            List of plugin names
        """
        return list(self._loaded.keys())

    def list_discovered(self) -> list[str]:
        """List names of discovered plugins.

        Returns:
            List of plugin entry point names
        """
        return list(self._entries.keys())

    def list_failed(self) -> dict[str, str]:
        """List plugins that failed to load with their error messages.

        Returns:
            Dictionary of plugin name to error message
        """
        return dict(self._failed)

    def is_loaded(self, name: str) -> bool:
        """Check if a plugin is loaded.

        Args:
            name: Plugin name

        Returns:
            True if loaded
        """
        return name in self._loaded

    def get_info(self) -> dict[str, Any]:
        """Get loader information.

        Returns:
            Dictionary with loader state
        """
        return {
            "entry_point_groups": self.entry_point_groups,
            "discovered": list(self._entries.keys()),
            "loaded": list(self._loaded.keys()),
            "failed": self._failed,
        }

    def clear(self) -> None:
        """Clear all discovered and loaded plugins."""
        self._entries.clear()
        self._loaded.clear()
        self._failed.clear()


def discover_plugins(
    groups: list[str] | None = None,
) -> Iterator[tuple[str, PluginEntry]]:
    """Generator to discover plugin entry points.

    Args:
        groups: Entry point groups to search

    Yields:
        Tuples of (name, PluginEntry)

    Example::

        for name, entry in discover_plugins():
            print(f"Found plugin: {name} ({entry.value})")
    """
    loader = PluginLoader(entry_point_groups=groups)
    entries = loader.discover_entries()
    for entry in entries:
        yield entry.name, entry


def load_plugin(name: str) -> Plugin | None:
    """Convenience function to load a single plugin.

    Args:
        name: Plugin entry point name

    Returns:
        Loaded plugin, or None if failed
    """
    loader = PluginLoader()
    return loader.load_plugin(name)


def load_all_plugins() -> list[Plugin]:
    """Convenience function to load all available plugins.

    Returns:
        List of loaded plugins
    """
    loader = PluginLoader()
    return loader.load_all()
