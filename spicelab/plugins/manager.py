"""Plugin manager for SpiceLab.

This module provides the central PluginManager class that handles
the complete plugin lifecycle: discovery, loading, activation,
deactivation, and unloading.

Example::

    from spicelab.plugins import PluginManager

    # Create manager and discover plugins
    manager = PluginManager()
    manager.discover()

    # Activate all discovered plugins
    manager.activate_all()

    # Get a specific plugin
    memristor = manager.get_plugin("spicelab-memristor")

    # Deactivate and unload
    manager.deactivate_all()
    manager.unload_all()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import Plugin, PluginRegistry, PluginState, PluginType
from .hooks import HookManager, HookType
from .loader import PluginLoader

logger = logging.getLogger(__name__)


@dataclass
class PluginManagerConfig:
    """Configuration for the plugin manager.

    Attributes:
        auto_discover: Automatically discover plugins on init
        auto_activate: Automatically activate plugins after loading
        entry_point_groups: Entry point groups to search
        disabled_plugins: Set of plugin names to not load
        plugin_settings: Per-plugin configuration settings
        settings_file: Path to settings file for persistence
    """

    auto_discover: bool = True
    auto_activate: bool = False
    entry_point_groups: list[str] = field(default_factory=lambda: ["spicelab.plugins"])
    disabled_plugins: set[str] = field(default_factory=set)
    plugin_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    settings_file: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "auto_discover": self.auto_discover,
            "auto_activate": self.auto_activate,
            "entry_point_groups": self.entry_point_groups,
            "disabled_plugins": list(self.disabled_plugins),
            "plugin_settings": self.plugin_settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginManagerConfig:
        """Create config from dictionary."""
        return cls(
            auto_discover=data.get("auto_discover", True),
            auto_activate=data.get("auto_activate", False),
            entry_point_groups=data.get("entry_point_groups", ["spicelab.plugins"]),
            disabled_plugins=set(data.get("disabled_plugins", [])),
            plugin_settings=data.get("plugin_settings", {}),
        )


class PluginManager:
    """Central manager for SpiceLab plugins.

    Handles the complete plugin lifecycle:
    1. Discovery - Find available plugins via entry points
    2. Loading - Import plugin modules and instantiate classes
    3. Activation - Initialize plugins and register their functionality
    4. Deactivation - Unregister functionality before unloading
    5. Unloading - Remove plugins from memory

    Example::

        manager = PluginManager()

        # Discovery and loading
        manager.discover()  # Find all plugins
        manager.load_all()  # Load all discovered plugins

        # Or load specific plugins
        manager.load_plugin("my-plugin")

        # Activation
        manager.activate_all()  # Activate all loaded plugins
        # Or
        manager.activate_plugin("my-plugin")

        # Query plugins
        plugin = manager.get_plugin("my-plugin")
        active = manager.list_active()
        by_type = manager.get_plugins_by_type(PluginType.COMPONENT)

        # Cleanup
        manager.deactivate_plugin("my-plugin")
        manager.unload_plugin("my-plugin")

    Attributes:
        config: Manager configuration
        registry: Plugin registry
        loader: Plugin loader
        hook_manager: Hook manager instance
    """

    _instance: PluginManager | None = None

    def __init__(
        self,
        config: PluginManagerConfig | None = None,
        singleton: bool = True,
    ) -> None:
        """Initialize the plugin manager.

        Args:
            config: Manager configuration
            singleton: If True, store as singleton instance
        """
        self.config = config or PluginManagerConfig()
        self.registry = PluginRegistry()
        self.loader = PluginLoader(entry_point_groups=self.config.entry_point_groups)
        self.hook_manager = HookManager.get_instance()
        self._dependency_order: list[str] = []

        if singleton:
            PluginManager._instance = self

        # Load settings if file exists
        if self.config.settings_file and self.config.settings_file.exists():
            self.load_settings()

        # Auto-discover if configured
        if self.config.auto_discover:
            self.discover()

    @classmethod
    def get_instance(cls) -> PluginManager | None:
        """Get the singleton instance if it exists."""
        return cls._instance

    def discover(self) -> list[str]:
        """Discover available plugins.

        Returns:
            List of discovered plugin names
        """
        entries = self.loader.discover_entries()
        names = [e.name for e in entries]
        logger.info(f"Discovered {len(names)} plugins: {names}")
        return names

    def load_plugin(self, name: str) -> Plugin | None:
        """Load a plugin by name.

        Args:
            name: Plugin name (entry point name)

        Returns:
            Loaded plugin, or None if failed
        """
        # Check if disabled
        if name in self.config.disabled_plugins:
            logger.info(f"Plugin {name} is disabled, not loading")
            return None

        # Load via loader
        plugin = self.loader.load_plugin(name)
        if plugin is None:
            return None

        # Register in our registry
        try:
            self.registry.register(plugin)
        except ValueError:
            # Already registered
            pass

        # Trigger hook
        self.hook_manager._trigger(
            HookType.PLUGIN_LOADED,
            plugin=plugin,
            manager=self,
        )

        # Auto-activate if configured
        if self.config.auto_activate:
            self.activate_plugin(name)

        return plugin

    def load_all(self) -> list[Plugin]:
        """Load all discovered plugins.

        Returns:
            List of loaded plugins
        """
        plugins = []
        for name in self.loader.list_discovered():
            if name not in self.config.disabled_plugins:
                plugin = self.load_plugin(name)
                if plugin:
                    plugins.append(plugin)
        return plugins

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin.

        Args:
            name: Plugin name

        Returns:
            True if unloaded
        """
        plugin = self.registry.get(name)
        if plugin is None:
            return False

        # Deactivate first if active
        if plugin.state == PluginState.ACTIVE:
            self.deactivate_plugin(name)

        # Unload
        plugin.unload()
        self.registry.unregister(name)
        self.loader.unload_plugin(name)

        logger.info(f"Unloaded plugin: {name}")
        return True

    def unload_all(self) -> int:
        """Unload all plugins.

        Returns:
            Number of plugins unloaded
        """
        count = 0
        for name in list(self.registry.list_names()):
            if self.unload_plugin(name):
                count += 1
        return count

    def activate_plugin(self, name: str) -> bool:
        """Activate a plugin.

        Args:
            name: Plugin name

        Returns:
            True if activated
        """
        plugin = self.registry.get(name)
        if plugin is None:
            # Try to load first
            plugin = self.load_plugin(name)
            if plugin is None:
                logger.error(f"Cannot activate {name}: not found")
                return False

        if plugin.state == PluginState.ACTIVE:
            logger.debug(f"Plugin {name} is already active")
            return True

        # Check dependencies
        if not self._check_dependencies(plugin):
            logger.error(f"Cannot activate {name}: dependencies not satisfied")
            return False

        try:
            # Apply plugin settings if available
            settings = self.config.plugin_settings.get(name, {})
            if settings and hasattr(plugin, "configure"):
                plugin.configure(settings)

            # Activate
            plugin.activate()
            plugin._state = PluginState.ACTIVE

            # Trigger hook
            self.hook_manager._trigger(
                HookType.PLUGIN_ACTIVATED,
                plugin=plugin,
                manager=self,
            )

            logger.info(f"Activated plugin: {name}")
            return True

        except Exception as e:
            plugin.set_error(str(e))
            self.hook_manager._trigger(
                HookType.PLUGIN_ERROR,
                plugin=plugin,
                error=e,
                manager=self,
            )
            logger.error(f"Failed to activate {name}: {e}")
            return False

    def activate_all(self) -> int:
        """Activate all loaded plugins in dependency order.

        Returns:
            Number of plugins activated
        """
        # Build dependency order
        self._build_dependency_order()

        count = 0
        for name in self._dependency_order:
            if self.activate_plugin(name):
                count += 1

        # Activate any remaining (no dependencies specified)
        for name in self.registry.list_names():
            if name not in self._dependency_order:
                plugin = self.registry.get(name)
                if plugin and plugin.state != PluginState.ACTIVE:
                    if self.activate_plugin(name):
                        count += 1

        return count

    def deactivate_plugin(self, name: str) -> bool:
        """Deactivate a plugin.

        Args:
            name: Plugin name

        Returns:
            True if deactivated
        """
        plugin = self.registry.get(name)
        if plugin is None:
            return False

        if plugin.state != PluginState.ACTIVE:
            logger.debug(f"Plugin {name} is not active")
            return True

        try:
            plugin.deactivate()
            plugin._state = PluginState.LOADED

            # Unregister plugin's hooks
            self.hook_manager.unregister_plugin_hooks(name)

            # Trigger hook
            self.hook_manager._trigger(
                HookType.PLUGIN_DEACTIVATED,
                plugin=plugin,
                manager=self,
            )

            logger.info(f"Deactivated plugin: {name}")
            return True

        except Exception as e:
            plugin.set_error(str(e))
            logger.error(f"Failed to deactivate {name}: {e}")
            return False

    def deactivate_all(self) -> int:
        """Deactivate all active plugins in reverse dependency order.

        Returns:
            Number of plugins deactivated
        """
        count = 0

        # Deactivate in reverse dependency order
        for name in reversed(self._dependency_order):
            if self.deactivate_plugin(name):
                count += 1

        # Deactivate any remaining
        for name in self.registry.list_names():
            plugin = self.registry.get(name)
            if plugin and plugin.state == PluginState.ACTIVE:
                if self.deactivate_plugin(name):
                    count += 1

        return count

    def _check_dependencies(self, plugin: Plugin) -> bool:
        """Check if plugin dependencies are satisfied.

        Args:
            plugin: Plugin to check

        Returns:
            True if all dependencies are satisfied
        """
        for dep_name in plugin.metadata.dependencies:
            dep_plugin = self.registry.get(dep_name)
            if dep_plugin is None:
                logger.warning(f"Dependency {dep_name} not found for {plugin.name}")
                return False
            if dep_plugin.state != PluginState.ACTIVE:
                logger.warning(f"Dependency {dep_name} not active for {plugin.name}")
                return False
        return True

    def _build_dependency_order(self) -> None:
        """Build topological order for plugin activation."""
        # Simple topological sort
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)

            plugin = self.registry.get(name)
            if plugin:
                for dep in plugin.metadata.dependencies:
                    visit(dep)
                order.append(name)

        for name in self.registry.list_names():
            visit(name)

        self._dependency_order = order

    def get_plugin(self, name: str) -> Plugin | None:
        """Get a plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin if found
        """
        return self.registry.get(name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> list[Plugin]:
        """Get all plugins of a specific type.

        Args:
            plugin_type: Type of plugins to get

        Returns:
            List of plugins
        """
        return self.registry.get_by_type(plugin_type)

    def list_all(self) -> list[str]:
        """List all registered plugin names.

        Returns:
            List of plugin names
        """
        return self.registry.list_names()

    def list_active(self) -> list[str]:
        """List active plugin names.

        Returns:
            List of active plugin names
        """
        return [
            name
            for name in self.registry.list_names()
            if self.registry.get(name).state == PluginState.ACTIVE
        ]

    def list_loaded(self) -> list[str]:
        """List loaded (but maybe not active) plugin names.

        Returns:
            List of loaded plugin names
        """
        return [
            name
            for name in self.registry.list_names()
            if self.registry.get(name).state in (PluginState.LOADED, PluginState.ACTIVE)
        ]

    def list_discovered(self) -> list[str]:
        """List discovered plugin names.

        Returns:
            List of discovered plugin names
        """
        return self.loader.list_discovered()

    def list_failed(self) -> dict[str, str]:
        """List plugins that failed to load.

        Returns:
            Dictionary of plugin name to error message
        """
        failed = self.loader.list_failed()
        # Also include plugins in ERROR state
        for plugin in self.registry:
            if plugin.state == PluginState.ERROR:
                failed[plugin.name] = plugin.error or "Unknown error"
        return failed

    def enable_plugin(self, name: str) -> None:
        """Enable a disabled plugin.

        Args:
            name: Plugin name
        """
        self.config.disabled_plugins.discard(name)

    def disable_plugin(self, name: str) -> None:
        """Disable a plugin (won't be loaded).

        Args:
            name: Plugin name
        """
        # Deactivate and unload if currently loaded
        self.unload_plugin(name)
        self.config.disabled_plugins.add(name)

    def set_plugin_settings(self, name: str, settings: dict[str, Any]) -> None:
        """Set settings for a plugin.

        Args:
            name: Plugin name
            settings: Settings dictionary
        """
        self.config.plugin_settings[name] = settings

        # Apply immediately if plugin is loaded and has configure method
        plugin = self.registry.get(name)
        if plugin and hasattr(plugin, "configure"):
            plugin.configure(settings)

    def get_plugin_settings(self, name: str) -> dict[str, Any]:
        """Get settings for a plugin.

        Args:
            name: Plugin name

        Returns:
            Settings dictionary
        """
        return self.config.plugin_settings.get(name, {})

    def save_settings(self, path: Path | None = None) -> None:
        """Save settings to file.

        Args:
            path: Path to save to, or use config.settings_file
        """
        path = path or self.config.settings_file
        if path is None:
            raise ValueError("No settings file path specified")

        data = self.config.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved plugin settings to {path}")

    def load_settings(self, path: Path | None = None) -> None:
        """Load settings from file.

        Args:
            path: Path to load from, or use config.settings_file
        """
        path = path or self.config.settings_file
        if path is None:
            raise ValueError("No settings file path specified")

        if not path.exists():
            return

        with open(path) as f:
            data = json.load(f)

        self.config = PluginManagerConfig.from_dict(data)
        logger.debug(f"Loaded plugin settings from {path}")

    def get_info(self) -> dict[str, Any]:
        """Get manager information.

        Returns:
            Dictionary with manager state
        """
        return {
            "config": self.config.to_dict(),
            "discovered": self.list_discovered(),
            "loaded": self.list_loaded(),
            "active": self.list_active(),
            "failed": self.list_failed(),
            "plugins": {
                name: plugin.get_info()
                for name, plugin in [(n, self.registry.get(n)) for n in self.registry.list_names()]
                if plugin
            },
        }

    def reload_plugin(self, name: str) -> Plugin | None:
        """Reload a plugin (unload and load again).

        Args:
            name: Plugin name

        Returns:
            Reloaded plugin, or None if failed
        """
        was_active = False
        plugin = self.registry.get(name)
        if plugin:
            was_active = plugin.state == PluginState.ACTIVE
            self.unload_plugin(name)

        # Reload the module
        import importlib
        import sys

        entry = self.loader._entries.get(name)
        if entry:
            module_name = entry.module_path
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])

        # Load again
        plugin = self.load_plugin(name)
        if plugin and was_active:
            self.activate_plugin(name)

        return plugin

    def __repr__(self) -> str:
        return (
            f"<PluginManager discovered={len(self.list_discovered())} "
            f"loaded={len(self.list_loaded())} active={len(self.list_active())}>"
        )
