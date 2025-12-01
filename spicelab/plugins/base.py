"""Base plugin classes and data structures.

This module defines the core plugin infrastructure:
- PluginMetadata: Information about a plugin
- PluginState: Current state of a plugin
- PluginType: Types of plugins supported
- Plugin: Base class for all plugins
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class PluginType(Enum):
    """Types of plugins supported by SpiceLab."""

    COMPONENT = auto()  # Adds new circuit components
    ENGINE = auto()  # Adds simulation engines
    MEASUREMENT = auto()  # Adds measurement functions
    ANALYSIS = auto()  # Adds analysis types
    VISUALIZATION = auto()  # Adds visualization tools
    EXPORT = auto()  # Adds export formats
    IMPORT = auto()  # Adds import formats
    GENERIC = auto()  # Generic plugin type


class PluginState(Enum):
    """Current state of a plugin."""

    DISCOVERED = auto()  # Found but not loaded
    LOADED = auto()  # Loaded into memory
    ACTIVE = auto()  # Fully initialized and active
    DISABLED = auto()  # Explicitly disabled
    ERROR = auto()  # Failed to load or initialize


@dataclass
class PluginMetadata:
    """Metadata describing a plugin.

    Attributes:
        name: Unique identifier for the plugin
        version: Semantic version string (e.g., "1.2.3")
        description: Human-readable description
        author: Plugin author name
        email: Contact email
        url: Project URL
        license: License identifier (e.g., "MIT")
        plugin_type: Type of plugin
        dependencies: List of required plugin names
        spicelab_version: Required SpiceLab version constraint
        entry_point: Entry point group name
        keywords: Search keywords
        classifiers: Package classifiers
    """

    name: str
    version: str
    description: str = ""
    author: str = ""
    email: str = ""
    url: str = ""
    license: str = ""
    plugin_type: PluginType = PluginType.GENERIC
    dependencies: list[str] = field(default_factory=list)
    spicelab_version: str = ">=0.1.0"
    entry_point: str = "spicelab.plugins"
    keywords: list[str] = field(default_factory=list)
    classifiers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "email": self.email,
            "url": self.url,
            "license": self.license,
            "plugin_type": self.plugin_type.name,
            "dependencies": self.dependencies,
            "spicelab_version": self.spicelab_version,
            "entry_point": self.entry_point,
            "keywords": self.keywords,
            "classifiers": self.classifiers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginMetadata:
        """Create metadata from dictionary."""
        plugin_type = data.get("plugin_type", "GENERIC")
        if isinstance(plugin_type, str):
            plugin_type = PluginType[plugin_type]
        return cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "0.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            email=data.get("email", ""),
            url=data.get("url", ""),
            license=data.get("license", ""),
            plugin_type=plugin_type,
            dependencies=data.get("dependencies", []),
            spicelab_version=data.get("spicelab_version", ">=0.1.0"),
            entry_point=data.get("entry_point", "spicelab.plugins"),
            keywords=data.get("keywords", []),
            classifiers=data.get("classifiers", []),
        )


class Plugin(ABC):
    """Base class for all SpiceLab plugins.

    All plugins must inherit from this class and implement the required methods.

    Example::

        class MyPlugin(Plugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="my-plugin",
                    version="1.0.0",
                    description="My custom plugin",
                    plugin_type=PluginType.COMPONENT,
                )

            def activate(self) -> None:
                # Register components, hooks, etc.
                pass

            def deactivate(self) -> None:
                # Cleanup
                pass
    """

    _state: PluginState = PluginState.DISCOVERED
    _error: str | None = None

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        ...

    @property
    def state(self) -> PluginState:
        """Return current plugin state."""
        return self._state

    @property
    def error(self) -> str | None:
        """Return error message if plugin is in ERROR state."""
        return self._error

    @property
    def name(self) -> str:
        """Return plugin name."""
        return self.metadata.name

    @property
    def version(self) -> str:
        """Return plugin version."""
        return self.metadata.version

    @property
    def plugin_type(self) -> PluginType:
        """Return plugin type."""
        return self.metadata.plugin_type

    def load(self) -> None:
        """Load the plugin into memory.

        This is called before activate() and should perform any
        necessary setup that doesn't require other plugins.
        """
        self._state = PluginState.LOADED

    @abstractmethod
    def activate(self) -> None:
        """Activate the plugin.

        This is called after all dependencies are loaded and should
        register any components, hooks, or other functionality.
        """
        ...

    @abstractmethod
    def deactivate(self) -> None:
        """Deactivate the plugin.

        This is called when the plugin is being unloaded and should
        unregister any components, hooks, or other functionality.
        """
        ...

    def unload(self) -> None:
        """Unload the plugin from memory.

        This is called after deactivate() and should perform any
        final cleanup.
        """
        self._state = PluginState.DISCOVERED

    def set_error(self, message: str) -> None:
        """Set plugin to error state with message."""
        self._state = PluginState.ERROR
        self._error = message

    def clear_error(self) -> None:
        """Clear error state."""
        self._error = None
        if self._state == PluginState.ERROR:
            self._state = PluginState.DISCOVERED

    def get_info(self) -> dict[str, Any]:
        """Get plugin information."""
        return {
            "metadata": self.metadata.to_dict(),
            "state": self.state.name,
            "error": self.error,
        }

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"<{cls_name} name={self.name!r} version={self.version!r} state={self.state.name}>"


@dataclass
class PluginDependency:
    """Represents a plugin dependency.

    Attributes:
        name: Name of the required plugin
        version: Version constraint (e.g., ">=1.0.0", "~=2.0")
        optional: Whether the dependency is optional
    """

    name: str
    version: str = "*"
    optional: bool = False

    def is_satisfied_by(self, available_version: str) -> bool:
        """Check if available version satisfies this dependency.

        Args:
            available_version: Version string to check

        Returns:
            True if version satisfies constraint
        """
        # Simple version comparison - could use packaging.version for full support
        if self.version == "*":
            return True

        try:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version

            spec = SpecifierSet(self.version)
            return Version(available_version) in spec
        except ImportError:
            # Fall back to simple string comparison
            if self.version.startswith(">="):
                return available_version >= self.version[2:]
            elif self.version.startswith("<="):
                return available_version <= self.version[2:]
            elif self.version.startswith("=="):
                return available_version == self.version[2:]
            elif self.version.startswith("~="):
                # Compatible release
                base = self.version[2:]
                return available_version.startswith(base.rsplit(".", 1)[0])
            return True


class PluginRegistry:
    """Registry for tracking available plugins.

    This is a simple in-memory registry. For persistence, use PluginManager.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._by_type: dict[PluginType, set[str]] = {t: set() for t in PluginType}

    def register(self, plugin: Plugin) -> None:
        """Register a plugin."""
        name = plugin.name
        if name in self._plugins:
            raise ValueError(f"Plugin {name!r} is already registered")
        self._plugins[name] = plugin
        self._by_type[plugin.plugin_type].add(name)

    def unregister(self, name: str) -> Plugin | None:
        """Unregister a plugin by name."""
        plugin = self._plugins.pop(name, None)
        if plugin:
            self._by_type[plugin.plugin_type].discard(name)
        return plugin

    def get(self, name: str) -> Plugin | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_by_type(self, plugin_type: PluginType) -> list[Plugin]:
        """Get all plugins of a specific type."""
        return [self._plugins[name] for name in self._by_type[plugin_type]]

    def list_all(self) -> list[Plugin]:
        """List all registered plugins."""
        return list(self._plugins.values())

    def list_names(self) -> list[str]:
        """List all plugin names."""
        return list(self._plugins.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._plugins

    def __len__(self) -> int:
        return len(self._plugins)

    def __iter__(self):
        return iter(self._plugins.values())
