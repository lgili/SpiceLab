"""Plugin system for SpiceLab.

This module provides a comprehensive plugin architecture for extending SpiceLab:
- PluginManager: Central manager for loading/unloading plugins
- Plugin protocols: ComponentPlugin, EnginePlugin, MeasurementPlugin, AnalysisPlugin
- HookManager: Event hooks for customizing behavior
- PluginMetadata: Plugin information and dependencies
- PluginMarketplace: Search, install, and manage plugins
- Testing utilities: Harnesses, fixtures, and validators

Example::

    from spicelab.plugins import PluginManager, ComponentPlugin

    # Load all installed plugins
    manager = PluginManager()
    manager.discover()

    # Get a specific plugin
    memristor_plugin = manager.get_plugin("spicelab-memristor")

    # Use hooks
    from spicelab.plugins import HookManager, HookType

    @HookManager.register(HookType.PRE_SIMULATION)
    def my_hook(circuit, analyses):
        print(f"About to simulate {circuit.name}")

    # Use marketplace
    from spicelab.plugins import PluginMarketplace

    marketplace = PluginMarketplace()
    results = marketplace.search("memristor")
    marketplace.install("spicelab-memristor")

"""

from .base import (
    Plugin,
    PluginDependency,
    PluginMetadata,
    PluginRegistry,
    PluginState,
    PluginType,
)
from .hooks import (
    DisableHooks,
    EnableOnlyHooks,
    Hook,
    HookManager,
    HookPriority,
    HookType,
)
from .loader import (
    PluginEntry,
    PluginLoader,
    discover_plugins,
    load_all_plugins,
    load_plugin,
)
from .manager import (
    PluginManager,
    PluginManagerConfig,
)
from .marketplace import (
    InstallResult,
    MarketplacePluginInfo,
    PluginMarketplace,
    install_plugin,
    list_installed_plugins,
    search_plugins,
)
from .protocols import (
    AnalysisPlugin,
    ComponentPlugin,
    EnginePlugin,
    ExportPlugin,
    ImportPlugin,
    MeasurementPlugin,
    VisualizationPlugin,
)
from .testing import (
    MockPluginFactory,
    PluginTestHarness,
    PluginValidator,
    ValidationResult,
    isolated_plugin_env,
    validate_plugin,
)

__all__ = [
    # Base
    "Plugin",
    "PluginDependency",
    "PluginMetadata",
    "PluginRegistry",
    "PluginState",
    "PluginType",
    # Hooks
    "DisableHooks",
    "EnableOnlyHooks",
    "Hook",
    "HookManager",
    "HookPriority",
    "HookType",
    # Loader
    "PluginEntry",
    "PluginLoader",
    "discover_plugins",
    "load_all_plugins",
    "load_plugin",
    # Manager
    "PluginManager",
    "PluginManagerConfig",
    # Marketplace
    "InstallResult",
    "MarketplacePluginInfo",
    "PluginMarketplace",
    "install_plugin",
    "list_installed_plugins",
    "search_plugins",
    # Protocols
    "AnalysisPlugin",
    "ComponentPlugin",
    "EnginePlugin",
    "ExportPlugin",
    "ImportPlugin",
    "MeasurementPlugin",
    "VisualizationPlugin",
    # Testing
    "MockPluginFactory",
    "PluginTestHarness",
    "PluginValidator",
    "ValidationResult",
    "isolated_plugin_env",
    "validate_plugin",
]
