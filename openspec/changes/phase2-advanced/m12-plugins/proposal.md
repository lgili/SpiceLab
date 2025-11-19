# M12: Plugin System

**Status:** Proposed
**Priority:** 🟡 MEDIUM
**Estimated Duration:** 8-10 weeks
**Dependencies:** M1 (testing), M3 (components), M7 (measurements)

## Problem Statement

SpiceLab currently has no extensibility mechanism beyond Python inheritance. Users cannot easily add custom component types, analysis methods, simulation engines, or measurement specs without modifying core code. There's no plugin marketplace, no standardized plugin API, and no way for the community to contribute extensions.

### Current Gaps
- ❌ No plugin architecture (entry points, hooks)
- ❌ No plugin marketplace/registry
- ❌ Cannot add custom component types via plugins
- ❌ Cannot add custom analysis types
- ❌ Cannot add custom simulation engines
- ❌ No hooks system (pre/post simulation callbacks)
- ❌ No plugin testing framework
- ❌ No plugin SDK documentation

### Impact
- **Extensibility:** Users must fork or modify core code
- **Community:** No ecosystem of community-contributed extensions
- **Vendor Integration:** Cannot easily integrate vendor tools
- **Innovation:** Limits experimental features

## Objectives

1. **Build plugin architecture** using entry points and import hooks
2. **Create plugin marketplace** with registry and discovery
3. **Enable custom component types** via plugins
4. **Support custom analysis types** through plugin API
5. **Allow custom simulation engines** with standard interface
6. **Implement hooks system** (pre/post simulation, validation, etc.)
7. **Build plugin testing framework** with fixtures
8. **Create plugin SDK** with documentation and templates
9. **Target: 5+ community plugins** within 6 months

## Technical Design

### 1. Plugin Architecture

```python
# spicelab/plugins/base.py
from typing import Protocol, Any
from abc import abstractmethod

class Plugin(Protocol):
    """Base protocol for all plugins."""

    name: str
    version: str
    description: str
    author: str

    @abstractmethod
    def load(self) -> None:
        """Called when plugin is loaded."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Called when plugin is unloaded."""
        ...

class PluginMetadata:
    """Plugin metadata."""

    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        author: str,
        dependencies: list[str] | None = None,
        entry_point: str | None = None,
    ):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.dependencies = dependencies or []
        self.entry_point = entry_point
```

### 2. Plugin Loader with Entry Points

```python
# spicelab/plugins/loader.py
from importlib.metadata import entry_points
import importlib
from typing import Type

class PluginLoader:
    """Load and manage plugins."""

    def __init__(self):
        self.loaded_plugins: dict[str, Plugin] = {}

    def discover_plugins(self) -> list[PluginMetadata]:
        """Discover available plugins via entry points."""
        discovered = []

        # Python 3.10+ entry_points API
        eps = entry_points(group='spicelab.plugins')

        for ep in eps:
            plugin_cls = ep.load()

            metadata = PluginMetadata(
                name=ep.name,
                version=getattr(plugin_cls, '__version__', '0.0.0'),
                description=getattr(plugin_cls, '__doc__', ''),
                author=getattr(plugin_cls, '__author__', 'Unknown'),
                entry_point=f"{ep.module}:{ep.attr}",
            )

            discovered.append(metadata)

        return discovered

    def load_plugin(self, plugin_name: str) -> Plugin:
        """Load a plugin by name."""
        if plugin_name in self.loaded_plugins:
            return self.loaded_plugins[plugin_name]

        # Find entry point
        eps = entry_points(group='spicelab.plugins', name=plugin_name)

        if not eps:
            raise ValueError(f"Plugin '{plugin_name}' not found")

        # Load plugin class
        plugin_cls = eps[0].load()

        # Instantiate and load
        plugin_instance = plugin_cls()
        plugin_instance.load()

        self.loaded_plugins[plugin_name] = plugin_instance

        return plugin_instance

    def unload_plugin(self, plugin_name: str):
        """Unload a plugin."""
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name]
            plugin.unload()
            del self.loaded_plugins[plugin_name]

# Global plugin loader
_loader = PluginLoader()

def load_plugin(name: str) -> Plugin:
    return _loader.load_plugin(name)

def discover_plugins() -> list[PluginMetadata]:
    return _loader.discover_plugins()
```

### 3. Plugin Types

#### Component Plugin
```python
# spicelab/plugins/component.py

class ComponentPlugin(Plugin):
    """Plugin that adds custom components."""

    @abstractmethod
    def get_components(self) -> list[Type[Component]]:
        """Return list of component classes."""
        ...

# Example plugin
class CustomComponentsPlugin:
    """Example plugin adding custom components."""

    name = "custom-components"
    version = "1.0.0"
    description = "Custom component library"
    author = "Example Author"

    def load(self):
        """Register custom components."""
        from spicelab.core.components import ComponentRegistry

        for comp_cls in self.get_components():
            ComponentRegistry.register(comp_cls)

    def unload(self):
        """Unregister components."""
        for comp_cls in self.get_components():
            ComponentRegistry.unregister(comp_cls.component_type)

    def get_components(self):
        return [
            MemristorComponent,
            CustomTransformerComponent,
            # ... more
        ]

class MemristorComponent(Component):
    """Custom memristor component."""

    component_type = "memristor"

    def __init__(self, ref: str, resistance_on: float, resistance_off: float):
        super().__init__(ref)
        self.resistance_on = resistance_on
        self.resistance_off = resistance_off

    def spice_card(self) -> str:
        # Memristor SPICE model
        return f"X{self.ref} ... MEMRISTOR_MODEL"
```

#### Engine Plugin
```python
# spicelab/plugins/engine.py

class EnginePlugin(Plugin):
    """Plugin that adds custom simulation engine."""

    @abstractmethod
    def get_engine(self) -> Type[SimulationEngine]:
        """Return engine class."""
        ...

# Example: QSPICE plugin
class QSPICEPlugin:
    """QSPICE simulation engine plugin."""

    name = "qspice-engine"
    version = "1.0.0"

    def load(self):
        from spicelab.engines.registry import EngineRegistry
        EngineRegistry.register('qspice', self.get_engine())

    def unload(self):
        from spicelab.engines.registry import EngineRegistry
        EngineRegistry.unregister('qspice')

    def get_engine(self):
        return QSPICEEngine

class QSPICEEngine(SimulationEngine):
    """QSPICE simulation engine implementation."""

    def __init__(self, executable_path: Path | None = None):
        super().__init__(name="qspice")
        self.executable_path = executable_path or self._find_qspice()

    def run(self, netlist: str, output_dir: Path) -> SimulationResult:
        # QSPICE-specific implementation
        ...
```

#### Measurement Plugin
```python
# spicelab/plugins/measurement.py

class MeasurementPlugin(Plugin):
    """Plugin that adds custom measurements."""

    @abstractmethod
    def get_measurements(self) -> list[Type[Measurement]]:
        """Return measurement classes."""
        ...

# Example: RF measurements plugin
class RFMeasurementsPlugin:
    """RF-specific measurements."""

    name = "rf-measurements"
    version = "1.0.0"

    def load(self):
        from spicelab.measurements import MeasurementRegistry

        for meas_cls in self.get_measurements():
            MeasurementRegistry.register(meas_cls)

    def get_measurements(self):
        return [
            S11Measurement,
            S21Measurement,
            VSWRMeasurement,
            GroupDelayMeasurement,
        ]
```

### 4. Hooks System

```python
# spicelab/plugins/hooks.py
from typing import Callable, Any
from enum import Enum

class HookType(Enum):
    """Available hook types."""
    PRE_SIMULATION = "pre_simulation"
    POST_SIMULATION = "post_simulation"
    PRE_ANALYSIS = "pre_analysis"
    POST_ANALYSIS = "post_analysis"
    PRE_NETLIST_BUILD = "pre_netlist_build"
    POST_NETLIST_BUILD = "post_netlist_build"
    PRE_VALIDATION = "pre_validation"
    POST_VALIDATION = "post_validation"

class HookManager:
    """Manage plugin hooks."""

    def __init__(self):
        self._hooks: dict[HookType, list[Callable]] = {
            hook_type: [] for hook_type in HookType
        }

    def register(self, hook_type: HookType, callback: Callable):
        """Register a hook callback."""
        self._hooks[hook_type].append(callback)

    def unregister(self, hook_type: HookType, callback: Callable):
        """Unregister a hook callback."""
        if callback in self._hooks[hook_type]:
            self._hooks[hook_type].remove(callback)

    def trigger(self, hook_type: HookType, **kwargs) -> dict[str, Any]:
        """Trigger all callbacks for a hook type."""
        results = {}

        for callback in self._hooks[hook_type]:
            try:
                result = callback(**kwargs)
                results[callback.__name__] = result
            except Exception as e:
                print(f"Hook {callback.__name__} failed: {e}")

        return results

# Global hook manager
_hook_manager = HookManager()

def register_hook(hook_type: HookType, callback: Callable):
    _hook_manager.register(hook_type, callback)

def trigger_hooks(hook_type: HookType, **kwargs):
    return _hook_manager.trigger(hook_type, **kwargs)

# Example usage in simulation code
def run_simulation(circuit, analyses, **kwargs):
    # Trigger pre-simulation hooks
    trigger_hooks(HookType.PRE_SIMULATION, circuit=circuit, analyses=analyses)

    # ... run simulation

    # Trigger post-simulation hooks
    trigger_hooks(HookType.POST_SIMULATION, result=result)

    return result

# Plugin with hooks
class LoggingPlugin:
    """Plugin that logs all simulations."""

    name = "logging"

    def load(self):
        register_hook(HookType.PRE_SIMULATION, self.log_pre_simulation)
        register_hook(HookType.POST_SIMULATION, self.log_post_simulation)

    def log_pre_simulation(self, circuit, analyses, **kwargs):
        print(f"Starting simulation: {circuit.name}")

    def log_post_simulation(self, result, **kwargs):
        print(f"Simulation completed: {result.status}")
```

### 5. Plugin Marketplace/Registry

```python
# spicelab/plugins/marketplace.py
import requests
from pydantic import BaseModel

class PluginPackage(BaseModel):
    """Plugin package metadata."""
    name: str
    version: str
    description: str
    author: str
    homepage: str
    repository: str
    downloads: int
    rating: float
    tags: list[str]

class PluginMarketplace:
    """Plugin marketplace client."""

    def __init__(self, registry_url: str = "https://plugins.spicelab.io/api"):
        self.registry_url = registry_url

    def search(self, query: str = "", tags: list[str] | None = None) -> list[PluginPackage]:
        """Search for plugins."""
        params = {"q": query}
        if tags:
            params["tags"] = ",".join(tags)

        response = requests.get(f"{self.registry_url}/search", params=params)
        response.raise_for_status()

        return [PluginPackage(**pkg) for pkg in response.json()]

    def install(self, plugin_name: str, version: str | None = None):
        """Install a plugin from marketplace."""
        import subprocess

        # Use pip to install
        package = f"spicelab-plugin-{plugin_name}"
        if version:
            package += f"=={version}"

        subprocess.run(["pip", "install", package], check=True)

        print(f"Installed {package}")

    def publish(self, plugin_dir: Path):
        """Publish plugin to marketplace."""
        # Build package
        subprocess.run(["python", "-m", "build"], cwd=plugin_dir, check=True)

        # Upload to registry (via twine or API)
        ...

# CLI commands
# spicelab plugin search "measurement"
# spicelab plugin install rf-measurements
# spicelab plugin list
```

### 6. Plugin Testing Framework

```python
# spicelab/plugins/testing.py
import pytest

class PluginTestHarness:
    """Test harness for plugins."""

    def __init__(self, plugin_cls: Type[Plugin]):
        self.plugin_cls = plugin_cls
        self.plugin = None

    def setup(self):
        """Setup plugin for testing."""
        self.plugin = self.plugin_cls()
        self.plugin.load()

    def teardown(self):
        """Teardown plugin after testing."""
        if self.plugin:
            self.plugin.unload()

    def test_load_unload(self):
        """Test basic load/unload."""
        assert self.plugin is not None

    def test_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.name
        assert self.plugin.version
        assert self.plugin.description

# Pytest fixture
@pytest.fixture
def plugin_harness():
    """Plugin test harness fixture."""
    harness = PluginTestHarness(MyPlugin)
    harness.setup()
    yield harness
    harness.teardown()

def test_custom_component_plugin(plugin_harness):
    """Test custom component plugin."""
    from spicelab.core.components import ComponentRegistry

    # Check component registered
    assert ComponentRegistry.has('memristor')

    # Create component
    m = ComponentRegistry.create('memristor', 'M1', r_on=100, r_off=1e6)
    assert m.ref == 'M1'
```

### 7. Plugin SDK Template

```
spicelab-plugin-template/
├── pyproject.toml
├── README.md
├── src/
│   └── spicelab_plugin_example/
│       ├── __init__.py
│       ├── plugin.py
│       ├── components.py
│       └── measurements.py
├── tests/
│   ├── test_plugin.py
│   └── test_components.py
└── docs/
    └── usage.md
```

```toml
# pyproject.toml
[project]
name = "spicelab-plugin-example"
version = "1.0.0"
description = "Example SpiceLab plugin"
authors = [{name = "Your Name", email = "you@example.com"}]
dependencies = ["spicelab>=0.3.0"]

[project.entry-points."spicelab.plugins"]
example = "spicelab_plugin_example.plugin:ExamplePlugin"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

## Implementation Plan

### Week 1-2: Plugin Architecture
- [ ] Design plugin protocols
- [ ] Implement PluginLoader
- [ ] Add entry points discovery
- [ ] Create plugin metadata model

### Week 3-4: Plugin Types
- [ ] Component plugin interface
- [ ] Engine plugin interface
- [ ] Measurement plugin interface
- [ ] Analysis plugin interface

### Week 5-6: Hooks System
- [ ] Define hook types
- [ ] Implement HookManager
- [ ] Integrate hooks into core
- [ ] Document hook usage

### Week 7: Plugin Marketplace
- [ ] Design registry API
- [ ] Implement marketplace client
- [ ] Add search/install/publish
- [ ] Create web UI for marketplace

### Week 8: Testing Framework
- [ ] Create PluginTestHarness
- [ ] Add pytest fixtures
- [ ] Write plugin test guide

### Week 9-10: SDK & Documentation
- [ ] Create plugin template
- [ ] Write SDK documentation
- [ ] Create example plugins (5+)
- [ ] Write tutorials

## Success Metrics

- [ ] Plugin architecture functional
- [ ] **5+ example plugins** created
- [ ] Plugin marketplace operational
- [ ] **10+ community plugins** (6 months post-release)
- [ ] SDK documentation complete
- [ ] Test coverage: **95%+**

## Dependencies

- M1 (testing framework)
- M3 (components for component plugins)
- M7 (measurements for measurement plugins)

## References

- [Python Entry Points](https://packaging.python.org/en/latest/specifications/entry-points/)
- [Plugin Architecture Patterns](https://github.com/pytest-dev/pytest)
- [WordPress Plugin API](https://developer.wordpress.org/plugins/)
