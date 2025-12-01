"""Plugin testing framework.

This module provides utilities for testing SpiceLab plugins,
including test harnesses, fixtures, and validation tools.

Example::

    from spicelab.plugins.testing import PluginTestHarness

    class TestMyPlugin(PluginTestHarness):
        plugin_class = MyPlugin

        def test_activation(self):
            # Plugin is automatically loaded and activated
            assert self.plugin.state == PluginState.ACTIVE

        def test_components(self):
            # Test component registration
            components = self.plugin.get_components()
            assert "MyComponent" in components

For pytest users::

    from spicelab.plugins.testing import plugin_manager, mock_plugin

    def test_with_manager(plugin_manager):
        plugin_manager.load_from_class(MyPlugin)
        assert "my-plugin" in plugin_manager.list_loaded()

    def test_with_mock(mock_plugin):
        mock_plugin.activate()
        assert mock_plugin.activated
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import pytest

from .base import Plugin, PluginMetadata, PluginRegistry, PluginState, PluginType
from .hooks import HookManager
from .loader import PluginLoader
from .manager import PluginManager, PluginManagerConfig
from .protocols import (
    AnalysisPlugin,
    ComponentPlugin,
    EnginePlugin,
    MeasurementPlugin,
)

P = TypeVar("P", bound=Plugin)


@dataclass
class ValidationResult:
    """Result of plugin validation.

    Attributes:
        valid: Whether plugin passed validation
        errors: List of error messages
        warnings: List of warning messages
        info: Additional information
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.valid


class PluginValidator:
    """Validator for SpiceLab plugins.

    Checks that plugins conform to the expected interface
    and follow best practices.

    Example::

        validator = PluginValidator()
        result = validator.validate(MyPlugin)

        if not result.valid:
            for error in result.errors:
                print(f"ERROR: {error}")
    """

    def validate(self, plugin_class: type[Plugin]) -> ValidationResult:
        """Validate a plugin class.

        Args:
            plugin_class: Plugin class to validate

        Returns:
            Validation result
        """
        errors: list[str] = []
        warnings: list[str] = []
        info: dict[str, Any] = {}

        # Check it's a proper subclass
        if not issubclass(plugin_class, Plugin):
            errors.append(f"{plugin_class.__name__} must inherit from Plugin")
            return ValidationResult(valid=False, errors=errors)

        # Try to instantiate
        try:
            plugin = plugin_class()
        except Exception as e:
            errors.append(f"Failed to instantiate: {e}")
            return ValidationResult(valid=False, errors=errors)

        # Check metadata
        try:
            metadata = plugin.metadata
            info["metadata"] = metadata.to_dict()

            if not metadata.name:
                errors.append("Plugin name cannot be empty")
            if not metadata.version:
                errors.append("Plugin version cannot be empty")
            if not self._is_valid_version(metadata.version):
                warnings.append(f"Version '{metadata.version}' may not be semver compliant")

        except Exception as e:
            errors.append(f"Failed to get metadata: {e}")
            return ValidationResult(valid=False, errors=errors)

        # Check required methods
        if not hasattr(plugin, "activate") or not callable(plugin.activate):
            errors.append("Plugin must implement activate() method")
        if not hasattr(plugin, "deactivate") or not callable(plugin.deactivate):
            errors.append("Plugin must implement deactivate() method")

        # Check plugin type-specific requirements
        if isinstance(plugin, ComponentPlugin):
            self._validate_component_plugin(plugin, errors, warnings)
        elif isinstance(plugin, EnginePlugin):
            self._validate_engine_plugin(plugin, errors, warnings)
        elif isinstance(plugin, MeasurementPlugin):
            self._validate_measurement_plugin(plugin, errors, warnings)
        elif isinstance(plugin, AnalysisPlugin):
            self._validate_analysis_plugin(plugin, errors, warnings)

        # Test lifecycle
        try:
            plugin.load()
            if plugin.state != PluginState.LOADED:
                warnings.append("Plugin state not set to LOADED after load()")
            plugin.unload()
        except Exception as e:
            warnings.append(f"Lifecycle test warning: {e}")

        valid = len(errors) == 0
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            info=info,
        )

    def _validate_component_plugin(
        self,
        plugin: ComponentPlugin,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Validate a component plugin."""
        try:
            components = plugin.get_components()
            if not isinstance(components, dict):
                errors.append("get_components() must return a dict")
            elif len(components) == 0:
                warnings.append("Plugin provides no components")
        except Exception as e:
            errors.append(f"get_components() failed: {e}")

    def _validate_engine_plugin(
        self,
        plugin: EnginePlugin,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Validate an engine plugin."""
        try:
            engines = plugin.get_engines()
            if not isinstance(engines, dict):
                errors.append("get_engines() must return a dict")
            elif len(engines) == 0:
                warnings.append("Plugin provides no engines")
        except Exception as e:
            errors.append(f"get_engines() failed: {e}")

    def _validate_measurement_plugin(
        self,
        plugin: MeasurementPlugin,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Validate a measurement plugin."""
        try:
            measurements = plugin.get_measurements()
            if not isinstance(measurements, dict):
                errors.append("get_measurements() must return a dict")
            elif len(measurements) == 0:
                warnings.append("Plugin provides no measurements")
            else:
                for name, func in measurements.items():
                    if not callable(func):
                        errors.append(f"Measurement '{name}' is not callable")
        except Exception as e:
            errors.append(f"get_measurements() failed: {e}")

    def _validate_analysis_plugin(
        self,
        plugin: AnalysisPlugin,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Validate an analysis plugin."""
        try:
            analyses = plugin.get_analyses()
            if not isinstance(analyses, dict):
                errors.append("get_analyses() must return a dict")
            elif len(analyses) == 0:
                warnings.append("Plugin provides no analyses")
        except Exception as e:
            errors.append(f"get_analyses() failed: {e}")

    def _is_valid_version(self, version: str) -> bool:
        """Check if version is semver compliant."""
        import re

        semver_pattern = r"^\d+\.\d+\.\d+(-[\w.]+)?(\+[\w.]+)?$"
        return bool(re.match(semver_pattern, version))


class PluginTestHarness:
    """Base class for plugin tests.

    Provides automatic setup and teardown of plugin environment.

    Example::

        class TestMyPlugin(PluginTestHarness):
            plugin_class = MyPlugin

            def test_activation(self):
                assert self.plugin.state == PluginState.ACTIVE

            def test_custom(self):
                result = self.plugin.do_something()
                assert result is not None
    """

    plugin_class: type[Plugin]
    plugin: Plugin
    manager: PluginManager

    @pytest.fixture(autouse=True)
    def setup_plugin(self) -> Generator[None, None, None]:
        """Set up plugin environment before each test."""
        # Reset hooks
        HookManager.reset()

        # Create manager
        config = PluginManagerConfig(auto_discover=False)
        self.manager = PluginManager(config=config, singleton=False)

        # Load and activate plugin
        self.plugin = self.manager.loader.load_from_class(self.plugin_class)
        self.manager.registry.register(self.plugin)
        self.manager.activate_plugin(self.plugin.name)

        yield

        # Teardown
        self.manager.deactivate_all()
        self.manager.unload_all()
        HookManager.reset()

    def get_plugin(self) -> Plugin:
        """Get the plugin instance."""
        return self.plugin


class MockPluginFactory:
    """Factory for creating mock plugins for testing.

    Example::

        factory = MockPluginFactory()

        # Create a simple mock
        plugin = factory.create_mock()

        # Create with custom metadata
        plugin = factory.create_mock(
            name="test-plugin",
            version="2.0.0",
            plugin_type=PluginType.COMPONENT,
        )

        # Create a component plugin mock
        comp_plugin = factory.create_component_plugin(
            components={"MyComp": MyCompClass}
        )
    """

    def create_mock(
        self,
        name: str = "mock-plugin",
        version: str = "1.0.0",
        plugin_type: PluginType = PluginType.GENERIC,
        **extra_metadata: Any,
    ) -> Plugin:
        """Create a mock plugin.

        Args:
            name: Plugin name
            version: Plugin version
            plugin_type: Plugin type
            **extra_metadata: Additional metadata fields

        Returns:
            Mock plugin instance
        """

        class MockPlugin(Plugin):
            _name = name
            _version = version
            _type = plugin_type
            _extra = extra_metadata
            activated = False
            deactivated = False
            configured = False
            config_data: dict[str, Any] = {}

            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name=self._name,
                    version=self._version,
                    plugin_type=self._type,
                    **self._extra,
                )

            def activate(self) -> None:
                self.activated = True

            def deactivate(self) -> None:
                self.deactivated = True

            def configure(self, settings: dict[str, Any]) -> None:
                self.configured = True
                self.config_data = settings

        return MockPlugin()

    def create_component_plugin(
        self,
        name: str = "mock-components",
        version: str = "1.0.0",
        components: dict[str, type[Any]] | None = None,
    ) -> ComponentPlugin:
        """Create a mock component plugin.

        Args:
            name: Plugin name
            version: Plugin version
            components: Dictionary of component name to class

        Returns:
            Mock component plugin
        """
        components = components or {}

        class MockComponentPlugin(ComponentPlugin):
            _name = name
            _version = version
            _components = components

            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name=self._name,
                    version=self._version,
                    plugin_type=PluginType.COMPONENT,
                )

            def get_components(self) -> dict[str, type[Any]]:
                return self._components

            def activate(self) -> None:
                pass

            def deactivate(self) -> None:
                pass

        return MockComponentPlugin()

    def create_measurement_plugin(
        self,
        name: str = "mock-measurements",
        version: str = "1.0.0",
        measurements: dict[str, Callable[..., Any]] | None = None,
    ) -> MeasurementPlugin:
        """Create a mock measurement plugin.

        Args:
            name: Plugin name
            version: Plugin version
            measurements: Dictionary of measurement name to function

        Returns:
            Mock measurement plugin
        """
        measurements = measurements or {}

        class MockMeasurementPlugin(MeasurementPlugin):
            _name = name
            _version = version
            _measurements = measurements

            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name=self._name,
                    version=self._version,
                    plugin_type=PluginType.MEASUREMENT,
                )

            def get_measurements(self) -> dict[str, Callable[..., Any]]:
                return self._measurements

            def activate(self) -> None:
                pass

            def deactivate(self) -> None:
                pass

        return MockMeasurementPlugin()


# Pytest fixtures


@pytest.fixture
def plugin_manager() -> Generator[PluginManager, None, None]:
    """Pytest fixture providing a clean plugin manager.

    Example::

        def test_with_manager(plugin_manager):
            plugin_manager.loader.load_from_class(MyPlugin)
            assert "my-plugin" in plugin_manager.list_loaded()
    """
    HookManager.reset()
    config = PluginManagerConfig(auto_discover=False)
    manager = PluginManager(config=config, singleton=False)

    yield manager

    manager.deactivate_all()
    manager.unload_all()
    HookManager.reset()


@pytest.fixture
def plugin_registry() -> Generator[PluginRegistry, None, None]:
    """Pytest fixture providing a clean plugin registry.

    Example::

        def test_with_registry(plugin_registry):
            plugin_registry.register(my_plugin)
            assert "my-plugin" in plugin_registry
    """
    registry = PluginRegistry()
    yield registry


@pytest.fixture
def plugin_loader() -> Generator[PluginLoader, None, None]:
    """Pytest fixture providing a clean plugin loader.

    Example::

        def test_with_loader(plugin_loader):
            plugin = plugin_loader.load_from_class(MyPlugin)
            assert plugin is not None
    """
    loader = PluginLoader()
    yield loader
    loader.clear()


@pytest.fixture
def mock_plugin() -> Generator[Plugin, None, None]:
    """Pytest fixture providing a mock plugin.

    Example::

        def test_with_mock(mock_plugin):
            mock_plugin.activate()
            assert mock_plugin.activated
    """
    factory = MockPluginFactory()
    plugin = factory.create_mock()
    yield plugin


@pytest.fixture
def plugin_factory() -> MockPluginFactory:
    """Pytest fixture providing a mock plugin factory.

    Example::

        def test_with_factory(plugin_factory):
            plugin = plugin_factory.create_mock(name="test")
            assert plugin.name == "test"
    """
    return MockPluginFactory()


@pytest.fixture
def hook_manager() -> Generator[HookManager, None, None]:
    """Pytest fixture providing a clean hook manager.

    Example::

        def test_with_hooks(hook_manager):
            @HookManager.register(HookType.PRE_SIMULATION)
            def my_hook():
                pass
            assert hook_manager.count_hooks() == 1
    """
    HookManager.reset()
    manager = HookManager.get_instance()
    yield manager
    HookManager.reset()


@pytest.fixture
def validator() -> PluginValidator:
    """Pytest fixture providing a plugin validator.

    Example::

        def test_validation(validator):
            result = validator.validate(MyPlugin)
            assert result.valid
    """
    return PluginValidator()


# Context managers


@contextmanager
def isolated_plugin_env() -> Generator[PluginManager, None, None]:
    """Context manager for isolated plugin testing.

    Creates a clean plugin environment and cleans up after.

    Example::

        with isolated_plugin_env() as manager:
            manager.loader.load_from_class(MyPlugin)
            manager.activate_all()
            # Test plugin functionality
    """
    HookManager.reset()
    config = PluginManagerConfig(auto_discover=False)
    manager = PluginManager(config=config, singleton=False)

    try:
        yield manager
    finally:
        manager.deactivate_all()
        manager.unload_all()
        HookManager.reset()


@contextmanager
def temp_plugin_package(
    name: str,
    plugin_code: str,
) -> Generator[Path, None, None]:
    """Context manager that creates a temporary plugin package.

    Useful for testing plugin loading from actual packages.

    Args:
        name: Package name
        plugin_code: Python code for the plugin module

    Yields:
        Path to the temporary package directory

    Example::

        code = '''
        from spicelab.plugins import Plugin, PluginMetadata

        class TestPlugin(Plugin):
            @property
            def metadata(self):
                return PluginMetadata(name="test", version="1.0.0")
            def activate(self): pass
            def deactivate(self): pass
        '''

        with temp_plugin_package("test_plugin", code) as pkg_path:
            # Import and test the plugin
            sys.path.insert(0, str(pkg_path.parent))
            import test_plugin
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = Path(tmpdir) / name
        pkg_dir.mkdir()

        # Create __init__.py
        init_file = pkg_dir / "__init__.py"
        init_file.write_text(plugin_code)

        # Add to path temporarily
        sys.path.insert(0, tmpdir)

        try:
            yield pkg_dir
        finally:
            sys.path.remove(tmpdir)
            # Remove from cache
            if name in sys.modules:
                del sys.modules[name]


def validate_plugin(plugin_class: type[Plugin]) -> ValidationResult:
    """Convenience function to validate a plugin.

    Args:
        plugin_class: Plugin class to validate

    Returns:
        Validation result
    """
    validator = PluginValidator()
    return validator.validate(plugin_class)
