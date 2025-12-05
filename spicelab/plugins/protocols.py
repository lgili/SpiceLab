"""Plugin protocol interfaces.

This module defines the protocol interfaces that specific plugin types
must implement to integrate with SpiceLab.

Available protocols:
- ComponentPlugin: For adding new circuit components
- EnginePlugin: For adding simulation engines
- MeasurementPlugin: For adding measurement functions
- AnalysisPlugin: For adding analysis types

Example::

    from spicelab.plugins import ComponentPlugin, PluginMetadata, PluginType

    class MemristorPlugin(ComponentPlugin):
        @property
        def metadata(self) -> PluginMetadata:
            return PluginMetadata(
                name="spicelab-memristor",
                version="1.0.0",
                plugin_type=PluginType.COMPONENT,
            )

        def get_components(self):
            return {"Memristor": MemristorComponent}

        def activate(self):
            self.register_components()

        def deactivate(self):
            self.unregister_components()
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .base import Plugin

if TYPE_CHECKING:
    pass


class ComponentPlugin(Plugin):
    """Protocol for plugins that add circuit components.

    Component plugins can register new component types that users
    can add to their circuits.

    Example::

        class MemristorPlugin(ComponentPlugin):
            def get_components(self):
                return {
                    "Memristor": MemristorComponent,
                    "IdealMemristor": IdealMemristorComponent,
                }

            def activate(self):
                self.register_components()

            def deactivate(self):
                self.unregister_components()
    """

    _registered_components: dict[str, type[Any]] = {}

    @abstractmethod
    def get_components(self) -> dict[str, type[Any]]:
        """Return dictionary of component name to component class.

        Returns:
            Dictionary mapping component names to their classes
        """
        ...

    def register_components(self) -> None:
        """Register all components with SpiceLab.

        This adds the components to the global component registry.
        """
        from spicelab.library.registry import ComponentRegistry  # type: ignore[attr-defined]

        components = self.get_components()
        for name, component_class in components.items():
            ComponentRegistry.register(name, component_class)
            self._registered_components[name] = component_class

    def unregister_components(self) -> None:
        """Unregister all components from SpiceLab."""
        from spicelab.library.registry import ComponentRegistry  # type: ignore[attr-defined]

        for name in self._registered_components:
            try:
                ComponentRegistry.unregister(name)
            except Exception:
                pass
        self._registered_components.clear()

    def get_component_info(self) -> list[dict[str, Any]]:
        """Get information about registered components."""
        components = self.get_components()
        return [
            {
                "name": name,
                "class": cls.__name__,
                "module": cls.__module__,
                "doc": cls.__doc__ or "",
            }
            for name, cls in components.items()
        ]


class EnginePlugin(Plugin):
    """Protocol for plugins that add simulation engines.

    Engine plugins can register new simulation backends (e.g., QSPICE, Xyce).

    Example::

        class QspicePlugin(EnginePlugin):
            def get_engines(self):
                return {
                    "qspice": QspiceEngine,
                }

            def activate(self):
                self.register_engines()

            def deactivate(self):
                self.unregister_engines()
    """

    _registered_engines: dict[str, type[Any]] = {}

    @abstractmethod
    def get_engines(self) -> dict[str, type[Any]]:
        """Return dictionary of engine name to engine class.

        Returns:
            Dictionary mapping engine names to their classes
        """
        ...

    def register_engines(self) -> None:
        """Register all engines with SpiceLab."""
        from spicelab.spice.registry import EngineRegistry  # type: ignore[attr-defined]

        engines = self.get_engines()
        for name, engine_class in engines.items():
            EngineRegistry.register(name, engine_class)
            self._registered_engines[name] = engine_class

    def unregister_engines(self) -> None:
        """Unregister all engines from SpiceLab."""
        from spicelab.spice.registry import EngineRegistry  # type: ignore[attr-defined]

        for name in self._registered_engines:
            try:
                EngineRegistry.unregister(name)
            except Exception:
                pass
        self._registered_engines.clear()

    def get_engine_info(self) -> list[dict[str, Any]]:
        """Get information about registered engines."""
        engines = self.get_engines()
        return [
            {
                "name": name,
                "class": cls.__name__,
                "module": cls.__module__,
                "doc": cls.__doc__ or "",
            }
            for name, cls in engines.items()
        ]


class MeasurementPlugin(Plugin):
    """Protocol for plugins that add measurement functions.

    Measurement plugins can register new measurement types for
    analyzing simulation results.

    Example::

        class RFMeasurementsPlugin(MeasurementPlugin):
            def get_measurements(self):
                return {
                    "s_parameters": measure_s_parameters,
                    "noise_figure": measure_noise_figure,
                    "ip3": measure_ip3,
                }

            def activate(self):
                self.register_measurements()

            def deactivate(self):
                self.unregister_measurements()
    """

    _registered_measurements: dict[str, Callable[..., Any]] = {}

    @abstractmethod
    def get_measurements(self) -> dict[str, Callable[..., Any]]:
        """Return dictionary of measurement name to measurement function.

        Returns:
            Dictionary mapping measurement names to functions
        """
        ...

    def register_measurements(self) -> None:
        """Register all measurements with SpiceLab."""
        from spicelab.measurements.registry import MeasurementRegistry

        measurements = self.get_measurements()
        for name, func in measurements.items():
            MeasurementRegistry.register(name, func)  # type: ignore[call-arg]
            self._registered_measurements[name] = func

    def unregister_measurements(self) -> None:
        """Unregister all measurements from SpiceLab."""
        from spicelab.measurements.registry import MeasurementRegistry

        for name in self._registered_measurements:
            try:
                MeasurementRegistry.unregister(name)  # type: ignore[attr-defined]
            except Exception:
                pass
        self._registered_measurements.clear()

    def get_measurement_info(self) -> list[dict[str, Any]]:
        """Get information about registered measurements."""
        measurements = self.get_measurements()
        return [
            {
                "name": name,
                "function": func.__name__,
                "module": func.__module__,
                "doc": func.__doc__ or "",
            }
            for name, func in measurements.items()
        ]


class AnalysisPlugin(Plugin):
    """Protocol for plugins that add analysis types.

    Analysis plugins can register new analysis types (e.g., harmonic balance,
    noise analysis, etc.).

    Example::

        class HarmonicBalancePlugin(AnalysisPlugin):
            def get_analyses(self):
                return {
                    "hb": HarmonicBalanceAnalysis,
                    "pss": PeriodicSteadyStateAnalysis,
                }

            def activate(self):
                self.register_analyses()

            def deactivate(self):
                self.unregister_analyses()
    """

    _registered_analyses: dict[str, type[Any]] = {}

    @abstractmethod
    def get_analyses(self) -> dict[str, type[Any]]:
        """Return dictionary of analysis name to analysis class.

        Returns:
            Dictionary mapping analysis names to their classes
        """
        ...

    def register_analyses(self) -> None:
        """Register all analyses with SpiceLab."""
        # Note: Assuming there's an AnalysisRegistry - create if needed
        analyses = self.get_analyses()
        for name, cls in analyses.items():
            # Registry registration would go here
            self._registered_analyses[name] = cls

    def unregister_analyses(self) -> None:
        """Unregister all analyses from SpiceLab."""
        self._registered_analyses.clear()

    def get_analysis_info(self) -> list[dict[str, Any]]:
        """Get information about registered analyses."""
        analyses = self.get_analyses()
        return [
            {
                "name": name,
                "class": cls.__name__,
                "module": cls.__module__,
                "doc": cls.__doc__ or "",
            }
            for name, cls in analyses.items()
        ]


class VisualizationPlugin(Plugin):
    """Protocol for plugins that add visualization tools.

    Visualization plugins can register new plot types and
    visualization functions.

    Example::

        class SmithChartPlugin(VisualizationPlugin):
            def get_visualizations(self):
                return {
                    "smith_chart": plot_smith_chart,
                    "polar_plot": plot_polar,
                }

            def activate(self):
                self.register_visualizations()

            def deactivate(self):
                self.unregister_visualizations()
    """

    _registered_visualizations: dict[str, Callable[..., Any]] = {}

    @abstractmethod
    def get_visualizations(self) -> dict[str, Callable[..., Any]]:
        """Return dictionary of visualization name to function.

        Returns:
            Dictionary mapping visualization names to functions
        """
        ...

    def register_visualizations(self) -> None:
        """Register all visualizations with SpiceLab."""
        visualizations = self.get_visualizations()
        for name, func in visualizations.items():
            self._registered_visualizations[name] = func

    def unregister_visualizations(self) -> None:
        """Unregister all visualizations from SpiceLab."""
        self._registered_visualizations.clear()

    def get_visualization_info(self) -> list[dict[str, Any]]:
        """Get information about registered visualizations."""
        visualizations = self.get_visualizations()
        return [
            {
                "name": name,
                "function": func.__name__,
                "module": func.__module__,
                "doc": func.__doc__ or "",
            }
            for name, func in visualizations.items()
        ]


class ExportPlugin(Plugin):
    """Protocol for plugins that add export formats.

    Export plugins can register new file formats for exporting
    circuits or results.

    Example::

        class GerberExportPlugin(ExportPlugin):
            def get_exporters(self):
                return {
                    "gerber": GerberExporter,
                    "pcb": PCBExporter,
                }

            def activate(self):
                self.register_exporters()

            def deactivate(self):
                self.unregister_exporters()
    """

    _registered_exporters: dict[str, type[Any]] = {}

    @abstractmethod
    def get_exporters(self) -> dict[str, type[Any]]:
        """Return dictionary of format name to exporter class.

        Returns:
            Dictionary mapping format names to exporter classes
        """
        ...

    def register_exporters(self) -> None:
        """Register all exporters with SpiceLab."""
        exporters = self.get_exporters()
        for name, cls in exporters.items():
            self._registered_exporters[name] = cls

    def unregister_exporters(self) -> None:
        """Unregister all exporters from SpiceLab."""
        self._registered_exporters.clear()

    def get_exporter_info(self) -> list[dict[str, Any]]:
        """Get information about registered exporters."""
        exporters = self.get_exporters()
        return [
            {
                "name": name,
                "class": cls.__name__,
                "module": cls.__module__,
                "doc": cls.__doc__ or "",
            }
            for name, cls in exporters.items()
        ]


class ImportPlugin(Plugin):
    """Protocol for plugins that add import formats.

    Import plugins can register new file formats for importing
    circuits or models.

    Example::

        class KiCadImportPlugin(ImportPlugin):
            def get_importers(self):
                return {
                    "kicad": KiCadImporter,
                    "eeschema": EeschemaImporter,
                }

            def activate(self):
                self.register_importers()

            def deactivate(self):
                self.unregister_importers()
    """

    _registered_importers: dict[str, type[Any]] = {}

    @abstractmethod
    def get_importers(self) -> dict[str, type[Any]]:
        """Return dictionary of format name to importer class.

        Returns:
            Dictionary mapping format names to importer classes
        """
        ...

    def register_importers(self) -> None:
        """Register all importers with SpiceLab."""
        importers = self.get_importers()
        for name, cls in importers.items():
            self._registered_importers[name] = cls

    def unregister_importers(self) -> None:
        """Unregister all importers from SpiceLab."""
        self._registered_importers.clear()

    def get_importer_info(self) -> list[dict[str, Any]]:
        """Get information about registered importers."""
        importers = self.get_importers()
        return [
            {
                "name": name,
                "class": cls.__name__,
                "module": cls.__module__,
                "doc": cls.__doc__ or "",
            }
            for name, cls in importers.items()
        ]
