"""Component library utilities and curated parts."""

from __future__ import annotations

# Import default libraries (diodes, etc.)
from . import capacitors, diodes, inductors, opamps, resistors, switches, transistors  # noqa: F401
from .factory_typing import (
    BjtFactory,
    CapacitorFactory,
    ComponentFactoryProtocol,
    ControlledSourceFactory,
    CurrentSourceFactory,
    DiodeFactory,
    IdealOpAmpFactory,
    InductorFactory,
    ISwitchFactory,
    MosfetFactory,
    OpAmpSubcktFactory,
    ResistorFactory,
    VSwitchFactory,
)
from .registry import (
    ComponentFactory,
    ComponentSpec,
    create_component,
    get_component_spec,
    list_components,
    register_component,
    search_components,
    unregister_component,
)
from .utils import apply_metadata_to_circuit

__all__ = [
    "ComponentFactory",
    "ComponentSpec",
    "ComponentFactoryProtocol",
    "ResistorFactory",
    "CapacitorFactory",
    "InductorFactory",
    "DiodeFactory",
    "CurrentSourceFactory",
    "VSwitchFactory",
    "ISwitchFactory",
    "MosfetFactory",
    "BjtFactory",
    "IdealOpAmpFactory",
    "OpAmpSubcktFactory",
    "ControlledSourceFactory",
    "apply_metadata_to_circuit",
    "register_component",
    "unregister_component",
    "get_component_spec",
    "create_component",
    "list_components",
    "search_components",
]
