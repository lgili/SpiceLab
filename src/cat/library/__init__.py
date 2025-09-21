"""Component library utilities and curated parts."""

from __future__ import annotations

# Import default libraries (diodes, etc.)
from . import capacitors, diodes, inductors, opamps, resistors, switches, transistors  # noqa: F401
from .registry import (
    ComponentFactory,
    ComponentSpec,
    create_component,
    get_component_spec,
    list_components,
    register_component,
    unregister_component,
)

__all__ = [
    "ComponentFactory",
    "ComponentSpec",
    "register_component",
    "unregister_component",
    "get_component_spec",
    "create_component",
    "list_components",
]
