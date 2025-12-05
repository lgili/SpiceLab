"""Clipboard support for circuit snippets.

Provides copy/paste functionality for circuits and components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.circuit import Circuit
    from ..core.components import Component


@dataclass
class ClipboardItem:
    """An item stored in the clipboard.

    Attributes:
        type: Type of item ("circuit", "component", "snippet")
        data: Serialized data
        source: Name of source circuit
        description: Optional description
    """

    type: str
    data: dict[str, Any]
    source: str = ""
    description: str = ""

    def __str__(self) -> str:
        return f"[{self.type}] {self.description or self.source}"


class CircuitClipboard:
    """Clipboard for circuit snippets and components.

    Supports copying and pasting circuits, components, and
    arbitrary snippets between circuits.

    Example:
        >>> from spicelab.ux import CircuitClipboard
        >>> from spicelab.core.circuit import Circuit
        >>> from spicelab.core.components import Resistor
        >>>
        >>> clipboard = CircuitClipboard()
        >>>
        >>> # Copy a component
        >>> R1 = Resistor(ref="1", resistance=1000)
        >>> clipboard.copy_component(R1)
        >>>
        >>> # Paste into another circuit
        >>> circuit = Circuit("test")
        >>> R1_copy = clipboard.paste_component()
        >>> circuit.add(R1_copy)
        >>>
        >>> # Copy entire circuit
        >>> clipboard.copy_circuit(circuit)
        >>>
        >>> # Paste as new circuit
        >>> new_circuit = clipboard.paste_circuit()
    """

    _instance: CircuitClipboard | None = None
    _items: list[ClipboardItem]
    _max_items: int

    def __new__(cls) -> CircuitClipboard:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._items = []
            cls._instance._max_items = 50
        return cls._instance

    def __init__(self) -> None:
        pass

    def copy_circuit(
        self,
        circuit: Circuit,
        description: str = "",
    ) -> None:
        """Copy a circuit to the clipboard.

        Args:
            circuit: Circuit to copy
            description: Optional description
        """
        data = self._serialize_circuit(circuit)
        item = ClipboardItem(
            type="circuit",
            data=data,
            source=circuit.name,
            description=description or f"Circuit: {circuit.name}",
        )
        self._add_item(item)

    def paste_circuit(self, name: str | None = None) -> Circuit:
        """Paste a circuit from the clipboard.

        Args:
            name: Optional new name for the circuit

        Returns:
            New Circuit instance

        Raises:
            ValueError: If no circuit in clipboard
        """
        item = self._find_item("circuit")
        if item is None:
            raise ValueError("No circuit in clipboard")

        circuit = self._deserialize_circuit(item.data)
        if name:
            circuit.name = name

        return circuit

    def copy_component(
        self,
        component: Component,
        description: str = "",
    ) -> None:
        """Copy a component to the clipboard.

        Args:
            component: Component to copy
            description: Optional description
        """
        data = self._serialize_component(component)
        item = ClipboardItem(
            type="component",
            data=data,
            source=component.ref,
            description=description or f"Component: {component.ref}",
        )
        self._add_item(item)

    def paste_component(self, new_ref: str | None = None) -> Component:
        """Paste a component from the clipboard.

        Args:
            new_ref: Optional new reference designator

        Returns:
            New Component instance

        Raises:
            ValueError: If no component in clipboard
        """
        item = self._find_item("component")
        if item is None:
            raise ValueError("No component in clipboard")

        component = self._deserialize_component(item.data)
        if new_ref:
            component.ref = new_ref

        return component

    def copy_components(
        self,
        components: list[Component],
        description: str = "",
    ) -> None:
        """Copy multiple components to the clipboard.

        Args:
            components: Components to copy
            description: Optional description
        """
        data = {
            "components": [self._serialize_component(c) for c in components],
        }
        item = ClipboardItem(
            type="components",
            data=data,
            source=", ".join(c.ref for c in components[:3]),
            description=description or f"Components: {len(components)} items",
        )
        self._add_item(item)

    def paste_components(
        self,
        ref_prefix: str = "",
    ) -> list[Component]:
        """Paste multiple components from the clipboard.

        Args:
            ref_prefix: Prefix to add to reference designators

        Returns:
            List of new Component instances

        Raises:
            ValueError: If no components in clipboard
        """
        item = self._find_item("components")
        if item is None:
            raise ValueError("No components in clipboard")

        components = []
        for comp_data in item.data.get("components", []):
            comp = self._deserialize_component(comp_data)
            if ref_prefix:
                comp.ref = f"{ref_prefix}{comp.ref}"
            components.append(comp)

        return components

    def copy_snippet(
        self,
        circuit: Circuit,
        components: list[Component],
        include_connections: bool = True,
        description: str = "",
    ) -> None:
        """Copy a circuit snippet (components with connections).

        Args:
            circuit: Source circuit
            components: Components to include
            include_connections: Include net connections
            description: Optional description
        """
        comp_refs = {c.ref for c in components}

        data: dict[str, Any] = {
            "components": [self._serialize_component(c) for c in components],
        }

        if include_connections:
            connections = []
            for port, net in circuit._port_to_net.items():
                if port.owner.ref in comp_refs:
                    connections.append(
                        {
                            "owner": port.owner.ref,
                            "port": port.name,
                            "net": getattr(net, "name", None),
                        }
                    )
            data["connections"] = connections

        item = ClipboardItem(
            type="snippet",
            data=data,
            source=circuit.name,
            description=description or f"Snippet: {len(components)} components",
        )
        self._add_item(item)

    def paste_snippet(
        self,
        circuit: Circuit,
        ref_prefix: str = "",
        net_prefix: str = "",
    ) -> list[Component]:
        """Paste a snippet into a circuit.

        Args:
            circuit: Target circuit
            ref_prefix: Prefix for component references
            net_prefix: Prefix for net names

        Returns:
            List of pasted components

        Raises:
            ValueError: If no snippet in clipboard
        """
        item = self._find_item("snippet")
        if item is None:
            raise ValueError("No snippet in clipboard")

        from ..core.net import GND, Net

        # Deserialize components
        components = []
        comp_map: dict[str, Component] = {}
        for comp_data in item.data.get("components", []):
            comp = self._deserialize_component(comp_data)
            original_ref = comp.ref
            if ref_prefix:
                comp.ref = f"{ref_prefix}{comp.ref}"
            components.append(comp)
            comp_map[original_ref] = comp
            circuit.add(comp)

        # Restore connections
        net_map: dict[str | None, Net] = {}
        for conn in item.data.get("connections", []):
            owner_ref = conn["owner"]
            port_name = conn["port"]
            net_name = conn["net"]

            comp_or_none = comp_map.get(owner_ref)
            if comp_or_none is None:
                continue
            comp = comp_or_none

            # Find port
            port = None
            for p in comp.ports:
                if p.name == port_name:
                    port = p
                    break
            if port is None:
                continue

            # Get or create net
            if net_name is None or net_name == "0":
                net = GND
            elif net_name in net_map:
                net = net_map[net_name]
            else:
                new_name = f"{net_prefix}{net_name}" if net_prefix else net_name
                net = Net(new_name)
                net_map[net_name] = net

            circuit.connect(port, net)

        return components

    def _serialize_circuit(self, circuit: Circuit) -> dict[str, Any]:
        """Serialize a circuit to a dictionary."""
        return {
            "name": circuit.name,
            "components": [self._serialize_component(c) for c in circuit._components],
            "connections": [
                {
                    "owner": port.owner.ref,
                    "port": port.name,
                    "net": getattr(net, "name", None),
                }
                for port, net in circuit._port_to_net.items()
            ],
            "directives": list(circuit._directives),
        }

    def _deserialize_circuit(self, data: dict[str, Any]) -> Circuit:
        """Deserialize a circuit from a dictionary."""
        from ..core.circuit import Circuit
        from ..core.net import GND, Net

        circuit = Circuit(data.get("name", "Pasted"))

        # Deserialize components
        comp_map: dict[str, Component] = {}
        for comp_data in data.get("components", []):
            comp = self._deserialize_component(comp_data)
            circuit.add(comp)
            comp_map[comp.ref] = comp

        # Restore connections
        net_map: dict[str | None, Net] = {}
        for conn in data.get("connections", []):
            owner_ref = conn["owner"]
            port_name = conn["port"]
            net_name = conn["net"]

            comp_or_none = comp_map.get(owner_ref)
            if comp_or_none is None:
                continue
            comp = comp_or_none

            port = None
            for p in comp.ports:
                if p.name == port_name:
                    port = p
                    break
            if port is None:
                continue

            if net_name is None or net_name == "0":
                net = GND
            elif net_name in net_map:
                net = net_map[net_name]
            else:
                net = Net(net_name)
                net_map[net_name] = net

            circuit.connect(port, net)

        # Restore directives
        for directive in data.get("directives", []):
            circuit.add_directive(directive)

        return circuit

    def _serialize_component(self, component: Component) -> dict[str, Any]:
        """Serialize a component to a dictionary."""
        import inspect

        attrs = {}

        # Get constructor parameters for the component class
        try:
            sig = inspect.signature(type(component).__init__)
            valid_params = set(sig.parameters.keys()) - {"self"}
        except (ValueError, TypeError):
            valid_params = set()

        for key in dir(component):
            if key.startswith("_") or key == "ports":
                continue
            # Only include if it's a valid constructor parameter
            if valid_params and key not in valid_params:
                continue
            try:
                value = getattr(component, key)
                if not callable(value):
                    json.dumps(value, default=str)
                    attrs[key] = value
            except Exception:
                pass

        return {
            "type": type(component).__name__,
            "ref": component.ref,
            "attrs": attrs,
        }

    def _deserialize_component(self, data: dict[str, Any]) -> Component:
        """Deserialize a component from a dictionary."""
        from ..core import components as comp_module

        comp_type = data.get("type", "")
        comp_class = getattr(comp_module, comp_type, None)
        if comp_class is None:
            raise ValueError(f"Unknown component type: {comp_type}")

        attrs = data.get("attrs", {})
        result: Component = comp_class(**attrs)
        return result

    def _add_item(self, item: ClipboardItem) -> None:
        """Add an item to the clipboard."""
        self._items.insert(0, item)
        # Limit history
        if len(self._items) > self._max_items:
            self._items.pop()

    def _find_item(self, item_type: str) -> ClipboardItem | None:
        """Find the most recent item of a given type."""
        for item in self._items:
            if item.type == item_type:
                return item
        return None

    def clear(self) -> None:
        """Clear the clipboard."""
        self._items.clear()

    def history(self) -> list[ClipboardItem]:
        """Get clipboard history.

        Returns:
            List of clipboard items from newest to oldest
        """
        return list(self._items)

    def is_empty(self) -> bool:
        """Check if clipboard is empty."""
        return len(self._items) == 0

    def __len__(self) -> int:
        """Return number of items in clipboard."""
        return len(self._items)


# Convenience functions using global clipboard


def copy_circuit(circuit: Circuit, description: str = "") -> None:
    """Copy a circuit to the global clipboard.

    Args:
        circuit: Circuit to copy
        description: Optional description
    """
    CircuitClipboard().copy_circuit(circuit, description)


def paste_circuit(name: str | None = None) -> Circuit:
    """Paste a circuit from the global clipboard.

    Args:
        name: Optional new name

    Returns:
        New Circuit instance
    """
    return CircuitClipboard().paste_circuit(name)


def copy_component(component: Component, description: str = "") -> None:
    """Copy a component to the global clipboard.

    Args:
        component: Component to copy
        description: Optional description
    """
    CircuitClipboard().copy_component(component, description)


def paste_component(new_ref: str | None = None) -> Component:
    """Paste a component from the global clipboard.

    Args:
        new_ref: Optional new reference

    Returns:
        New Component instance
    """
    return CircuitClipboard().paste_component(new_ref)


__all__ = [
    "ClipboardItem",
    "CircuitClipboard",
    "copy_circuit",
    "paste_circuit",
    "copy_component",
    "paste_component",
]
