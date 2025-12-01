"""Undo/redo support for circuit modifications.

Provides history tracking with snapshot-based state management.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from ..core.circuit import Circuit

T = TypeVar("T")


@dataclass
class Snapshot:
    """A snapshot of circuit state at a point in time.

    Attributes:
        state: Serialized circuit state
        description: Human-readable description of the change
        timestamp: Unix timestamp when snapshot was created
        checksum: Hash of the state for integrity checking
    """

    state: dict[str, Any]
    description: str
    timestamp: float = field(default_factory=time.time)
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute a checksum of the state."""
        state_str = json.dumps(self.state, sort_keys=True, default=str)
        return hashlib.md5(state_str.encode()).hexdigest()[:8]

    def __str__(self) -> str:
        time_str = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return f"[{time_str}] {self.description} ({self.checksum})"


class CircuitHistory:
    """History manager for a single circuit with undo/redo support.

    Maintains a stack of snapshots allowing users to navigate through
    previous circuit states.

    Example:
        >>> from spicelab.ux import CircuitHistory
        >>> from spicelab.core.circuit import Circuit
        >>> from spicelab.core.components import Resistor
        >>> from spicelab.core.net import Net, GND
        >>>
        >>> circuit = Circuit("test")
        >>> history = CircuitHistory(circuit)
        >>>
        >>> # Make changes
        >>> R1 = Resistor(ref="1", resistance=1000)
        >>> circuit.add(R1)
        >>> history.save("Added R1")
        >>>
        >>> R2 = Resistor(ref="2", resistance=2000)
        >>> circuit.add(R2)
        >>> history.save("Added R2")
        >>>
        >>> # Undo
        >>> history.undo()  # Removes R2
        >>> print(len(circuit._components))  # 1
        >>>
        >>> # Redo
        >>> history.redo()  # Restores R2
        >>> print(len(circuit._components))  # 2
    """

    def __init__(
        self,
        circuit: Circuit,
        max_history: int = 100,
        auto_save: bool = False,
    ) -> None:
        """Initialize circuit history.

        Args:
            circuit: Circuit to track
            max_history: Maximum number of snapshots to keep
            auto_save: Automatically save on circuit modifications
        """
        self._circuit = circuit
        self._max_history = max_history
        self._auto_save = auto_save

        self._undo_stack: list[Snapshot] = []
        self._redo_stack: list[Snapshot] = []

        # Save initial state
        self._save_snapshot("Initial state")

    def save(self, description: str = "Change") -> None:
        """Save current circuit state to history.

        Args:
            description: Description of the change
        """
        self._save_snapshot(description)
        # Clear redo stack on new change
        self._redo_stack.clear()

    def _save_snapshot(self, description: str) -> None:
        """Internal method to save a snapshot."""
        state = self._serialize_circuit()
        snapshot = Snapshot(state=state, description=description)

        self._undo_stack.append(snapshot)

        # Limit history size
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)

    def _serialize_circuit(self) -> dict[str, Any]:
        """Serialize circuit state to a dictionary."""
        return {
            "name": self._circuit.name,
            "components": [
                {
                    "type": type(comp).__name__,
                    "ref": comp.ref,
                    "attrs": self._serialize_component_attrs(comp),
                }
                for comp in self._circuit._components
            ],
            "connections": [
                {
                    "port_owner": port.owner.ref,
                    "port_name": port.name,
                    "net_name": getattr(net, "name", None),
                }
                for port, net in self._circuit._port_to_net.items()
            ],
            "directives": list(self._circuit._directives),
        }

    def _serialize_component_attrs(self, comp: Any) -> dict[str, Any]:
        """Serialize component attributes."""
        import inspect

        attrs = {}

        # Get constructor parameters for the component class
        try:
            sig = inspect.signature(type(comp).__init__)
            valid_params = set(sig.parameters.keys()) - {"self"}
        except (ValueError, TypeError):
            valid_params = set()

        # Only serialize attributes that are constructor parameters
        for key in dir(comp):
            if key.startswith("_") or key == "ports":
                continue
            # Only include if it's a valid constructor parameter
            if valid_params and key not in valid_params:
                continue
            try:
                value = getattr(comp, key)
                if not callable(value):
                    # Try to serialize
                    json.dumps(value, default=str)
                    attrs[key] = value
            except Exception:
                pass
        return attrs

    def _restore_circuit(self, state: dict[str, Any]) -> None:
        """Restore circuit from serialized state."""
        from ..core import components as comp_module
        from ..core.net import GND, Net

        # Clear current circuit
        self._circuit._components.clear()
        self._circuit._port_to_net.clear()
        self._circuit._directives.clear()
        self._circuit._net_ids.clear()

        # Restore name
        self._circuit.name = state["name"]

        # Restore components
        comp_map: dict[str, Any] = {}
        for comp_data in state["components"]:
            comp_type = comp_data["type"]
            comp_class = getattr(comp_module, comp_type, None)
            if comp_class is None:
                continue

            attrs = comp_data["attrs"]
            try:
                comp = comp_class(**attrs)
                self._circuit.add(comp)
                comp_map[comp_data["ref"]] = comp
            except Exception:
                pass

        # Restore connections
        net_map: dict[str | None, Net] = {None: GND}
        for conn in state["connections"]:
            owner_ref = conn["port_owner"]
            port_name = conn["port_name"]
            net_name = conn["net_name"]

            comp = comp_map.get(owner_ref)
            if comp is None:
                continue

            # Find port
            port = None
            for p in comp.ports:
                if p.name == port_name:
                    port = p
                    break
            if port is None:
                continue

            # Get or create net
            if net_name not in net_map:
                if net_name == "0" or net_name is None:
                    net_map[net_name] = GND
                else:
                    net_map[net_name] = Net(net_name)

            net = net_map[net_name]
            self._circuit.connect(port, net)

        # Restore directives
        for directive in state["directives"]:
            self._circuit.add_directive(directive)

    def undo(self) -> bool:
        """Undo the last change.

        Returns:
            True if undo was successful, False if nothing to undo
        """
        if len(self._undo_stack) <= 1:
            return False

        # Move current state to redo stack
        current = self._undo_stack.pop()
        self._redo_stack.append(current)

        # Restore previous state
        previous = self._undo_stack[-1]
        self._restore_circuit(previous.state)

        return True

    def redo(self) -> bool:
        """Redo the last undone change.

        Returns:
            True if redo was successful, False if nothing to redo
        """
        if not self._redo_stack:
            return False

        # Move state from redo to undo stack
        state = self._redo_stack.pop()
        self._undo_stack.append(state)

        # Restore that state
        self._restore_circuit(state.state)

        return True

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 1

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    def clear(self) -> None:
        """Clear all history."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._save_snapshot("Cleared history")

    def history(self) -> list[Snapshot]:
        """Get all snapshots in history.

        Returns:
            List of snapshots from oldest to newest
        """
        return list(self._undo_stack)

    def __len__(self) -> int:
        """Return number of snapshots in undo stack."""
        return len(self._undo_stack)

    def __str__(self) -> str:
        lines = [
            f"CircuitHistory for '{self._circuit.name}'",
            f"  Undo available: {len(self._undo_stack) - 1}",
            f"  Redo available: {len(self._redo_stack)}",
            "",
            "History:",
        ]
        for i, snapshot in enumerate(self._undo_stack):
            marker = "→" if i == len(self._undo_stack) - 1 else " "
            lines.append(f"  {marker} {snapshot}")
        return "\n".join(lines)


class HistoryManager:
    """Global history manager for multiple circuits.

    Provides a singleton-like interface for managing history
    across multiple circuits.

    Example:
        >>> from spicelab.ux import HistoryManager
        >>>
        >>> manager = HistoryManager()
        >>> manager.track(circuit)
        >>>
        >>> # Make changes
        >>> circuit.add(R1)
        >>> manager.save(circuit, "Added R1")
        >>>
        >>> # Undo across any tracked circuit
        >>> manager.undo(circuit)
    """

    _instance: HistoryManager | None = None
    _histories: dict[int, CircuitHistory]

    def __new__(cls) -> HistoryManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._histories = {}
        return cls._instance

    def __init__(self) -> None:
        # Already initialized in __new__
        pass

    def track(self, circuit: Circuit, max_history: int = 100) -> CircuitHistory:
        """Start tracking a circuit.

        Args:
            circuit: Circuit to track
            max_history: Maximum history size

        Returns:
            CircuitHistory instance for the circuit
        """
        if id(circuit) not in self._histories:
            self._histories[id(circuit)] = CircuitHistory(circuit, max_history)
        return self._histories[id(circuit)]

    def untrack(self, circuit: Circuit) -> None:
        """Stop tracking a circuit.

        Args:
            circuit: Circuit to stop tracking
        """
        self._histories.pop(id(circuit), None)

    def get(self, circuit: Circuit) -> CircuitHistory | None:
        """Get history for a circuit.

        Args:
            circuit: Circuit to get history for

        Returns:
            CircuitHistory or None if not tracked
        """
        return self._histories.get(id(circuit))

    def save(self, circuit: Circuit, description: str = "Change") -> None:
        """Save current state of a tracked circuit.

        Args:
            circuit: Circuit to save
            description: Description of the change
        """
        history = self.get(circuit)
        if history:
            history.save(description)

    def undo(self, circuit: Circuit) -> bool:
        """Undo last change on a circuit.

        Args:
            circuit: Circuit to undo

        Returns:
            True if successful
        """
        history = self.get(circuit)
        return history.undo() if history else False

    def redo(self, circuit: Circuit) -> bool:
        """Redo last undone change on a circuit.

        Args:
            circuit: Circuit to redo

        Returns:
            True if successful
        """
        history = self.get(circuit)
        return history.redo() if history else False

    def clear_all(self) -> None:
        """Clear all tracked histories."""
        self._histories.clear()


__all__ = [
    "Snapshot",
    "CircuitHistory",
    "HistoryManager",
]
