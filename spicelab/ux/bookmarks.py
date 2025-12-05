"""Bookmarks/favorites for frequently used configurations.

Provides persistent storage for circuit configurations and analysis presets.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from builtins import list as List  # Avoid shadowing by method name
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.circuit import Circuit
    from ..core.types import AnalysisSpec


@dataclass
class Bookmark:
    """A saved circuit configuration or analysis preset.

    Attributes:
        name: Unique name for the bookmark
        description: Human-readable description
        type: Type of bookmark (circuit, analysis, config)
        data: Serialized data
        tags: Tags for organization
        created: Creation timestamp
        modified: Last modification timestamp
    """

    name: str
    description: str = ""
    type: str = "circuit"
    data: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    modified: float = field(default_factory=time.time)

    def __str__(self) -> str:
        tags_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"[{self.type}] {self.name}{tags_str}: {self.description}"

    def to_dict(self) -> dict[str, Any]:
        """Convert bookmark to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Bookmark:
        """Create bookmark from dictionary."""
        return cls(**data)


class BookmarkManager:
    """Manager for circuit bookmarks and favorites.

    Handles persistent storage and retrieval of bookmarks.

    Example:
        >>> from spicelab.ux import BookmarkManager
        >>> from spicelab.core.circuit import Circuit
        >>> from spicelab.core.components import Resistor
        >>>
        >>> manager = BookmarkManager()
        >>>
        >>> # Create a circuit
        >>> circuit = Circuit("my_filter")
        >>> circuit.add(Resistor(ref="1", resistance=1000))
        >>>
        >>> # Save as bookmark
        >>> manager.save_circuit("rc_filter_1k", circuit, "RC filter with 1kΩ")
        >>>
        >>> # Later, load it back
        >>> loaded = manager.load_circuit("rc_filter_1k")
        >>>
        >>> # List all bookmarks
        >>> for bm in manager.list():
        ...     print(bm)
    """

    DEFAULT_PATH = Path.home() / ".spicelab" / "bookmarks.json"

    def __init__(self, path: str | Path | None = None) -> None:
        """Initialize bookmark manager.

        Args:
            path: Path to bookmarks file (default: ~/.spicelab/bookmarks.json)
        """
        self._path = Path(path) if path else self.DEFAULT_PATH
        self._bookmarks: dict[str, Bookmark] = {}
        self._load()

    def _load(self) -> None:
        """Load bookmarks from disk."""
        if not self._path.exists():
            return

        try:
            with open(self._path) as f:
                data = json.load(f)
            self._bookmarks = {name: Bookmark.from_dict(bm_data) for name, bm_data in data.items()}
        except Exception:
            self._bookmarks = {}

    def _save(self) -> None:
        """Save bookmarks to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            data = {name: bm.to_dict() for name, bm in self._bookmarks.items()}
            json.dump(data, f, indent=2, default=str)

    def save_circuit(
        self,
        name: str,
        circuit: Circuit,
        description: str = "",
        tags: list[str] | None = None,
    ) -> Bookmark:
        """Save a circuit as a bookmark.

        Args:
            name: Unique name for the bookmark
            circuit: Circuit to save
            description: Description
            tags: Tags for organization

        Returns:
            Created bookmark
        """
        data = self._serialize_circuit(circuit)
        bookmark = Bookmark(
            name=name,
            description=description or f"Circuit: {circuit.name}",
            type="circuit",
            data=data,
            tags=tags or [],
        )
        self._bookmarks[name] = bookmark
        self._save()
        return bookmark

    def load_circuit(self, name: str) -> Circuit:
        """Load a circuit from a bookmark.

        Args:
            name: Bookmark name

        Returns:
            Circuit instance

        Raises:
            KeyError: If bookmark not found
            ValueError: If bookmark is not a circuit
        """
        bookmark = self._bookmarks.get(name)
        if bookmark is None:
            raise KeyError(f"Bookmark not found: {name}")
        if bookmark.type != "circuit":
            raise ValueError(f"Bookmark '{name}' is not a circuit")

        return self._deserialize_circuit(bookmark.data)

    def save_analysis(
        self,
        name: str,
        analyses: list[AnalysisSpec],
        description: str = "",
        tags: list[str] | None = None,
    ) -> Bookmark:
        """Save analysis presets as a bookmark.

        Args:
            name: Unique name for the bookmark
            analyses: Analysis specifications to save
            description: Description
            tags: Tags for organization

        Returns:
            Created bookmark
        """
        data = {
            "analyses": [{"mode": a.mode, "args": a.args} for a in analyses],
        }
        bookmark = Bookmark(
            name=name,
            description=description or f"Analysis preset: {len(analyses)} analyses",
            type="analysis",
            data=data,
            tags=tags or [],
        )
        self._bookmarks[name] = bookmark
        self._save()
        return bookmark

    def load_analysis(self, name: str) -> list[AnalysisSpec]:
        """Load analysis presets from a bookmark.

        Args:
            name: Bookmark name

        Returns:
            List of AnalysisSpec instances

        Raises:
            KeyError: If bookmark not found
            ValueError: If bookmark is not an analysis preset
        """
        from ..core.types import AnalysisSpec

        bookmark = self._bookmarks.get(name)
        if bookmark is None:
            raise KeyError(f"Bookmark not found: {name}")
        if bookmark.type != "analysis":
            raise ValueError(f"Bookmark '{name}' is not an analysis preset")

        return [AnalysisSpec(a["mode"], a["args"]) for a in bookmark.data.get("analyses", [])]

    def save_config(
        self,
        name: str,
        config: dict[str, Any],
        description: str = "",
        tags: list[str] | None = None,
    ) -> Bookmark:
        """Save a configuration as a bookmark.

        Args:
            name: Unique name for the bookmark
            config: Configuration dictionary
            description: Description
            tags: Tags for organization

        Returns:
            Created bookmark
        """
        bookmark = Bookmark(
            name=name,
            description=description or "Configuration",
            type="config",
            data=config,
            tags=tags or [],
        )
        self._bookmarks[name] = bookmark
        self._save()
        return bookmark

    def load_config(self, name: str) -> dict[str, Any]:
        """Load a configuration from a bookmark.

        Args:
            name: Bookmark name

        Returns:
            Configuration dictionary

        Raises:
            KeyError: If bookmark not found
        """
        bookmark = self._bookmarks.get(name)
        if bookmark is None:
            raise KeyError(f"Bookmark not found: {name}")

        return dict(bookmark.data)

    def get(self, name: str) -> Bookmark | None:
        """Get a bookmark by name.

        Args:
            name: Bookmark name

        Returns:
            Bookmark or None if not found
        """
        return self._bookmarks.get(name)

    def delete(self, name: str) -> bool:
        """Delete a bookmark.

        Args:
            name: Bookmark name

        Returns:
            True if deleted, False if not found
        """
        if name in self._bookmarks:
            del self._bookmarks[name]
            self._save()
            return True
        return False

    def rename(self, old_name: str, new_name: str) -> bool:
        """Rename a bookmark.

        Args:
            old_name: Current name
            new_name: New name

        Returns:
            True if renamed, False if not found
        """
        if old_name not in self._bookmarks:
            return False
        if new_name in self._bookmarks:
            raise ValueError(f"Bookmark already exists: {new_name}")

        bookmark = self._bookmarks.pop(old_name)
        bookmark.name = new_name
        bookmark.modified = time.time()
        self._bookmarks[new_name] = bookmark
        self._save()
        return True

    def list(
        self,
        type_filter: str | None = None,
        tag_filter: str | None = None,
    ) -> list[Bookmark]:
        """List all bookmarks with optional filtering.

        Args:
            type_filter: Filter by type (circuit, analysis, config)
            tag_filter: Filter by tag

        Returns:
            List of matching bookmarks
        """
        bookmarks = list(self._bookmarks.values())

        if type_filter:
            bookmarks = [b for b in bookmarks if b.type == type_filter]

        if tag_filter:
            bookmarks = [b for b in bookmarks if tag_filter in b.tags]

        return sorted(bookmarks, key=lambda b: b.name)

    def search(self, query: str) -> List[Bookmark]:
        """Search bookmarks by name or description.

        Args:
            query: Search query

        Returns:
            List of matching bookmarks
        """
        query = query.lower()
        return [
            b
            for b in self._bookmarks.values()
            if query in b.name.lower() or query in b.description.lower()
        ]

    def tags(self) -> List[str]:
        """Get all unique tags.

        Returns:
            List of tag names
        """
        all_tags: set[str] = set()
        for bookmark in self._bookmarks.values():
            all_tags.update(bookmark.tags)
        return sorted(all_tags)

    def export_to_file(self, path: str | Path) -> None:
        """Export bookmarks to a file.

        Args:
            path: Output file path
        """
        path = Path(path)
        with open(path, "w") as f:
            data = {name: bm.to_dict() for name, bm in self._bookmarks.items()}
            json.dump(data, f, indent=2, default=str)

    def import_from_file(self, path: str | Path, overwrite: bool = False) -> int:
        """Import bookmarks from a file.

        Args:
            path: Input file path
            overwrite: Overwrite existing bookmarks

        Returns:
            Number of bookmarks imported
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        imported = 0
        for name, bm_data in data.items():
            if name in self._bookmarks and not overwrite:
                continue
            self._bookmarks[name] = Bookmark.from_dict(bm_data)
            imported += 1

        if imported:
            self._save()

        return imported

    def clear(self) -> None:
        """Clear all bookmarks."""
        self._bookmarks.clear()
        self._save()

    def _serialize_circuit(self, circuit: Circuit) -> dict[str, Any]:
        """Serialize a circuit to a dictionary."""
        import inspect
        import json

        def serialize_component(comp: Any) -> dict[str, Any]:
            attrs = {}

            # Get constructor parameters for the component class
            try:
                sig = inspect.signature(type(comp).__init__)
                valid_params = set(sig.parameters.keys()) - {"self"}
            except (ValueError, TypeError):
                valid_params = set()

            for key in dir(comp):
                if key.startswith("_") or key == "ports":
                    continue
                # Only include if it's a valid constructor parameter
                if valid_params and key not in valid_params:
                    continue
                try:
                    value = getattr(comp, key)
                    if not callable(value):
                        json.dumps(value, default=str)
                        attrs[key] = value
                except Exception:
                    pass
            return {
                "type": type(comp).__name__,
                "ref": comp.ref,
                "attrs": attrs,
            }

        return {
            "name": circuit.name,
            "components": [serialize_component(c) for c in circuit._components],
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
        from ..core import components as comp_module
        from ..core.circuit import Circuit
        from ..core.net import GND, Net

        circuit = Circuit(data.get("name", "Loaded"))

        # Deserialize components
        comp_map: dict[str, Any] = {}
        for comp_data in data.get("components", []):
            comp_type = comp_data.get("type", "")
            comp_class = getattr(comp_module, comp_type, None)
            if comp_class is None:
                continue

            attrs = comp_data.get("attrs", {})
            try:
                comp = comp_class(**attrs)
                circuit.add(comp)
                comp_map[comp_data["ref"]] = comp
            except Exception:
                pass

        # Restore connections
        net_map: dict[str | None, Net] = {}
        for conn in data.get("connections", []):
            owner_ref = conn["owner"]
            port_name = conn["port"]
            net_name = conn["net"]

            comp = comp_map.get(owner_ref)
            if comp is None:
                continue

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

    def __len__(self) -> int:
        """Return number of bookmarks."""
        return len(self._bookmarks)

    def __contains__(self, name: str) -> bool:
        """Check if bookmark exists."""
        return name in self._bookmarks


# Global instance for convenience functions
_default_manager: BookmarkManager | None = None


def _get_manager() -> BookmarkManager:
    """Get or create the default bookmark manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = BookmarkManager()
    return _default_manager


def save_bookmark(
    name: str,
    obj: Any,
    description: str = "",
    tags: list[str] | None = None,
) -> Bookmark:
    """Save an object as a bookmark.

    Args:
        name: Unique name for the bookmark
        obj: Object to save (Circuit, AnalysisSpec list, or dict)
        description: Description
        tags: Tags for organization

    Returns:
        Created bookmark
    """
    from ..core.circuit import Circuit

    manager = _get_manager()

    if isinstance(obj, Circuit):
        return manager.save_circuit(name, obj, description, tags)
    elif isinstance(obj, list):
        return manager.save_analysis(name, obj, description, tags)
    elif isinstance(obj, dict):
        return manager.save_config(name, obj, description, tags)
    else:
        raise TypeError(f"Cannot bookmark object of type {type(obj)}")


def load_bookmark(name: str) -> Any:
    """Load an object from a bookmark.

    Args:
        name: Bookmark name

    Returns:
        Circuit, list of AnalysisSpec, or dict depending on bookmark type
    """
    manager = _get_manager()
    bookmark = manager.get(name)
    if bookmark is None:
        raise KeyError(f"Bookmark not found: {name}")

    if bookmark.type == "circuit":
        return manager.load_circuit(name)
    elif bookmark.type == "analysis":
        return manager.load_analysis(name)
    else:
        return manager.load_config(name)


def list_bookmarks(
    type_filter: str | None = None,
    tag_filter: str | None = None,
) -> list[Bookmark]:
    """List all bookmarks.

    Args:
        type_filter: Filter by type
        tag_filter: Filter by tag

    Returns:
        List of bookmarks
    """
    return _get_manager().list(type_filter, tag_filter)


__all__ = [
    "Bookmark",
    "BookmarkManager",
    "save_bookmark",
    "load_bookmark",
    "list_bookmarks",
]
