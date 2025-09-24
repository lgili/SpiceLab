"""Registry of reusable component factories.

Users can register custom component factories and later instantiate them by
name. Factories receive arbitrary positional/keyword arguments and must return a
:class:`spicelab.core.components.Component` instance.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass

from ..core.components import Component
from .metadata import validate_metadata

ComponentFactory = Callable[..., Component]


@dataclass(frozen=True)
class ComponentSpec:
    """Metadata describing a registered component factory."""

    name: str
    factory: ComponentFactory
    category: str | None = None
    metadata: Mapping[str, object] | None = None


_registry: MutableMapping[str, ComponentSpec] = {}


def register_component(
    name: str,
    factory: ComponentFactory,
    *,
    category: str | None = None,
    metadata: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> None:
    """Register a new component factory under ``name``.

    Parameters
    ----------
    name:
        Unique identifier. Names are case-sensitive; a dotted naming convention
        (``"diode.1n4007"``) is encouraged to group related parts.
    factory:
        Callable returning a :class:`Component` instance.
    category:
        Optional high-level category (e.g. ``"diode"``).
    metadata:
        Optional arbitrary information associated with this entry (e.g. data
        sheet URL, recommended ``.model`` line).
    overwrite:
        If ``False`` (default) attempting to re-register an existing name raises
        ``ValueError``.
    """

    if not overwrite and name in _registry:
        raise ValueError(f"Component '{name}' already registered")

    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping if provided")
    if category and metadata:
        validate_metadata(category, metadata)
    spec = ComponentSpec(name=name, factory=factory, category=category, metadata=metadata)
    _registry[name] = spec


def unregister_component(name: str) -> None:
    """Remove a previously registered component factory."""

    _registry.pop(name, None)


def get_component_spec(name: str) -> ComponentSpec:
    """Return the :class:`ComponentSpec` associated with ``name``."""

    try:
        return _registry[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Component '{name}' is not registered") from exc


def create_component(name: str, *args: object, **kwargs: object) -> Component:
    """Instantiate a registered component factory."""

    spec = get_component_spec(name)
    return spec.factory(*args, **kwargs)


def list_components(category: str | None = None) -> list[ComponentSpec]:
    """Return registered component specs, optionally filtered by ``category``."""

    specs: Iterable[ComponentSpec] = _registry.values()
    if category is not None:
        specs = [spec for spec in specs if spec.category == category]
    return list(specs)


__all__ = [
    "ComponentFactory",
    "ComponentSpec",
    "register_component",
    "unregister_component",
    "get_component_spec",
    "create_component",
    "list_components",
]


def search_components(
    *,
    name_contains: str | None = None,
    category: str | None = None,
    metadata: Mapping[str, object] | None = None,
    predicate: Callable[[ComponentSpec], bool] | None = None,
) -> list[ComponentSpec]:
    """Return component specs matching optional filters.

    Parameters
    ----------
    name_contains:
        Case-insensitive substring that must appear in the component name.
    category:
        Optional category filter (exact match).
    metadata:
        Mapping of metadata keys/values that must be present in the component
        metadata (comparison uses ``==``).
    predicate:
        Optional callable for custom filtering; receives the spec and must
        return ``True`` to keep it.
    """

    specs: Iterable[ComponentSpec] = _registry.values()

    if name_contains is not None:
        lowered = name_contains.lower()
        specs = [spec for spec in specs if lowered in spec.name.lower()]

    if category is not None:
        specs = [spec for spec in specs if spec.category == category]

    if metadata:

        def _metadata_matches(spec: ComponentSpec) -> bool:
            meta = spec.metadata or {}
            for key, expected in metadata.items():
                if meta.get(key) != expected:
                    return False
            return True

        specs = [spec for spec in specs if _metadata_matches(spec)]

    if predicate is not None:
        specs = [spec for spec in specs if predicate(spec)]

    return list(specs)


__all__.append("search_components")
