"""Registry of reusable component factories.

Users can register custom component factories and later instantiate them by
name. Factories receive arbitrary positional/keyword arguments and must return a
:class:`cat.core.components.Component` instance.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass

from ..core.components import Component

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
