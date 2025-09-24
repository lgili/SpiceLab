"""Helpers for importing external component definitions into the registry."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Callable as _Callable
from pathlib import Path
from typing import Any, cast

from .registry import register_component


class ImportError(Exception):
    """Raised when importing a catalog fails."""


def _ensure_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise ImportError(f"File not found: {p}")
    return p


def import_catalog_from_mapping(
    entries: Iterable[Mapping[str, Any]],
    *,
    name_key: str = "name",
    factory_key: str = "factory",
    category_key: str | None = "category",
    metadata_key: str | None = "metadata",
    factory_builder: Callable[[Mapping[str, Any]], Callable[..., Any]] | None = None,
    overwrite: bool = False,
) -> None:
    """Register entries from an in-memory iterable of mappings.

    Each entry must include at least `name` and `factory` information. If
    `factory_builder` is provided it will be called with the entry mapping and
    must return the callable to register.
    """

    for entry in entries:
        try:
            name = str(entry[name_key])
        except KeyError as exc:  # pragma: no cover - defensive
            raise ImportError("Missing 'name' field in catalog entry") from exc

        factory: Any = None
        if factory_builder is not None:
            factory = factory_builder(entry)
        else:
            factory = entry.get(factory_key)
        if not callable(factory):
            raise ImportError(f"Entry '{name}' missing or invalid factory information")

        # mypy: entry.get(...) can be Any | None; cast to a callable after runtime check
        factory_callable = cast(_Callable[..., Any], factory)

        category = entry.get(category_key) if category_key else None
        metadata = entry.get(metadata_key) if metadata_key else None

        register_component(
            name,
            factory_callable,
            category=category,
            metadata=metadata,
            overwrite=overwrite,
        )


def import_catalog_from_json(
    path: str | Path,
    *,
    factory_builder: Callable[[Mapping[str, Any]], Callable[..., Any]] | None = None,
    overwrite: bool = False,
) -> None:
    """Load a JSON file containing a list of entries and register them."""

    data = json.loads(_ensure_path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Sequence):
        raise ImportError("JSON catalog must contain a list of entries")
    import_catalog_from_mapping(
        data,
        factory_builder=factory_builder,
        overwrite=overwrite,
    )


def import_catalog_from_csv(
    path: str | Path,
    *,
    factory_builder: Callable[[Mapping[str, Any]], Callable[..., Any]] | None = None,
    overwrite: bool = False,
) -> None:
    """Load a CSV file and register entries.

    The CSV header must include at least a `name` column. If `factory_builder`
    is not provided, a column `factory` is expected and should refer to a
    dotted-path to a callable (importable via ``importlib``). For custom
    factories it's recommended to provide a builder.
    """

    path = _ensure_path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        import_catalog_from_mapping(
            reader,
            factory_builder=factory_builder,
            overwrite=overwrite,
        )


__all__ = [
    "ImportError",
    "import_catalog_from_mapping",
    "import_catalog_from_json",
    "import_catalog_from_csv",
]
