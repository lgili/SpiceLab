"""Utilities for applying component metadata to circuits."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterable as TypingIterable

from ..core.circuit import Circuit
from .registry import ComponentSpec


def _coerce_to_lines(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        lines: list[str] = []
        for item in value:
            if item is None:
                continue
            lines.append(str(item))
        return lines
    return []


def apply_metadata_to_circuit(
    circuit: Circuit,
    spec: ComponentSpec,
    *,
    include_keys: TypingIterable[str] = ("include", "includes"),
    model_keys: TypingIterable[str] = ("model_card",),
    directive_keys: TypingIterable[str] = (),
) -> list[str]:
    """Insert model/include directives from a component spec into ``circuit``.

    Returns a list of directives actually added (new lines only)."""

    added: list[str] = []
    metadata = spec.metadata or {}

    for key in include_keys:
        value = metadata.get(key)
        for raw in _coerce_to_lines(value):
            line = raw if raw.strip().lower().startswith(".include") else f".include {raw}"
            before = len(circuit._directives)
            circuit.add_directive_once(line)
            if len(circuit._directives) > before:
                added.append(line)

    for key in model_keys:
        value = metadata.get(key)
        for line in _coerce_to_lines(value):
            before = len(circuit._directives)
            circuit.add_directive_once(line)
            if len(circuit._directives) > before:
                added.append(line)

    for key in directive_keys:
        value = metadata.get(key)
        for line in _coerce_to_lines(value):
            before = len(circuit._directives)
            circuit.add_directive_once(line)
            if len(circuit._directives) > before:
                added.append(line)

    return added


__all__ = ["apply_metadata_to_circuit"]
