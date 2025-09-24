from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from typing import Any, Literal, Protocol, runtime_checkable

# ---- Analysis & Sweep contracts (stable surface for engines/orchestrator) ----

AnalysisMode = Literal["op", "dc", "ac", "tran", "noise"]


@dataclass(frozen=True)
class AnalysisSpec:
    """Specifies a single analysis type and its arguments.

    Arguments are backend-agnostic (strings or numbers). Backends are responsible
    for mapping these to their dialect (cards/CLI args).
    """

    mode: AnalysisMode
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SweepSpec:
    """Declarative sweep definition.

    variables: mapping of parameter name to list of numeric values.
    Backends expand these according to their native support (e.g. .step / param).
    """

    variables: dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class ResultMeta:
    """Minimal metadata attached to simulation results/datasets."""

    engine: str
    engine_version: str | None = None
    netlist_hash: str | None = None
    analyses: list[AnalysisSpec] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ResultHandle(Protocol):
    """A lightweight handle to access result data in common formats.

    Implementations should provide zero/lazy copy access wherever possible.
    """

    def dataset(self) -> Any:  # expected: xarray.Dataset
        """Return results as an xarray.Dataset (or raise if unavailable)."""

    def to_polars(self) -> Any:  # expected: polars.DataFrame
        """Return results as a polars.DataFrame (or raise if unavailable)."""

    def attrs(self) -> Mapping[str, Any]:
        """Return a mapping with metadata attributes."""


# ---- Hash helpers (deterministic) ----


def _to_canonical(obj: Any) -> Any:
    """Convert known structures to JSON-serializable stable representation.

    - dataclasses -> dict
    - sets -> sorted lists
    - mapping -> sorted items by key
    - objects with ``build_netlist()`` -> that string (useful for Circuit)
    - fallback to ``repr``
    """

    try:
        from dataclasses import is_dataclass

        # asdict expects an instance, not a dataclass type
        if is_dataclass(obj) and not isinstance(obj, type):
            return {k: _to_canonical(v) for k, v in asdict(obj).items()}
    except Exception:
        pass

    if hasattr(obj, "build_netlist") and callable(obj.build_netlist):
        try:
            return str(obj.build_netlist())
        except Exception:
            return repr(obj)

    if isinstance(obj, dict):
        return {str(k): _to_canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, list | tuple):
        return [_to_canonical(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_to_canonical(v) for v in obj)
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    return repr(obj)


def stable_hash(obj: Any) -> str:
    """Return a deterministic short hash (sha1 hex, 12 chars) for ``obj``.

    Uses a canonical JSON representation so ordering differences don't change the hash.
    """

    canonical = _to_canonical(obj)
    data = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return sha1(data.encode("utf-8")).hexdigest()[:12]


def circuit_hash(circuit: Any, *, extra: Mapping[str, Any] | None = None) -> str:
    """Hash a Circuit-like object deterministically.

    If ``circuit`` provides ``build_netlist()``, that string is used as core input.
    ``extra`` can inject engine/version/args to bind caches to full run context.
    """

    payload: dict[str, Any] = {"circuit": circuit}
    if extra:
        # sort by key for stability
        pairs = ((str(k), v) for k, v in extra.items())
        payload["extra"] = dict(sorted(pairs, key=lambda kv: kv[0]))
    return stable_hash(payload)


__all__ = [
    "AnalysisMode",
    "AnalysisSpec",
    "SweepSpec",
    "ResultMeta",
    "ResultHandle",
    "stable_hash",
    "circuit_hash",
]
