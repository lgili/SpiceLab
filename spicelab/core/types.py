from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from typing import Any, Literal, Protocol, cast, runtime_checkable

try:  # pragma: no cover - import guard
    from pydantic import BaseModel, Field, field_validator, model_validator
except Exception as _pye:  # pragma: no cover
    raise RuntimeError(
        "pydantic is required after M1 refactor. Install with 'pip install pydantic>=2.7'."
    ) from _pye

# ---- Analysis & Sweep contracts (stable surface for engines/orchestrator) ----

AnalysisMode = Literal["op", "dc", "ac", "tran", "noise"]


class AnalysisSpec(BaseModel):  # Pydantic model (breaking change M1 full)
    """Specifies a single analysis type and its arguments (validated)."""

    mode: AnalysisMode
    args: dict[str, Any] = Field(default_factory=dict)

    # Backward-compatible positional init: AnalysisSpec(mode, args)
    def __init__(
        self,
        mode: AnalysisMode | None = None,
        args: dict[str, Any] | None = None,
        **data: Any,
    ):
        if mode is not None:
            data.setdefault("mode", mode)
        if args is not None:
            data.setdefault("args", args)
        super().__init__(**data)

    @field_validator("args")
    @classmethod
    def _args_plain(cls, v: dict[str, Any]) -> dict[str, Any]:
        # ensure primitives or simple sequences only
        for k, val in v.items():
            if k in {"callbacks", "external_sources"} and isinstance(val, dict):
                for item in val.values():
                    if item is None:
                        continue
                    if not callable(item):
                        raise ValueError(f"args[{k}] requires callables or None, got {type(item)}")
                continue
            if isinstance(val, list | tuple):
                for item in val:
                    if not isinstance(item, str | int | float | bool) and item is not None:
                        raise ValueError(f"args[{k}] element not primitive: {item!r}")
            elif not isinstance(val, str | int | float | bool) and val is not None:
                raise ValueError(f"args[{k}] must be primitive, got {type(val)}")
        return v


class SweepSpec(BaseModel):
    """Declarative sweep definition with unit/number normalization."""

    variables: dict[str, list[float | str]] = Field(default_factory=dict)

    def __init__(self, variables: dict[str, list[float | str]] | None = None, **data: Any):
        if variables is not None:
            data.setdefault("variables", variables)
        super().__init__(**data)

    @model_validator(mode="after")
    def _normalize(self) -> SweepSpec:
        from ..utils.units import to_float

        norm: dict[str, list[float]] = {}
        for name, seq in self.variables.items():
            if not seq:
                raise ValueError(f"Sweep variable '{name}' empty")
            parsed: list[float] = []
            for v in seq:
                if isinstance(v, int | float):
                    parsed.append(float(v))
                elif isinstance(v, str):
                    parsed.append(to_float(v))
                else:
                    raise ValueError(f"Unsupported sweep value type: {type(v)}")
            norm[name] = parsed
        object.__setattr__(self, "variables", norm)  # bypass pydantic immutability
        return self


@dataclass(frozen=True)
class ResultMeta:
    engine: str
    engine_version: str | None = None
    netlist_hash: str | None = None
    analyses: list[AnalysisSpec] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


class Probe(BaseModel):
    """Measurement point specification.

    kind:
      - voltage: target is node name (e.g. 'out')
      - current: target is element reference (e.g. 'R1' for current through resistor)
    """

    kind: Literal["voltage", "current"]
    target: str

    @field_validator("target")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("target must be non-empty")
        return v

    # Convenience helpers
    @classmethod
    def v(cls, node: str) -> Probe:  # voltage probe
        return cls(kind="voltage", target=node)

    @classmethod
    def i(cls, ref: str) -> Probe:  # current probe
        return cls(kind="current", target=ref)


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

    # Pydantic models (v2) provide model_dump
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            dumped = obj.model_dump()
            return {k: _to_canonical(v) for k, v in sorted(dumped.items(), key=lambda kv: kv[0])}
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
    "Probe",
    "ResultHandle",
    "stable_hash",
    "circuit_hash",
]

# ----------------------------------------------------------------------------------
# Normalizers to/from Pydantic models (introduced in migration M1 full)
# ----------------------------------------------------------------------------------
# (Legacy Pydantic migration placeholders removed; models consolidated here.)


def ensure_analysis_spec(obj: AnalysisSpec | dict[str, object] | object) -> AnalysisSpec:
    """Normalize arbitrary input into an ``AnalysisSpec``.

    Accepted forms:
      - ``AnalysisSpec`` instance (returned as-is)
      - ``dict`` with keys ``mode`` and optional ``args`` (values validated)

    Raises:
      - ``ValueError`` for invalid mode or args type
      - ``TypeError`` for unsupported input type
    """
    if isinstance(obj, AnalysisSpec):  # fast path
        return obj
    if isinstance(obj, dict):
        mode = obj.get("mode")
        if mode not in ("op", "dc", "ac", "tran", "noise"):
            raise ValueError(f"Invalid analysis mode: {mode!r}")
        args = obj.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("args must be a dict")
        mode_lit = cast(AnalysisMode, mode)
        return AnalysisSpec(mode=mode_lit, args=dict(args))
    raise TypeError(f"Cannot convert to AnalysisSpec: {obj!r}")


def ensure_sweep_spec(obj: SweepSpec | dict[str, object] | object | None) -> SweepSpec | None:
    if obj is None:
        return None
    if isinstance(obj, SweepSpec):
        return obj
    if isinstance(obj, dict):
        vars_obj = obj.get("variables", {})
        if not isinstance(vars_obj, dict):
            raise ValueError("variables must be a dict")
        converted: dict[str, list[float | str]] = {}
        for name, seq in vars_obj.items():
            if not isinstance(seq, list):
                raise ValueError(f"Sweep variable '{name}' must be a list")
            converted[name] = list(seq)  # keep original (normalização no modelo)
        return SweepSpec(variables=converted)
    # (No legacy model types anymore.)
    raise TypeError(f"Cannot convert to SweepSpec: {obj!r}")


def ensure_probe(obj: Probe | dict[str, str] | str) -> Probe:
    """Normalize flexible probe input to a ``Probe``.

    Accepted forms:
      - ``Probe`` instance
      - ``dict``: ``{"kind": "voltage"|"current", "target": <str>}``
      - string shorthand: ``V(node)``, ``I(R1)`` (case-insensitive) or bare ``node`` (voltage)

    Bare string fallback: treated as a voltage probe on the given node.
    """
    if isinstance(obj, Probe):
        return obj
    if isinstance(obj, dict):
        k = obj.get("kind")
        t = obj.get("target")
        if k not in ("voltage", "current") or not isinstance(t, str):
            raise ValueError(f"Invalid probe dict: {obj!r}")
        k_lit = cast(Literal["voltage", "current"], k)
        return Probe(kind=k_lit, target=t)
    if isinstance(obj, str):
        s = obj.strip()
        if s.lower().startswith("v(") and s.endswith(")"):
            inner = s[2:-1]
            return Probe(kind="voltage", target=inner)
        if s.lower().startswith("i(") and s.endswith(")"):
            inner = s[2:-1]
            return Probe(kind="current", target=inner)
        # bare node name -> assume voltage
        return Probe.v(s)
    raise TypeError(f"Cannot convert to Probe: {obj!r}")


def ensure_probes(seq: list[Probe | dict[str, str] | str] | None) -> list[Probe]:
    """Normalize a sequence of heterogeneous probe inputs to concrete ``Probe`` objects.

    Returns an empty list if ``seq`` is falsy. Does not enforce uniqueness.
    """
    if not seq:
        return []
    return [ensure_probe(p) for p in seq]


__all__ += ["ensure_analysis_spec", "ensure_sweep_spec", "ensure_probe", "ensure_probes"]
