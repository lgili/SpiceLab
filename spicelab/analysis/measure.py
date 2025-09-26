"""Helpers that replicate classic SPICE ``.meas`` workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from ..core.types import ResultHandle
from ..io.raw_reader import TraceSet
from .result import AnalysisResult


@dataclass(frozen=True)
class SignalData:
    axis_name: str
    axis: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class GainSpec:
    """Measure the small-signal gain at a given frequency."""

    name: str
    numerator: str
    freq: float
    denominator: str | None = None
    kind: Literal["mag", "db"] = "db"


@dataclass(frozen=True)
class OvershootSpec:
    """Measure peak overshoot relative to a target value."""

    name: str
    signal: str
    target: float
    reference: float | None = None
    percent: bool = True


@dataclass(frozen=True)
class SettlingTimeSpec:
    """Measure when a signal stays within a tolerance band."""

    name: str
    signal: str
    target: float
    tolerance: float
    tolerance_kind: Literal["abs", "pct"] = "pct"
    start_time: float = 0.0


Spec = GainSpec | OvershootSpec | SettlingTimeSpec


def _import_polars() -> Any:
    try:
        return __import__("polars")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "polars is required for measure(); install with 'pip install polars'"
        ) from exc


class _SignalExtractor:
    def __init__(self, source: object) -> None:
        self._kind: Literal["dataset", "traces"]
        self._var_lookup: dict[str, str]
        self._dataset: Any | None = None
        self._traces: TraceSet | None = None
        if hasattr(source, "dataset"):
            dataset = source.dataset()
            self._dataset = dataset
            self._kind = "dataset"
            self._var_lookup = {name.lower(): name for name in getattr(dataset, "data_vars", {})}
        elif hasattr(source, "traces"):
            traces = source.traces
            self._traces = traces
            self._kind = "traces"
            self._var_lookup = {name.lower(): name for name in getattr(traces, "names", [])}
        elif isinstance(source, TraceSet):
            self._kind = "traces"
            self._traces = source
            self._var_lookup = {name.lower(): name for name in source.names}
        else:
            # assume xarray.Dataset-like
            self._kind = "dataset"
            self._dataset = source
            self._var_lookup = {name.lower(): name for name in getattr(source, "data_vars", {})}

        if not self._var_lookup:
            raise ValueError("No signals available for measurement")

    def get(self, name: str) -> SignalData:
        key = self._resolve(name)
        if self._kind == "dataset":
            return self._from_dataset(key)
        return self._from_traces(key)

    def _resolve(self, name: str) -> str:
        lowered = name.lower()
        if lowered in self._var_lookup:
            return self._var_lookup[lowered]
        raise KeyError(f"Signal '{name}' not available. Known: {tuple(self._var_lookup.values())}")

    def _from_dataset(self, key: str) -> SignalData:
        if self._dataset is None:
            raise RuntimeError("Expected dataset source for measurement")
        ds = self._dataset
        data = ds[key]
        if not getattr(data, "dims", None):
            raise ValueError(f"Signal '{key}' is scalar; cannot measure")
        axis_name = self._pick_axis_name(data)
        axis = self._coord_values(data, axis_name)
        values = np.asarray(data.values, dtype=float)
        return SignalData(axis_name=axis_name, axis=axis, values=values)

    def _from_traces(self, key: str) -> SignalData:
        if self._traces is None:
            raise RuntimeError("Expected TraceSet source for measurement")
        trace = self._traces[key]
        axis_trace = self._traces.x
        axis = np.asarray(axis_trace.values, dtype=float)
        axis_name = axis_trace.name or "index"
        values = np.asarray(trace.magnitude(), dtype=float)
        return SignalData(axis_name=axis_name, axis=axis, values=values)

    @staticmethod
    def _pick_axis_name(data: Any) -> str:
        preferred = ("time", "freq", "frequency")
        coords = getattr(data, "coords", {})
        for cand in preferred:
            if cand in coords:
                return cand
        if data.dims:
            first_dim = data.dims[0]
            return cast(str, first_dim)
        return "index"

    @staticmethod
    def _coord_values(data: Any, axis_name: str) -> np.ndarray:
        coords = getattr(data, "coords", {})
        if axis_name in coords:
            return np.asarray(coords[axis_name].values, dtype=float)
        # fallback to raw dimension coordinate
        coord = cast(Any, data)[axis_name]
        values = getattr(coord, "values", coord)
        return np.asarray(values, dtype=float)


def measure(source: ResultHandle | AnalysisResult | TraceSet | Any, specs: Sequence[Spec]) -> Any:
    """Evaluate measurement specs and return a ``polars.DataFrame``."""

    extractor = _SignalExtractor(source)
    rows = [_apply_spec(extractor, spec) for spec in specs]
    pl = _import_polars()
    return pl.DataFrame(rows)


def _apply_spec(extractor: _SignalExtractor, spec: Spec) -> dict[str, float | str]:
    if isinstance(spec, GainSpec):
        return _measure_gain(extractor, spec)
    if isinstance(spec, OvershootSpec):
        return _measure_overshoot(extractor, spec)
    if isinstance(spec, SettlingTimeSpec):
        return _measure_settling_time(extractor, spec)
    raise TypeError(f"Unsupported spec: {spec!r}")


def _measure_gain(extractor: _SignalExtractor, spec: GainSpec) -> dict[str, float | str]:
    signal = extractor.get(spec.numerator)
    denom = None
    if spec.denominator:
        denom = extractor.get(spec.denominator)
        _ensure_axis(signal, denom)

    axis = signal.axis
    if axis.size == 0:
        raise ValueError("Gain measurement requires non-empty signal")
    freq_idx = int(np.argmin(np.abs(axis - spec.freq)))
    freq_val = float(axis[freq_idx])
    num_val = float(np.abs(signal.values[freq_idx]))
    denom_val = 1.0
    if denom is not None:
        denom_val = float(np.abs(denom.values[freq_idx]))
        if denom_val == 0.0:
            return {
                "measure": spec.name,
                "type": "gain",
                "value": float("inf"),
                "units": "dB" if spec.kind == "db" else "V/V",
                "freq": freq_val,
            }

    ratio = num_val / denom_val
    if spec.kind == "db":
        value = 20.0 * np.log10(ratio) if ratio > 0 else float("-inf")
        units = "dB"
    else:
        value = ratio
        units = "V/V"
    return {
        "measure": spec.name,
        "type": "gain",
        "value": float(value),
        "units": units,
        "freq": freq_val,
        "numerator": spec.numerator,
        "denominator": spec.denominator or "1",
    }


def _measure_overshoot(extractor: _SignalExtractor, spec: OvershootSpec) -> dict[str, float | str]:
    signal = extractor.get(spec.signal)
    values = signal.values
    if values.size == 0:
        raise ValueError("Overshoot measurement requires non-empty signal")
    baseline = spec.reference if spec.reference is not None else float(values[0])
    amplitude = spec.target - baseline
    if amplitude == 0:
        overshoot = 0.0
    else:
        if spec.target >= baseline:
            peak_value = float(np.max(values))
        else:
            peak_value = float(np.min(values))
        overshoot = (peak_value - spec.target) / amplitude
    if spec.percent:
        overshoot *= 100.0
        units = "%"
    else:
        units = "ratio"
    idx_peak = int(np.argmax(values))
    peak_time = float(signal.axis[idx_peak])
    return {
        "measure": spec.name,
        "type": "overshoot",
        "value": float(overshoot),
        "units": units,
        "signal": spec.signal,
        "peak_time": peak_time,
        "target": spec.target,
    }


def _measure_settling_time(
    extractor: _SignalExtractor, spec: SettlingTimeSpec
) -> dict[str, float | str]:
    signal = extractor.get(spec.signal)
    time = signal.axis
    values = signal.values
    if values.size == 0:
        raise ValueError("Settling time measurement requires non-empty signal")
    if time.size != values.size:
        raise ValueError("Signal axis and values must align")
    tol_abs = spec.tolerance
    if spec.tolerance_kind == "pct":
        tol_abs = abs(spec.target) * spec.tolerance
    lower = spec.target - tol_abs
    upper = spec.target + tol_abs
    settled_time = float("nan")
    mask = time >= spec.start_time
    candidate_indices = np.flatnonzero(mask)
    for idx in candidate_indices:
        window = values[idx:]
        if np.all((window >= lower) & (window <= upper)):
            settled_time = float(time[idx])
            break
    return {
        "measure": spec.name,
        "type": "settling",
        "value": settled_time,
        "units": "s",
        "signal": spec.signal,
        "target": spec.target,
        "tolerance": tol_abs,
        "tolerance_kind": spec.tolerance_kind,
    }


def _ensure_axis(a: SignalData, b: SignalData) -> None:
    if a.axis.shape != b.axis.shape:
        raise ValueError("Signals must share the same axis for gain measurement")
    if not np.allclose(a.axis, b.axis):
        raise ValueError("Signal axes do not match for gain measurement")


__all__ = [
    "measure",
    "GainSpec",
    "OvershootSpec",
    "SettlingTimeSpec",
]
