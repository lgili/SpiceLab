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


@dataclass(frozen=True)
class PhaseMarginSpec:
    """Phase margin at unity-gain crossover of H = numerator/denominator.

    Returns the classical PM = 180 + angle(H) [deg] at |H| = 1.
    """

    name: str
    numerator: str
    denominator: str


@dataclass(frozen=True)
class GainBandwidthSpec:
    """Unity-gain frequency for H = numerator/denominator (GBW for open-loop A)."""

    name: str
    numerator: str
    denominator: str


@dataclass(frozen=True)
class GainMarginSpec:
    """Gain margin at phase = -180° (mod 360) for H = numerator/denominator.

    Returns the classical GM in dB: GM_dB = -20*log10(|H|) evaluated at the
    phase-crossing (closest sample to -180° modulo 360). If no sample is within
    `tolerance_deg` of -180°, returns +inf (no crossing within range).
    """

    name: str
    numerator: str
    denominator: str
    tolerance_deg: float = 15.0


@dataclass(frozen=True)
class RiseTimeSpec:
    """10-90 (or custom) rise time between threshold crossings.

    Computes the time difference between the first crossings of low and high
    thresholds, where thresholds are defined relative to baseline/reference and
    target.
    """

    name: str
    signal: str
    target: float | None = None
    reference: float | None = None
    low_pct: float = 0.1
    high_pct: float = 0.9


@dataclass(frozen=True)
class THDSpec:
    """Total Harmonic Distortion of a steady-state tone.

    Returns THD in percent. Fundamental can be provided (f0); otherwise the
    dominant bin (ignoring DC) is used. Uses a Hann window by default via FFT helper.
    """

    name: str
    signal: str
    harmonics: int = 5
    f0: float | None = None


@dataclass(frozen=True)
class ENOBSpec:
    """Effective number of bits estimated from SINAD of a sine wave.

    ENOB = (SINAD_dB - 1.76) / 6.02
    """

    name: str
    signal: str
    harmonics: int = 5
    f0: float | None = None


Spec = (
    GainSpec
    | OvershootSpec
    | SettlingTimeSpec
    | PhaseMarginSpec
    | GainBandwidthSpec
    | GainMarginSpec
    | RiseTimeSpec
    | THDSpec
    | ENOBSpec
)


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
        # Preserve complex dtype when available (AC analysis)
        values = np.asarray(data.values)
        return SignalData(axis_name=axis_name, axis=axis, values=values)

    def _from_traces(self, key: str) -> SignalData:
        if self._traces is None:
            raise RuntimeError("Expected TraceSet source for measurement")
        trace = self._traces[key]
        axis_trace = self._traces.x
        axis = np.asarray(axis_trace.values, dtype=float)
        axis_name = axis_trace.name or "index"
        # Prefer complex values if available on the trace; fallback to magnitude.
        vals = getattr(trace, "values", None)
        if vals is not None:
            values = np.asarray(vals)
        else:
            values = np.asarray(trace.magnitude(), dtype=float)
        return SignalData(axis_name=axis_name, axis=axis, values=values)

    @staticmethod
    def _pick_axis_name(data: Any) -> str:
        preferred = ("time", "freq", "frequency")
        coords = getattr(data, "coords", {})
        for cand in preferred:
            if cand in coords:
                return cand
        dims = getattr(data, "dims", ())
        if dims:
            first_dim = dims[0]
            return str(first_dim)
        return "index"

    @staticmethod
    def _coord_values(data: Any, axis_name: str) -> np.ndarray:
        coords = getattr(data, "coords", {})
        if axis_name in coords:
            return np.asarray(coords[axis_name].values, dtype=float)
        # fallback to raw dimension coordinate
        coord = cast(Any, data)[axis_name]
        values = getattr(coord, "values", coord)
        arr = np.asarray(values, dtype=float)
        return cast(np.ndarray, arr)


def measure(
    source: ResultHandle | AnalysisResult | TraceSet | Any,
    specs: Sequence[Spec],
    *,
    return_as: Literal["polars", "python"] = "polars",
) -> Any:
    """Evaluate measurement specs and return a table of rows.

    By default returns a ``polars.DataFrame``; set return_as="python" to get
    a plain ``list[dict]`` which is convenient for lightweight environments/tests.
    """

    extractor = _SignalExtractor(source)
    rows = [_apply_spec(extractor, spec) for spec in specs]
    if return_as == "python":
        return rows
    pl = _import_polars()
    return pl.DataFrame(rows)


def _apply_spec(extractor: _SignalExtractor, spec: Spec) -> dict[str, float | str]:
    if isinstance(spec, GainSpec):
        return _measure_gain(extractor, spec)
    if isinstance(spec, OvershootSpec):
        return _measure_overshoot(extractor, spec)
    if isinstance(spec, SettlingTimeSpec):
        return _measure_settling_time(extractor, spec)
    if isinstance(spec, PhaseMarginSpec):
        return _measure_phase_margin(extractor, spec)
    if isinstance(spec, GainBandwidthSpec):
        return _measure_gain_bandwidth(extractor, spec)
    if isinstance(spec, GainMarginSpec):
        return _measure_gain_margin(extractor, spec)
    if isinstance(spec, RiseTimeSpec):
        return _measure_rise_time(extractor, spec)
    if isinstance(spec, THDSpec):
        return _measure_thd(extractor, spec)
    if isinstance(spec, ENOBSpec):
        return _measure_enob(extractor, spec)
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


def _complex_ratio(num: SignalData, den: SignalData) -> np.ndarray:
    _ensure_axis(num, den)
    num_v = np.asarray(num.values)
    den_v = np.asarray(den.values)
    # Guard zeros to avoid division warnings; let inf propagate meaningfully
    with np.errstate(divide="ignore", invalid="ignore"):
        h = num_v / den_v
    return np.asarray(h)


def _unwrap_deg(phase_rad: np.ndarray) -> np.ndarray:
    return np.asarray(np.unwrap(phase_rad) * (180.0 / np.pi))


def _nearest_index(arr: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(arr - target)))


def _safe_log10(x: np.ndarray) -> np.ndarray:
    # Guard extremely small or non-positive values for log scale
    x = np.asarray(x, dtype=float)
    eps = np.finfo(float).tiny
    out = np.log10(np.clip(x, eps, None))
    return np.asarray(out)


def _interp_f_at_y_logf(freq: np.ndarray, y: np.ndarray, target: float) -> float | None:
    """Interpolate frequency where y crosses target using linear interp vs log10(f).

    Returns None if no bracketing segment is found.
    """
    if freq.size < 2:
        return None
    xf = _safe_log10(freq)
    dy = y - target
    signs = np.sign(dy)
    # Find segments where the sign changes (crossing)
    changes = np.nonzero(signs[:-1] * signs[1:] <= 0)[0]
    if changes.size == 0:
        return None
    # If multiple, pick the one whose midpoint y is closest to target
    best_i = None
    best_err = float("inf")
    for i in changes:
        y_mid = 0.5 * (y[i] + y[i + 1])
        err = abs(y_mid - target)
        if err < best_err:
            best_err = err
            best_i = int(i)
    i = int(best_i) if best_i is not None else int(changes[0])
    x0, x1 = xf[i], xf[i + 1]
    y0, y1 = y[i], y[i + 1]
    if y1 == y0:
        xf_t = x0
    else:
        xf_t = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return float(10.0**xf_t)


def _interp_y_at_f_logf(freq: np.ndarray, y: np.ndarray, f_target: float) -> float:
    """Interpolate y at a given frequency using linear interp vs log10(f).

    Falls back to nearest sample when out-of-range or degenerate.
    """
    if freq.size < 2:
        return float(y[0]) if y.size else float("nan")
    xf = _safe_log10(freq)
    xf_t = np.log10(max(f_target, np.finfo(float).tiny))
    # Find neighbor indices
    idx = int(np.searchsorted(xf, xf_t))
    if idx <= 0:
        return float(y[0])
    if idx >= xf.size:
        return float(y[-1])
    x0, x1 = xf[idx - 1], xf[idx]
    y0, y1 = float(y[idx - 1]), float(y[idx])
    if x1 == x0:
        return float(y0)
    w = (xf_t - x0) / (x1 - x0)
    ret = (1.0 - w) * y0 + w * y1
    return float(ret)


def _measure_phase_margin(
    extractor: _SignalExtractor, spec: PhaseMarginSpec
) -> dict[str, float | str]:
    num = extractor.get(spec.numerator)
    den = extractor.get(spec.denominator)
    h = _complex_ratio(num, den)
    freq = num.axis
    mag = np.abs(h)
    phase_deg = _unwrap_deg(np.angle(h))
    # Interpolate unity-gain crossing in dB vs log f
    mag_db = 20.0 * np.log10(np.clip(mag, np.finfo(float).tiny, None))
    f_unity = _interp_f_at_y_logf(freq, mag_db, 0.0)
    if f_unity is None:
        # Fallback to nearest sample
        idx = _nearest_index(mag, 1.0)
        pm = 180.0 + float(phase_deg[idx])
        f_at = float(freq[idx])
    else:
        # Interpolate phase at f_unity (phase vs log f)
        phi_at = _interp_y_at_f_logf(freq, phase_deg, f_unity)
        pm = 180.0 + float(phi_at)
        f_at = float(f_unity)
    return {
        "measure": spec.name,
        "type": "phase_margin",
        "value": pm,
        "units": "deg",
        "freq": f_at,
        "numerator": spec.numerator,
        "denominator": spec.denominator,
    }


def _measure_gain_bandwidth(
    extractor: _SignalExtractor, spec: GainBandwidthSpec
) -> dict[str, float | str]:
    num = extractor.get(spec.numerator)
    den = extractor.get(spec.denominator)
    h = _complex_ratio(num, den)
    freq = num.axis
    mag = np.abs(h)
    mag_db = 20.0 * np.log10(np.clip(mag, np.finfo(float).tiny, None))
    f_unity = _interp_f_at_y_logf(freq, mag_db, 0.0)
    if f_unity is None:
        idx = _nearest_index(mag, 1.0)
        f_unity = float(freq[idx])
    return {
        "measure": spec.name,
        "type": "gbw",
        "value": f_unity,
        "units": "Hz",
        "numerator": spec.numerator,
        "denominator": spec.denominator,
    }


def _measure_gain_margin(
    extractor: _SignalExtractor, spec: GainMarginSpec
) -> dict[str, float | str]:
    num = extractor.get(spec.numerator)
    den = extractor.get(spec.denominator)
    h = _complex_ratio(num, den)
    freq = num.axis
    mag = np.abs(h)
    phase_deg = _unwrap_deg(np.angle(h))
    # Interpolate frequency where phase crosses -180 using phase vs log f
    phi_shift = phase_deg + 180.0
    f_phase = _interp_f_at_y_logf(freq, phi_shift, 0.0)
    if f_phase is None:
        # Fallback to previous heuristic near -180 with tolerance
        phi_mod = np.mod(np.mod(phase_deg, 360.0) + 360.0, 360.0)
        dist = np.abs(phi_mod - 180.0)
        candidates = np.flatnonzero(dist <= spec.tolerance_deg)
        if candidates.size == 0:
            idx = int(np.argmin(dist))
            gm_db = float("inf")
            f_at = float(freq[idx])
        else:
            local_mags = mag[candidates]
            best_local = int(np.argmin(np.abs(local_mags - 1.0)))
            idx = int(candidates[best_local])
            m = float(mag[idx])
            gm_db = -20.0 * np.log10(m) if m > 0 else float("inf")
            f_at = float(freq[idx])
    else:
        # Interpolate magnitude in dB at f_phase
        mag_db = 20.0 * np.log10(np.clip(mag, np.finfo(float).tiny, None))
        mag_db_at = _interp_y_at_f_logf(freq, mag_db, f_phase)
        gm_db = -float(mag_db_at)
        f_at = float(f_phase)
    return {
        "measure": spec.name,
        "type": "gain_margin",
        "value": float(gm_db),
        "units": "dB",
        "freq": f_at,
        "numerator": spec.numerator,
        "denominator": spec.denominator,
        "tolerance_deg": spec.tolerance_deg,
    }


def _interp_crossing(
    time: np.ndarray, values: np.ndarray, threshold: float, start_idx: int = 0
) -> float:
    """Find first threshold crossing at or after start_idx using linear interpolation.

    Returns NaN if not found.
    """
    from collections.abc import Callable

    rising = bool(values[-1] >= values[0])

    def _ge(a: float, b: float) -> bool:
        return a >= b

    def _le(a: float, b: float) -> bool:
        return a <= b

    cmp: Callable[[float, float], bool] = _ge if rising else _le
    for i in range(start_idx, len(values) - 1):
        a, b = float(values[i]), float(values[i + 1])
        ta, tb = float(time[i]), float(time[i + 1])
        if (cmp(a, threshold) and not cmp(a, threshold)) or (
            cmp(b, threshold) and not cmp(b, threshold)
        ):
            # unreachable branch; keep for symmetry
            pass
        # Check if threshold is bracketed between a and b
        if (a - threshold) * (b - threshold) <= 0.0:
            if a == b:
                return ta
            frac = (threshold - a) / (b - a)
            return ta + frac * (tb - ta)
    return float("nan")


def _measure_rise_time(extractor: _SignalExtractor, spec: RiseTimeSpec) -> dict[str, float | str]:
    sig = extractor.get(spec.signal)
    t = sig.axis
    y = np.asarray(sig.values, dtype=float)
    if y.size < 2:
        raise ValueError("Rise time requires at least 2 samples")
    y0 = float(y[0]) if spec.reference is None else float(spec.reference)
    y1 = float(y[-1]) if spec.target is None else float(spec.target)
    dy = y1 - y0
    lo = y0 + spec.low_pct * dy
    hi = y0 + spec.high_pct * dy
    t_lo = _interp_crossing(t, y, lo, 0)
    start_idx = int(np.searchsorted(t, t_lo)) if np.isfinite(t_lo) else 0
    t_hi = _interp_crossing(t, y, hi, start_idx)
    tr = float(t_hi - t_lo) if np.isfinite(t_lo) and np.isfinite(t_hi) else float("nan")
    return {
        "measure": spec.name,
        "type": "rise_time",
        "value": tr,
        "units": "s",
        "signal": spec.signal,
        "low_pct": spec.low_pct,
        "high_pct": spec.high_pct,
    }


def _fft_of_signal(sig: SignalData) -> tuple[np.ndarray, np.ndarray]:
    from .signal import rfft_coherent

    t = sig.axis
    y = np.asarray(sig.values, dtype=float)
    if t.size < 2:
        raise ValueError("Signal too short for FFT")
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt
    res = rfft_coherent(y, fs, win="hann")
    freq = res.freq
    mag = np.abs(res.spectrum)
    return freq, mag


def _find_fundamental(freq: np.ndarray, mag: np.ndarray, f0: float | None) -> int:
    # skip DC bin
    if f0 is not None:
        return int(np.argmin(np.abs(freq - f0)))
    idx = 1 + int(np.argmax(mag[1:]))
    return idx


def _measure_thd(extractor: _SignalExtractor, spec: THDSpec) -> dict[str, float | str]:
    sig = extractor.get(spec.signal)
    freq, mag = _fft_of_signal(sig)
    fund_idx = _find_fundamental(freq, mag, spec.f0)
    fund_mag = float(mag[fund_idx])
    n = spec.harmonics
    total_harm = 0.0
    for k in range(2, n + 1):
        idx = fund_idx * k
        if idx >= mag.size:
            break
        total_harm += float(mag[idx]) ** 2
    thd_ratio = np.sqrt(total_harm) / max(fund_mag, 1e-20)
    thd_pct = 100.0 * float(thd_ratio)
    return {
        "measure": spec.name,
        "type": "thd",
        "value": thd_pct,
        "units": "%",
        "signal": spec.signal,
        "harmonics": float(spec.harmonics),
        "f0": float(freq[fund_idx]),
    }


def _measure_enob(extractor: _SignalExtractor, spec: ENOBSpec) -> dict[str, float | str]:
    # Time-domain sine fitting to avoid FFT normalization pitfalls
    sig = extractor.get(spec.signal)
    t = sig.axis
    y = np.asarray(sig.values, dtype=float)
    if t.size < 4:
        raise ValueError("Signal too short for ENOB estimate")
    # Determine f0 if not provided
    f0 = spec.f0
    if f0 is None:
        freq, mag = _fft_of_signal(sig)
        idx = _find_fundamental(freq, mag, None)
        f0 = float(freq[idx])
    w0 = 2.0 * np.pi * float(f0)
    s = np.sin(w0 * t)
    c = np.cos(w0 * t)
    X = np.column_stack([s, c, np.ones_like(t)])
    coef_tuple = np.linalg.lstsq(X, y, rcond=None)
    coef = coef_tuple[0]
    a, b, dco = float(coef[0]), float(coef[1]), float(coef[2])
    y_hat = a * s + b * c + dco
    # Signal power is the mean-square of the AC component (exclude DC offset)
    signal_ac = a * s + b * c
    p_signal = float(np.mean(signal_ac**2))
    # Noise+distortion is the residual power
    resid = y - y_hat
    p_noise_dist = float(np.mean(resid**2))
    p_noise_dist = max(p_noise_dist, 1e-30)
    sinad_db = 10.0 * np.log10(p_signal / p_noise_dist)
    enob = (sinad_db - 1.76) / 6.02
    return {
        "measure": spec.name,
        "type": "enob",
        "value": float(enob),
        "units": "bits",
        "signal": spec.signal,
        "sinad_db": float(sinad_db),
        "f0": float(f0),
    }


__all__ = [
    "measure",
    "GainSpec",
    "OvershootSpec",
    "SettlingTimeSpec",
    "PhaseMarginSpec",
    "GainBandwidthSpec",
    "GainMarginSpec",
    "RiseTimeSpec",
    "THDSpec",
    "ENOBSpec",
]
