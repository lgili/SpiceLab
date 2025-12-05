"""Transient analysis measurements.

Measurements for time-domain analysis including:
- Slew rate (dV/dt)
- Settling time
- Rise time / Fall time
- Overshoot / Undershoot
- Propagation delay
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import BaseMeasurement, MeasurementResult
from .registry import measurement

if TYPE_CHECKING:
    import xarray as xr


def _find_threshold_crossing(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float,
    rising: bool = True,
    start_idx: int = 0,
) -> tuple[float | None, int | None]:
    """Find time where signal crosses threshold.

    Args:
        time: Time array
        values: Signal values
        threshold: Threshold value
        rising: True for rising edge, False for falling
        start_idx: Index to start searching from

    Returns:
        (crossing_time, index) or (None, None) if not found
    """
    for i in range(start_idx, len(values) - 1):
        v0, v1 = values[i], values[i + 1]
        t0, t1 = time[i], time[i + 1]

        if rising:
            crossed = v0 < threshold <= v1
        else:
            crossed = v0 > threshold >= v1

        if crossed:
            # Linear interpolation
            if v1 == v0:
                t_cross = t0
            else:
                frac = (threshold - v0) / (v1 - v0)
                t_cross = t0 + frac * (t1 - t0)
            return float(t_cross), i

    return None, None


@measurement("slew_rate")
class SlewRateMeasurement(BaseMeasurement):
    """Measure slew rate (dV/dt) of a signal.

    Calculates the rate of change between low and high thresholds
    (default 10% to 90% of signal swing).

    Example:
        >>> sr = SlewRateMeasurement(node="vout", edge="rising")
        >>> result = sr.measure(dataset)
        >>> print(f"Slew rate: {result.value/1e6:.2f} V/µs")
    """

    name = "slew_rate"
    description = "Maximum rate of change (dV/dt)"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        edge: str = "rising",
        low_pct: float = 0.1,
        high_pct: float = 0.9,
    ):
        """Initialize slew rate measurement.

        Args:
            node: Signal node name
            edge: "rising", "falling", or "both"
            low_pct: Low threshold as fraction of swing (default 0.1 = 10%)
            high_pct: High threshold as fraction of swing (default 0.9 = 90%)
        """
        self.node = node
        self.edge = edge
        self.low_pct = low_pct
        self.high_pct = high_pct

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate slew rate from transient data."""
        # Get time and signal
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        sig_key = self._find_signal_key(dataset, self.node)
        voltage = np.asarray(dataset[sig_key].values)

        # Normalize signal to 0-1
        v_min, v_max = float(voltage.min()), float(voltage.max())
        v_swing = v_max - v_min

        if v_swing == 0:
            return MeasurementResult(
                value=0.0,
                unit="V/s",
                metadata={"error": "No signal swing detected"},
            )

        # Calculate thresholds
        v_low = v_min + self.low_pct * v_swing
        v_high = v_min + self.high_pct * v_swing

        slew_rates = []

        if self.edge in ("rising", "both"):
            t_low, idx_low = _find_threshold_crossing(
                time_values, voltage, v_low, rising=True
            )
            if t_low is not None and idx_low is not None:
                t_high, _ = _find_threshold_crossing(
                    time_values, voltage, v_high, rising=True, start_idx=idx_low
                )
                if t_high is not None:
                    dt = t_high - t_low
                    dv = (self.high_pct - self.low_pct) * v_swing
                    if dt > 0:
                        slew_rates.append(dv / dt)

        if self.edge in ("falling", "both"):
            t_high, idx_high = _find_threshold_crossing(
                time_values, voltage, v_high, rising=False
            )
            if t_high is not None and idx_high is not None:
                t_low, _ = _find_threshold_crossing(
                    time_values, voltage, v_low, rising=False, start_idx=idx_high
                )
                if t_low is not None:
                    dt = t_low - t_high
                    dv = (self.high_pct - self.low_pct) * v_swing
                    if dt > 0:
                        slew_rates.append(dv / dt)

        if not slew_rates:
            return MeasurementResult(
                value=float("nan"),
                unit="V/s",
                metadata={"error": "Could not find threshold crossings"},
            )

        # Return average or max depending on edge type
        if self.edge == "both":
            slew = sum(slew_rates) / len(slew_rates)
        else:
            slew = slew_rates[0]

        return MeasurementResult(
            value=slew,
            unit="V/s",
            metadata={
                "edge": self.edge,
                "v_swing": v_swing,
                "low_threshold": v_low,
                "high_threshold": v_high,
            },
        )

    def _find_signal_key(self, dataset: xr.Dataset, node: str) -> str:
        if node in dataset.data_vars:
            return node
        node_lower = node.lower()
        for key in dataset.data_vars:
            key_str = str(key)
            if key_str.lower() == node_lower:
                return key_str
        raise KeyError(f"Signal '{node}' not found in dataset")


@measurement("settling_time")
class SettlingTimeMeasurement(BaseMeasurement):
    """Measure settling time to within an error band.

    Finds the time after which the signal stays within a specified
    percentage of the final value.

    Example:
        >>> st = SettlingTimeMeasurement(node="vout", error_pct=1.0)
        >>> result = st.measure(dataset)
        >>> print(f"Settling time: {result.value*1e6:.2f} µs")
    """

    name = "settling_time"
    description = "Time to settle within error band"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        final_value: float | None = None,
        error_pct: float = 1.0,
    ):
        """Initialize settling time measurement.

        Args:
            node: Signal node name
            final_value: Expected final value. None = auto-detect from last 10%
            error_pct: Error band as percentage (default 1.0 = ±1%)
        """
        self.node = node
        self.final_value = final_value
        self.error_pct = error_pct

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate settling time."""
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        sig_key = self._find_signal_key(dataset, self.node)
        voltage = np.asarray(dataset[sig_key].values)

        # Determine final value
        if self.final_value is None:
            # Use mean of last 10% of samples
            n_samples = max(1, len(voltage) // 10)
            final_value = float(np.mean(voltage[-n_samples:]))
        else:
            final_value = self.final_value

        # Calculate error band
        error_band = abs(final_value * self.error_pct / 100.0)
        upper = final_value + error_band
        lower = final_value - error_band

        # Find last time outside error band
        outside = (voltage < lower) | (voltage > upper)
        idx_outside = np.where(outside)[0]

        if len(idx_outside) == 0:
            settling_time = float(time_values[0])
        else:
            settling_time = float(time_values[idx_outside[-1]])

        return MeasurementResult(
            value=settling_time,
            unit="s",
            metadata={
                "final_value": final_value,
                "error_band_pct": self.error_pct,
                "error_band": error_band,
                "upper_bound": upper,
                "lower_bound": lower,
            },
        )

    def _find_signal_key(self, dataset: xr.Dataset, node: str) -> str:
        if node in dataset.data_vars:
            return node
        node_lower = node.lower()
        for key in dataset.data_vars:
            key_str = str(key)
            if key_str.lower() == node_lower:
                return key_str
        raise KeyError(f"Signal '{node}' not found in dataset")


@measurement("rise_time")
class RiseTimeMeasurement(BaseMeasurement):
    """Measure rise time (10%-90% or custom thresholds).

    Example:
        >>> rt = RiseTimeMeasurement(node="vout")
        >>> result = rt.measure(dataset)
        >>> print(f"Rise time: {result.value*1e9:.2f} ns")
    """

    name = "rise_time"
    description = "Time from low to high threshold"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        target: float | None = None,
        reference: float | None = None,
        low_pct: float = 0.1,
        high_pct: float = 0.9,
    ):
        """Initialize rise time measurement.

        Args:
            node: Signal node name
            target: High level (None = auto from signal max)
            reference: Low level (None = auto from signal min)
            low_pct: Low threshold fraction (default 0.1 = 10%)
            high_pct: High threshold fraction (default 0.9 = 90%)
        """
        self.node = node
        self.target = target
        self.reference = reference
        self.low_pct = low_pct
        self.high_pct = high_pct

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate rise time."""
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        sig_key = self._find_signal_key(dataset, self.node)
        voltage = np.asarray(dataset[sig_key].values)

        # Determine reference and target levels
        y0 = float(voltage[0]) if self.reference is None else self.reference
        y1 = float(voltage[-1]) if self.target is None else self.target

        # Calculate thresholds
        dy = y1 - y0
        v_low = y0 + self.low_pct * dy
        v_high = y0 + self.high_pct * dy

        # Find crossings
        t_low, idx_low = _find_threshold_crossing(
            time_values, voltage, v_low, rising=True
        )
        if t_low is None or idx_low is None:
            return MeasurementResult(
                value=float("nan"),
                unit="s",
                metadata={"error": "Low threshold crossing not found"},
            )

        t_high, _ = _find_threshold_crossing(
            time_values, voltage, v_high, rising=True, start_idx=idx_low
        )
        if t_high is None:
            return MeasurementResult(
                value=float("nan"),
                unit="s",
                metadata={"error": "High threshold crossing not found"},
            )

        rise_time = t_high - t_low

        return MeasurementResult(
            value=rise_time,
            unit="s",
            metadata={
                "t_low": t_low,
                "t_high": t_high,
                "v_low": v_low,
                "v_high": v_high,
                "low_pct": self.low_pct,
                "high_pct": self.high_pct,
            },
        )

    def _find_signal_key(self, dataset: xr.Dataset, node: str) -> str:
        if node in dataset.data_vars:
            return node
        node_lower = node.lower()
        for key in dataset.data_vars:
            key_str = str(key)
            if key_str.lower() == node_lower:
                return key_str
        raise KeyError(f"Signal '{node}' not found in dataset")


@measurement("fall_time")
class FallTimeMeasurement(BaseMeasurement):
    """Measure fall time (90%-10% or custom thresholds).

    Example:
        >>> ft = FallTimeMeasurement(node="vout")
        >>> result = ft.measure(dataset)
        >>> print(f"Fall time: {result.value*1e9:.2f} ns")
    """

    name = "fall_time"
    description = "Time from high to low threshold"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        target: float | None = None,
        reference: float | None = None,
        high_pct: float = 0.9,
        low_pct: float = 0.1,
    ):
        """Initialize fall time measurement.

        Args:
            node: Signal node name
            target: Low level (None = auto from signal)
            reference: High level (None = auto from signal)
            high_pct: High threshold fraction (default 0.9 = 90%)
            low_pct: Low threshold fraction (default 0.1 = 10%)
        """
        self.node = node
        self.target = target
        self.reference = reference
        self.high_pct = high_pct
        self.low_pct = low_pct

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate fall time."""
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        sig_key = self._find_signal_key(dataset, self.node)
        voltage = np.asarray(dataset[sig_key].values)

        # Determine reference (high) and target (low) levels
        v_min = float(voltage.min())
        v_max = float(voltage.max())
        y_high = v_max if self.reference is None else self.reference
        y_low = v_min if self.target is None else self.target

        # Calculate thresholds
        dy = y_high - y_low
        v_high = y_low + self.high_pct * dy
        v_low = y_low + self.low_pct * dy

        # Find crossings (falling edge)
        t_high, idx_high = _find_threshold_crossing(
            time_values, voltage, v_high, rising=False
        )
        if t_high is None or idx_high is None:
            return MeasurementResult(
                value=float("nan"),
                unit="s",
                metadata={"error": "High threshold crossing not found"},
            )

        t_low, _ = _find_threshold_crossing(
            time_values, voltage, v_low, rising=False, start_idx=idx_high
        )
        if t_low is None:
            return MeasurementResult(
                value=float("nan"),
                unit="s",
                metadata={"error": "Low threshold crossing not found"},
            )

        fall_time = t_low - t_high

        return MeasurementResult(
            value=fall_time,
            unit="s",
            metadata={
                "t_high": t_high,
                "t_low": t_low,
                "v_high": v_high,
                "v_low": v_low,
                "high_pct": self.high_pct,
                "low_pct": self.low_pct,
            },
        )

    def _find_signal_key(self, dataset: xr.Dataset, node: str) -> str:
        if node in dataset.data_vars:
            return node
        node_lower = node.lower()
        for key in dataset.data_vars:
            key_str = str(key)
            if key_str.lower() == node_lower:
                return key_str
        raise KeyError(f"Signal '{node}' not found in dataset")


@measurement("overshoot")
class OvershootMeasurement(BaseMeasurement):
    """Measure overshoot as percentage of step size.

    Overshoot = (peak - target) / (target - initial) * 100%

    Example:
        >>> os = OvershootMeasurement(node="vout", target=5.0, reference=0.0)
        >>> result = os.measure(dataset)
        >>> print(f"Overshoot: {result.value:.1f}%")
    """

    name = "overshoot"
    description = "Peak overshoot relative to target"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        target: float,
        reference: float | None = None,
    ):
        """Initialize overshoot measurement.

        Args:
            node: Signal node name
            target: Target/final value
            reference: Initial/reference value (None = first sample)
        """
        self.node = node
        self.target = target
        self.reference = reference

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate overshoot percentage."""
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        sig_key = self._find_signal_key(dataset, self.node)
        voltage = np.asarray(dataset[sig_key].values)

        # Reference value
        baseline = float(voltage[0]) if self.reference is None else self.reference

        # Step size
        step_size = self.target - baseline

        if step_size == 0:
            return MeasurementResult(
                value=0.0,
                unit="%",
                metadata={"error": "Zero step size"},
            )

        # Find peak (max for positive step, min for negative)
        if step_size > 0:
            peak = float(voltage.max())
            peak_idx = int(np.argmax(voltage))
        else:
            peak = float(voltage.min())
            peak_idx = int(np.argmin(voltage))

        # Calculate overshoot
        overshoot = (peak - self.target) / abs(step_size) * 100.0

        return MeasurementResult(
            value=max(0.0, overshoot),  # Overshoot is always positive
            unit="%",
            metadata={
                "peak": peak,
                "peak_time": float(time_values[peak_idx]),
                "target": self.target,
                "reference": baseline,
                "step_size": step_size,
            },
        )

    def _find_signal_key(self, dataset: xr.Dataset, node: str) -> str:
        if node in dataset.data_vars:
            return node
        node_lower = node.lower()
        for key in dataset.data_vars:
            key_str = str(key)
            if key_str.lower() == node_lower:
                return key_str
        raise KeyError(f"Signal '{node}' not found in dataset")


@measurement("propagation_delay")
class PropagationDelayMeasurement(BaseMeasurement):
    """Measure propagation delay between input and output.

    Measures time from input crossing 50% to output crossing 50%.

    Example:
        >>> pd = PropagationDelayMeasurement(
        ...     input_node="vin", output_node="vout", edge="rising"
        ... )
        >>> result = pd.measure(dataset)
        >>> print(f"Propagation delay: {result.value*1e9:.2f} ns")
    """

    name = "propagation_delay"
    description = "Delay from input to output transition"
    required_analyses = ["tran"]

    def __init__(
        self,
        input_node: str,
        output_node: str,
        edge: str = "rising",
        threshold_pct: float = 0.5,
    ):
        """Initialize propagation delay measurement.

        Args:
            input_node: Input signal node
            output_node: Output signal node
            edge: "rising", "falling", or "both" (average)
            threshold_pct: Threshold as fraction of swing (default 0.5 = 50%)
        """
        self.input_node = input_node
        self.output_node = output_node
        self.edge = edge
        self.threshold_pct = threshold_pct

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate propagation delay."""
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        in_key = self._find_signal_key(dataset, self.input_node)
        out_key = self._find_signal_key(dataset, self.output_node)

        v_in = np.asarray(dataset[in_key].values)
        v_out = np.asarray(dataset[out_key].values)

        delays = []

        for sig, name in [(v_in, "input"), (v_out, "output")]:
            v_min, v_max = float(sig.min()), float(sig.max())
            setattr(self, f"_{name}_threshold", v_min + self.threshold_pct * (v_max - v_min))

        in_thresh = self._input_threshold  # type: ignore
        out_thresh = self._output_threshold  # type: ignore

        if self.edge in ("rising", "both"):
            t_in, _ = _find_threshold_crossing(time_values, v_in, in_thresh, rising=True)
            t_out, _ = _find_threshold_crossing(time_values, v_out, out_thresh, rising=True)
            if t_in is not None and t_out is not None:
                delays.append(t_out - t_in)

        if self.edge in ("falling", "both"):
            t_in, _ = _find_threshold_crossing(time_values, v_in, in_thresh, rising=False)
            t_out, _ = _find_threshold_crossing(time_values, v_out, out_thresh, rising=False)
            if t_in is not None and t_out is not None:
                delays.append(t_out - t_in)

        if not delays:
            return MeasurementResult(
                value=float("nan"),
                unit="s",
                metadata={"error": "Could not find crossings"},
            )

        delay = sum(delays) / len(delays)

        return MeasurementResult(
            value=delay,
            unit="s",
            metadata={
                "edge": self.edge,
                "threshold_pct": self.threshold_pct,
                "input_threshold": in_thresh,
                "output_threshold": out_thresh,
            },
        )

    def _find_signal_key(self, dataset: xr.Dataset, node: str) -> str:
        if node in dataset.data_vars:
            return node
        node_lower = node.lower()
        for key in dataset.data_vars:
            key_str = str(key)
            if key_str.lower() == node_lower:
                return key_str
        raise KeyError(f"Signal '{node}' not found in dataset")


__all__ = [
    "SlewRateMeasurement",
    "SettlingTimeMeasurement",
    "RiseTimeMeasurement",
    "FallTimeMeasurement",
    "OvershootMeasurement",
    "PropagationDelayMeasurement",
]
