"""Digital signal measurements.

Measurements for high-speed digital signals including:
- Eye diagram generation
- Eye opening (height and width)
- Jitter (period, cycle-to-cycle, TIE)
- Timing measurements (setup/hold)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .base import BaseMeasurement, MeasurementResult
from .registry import measurement

if TYPE_CHECKING:
    import xarray as xr


@dataclass
class EyeDiagramData:
    """Eye diagram data structure.

    Attributes:
        time_ui: Normalized time axis (0 to 1 UI or 0 to 2 UI)
        traces: Array of overlaid voltage traces (n_traces x n_samples)
        bit_period: Unit interval duration in seconds
        n_traces: Number of overlaid traces
    """

    time_ui: np.ndarray
    traces: np.ndarray
    bit_period: float
    n_traces: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_traces = len(self.traces)

    def get_histogram(
        self, sample_point: float = 0.5, n_bins: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get histogram of values at sample point.

        Args:
            sample_point: Position in UI (0-1) to sample
            n_bins: Number of histogram bins

        Returns:
            (bin_edges, counts)
        """
        sample_idx = int(sample_point * self.time_ui.size)
        if sample_idx >= self.traces.shape[1]:
            sample_idx = self.traces.shape[1] - 1

        samples = self.traces[:, sample_idx]
        counts, bin_edges = np.histogram(samples, bins=n_bins)
        return bin_edges, counts


class EyeDiagram:
    """Generate eye diagram from transient data.

    Example:
        >>> eye = EyeDiagram(node="vout", bit_period=1e-9)
        >>> data = eye.generate(dataset)
        >>> print(f"Generated {data.n_traces} traces")
        >>> eye.plot(dataset)  # matplotlib plot
    """

    def __init__(
        self,
        node: str,
        bit_period: float,
        n_samples_per_ui: int = 100,
        n_ui: int = 2,
    ):
        """Initialize eye diagram generator.

        Args:
            node: Signal node name
            bit_period: Bit period (unit interval) in seconds
            n_samples_per_ui: Samples per unit interval for resampling
            n_ui: Number of unit intervals to show (1 or 2)
        """
        self.node = node
        self.bit_period = bit_period
        self.n_samples_per_ui = n_samples_per_ui
        self.n_ui = n_ui

    def generate(self, dataset: xr.Dataset) -> EyeDiagramData:
        """Generate eye diagram data from dataset.

        Args:
            dataset: xarray Dataset with transient simulation results

        Returns:
            EyeDiagramData with normalized traces
        """
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        sig_key = self._find_signal_key(dataset, self.node)
        voltage = np.asarray(dataset[sig_key].values)

        # Calculate number of unit intervals in data
        total_time = time_values[-1] - time_values[0]
        n_uis = int(total_time / self.bit_period)

        if n_uis < 2:
            raise ValueError(
                f"Not enough data for eye diagram: {n_uis} UIs "
                f"(need at least 2, total time = {total_time}s, "
                f"bit period = {self.bit_period}s)"
            )

        # Time axis for one eye (in UI units, normalized 0 to n_ui)
        time_ui = np.linspace(0, self.n_ui, self.n_samples_per_ui * self.n_ui)

        # Collect traces
        traces = []
        for ui_start in range(0, n_uis - self.n_ui + 1):
            t_start = time_values[0] + ui_start * self.bit_period
            t_end = t_start + self.n_ui * self.bit_period

            # Find samples in this window
            mask = (time_values >= t_start) & (time_values < t_end)
            t_window = time_values[mask]
            v_window = voltage[mask]

            if len(t_window) < 2:
                continue

            # Normalize time to UI units
            t_norm = (t_window - t_start) / self.bit_period

            # Interpolate to uniform grid
            v_resampled = np.interp(time_ui, t_norm, v_window)
            traces.append(v_resampled)

        if not traces:
            raise ValueError("Could not generate any eye traces")

        return EyeDiagramData(
            time_ui=time_ui,
            traces=np.array(traces),
            bit_period=self.bit_period,
        )

    def plot(
        self,
        dataset: xr.Dataset,
        alpha: float = 0.1,
        color: str = "blue",
        show: bool = True,
    ) -> None:
        """Plot eye diagram using matplotlib.

        Args:
            dataset: xarray Dataset with transient data
            alpha: Transparency for traces (0-1)
            color: Line color
            show: Whether to call plt.show()
        """
        import matplotlib.pyplot as plt

        data = self.generate(dataset)

        fig, ax = plt.subplots(figsize=(10, 6))
        for trace in data.traces:
            ax.plot(data.time_ui, trace, color=color, alpha=alpha, linewidth=0.5)

        ax.set_xlabel("Time (UI)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(f"Eye Diagram - {self.node}")
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

        return fig  # type: ignore

    def _find_signal_key(self, dataset: xr.Dataset, node: str) -> str:
        if node in dataset.data_vars:
            return node
        node_lower = node.lower()
        for key in dataset.data_vars:
            if key.lower() == node_lower:
                return key
        raise KeyError(f"Signal '{node}' not found in dataset")


@measurement("eye_opening")
class EyeOpeningMeasurement(BaseMeasurement):
    """Measure eye opening (height and width).

    Eye height: Minimum vertical opening at sample point
    Eye width: Horizontal opening at mid-level

    Example:
        >>> eye = EyeOpeningMeasurement(node="vout", bit_period=1e-9)
        >>> result = eye.measure(dataset)
        >>> print(f"Eye height: {result.value:.3f} V")
        >>> print(f"Eye width: {result.metadata['eye_width_ui']:.3f} UI")
    """

    name = "eye_opening"
    description = "Eye diagram opening (height and width)"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        bit_period: float,
        sample_point: float = 0.5,
    ):
        """Initialize eye opening measurement.

        Args:
            node: Signal node name
            bit_period: Bit period (unit interval) in seconds
            sample_point: Position in UI (0-1) to measure (default 0.5 = center)
        """
        self.node = node
        self.bit_period = bit_period
        self.sample_point = sample_point

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate eye opening."""
        eye_gen = EyeDiagram(self.node, self.bit_period, n_ui=2)
        try:
            data = eye_gen.generate(dataset)
        except ValueError as e:
            return MeasurementResult(
                value=float("nan"),
                unit="V",
                metadata={"error": str(e)},
            )

        # Sample at center of eye (for 2 UI, center is at 1.0)
        sample_idx = int(self.sample_point * data.time_ui.size / 2 + data.time_ui.size / 4)
        if sample_idx >= data.traces.shape[1]:
            sample_idx = data.traces.shape[1] - 1

        samples = data.traces[:, sample_idx]

        # Separate high and low levels using threshold
        threshold = np.median(samples)
        high_samples = samples[samples > threshold]
        low_samples = samples[samples <= threshold]

        if len(high_samples) == 0 or len(low_samples) == 0:
            return MeasurementResult(
                value=0.0,
                unit="V",
                metadata={"error": "Could not separate high/low levels"},
            )

        # Eye height = min(high) - max(low)
        eye_height = float(np.min(high_samples) - np.max(low_samples))

        # Eye width measurement (horizontal opening)
        # Find 50% crossing points for each trace
        mid_voltage = (np.min(high_samples) + np.max(low_samples)) / 2

        crossing_times = []
        for trace in data.traces:
            # Find all crossings
            crossings = []
            for i in range(len(trace) - 1):
                if (trace[i] - mid_voltage) * (trace[i + 1] - mid_voltage) < 0:
                    # Interpolate crossing time
                    frac = (mid_voltage - trace[i]) / (trace[i + 1] - trace[i])
                    t_cross = data.time_ui[i] + frac * (
                        data.time_ui[i + 1] - data.time_ui[i]
                    )
                    crossings.append(t_cross)

            if len(crossings) >= 2:
                # Eye width = difference between adjacent crossings
                crossing_times.extend(np.diff(crossings).tolist())

        if crossing_times:
            eye_width_ui = float(np.mean(crossing_times))
            eye_width = eye_width_ui * self.bit_period
        else:
            eye_width_ui = 0.0
            eye_width = 0.0

        return MeasurementResult(
            value=eye_height,
            unit="V",
            metadata={
                "eye_width": eye_width,
                "eye_width_ui": eye_width_ui,
                "high_level_min": float(np.min(high_samples)),
                "low_level_max": float(np.max(low_samples)),
                "mid_voltage": mid_voltage,
                "n_traces": data.n_traces,
                "sample_point": self.sample_point,
            },
        )


@measurement("jitter")
class JitterMeasurement(BaseMeasurement):
    """Measure jitter statistics.

    Calculates:
    - Period jitter (RMS deviation from mean period)
    - Cycle-to-cycle jitter (RMS of consecutive period differences)
    - Peak-to-peak jitter

    Example:
        >>> jitter = JitterMeasurement(node="clk", edge="rising")
        >>> result = jitter.measure(dataset)
        >>> print(f"Period jitter: {result.value*1e12:.2f} ps RMS")
    """

    name = "jitter"
    description = "Period and cycle-to-cycle jitter"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        edge: str = "rising",
        threshold: float | None = None,
    ):
        """Initialize jitter measurement.

        Args:
            node: Signal node name
            edge: "rising" or "falling"
            threshold: Crossing threshold (None = auto 50%)
        """
        self.node = node
        self.edge = edge
        self.threshold = threshold

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate jitter statistics."""
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        sig_key = self._find_signal_key(dataset, self.node)
        voltage = np.asarray(dataset[sig_key].values)

        # Auto threshold at 50%
        if self.threshold is None:
            threshold = (voltage.max() + voltage.min()) / 2
        else:
            threshold = self.threshold

        # Find edge crossings
        crossing_times = []
        for i in range(len(voltage) - 1):
            v0, v1 = voltage[i], voltage[i + 1]
            t0, t1 = time_values[i], time_values[i + 1]

            if self.edge == "rising":
                crossed = v0 < threshold <= v1
            else:
                crossed = v0 > threshold >= v1

            if crossed:
                # Interpolate crossing time
                if v1 != v0:
                    frac = (threshold - v0) / (v1 - v0)
                    t_cross = t0 + frac * (t1 - t0)
                else:
                    t_cross = t0
                crossing_times.append(t_cross)

        crossing_times = np.array(crossing_times)

        if len(crossing_times) < 3:
            return MeasurementResult(
                value=float("nan"),
                unit="s",
                metadata={"error": "Not enough edges for jitter analysis"},
            )

        # Calculate periods
        periods = np.diff(crossing_times)

        if len(periods) < 2:
            return MeasurementResult(
                value=float("nan"),
                unit="s",
                metadata={"error": "Not enough periods"},
            )

        # Statistics
        mean_period = float(np.mean(periods))
        period_jitter_rms = float(np.std(periods))
        period_jitter_pp = float(np.max(periods) - np.min(periods))

        # Cycle-to-cycle jitter
        period_diffs = np.diff(periods)
        if len(period_diffs) > 0:
            cycle_jitter_rms = float(np.std(period_diffs))
        else:
            cycle_jitter_rms = 0.0

        return MeasurementResult(
            value=period_jitter_rms,
            unit="s",
            metadata={
                "mean_period": mean_period,
                "frequency": 1.0 / mean_period if mean_period > 0 else 0,
                "period_jitter_rms": period_jitter_rms,
                "period_jitter_pp": period_jitter_pp,
                "period_jitter_pct": (period_jitter_rms / mean_period * 100)
                if mean_period > 0
                else 0,
                "cycle_to_cycle_jitter": cycle_jitter_rms,
                "n_edges": len(crossing_times),
                "n_periods": len(periods),
                "edge": self.edge,
                "threshold": threshold,
            },
        )

    def _find_signal_key(self, dataset: xr.Dataset, node: str) -> str:
        if node in dataset.data_vars:
            return node
        node_lower = node.lower()
        for key in dataset.data_vars:
            if key.lower() == node_lower:
                return key
        raise KeyError(f"Signal '{node}' not found in dataset")


@measurement("duty_cycle")
class DutyCycleMeasurement(BaseMeasurement):
    """Measure duty cycle of a periodic signal.

    Duty cycle = high time / period * 100%

    Example:
        >>> dc = DutyCycleMeasurement(node="pwm", threshold=2.5)
        >>> result = dc.measure(dataset)
        >>> print(f"Duty cycle: {result.value:.1f}%")
    """

    name = "duty_cycle"
    description = "Duty cycle of periodic signal"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        threshold: float | None = None,
    ):
        """Initialize duty cycle measurement.

        Args:
            node: Signal node name
            threshold: Threshold voltage (None = auto 50%)
        """
        self.node = node
        self.threshold = threshold

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate duty cycle."""
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        sig_key = self._find_signal_key(dataset, self.node)
        voltage = np.asarray(dataset[sig_key].values)

        # Auto threshold
        if self.threshold is None:
            threshold = (voltage.max() + voltage.min()) / 2
        else:
            threshold = self.threshold

        # Find all crossings
        rising_times = []
        falling_times = []

        for i in range(len(voltage) - 1):
            v0, v1 = voltage[i], voltage[i + 1]
            t0, t1 = time_values[i], time_values[i + 1]

            if v0 < threshold <= v1:  # Rising edge
                frac = (threshold - v0) / (v1 - v0) if v1 != v0 else 0
                rising_times.append(t0 + frac * (t1 - t0))
            elif v0 > threshold >= v1:  # Falling edge
                frac = (threshold - v0) / (v1 - v0) if v1 != v0 else 0
                falling_times.append(t0 + frac * (t1 - t0))

        if len(rising_times) < 2 or len(falling_times) < 1:
            return MeasurementResult(
                value=float("nan"),
                unit="%",
                metadata={"error": "Not enough edges for duty cycle"},
            )

        # Calculate duty cycles for each complete period
        duty_cycles = []
        rising_times = np.array(rising_times)
        falling_times = np.array(falling_times)

        for i in range(len(rising_times) - 1):
            t_rise = rising_times[i]
            t_rise_next = rising_times[i + 1]

            # Find falling edge between these two rising edges
            fall_idx = np.searchsorted(falling_times, t_rise)
            if fall_idx < len(falling_times) and falling_times[fall_idx] < t_rise_next:
                t_fall = falling_times[fall_idx]
                period = t_rise_next - t_rise
                high_time = t_fall - t_rise
                duty_cycles.append(high_time / period * 100)

        if not duty_cycles:
            return MeasurementResult(
                value=float("nan"),
                unit="%",
                metadata={"error": "Could not calculate duty cycle"},
            )

        mean_duty = float(np.mean(duty_cycles))

        return MeasurementResult(
            value=mean_duty,
            unit="%",
            metadata={
                "duty_cycle_std": float(np.std(duty_cycles)),
                "n_periods": len(duty_cycles),
                "threshold": threshold,
            },
        )

    def _find_signal_key(self, dataset: xr.Dataset, node: str) -> str:
        if node in dataset.data_vars:
            return node
        node_lower = node.lower()
        for key in dataset.data_vars:
            if key.lower() == node_lower:
                return key
        raise KeyError(f"Signal '{node}' not found in dataset")


__all__ = [
    "EyeDiagram",
    "EyeDiagramData",
    "EyeOpeningMeasurement",
    "JitterMeasurement",
    "DutyCycleMeasurement",
]
