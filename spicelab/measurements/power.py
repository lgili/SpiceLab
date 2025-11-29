"""Power integrity measurements.

Measurements for power supply and PDN analysis including:
- Supply ripple (peak-to-peak or at specific frequency)
- PDN impedance
- Load transient response
- Voltage droop
- Power efficiency
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import BaseMeasurement, MeasurementResult
from .registry import measurement

if TYPE_CHECKING:
    import xarray as xr


@measurement("ripple")
class RippleMeasurement(BaseMeasurement):
    """Measure supply ripple voltage.

    Can measure peak-to-peak ripple or ripple at a specific frequency.

    Example:
        >>> ripple = RippleMeasurement(node="vdd", ripple_freq=100e3)
        >>> result = ripple.measure(dataset)
        >>> print(f"Ripple at 100kHz: {result.value*1e3:.2f} mV")
    """

    name = "ripple"
    description = "Power supply ripple voltage"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        ripple_freq: float | None = None,
        steady_state_start: float | None = None,
    ):
        """Initialize ripple measurement.

        Args:
            node: Power supply node name
            ripple_freq: Specific frequency to measure (None = peak-to-peak)
            steady_state_start: Time to start measurement (skip startup transient)
        """
        self.node = node
        self.ripple_freq = ripple_freq
        self.steady_state_start = steady_state_start

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate supply ripple."""
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        sig_key = self._find_signal_key(dataset, self.node)
        voltage = np.asarray(dataset[sig_key].values)

        # Apply steady-state filter
        if self.steady_state_start is not None:
            mask = time_values >= self.steady_state_start
            time_values = time_values[mask]
            voltage = voltage[mask]

        if len(voltage) < 2:
            return MeasurementResult(
                value=float("nan"),
                unit="V",
                metadata={"error": "Not enough data points"},
            )

        if self.ripple_freq is not None:
            # FFT-based measurement at specific frequency
            dt = np.mean(np.diff(time_values))
            fs = 1.0 / dt

            # Remove DC component
            voltage_ac = voltage - np.mean(voltage)

            # FFT
            fft_vals = np.fft.rfft(voltage_ac)
            freqs = np.fft.rfftfreq(len(voltage_ac), dt)

            # Find target frequency bin
            freq_idx = int(np.argmin(np.abs(freqs - self.ripple_freq)))
            actual_freq = float(freqs[freq_idx])

            # Peak amplitude (2x for one-sided, /N for normalization)
            ripple = 2.0 * np.abs(fft_vals[freq_idx]) / len(voltage_ac)

            return MeasurementResult(
                value=float(ripple),
                unit="V",
                metadata={
                    "measurement_type": "fft",
                    "target_freq": self.ripple_freq,
                    "actual_freq": actual_freq,
                    "dc_level": float(np.mean(voltage)),
                },
            )
        else:
            # Peak-to-peak measurement
            v_max = float(np.max(voltage))
            v_min = float(np.min(voltage))
            ripple_pp = v_max - v_min

            return MeasurementResult(
                value=ripple_pp,
                unit="V",
                metadata={
                    "measurement_type": "peak_to_peak",
                    "v_max": v_max,
                    "v_min": v_min,
                    "dc_level": float(np.mean(voltage)),
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


@measurement("pdn_impedance")
class PDNImpedanceMeasurement(BaseMeasurement):
    """Power Distribution Network impedance measurement.

    Measures PDN impedance from AC analysis with current injection.
    Z = V / I at each frequency.

    Example:
        >>> pdn = PDNImpedanceMeasurement(
        ...     voltage_node="vdd", current_node="i(iac)", target_z=0.1
        ... )
        >>> result = pdn.measure(dataset)
        >>> print(f"Max impedance: {result.value*1e3:.2f} mΩ")
    """

    name = "pdn_impedance"
    description = "PDN impedance vs frequency"
    required_analyses = ["ac"]

    def __init__(
        self,
        voltage_node: str,
        current_node: str | None = None,
        current_amplitude: float = 1.0,
        target_impedance: float | None = None,
    ):
        """Initialize PDN impedance measurement.

        Args:
            voltage_node: Power net voltage node
            current_node: AC current source node (None = assume 1A injection)
            current_amplitude: AC current amplitude (default 1A)
            target_impedance: Target impedance for pass/fail (Ohms)
        """
        self.voltage_node = voltage_node
        self.current_node = current_node
        self.current_amplitude = current_amplitude
        self.target_impedance = target_impedance

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate PDN impedance."""
        freq = dataset.coords.get("frequency", dataset.coords.get("freq"))
        if freq is None:
            raise ValueError("Dataset must have 'frequency' coordinate")
        freq_values = np.asarray(freq.values)

        v_key = self._find_signal_key(dataset, self.voltage_node)
        voltage = np.asarray(dataset[v_key].values)

        # Get current if available, otherwise assume unit current
        if self.current_node:
            i_key = self._find_signal_key(dataset, self.current_node)
            current = np.asarray(dataset[i_key].values)
        else:
            current = self.current_amplitude * np.ones_like(voltage)

        # Calculate impedance magnitude
        with np.errstate(divide="ignore", invalid="ignore"):
            impedance = np.abs(voltage) / np.abs(current)

        # Replace inf/nan with large value
        impedance = np.nan_to_num(impedance, nan=0.0, posinf=1e12, neginf=0.0)

        # Find maximum impedance
        max_z = float(np.max(impedance))
        max_z_idx = int(np.argmax(impedance))
        max_z_freq = float(freq_values[max_z_idx])

        # Check against target
        passed = None
        if self.target_impedance is not None:
            passed = max_z <= self.target_impedance

        return MeasurementResult(
            value=max_z,
            unit="Ohm",
            passed=passed,
            metadata={
                "frequency_at_max_z": max_z_freq,
                "target_impedance": self.target_impedance,
                "min_z": float(np.min(impedance)),
                "freq_min": float(freq_values[0]),
                "freq_max": float(freq_values[-1]),
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


@measurement("load_transient")
class LoadTransientMeasurement(BaseMeasurement):
    """Measure load transient response characteristics.

    Measures voltage droop/overshoot and recovery time during load step.

    Example:
        >>> lt = LoadTransientMeasurement(
        ...     voltage_node="vout",
        ...     step_time=1e-6,
        ...     nominal_voltage=3.3
        ... )
        >>> result = lt.measure(dataset)
        >>> print(f"Droop: {result.value*1e3:.1f} mV")
    """

    name = "load_transient"
    description = "Load transient droop and recovery"
    required_analyses = ["tran"]

    def __init__(
        self,
        voltage_node: str,
        step_time: float,
        nominal_voltage: float | None = None,
        recovery_tolerance: float = 0.01,
    ):
        """Initialize load transient measurement.

        Args:
            voltage_node: Output voltage node
            step_time: Time of load step
            nominal_voltage: Expected voltage (None = auto from pre-step)
            recovery_tolerance: Recovery band as fraction (default 1%)
        """
        self.voltage_node = voltage_node
        self.step_time = step_time
        self.nominal_voltage = nominal_voltage
        self.recovery_tolerance = recovery_tolerance

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate load transient response."""
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        sig_key = self._find_signal_key(dataset, self.voltage_node)
        voltage = np.asarray(dataset[sig_key].values)

        # Find step time index
        step_idx = int(np.searchsorted(time_values, self.step_time))
        if step_idx >= len(time_values):
            step_idx = len(time_values) - 1

        # Nominal voltage (pre-step average)
        if self.nominal_voltage is None:
            pre_step_samples = max(10, step_idx // 10)
            nominal = float(np.mean(voltage[max(0, step_idx - pre_step_samples):step_idx]))
        else:
            nominal = self.nominal_voltage

        # Find post-step extremes
        post_voltage = voltage[step_idx:]
        if len(post_voltage) == 0:
            return MeasurementResult(
                value=float("nan"),
                unit="V",
                metadata={"error": "No post-step data"},
            )

        v_min_post = float(np.min(post_voltage))
        v_max_post = float(np.max(post_voltage))

        # Determine if droop (undershoot) or overshoot dominates
        droop = nominal - v_min_post
        overshoot = v_max_post - nominal

        if droop > overshoot:
            peak_deviation = droop
            peak_voltage = v_min_post
            deviation_type = "droop"
        else:
            peak_deviation = overshoot
            peak_voltage = v_max_post
            deviation_type = "overshoot"

        # Find recovery time
        recovery_band = nominal * self.recovery_tolerance
        upper = nominal + recovery_band
        lower = nominal - recovery_band

        # Find when voltage stays within band
        recovery_time = float("nan")
        post_time = time_values[step_idx:]

        for i in range(len(post_voltage)):
            if lower <= post_voltage[i] <= upper:
                # Check if it stays within band
                remaining = post_voltage[i:]
                if np.all((remaining >= lower) & (remaining <= upper)):
                    recovery_time = float(post_time[i] - self.step_time)
                    break

        return MeasurementResult(
            value=peak_deviation,
            unit="V",
            metadata={
                "deviation_type": deviation_type,
                "peak_voltage": peak_voltage,
                "nominal_voltage": nominal,
                "droop": droop,
                "overshoot": overshoot,
                "recovery_time": recovery_time,
                "recovery_tolerance": self.recovery_tolerance,
                "step_time": self.step_time,
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


@measurement("efficiency")
class EfficiencyMeasurement(BaseMeasurement):
    """Measure power conversion efficiency.

    Efficiency = P_out / P_in * 100%

    Example:
        >>> eff = EfficiencyMeasurement(
        ...     output_voltage="vout", output_current="i(rload)",
        ...     input_voltage="vin", input_current="i(vin)"
        ... )
        >>> result = eff.measure(dataset)
        >>> print(f"Efficiency: {result.value:.1f}%")
    """

    name = "efficiency"
    description = "Power conversion efficiency"
    required_analyses = ["tran"]

    def __init__(
        self,
        output_voltage: str,
        output_current: str,
        input_voltage: str,
        input_current: str,
        steady_state_start: float | None = None,
    ):
        """Initialize efficiency measurement.

        Args:
            output_voltage: Output voltage node
            output_current: Output current node
            input_voltage: Input voltage node
            input_current: Input current node
            steady_state_start: Time to start measurement (skip startup)
        """
        self.output_voltage = output_voltage
        self.output_current = output_current
        self.input_voltage = input_voltage
        self.input_current = input_current
        self.steady_state_start = steady_state_start

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate efficiency."""
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        # Get signals
        v_out = np.asarray(
            dataset[self._find_signal_key(dataset, self.output_voltage)].values
        )
        i_out = np.asarray(
            dataset[self._find_signal_key(dataset, self.output_current)].values
        )
        v_in = np.asarray(
            dataset[self._find_signal_key(dataset, self.input_voltage)].values
        )
        i_in = np.asarray(
            dataset[self._find_signal_key(dataset, self.input_current)].values
        )

        # Apply steady-state filter
        if self.steady_state_start is not None:
            mask = time_values >= self.steady_state_start
            v_out = v_out[mask]
            i_out = i_out[mask]
            v_in = v_in[mask]
            i_in = i_in[mask]

        if len(v_out) < 2:
            return MeasurementResult(
                value=float("nan"),
                unit="%",
                metadata={"error": "Not enough data points"},
            )

        # Calculate average power
        p_out = float(np.mean(np.abs(v_out * i_out)))
        p_in = float(np.mean(np.abs(v_in * i_in)))

        if p_in == 0:
            efficiency = 0.0
        else:
            efficiency = (p_out / p_in) * 100.0

        return MeasurementResult(
            value=efficiency,
            unit="%",
            metadata={
                "p_out": p_out,
                "p_in": p_in,
                "p_loss": p_in - p_out,
                "v_out_avg": float(np.mean(v_out)),
                "v_in_avg": float(np.mean(v_in)),
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
    "RippleMeasurement",
    "PDNImpedanceMeasurement",
    "LoadTransientMeasurement",
    "EfficiencyMeasurement",
]
