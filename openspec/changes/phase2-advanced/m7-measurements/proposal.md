# M7: Measurement & Analysis Library

**Status:** Proposed
**Priority:** 🟠 HIGH
**Estimated Duration:** 8-10 weeks
**Dependencies:** M1 (testing framework), M2 (performance baseline), M4 (error handling)

## Problem Statement

SpiceLab currently lacks a comprehensive measurement and analysis library for extracting circuit performance specifications from simulation results. Users must manually write custom analysis code for each measurement (gain, bandwidth, phase margin, THD, etc.), which is time-consuming, error-prone, and not reusable across projects.

### Current Gaps
- ❌ No ready-to-use measurement specs (gain, bandwidth, phase margin, etc.)
- ❌ Manual result processing for every common measurement
- ❌ No custom measurement DSL for creating reusable specs
- ❌ Limited statistical analysis helpers
- ❌ Basic FFT support without windowing or advanced features
- ❌ No eye diagram generation capabilities
- ❌ No jitter analysis tools
- ❌ No power integrity measurement helpers

### Impact
- **User Experience:** Frustration from writing boilerplate analysis code
- **Productivity:** Hours wasted reimplementing common measurements
- **Reliability:** Inconsistent measurement implementations across projects
- **Adoption:** Users choose tools with built-in measurement libraries

## Objectives

1. **Provide 30+ ready-to-use measurement specs** (gain, BW, phase margin, THD, PSRR, CMRR, slew rate, settling time, noise, etc.)
2. **Create custom measurement DSL** for defining new measurements declaratively
3. **Implement statistical analysis helpers** (mean, std, percentiles, distributions)
4. **Build advanced FFT/spectrum analysis** with windowing functions
5. **Add eye diagram generation** for high-speed signals
6. **Implement jitter analysis** (period, cycle-to-cycle, TIE)
7. **Create power integrity checks** (PDN impedance, supply ripple, transient response)
8. **Achieve 90% coverage** of common analog/mixed-signal specs

## Technical Design

### 1. Measurement Specification Framework

**Strategy:** Define measurements as composable, reusable specifications.

#### Base Measurement Protocol
```python
# spicelab/measurements/base.py
from typing import Protocol, TypeVar, Generic
import xarray as xr
from pydantic import BaseModel

T = TypeVar('T')

class MeasurementResult(BaseModel, Generic[T]):
    """Result of a measurement with metadata."""
    value: T
    unit: str
    passed: bool | None = None  # For spec checking
    metadata: dict = {}

class Measurement(Protocol):
    """Protocol for all measurements."""

    name: str
    description: str
    required_analyses: list[str]  # ['ac', 'tran', 'noise', etc.]

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Extract measurement from simulation results."""
        ...

    def check_spec(self, result: MeasurementResult, spec: dict) -> bool:
        """Check if measurement meets specification."""
        ...
```

#### Measurement Registry
```python
# spicelab/measurements/registry.py
from typing import Type, Callable
from functools import wraps

class MeasurementRegistry:
    """Central registry of available measurements."""

    _measurements: dict[str, Type[Measurement]] = {}

    @classmethod
    def register(cls, name: str | None = None) -> Callable:
        """Decorator to register a measurement."""
        def decorator(measurement_cls: Type[Measurement]) -> Type[Measurement]:
            reg_name = name or measurement_cls.__name__.lower()
            cls._measurements[reg_name] = measurement_cls
            return measurement_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[Measurement]:
        """Get measurement by name."""
        if name not in cls._measurements:
            raise ValueError(f"Measurement '{name}' not found. Available: {list(cls._measurements.keys())}")
        return cls._measurements[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered measurements."""
        return sorted(cls._measurements.keys())

# Convenience decorator
def measurement(name: str | None = None):
    """Register a measurement class."""
    return MeasurementRegistry.register(name)
```

### 2. Common Analog Measurements

#### AC Analysis Measurements
```python
# spicelab/measurements/ac.py
import numpy as np
import xarray as xr
from scipy import signal

@measurement("gain")
class GainMeasurement:
    """Measure gain (magnitude) at specified frequency."""

    name = "gain"
    description = "DC or AC gain at specified frequency"
    required_analyses = ["ac", "dc"]

    def __init__(
        self,
        output_node: str,
        input_node: str | None = None,
        frequency: float | None = None,  # Hz, None = DC
        unit: str = "dB"  # "dB", "V/V", "abs"
    ):
        self.output_node = output_node
        self.input_node = input_node
        self.frequency = frequency
        self.unit = unit

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Extract gain from AC analysis."""
        if self.frequency is None:
            # DC gain from .op or low-frequency AC
            mag = float(dataset[self.output_node].sel(frequency=dataset.frequency.min(), method='nearest'))
        else:
            # AC gain at specific frequency
            mag = float(dataset[self.output_node].sel(frequency=self.frequency, method='nearest'))

        # Convert to requested unit
        if self.unit == "dB":
            value = 20 * np.log10(mag)
        elif self.unit == "abs":
            value = mag
        else:  # V/V
            value = mag

        return MeasurementResult(
            value=value,
            unit=self.unit,
            metadata={
                "frequency": self.frequency or 0,
                "output_node": self.output_node,
            }
        )

    def check_spec(self, result: MeasurementResult, spec: dict) -> bool:
        """Check gain against spec (min/max)."""
        if "min" in spec and result.value < spec["min"]:
            return False
        if "max" in spec and result.value > spec["max"]:
            return False
        return True

@measurement("bandwidth")
class BandwidthMeasurement:
    """Measure -3dB bandwidth from AC analysis."""

    name = "bandwidth"
    description = "Frequency at -3dB from DC gain"
    required_analyses = ["ac"]

    def __init__(
        self,
        output_node: str,
        reference_gain: float | None = None,  # dB, None = auto from DC
        n_db: float = 3.0  # -3dB default
    ):
        self.output_node = output_node
        self.reference_gain = reference_gain
        self.n_db = n_db

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Find -NdB frequency."""
        freq = dataset.frequency.values
        mag_db = 20 * np.log10(np.abs(dataset[self.output_node].values))

        # Reference gain (DC or specified)
        if self.reference_gain is None:
            ref_gain = mag_db[0]  # Assume first point is DC
        else:
            ref_gain = self.reference_gain

        # Find -3dB point
        target = ref_gain - self.n_db
        idx = np.where(mag_db <= target)[0]

        if len(idx) == 0:
            # Never reached -3dB
            bw = freq[-1]  # Upper limit
            metadata = {"reached_target": False}
        else:
            bw = freq[idx[0]]
            metadata = {"reached_target": True}

        return MeasurementResult(
            value=float(bw),
            unit="Hz",
            metadata={
                "reference_gain_db": ref_gain,
                "target_gain_db": target,
                **metadata
            }
        )

@measurement("phase_margin")
class PhaseMarginMeasurement:
    """Measure phase margin at unity-gain frequency."""

    name = "phase_margin"
    description = "Phase margin at 0dB crossing"
    required_analyses = ["ac"]

    def __init__(self, output_node: str, input_node: str | None = None):
        self.output_node = output_node
        self.input_node = input_node

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Calculate phase margin."""
        freq = dataset.frequency.values

        # Get magnitude and phase
        complex_response = dataset[self.output_node].values
        mag_db = 20 * np.log10(np.abs(complex_response))
        phase_deg = np.angle(complex_response, deg=True)

        # Find unity-gain frequency (0dB crossing)
        zero_crossings = np.where(np.diff(np.sign(mag_db)))[0]

        if len(zero_crossings) == 0:
            raise ValueError("No unity-gain frequency found")

        # First crossing (ascending or descending)
        ugf_idx = zero_crossings[0]
        ugf = freq[ugf_idx]

        # Phase margin = 180° + phase at UGF
        phase_at_ugf = phase_deg[ugf_idx]
        pm = 180 + phase_at_ugf

        return MeasurementResult(
            value=float(pm),
            unit="degrees",
            passed=pm > 0,  # Negative PM = unstable
            metadata={
                "unity_gain_frequency": ugf,
                "phase_at_ugf": phase_at_ugf,
            }
        )

@measurement("gain_margin")
class GainMarginMeasurement:
    """Measure gain margin at 180° phase crossing."""

    name = "gain_margin"
    description = "Gain margin at -180° phase"
    required_analyses = ["ac"]

    def __init__(self, output_node: str):
        self.output_node = output_node

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Calculate gain margin."""
        freq = dataset.frequency.values
        complex_response = dataset[self.output_node].values
        mag_db = 20 * np.log10(np.abs(complex_response))
        phase_deg = np.angle(complex_response, deg=True)

        # Find -180° crossing
        phase_target = -180
        crossings = np.where(np.diff(np.sign(phase_deg - phase_target)))[0]

        if len(crossings) == 0:
            raise ValueError("No -180° phase crossing found")

        crossing_idx = crossings[0]
        crossing_freq = freq[crossing_idx]
        mag_at_crossing = mag_db[crossing_idx]

        # Gain margin = -magnitude at -180° (should be negative)
        gm = -mag_at_crossing

        return MeasurementResult(
            value=float(gm),
            unit="dB",
            passed=gm > 0,  # Negative GM = unstable
            metadata={
                "crossing_frequency": crossing_freq,
                "magnitude_at_crossing": mag_at_crossing,
            }
        )
```

#### Transient Analysis Measurements
```python
# spicelab/measurements/transient.py

@measurement("slew_rate")
class SlewRateMeasurement:
    """Measure slew rate (rise/fall time)."""

    name = "slew_rate"
    description = "Maximum rate of change (dV/dt)"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        edge: str = "rising",  # "rising", "falling", "both"
        v_low: float = 0.1,  # 10%
        v_high: float = 0.9,  # 90%
    ):
        self.node = node
        self.edge = edge
        self.v_low = v_low
        self.v_high = v_high

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Calculate slew rate from transient."""
        time = dataset.time.values
        voltage = dataset[self.node].values

        # Normalize to 0-1
        v_min, v_max = voltage.min(), voltage.max()
        v_norm = (voltage - v_min) / (v_max - v_min)

        # Find 10%-90% crossing times
        idx_low = np.where(v_norm >= self.v_low)[0]
        idx_high = np.where(v_norm >= self.v_high)[0]

        if len(idx_low) == 0 or len(idx_high) == 0:
            raise ValueError("Could not find threshold crossings")

        t_low = time[idx_low[0]]
        t_high = time[idx_high[0]]

        # Slew rate = ΔV / Δt
        dv = (self.v_high - self.v_low) * (v_max - v_min)
        dt = t_high - t_low
        slew = dv / dt

        return MeasurementResult(
            value=float(slew),
            unit="V/s",
            metadata={
                "rise_time": dt,
                "v_swing": dv,
            }
        )

@measurement("settling_time")
class SettlingTimeMeasurement:
    """Measure settling time to within specified error band."""

    name = "settling_time"
    description = "Time to settle within error band"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        final_value: float | None = None,  # None = auto-detect
        error_pct: float = 1.0,  # 1% error band
    ):
        self.node = node
        self.final_value = final_value
        self.error_pct = error_pct

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Calculate settling time."""
        time = dataset.time.values
        voltage = dataset[self.node].values

        # Final value (last 10% of simulation)
        if self.final_value is None:
            final_value = np.mean(voltage[int(0.9 * len(voltage)):])
        else:
            final_value = self.final_value

        # Error band
        error_band = abs(final_value * self.error_pct / 100)
        upper = final_value + error_band
        lower = final_value - error_band

        # Find last time outside error band
        outside = (voltage < lower) | (voltage > upper)
        idx_outside = np.where(outside)[0]

        if len(idx_outside) == 0:
            settling_time = time[0]  # Already settled
        else:
            settling_time = time[idx_outside[-1]]

        return MeasurementResult(
            value=float(settling_time),
            unit="s",
            metadata={
                "final_value": final_value,
                "error_band_pct": self.error_pct,
                "error_band": error_band,
            }
        )
```

### 3. Advanced FFT and Spectrum Analysis

```python
# spicelab/measurements/spectrum.py
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq

@measurement("thd")
class THDMeasurement:
    """Total Harmonic Distortion measurement."""

    name = "thd"
    description = "Total harmonic distortion (THD)"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        fundamental_freq: float,
        n_harmonics: int = 10,
        window: str = "hann",  # "hann", "hamming", "blackman", "flattop"
    ):
        self.node = node
        self.fundamental_freq = fundamental_freq
        self.n_harmonics = n_harmonics
        self.window = window

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Calculate THD from FFT."""
        time = dataset.time.values
        voltage = dataset[self.node].values

        # Apply window
        if self.window != "none":
            window_func = sp_signal.get_window(self.window, len(voltage))
            voltage = voltage * window_func

        # FFT
        fft_vals = fft(voltage)
        freqs = fftfreq(len(voltage), time[1] - time[0])

        # Positive frequencies only
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        fft_vals = np.abs(fft_vals[pos_mask])

        # Find fundamental
        fund_idx = np.argmin(np.abs(freqs - self.fundamental_freq))
        fund_mag = fft_vals[fund_idx]

        # Find harmonics
        harmonic_mags = []
        for n in range(2, self.n_harmonics + 2):
            harm_freq = n * self.fundamental_freq
            harm_idx = np.argmin(np.abs(freqs - harm_freq))
            harmonic_mags.append(fft_vals[harm_idx])

        # THD = sqrt(sum of harmonic powers) / fundamental
        thd = np.sqrt(np.sum(np.array(harmonic_mags)**2)) / fund_mag
        thd_pct = thd * 100
        thd_db = 20 * np.log10(thd)

        return MeasurementResult(
            value=float(thd_pct),
            unit="%",
            metadata={
                "thd_db": thd_db,
                "fundamental_freq": self.fundamental_freq,
                "fundamental_magnitude": fund_mag,
                "harmonic_magnitudes": harmonic_mags.tolist(),
                "window": self.window,
            }
        )

@measurement("snr")
class SNRMeasurement:
    """Signal-to-Noise Ratio measurement."""

    name = "snr"
    description = "Signal-to-noise ratio"
    required_analyses = ["tran", "noise"]

    def __init__(
        self,
        signal_node: str,
        signal_freq: float | None = None,  # Hz, None = RMS
        noise_floor_band: tuple[float, float] | None = None,  # (fmin, fmax)
    ):
        self.signal_node = signal_node
        self.signal_freq = signal_freq
        self.noise_floor_band = noise_floor_band

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Calculate SNR."""
        voltage = dataset[self.signal_node].values

        if self.signal_freq:
            # FFT to get signal power at frequency
            fft_vals = fft(voltage)
            freqs = fftfreq(len(voltage), dataset.time.values[1] - dataset.time.values[0])
            pos_mask = freqs > 0
            freqs = freqs[pos_mask]
            fft_vals = np.abs(fft_vals[pos_mask])

            sig_idx = np.argmin(np.abs(freqs - self.signal_freq))
            signal_power = fft_vals[sig_idx]**2

            # Noise power (excluding signal bin)
            if self.noise_floor_band:
                noise_mask = (freqs >= self.noise_floor_band[0]) & (freqs <= self.noise_floor_band[1])
                noise_mask[sig_idx] = False  # Exclude signal
                noise_power = np.mean(fft_vals[noise_mask]**2)
            else:
                # All bins except signal
                noise_power = np.mean(np.delete(fft_vals, sig_idx)**2)
        else:
            # RMS-based SNR
            signal_power = np.mean(voltage**2)
            # Assume noise from high-frequency content
            noise_power = np.var(voltage - sp_signal.savgol_filter(voltage, 51, 3))

        snr = signal_power / noise_power
        snr_db = 10 * np.log10(snr)

        return MeasurementResult(
            value=float(snr_db),
            unit="dB",
            metadata={
                "signal_power": signal_power,
                "noise_power": noise_power,
            }
        )
```

### 4. PSRR and CMRR Measurements

```python
# spicelab/measurements/rejection.py

@measurement("psrr")
class PSRRMeasurement:
    """Power Supply Rejection Ratio."""

    name = "psrr"
    description = "Power supply rejection ratio"
    required_analyses = ["ac"]

    def __init__(
        self,
        output_node: str,
        supply_node: str,
        frequency: float,
    ):
        self.output_node = output_node
        self.supply_node = supply_node
        self.frequency = frequency

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Calculate PSRR = 20*log10(ΔVsupply/ΔVout)."""
        # Response to supply variation
        v_out = dataset[self.output_node].sel(frequency=self.frequency, method='nearest')
        v_supply = dataset[self.supply_node].sel(frequency=self.frequency, method='nearest')

        # PSRR = ΔVsupply / ΔVout
        psrr_abs = abs(v_supply / v_out)
        psrr_db = 20 * np.log10(psrr_abs)

        return MeasurementResult(
            value=float(psrr_db),
            unit="dB",
            metadata={
                "frequency": self.frequency,
            }
        )

@measurement("cmrr")
class CMRRMeasurement:
    """Common-Mode Rejection Ratio."""

    name = "cmrr"
    description = "Common-mode rejection ratio"
    required_analyses = ["ac"]

    def __init__(
        self,
        output_node: str,
        input_p: str,
        input_n: str,
        frequency: float,
    ):
        self.output_node = output_node
        self.input_p = input_p
        self.input_n = input_n
        self.frequency = frequency

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Calculate CMRR."""
        # Differential gain (from differential AC simulation)
        # Common-mode gain (from common-mode AC simulation)
        # CMRR = Adiff / Acm

        # This requires TWO AC simulations:
        # 1. Differential (in+ = +1V, in- = -1V)
        # 2. Common-mode (in+ = in- = 1V)

        # For simplicity, assume dataset contains both
        # Real implementation needs proper sweep setup

        raise NotImplementedError("CMRR requires differential/common-mode sweep")
```

### 5. Eye Diagram and Jitter Analysis

```python
# spicelab/measurements/digital.py

class EyeDiagram:
    """Generate eye diagram from transient data."""

    def __init__(
        self,
        node: str,
        bit_period: float,
        n_samples_per_ui: int = 100,
    ):
        self.node = node
        self.bit_period = bit_period
        self.n_samples_per_ui = n_samples_per_ui

    def generate(self, dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Generate eye diagram data.

        Returns:
            (time_normalized, voltage_traces)
        """
        time = dataset.time.values
        voltage = dataset[self.node].values

        # Number of UIs (unit intervals) in data
        n_uis = int((time[-1] - time[0]) / self.bit_period)

        # Resample each UI
        traces = []
        time_ui = np.linspace(0, self.bit_period, self.n_samples_per_ui)

        for ui in range(n_uis):
            t_start = ui * self.bit_period
            t_end = (ui + 1) * self.bit_period

            # Extract this UI
            mask = (time >= t_start) & (time < t_end)
            t_ui_actual = time[mask]
            v_ui = voltage[mask]

            # Interpolate to standard time grid
            if len(t_ui_actual) > 1:
                v_ui_resampled = np.interp(time_ui + t_start, t_ui_actual, v_ui)
                traces.append(v_ui_resampled)

        traces = np.array(traces)
        return time_ui, traces

    def plot(self, dataset: xr.Dataset, alpha: float = 0.1):
        """Plot eye diagram."""
        import matplotlib.pyplot as plt

        time_ui, traces = self.generate(dataset)

        plt.figure(figsize=(10, 6))
        for trace in traces:
            plt.plot(time_ui * 1e9, trace, 'b-', alpha=alpha)

        plt.xlabel("Time (ns)")
        plt.ylabel("Voltage (V)")
        plt.title(f"Eye Diagram - {self.node}")
        plt.grid(True, alpha=0.3)
        return plt.gcf()

@measurement("eye_opening")
class EyeOpeningMeasurement:
    """Measure eye opening (height and width)."""

    name = "eye_opening"
    description = "Eye diagram opening (vertical and horizontal)"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        bit_period: float,
        sample_point: float = 0.5,  # UI fraction (0.5 = center)
    ):
        self.node = node
        self.bit_period = bit_period
        self.sample_point = sample_point

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Calculate eye opening."""
        eye = EyeDiagram(self.node, self.bit_period)
        time_ui, traces = eye.generate(dataset)

        # Sample at center of UI
        sample_idx = int(self.sample_point * len(time_ui))
        samples = traces[:, sample_idx]

        # Separate high and low levels (k-means or threshold)
        threshold = np.median(samples)
        high_samples = samples[samples > threshold]
        low_samples = samples[samples <= threshold]

        # Eye height = min(high) - max(low)
        if len(high_samples) > 0 and len(low_samples) > 0:
            eye_height = np.min(high_samples) - np.max(low_samples)
        else:
            eye_height = 0

        # Eye width (time at threshold crossings)
        # Simplified: measure at 50% of eye height
        mid_voltage = (np.min(high_samples) + np.max(low_samples)) / 2

        # Find crossing times
        crossings = []
        for trace in traces:
            cross_idx = np.where(np.diff(np.sign(trace - mid_voltage)))[0]
            if len(cross_idx) >= 2:
                eye_width_ui = (cross_idx[-1] - cross_idx[0]) / len(time_ui)
                crossings.append(eye_width_ui)

        eye_width = np.mean(crossings) * self.bit_period if crossings else 0

        return MeasurementResult(
            value=float(eye_height),
            unit="V",
            metadata={
                "eye_width": eye_width,
                "eye_width_ui": eye_width / self.bit_period if self.bit_period else 0,
                "n_traces": len(traces),
            }
        )

@measurement("jitter")
class JitterMeasurement:
    """Measure jitter (period, cycle-to-cycle)."""

    name = "jitter"
    description = "Period jitter and cycle-to-cycle jitter"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        edge: str = "rising",  # "rising", "falling"
        threshold: float | None = None,  # None = auto (50%)
    ):
        self.node = node
        self.edge = edge
        self.threshold = threshold

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Calculate jitter statistics."""
        time = dataset.time.values
        voltage = dataset[self.node].values

        # Auto threshold
        if self.threshold is None:
            threshold = (voltage.max() + voltage.min()) / 2
        else:
            threshold = self.threshold

        # Find edge crossings
        if self.edge == "rising":
            crossings = np.where((voltage[:-1] < threshold) & (voltage[1:] >= threshold))[0]
        else:
            crossings = np.where((voltage[:-1] > threshold) & (voltage[1:] <= threshold))[0]

        crossing_times = time[crossings]

        # Period = time between consecutive edges
        periods = np.diff(crossing_times)

        # Period jitter (RMS deviation from mean)
        mean_period = np.mean(periods)
        period_jitter = np.std(periods)

        # Cycle-to-cycle jitter (consecutive period differences)
        cycle_jitter = np.std(np.diff(periods)) if len(periods) > 1 else 0

        return MeasurementResult(
            value=float(period_jitter),
            unit="s",
            metadata={
                "mean_period": mean_period,
                "period_jitter_rms": period_jitter,
                "period_jitter_pct": (period_jitter / mean_period * 100) if mean_period else 0,
                "cycle_to_cycle_jitter": cycle_jitter,
                "n_edges": len(crossing_times),
            }
        )
```

### 6. Power Integrity Measurements

```python
# spicelab/measurements/power.py

@measurement("pdn_impedance")
class PDNImpedanceMeasurement:
    """Power Distribution Network impedance."""

    name = "pdn_impedance"
    description = "PDN impedance vs frequency"
    required_analyses = ["ac"]

    def __init__(
        self,
        power_node: str,
        gnd_node: str = "0",
        target_impedance: float | None = None,  # Ohms
    ):
        self.power_node = power_node
        self.gnd_node = gnd_node
        self.target_impedance = target_impedance

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Calculate PDN impedance."""
        # Impedance = V / I (from AC analysis with current source)
        # Requires simulation with AC current source on power net

        freq = dataset.frequency.values
        voltage = dataset[self.power_node].values

        # Assume 1A AC current injection (need to verify simulation setup)
        current = 1.0
        impedance = np.abs(voltage) / current

        # Find maximum impedance
        max_z = np.max(impedance)
        max_z_freq = freq[np.argmax(impedance)]

        # Check against target
        passed = None
        if self.target_impedance:
            passed = max_z < self.target_impedance

        return MeasurementResult(
            value=float(max_z),
            unit="Ohm",
            passed=passed,
            metadata={
                "frequency_at_max_z": max_z_freq,
                "impedance_spectrum": impedance.tolist()[:100],  # First 100 points
            }
        )

@measurement("ripple")
class SupplyRippleMeasurement:
    """Measure supply ripple voltage."""

    name = "ripple"
    description = "Power supply ripple (AC component)"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        ripple_freq: float | None = None,  # Hz, None = peak-to-peak
    ):
        self.node = node
        self.ripple_freq = ripple_freq

    def measure(self, dataset: xr.Dataset) -> MeasurementResult:
        """Calculate ripple."""
        voltage = dataset[self.node].values

        if self.ripple_freq:
            # FFT at specific frequency
            fft_vals = fft(voltage)
            freqs = fftfreq(len(voltage), dataset.time.values[1] - dataset.time.values[0])

            freq_idx = np.argmin(np.abs(freqs - self.ripple_freq))
            ripple = 2 * np.abs(fft_vals[freq_idx]) / len(voltage)  # Peak amplitude
        else:
            # Peak-to-peak
            ripple = voltage.max() - voltage.min()

        return MeasurementResult(
            value=float(ripple),
            unit="V",
            metadata={
                "measurement_type": "fft" if self.ripple_freq else "peak_to_peak",
            }
        )
```

### 7. Measurement DSL

```python
# spicelab/measurements/dsl.py

class MeasurementDSL:
    """DSL for defining custom measurements."""

    @staticmethod
    def define(
        name: str,
        description: str,
        required_analyses: list[str],
        measure_fn: Callable[[xr.Dataset], float],
        check_fn: Callable[[float, dict], bool] | None = None,
    ) -> Type[Measurement]:
        """Define a custom measurement.

        Example:
            >>> def measure_vout_max(ds):
            ...     return float(ds['vout'].max())
            >>>
            >>> VoutMax = MeasurementDSL.define(
            ...     name="vout_max",
            ...     description="Maximum output voltage",
            ...     required_analyses=["tran"],
            ...     measure_fn=measure_vout_max,
            ... )
            >>>
            >>> result = VoutMax().measure(dataset)
        """

        @measurement(name)
        class CustomMeasurement:
            name = name
            description = description
            required_analyses = required_analyses

            def measure(self, dataset: xr.Dataset) -> MeasurementResult:
                value = measure_fn(dataset)
                return MeasurementResult(value=value, unit="")

            def check_spec(self, result: MeasurementResult, spec: dict) -> bool:
                if check_fn:
                    return check_fn(result.value, spec)
                # Default: min/max checking
                if "min" in spec and result.value < spec["min"]:
                    return False
                if "max" in spec and result.value > spec["max"]:
                    return False
                return True

        return CustomMeasurement
```

## Implementation Plan

### Week 1-2: Framework and Infrastructure
- [ ] Create `spicelab/measurements/` package structure
- [ ] Implement base `Measurement` protocol
- [ ] Create `MeasurementRegistry` with decorator
- [ ] Build `MeasurementResult` data model
- [ ] Implement DSL for custom measurements
- [ ] Write framework unit tests

### Week 3-4: AC Analysis Measurements
- [ ] Implement gain measurement (DC, AC)
- [ ] Create bandwidth measurement (-3dB)
- [ ] Add phase margin measurement
- [ ] Implement gain margin measurement
- [ ] Create input/output impedance measurements
- [ ] Add PSRR measurement
- [ ] Implement CMRR measurement (requires sweep)
- [ ] Write tests for all AC measurements

### Week 5-6: Transient Analysis Measurements
- [ ] Implement slew rate measurement
- [ ] Create settling time measurement
- [ ] Add overshoot/undershoot measurement
- [ ] Implement rise/fall time measurement
- [ ] Create propagation delay measurement
- [ ] Write tests for transient measurements

### Week 7: Spectrum and Distortion Analysis
- [ ] Implement FFT helper with windowing
- [ ] Create THD measurement
- [ ] Add THD+N measurement
- [ ] Implement SNR/SINAD measurements
- [ ] Create SFDR (spurious-free dynamic range)
- [ ] Add spectrum plotting utilities
- [ ] Write tests for spectrum analysis

### Week 8: Digital Signal Measurements
- [ ] Implement eye diagram generation
- [ ] Create eye opening measurement
- [ ] Add jitter measurement (period, cycle-to-cycle, TIE)
- [ ] Implement bit error rate estimation
- [ ] Create timing measurements (setup/hold)
- [ ] Write tests for digital measurements

### Week 9: Power Integrity
- [ ] Implement PDN impedance measurement
- [ ] Create supply ripple measurement
- [ ] Add load transient response
- [ ] Implement voltage droop measurement
- [ ] Create efficiency calculation
- [ ] Write tests for power measurements

### Week 10: Documentation and Integration
- [ ] Document all 30+ measurements with examples
- [ ] Create measurement cookbook (common circuits)
- [ ] Add measurement chaining/composition examples
- [ ] Write integration tests (full circuits)
- [ ] Create performance benchmarks
- [ ] Update main documentation

## Success Metrics

### Completeness
- [ ] **30+ measurements** implemented and tested
- [ ] Coverage: **90% of common analog/mixed-signal specs**
- [ ] All measurements documented with examples
- [ ] Custom measurement DSL functional

### Quality
- [ ] **95%+ test coverage** for measurements module
- [ ] All measurements validated against known circuits
- [ ] Measurement results match manual calculations (±1%)
- [ ] Error handling for invalid inputs

### Performance
- [ ] Measurement extraction: <100ms per measurement
- [ ] FFT with 100k points: <500ms
- [ ] Eye diagram generation: <1s for 1M points
- [ ] Batch measurements: <1s for 10 measurements

### Usability
- [ ] One-line measurement syntax
- [ ] Automatic unit conversion
- [ ] Helpful error messages
- [ ] Spec checking built-in

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Measurement accuracy | HIGH | Validate against manual calculations and commercial tools |
| Edge cases (noise, sparse data) | MEDIUM | Robust interpolation, require minimum data points |
| Performance (large datasets) | MEDIUM | Lazy evaluation, chunked processing |
| FFT windowing artifacts | MEDIUM | Document window selection guide, provide presets |
| CMRR/PSRR require special setup | MEDIUM | Provide simulation templates, clear documentation |

## Dependencies

**Required:**
- M1 (testing framework)
- M2 (performance baseline)
- M4 (error handling)
- scipy (FFT, signal processing)
- numpy
- xarray

**Optional:**
- matplotlib (plotting)
- scikit-learn (clustering for eye diagrams)

## Future Enhancements

- **M9:** Integration with optimization (measurement as objective)
- **M11:** Interactive measurement visualization (web UI)
- **M13:** Mixed-signal measurements (protocol decoding)
- **M20:** ML-based measurement extraction

## References

- [Analog Circuit Design Specs](https://www.analog.com/en/education/education-library/tutorials.html)
- [Eye Diagram Analysis](https://www.tek.com/en/documents/primer/primer-understanding-eye-diagrams)
- [Jitter Analysis Fundamentals](https://www.ti.com/lit/an/slaa609/slaa609.pdf)
- [FFT Windowing](https://www.sjsu.edu/people/burford.furman/docs/me120/FFT_tutorial_NI.pdf)
- [Power Integrity](https://www.powerelectronictips.com/understanding-power-distribution-network-pdn-impedance/)
