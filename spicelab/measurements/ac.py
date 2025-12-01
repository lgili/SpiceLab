"""AC analysis measurements.

Measurements for frequency-domain analysis including:
- Gain (DC and AC)
- Bandwidth (-3dB or custom)
- Phase margin
- Gain margin
- PSRR (Power Supply Rejection Ratio)
- CMRR (Common-Mode Rejection Ratio)
- Input/Output impedance
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import BaseMeasurement, MeasurementResult
from .registry import measurement

if TYPE_CHECKING:
    import xarray as xr


def _safe_log10(x: np.ndarray) -> np.ndarray:
    """Compute log10 safely, avoiding log(0)."""
    eps = np.finfo(float).tiny
    return np.log10(np.clip(x, eps, None))


def _interp_crossing(
    freq: np.ndarray, values: np.ndarray, target: float
) -> tuple[float | None, int | None]:
    """Find frequency where values cross target using linear interpolation.

    Returns (frequency, index) of crossing, or (None, None) if not found.
    """
    if len(freq) < 2:
        return None, None

    # Find sign changes
    diff = values - target
    signs = np.sign(diff)
    crossings = np.where(np.diff(signs) != 0)[0]

    if len(crossings) == 0:
        return None, None

    # Use first crossing
    idx = crossings[0]
    f0, f1 = freq[idx], freq[idx + 1]
    v0, v1 = values[idx], values[idx + 1]

    # Linear interpolation
    if v1 == v0:
        f_cross = f0
    else:
        frac = (target - v0) / (v1 - v0)
        f_cross = f0 + frac * (f1 - f0)

    return float(f_cross), int(idx)


@measurement("gain")
class GainMeasurement(BaseMeasurement):
    """Measure gain (magnitude) at a specified frequency.

    Supports both DC gain (frequency=None or 0) and AC gain at any frequency.
    Can output in dB, V/V, or absolute magnitude.

    Example:
        >>> gain = GainMeasurement(output_node="vout", frequency=1000, unit="dB")
        >>> result = gain.measure(dataset)
        >>> print(f"Gain at 1kHz: {result.value} dB")
    """

    name = "gain"
    description = "DC or AC gain at specified frequency"
    required_analyses = ["ac"]

    def __init__(
        self,
        output_node: str,
        input_node: str | None = None,
        frequency: float | None = None,
        unit: str = "dB",
    ):
        """Initialize gain measurement.

        Args:
            output_node: Output signal node name (e.g., "V(out)")
            input_node: Optional input signal node for ratio (e.g., "V(in)")
            frequency: Frequency in Hz. None or 0 means DC/lowest frequency
            unit: Output unit - "dB", "V/V", or "abs"
        """
        self.output_node = output_node
        self.input_node = input_node
        self.frequency = frequency
        self.unit = unit

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Extract gain from AC analysis results."""
        # Get frequency axis
        freq = dataset.coords.get("frequency", dataset.coords.get("freq"))
        if freq is None:
            raise ValueError("Dataset must have 'frequency' or 'freq' coordinate")
        freq_values = np.asarray(freq.values)

        # Get output signal
        out_key = self._find_signal_key(dataset, self.output_node)
        out_signal = np.asarray(dataset[out_key].values)

        # Determine target frequency
        if self.frequency is None or self.frequency == 0:
            # DC gain - use lowest frequency
            freq_idx = 0
            target_freq = float(freq_values[0])
        else:
            # Find nearest frequency
            freq_idx = int(np.argmin(np.abs(freq_values - self.frequency)))
            target_freq = float(freq_values[freq_idx])

        # Get output magnitude
        out_mag = float(np.abs(out_signal[freq_idx]))

        # Get input magnitude if specified
        if self.input_node:
            in_key = self._find_signal_key(dataset, self.input_node)
            in_signal = np.asarray(dataset[in_key].values)
            in_mag = float(np.abs(in_signal[freq_idx]))
            if in_mag == 0:
                ratio = float("inf")
            else:
                ratio = out_mag / in_mag
        else:
            ratio = out_mag

        # Convert to requested unit
        if self.unit == "dB":
            if ratio <= 0:
                value = float("-inf")
            else:
                value = 20.0 * np.log10(ratio)
        else:
            value = ratio

        return MeasurementResult(
            value=value,
            unit=self.unit,
            metadata={
                "frequency": target_freq,
                "output_node": self.output_node,
                "input_node": self.input_node,
                "output_magnitude": out_mag,
            },
        )

    def _find_signal_key(self, dataset: xr.Dataset, node: str) -> str:
        """Find the actual key in dataset for a node name."""
        # Try exact match first
        if node in dataset.data_vars:
            return node

        # Try case-insensitive match
        node_lower = node.lower()
        for key in dataset.data_vars:
            if key.lower() == node_lower:
                return key

        # Try with common prefixes
        for prefix in ["V(", "v(", "I(", "i("]:
            for suffix in [")", ""]:
                test_key = f"{prefix}{node}{suffix}"
                if test_key in dataset.data_vars:
                    return test_key
                for key in dataset.data_vars:
                    if key.lower() == test_key.lower():
                        return key

        raise KeyError(
            f"Signal '{node}' not found in dataset. " f"Available: {list(dataset.data_vars)}"
        )


@measurement("bandwidth")
class BandwidthMeasurement(BaseMeasurement):
    """Measure -3dB (or custom) bandwidth from AC analysis.

    Finds the frequency at which gain drops by n_db from reference.

    Example:
        >>> bw = BandwidthMeasurement(output_node="vout", n_db=3.0)
        >>> result = bw.measure(dataset)
        >>> print(f"Bandwidth: {result.value/1e6:.2f} MHz")
    """

    name = "bandwidth"
    description = "Frequency at -NdB from DC gain"
    required_analyses = ["ac"]

    def __init__(
        self,
        output_node: str,
        reference_gain_db: float | None = None,
        n_db: float = 3.0,
    ):
        """Initialize bandwidth measurement.

        Args:
            output_node: Output signal node name
            reference_gain_db: Reference gain in dB. None = use DC gain
            n_db: Number of dB below reference (default 3.0 for -3dB)
        """
        self.output_node = output_node
        self.reference_gain_db = reference_gain_db
        self.n_db = n_db

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Find -NdB bandwidth frequency."""
        # Get frequency axis
        freq = dataset.coords.get("frequency", dataset.coords.get("freq"))
        if freq is None:
            raise ValueError("Dataset must have 'frequency' coordinate")
        freq_values = np.asarray(freq.values)

        # Get signal and convert to dB
        out_key = self._find_signal_key(dataset, self.output_node)
        out_signal = np.asarray(dataset[out_key].values)
        mag_db = 20.0 * _safe_log10(np.abs(out_signal))

        # Reference gain (DC or specified)
        if self.reference_gain_db is None:
            ref_gain_db = float(mag_db[0])
        else:
            ref_gain_db = self.reference_gain_db

        # Target gain
        target_db = ref_gain_db - self.n_db

        # Find crossing
        f_bw, _idx = _interp_crossing(freq_values, mag_db, target_db)

        if f_bw is None:
            # Never reached target - use upper limit
            f_bw = float(freq_values[-1])
            reached = False
        else:
            reached = True

        return MeasurementResult(
            value=f_bw,
            unit="Hz",
            metadata={
                "reference_gain_db": ref_gain_db,
                "target_gain_db": target_db,
                "n_db": self.n_db,
                "reached_target": reached,
            },
        )

    def _find_signal_key(self, dataset: xr.Dataset, node: str) -> str:
        """Find the actual key in dataset for a node name."""
        if node in dataset.data_vars:
            return node
        node_lower = node.lower()
        for key in dataset.data_vars:
            if key.lower() == node_lower:
                return key
        raise KeyError(f"Signal '{node}' not found in dataset")


@measurement("phase_margin")
class PhaseMarginMeasurement(BaseMeasurement):
    """Measure phase margin at unity-gain frequency.

    Phase margin = 180° + phase at |H| = 1 (0 dB).
    Positive phase margin indicates stability.

    Example:
        >>> pm = PhaseMarginMeasurement(output_node="vout", input_node="vin")
        >>> result = pm.measure(dataset)
        >>> print(f"Phase margin: {result.value:.1f}°")
        >>> if result.passed is False:
        ...     print("Warning: System may be unstable!")
    """

    name = "phase_margin"
    description = "Phase margin at unity-gain crossing"
    required_analyses = ["ac"]

    def __init__(self, output_node: str, input_node: str | None = None):
        """Initialize phase margin measurement.

        Args:
            output_node: Output signal node name
            input_node: Input signal node name (for transfer function)
        """
        self.output_node = output_node
        self.input_node = input_node

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate phase margin at unity-gain frequency."""
        # Get frequency axis
        freq = dataset.coords.get("frequency", dataset.coords.get("freq"))
        if freq is None:
            raise ValueError("Dataset must have 'frequency' coordinate")
        freq_values = np.asarray(freq.values)

        # Get transfer function H = Vout / Vin (or just Vout if no Vin)
        out_key = self._find_signal_key(dataset, self.output_node)
        out_signal = np.asarray(dataset[out_key].values)

        if self.input_node:
            in_key = self._find_signal_key(dataset, self.input_node)
            in_signal = np.asarray(dataset[in_key].values)
            with np.errstate(divide="ignore", invalid="ignore"):
                h = out_signal / in_signal
        else:
            h = out_signal

        # Magnitude and phase
        mag = np.abs(h)
        mag_db = 20.0 * _safe_log10(mag)
        phase_deg = np.unwrap(np.angle(h)) * (180.0 / np.pi)

        # Find unity-gain frequency (0 dB crossing)
        f_unity, idx = _interp_crossing(freq_values, mag_db, 0.0)

        if f_unity is None:
            # Fallback: find sample closest to 0 dB
            idx = int(np.argmin(np.abs(mag_db)))
            f_unity = float(freq_values[idx])
            phase_at_ugf = float(phase_deg[idx])
        else:
            # Interpolate phase at unity-gain frequency
            if idx is not None and idx < len(phase_deg) - 1:
                # Linear interpolation
                f0, f1 = freq_values[idx], freq_values[idx + 1]
                p0, p1 = phase_deg[idx], phase_deg[idx + 1]
                frac = (f_unity - f0) / (f1 - f0) if f1 != f0 else 0
                phase_at_ugf = p0 + frac * (p1 - p0)
            else:
                phase_at_ugf = float(phase_deg[idx or 0])

        # Phase margin = 180° + phase at UGF
        pm = 180.0 + float(phase_at_ugf)

        return MeasurementResult(
            value=pm,
            unit="degrees",
            passed=pm > 0,  # Negative PM = unstable
            metadata={
                "unity_gain_frequency": f_unity,
                "phase_at_ugf": float(phase_at_ugf),
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


@measurement("gain_margin")
class GainMarginMeasurement(BaseMeasurement):
    """Measure gain margin at -180° phase crossing.

    Gain margin = -|H|_dB at phase = -180°.
    Positive gain margin indicates stability.

    Example:
        >>> gm = GainMarginMeasurement(output_node="vout", input_node="vin")
        >>> result = gm.measure(dataset)
        >>> print(f"Gain margin: {result.value:.1f} dB")
    """

    name = "gain_margin"
    description = "Gain margin at -180° phase crossing"
    required_analyses = ["ac"]

    def __init__(self, output_node: str, input_node: str | None = None):
        """Initialize gain margin measurement.

        Args:
            output_node: Output signal node name
            input_node: Input signal node name (for transfer function)
        """
        self.output_node = output_node
        self.input_node = input_node

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate gain margin at -180° phase."""
        # Get frequency axis
        freq = dataset.coords.get("frequency", dataset.coords.get("freq"))
        if freq is None:
            raise ValueError("Dataset must have 'frequency' coordinate")
        freq_values = np.asarray(freq.values)

        # Get transfer function
        out_key = self._find_signal_key(dataset, self.output_node)
        out_signal = np.asarray(dataset[out_key].values)

        if self.input_node:
            in_key = self._find_signal_key(dataset, self.input_node)
            in_signal = np.asarray(dataset[in_key].values)
            with np.errstate(divide="ignore", invalid="ignore"):
                h = out_signal / in_signal
        else:
            h = out_signal

        # Magnitude and phase
        mag = np.abs(h)
        mag_db = 20.0 * _safe_log10(mag)
        phase_deg = np.unwrap(np.angle(h)) * (180.0 / np.pi)

        # Find -180° phase crossing
        # Shift phase so we're looking for 0 crossing
        phase_shifted = phase_deg + 180.0
        f_phase, idx = _interp_crossing(freq_values, phase_shifted, 0.0)

        if f_phase is None:
            # No crossing found
            return MeasurementResult(
                value=float("inf"),
                unit="dB",
                passed=True,
                metadata={
                    "crossing_frequency": None,
                    "no_crossing": True,
                },
            )

        # Interpolate magnitude at phase crossing
        if idx is not None and idx < len(mag_db) - 1:
            f0, f1 = freq_values[idx], freq_values[idx + 1]
            m0, m1 = mag_db[idx], mag_db[idx + 1]
            frac = (f_phase - f0) / (f1 - f0) if f1 != f0 else 0
            mag_at_crossing = m0 + frac * (m1 - m0)
        else:
            mag_at_crossing = float(mag_db[idx or 0])

        # Gain margin = -magnitude at -180°
        gm = -float(mag_at_crossing)

        return MeasurementResult(
            value=gm,
            unit="dB",
            passed=gm > 0,  # Negative GM = unstable
            metadata={
                "crossing_frequency": f_phase,
                "magnitude_at_crossing": float(mag_at_crossing),
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


@measurement("psrr")
class PSRRMeasurement(BaseMeasurement):
    """Power Supply Rejection Ratio measurement.

    PSRR = 20*log10(ΔVsupply / ΔVout) in dB.
    Higher PSRR indicates better supply noise rejection.

    Note: Requires AC analysis with supply as the stimulus.

    Example:
        >>> psrr = PSRRMeasurement(output_node="vout", supply_node="vdd", frequency=1000)
        >>> result = psrr.measure(dataset)
        >>> print(f"PSRR at 1kHz: {result.value:.1f} dB")
    """

    name = "psrr"
    description = "Power supply rejection ratio"
    required_analyses = ["ac"]

    def __init__(self, output_node: str, supply_node: str, frequency: float):
        """Initialize PSRR measurement.

        Args:
            output_node: Output signal node
            supply_node: Supply voltage node (AC stimulus source)
            frequency: Measurement frequency in Hz
        """
        self.output_node = output_node
        self.supply_node = supply_node
        self.frequency = frequency

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate PSRR at specified frequency."""
        # Get frequency axis
        freq = dataset.coords.get("frequency", dataset.coords.get("freq"))
        if freq is None:
            raise ValueError("Dataset must have 'frequency' coordinate")
        freq_values = np.asarray(freq.values)

        # Find nearest frequency
        freq_idx = int(np.argmin(np.abs(freq_values - self.frequency)))
        actual_freq = float(freq_values[freq_idx])

        # Get output and supply signals
        out_key = self._find_signal_key(dataset, self.output_node)
        supply_key = self._find_signal_key(dataset, self.supply_node)

        v_out = np.abs(dataset[out_key].values[freq_idx])
        v_supply = np.abs(dataset[supply_key].values[freq_idx])

        # PSRR = ΔVsupply / ΔVout (in dB, higher is better)
        if v_out == 0:
            psrr_db = float("inf")
        else:
            psrr_abs = v_supply / v_out
            psrr_db = 20.0 * np.log10(psrr_abs)

        return MeasurementResult(
            value=float(psrr_db),
            unit="dB",
            metadata={
                "frequency": actual_freq,
                "v_out": float(v_out),
                "v_supply": float(v_supply),
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


@measurement("input_impedance")
class InputImpedanceMeasurement(BaseMeasurement):
    """Measure input impedance at a specified frequency.

    Z_in = V_in / I_in

    Example:
        >>> z_in = InputImpedanceMeasurement(
        ...     voltage_node="vin", current_node="i(vin)", frequency=1000
        ... )
        >>> result = z_in.measure(dataset)
        >>> print(f"Input impedance: {result.value/1000:.2f} kΩ")
    """

    name = "input_impedance"
    description = "Input impedance at specified frequency"
    required_analyses = ["ac"]

    def __init__(self, voltage_node: str, current_node: str, frequency: float):
        """Initialize input impedance measurement.

        Args:
            voltage_node: Input voltage node
            current_node: Input current node (e.g., "I(Vin)")
            frequency: Measurement frequency in Hz
        """
        self.voltage_node = voltage_node
        self.current_node = current_node
        self.frequency = frequency

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate input impedance."""
        freq = dataset.coords.get("frequency", dataset.coords.get("freq"))
        if freq is None:
            raise ValueError("Dataset must have 'frequency' coordinate")
        freq_values = np.asarray(freq.values)

        freq_idx = int(np.argmin(np.abs(freq_values - self.frequency)))
        actual_freq = float(freq_values[freq_idx])

        v_key = self._find_signal_key(dataset, self.voltage_node)
        i_key = self._find_signal_key(dataset, self.current_node)

        v_in = dataset[v_key].values[freq_idx]
        i_in = dataset[i_key].values[freq_idx]

        if np.abs(i_in) == 0:
            z_mag = float("inf")
            z_phase = 0.0
        else:
            z = v_in / i_in
            z_mag = float(np.abs(z))
            z_phase = float(np.angle(z, deg=True))

        return MeasurementResult(
            value=z_mag,
            unit="Ohm",
            metadata={
                "frequency": actual_freq,
                "phase_deg": z_phase,
                "v_in": complex(v_in),
                "i_in": complex(i_in),
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


@measurement("cmrr")
class CMRRMeasurement(BaseMeasurement):
    """Common-Mode Rejection Ratio measurement.

    CMRR = 20*log10(Adiff / Acm) in dB.
    Higher CMRR indicates better common-mode noise rejection.

    Requires either:
    1. Two separate AC analyses (differential and common-mode), or
    2. A single analysis with pre-computed differential and common-mode gains.

    Note: This measurement supports multiple input modes:
    - Two separate datasets (diff_gain, cm_gain as metadata)
    - Single dataset with differential and common-mode output nodes
    - Pre-computed gains passed as parameters

    Example:
        >>> # From differential and common-mode output nodes
        >>> cmrr = CMRRMeasurement(
        ...     diff_output_node="vout_diff",
        ...     cm_output_node="vout_cm",
        ...     frequency=1000
        ... )
        >>> result = cmrr.measure(dataset)
        >>> print(f"CMRR at 1kHz: {result.value:.1f} dB")

        >>> # From pre-computed gains
        >>> cmrr = CMRRMeasurement(
        ...     diff_gain=1000,  # V/V
        ...     cm_gain=0.01,    # V/V
        ... )
        >>> result = cmrr.measure(dataset)  # dataset ignored
    """

    name = "cmrr"
    description = "Common-mode rejection ratio"
    required_analyses = ["ac"]

    def __init__(
        self,
        diff_output_node: str | None = None,
        cm_output_node: str | None = None,
        input_node: str | None = None,
        frequency: float | None = None,
        diff_gain: float | None = None,
        cm_gain: float | None = None,
    ):
        """Initialize CMRR measurement.

        Args:
            diff_output_node: Differential output node (for dataset-based measurement)
            cm_output_node: Common-mode output node (for dataset-based measurement)
            input_node: Input node (optional, for ratio calculation)
            frequency: Measurement frequency in Hz (None = DC/lowest)
            diff_gain: Pre-computed differential gain in V/V (optional)
            cm_gain: Pre-computed common-mode gain in V/V (optional)
        """
        self.diff_output_node = diff_output_node
        self.cm_output_node = cm_output_node
        self.input_node = input_node
        self.frequency = frequency
        self.diff_gain = diff_gain
        self.cm_gain = cm_gain

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate CMRR."""
        # Mode 1: Pre-computed gains
        if self.diff_gain is not None and self.cm_gain is not None:
            adiff = self.diff_gain
            acm = self.cm_gain

            if acm == 0:
                cmrr_db = float("inf")
            else:
                cmrr_db = 20.0 * np.log10(abs(adiff) / abs(acm))

            return MeasurementResult(
                value=float(cmrr_db),
                unit="dB",
                metadata={
                    "diff_gain": adiff,
                    "cm_gain": acm,
                    "mode": "pre_computed",
                },
            )

        # Mode 2: Extract from dataset
        if self.diff_output_node is None or self.cm_output_node is None:
            raise ValueError(
                "Must provide either (diff_gain, cm_gain) or " "(diff_output_node, cm_output_node)"
            )

        # Get frequency axis
        freq = dataset.coords.get("frequency", dataset.coords.get("freq"))
        if freq is None:
            raise ValueError("Dataset must have 'frequency' coordinate")
        freq_values = np.asarray(freq.values)

        # Determine target frequency
        if self.frequency is None or self.frequency == 0:
            freq_idx = 0
            target_freq = float(freq_values[0])
        else:
            freq_idx = int(np.argmin(np.abs(freq_values - self.frequency)))
            target_freq = float(freq_values[freq_idx])

        # Get differential and common-mode outputs
        diff_key = self._find_signal_key(dataset, self.diff_output_node)
        cm_key = self._find_signal_key(dataset, self.cm_output_node)

        diff_mag = float(np.abs(dataset[diff_key].values[freq_idx]))
        cm_mag = float(np.abs(dataset[cm_key].values[freq_idx]))

        # If input node specified, compute actual gains
        if self.input_node:
            in_key = self._find_signal_key(dataset, self.input_node)
            in_mag = float(np.abs(dataset[in_key].values[freq_idx]))
            if in_mag > 0:
                diff_mag = diff_mag / in_mag
                cm_mag = cm_mag / in_mag

        # Calculate CMRR
        if cm_mag == 0:
            cmrr_db = float("inf")
        else:
            cmrr_db = 20.0 * np.log10(diff_mag / cm_mag)

        return MeasurementResult(
            value=float(cmrr_db),
            unit="dB",
            metadata={
                "frequency": target_freq,
                "diff_gain": diff_mag,
                "cm_gain": cm_mag,
                "mode": "dataset",
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
    "GainMeasurement",
    "BandwidthMeasurement",
    "PhaseMarginMeasurement",
    "GainMarginMeasurement",
    "PSRRMeasurement",
    "InputImpedanceMeasurement",
    "CMRRMeasurement",
]
