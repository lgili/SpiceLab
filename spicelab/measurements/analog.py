"""Analog circuit measurements.

High-level measurement functions for analog circuit characterization:
- PSRR (Power Supply Rejection Ratio) - single point and vs frequency
- CMRR (Common Mode Rejection Ratio) - single point and vs frequency
- Noise Analysis - spot noise, integrated noise, noise spectral density

These functions provide complete workflows including circuit setup suggestions
and result processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseMeasurement, MeasurementResult
from .registry import measurement

if TYPE_CHECKING:
    import xarray as xr


@dataclass
class PSRRResult:
    """Results from PSRR measurement.

    Attributes:
        value_db: PSRR value in dB at target frequency
        frequency: Target frequency in Hz
        frequencies: Frequency array for sweep (if available)
        psrr_vs_freq: PSRR vs frequency array in dB (if available)
        v_out: Output voltage magnitude
        v_supply: Supply voltage magnitude
        metadata: Additional measurement info
    """

    value_db: float
    frequency: float
    frequencies: np.ndarray | None = None
    psrr_vs_freq: np.ndarray | None = None
    v_out: float = 0.0
    v_supply: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"PSRR: {self.value_db:.1f} dB at {self.frequency:.0f} Hz"

    @property
    def passed(self) -> bool | None:
        """Check if PSRR meets typical specs (>60dB at 1kHz)."""
        if self.frequency <= 1000:
            return self.value_db >= 60
        return None


@dataclass
class CMRRResult:
    """Results from CMRR measurement.

    Attributes:
        value_db: CMRR value in dB at target frequency
        frequency: Target frequency in Hz
        frequencies: Frequency array for sweep (if available)
        cmrr_vs_freq: CMRR vs frequency array in dB (if available)
        diff_gain: Differential gain (V/V)
        cm_gain: Common-mode gain (V/V)
        metadata: Additional measurement info
    """

    value_db: float
    frequency: float
    frequencies: np.ndarray | None = None
    cmrr_vs_freq: np.ndarray | None = None
    diff_gain: float = 0.0
    cm_gain: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"CMRR: {self.value_db:.1f} dB at {self.frequency:.0f} Hz"

    @property
    def passed(self) -> bool | None:
        """Check if CMRR meets typical specs (>80dB at DC)."""
        if self.frequency <= 100:
            return self.value_db >= 80
        return None


@dataclass
class NoiseResult:
    """Results from noise analysis.

    Attributes:
        spot_noise: Spot noise density at target frequency (V/sqrt(Hz) or A/sqrt(Hz))
        integrated_noise: Integrated RMS noise over bandwidth (V or A)
        frequency: Target frequency for spot noise (Hz)
        freq_min: Lower frequency bound for integration (Hz)
        freq_max: Upper frequency bound for integration (Hz)
        frequencies: Frequency array (if available)
        noise_density: Noise spectral density array (if available)
        equivalent_input_noise: Input-referred noise (if gain provided)
        unit: Unit of noise measurement
        metadata: Additional measurement info
    """

    spot_noise: float
    integrated_noise: float
    frequency: float
    freq_min: float
    freq_max: float
    frequencies: np.ndarray | None = None
    noise_density: np.ndarray | None = None
    equivalent_input_noise: float | None = None
    unit: str = "V/sqrt(Hz)"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        spot_str = f"Spot noise: {self.spot_noise:.2e} {self.unit} at {self.frequency:.0f} Hz"
        int_str = f"Integrated: {self.integrated_noise:.2e} V rms ({self.freq_min:.0f}-{self.freq_max:.0f} Hz)"
        return f"{spot_str}\n{int_str}"


def _safe_db(value: float, ref: float = 1.0) -> float:
    """Calculate dB safely, handling zero values."""
    if value <= 0 or ref <= 0:
        return float("-inf")
    return 20.0 * np.log10(abs(value) / abs(ref))


def _find_signal_key(dataset: xr.Dataset, node: str) -> str:
    """Find the actual key in dataset for a node name."""
    if node in dataset.data_vars:
        return node

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

    raise KeyError(f"Signal '{node}' not found in dataset. Available: {list(dataset.data_vars)}")


@measurement("psrr_sweep")
class PSRRSweepMeasurement(BaseMeasurement):
    """PSRR measurement with frequency sweep capability.

    Power Supply Rejection Ratio (PSRR) measures how well a circuit
    rejects noise on its power supply.

    PSRR = 20*log10(ΔVsupply / ΔVout) in dB

    Higher values indicate better supply noise rejection.

    Setup Requirements:
        - AC analysis with supply node as stimulus (AC source on VDD)
        - Output node must be specified
        - For accurate measurement, input should be at DC or grounded

    Example:
        >>> # Measure PSRR at 1kHz
        >>> psrr = PSRRSweepMeasurement(
        ...     output_node="vout",
        ...     supply_node="vdd",
        ...     frequency=1000
        ... )
        >>> result = psrr.measure(dataset)
        >>> print(f"PSRR: {result.value} dB")

        >>> # Get PSRR vs frequency
        >>> psrr = PSRRSweepMeasurement(
        ...     output_node="vout",
        ...     supply_node="vdd",
        ...     return_sweep=True
        ... )
        >>> result = psrr.measure(dataset)
        >>> # result.metadata contains frequency array and PSRR array
    """

    name = "psrr_sweep"
    description = "Power supply rejection ratio with frequency sweep"
    required_analyses = ["ac"]

    def __init__(
        self,
        output_node: str,
        supply_node: str,
        frequency: float | None = None,
        return_sweep: bool = False,
    ):
        """Initialize PSRR measurement.

        Args:
            output_node: Output signal node (e.g., "vout")
            supply_node: Supply voltage node with AC stimulus (e.g., "vdd")
            frequency: Target frequency for single-point measurement (Hz)
                      If None and return_sweep=False, uses 1kHz
            return_sweep: If True, include full frequency sweep in metadata
        """
        self.output_node = output_node
        self.supply_node = supply_node
        self.frequency = frequency if frequency is not None else 1000.0
        self.return_sweep = return_sweep

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate PSRR from AC analysis results."""
        # Get frequency axis
        freq = dataset.coords.get("frequency", dataset.coords.get("freq"))
        if freq is None:
            raise ValueError(
                "Dataset must have 'frequency' or 'freq' coordinate. "
                "Ensure you ran an AC analysis."
            )
        freq_values = np.asarray(freq.values)

        # Get output and supply signals
        out_key = _find_signal_key(dataset, self.output_node)
        supply_key = _find_signal_key(dataset, self.supply_node)

        out_signal = np.asarray(dataset[out_key].values)
        supply_signal = np.asarray(dataset[supply_key].values)

        # Calculate PSRR at all frequencies
        out_mag = np.abs(out_signal)
        supply_mag = np.abs(supply_signal)

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            psrr_ratio = supply_mag / np.where(out_mag > 0, out_mag, np.inf)
            psrr_db = 20.0 * np.log10(np.where(psrr_ratio > 0, psrr_ratio, 1e-30))

        # Find nearest frequency to target
        freq_idx = int(np.argmin(np.abs(freq_values - self.frequency)))
        actual_freq = float(freq_values[freq_idx])
        psrr_at_freq = float(psrr_db[freq_idx])

        # Build metadata
        metadata = {
            "frequency": actual_freq,
            "output_node": self.output_node,
            "supply_node": self.supply_node,
            "v_out": float(out_mag[freq_idx]),
            "v_supply": float(supply_mag[freq_idx]),
        }

        if self.return_sweep:
            metadata["frequencies"] = freq_values.tolist()
            metadata["psrr_vs_freq"] = psrr_db.tolist()

        return MeasurementResult(
            value=psrr_at_freq,
            unit="dB",
            metadata=metadata,
        )


@measurement("cmrr_sweep")
class CMRRSweepMeasurement(BaseMeasurement):
    """CMRR measurement with frequency sweep capability.

    Common-Mode Rejection Ratio (CMRR) measures how well a differential
    amplifier rejects signals common to both inputs.

    CMRR = 20*log10(A_diff / A_cm) in dB

    Higher values indicate better common-mode noise rejection.

    Measurement Methods:
        1. Two separate analyses: Run differential and common-mode AC analyses
        2. Single analysis: Use circuit with both outputs available
        3. Pre-computed: Provide diff_gain and cm_gain directly

    Example:
        >>> # From two output nodes (single analysis with both measurements)
        >>> cmrr = CMRRSweepMeasurement(
        ...     diff_output_node="vout_diff",
        ...     cm_output_node="vout_cm",
        ...     input_node="vin",
        ...     frequency=1000
        ... )
        >>> result = cmrr.measure(dataset)

        >>> # From pre-computed gains
        >>> cmrr = CMRRSweepMeasurement(diff_gain=1000, cm_gain=0.01)
        >>> result = cmrr.measure(dataset)  # dataset ignored
    """

    name = "cmrr_sweep"
    description = "Common-mode rejection ratio with frequency sweep"
    required_analyses = ["ac"]

    def __init__(
        self,
        diff_output_node: str | None = None,
        cm_output_node: str | None = None,
        input_node: str | None = None,
        frequency: float | None = None,
        diff_gain: float | None = None,
        cm_gain: float | None = None,
        return_sweep: bool = False,
    ):
        """Initialize CMRR measurement.

        Args:
            diff_output_node: Differential output node
            cm_output_node: Common-mode output node
            input_node: Input node (for gain calculation)
            frequency: Target frequency (Hz). If None, uses DC/lowest
            diff_gain: Pre-computed differential gain (V/V)
            cm_gain: Pre-computed common-mode gain (V/V)
            return_sweep: If True, include full frequency sweep in metadata
        """
        self.diff_output_node = diff_output_node
        self.cm_output_node = cm_output_node
        self.input_node = input_node
        self.frequency = frequency
        self.diff_gain = diff_gain
        self.cm_gain = cm_gain
        self.return_sweep = return_sweep

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate CMRR."""
        # Mode 1: Pre-computed gains
        if self.diff_gain is not None and self.cm_gain is not None:
            cmrr_db = _safe_db(self.diff_gain, self.cm_gain)
            return MeasurementResult(
                value=cmrr_db,
                unit="dB",
                metadata={
                    "diff_gain": self.diff_gain,
                    "cm_gain": self.cm_gain,
                    "mode": "pre_computed",
                },
            )

        # Mode 2: From dataset
        if self.diff_output_node is None or self.cm_output_node is None:
            raise ValueError(
                "Must provide either (diff_gain, cm_gain) or " "(diff_output_node, cm_output_node)"
            )

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
        diff_key = _find_signal_key(dataset, self.diff_output_node)
        cm_key = _find_signal_key(dataset, self.cm_output_node)

        diff_signal = np.asarray(dataset[diff_key].values)
        cm_signal = np.asarray(dataset[cm_key].values)

        diff_mag = np.abs(diff_signal)
        cm_mag = np.abs(cm_signal)

        # If input node specified, compute actual gains
        if self.input_node:
            in_key = _find_signal_key(dataset, self.input_node)
            in_signal = np.asarray(dataset[in_key].values)
            in_mag = np.abs(in_signal)
            with np.errstate(divide="ignore", invalid="ignore"):
                diff_mag = diff_mag / np.where(in_mag > 0, in_mag, 1)
                cm_mag = cm_mag / np.where(in_mag > 0, in_mag, 1)

        # Calculate CMRR at all frequencies
        with np.errstate(divide="ignore", invalid="ignore"):
            cmrr_ratio = diff_mag / np.where(cm_mag > 0, cm_mag, 1e-30)
            cmrr_db = 20.0 * np.log10(np.where(cmrr_ratio > 0, cmrr_ratio, 1e-30))

        cmrr_at_freq = float(cmrr_db[freq_idx])

        metadata = {
            "frequency": target_freq,
            "diff_output_node": self.diff_output_node,
            "cm_output_node": self.cm_output_node,
            "diff_gain": float(diff_mag[freq_idx]),
            "cm_gain": float(cm_mag[freq_idx]),
            "mode": "dataset",
        }

        if self.return_sweep:
            metadata["frequencies"] = freq_values.tolist()
            metadata["cmrr_vs_freq"] = cmrr_db.tolist()

        return MeasurementResult(
            value=cmrr_at_freq,
            unit="dB",
            metadata=metadata,
        )


@measurement("noise_analysis")
class NoiseAnalysisMeasurement(BaseMeasurement):
    """Noise analysis measurement.

    Extracts noise spectral density from SPICE noise analysis results.
    Provides both spot noise (at a specific frequency) and integrated
    noise over a bandwidth.

    Noise Types:
        - Spot noise: Noise density at a specific frequency (V/sqrt(Hz))
        - Integrated noise: Total RMS noise over a frequency range (V rms)
        - Input-referred noise: Output noise divided by gain

    Setup Requirements:
        - SPICE noise analysis must be run
        - Dataset should contain 'onoise' (output noise) or 'inoise' (input noise)

    Example:
        >>> # Measure output noise
        >>> noise = NoiseAnalysisMeasurement(
        ...     noise_node="onoise",
        ...     frequency=1000,
        ...     freq_min=10,
        ...     freq_max=100000
        ... )
        >>> result = noise.measure(dataset)
        >>> print(f"Spot noise at 1kHz: {result.metadata['spot_noise']:.2e} V/sqrt(Hz)")
        >>> print(f"Integrated noise: {result.value:.2e} V rms")
    """

    name = "noise_analysis"
    description = "Noise spectral density and integrated noise"
    required_analyses = ["noise"]

    def __init__(
        self,
        noise_node: str = "onoise",
        frequency: float = 1000.0,
        freq_min: float = 10.0,
        freq_max: float = 100000.0,
        gain: float | None = None,
        return_spectrum: bool = False,
    ):
        """Initialize noise measurement.

        Args:
            noise_node: Noise output node (typically "onoise" or "inoise")
            frequency: Target frequency for spot noise (Hz)
            freq_min: Lower frequency bound for integration (Hz)
            freq_max: Upper frequency bound for integration (Hz)
            gain: Circuit gain for input-referred noise calculation
            return_spectrum: If True, include noise spectrum in metadata
        """
        self.noise_node = noise_node
        self.frequency = frequency
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.gain = gain
        self.return_spectrum = return_spectrum

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Extract noise from noise analysis results."""
        # Get frequency axis
        freq = dataset.coords.get("frequency", dataset.coords.get("freq"))
        if freq is None:
            raise ValueError(
                "Dataset must have 'frequency' coordinate. " "Ensure you ran a noise analysis."
            )
        freq_values = np.asarray(freq.values)

        # Get noise data
        noise_key = _find_signal_key(dataset, self.noise_node)
        noise_data = np.asarray(dataset[noise_key].values)

        # For noise analysis, data is typically real (spectral density)
        # but might be complex from some simulators
        noise_density = np.abs(noise_data)

        # Spot noise at target frequency
        spot_idx = int(np.argmin(np.abs(freq_values - self.frequency)))
        spot_noise = float(noise_density[spot_idx])
        actual_spot_freq = float(freq_values[spot_idx])

        # Integrated noise over bandwidth
        # Find indices for integration range
        freq_mask = (freq_values >= self.freq_min) & (freq_values <= self.freq_max)

        if not np.any(freq_mask):
            raise ValueError(f"No frequencies in range [{self.freq_min}, {self.freq_max}] Hz")

        freq_band = freq_values[freq_mask]
        noise_band = noise_density[freq_mask]

        # Integrate noise power (noise_density^2) and take sqrt
        # Use trapezoidal integration
        noise_power = np.trapezoid(noise_band**2, freq_band)
        integrated_noise = float(np.sqrt(noise_power))

        # Input-referred noise if gain is provided
        input_referred = None
        if self.gain is not None and self.gain != 0:
            input_referred = integrated_noise / abs(self.gain)

        metadata = {
            "spot_noise": spot_noise,
            "spot_frequency": actual_spot_freq,
            "integrated_freq_min": self.freq_min,
            "integrated_freq_max": self.freq_max,
            "noise_node": self.noise_node,
        }

        if input_referred is not None:
            metadata["input_referred_noise"] = input_referred
            metadata["gain"] = self.gain

        if self.return_spectrum:
            metadata["frequencies"] = freq_values.tolist()
            metadata["noise_density"] = noise_density.tolist()

        return MeasurementResult(
            value=integrated_noise,
            unit="V rms",
            metadata=metadata,
        )


@measurement("spot_noise")
class SpotNoiseMeasurement(BaseMeasurement):
    """Spot noise measurement at a specific frequency.

    Returns the noise spectral density at a single frequency point.

    Example:
        >>> spot = SpotNoiseMeasurement(noise_node="onoise", frequency=1000)
        >>> result = spot.measure(dataset)
        >>> print(f"Noise at 1kHz: {result.value:.2e} V/sqrt(Hz)")
    """

    name = "spot_noise"
    description = "Spot noise at specified frequency"
    required_analyses = ["noise"]

    def __init__(self, noise_node: str = "onoise", frequency: float = 1000.0):
        """Initialize spot noise measurement.

        Args:
            noise_node: Noise output node (typically "onoise" or "inoise")
            frequency: Target frequency (Hz)
        """
        self.noise_node = noise_node
        self.frequency = frequency

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Extract spot noise at target frequency."""
        freq = dataset.coords.get("frequency", dataset.coords.get("freq"))
        if freq is None:
            raise ValueError("Dataset must have 'frequency' coordinate")
        freq_values = np.asarray(freq.values)

        noise_key = _find_signal_key(dataset, self.noise_node)
        noise_data = np.asarray(dataset[noise_key].values)
        noise_density = np.abs(noise_data)

        freq_idx = int(np.argmin(np.abs(freq_values - self.frequency)))
        spot_noise = float(noise_density[freq_idx])
        actual_freq = float(freq_values[freq_idx])

        return MeasurementResult(
            value=spot_noise,
            unit="V/sqrt(Hz)",
            metadata={
                "frequency": actual_freq,
                "noise_node": self.noise_node,
            },
        )


@measurement("integrated_noise")
class IntegratedNoiseMeasurement(BaseMeasurement):
    """Integrated noise over a frequency range.

    Calculates total RMS noise by integrating the noise spectral density
    over a specified bandwidth.

    integrated_noise = sqrt(integral(noise_density^2 * df))

    Example:
        >>> int_noise = IntegratedNoiseMeasurement(
        ...     noise_node="onoise",
        ...     freq_min=10,
        ...     freq_max=100000
        ... )
        >>> result = int_noise.measure(dataset)
        >>> print(f"RMS noise: {result.value:.2e} V")
    """

    name = "integrated_noise"
    description = "Integrated RMS noise over frequency band"
    required_analyses = ["noise"]

    def __init__(
        self,
        noise_node: str = "onoise",
        freq_min: float = 10.0,
        freq_max: float = 100000.0,
    ):
        """Initialize integrated noise measurement.

        Args:
            noise_node: Noise output node
            freq_min: Lower frequency bound (Hz)
            freq_max: Upper frequency bound (Hz)
        """
        self.noise_node = noise_node
        self.freq_min = freq_min
        self.freq_max = freq_max

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate integrated noise over bandwidth."""
        freq = dataset.coords.get("frequency", dataset.coords.get("freq"))
        if freq is None:
            raise ValueError("Dataset must have 'frequency' coordinate")
        freq_values = np.asarray(freq.values)

        noise_key = _find_signal_key(dataset, self.noise_node)
        noise_data = np.asarray(dataset[noise_key].values)
        noise_density = np.abs(noise_data)

        # Find integration range
        freq_mask = (freq_values >= self.freq_min) & (freq_values <= self.freq_max)

        if not np.any(freq_mask):
            raise ValueError(f"No frequencies in range [{self.freq_min}, {self.freq_max}] Hz")

        freq_band = freq_values[freq_mask]
        noise_band = noise_density[freq_mask]

        # Integrate noise power
        noise_power = np.trapezoid(noise_band**2, freq_band)
        integrated = float(np.sqrt(noise_power))

        return MeasurementResult(
            value=integrated,
            unit="V rms",
            metadata={
                "freq_min": self.freq_min,
                "freq_max": self.freq_max,
                "noise_node": self.noise_node,
                "bandwidth": self.freq_max - self.freq_min,
            },
        )


# Convenience functions for quick measurements


def measure_psrr(
    dataset: xr.Dataset,
    output_node: str,
    supply_node: str,
    frequency: float = 1000.0,
) -> PSRRResult:
    """Measure PSRR at a specific frequency.

    Convenience function for quick PSRR measurement.

    Args:
        dataset: AC analysis results with supply as AC stimulus
        output_node: Output node name
        supply_node: Supply node name (AC source)
        frequency: Target frequency in Hz (default 1kHz)

    Returns:
        PSRRResult with measurement data

    Example:
        >>> result = measure_psrr(dataset, "vout", "vdd", frequency=1000)
        >>> print(f"PSRR: {result.value_db:.1f} dB")
    """
    meas = PSRRSweepMeasurement(
        output_node=output_node,
        supply_node=supply_node,
        frequency=frequency,
        return_sweep=True,
    )
    result = meas.measure(dataset)

    return PSRRResult(
        value_db=result.value,
        frequency=result.metadata.get("frequency", frequency),
        frequencies=np.array(result.metadata.get("frequencies", [])),
        psrr_vs_freq=np.array(result.metadata.get("psrr_vs_freq", [])),
        v_out=result.metadata.get("v_out", 0.0),
        v_supply=result.metadata.get("v_supply", 0.0),
        metadata=result.metadata,
    )


def measure_cmrr(
    dataset: xr.Dataset,
    diff_output_node: str,
    cm_output_node: str,
    input_node: str | None = None,
    frequency: float | None = None,
) -> CMRRResult:
    """Measure CMRR at a specific frequency.

    Convenience function for quick CMRR measurement.

    Args:
        dataset: AC analysis results
        diff_output_node: Differential output node
        cm_output_node: Common-mode output node
        input_node: Input node (optional, for gain calculation)
        frequency: Target frequency in Hz (None = DC)

    Returns:
        CMRRResult with measurement data

    Example:
        >>> result = measure_cmrr(dataset, "vout_diff", "vout_cm", frequency=1000)
        >>> print(f"CMRR: {result.value_db:.1f} dB")
    """
    meas = CMRRSweepMeasurement(
        diff_output_node=diff_output_node,
        cm_output_node=cm_output_node,
        input_node=input_node,
        frequency=frequency,
        return_sweep=True,
    )
    result = meas.measure(dataset)

    return CMRRResult(
        value_db=result.value,
        frequency=result.metadata.get("frequency", frequency or 0),
        frequencies=np.array(result.metadata.get("frequencies", [])),
        cmrr_vs_freq=np.array(result.metadata.get("cmrr_vs_freq", [])),
        diff_gain=result.metadata.get("diff_gain", 0.0),
        cm_gain=result.metadata.get("cm_gain", 0.0),
        metadata=result.metadata,
    )


def measure_noise(
    dataset: xr.Dataset,
    noise_node: str = "onoise",
    frequency: float = 1000.0,
    freq_min: float = 10.0,
    freq_max: float = 100000.0,
    gain: float | None = None,
) -> NoiseResult:
    """Measure noise - spot and integrated.

    Convenience function for complete noise characterization.

    Args:
        dataset: Noise analysis results
        noise_node: Noise output node (typically "onoise" or "inoise")
        frequency: Target frequency for spot noise (Hz)
        freq_min: Lower frequency bound for integration (Hz)
        freq_max: Upper frequency bound for integration (Hz)
        gain: Circuit gain for input-referred calculation

    Returns:
        NoiseResult with spot and integrated noise

    Example:
        >>> result = measure_noise(dataset, "onoise", frequency=1000)
        >>> print(f"Spot noise: {result.spot_noise:.2e} V/sqrt(Hz)")
        >>> print(f"Integrated: {result.integrated_noise:.2e} V rms")
    """
    meas = NoiseAnalysisMeasurement(
        noise_node=noise_node,
        frequency=frequency,
        freq_min=freq_min,
        freq_max=freq_max,
        gain=gain,
        return_spectrum=True,
    )
    result = meas.measure(dataset)

    return NoiseResult(
        spot_noise=result.metadata.get("spot_noise", 0.0),
        integrated_noise=result.value,
        frequency=result.metadata.get("spot_frequency", frequency),
        freq_min=freq_min,
        freq_max=freq_max,
        frequencies=np.array(result.metadata.get("frequencies", [])),
        noise_density=np.array(result.metadata.get("noise_density", [])),
        equivalent_input_noise=result.metadata.get("input_referred_noise"),
        unit="V/sqrt(Hz)",
        metadata=result.metadata,
    )


__all__ = [
    # Result dataclasses
    "PSRRResult",
    "CMRRResult",
    "NoiseResult",
    # Measurement classes
    "PSRRSweepMeasurement",
    "CMRRSweepMeasurement",
    "NoiseAnalysisMeasurement",
    "SpotNoiseMeasurement",
    "IntegratedNoiseMeasurement",
    # Convenience functions
    "measure_psrr",
    "measure_cmrr",
    "measure_noise",
]
