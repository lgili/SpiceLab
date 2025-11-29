"""Spectrum and distortion measurements.

Measurements for frequency-domain signal quality including:
- THD (Total Harmonic Distortion)
- THD+N (THD plus noise)
- SNR (Signal-to-Noise Ratio)
- SINAD (Signal-to-Noise and Distortion)
- SFDR (Spurious-Free Dynamic Range)
- ENOB (Effective Number of Bits)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .base import BaseMeasurement, MeasurementResult
from .registry import measurement

if TYPE_CHECKING:
    import xarray as xr

WindowType = Literal["rect", "hann", "hamming", "blackman", "flattop"]


def _get_window(n: int, window_type: WindowType) -> np.ndarray:
    """Generate window function."""
    if window_type == "rect":
        return np.ones(n)
    k = np.arange(n)
    if window_type == "hann":
        return 0.5 * (1 - np.cos(2 * np.pi * k / (n - 1)))
    if window_type == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * k / (n - 1))
    if window_type == "blackman":
        return (
            0.42
            - 0.5 * np.cos(2 * np.pi * k / (n - 1))
            + 0.08 * np.cos(4 * np.pi * k / (n - 1))
        )
    if window_type == "flattop":
        return (
            0.21557895
            - 0.41663158 * np.cos(2 * np.pi * k / (n - 1))
            + 0.277263158 * np.cos(4 * np.pi * k / (n - 1))
            - 0.083578947 * np.cos(6 * np.pi * k / (n - 1))
            + 0.006947368 * np.cos(8 * np.pi * k / (n - 1))
        )
    raise ValueError(f"Unknown window type: {window_type}")


def _compute_fft(
    signal: np.ndarray, fs: float, window_type: WindowType = "hann"
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FFT with windowing and return (frequencies, magnitudes).

    Returns positive frequencies only with proper amplitude scaling.
    """
    n = len(signal)
    window = _get_window(n, window_type)

    # Apply window and normalize for coherent gain
    windowed = signal * window
    coherent_gain = np.sum(window)

    # Compute FFT
    fft_vals = np.fft.rfft(windowed)

    # Scale for amplitude (2x for one-sided, divide by coherent gain)
    magnitudes = 2.0 * np.abs(fft_vals) / coherent_gain

    # DC and Nyquist don't need 2x factor
    magnitudes[0] /= 2.0
    if n % 2 == 0:
        magnitudes[-1] /= 2.0

    # Frequency axis
    frequencies = np.fft.rfftfreq(n, 1.0 / fs)

    return frequencies, magnitudes


def _find_bin(frequencies: np.ndarray, target_freq: float) -> int:
    """Find FFT bin index closest to target frequency."""
    return int(np.argmin(np.abs(frequencies - target_freq)))


def _find_fundamental(
    frequencies: np.ndarray, magnitudes: np.ndarray, f0: float | None
) -> tuple[int, float]:
    """Find fundamental frequency bin.

    Args:
        frequencies: Frequency axis
        magnitudes: FFT magnitudes
        f0: Specified fundamental frequency (None = auto-detect)

    Returns:
        (bin_index, frequency)
    """
    if f0 is not None:
        idx = _find_bin(frequencies, f0)
        return idx, float(frequencies[idx])

    # Auto-detect: find largest bin (excluding DC)
    idx = 1 + int(np.argmax(magnitudes[1:]))
    return idx, float(frequencies[idx])


@measurement("thd")
class THDMeasurement(BaseMeasurement):
    """Total Harmonic Distortion measurement.

    THD = sqrt(sum(H2^2 + H3^2 + ... + Hn^2)) / H1
    where H1 is fundamental, H2..Hn are harmonics.

    Example:
        >>> thd = THDMeasurement(node="vout", fundamental_freq=1000, n_harmonics=10)
        >>> result = thd.measure(dataset)
        >>> print(f"THD: {result.value:.3f}%")
    """

    name = "thd"
    description = "Total harmonic distortion"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        fundamental_freq: float | None = None,
        n_harmonics: int = 10,
        window: WindowType = "hann",
    ):
        """Initialize THD measurement.

        Args:
            node: Signal node name
            fundamental_freq: Fundamental frequency (None = auto-detect)
            n_harmonics: Number of harmonics to include (default 10)
            window: Window function type
        """
        self.node = node
        self.fundamental_freq = fundamental_freq
        self.n_harmonics = n_harmonics
        self.window = window

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate THD from transient data."""
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        sig_key = self._find_signal_key(dataset, self.node)
        voltage = np.asarray(dataset[sig_key].values)

        # Calculate sample rate
        dt = np.mean(np.diff(time_values))
        fs = 1.0 / dt

        # Compute FFT
        frequencies, magnitudes = _compute_fft(voltage, fs, self.window)

        # Find fundamental
        fund_idx, fund_freq = _find_fundamental(
            frequencies, magnitudes, self.fundamental_freq
        )
        fund_mag = magnitudes[fund_idx]

        if fund_mag == 0:
            return MeasurementResult(
                value=float("inf"),
                unit="%",
                metadata={"error": "Zero fundamental magnitude"},
            )

        # Sum harmonic powers
        harmonic_power = 0.0
        harmonic_mags = []

        for n in range(2, self.n_harmonics + 2):
            harm_freq = n * fund_freq
            if harm_freq > frequencies[-1]:
                break
            harm_idx = _find_bin(frequencies, harm_freq)
            harm_mag = magnitudes[harm_idx]
            harmonic_mags.append(float(harm_mag))
            harmonic_power += harm_mag**2

        # THD = sqrt(sum of harmonic powers) / fundamental
        thd = np.sqrt(harmonic_power) / fund_mag
        thd_pct = thd * 100.0
        thd_db = 20.0 * np.log10(thd) if thd > 0 else float("-inf")

        return MeasurementResult(
            value=float(thd_pct),
            unit="%",
            metadata={
                "thd_db": thd_db,
                "fundamental_freq": fund_freq,
                "fundamental_magnitude": float(fund_mag),
                "harmonic_magnitudes": harmonic_mags,
                "n_harmonics": len(harmonic_mags),
                "window": self.window,
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


@measurement("snr")
class SNRMeasurement(BaseMeasurement):
    """Signal-to-Noise Ratio measurement.

    SNR = 10*log10(signal_power / noise_power)

    Example:
        >>> snr = SNRMeasurement(node="vout", signal_freq=1000)
        >>> result = snr.measure(dataset)
        >>> print(f"SNR: {result.value:.1f} dB")
    """

    name = "snr"
    description = "Signal-to-noise ratio"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        signal_freq: float | None = None,
        noise_band: tuple[float, float] | None = None,
        window: WindowType = "hann",
    ):
        """Initialize SNR measurement.

        Args:
            node: Signal node name
            signal_freq: Signal frequency (None = auto-detect)
            noise_band: (fmin, fmax) for noise measurement (None = full band)
            window: Window function type
        """
        self.node = node
        self.signal_freq = signal_freq
        self.noise_band = noise_band
        self.window = window

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate SNR."""
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        sig_key = self._find_signal_key(dataset, self.node)
        voltage = np.asarray(dataset[sig_key].values)

        dt = np.mean(np.diff(time_values))
        fs = 1.0 / dt

        frequencies, magnitudes = _compute_fft(voltage, fs, self.window)

        # Find signal
        sig_idx, sig_freq = _find_fundamental(frequencies, magnitudes, self.signal_freq)
        sig_power = magnitudes[sig_idx] ** 2

        # Calculate noise power (excluding signal bin and DC)
        noise_mask = np.ones(len(magnitudes), dtype=bool)
        noise_mask[0] = False  # Exclude DC
        noise_mask[sig_idx] = False  # Exclude signal

        # Also exclude harmonics
        for n in range(2, 11):
            harm_freq = n * sig_freq
            if harm_freq < frequencies[-1]:
                harm_idx = _find_bin(frequencies, harm_freq)
                noise_mask[harm_idx] = False

        # Apply noise band filter if specified
        if self.noise_band:
            band_mask = (frequencies >= self.noise_band[0]) & (
                frequencies <= self.noise_band[1]
            )
            noise_mask &= band_mask

        noise_magnitudes = magnitudes[noise_mask]
        noise_power = float(np.sum(noise_magnitudes**2))

        if noise_power == 0:
            snr_db = float("inf")
        else:
            snr_db = 10.0 * np.log10(sig_power / noise_power)

        return MeasurementResult(
            value=float(snr_db),
            unit="dB",
            metadata={
                "signal_freq": sig_freq,
                "signal_power": float(sig_power),
                "noise_power": noise_power,
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


@measurement("sinad")
class SINADMeasurement(BaseMeasurement):
    """Signal-to-Noise-and-Distortion measurement.

    SINAD = 10*log10(signal_power / (noise_power + distortion_power))

    Example:
        >>> sinad = SINADMeasurement(node="vout", signal_freq=1000)
        >>> result = sinad.measure(dataset)
        >>> print(f"SINAD: {result.value:.1f} dB")
    """

    name = "sinad"
    description = "Signal to noise and distortion ratio"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        signal_freq: float | None = None,
        window: WindowType = "hann",
    ):
        """Initialize SINAD measurement.

        Args:
            node: Signal node name
            signal_freq: Signal frequency (None = auto-detect)
            window: Window function type
        """
        self.node = node
        self.signal_freq = signal_freq
        self.window = window

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate SINAD."""
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        sig_key = self._find_signal_key(dataset, self.node)
        voltage = np.asarray(dataset[sig_key].values)

        dt = np.mean(np.diff(time_values))
        fs = 1.0 / dt

        frequencies, magnitudes = _compute_fft(voltage, fs, self.window)

        # Find signal
        sig_idx, sig_freq = _find_fundamental(frequencies, magnitudes, self.signal_freq)
        sig_power = magnitudes[sig_idx] ** 2

        # Total power (excluding DC)
        total_power = float(np.sum(magnitudes[1:] ** 2))

        # Noise + distortion = total - signal
        nad_power = total_power - sig_power

        if nad_power <= 0:
            sinad_db = float("inf")
        else:
            sinad_db = 10.0 * np.log10(sig_power / nad_power)

        return MeasurementResult(
            value=float(sinad_db),
            unit="dB",
            metadata={
                "signal_freq": sig_freq,
                "signal_power": float(sig_power),
                "noise_and_distortion_power": nad_power,
                "total_power": total_power,
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


@measurement("sfdr")
class SFDRMeasurement(BaseMeasurement):
    """Spurious-Free Dynamic Range measurement.

    SFDR = fundamental magnitude / largest spurious magnitude (in dB)

    Example:
        >>> sfdr = SFDRMeasurement(node="vout", fundamental_freq=1000)
        >>> result = sfdr.measure(dataset)
        >>> print(f"SFDR: {result.value:.1f} dBc")
    """

    name = "sfdr"
    description = "Spurious-free dynamic range"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        fundamental_freq: float | None = None,
        window: WindowType = "hann",
    ):
        """Initialize SFDR measurement.

        Args:
            node: Signal node name
            fundamental_freq: Fundamental frequency (None = auto-detect)
            window: Window function type
        """
        self.node = node
        self.fundamental_freq = fundamental_freq
        self.window = window

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate SFDR."""
        time = dataset.coords.get("time")
        if time is None:
            raise ValueError("Dataset must have 'time' coordinate")
        time_values = np.asarray(time.values)

        sig_key = self._find_signal_key(dataset, self.node)
        voltage = np.asarray(dataset[sig_key].values)

        dt = np.mean(np.diff(time_values))
        fs = 1.0 / dt

        frequencies, magnitudes = _compute_fft(voltage, fs, self.window)

        # Find fundamental
        fund_idx, fund_freq = _find_fundamental(
            frequencies, magnitudes, self.fundamental_freq
        )
        fund_mag = magnitudes[fund_idx]

        # Find largest spur (excluding fundamental and DC)
        spur_mask = np.ones(len(magnitudes), dtype=bool)
        spur_mask[0] = False  # Exclude DC
        spur_mask[fund_idx] = False  # Exclude fundamental

        spur_magnitudes = magnitudes[spur_mask]
        if len(spur_magnitudes) == 0:
            return MeasurementResult(
                value=float("inf"),
                unit="dBc",
                metadata={"error": "No spurs found"},
            )

        max_spur_idx_relative = int(np.argmax(spur_magnitudes))
        max_spur_mag = spur_magnitudes[max_spur_idx_relative]

        # Find actual frequency of largest spur
        spur_indices = np.where(spur_mask)[0]
        max_spur_idx = spur_indices[max_spur_idx_relative]
        max_spur_freq = float(frequencies[max_spur_idx])

        if max_spur_mag == 0:
            sfdr_db = float("inf")
        else:
            sfdr_db = 20.0 * np.log10(fund_mag / max_spur_mag)

        return MeasurementResult(
            value=float(sfdr_db),
            unit="dBc",
            metadata={
                "fundamental_freq": fund_freq,
                "fundamental_magnitude": float(fund_mag),
                "max_spur_freq": max_spur_freq,
                "max_spur_magnitude": float(max_spur_mag),
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


@measurement("enob")
class ENOBMeasurement(BaseMeasurement):
    """Effective Number of Bits measurement.

    ENOB = (SINAD - 1.76) / 6.02

    Example:
        >>> enob = ENOBMeasurement(node="vout", signal_freq=1000)
        >>> result = enob.measure(dataset)
        >>> print(f"ENOB: {result.value:.2f} bits")
    """

    name = "enob"
    description = "Effective number of bits"
    required_analyses = ["tran"]

    def __init__(
        self,
        node: str,
        signal_freq: float | None = None,
        window: WindowType = "hann",
    ):
        """Initialize ENOB measurement.

        Args:
            node: Signal node name
            signal_freq: Signal frequency (None = auto-detect)
            window: Window function type
        """
        self.node = node
        self.signal_freq = signal_freq
        self.window = window

    def measure(self, dataset: xr.Dataset) -> MeasurementResult[float]:
        """Calculate ENOB via SINAD."""
        # Use SINAD measurement
        sinad = SINADMeasurement(self.node, self.signal_freq, self.window)
        sinad_result = sinad.measure(dataset)

        sinad_db = sinad_result.value
        enob = (sinad_db - 1.76) / 6.02

        return MeasurementResult(
            value=float(enob),
            unit="bits",
            metadata={
                "sinad_db": sinad_db,
                **sinad_result.metadata,
            },
        )


__all__ = [
    "THDMeasurement",
    "SNRMeasurement",
    "SINADMeasurement",
    "SFDRMeasurement",
    "ENOBMeasurement",
]
