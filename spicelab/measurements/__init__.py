"""Measurement & Analysis Library for SpiceLab.

This module provides a comprehensive set of measurements for extracting
circuit performance specifications from simulation results.

Features:
- MeasurementResult: Structured result with value, unit, pass/fail, and metadata
- MeasurementRegistry: Central registry with @measurement decorator
- 30+ ready-to-use measurements (AC, transient, spectrum, digital, power, analog)
- Custom measurement DSL for defining new measurements

Available Measurements:
    AC Analysis:
        - gain: DC or AC gain at specified frequency
        - bandwidth: -NdB bandwidth frequency
        - phase_margin: Phase margin at unity gain
        - gain_margin: Gain margin at -180° phase
        - psrr: Power supply rejection ratio
        - cmrr: Common-mode rejection ratio
        - input_impedance: Input impedance vs frequency

    Analog Measurements (NEW):
        - psrr_sweep: PSRR with frequency sweep
        - cmrr_sweep: CMRR with frequency sweep
        - noise_analysis: Spot noise and integrated noise
        - spot_noise: Noise at specific frequency
        - integrated_noise: Integrated RMS noise over bandwidth

    Transient Analysis:
        - slew_rate: Maximum rate of change (dV/dt)
        - settling_time: Time to settle within error band
        - rise_time: 10-90% rise time
        - fall_time: 90-10% fall time
        - overshoot: Peak overshoot percentage
        - propagation_delay: Input to output delay

    Spectrum Analysis:
        - thd: Total harmonic distortion
        - thd_n: Total harmonic distortion plus noise
        - snr: Signal-to-noise ratio
        - sinad: Signal-to-noise-and-distortion
        - sfdr: Spurious-free dynamic range
        - enob: Effective number of bits

    Digital Analysis:
        - eye_opening: Eye diagram height and width
        - jitter: Period and cycle-to-cycle jitter
        - duty_cycle: Duty cycle percentage

    Power Analysis:
        - ripple: Supply ripple voltage
        - pdn_impedance: PDN impedance vs frequency
        - load_transient: Load transient droop/recovery
        - voltage_droop: Maximum voltage droop from nominal
        - efficiency: Power conversion efficiency

Example:
    >>> from spicelab.measurements import MeasurementRegistry, measure
    >>> from spicelab.measurements.ac import GainMeasurement, BandwidthMeasurement
    >>>
    >>> # Get measurement by name
    >>> gain_cls = MeasurementRegistry.get("gain")
    >>> gain = gain_cls(output_node="vout", frequency=1000)
    >>> result = gain.measure(dataset)
    >>> print(f"Gain: {result.value} {result.unit}")
    >>>
    >>> # Use convenience function
    >>> result = measure(dataset, "bandwidth", output_node="vout")
    >>>
    >>> # List all available measurements
    >>> print(MeasurementRegistry.list_all())

    >>> # Analog measurements with convenience functions
    >>> from spicelab.measurements.analog import measure_psrr, measure_noise
    >>> psrr = measure_psrr(dataset, "vout", "vdd", frequency=1000)
    >>> noise = measure_noise(dataset, "onoise", freq_min=10, freq_max=100000)
"""

from __future__ import annotations

# Import all measurement modules to register them
from . import (
    ac,  # noqa: F401
    analog,  # noqa: F401
    digital,  # noqa: F401
    power,  # noqa: F401
    spectrum,  # noqa: F401
    transient,  # noqa: F401
)
from .analog import (
    CMRRResult,
    CMRRSweepMeasurement,
    IntegratedNoiseMeasurement,
    NoiseAnalysisMeasurement,
    NoiseResult,
    PSRRResult,
    PSRRSweepMeasurement,
    SpotNoiseMeasurement,
    measure_cmrr,
    measure_noise,
    measure_psrr,
)
from .base import BaseMeasurement, Measurement, MeasurementResult
from .registry import MeasurementRegistry, measure, measurement

__all__ = [
    # Core classes
    "BaseMeasurement",
    "Measurement",
    "MeasurementResult",
    "MeasurementRegistry",
    "measure",
    "measurement",
    # Analog measurement exports
    "PSRRResult",
    "CMRRResult",
    "NoiseResult",
    "PSRRSweepMeasurement",
    "CMRRSweepMeasurement",
    "NoiseAnalysisMeasurement",
    "SpotNoiseMeasurement",
    "IntegratedNoiseMeasurement",
    "measure_psrr",
    "measure_cmrr",
    "measure_noise",
    # Module re-exports for convenience
    "ac",
    "analog",
    "digital",
    "power",
    "spectrum",
    "transient",
]
