"""Measurement & Analysis Library for SpiceLab.

This module provides a comprehensive set of measurements for extracting
circuit performance specifications from simulation results.

Features:
- MeasurementResult: Structured result with value, unit, pass/fail, and metadata
- MeasurementRegistry: Central registry with @measurement decorator
- 30+ ready-to-use measurements (AC, transient, spectrum, digital, power)
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
"""

from __future__ import annotations

# Import all measurement modules to register them
from . import (
    ac,  # noqa: F401
    digital,  # noqa: F401
    power,  # noqa: F401
    spectrum,  # noqa: F401
    transient,  # noqa: F401
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
    # Module re-exports for convenience
    "ac",
    "digital",
    "power",
    "spectrum",
    "transient",
]
