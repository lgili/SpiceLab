"""Circuit templates for common topologies.

Provides pre-built circuit templates to accelerate design workflows.

Available templates:

Filters (1st order):
- rc_lowpass: Simple RC lowpass filter
- rc_highpass: Simple RC highpass filter

Filters (2nd order):
- sallen_key_lowpass: Sallen-Key lowpass with configurable Q
- rlc_bandpass: Series RLC bandpass filter

Filters (Higher order - Butterworth, Chebyshev, Bessel):
- butterworth_lowpass: Maximally flat passband (orders 1-6)
- chebyshev_lowpass: Steep rolloff with passband ripple (orders 1-6)
- bessel_lowpass: Maximally flat group delay (orders 1-6)

Amplifiers:
- voltage_follower: Unity gain buffer
- inverting_amplifier: Inverting opamp configuration
- non_inverting_amplifier: Non-inverting opamp configuration
- differential_amplifier: Difference amplifier
- summing_amplifier: Inverting summer (multiple inputs)

Bias circuits:
- current_mirror: Simple current mirror

Frequency compensation:
- dominant_pole_compensation: Simple RC pole for gain rolloff
- lead_compensation: Phase boost network (zero + pole)
- lead_lag_compensation: Combined lead-lag for phase margin + DC gain
- miller_compensation: Capacitor across gain stage (pole splitting)

Power:
- voltage_divider: Resistive divider

Example:
    >>> from spicelab.templates import butterworth_lowpass, chebyshev_lowpass
    >>> butter = butterworth_lowpass(fc=1000, order=4)  # 4th-order Butterworth
    >>> cheby = chebyshev_lowpass(fc=1000, order=4)  # 4th-order Chebyshev

    >>> from spicelab.templates import voltage_follower, differential_amplifier
    >>> buffer = voltage_follower()  # Unity gain buffer
    >>> diff_amp = differential_amplifier(gain=10)  # 10x differential gain

    >>> from spicelab.templates import lead_compensation
    >>> comp = lead_compensation(zero_frequency=1000, pole_frequency=10000)
    >>> print(f"Phase boost: {comp.phase_margin_boost:.1f}°")

ADC/DAC building blocks:
- r2r_dac_ladder: R-2R resistor ladder for digital-to-analog conversion
- sample_and_hold: Capacitor + switch for sampling analog signals
- comparator_bank: Reference ladder for flash ADC

    >>> from spicelab.templates import r2r_dac_ladder
    >>> dac = r2r_dac_ladder(bits=8)  # 8-bit R-2R DAC
"""

from __future__ import annotations

from .circuits import (
    AmplifierResult,
    BiasResult,
    CompensationResult,
    ConverterResult,
    FilterResult,
    bessel_lowpass,
    butterworth_lowpass,
    chebyshev_lowpass,
    comparator_bank,
    current_mirror,
    differential_amplifier,
    dominant_pole_compensation,
    inverting_amplifier,
    lead_compensation,
    lead_lag_compensation,
    miller_compensation,
    non_inverting_amplifier,
    r2r_dac_ladder,
    rc_highpass,
    rc_lowpass,
    rlc_bandpass,
    sallen_key_lowpass,
    sample_and_hold,
    summing_amplifier,
    voltage_divider,
    voltage_follower,
)

__all__ = [
    # 1st order filters
    "rc_lowpass",
    "rc_highpass",
    # 2nd order filters
    "sallen_key_lowpass",
    "rlc_bandpass",
    # Higher order filters
    "butterworth_lowpass",
    "chebyshev_lowpass",
    "bessel_lowpass",
    # Amplifiers
    "inverting_amplifier",
    "non_inverting_amplifier",
    "voltage_follower",
    "differential_amplifier",
    "summing_amplifier",
    # Bias
    "current_mirror",
    # Compensation
    "dominant_pole_compensation",
    "lead_compensation",
    "lead_lag_compensation",
    "miller_compensation",
    # ADC/DAC
    "r2r_dac_ladder",
    "sample_and_hold",
    "comparator_bank",
    # Power
    "voltage_divider",
    # Result types
    "FilterResult",
    "AmplifierResult",
    "BiasResult",
    "CompensationResult",
    "ConverterResult",
]
