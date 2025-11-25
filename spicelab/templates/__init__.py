"""Circuit templates for common topologies.

Provides pre-built circuit templates to accelerate design workflows.

Available templates:
- Filters: rc_lowpass, rc_highpass, rlc_bandpass, sallen_key_lowpass
- Amplifiers: inverting_amplifier, non_inverting_amplifier
- Power: voltage_divider

Example:
    >>> from spicelab.templates import rc_lowpass, sallen_key_lowpass
    >>> lpf = rc_lowpass(fc=1000)  # 1kHz 1st-order lowpass
    >>> sk = sallen_key_lowpass(fc=1000, q=0.707)  # 1kHz Butterworth
"""

from __future__ import annotations

from .circuits import (
    FilterResult,
    inverting_amplifier,
    non_inverting_amplifier,
    rc_highpass,
    rc_lowpass,
    rlc_bandpass,
    sallen_key_lowpass,
    voltage_divider,
)

__all__ = [
    "rc_lowpass",
    "rc_highpass",
    "rlc_bandpass",
    "sallen_key_lowpass",
    "voltage_divider",
    "inverting_amplifier",
    "non_inverting_amplifier",
    "FilterResult",
]
