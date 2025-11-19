"""Circuit templates for common topologies.

Provides pre-built circuit templates to accelerate design workflows.
"""

from __future__ import annotations

from .circuits import rc_lowpass, voltage_divider

__all__ = [
    "rc_lowpass",
    "voltage_divider",
]
