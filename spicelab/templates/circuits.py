"""Common circuit templates for quick prototyping.

Usage:
    >>> from spicelab.templates import rc_lowpass
    >>> circuit = rc_lowpass(fc=1000)  # 1 kHz cutoff
"""

from __future__ import annotations

import math

from ..core.circuit import Circuit
from ..core.components import Capacitor, Resistor
from ..core.net import GND, Net

__all__ = ["rc_lowpass", "voltage_divider"]


def rc_lowpass(
    fc: float,
    impedance: float = 10_000,
    name: str = "RC_Filter",
) -> Circuit:
    """Create RC low-pass filter with specified cutoff frequency.

    Args:
        fc: Cutoff frequency in Hz (-3dB point)
        impedance: Characteristic impedance (default 10kΩ)
        name: Circuit name

    Returns:
        Circuit with R and C sized for fc = 1/(2π RC)

    Example:
        >>> circuit = rc_lowpass(fc=1000, impedance=10_000)
        >>> # Creates R=10kΩ, C≈15.9nF for fc=1kHz
    """
    # Calculate C from fc = 1/(2π RC)
    R = impedance
    C = 1 / (2 * math.pi * fc * R)

    circuit = Circuit(name)

    # Create nodes
    vin = Net("vin")
    vout = Net("vout")

    # Create components
    R1 = Resistor(ref="1", resistance=R)
    C1 = Capacitor(ref="1", capacitance=C)

    # Add components to circuit
    circuit.add(R1, C1)

    # Connect components
    circuit.connect(R1.ports[0], vin)
    circuit.connect(R1.ports[1], vout)
    circuit.connect(C1.ports[0], vout)
    circuit.connect(C1.ports[1], GND)

    return circuit


def voltage_divider(
    ratio: float,
    total_resistance: float = 10_000,
    name: str = "Voltage_Divider",
) -> Circuit:
    """Create resistive voltage divider.

    Args:
        ratio: Output voltage ratio (Vout/Vin), between 0 and 1
        total_resistance: Total series resistance (default 10kΩ)
        name: Circuit name

    Returns:
        Circuit with R1 and R2 for Vout = Vin * ratio

    Raises:
        ValueError: If ratio not in (0, 1)

    Example:
        >>> circuit = voltage_divider(ratio=0.5, total_resistance=10_000)
        >>> # Creates R1=5kΩ, R2=5kΩ for 50% division
    """
    if not 0 < ratio < 1:
        raise ValueError(f"Voltage divider ratio must be in (0, 1), got {ratio}")

    # R2 / (R1 + R2) = ratio
    # R1 + R2 = total_resistance
    R2 = ratio * total_resistance
    R1 = total_resistance - R2

    circuit = Circuit(name)

    # Create nodes
    vin = Net("vin")
    vout = Net("vout")

    # Create components
    R1_comp = Resistor(ref="1", resistance=R1)
    R2_comp = Resistor(ref="2", resistance=R2)

    # Add components to circuit
    circuit.add(R1_comp, R2_comp)

    # Connect components
    circuit.connect(R1_comp.ports[0], vin)
    circuit.connect(R1_comp.ports[1], vout)
    circuit.connect(R2_comp.ports[0], vout)
    circuit.connect(R2_comp.ports[1], GND)

    return circuit
