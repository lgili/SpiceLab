"""Common circuit templates for quick prototyping.

This module provides pre-built circuit templates for common topologies:
- Filters: RC lowpass, RC highpass, RLC bandpass, Sallen-Key
- Amplifiers: Inverting, Non-inverting, Differential
- Power: Voltage divider

Usage:
    >>> from spicelab.templates import rc_lowpass, rc_highpass
    >>> lpf = rc_lowpass(fc=1000)  # 1 kHz cutoff
    >>> hpf = rc_highpass(fc=100)  # 100 Hz cutoff
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.circuit import Circuit
from ..core.components import Capacitor, Inductor, Resistor
from ..core.net import GND, Net

if TYPE_CHECKING:
    from ..core.components import Component

__all__ = [
    "rc_lowpass",
    "rc_highpass",
    "rlc_bandpass",
    "voltage_divider",
    "sallen_key_lowpass",
    "inverting_amplifier",
    "non_inverting_amplifier",
    "FilterResult",
]


@dataclass
class FilterResult:
    """Result from filter template creation.

    Contains the circuit and key component references for easy access.
    """

    circuit: Circuit
    components: dict[str, Component]
    cutoff_frequency: float
    q_factor: float | None = None

    def __repr__(self) -> str:
        return f"FilterResult(circuit={self.circuit.name!r}, fc={self.cutoff_frequency:.2f}Hz)"


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


# =============================================================================
# Filter Templates
# =============================================================================


def rc_highpass(
    fc: float,
    impedance: float = 10_000,
    name: str = "RC_Highpass",
) -> Circuit:
    """Create RC high-pass filter with specified cutoff frequency.

    The high-pass filter passes frequencies above fc and attenuates lower frequencies.
    The -3dB point occurs at fc = 1/(2π RC).

    Circuit topology:
        vin ---||---+--- vout
              C1   |
                   R1
                   |
                  GND

    Args:
        fc: Cutoff frequency in Hz (-3dB point)
        impedance: Characteristic impedance (default 10kΩ)
        name: Circuit name

    Returns:
        Circuit with C and R sized for fc = 1/(2π RC)

    Example:
        >>> circuit = rc_highpass(fc=100, impedance=10_000)
        >>> # Creates R=10kΩ, C≈159nF for fc=100Hz
    """
    R = impedance
    C = 1 / (2 * math.pi * fc * R)

    circuit = Circuit(name)

    vin = Net("vin")
    vout = Net("vout")

    C1 = Capacitor(ref="1", capacitance=C)
    R1 = Resistor(ref="1", resistance=R)

    circuit.add(C1, R1)

    # Capacitor in series, resistor to ground
    circuit.connect(C1.ports[0], vin)
    circuit.connect(C1.ports[1], vout)
    circuit.connect(R1.ports[0], vout)
    circuit.connect(R1.ports[1], GND)

    return circuit


def rlc_bandpass(
    fc: float,
    bandwidth: float,
    impedance: float = 1000,
    name: str = "RLC_Bandpass",
) -> FilterResult:
    """Create series RLC bandpass filter.

    The bandpass filter passes frequencies near fc and attenuates others.
    Q factor determines the selectivity: Q = fc / bandwidth.

    Circuit topology:
        vin ---[R]---[L]---+--- vout
                          |
                         [C]
                          |
                         GND

    Args:
        fc: Center frequency in Hz
        bandwidth: 3dB bandwidth in Hz
        impedance: Characteristic impedance (default 1kΩ)
        name: Circuit name

    Returns:
        FilterResult with circuit, components, fc, and Q factor

    Example:
        >>> result = rlc_bandpass(fc=1000, bandwidth=100)
        >>> # Creates bandpass with Q=10 centered at 1kHz
        >>> print(f"Q factor: {result.q_factor}")
    """
    Q = fc / bandwidth
    omega_0 = 2 * math.pi * fc

    # Design equations:
    # Q = (1/R) * sqrt(L/C)
    # omega_0 = 1/sqrt(LC)
    # Choose R, then calculate L and C

    R = impedance
    L = (Q * R) / omega_0
    C = 1 / (omega_0 * Q * R)

    circuit = Circuit(name)

    vin = Net("vin")
    vout = Net("vout")
    n1 = Net("n1")

    R1 = Resistor(ref="1", resistance=R)
    L1 = Inductor(ref="1", inductance=L)
    C1 = Capacitor(ref="1", capacitance=C)

    circuit.add(R1, L1, C1)

    circuit.connect(R1.ports[0], vin)
    circuit.connect(R1.ports[1], n1)
    circuit.connect(L1.ports[0], n1)
    circuit.connect(L1.ports[1], vout)
    circuit.connect(C1.ports[0], vout)
    circuit.connect(C1.ports[1], GND)

    return FilterResult(
        circuit=circuit,
        components={"R1": R1, "L1": L1, "C1": C1},
        cutoff_frequency=fc,
        q_factor=Q,
    )


def sallen_key_lowpass(
    fc: float,
    q: float = 0.707,
    impedance: float = 10_000,
    name: str = "Sallen_Key_LPF",
) -> FilterResult:
    """Create 2nd-order Sallen-Key lowpass filter.

    The Sallen-Key topology provides a 2nd-order response with -40dB/decade
    rolloff above the cutoff frequency. Q=0.707 gives Butterworth (maximally flat).

    Circuit topology (unity gain):
        vin ---[R1]---+---[R2]---+--- vout
                      |         |
                     [C1]     [C2]
                      |         |
                     GND     (to opamp -)

    Note: This creates the passive portion. Add an opamp buffer for the
    complete Sallen-Key filter.

    Args:
        fc: Cutoff frequency in Hz (-3dB point)
        q: Quality factor (0.707 for Butterworth, 1.0 for Bessel-like)
        impedance: Base impedance for R1=R2 (default 10kΩ)
        name: Circuit name

    Returns:
        FilterResult with circuit, components, fc, and Q factor

    Example:
        >>> result = sallen_key_lowpass(fc=1000, q=0.707)
        >>> # Creates Butterworth 2nd-order LPF at 1kHz
    """
    omega_0 = 2 * math.pi * fc

    # Unity-gain Sallen-Key design with R1=R2=R:
    # fc = 1/(2π * R * sqrt(C1*C2))
    # Q = sqrt(C1*C2) / (C1 + C2) * (1/R) -- simplified for equal R

    # For equal R: C2 = C1 / (4 * Q^2)
    # And: C1 = 2*Q / (omega_0 * R)

    R = impedance
    C1 = (2 * q) / (omega_0 * R)
    C2 = 1 / (4 * q * q * omega_0 * R)

    circuit = Circuit(name)

    vin = Net("vin")
    n1 = Net("n1")
    vout = Net("vout")

    R1 = Resistor(ref="1", resistance=R)
    R2 = Resistor(ref="2", resistance=R)
    C1_comp = Capacitor(ref="1", capacitance=C1)
    C2_comp = Capacitor(ref="2", capacitance=C2)

    circuit.add(R1, R2, C1_comp, C2_comp)

    circuit.connect(R1.ports[0], vin)
    circuit.connect(R1.ports[1], n1)
    circuit.connect(R2.ports[0], n1)
    circuit.connect(R2.ports[1], vout)
    circuit.connect(C1_comp.ports[0], n1)
    circuit.connect(C1_comp.ports[1], GND)
    circuit.connect(C2_comp.ports[0], vout)
    circuit.connect(C2_comp.ports[1], GND)

    return FilterResult(
        circuit=circuit,
        components={"R1": R1, "R2": R2, "C1": C1_comp, "C2": C2_comp},
        cutoff_frequency=fc,
        q_factor=q,
    )


# =============================================================================
# Amplifier Templates
# =============================================================================


def inverting_amplifier(
    gain: float,
    input_impedance: float = 10_000,
    name: str = "Inverting_Amp",
) -> Circuit:
    """Create inverting amplifier resistor network.

    The inverting amplifier has gain = -Rf/Rin.
    Add an opamp with the output connected to the feedback network.

    Circuit topology:
        vin ---[Rin]---+---[Rf]--- vout (opamp output)
                       |
                    (opamp -)

    Args:
        gain: Magnitude of voltage gain (positive number, actual gain is negative)
        input_impedance: Input resistance Rin (default 10kΩ)
        name: Circuit name

    Returns:
        Circuit with Rin and Rf for specified gain

    Example:
        >>> circuit = inverting_amplifier(gain=10, input_impedance=10_000)
        >>> # Creates Rin=10kΩ, Rf=100kΩ for gain of -10
    """
    if gain <= 0:
        raise ValueError(f"Gain must be positive, got {gain}")

    Rin = input_impedance
    Rf = gain * Rin

    circuit = Circuit(name)

    vin = Net("vin")
    vminus = Net("vminus")  # Opamp inverting input
    vout = Net("vout")

    R_in = Resistor(ref="in", resistance=Rin)
    R_f = Resistor(ref="f", resistance=Rf)

    circuit.add(R_in, R_f)

    circuit.connect(R_in.ports[0], vin)
    circuit.connect(R_in.ports[1], vminus)
    circuit.connect(R_f.ports[0], vminus)
    circuit.connect(R_f.ports[1], vout)

    return circuit


def non_inverting_amplifier(
    gain: float,
    feedback_resistance: float = 10_000,
    name: str = "NonInverting_Amp",
) -> Circuit:
    """Create non-inverting amplifier resistor network.

    The non-inverting amplifier has gain = 1 + Rf/R1.
    Add an opamp with vin to +input and the feedback network to -input.

    Circuit topology:
        vin --- (opamp +)
                            vout (opamp output)
                              |
               GND---[R1]---+-[Rf]
                            |
                         (opamp -)

    Args:
        gain: Voltage gain (must be >= 1)
        feedback_resistance: Feedback resistance Rf (default 10kΩ)
        name: Circuit name

    Returns:
        Circuit with R1 and Rf for specified gain

    Raises:
        ValueError: If gain < 1

    Example:
        >>> circuit = non_inverting_amplifier(gain=11, feedback_resistance=10_000)
        >>> # Creates Rf=10kΩ, R1=1kΩ for gain of 11
    """
    if gain < 1:
        raise ValueError(f"Non-inverting gain must be >= 1, got {gain}")

    Rf = feedback_resistance
    # gain = 1 + Rf/R1  =>  R1 = Rf / (gain - 1)
    if gain == 1:
        # Unity gain buffer: just wire through (no feedback resistor needed)
        R1 = float("inf")  # Open circuit
        Rf = 0  # Short circuit
    else:
        R1 = Rf / (gain - 1)

    circuit = Circuit(name)

    vminus = Net("vminus")  # Opamp inverting input
    vout = Net("vout")

    R_1 = Resistor(ref="1", resistance=R1 if R1 != float("inf") else 1e12)
    R_f = Resistor(ref="f", resistance=Rf if Rf != 0 else 0.001)

    circuit.add(R_1, R_f)

    circuit.connect(R_1.ports[0], GND)
    circuit.connect(R_1.ports[1], vminus)
    circuit.connect(R_f.ports[0], vminus)
    circuit.connect(R_f.ports[1], vout)

    return circuit
