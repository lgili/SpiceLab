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
from ..core.components import Capacitor, Inductor, OpAmpIdeal, Resistor, Vdc
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
    "voltage_follower",
    "differential_amplifier",
    "summing_amplifier",
    "FilterResult",
    "AmplifierResult",
    # Advanced filter patterns
    "butterworth_lowpass",
    "chebyshev_lowpass",
    "bessel_lowpass",
    # Bias patterns
    "current_mirror",
    "BiasResult",
    # Compensation patterns
    "dominant_pole_compensation",
    "lead_compensation",
    "lead_lag_compensation",
    "miller_compensation",
    "CompensationResult",
    # ADC/DAC building blocks
    "r2r_dac_ladder",
    "sample_and_hold",
    "comparator_bank",
    "ConverterResult",
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


@dataclass
class AmplifierResult:
    """Result from amplifier template creation.

    Contains the circuit, component references, and key specifications.
    """

    circuit: Circuit
    components: dict[str, Component]
    gain: float
    input_net: Net
    output_net: Net

    def __repr__(self) -> str:
        return f"AmplifierResult(circuit={self.circuit.name!r}, gain={self.gain:.2f})"


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


# =============================================================================
# Complete Amplifier Templates (with OpAmp)
# =============================================================================


def voltage_follower(
    name: str = "Voltage_Follower",
    vcc: float = 15.0,
    vee: float = -15.0,
) -> AmplifierResult:
    """Create unity-gain voltage follower (buffer) with ideal opamp.

    The voltage follower provides unity gain with high input impedance
    and low output impedance. Useful for impedance matching.

    Circuit topology:
        vin -----(+)\\
                    >---- vout
             +--(-)/      |
             |            |
             +------------+

    Args:
        name: Circuit name
        vcc: Positive supply voltage (default +15V)
        vee: Negative supply voltage (default -15V)

    Returns:
        AmplifierResult with circuit, components, gain=1, input/output nets

    Example:
        >>> result = voltage_follower()
        >>> circuit = result.circuit
        >>> print(f"Gain: {result.gain}")  # 1.0
    """
    circuit = Circuit(name)

    vin = Net("vin")
    vout = Net("vout")
    vcc_net = Net("vcc")
    vee_net = Net("vee")

    # Power supplies
    V_cc = Vdc(ref="CC", value=str(vcc))
    V_ee = Vdc(ref="EE", value=str(abs(vee)))

    # Ideal opamp
    opamp = OpAmpIdeal(ref="1")

    circuit.add(V_cc, V_ee, opamp)

    # Power supply connections
    circuit.connect(V_cc.ports[0], vcc_net)
    circuit.connect(V_cc.ports[1], GND)
    circuit.connect(V_ee.ports[0], GND)
    circuit.connect(V_ee.ports[1], vee_net)

    # Opamp connections: inp(+), inn(-), out
    circuit.connect(opamp.ports[0], vin)  # + input
    circuit.connect(opamp.ports[1], vout)  # - input (feedback)
    circuit.connect(opamp.ports[2], vout)  # output

    return AmplifierResult(
        circuit=circuit,
        components={"opamp": opamp, "Vcc": V_cc, "Vee": V_ee},
        gain=1.0,
        input_net=vin,
        output_net=vout,
    )


def differential_amplifier(
    gain: float = 1.0,
    input_impedance: float = 10_000,
    name: str = "Differential_Amp",
    vcc: float = 15.0,
    vee: float = -15.0,
) -> AmplifierResult:
    """Create differential amplifier with ideal opamp.

    The differential amplifier outputs the difference between two inputs
    multiplied by the gain: Vout = gain * (Vp - Vn).

    Circuit topology:
        vinp ---[R1]---+---(+)\\
                       |      >---- vout
        vinn ---[R2]---+--(-)/      |
                       |      [Rf]--+
                      [Rg]
                       |
                      GND

    For unity gain: R1=R2=Rg=Rf
    For gain G: Rf/R1 = Rg/R2 = G

    Args:
        gain: Differential voltage gain (default 1.0)
        input_impedance: Input resistance R1=R2 (default 10kΩ)
        name: Circuit name
        vcc: Positive supply voltage (default +15V)
        vee: Negative supply voltage (default -15V)

    Returns:
        AmplifierResult with circuit, components, and specifications

    Example:
        >>> result = differential_amplifier(gain=10)
        >>> # Vout = 10 * (Vp - Vn)
    """
    R1 = input_impedance
    R2 = input_impedance
    Rf = gain * R1
    Rg = gain * R2

    circuit = Circuit(name)

    vinp = Net("vinp")  # Positive input
    vinn = Net("vinn")  # Negative input
    vout = Net("vout")
    vminus = Net("vminus")
    vplus = Net("vplus")
    vcc_net = Net("vcc")
    vee_net = Net("vee")

    # Power supplies
    V_cc = Vdc(ref="CC", value=str(vcc))
    V_ee = Vdc(ref="EE", value=str(abs(vee)))

    # Resistors
    R_1 = Resistor(ref="1", resistance=R1)
    R_2 = Resistor(ref="2", resistance=R2)
    R_f = Resistor(ref="f", resistance=Rf)
    R_g = Resistor(ref="g", resistance=Rg)

    # Opamp
    opamp = OpAmpIdeal(ref="1")

    circuit.add(V_cc, V_ee, R_1, R_2, R_f, R_g, opamp)

    # Power supply connections
    circuit.connect(V_cc.ports[0], vcc_net)
    circuit.connect(V_cc.ports[1], GND)
    circuit.connect(V_ee.ports[0], GND)
    circuit.connect(V_ee.ports[1], vee_net)

    # Input network
    circuit.connect(R_1.ports[0], vinn)
    circuit.connect(R_1.ports[1], vminus)
    circuit.connect(R_2.ports[0], vinp)
    circuit.connect(R_2.ports[1], vplus)

    # Feedback network
    circuit.connect(R_f.ports[0], vminus)
    circuit.connect(R_f.ports[1], vout)
    circuit.connect(R_g.ports[0], vplus)
    circuit.connect(R_g.ports[1], GND)

    # Opamp connections
    circuit.connect(opamp.ports[0], vplus)  # + input
    circuit.connect(opamp.ports[1], vminus)  # - input
    circuit.connect(opamp.ports[2], vout)  # output

    return AmplifierResult(
        circuit=circuit,
        components={
            "opamp": opamp,
            "R1": R_1,
            "R2": R_2,
            "Rf": R_f,
            "Rg": R_g,
            "Vcc": V_cc,
            "Vee": V_ee,
        },
        gain=gain,
        input_net=vinp,  # Primary input
        output_net=vout,
    )


def summing_amplifier(
    num_inputs: int = 2,
    gain: float = 1.0,
    input_impedance: float = 10_000,
    name: str = "Summing_Amp",
    vcc: float = 15.0,
    vee: float = -15.0,
) -> AmplifierResult:
    """Create inverting summing amplifier with ideal opamp.

    The summing amplifier outputs the inverted sum of inputs:
    Vout = -Rf/Rin * (V1 + V2 + ... + Vn)

    For equal weighting with gain G: Rf = G * Rin

    Circuit topology:
        vin1 ---[R1]---+
        vin2 ---[R2]---+--(-)/
        ...            |    >---- vout
        vinN ---[Rn]---+   |
                       +--[Rf]
                  GND--(+)/

    Args:
        num_inputs: Number of input channels (default 2)
        gain: Gain per input channel (default 1.0)
        input_impedance: Input resistance for each channel (default 10kΩ)
        name: Circuit name
        vcc: Positive supply voltage (default +15V)
        vee: Negative supply voltage (default -15V)

    Returns:
        AmplifierResult with circuit, components, and specifications

    Example:
        >>> result = summing_amplifier(num_inputs=3, gain=2)
        >>> # Vout = -2 * (V1 + V2 + V3)
    """
    if num_inputs < 1:
        raise ValueError(f"num_inputs must be >= 1, got {num_inputs}")

    Rin = input_impedance
    Rf = gain * Rin

    circuit = Circuit(name)

    vminus = Net("vminus")
    vout = Net("vout")
    vcc_net = Net("vcc")
    vee_net = Net("vee")

    # Power supplies
    V_cc = Vdc(ref="CC", value=str(vcc))
    V_ee = Vdc(ref="EE", value=str(abs(vee)))

    # Feedback resistor
    R_f = Resistor(ref="f", resistance=Rf)

    # Opamp
    opamp = OpAmpIdeal(ref="1")

    circuit.add(V_cc, V_ee, R_f, opamp)

    components: dict[str, Component] = {
        "opamp": opamp,
        "Rf": R_f,
        "Vcc": V_cc,
        "Vee": V_ee,
    }

    # Input resistors
    input_nets: list[Net] = []
    for i in range(num_inputs):
        vin_i = Net(f"vin{i + 1}")
        input_nets.append(vin_i)
        R_i = Resistor(ref=f"in{i + 1}", resistance=Rin)
        circuit.add(R_i)
        circuit.connect(R_i.ports[0], vin_i)
        circuit.connect(R_i.ports[1], vminus)
        components[f"Rin{i + 1}"] = R_i

    # Power supply connections
    circuit.connect(V_cc.ports[0], vcc_net)
    circuit.connect(V_cc.ports[1], GND)
    circuit.connect(V_ee.ports[0], GND)
    circuit.connect(V_ee.ports[1], vee_net)

    # Feedback
    circuit.connect(R_f.ports[0], vminus)
    circuit.connect(R_f.ports[1], vout)

    # Opamp connections
    circuit.connect(opamp.ports[0], GND)  # + input to ground
    circuit.connect(opamp.ports[1], vminus)  # - input
    circuit.connect(opamp.ports[2], vout)  # output

    return AmplifierResult(
        circuit=circuit,
        components=components,
        gain=-gain,  # Inverting
        input_net=input_nets[0],  # First input
        output_net=vout,
    )


# =============================================================================
# Advanced Filter Patterns (Butterworth, Chebyshev, Bessel)
# =============================================================================


# Pre-computed normalized Butterworth pole Q factors for orders 1-6
# Q = 1 / (2 * cos(theta)) where theta = pi*(2k-1)/(2n) for k-th pole pair
_BUTTERWORTH_Q: dict[int, list[float]] = {
    1: [],  # First order has no complex poles
    2: [0.7071],  # 1/sqrt(2)
    3: [1.0],  # Single complex pair
    4: [0.5412, 1.3065],
    5: [0.6180, 1.6180],
    6: [0.5176, 0.7071, 1.9319],
}

# Pre-computed normalized Chebyshev Type I pole Q factors (0.5 dB ripple)
_CHEBYSHEV_Q_05DB: dict[int, list[float]] = {
    1: [],
    2: [0.8637],
    3: [1.7062],
    4: [0.7846, 3.5590],
    5: [0.9565, 6.5131],
    6: [0.7609, 1.5963, 11.691],
}

# Pre-computed normalized Bessel pole Q factors for orders 1-6
_BESSEL_Q: dict[int, list[float]] = {
    1: [],
    2: [0.5773],
    3: [0.6910],
    4: [0.5219, 0.8055],
    5: [0.5635, 0.9165],
    6: [0.5103, 0.6112, 1.0234],
}


@dataclass
class BiasResult:
    """Result from bias circuit template creation.

    Contains the circuit and key specifications for biasing.
    """

    circuit: Circuit
    components: dict[str, Component]
    output_current: float
    mirror_ratio: float = 1.0

    def __repr__(self) -> str:
        return f"BiasResult(circuit={self.circuit.name!r}, Iout={self.output_current:.3g}A)"


@dataclass
class CompensationResult:
    """Result from compensation network template creation.

    Contains the circuit and key specifications for frequency compensation.
    """

    circuit: Circuit
    components: dict[str, Component]
    pole_frequency: float | None = None
    zero_frequency: float | None = None
    phase_margin_boost: float | None = None  # Degrees of phase boost at crossover

    def __repr__(self) -> str:
        parts = [f"circuit={self.circuit.name!r}"]
        if self.pole_frequency:
            parts.append(f"fp={self.pole_frequency:.2g}Hz")
        if self.zero_frequency:
            parts.append(f"fz={self.zero_frequency:.2g}Hz")
        return f"CompensationResult({', '.join(parts)})"


@dataclass
class ConverterResult:
    """Result from ADC/DAC building block template creation.

    Contains the circuit and key specifications for data converters.
    """

    circuit: Circuit
    components: dict[str, Component]
    resolution_bits: int
    input_nets: list[Net]
    output_net: Net

    def __repr__(self) -> str:
        return f"ConverterResult(circuit={self.circuit.name!r}, bits={self.resolution_bits})"


def butterworth_lowpass(
    fc: float,
    order: int = 2,
    impedance: float = 10_000,
    name: str = "Butterworth_LPF",
) -> FilterResult:
    """Create Butterworth lowpass filter using cascaded Sallen-Key stages.

    Butterworth filters have maximally flat passband response. Each 2nd-order
    section uses the Sallen-Key topology with Q values calculated for
    Butterworth response.

    Supported orders: 1-6. Odd orders include a first-order RC section.

    Args:
        fc: Cutoff frequency in Hz (-3dB point)
        order: Filter order (1-6, default 2)
        impedance: Base impedance for resistors (default 10kΩ)
        name: Circuit name

    Returns:
        FilterResult with cascaded filter circuit

    Raises:
        ValueError: If order not in 1-6

    Example:
        >>> result = butterworth_lowpass(fc=1000, order=4)
        >>> # Creates 4th-order Butterworth with -80dB/decade rolloff
        >>> print(f"Components: {len(result.components)}")
    """
    if order < 1 or order > 6:
        raise ValueError(f"Order must be 1-6, got {order}")

    q_values = _BUTTERWORTH_Q[order]
    return _build_cascaded_filter(fc, order, q_values, impedance, name)


def chebyshev_lowpass(
    fc: float,
    order: int = 2,
    ripple_db: float = 0.5,
    impedance: float = 10_000,
    name: str = "Chebyshev_LPF",
) -> FilterResult:
    """Create Chebyshev Type I lowpass filter using cascaded Sallen-Key stages.

    Chebyshev filters have steeper rolloff than Butterworth but with passband
    ripple. Lower ripple values give response closer to Butterworth.

    Note: Currently only supports 0.5dB ripple. Other values are approximated.

    Args:
        fc: Cutoff frequency in Hz (edge of passband ripple)
        order: Filter order (1-6, default 2)
        ripple_db: Passband ripple in dB (default 0.5, only 0.5 fully supported)
        impedance: Base impedance for resistors (default 10kΩ)
        name: Circuit name

    Returns:
        FilterResult with cascaded filter circuit

    Raises:
        ValueError: If order not in 1-6

    Example:
        >>> result = chebyshev_lowpass(fc=1000, order=4)
        >>> # Creates 4th-order Chebyshev with 0.5dB ripple
    """
    if order < 1 or order > 6:
        raise ValueError(f"Order must be 1-6, got {order}")

    # Use 0.5dB ripple values (most common)
    # Scale Q slightly for different ripple values
    q_values = _CHEBYSHEV_Q_05DB[order].copy()
    if ripple_db != 0.5 and q_values:
        # Approximate scaling: higher ripple = higher Q
        scale = math.sqrt(ripple_db / 0.5)
        q_values = [q * scale for q in q_values]

    return _build_cascaded_filter(fc, order, q_values, impedance, name)


def bessel_lowpass(
    fc: float,
    order: int = 2,
    impedance: float = 10_000,
    name: str = "Bessel_LPF",
) -> FilterResult:
    """Create Bessel lowpass filter using cascaded Sallen-Key stages.

    Bessel filters have maximally flat group delay, making them ideal for
    pulse and transient applications where waveform shape must be preserved.
    The frequency response rolls off more gradually than Butterworth.

    Supported orders: 1-6.

    Args:
        fc: Cutoff frequency in Hz (-3dB point)
        order: Filter order (1-6, default 2)
        impedance: Base impedance for resistors (default 10kΩ)
        name: Circuit name

    Returns:
        FilterResult with cascaded filter circuit

    Raises:
        ValueError: If order not in 1-6

    Example:
        >>> result = bessel_lowpass(fc=1000, order=4)
        >>> # Creates 4th-order Bessel with linear phase response
    """
    if order < 1 or order > 6:
        raise ValueError(f"Order must be 1-6, got {order}")

    q_values = _BESSEL_Q[order]
    return _build_cascaded_filter(fc, order, q_values, impedance, name)


def _build_cascaded_filter(
    fc: float,
    order: int,
    q_values: list[float],
    impedance: float,
    name: str,
) -> FilterResult:
    """Build cascaded Sallen-Key filter with given Q values.

    Internal helper for Butterworth, Chebyshev, and Bessel filters.
    """
    circuit = Circuit(name)
    components: dict[str, Component] = {}

    omega_0 = 2 * math.pi * fc
    R = impedance

    # Track current input/output nets for cascading
    current_input = Net("vin")

    stage_num = 0

    # Handle first-order section for odd orders
    if order % 2 == 1:
        stage_num += 1
        # Simple RC lowpass for first-order section
        C_val = 1 / (omega_0 * R)

        R_stage = Resistor(ref=f"s{stage_num}_R", resistance=R)
        C_stage = Capacitor(ref=f"s{stage_num}_C", capacitance=C_val)

        circuit.add(R_stage, C_stage)
        components[f"R_s{stage_num}"] = R_stage
        components[f"C_s{stage_num}"] = C_stage

        if len(q_values) > 0:
            stage_output = Net(f"n{stage_num}")
        else:
            stage_output = Net("vout")

        circuit.connect(R_stage.ports[0], current_input)
        circuit.connect(R_stage.ports[1], stage_output)
        circuit.connect(C_stage.ports[0], stage_output)
        circuit.connect(C_stage.ports[1], GND)

        current_input = stage_output

    # Add 2nd-order Sallen-Key sections
    for i, q in enumerate(q_values):
        stage_num += 1

        # Sallen-Key design for equal R: C1 = 2*Q/(omega*R), C2 = 1/(4*Q*omega*R)
        C1_val = (2 * q) / (omega_0 * R)
        C2_val = 1 / (4 * q * omega_0 * R)

        R1 = Resistor(ref=f"s{stage_num}_R1", resistance=R)
        R2 = Resistor(ref=f"s{stage_num}_R2", resistance=R)
        C1 = Capacitor(ref=f"s{stage_num}_C1", capacitance=C1_val)
        C2 = Capacitor(ref=f"s{stage_num}_C2", capacitance=C2_val)

        circuit.add(R1, R2, C1, C2)
        components[f"R1_s{stage_num}"] = R1
        components[f"R2_s{stage_num}"] = R2
        components[f"C1_s{stage_num}"] = C1
        components[f"C2_s{stage_num}"] = C2

        # Internal node between R1 and R2
        n_mid = Net(f"n{stage_num}_mid")

        # Output node
        if i == len(q_values) - 1:
            stage_output = Net("vout")
        else:
            stage_output = Net(f"n{stage_num}")

        # Connect Sallen-Key topology
        circuit.connect(R1.ports[0], current_input)
        circuit.connect(R1.ports[1], n_mid)
        circuit.connect(R2.ports[0], n_mid)
        circuit.connect(R2.ports[1], stage_output)
        circuit.connect(C1.ports[0], n_mid)
        circuit.connect(C1.ports[1], GND)
        circuit.connect(C2.ports[0], stage_output)
        circuit.connect(C2.ports[1], GND)

        current_input = stage_output

    return FilterResult(
        circuit=circuit,
        components=components,
        cutoff_frequency=fc,
        q_factor=q_values[-1] if q_values else None,
    )


# =============================================================================
# Bias Circuit Patterns
# =============================================================================


def current_mirror(
    reference_current: float,
    mirror_ratio: float = 1.0,
    vcc: float = 5.0,
    name: str = "Current_Mirror",
) -> BiasResult:
    """Create simple NPN current mirror.

    A current mirror copies a reference current to one or more outputs.
    The mirror ratio determines the output current: Iout = ratio * Iref.

    Circuit topology:
        VCC ----+--------+
                |        |
               [Rref]   (output)
                |        |
        Iref -->|   +--->| Iout
                Q1  |    Q2
                |---+    |
               GND      GND

    Note: Uses ideal behavioral current source for simplicity.
    For real transistor implementation, add appropriate models.

    Args:
        reference_current: Reference current in Amps
        mirror_ratio: Output/Reference current ratio (default 1.0)
        vcc: Supply voltage (default 5V)
        name: Circuit name

    Returns:
        BiasResult with circuit, components, and output current

    Example:
        >>> result = current_mirror(reference_current=1e-3, mirror_ratio=2)
        >>> # Creates mirror with Iref=1mA, Iout=2mA
    """
    from ..core.components import Idc

    circuit = Circuit(name)

    vcc_net = Net("vcc")
    iout_net = Net("iout")

    # Supply
    V_cc = Vdc(ref="CC", value=str(vcc))

    # Reference current source (behavioral)
    I_ref = Idc(ref="ref", value=str(reference_current))

    # Output current source (mirrored)
    I_out = Idc(ref="out", value=str(reference_current * mirror_ratio))

    circuit.add(V_cc, I_ref, I_out)

    # Supply connections
    circuit.connect(V_cc.ports[0], vcc_net)
    circuit.connect(V_cc.ports[1], GND)

    # Reference current (flows into ground)
    circuit.connect(I_ref.ports[0], vcc_net)
    circuit.connect(I_ref.ports[1], GND)

    # Output current
    circuit.connect(I_out.ports[0], iout_net)
    circuit.connect(I_out.ports[1], GND)

    return BiasResult(
        circuit=circuit,
        components={"Vcc": V_cc, "Iref": I_ref, "Iout": I_out},
        output_current=reference_current * mirror_ratio,
        mirror_ratio=mirror_ratio,
    )


# =============================================================================
# Frequency Compensation Patterns
# =============================================================================


def dominant_pole_compensation(
    pole_frequency: float,
    impedance: float = 10_000,
    name: str = "Dominant_Pole",
) -> CompensationResult:
    """Create dominant pole compensation network.

    A simple RC network that adds a low-frequency pole to roll off gain
    before the system's higher-frequency poles can cause instability.
    Used as the simplest form of frequency compensation.

    Circuit topology:
        in ---[R]---+--- out
                    |
                   [C]
                    |
                   GND

    Pole frequency: fp = 1/(2π RC)

    Args:
        pole_frequency: Desired pole frequency in Hz
        impedance: Resistor value (default 10kΩ)
        name: Circuit name

    Returns:
        CompensationResult with circuit and pole frequency

    Example:
        >>> result = dominant_pole_compensation(pole_frequency=100)
        >>> # Creates RC network with pole at 100Hz
    """
    R = impedance
    C = 1 / (2 * math.pi * pole_frequency * R)

    circuit = Circuit(name)

    vin = Net("in")
    vout = Net("out")

    R_comp = Resistor(ref="comp", resistance=R)
    C_comp = Capacitor(ref="comp", capacitance=C)

    circuit.add(R_comp, C_comp)

    circuit.connect(R_comp.ports[0], vin)
    circuit.connect(R_comp.ports[1], vout)
    circuit.connect(C_comp.ports[0], vout)
    circuit.connect(C_comp.ports[1], GND)

    return CompensationResult(
        circuit=circuit,
        components={"R_comp": R_comp, "C_comp": C_comp},
        pole_frequency=pole_frequency,
    )


def lead_compensation(
    zero_frequency: float,
    pole_frequency: float,
    impedance: float = 10_000,
    name: str = "Lead_Comp",
) -> CompensationResult:
    """Create lead compensation network for phase boost.

    Lead compensation adds a zero before a pole to provide phase lead
    (positive phase shift) near the crossover frequency, improving
    phase margin. The zero-to-pole ratio determines the maximum phase boost.

    Circuit topology:
        in ---[R1]---+---[C]---+--- out
                     |         |
                    [R2]-------+

    Transfer function: H(s) = (1 + s/wz) / (1 + s/wp)
    where wz = 1/(R1*C), wp = 1/((R1||R2)*C)

    Maximum phase boost: φmax = arcsin((wp-wz)/(wp+wz))
    at geometric mean frequency: f = sqrt(fz * fp)

    Args:
        zero_frequency: Zero frequency in Hz (should be < pole_frequency)
        pole_frequency: Pole frequency in Hz
        impedance: Base impedance R1 (default 10kΩ)
        name: Circuit name

    Returns:
        CompensationResult with circuit, frequencies, and phase boost

    Raises:
        ValueError: If zero_frequency >= pole_frequency

    Example:
        >>> result = lead_compensation(zero_frequency=1000, pole_frequency=10000)
        >>> # Creates lead network with ~55° max phase boost
    """
    if zero_frequency >= pole_frequency:
        raise ValueError(
            f"Zero frequency ({zero_frequency}Hz) must be less than "
            f"pole frequency ({pole_frequency}Hz) for lead compensation"
        )

    # Design equations:
    # wz = 1/(R1*C)  =>  C = 1/(wz*R1)
    # wp = 1/((R1||R2)*C) = (R1+R2)/(R1*R2*C)
    # Ratio: wp/wz = (R1+R2)/R2 = 1 + R1/R2
    # => R2 = R1 / (wp/wz - 1)

    wz = 2 * math.pi * zero_frequency
    wp = 2 * math.pi * pole_frequency

    R1 = impedance
    C = 1 / (wz * R1)
    R2 = R1 / (wp / wz - 1)

    # Maximum phase boost
    phase_boost = math.degrees(math.asin((wp - wz) / (wp + wz)))

    circuit = Circuit(name)

    vin = Net("in")
    vout = Net("out")
    n1 = Net("n1")

    R1_comp = Resistor(ref="1", resistance=R1)
    R2_comp = Resistor(ref="2", resistance=R2)
    C_comp = Capacitor(ref="comp", capacitance=C)

    circuit.add(R1_comp, R2_comp, C_comp)

    circuit.connect(R1_comp.ports[0], vin)
    circuit.connect(R1_comp.ports[1], n1)
    circuit.connect(R2_comp.ports[0], n1)
    circuit.connect(R2_comp.ports[1], vout)
    circuit.connect(C_comp.ports[0], n1)
    circuit.connect(C_comp.ports[1], vout)

    return CompensationResult(
        circuit=circuit,
        components={"R1": R1_comp, "R2": R2_comp, "C": C_comp},
        pole_frequency=pole_frequency,
        zero_frequency=zero_frequency,
        phase_margin_boost=phase_boost,
    )


def lead_lag_compensation(
    lead_zero_freq: float,
    lead_pole_freq: float,
    lag_pole_freq: float,
    impedance: float = 10_000,
    name: str = "Lead_Lag_Comp",
) -> CompensationResult:
    """Create combined lead-lag compensation network.

    Combines lead compensation (for phase boost) with lag compensation
    (for low-frequency gain). The lag portion adds gain at DC while
    the lead portion provides phase margin improvement.

    Circuit topology (series connection):
        in ---[Lead Network]---[Lag Network]--- out

    Lead: Provides phase boost near crossover
    Lag: Provides high DC gain, rolls off at lag_pole_freq

    Args:
        lead_zero_freq: Lead zero frequency in Hz
        lead_pole_freq: Lead pole frequency in Hz (must be > lead_zero_freq)
        lag_pole_freq: Lag pole frequency in Hz (typically << lead_zero_freq)
        impedance: Base impedance (default 10kΩ)
        name: Circuit name

    Returns:
        CompensationResult with circuit and key frequencies

    Example:
        >>> result = lead_lag_compensation(
        ...     lead_zero_freq=1000,
        ...     lead_pole_freq=10000,
        ...     lag_pole_freq=10
        ... )
    """
    if lead_zero_freq >= lead_pole_freq:
        raise ValueError(f"Lead zero ({lead_zero_freq}Hz) must be < lead pole ({lead_pole_freq}Hz)")

    circuit = Circuit(name)

    # Lead network design
    wz_lead = 2 * math.pi * lead_zero_freq
    wp_lead = 2 * math.pi * lead_pole_freq

    R1_lead = impedance
    C_lead = 1 / (wz_lead * R1_lead)
    R2_lead = R1_lead / (wp_lead / wz_lead - 1)

    # Lag network design (simple RC)
    wp_lag = 2 * math.pi * lag_pole_freq
    R_lag = impedance
    C_lag = 1 / (wp_lag * R_lag)

    # Create nets
    vin = Net("in")
    n1 = Net("n1")
    n2 = Net("n2")
    vout = Net("out")

    # Lead components
    R1_l = Resistor(ref="lead_1", resistance=R1_lead)
    R2_l = Resistor(ref="lead_2", resistance=R2_lead)
    C_l = Capacitor(ref="lead", capacitance=C_lead)

    # Lag components
    R_lag_comp = Resistor(ref="lag", resistance=R_lag)
    C_lag_comp = Capacitor(ref="lag", capacitance=C_lag)

    circuit.add(R1_l, R2_l, C_l, R_lag_comp, C_lag_comp)

    # Lead network connections
    circuit.connect(R1_l.ports[0], vin)
    circuit.connect(R1_l.ports[1], n1)
    circuit.connect(R2_l.ports[0], n1)
    circuit.connect(R2_l.ports[1], n2)
    circuit.connect(C_l.ports[0], n1)
    circuit.connect(C_l.ports[1], n2)

    # Lag network connections
    circuit.connect(R_lag_comp.ports[0], n2)
    circuit.connect(R_lag_comp.ports[1], vout)
    circuit.connect(C_lag_comp.ports[0], vout)
    circuit.connect(C_lag_comp.ports[1], GND)

    # Phase boost from lead network
    phase_boost = math.degrees(math.asin((wp_lead - wz_lead) / (wp_lead + wz_lead)))

    return CompensationResult(
        circuit=circuit,
        components={
            "R1_lead": R1_l,
            "R2_lead": R2_l,
            "C_lead": C_l,
            "R_lag": R_lag_comp,
            "C_lag": C_lag_comp,
        },
        pole_frequency=lead_pole_freq,
        zero_frequency=lead_zero_freq,
        phase_margin_boost=phase_boost,
    )


def miller_compensation(
    pole_frequency: float,
    gain: float = 100,
    impedance: float = 10_000,
    name: str = "Miller_Comp",
) -> CompensationResult:
    """Create Miller compensation network.

    Miller compensation uses a capacitor across a gain stage to create
    pole splitting - moving the dominant pole to lower frequency while
    pushing non-dominant poles higher. This is the most common internal
    compensation technique in opamps.

    The effective capacitance is multiplied by (1 + A) where A is the
    stage gain (Miller effect), allowing small capacitors to create
    low-frequency poles.

    Circuit topology (compensation capacitor across inverting stage):
        in ---+---[Gain=-A]---+--- out
              |               |
              +-----[Cc]------+

    Effective capacitance: Ceff = Cc * (1 + A)
    Pole frequency: fp ≈ 1/(2π * Rout * Cc * A)

    Args:
        pole_frequency: Desired dominant pole frequency in Hz
        gain: Gain of the stage being compensated (default 100)
        impedance: Output impedance of the stage (default 10kΩ)
        name: Circuit name

    Returns:
        CompensationResult with circuit and pole frequency

    Example:
        >>> result = miller_compensation(pole_frequency=10, gain=100)
        >>> # Creates compensation network for 10Hz dominant pole
    """
    # Cc = 1 / (2π * fp * Rout * A)
    # But we model it as feedback capacitor with effective value
    Rout = impedance
    Cc = 1 / (2 * math.pi * pole_frequency * Rout * gain)

    circuit = Circuit(name)

    vin = Net("in")
    vout = Net("out")

    # Model as simple feedback capacitor
    # In practice, this would be across an inverting gain stage
    C_miller = Capacitor(ref="miller", capacitance=Cc)

    # Add a resistor to represent stage output impedance
    R_out = Resistor(ref="out", resistance=Rout)

    circuit.add(C_miller, R_out)

    # The capacitor provides feedback from output to input
    circuit.connect(C_miller.ports[0], vin)
    circuit.connect(C_miller.ports[1], vout)
    circuit.connect(R_out.ports[0], vout)
    circuit.connect(R_out.ports[1], GND)

    return CompensationResult(
        circuit=circuit,
        components={"C_miller": C_miller, "R_out": R_out},
        pole_frequency=pole_frequency,
    )


# =============================================================================
# ADC/DAC Building Blocks
# =============================================================================


def r2r_dac_ladder(
    bits: int = 4,
    r_value: float = 10_000,
    name: str = "R2R_DAC",
) -> ConverterResult:
    """Create R-2R resistor ladder DAC.

    The R-2R ladder is a simple DAC topology using only two resistor values.
    Each bit input contributes a binary-weighted current to the output.

    Circuit topology (4-bit example):
        B3 --[2R]--+--[R]--+--[R]--+--[R]--+-- Vout
                   |       |       |       |
        B2 --[2R]--+       |       |       |
                           |       |       |
        B1 --------[2R]----+       |       |
                                   |       |
        B0 ----------------[2R]----+       |
                                           |
                                   [2R]----+
                                           |
                                          GND

    Output voltage: Vout = Vref * (B3*8 + B2*4 + B1*2 + B0) / 16

    Args:
        bits: Number of bits (1-8, default 4)
        r_value: Base resistance R (default 10kΩ)
        name: Circuit name

    Returns:
        ConverterResult with circuit, bit inputs, and output

    Raises:
        ValueError: If bits not in 1-8

    Example:
        >>> result = r2r_dac_ladder(bits=4)
        >>> # 4-bit DAC with 16 output levels
        >>> print(f"Inputs: {[n.name for n in result.input_nets]}")
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"Bits must be 1-8, got {bits}")

    circuit = Circuit(name)
    components: dict[str, Component] = {}
    input_nets: list[Net] = []

    R = r_value
    R2 = 2 * r_value

    # Create bit input nets (LSB to MSB)
    for i in range(bits):
        input_nets.append(Net(f"b{i}"))

    # Output net
    vout = Net("vout")

    # Build the ladder from LSB (right) to MSB (left)
    prev_node = vout

    for i in range(bits):
        # Each bit position has a 2R to the bit input
        R_bit = Resistor(ref=f"2r_b{i}", resistance=R2)
        circuit.add(R_bit)
        components[f"R2_b{i}"] = R_bit

        circuit.connect(R_bit.ports[0], input_nets[i])
        circuit.connect(R_bit.ports[1], prev_node)

        # Add series R between stages (except after MSB)
        if i < bits - 1:
            next_node = Net(f"n{i}")
            R_series = Resistor(ref=f"r_s{i}", resistance=R)
            circuit.add(R_series)
            components[f"R_s{i}"] = R_series

            circuit.connect(R_series.ports[0], prev_node)
            circuit.connect(R_series.ports[1], next_node)
            prev_node = next_node

    # Termination resistor 2R to ground at output
    R_term = Resistor(ref="term", resistance=R2)
    circuit.add(R_term)
    components["R_term"] = R_term
    circuit.connect(R_term.ports[0], vout)
    circuit.connect(R_term.ports[1], GND)

    return ConverterResult(
        circuit=circuit,
        components=components,
        resolution_bits=bits,
        input_nets=input_nets,
        output_net=vout,
    )


def sample_and_hold(
    hold_capacitance: float = 100e-12,
    buffer_impedance: float = 10_000,
    name: str = "Sample_Hold",
) -> ConverterResult:
    """Create sample and hold circuit.

    The sample-and-hold circuit captures an analog voltage at a specific time
    and holds it constant for ADC conversion. Uses a switch and capacitor
    with an output buffer.

    Circuit topology:
        vin ---[Switch]---+--- vhold
                          |
                         [C]
                          |
                         GND

    In sample mode: switch closed, capacitor tracks input
    In hold mode: switch open, capacitor holds voltage

    Note: This creates the passive S/H network. The switch is modeled as
    a resistor (switch_on_resistance) for simulation.

    Args:
        hold_capacitance: Hold capacitor value (default 100pF)
        buffer_impedance: Switch/buffer impedance (default 10kΩ)
        name: Circuit name

    Returns:
        ConverterResult with circuit, input/output nets

    Example:
        >>> result = sample_and_hold(hold_capacitance=100e-12)
        >>> # Creates S/H with 100pF hold capacitor
    """
    circuit = Circuit(name)

    vin = Net("vin")
    vhold = Net("vhold")

    # Model switch as resistor (for AC/DC analysis)
    # In practice, would use a transistor switch
    R_sw = Resistor(ref="sw", resistance=buffer_impedance)
    C_hold = Capacitor(ref="hold", capacitance=hold_capacitance)

    circuit.add(R_sw, C_hold)

    # Switch between input and hold node
    circuit.connect(R_sw.ports[0], vin)
    circuit.connect(R_sw.ports[1], vhold)

    # Hold capacitor
    circuit.connect(C_hold.ports[0], vhold)
    circuit.connect(C_hold.ports[1], GND)

    return ConverterResult(
        circuit=circuit,
        components={"R_switch": R_sw, "C_hold": C_hold},
        resolution_bits=0,  # Analog, no discrete bits
        input_nets=[vin],
        output_net=vhold,
    )


def comparator_bank(
    bits: int = 3,
    vref_range: tuple[float, float] = (0.0, 5.0),
    name: str = "Comparator_Bank",
) -> ConverterResult:
    """Create comparator reference ladder for flash ADC.

    A flash ADC uses 2^N - 1 comparators with equally spaced reference
    voltages to convert an analog input to N bits in a single cycle.
    This creates the resistor ladder that generates the reference voltages.

    Circuit topology (3-bit = 7 comparators):
        Vref_high
           |
          [R]--+-- vref7 (7/8 * Vrange)
          [R]--+-- vref6 (6/8 * Vrange)
          [R]--+-- vref5 (5/8 * Vrange)
          [R]--+-- vref4 (4/8 * Vrange)
          [R]--+-- vref3 (3/8 * Vrange)
          [R]--+-- vref2 (2/8 * Vrange)
          [R]--+-- vref1 (1/8 * Vrange)
          [R]
           |
        Vref_low

    Args:
        bits: Number of ADC bits (1-4, default 3)
        vref_range: Reference voltage range (vlow, vhigh), default (0V, 5V)
        name: Circuit name

    Returns:
        ConverterResult with circuit and reference voltage nets

    Raises:
        ValueError: If bits not in 1-4

    Example:
        >>> result = comparator_bank(bits=3, vref_range=(0, 3.3))
        >>> # Creates 7 reference voltages for 3-bit flash ADC
    """
    if bits < 1 or bits > 4:
        raise ValueError(f"Bits must be 1-4 for flash ADC, got {bits}")

    num_comparators = (2**bits) - 1
    num_resistors = num_comparators + 1
    vlow, vhigh = vref_range

    circuit = Circuit(name)
    components: dict[str, Component] = {}
    ref_nets: list[Net] = []

    # Calculate resistor value for equal division
    # Total resistance is arbitrary; use 10k per segment
    R = 10_000

    # Create reference voltage nets
    for i in range(1, num_comparators + 1):
        ref_nets.append(Net(f"vref{i}"))

    # Create supply voltage sources
    V_high = Vdc(ref="refH", value=str(vhigh))
    circuit.add(V_high)
    components["V_high"] = V_high

    vhigh_net = Net("vref_high")
    vlow_net: Net

    # Connect voltage sources
    circuit.connect(V_high.ports[0], vhigh_net)
    circuit.connect(V_high.ports[1], GND)

    if vlow != 0:
        V_low = Vdc(ref="refL", value=str(abs(vlow)))
        circuit.add(V_low)
        components["V_low"] = V_low
        vlow_net = Net("vref_low")
        circuit.connect(V_low.ports[0], vlow_net)
        circuit.connect(V_low.ports[1], GND)
    else:
        vlow_net = GND

    # Build resistor ladder from high to low
    prev_node = vhigh_net

    for i in range(num_resistors):
        R_i = Resistor(ref=f"div{i + 1}", resistance=R)
        circuit.add(R_i)
        components[f"R{i + 1}"] = R_i

        circuit.connect(R_i.ports[0], prev_node)

        if i < num_comparators:
            # Tap point for reference voltage
            circuit.connect(R_i.ports[1], ref_nets[num_comparators - 1 - i])
            prev_node = ref_nets[num_comparators - 1 - i]
        else:
            # Last resistor to ground/vlow
            circuit.connect(R_i.ports[1], vlow_net)

    # Output is the highest reference (vin connects to all comparators)
    output_net = Net("vin_compare")

    return ConverterResult(
        circuit=circuit,
        components=components,
        resolution_bits=bits,
        input_nets=ref_nets,  # Reference voltages for comparators
        output_net=output_net,
    )
