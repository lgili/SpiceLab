"""Reference Circuits Library - Educational and Application Note Examples.

This module provides pre-built reference circuits organized by category:

1. **Educational Circuits**: Basic circuits for learning electronics
   - Ohm's law demonstration
   - Kirchhoff's laws verification
   - RC/RL time constants
   - Resonance demonstration

2. **Application Notes**: Common real-world designs
   - LED driver with current limiting
   - Sensor signal conditioning
   - Audio amplifier (LM386 style)
   - Voltage regulator simulation

3. **Test & Validation Circuits**: For verifying simulation accuracy
   - Known response circuits for AC/DC/Transient
   - Stability test circuits

Part of Sprint 3 (M15) - Reference Circuits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.circuit import Circuit
from ..core.components import (
    Capacitor,
    Diode,
    Inductor,
    Resistor,
    Vdc,
    Vpulse,
    Vsin,
)
from ..core.net import GND, Net

if TYPE_CHECKING:
    from ..core.components import Component


@dataclass
class ReferenceCircuit:
    """A reference circuit with metadata and educational content.

    Attributes:
        circuit: The Circuit object
        name: Descriptive name
        category: Category (educational, appnote, test)
        description: What the circuit does
        learning_objectives: What can be learned from this circuit
        expected_results: What results to expect from simulation
        components: Dictionary of key components
        nodes: Dictionary of key nodes
        suggested_analyses: List of suggested SPICE analyses
        parameters: Design parameters used
    """

    circuit: Circuit
    name: str
    category: str
    description: str
    learning_objectives: list[str] = field(default_factory=list)
    expected_results: dict[str, Any] = field(default_factory=dict)
    components: dict[str, Component] = field(default_factory=dict)
    nodes: dict[str, Net] = field(default_factory=dict)
    suggested_analyses: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"ReferenceCircuit({self.name!r}, category={self.category!r})"

    def summary(self) -> str:
        """Get a summary of the circuit for display."""
        lines = [
            f"=== {self.name} ===",
            f"Category: {self.category}",
            f"Description: {self.description}",
            "",
            "Learning Objectives:",
        ]
        for obj in self.learning_objectives:
            lines.append(f"  - {obj}")

        lines.append("")
        lines.append("Suggested Analyses:")
        for analysis in self.suggested_analyses:
            lines.append(f"  - {analysis}")

        if self.expected_results:
            lines.append("")
            lines.append("Expected Results:")
            for key, value in self.expected_results.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


# =============================================================================
# Educational Circuits
# =============================================================================


def ohms_law_demo(
    voltage: float = 5.0,
    resistance: float = 1000.0,
) -> ReferenceCircuit:
    """Ohm's Law demonstration circuit.

    Simple circuit to verify V = I × R.

    Args:
        voltage: Source voltage (V)
        resistance: Load resistance (Ω)

    Returns:
        ReferenceCircuit with expected current I = V/R
    """
    circuit = Circuit("ohms_law_demo")

    v1 = Vdc("V1", voltage)
    r1 = Resistor("R1", resistance)

    circuit.add(v1, r1)

    n_vcc = Net("vcc")
    circuit.connect(v1.ports[0], n_vcc)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], n_vcc)
    circuit.connect(r1.ports[1], GND)

    expected_current = voltage / resistance

    return ReferenceCircuit(
        circuit=circuit,
        name="Ohm's Law Demonstration",
        category="educational",
        description=f"Simple circuit demonstrating V = I × R with V={voltage}V, R={resistance}Ω",
        learning_objectives=[
            "Understand Ohm's Law relationship: V = I × R",
            "Verify simulation accuracy with known results",
            "Learn to read current through components",
        ],
        expected_results={
            "voltage_across_R1": f"{voltage} V",
            "current_through_R1": f"{expected_current * 1000:.3f} mA",
            "power_dissipated": f"{voltage * expected_current * 1000:.3f} mW",
        },
        components={"V1": v1, "R1": r1},
        nodes={"vcc": n_vcc},
        suggested_analyses=["DC operating point (.op)"],
        parameters={"voltage": voltage, "resistance": resistance},
    )


def voltage_divider_demo(
    vin: float = 10.0,
    r1: float = 10000.0,
    r2: float = 10000.0,
) -> ReferenceCircuit:
    """Voltage divider demonstration circuit.

    Shows how voltage divides across series resistors.

    Args:
        vin: Input voltage (V)
        r1: Upper resistor (Ω)
        r2: Lower resistor (Ω)

    Returns:
        ReferenceCircuit with expected Vout = Vin × R2/(R1+R2)
    """
    circuit = Circuit("voltage_divider_demo")

    v1 = Vdc("V1", vin)
    r1_comp = Resistor("R1", r1)
    r2_comp = Resistor("R2", r2)

    circuit.add(v1, r1_comp, r2_comp)

    n_vin = Net("vin")
    n_vout = Net("vout")

    circuit.connect(v1.ports[0], n_vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1_comp.ports[0], n_vin)
    circuit.connect(r1_comp.ports[1], n_vout)
    circuit.connect(r2_comp.ports[0], n_vout)
    circuit.connect(r2_comp.ports[1], GND)

    vout = vin * r2 / (r1 + r2)
    ratio = r2 / (r1 + r2)

    return ReferenceCircuit(
        circuit=circuit,
        name="Voltage Divider Demonstration",
        category="educational",
        description=f"Voltage divider with R1={r1/1000:.1f}kΩ, R2={r2/1000:.1f}kΩ",
        learning_objectives=[
            "Understand voltage division: Vout = Vin × R2/(R1+R2)",
            "Learn about series resistor relationships",
            "See current flow in series circuits",
        ],
        expected_results={
            "input_voltage": f"{vin} V",
            "output_voltage": f"{vout:.3f} V",
            "division_ratio": f"{ratio:.3f}",
        },
        components={"V1": v1, "R1": r1_comp, "R2": r2_comp},
        nodes={"vin": n_vin, "vout": n_vout},
        suggested_analyses=["DC operating point (.op)"],
        parameters={"vin": vin, "r1": r1, "r2": r2},
    )


def rc_time_constant_demo(
    tau_ms: float = 1.0,
    vin: float = 5.0,
) -> ReferenceCircuit:
    """RC time constant demonstration circuit.

    Shows capacitor charging/discharging with step input.

    Args:
        tau_ms: Time constant in milliseconds
        vin: Step voltage (V)

    Returns:
        ReferenceCircuit with RC network and expected charging curve
    """
    tau = tau_ms / 1000  # Convert to seconds

    # Choose R=10k, calculate C for desired tau
    R = 10000
    C = tau / R

    circuit = Circuit("rc_time_constant")

    # Use pulse source for step response
    v1 = Vpulse("V1", v1=0, v2=vin, td=0, tr=1e-9, tf=1e-9, pw=tau * 10, per=tau * 20)
    r1 = Resistor("R1", R)
    c1 = Capacitor("C1", C)

    circuit.add(v1, r1, c1)

    n_vin = Net("vin")
    n_vout = Net("vout")

    circuit.connect(v1.ports[0], n_vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], n_vin)
    circuit.connect(r1.ports[1], n_vout)
    circuit.connect(c1.ports[0], n_vout)
    circuit.connect(c1.ports[1], GND)

    return ReferenceCircuit(
        circuit=circuit,
        name="RC Time Constant Demonstration",
        category="educational",
        description=f"RC circuit with τ = {tau_ms} ms showing capacitor charging",
        learning_objectives=[
            "Understand RC time constant: τ = R × C",
            "Learn capacitor charging equation: V(t) = Vfinal × (1 - e^(-t/τ))",
            "Observe 63.2% charge at t = τ, 99.3% at t = 5τ",
        ],
        expected_results={
            "time_constant": f"{tau_ms} ms",
            "voltage_at_1_tau": f"{vin * 0.632:.3f} V (63.2%)",
            "voltage_at_3_tau": f"{vin * 0.950:.3f} V (95.0%)",
            "voltage_at_5_tau": f"{vin * 0.993:.3f} V (99.3%)",
        },
        components={"V1": v1, "R1": r1, "C1": c1},
        nodes={"vin": n_vin, "vout": n_vout},
        suggested_analyses=[f"Transient (.tran {tau_ms * 10}m)"],
        parameters={"tau_ms": tau_ms, "R": R, "C": C},
    )


def rl_time_constant_demo(
    tau_us: float = 100.0,
    vin: float = 5.0,
) -> ReferenceCircuit:
    """RL time constant demonstration circuit.

    Shows inductor current buildup with step input.

    Args:
        tau_us: Time constant in microseconds
        vin: Step voltage (V)

    Returns:
        ReferenceCircuit with RL network
    """
    tau = tau_us / 1e6  # Convert to seconds

    # Choose L=1mH, calculate R for desired tau
    L = 1e-3
    R = L / tau

    circuit = Circuit("rl_time_constant")

    v1 = Vpulse("V1", v1=0, v2=vin, td=0, tr=1e-9, tf=1e-9, pw=tau * 10, per=tau * 20)
    r1 = Resistor("R1", R)
    l1 = Inductor("L1", L)

    circuit.add(v1, r1, l1)

    n_vin = Net("vin")
    n_mid = Net("mid")

    circuit.connect(v1.ports[0], n_vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], n_vin)
    circuit.connect(r1.ports[1], n_mid)
    circuit.connect(l1.ports[0], n_mid)
    circuit.connect(l1.ports[1], GND)

    i_final = vin / R

    return ReferenceCircuit(
        circuit=circuit,
        name="RL Time Constant Demonstration",
        category="educational",
        description=f"RL circuit with τ = {tau_us} μs showing current buildup",
        learning_objectives=[
            "Understand RL time constant: τ = L / R",
            "Learn inductor current equation: I(t) = Ifinal × (1 - e^(-t/τ))",
            "Compare to RC charging behavior",
        ],
        expected_results={
            "time_constant": f"{tau_us} μs",
            "final_current": f"{i_final * 1000:.3f} mA",
            "current_at_1_tau": f"{i_final * 0.632 * 1000:.3f} mA (63.2%)",
        },
        components={"V1": v1, "R1": r1, "L1": l1},
        nodes={"vin": n_vin, "mid": n_mid},
        suggested_analyses=[f"Transient (.tran {tau_us * 10}u)"],
        parameters={"tau_us": tau_us, "L": L, "R": R},
    )


def rlc_resonance_demo(
    f0_hz: float = 1000.0,
    q_factor: float = 10.0,
) -> ReferenceCircuit:
    """RLC resonance demonstration circuit.

    Series RLC circuit showing resonance behavior.

    Args:
        f0_hz: Resonant frequency in Hz
        q_factor: Quality factor

    Returns:
        ReferenceCircuit with series RLC at resonance
    """
    w0 = 2 * math.pi * f0_hz

    # Design approach: Choose L, calculate C and R
    L = 10e-3  # 10 mH
    C = 1 / (w0**2 * L)
    R = w0 * L / q_factor  # R = w0*L/Q for series RLC

    bw = f0_hz / q_factor  # Bandwidth

    circuit = Circuit("rlc_resonance")

    # AC source for frequency sweep
    v1 = Vsin("V1", f"0 1.0 {f0_hz}")
    r1 = Resistor("R1", R)
    l1 = Inductor("L1", L)
    c1 = Capacitor("C1", C)

    circuit.add(v1, r1, l1, c1)

    n_vin = Net("vin")
    n_vr = Net("vr")
    n_vl = Net("vl")

    circuit.connect(v1.ports[0], n_vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], n_vin)
    circuit.connect(r1.ports[1], n_vr)
    circuit.connect(l1.ports[0], n_vr)
    circuit.connect(l1.ports[1], n_vl)
    circuit.connect(c1.ports[0], n_vl)
    circuit.connect(c1.ports[1], GND)

    return ReferenceCircuit(
        circuit=circuit,
        name="RLC Resonance Demonstration",
        category="educational",
        description=f"Series RLC resonant at {f0_hz} Hz with Q={q_factor}",
        learning_objectives=[
            "Understand resonance: f0 = 1/(2π√(LC))",
            "Learn about Q factor and bandwidth: BW = f0/Q",
            "See phase relationship between V and I at resonance",
            "Observe impedance minimum at resonance",
        ],
        expected_results={
            "resonant_frequency": f"{f0_hz} Hz",
            "q_factor": f"{q_factor}",
            "bandwidth": f"{bw:.1f} Hz",
            "impedance_at_resonance": f"{R:.2f} Ω (purely resistive)",
        },
        components={"V1": v1, "R1": r1, "L1": l1, "C1": c1},
        nodes={"vin": n_vin, "vr": n_vr, "vl": n_vl},
        suggested_analyses=[
            f"AC sweep (.ac dec 100 {f0_hz / 100} {f0_hz * 100})",
            f"Transient (.tran {10 / f0_hz})",
        ],
        parameters={"f0": f0_hz, "Q": q_factor, "L": L, "C": C, "R": R},
    )


# =============================================================================
# Application Note Circuits
# =============================================================================


def led_driver(
    vcc: float = 5.0,
    led_vf: float = 2.0,
    led_current_ma: float = 20.0,
) -> ReferenceCircuit:
    """LED driver with current limiting resistor.

    Common application for driving LEDs safely.

    Args:
        vcc: Supply voltage (V)
        led_vf: LED forward voltage (V)
        led_current_ma: Desired LED current (mA)

    Returns:
        ReferenceCircuit with LED and current limiting resistor
    """
    i_led = led_current_ma / 1000
    r_limit = (vcc - led_vf) / i_led
    power_dissipation = (vcc - led_vf) * i_led

    circuit = Circuit("led_driver")

    v1 = Vdc("V1", vcc)
    r1 = Resistor("R1", r_limit)
    # Model LED as diode with specified Vf
    d1 = Diode("D1", model="LED")

    circuit.add(v1, r1, d1)

    n_vcc = Net("vcc")
    n_anode = Net("led_anode")

    circuit.connect(v1.ports[0], n_vcc)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], n_vcc)
    circuit.connect(r1.ports[1], n_anode)
    circuit.connect(d1.ports[0], n_anode)  # Anode
    circuit.connect(d1.ports[1], GND)  # Cathode

    return ReferenceCircuit(
        circuit=circuit,
        name="LED Driver Circuit",
        category="appnote",
        description=f"LED driver for {led_current_ma}mA from {vcc}V supply",
        learning_objectives=[
            "Calculate current limiting resistor: R = (Vcc - Vf) / I",
            "Understand LED forward voltage drop",
            "Consider resistor power rating",
        ],
        expected_results={
            "led_current": f"{led_current_ma} mA",
            "resistor_value": f"{r_limit:.0f} Ω",
            "resistor_power": f"{power_dissipation * 1000:.1f} mW",
        },
        components={"V1": v1, "R1": r1, "D1": d1},
        nodes={"vcc": n_vcc, "led_anode": n_anode},
        suggested_analyses=["DC operating point (.op)"],
        parameters={
            "vcc": vcc,
            "led_vf": led_vf,
            "led_current_ma": led_current_ma,
        },
    )


def rc_lowpass_filter(
    fc_hz: float = 1000.0,
    impedance: float = 10000.0,
) -> ReferenceCircuit:
    """RC lowpass filter application.

    Common filter for signal conditioning.

    Args:
        fc_hz: Cutoff frequency (-3dB) in Hz
        impedance: Characteristic impedance (Ω)

    Returns:
        ReferenceCircuit with RC lowpass filter
    """
    R = impedance
    C = 1 / (2 * math.pi * fc_hz * R)

    circuit = Circuit("rc_lowpass")

    v1 = Vsin("V1", f"0 1.0 {fc_hz}")
    r1 = Resistor("R1", R)
    c1 = Capacitor("C1", C)

    circuit.add(v1, r1, c1)

    n_vin = Net("vin")
    n_vout = Net("vout")

    circuit.connect(v1.ports[0], n_vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], n_vin)
    circuit.connect(r1.ports[1], n_vout)
    circuit.connect(c1.ports[0], n_vout)
    circuit.connect(c1.ports[1], GND)

    return ReferenceCircuit(
        circuit=circuit,
        name="RC Lowpass Filter",
        category="appnote",
        description=f"First-order lowpass filter with fc = {fc_hz} Hz",
        learning_objectives=[
            "Design RC filter: fc = 1/(2πRC)",
            "Understand -3dB point and rolloff",
            "See phase shift at cutoff frequency (-45°)",
        ],
        expected_results={
            "cutoff_frequency": f"{fc_hz} Hz",
            "gain_at_fc": "-3 dB (0.707×)",
            "phase_at_fc": "-45°",
            "rolloff_rate": "-20 dB/decade",
        },
        components={"V1": v1, "R1": r1, "C1": c1},
        nodes={"vin": n_vin, "vout": n_vout},
        suggested_analyses=[
            f"AC sweep (.ac dec 20 {fc_hz / 100} {fc_hz * 100})",
        ],
        parameters={"fc": fc_hz, "R": R, "C": C},
    )


def decoupling_network(
    vcc: float = 5.0,
) -> ReferenceCircuit:
    """Power supply decoupling network.

    Typical decoupling configuration for digital ICs.

    Args:
        vcc: Supply voltage (V)

    Returns:
        ReferenceCircuit with multi-capacitor decoupling
    """
    circuit = Circuit("decoupling_network")

    v1 = Vdc("V1", vcc)

    # Multi-cap decoupling: bulk + ceramic
    c_bulk = Capacitor("C1", 10e-6)  # 10uF bulk capacitor
    c_ceramic1 = Capacitor("C2", 100e-9)  # 100nF ceramic
    c_ceramic2 = Capacitor("C3", 10e-9)  # 10nF ceramic (high frequency)

    # Small ESR/ESL model resistors
    r_bulk = Resistor("R_ESR1", 0.1)  # ESR of bulk cap
    r_load = Resistor("R_Load", 100)  # Simulated load

    circuit.add(v1, c_bulk, c_ceramic1, c_ceramic2, r_bulk, r_load)

    n_vcc = Net("vcc")
    n_bulk = Net("bulk")

    circuit.connect(v1.ports[0], n_vcc)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r_bulk.ports[0], n_vcc)
    circuit.connect(r_bulk.ports[1], n_bulk)
    circuit.connect(c_bulk.ports[0], n_bulk)
    circuit.connect(c_bulk.ports[1], GND)
    circuit.connect(c_ceramic1.ports[0], n_vcc)
    circuit.connect(c_ceramic1.ports[1], GND)
    circuit.connect(c_ceramic2.ports[0], n_vcc)
    circuit.connect(c_ceramic2.ports[1], GND)
    circuit.connect(r_load.ports[0], n_vcc)
    circuit.connect(r_load.ports[1], GND)

    return ReferenceCircuit(
        circuit=circuit,
        name="Power Supply Decoupling Network",
        category="appnote",
        description="Multi-capacitor decoupling for digital IC power supply",
        learning_objectives=[
            "Understand why multiple capacitor values are used",
            "Learn about capacitor ESR and ESL effects",
            "See impedance vs frequency of decoupling network",
        ],
        expected_results={
            "bulk_capacitance": "10 μF (low frequency energy)",
            "ceramic_100nF": "100 nF (mid frequency filtering)",
            "ceramic_10nF": "10 nF (high frequency filtering)",
        },
        components={
            "V1": v1,
            "C_bulk": c_bulk,
            "C_100n": c_ceramic1,
            "C_10n": c_ceramic2,
        },
        nodes={"vcc": n_vcc, "bulk": n_bulk},
        suggested_analyses=[
            "AC sweep (.ac dec 20 1 100Meg)",
            "Transient with load step",
        ],
        parameters={"vcc": vcc},
    )


# =============================================================================
# Test Circuits
# =============================================================================


def dc_test_circuit() -> ReferenceCircuit:
    """DC operating point test circuit.

    Simple circuit with known DC solution for verifying simulator.

    Returns:
        ReferenceCircuit with known DC operating point
    """
    circuit = Circuit("dc_test")

    v1 = Vdc("V1", 10.0)
    r1 = Resistor("R1", 1000)
    r2 = Resistor("R2", 2000)
    r3 = Resistor("R3", 3000)

    circuit.add(v1, r1, r2, r3)

    n_v1 = Net("v1")
    n_v2 = Net("v2")

    circuit.connect(v1.ports[0], n_v1)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], n_v1)
    circuit.connect(r1.ports[1], n_v2)
    circuit.connect(r2.ports[0], n_v2)
    circuit.connect(r2.ports[1], GND)
    circuit.connect(r3.ports[0], n_v2)
    circuit.connect(r3.ports[1], GND)

    # Calculate expected values
    r_parallel = (2000 * 3000) / (2000 + 3000)  # R2 || R3 = 1200 Ω
    v2_expected = 10.0 * r_parallel / (1000 + r_parallel)  # Voltage divider

    return ReferenceCircuit(
        circuit=circuit,
        name="DC Operating Point Test",
        category="test",
        description="Known DC solution for simulator verification",
        learning_objectives=[
            "Verify simulator DC operating point accuracy",
            "Practice nodal analysis by hand",
        ],
        expected_results={
            "V(v1)": "10.000 V",
            "V(v2)": f"{v2_expected:.3f} V",
            "I(R1)": f"{(10.0 - v2_expected) / 1000 * 1000:.3f} mA",
        },
        components={"V1": v1, "R1": r1, "R2": r2, "R3": r3},
        nodes={"v1": n_v1, "v2": n_v2},
        suggested_analyses=["DC operating point (.op)"],
        parameters={},
    )


def ac_test_circuit(fc_hz: float = 1000.0) -> ReferenceCircuit:
    """AC analysis test circuit.

    RC filter with known frequency response.

    Args:
        fc_hz: Cutoff frequency in Hz

    Returns:
        ReferenceCircuit with known AC response
    """
    R = 10000
    C = 1 / (2 * math.pi * fc_hz * R)

    circuit = Circuit("ac_test")

    v1 = Vsin("V1", f"0 1.0 {fc_hz}")
    r1 = Resistor("R1", R)
    c1 = Capacitor("C1", C)

    circuit.add(v1, r1, c1)

    n_vin = Net("vin")
    n_vout = Net("vout")

    circuit.connect(v1.ports[0], n_vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], n_vin)
    circuit.connect(r1.ports[1], n_vout)
    circuit.connect(c1.ports[0], n_vout)
    circuit.connect(c1.ports[1], GND)

    return ReferenceCircuit(
        circuit=circuit,
        name="AC Analysis Test",
        category="test",
        description=f"RC lowpass with fc = {fc_hz} Hz for AC verification",
        learning_objectives=[
            "Verify simulator AC analysis accuracy",
            "Check gain and phase at known frequencies",
        ],
        expected_results={
            "gain_at_fc/10": "-0.04 dB, -5.7° phase",
            "gain_at_fc": "-3.01 dB, -45° phase",
            "gain_at_fc*10": "-20.04 dB, -84.3° phase",
        },
        components={"V1": v1, "R1": r1, "C1": c1},
        nodes={"vin": n_vin, "vout": n_vout},
        suggested_analyses=[
            f"AC sweep (.ac dec 20 {fc_hz / 100} {fc_hz * 100})",
        ],
        parameters={"fc": fc_hz, "R": R, "C": C},
    )


# =============================================================================
# Circuit Library Access
# =============================================================================


class ReferenceLibrary:
    """Library of reference circuits organized by category.

    Example:
        >>> lib = ReferenceLibrary()
        >>> circuits = lib.list_circuits()
        >>> ohms = lib.get("ohms_law_demo")
        >>> print(ohms.summary())
    """

    _circuits: dict[str, callable] = {
        # Educational
        "ohms_law_demo": ohms_law_demo,
        "voltage_divider_demo": voltage_divider_demo,
        "rc_time_constant_demo": rc_time_constant_demo,
        "rl_time_constant_demo": rl_time_constant_demo,
        "rlc_resonance_demo": rlc_resonance_demo,
        # Application Notes
        "led_driver": led_driver,
        "rc_lowpass_filter": rc_lowpass_filter,
        "decoupling_network": decoupling_network,
        # Test
        "dc_test_circuit": dc_test_circuit,
        "ac_test_circuit": ac_test_circuit,
    }

    _categories: dict[str, list[str]] = {
        "educational": [
            "ohms_law_demo",
            "voltage_divider_demo",
            "rc_time_constant_demo",
            "rl_time_constant_demo",
            "rlc_resonance_demo",
        ],
        "appnote": [
            "led_driver",
            "rc_lowpass_filter",
            "decoupling_network",
        ],
        "test": [
            "dc_test_circuit",
            "ac_test_circuit",
        ],
    }

    def list_circuits(self, category: str | None = None) -> list[str]:
        """List available circuits, optionally filtered by category."""
        if category is None:
            return list(self._circuits.keys())
        return self._categories.get(category, [])

    def list_categories(self) -> list[str]:
        """List available categories."""
        return list(self._categories.keys())

    def get(self, name: str, **kwargs) -> ReferenceCircuit:
        """Get a reference circuit by name.

        Args:
            name: Circuit name
            **kwargs: Parameters passed to circuit constructor

        Returns:
            ReferenceCircuit instance

        Raises:
            KeyError: If circuit not found
        """
        if name not in self._circuits:
            available = ", ".join(self._circuits.keys())
            raise KeyError(f"Circuit '{name}' not found. Available: {available}")

        return self._circuits[name](**kwargs)

    def get_all_in_category(self, category: str) -> list[ReferenceCircuit]:
        """Get all circuits in a category with default parameters."""
        circuits = []
        for name in self._categories.get(category, []):
            circuits.append(self._circuits[name]())
        return circuits


# Module-level library instance
library = ReferenceLibrary()


__all__ = [
    # Data classes
    "ReferenceCircuit",
    "ReferenceLibrary",
    # Educational circuits
    "ohms_law_demo",
    "voltage_divider_demo",
    "rc_time_constant_demo",
    "rl_time_constant_demo",
    "rlc_resonance_demo",
    # Application notes
    "led_driver",
    "rc_lowpass_filter",
    "decoupling_network",
    # Test circuits
    "dc_test_circuit",
    "ac_test_circuit",
    # Library instance
    "library",
]
