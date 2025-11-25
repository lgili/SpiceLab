"""Tutorial 03: Voltage Sources

This tutorial teaches you how to use different voltage sources:
1. DC voltage source (Vdc)
2. AC voltage source (Vac)
3. Pulse source (Vpulse)
4. Sinusoidal source (Vsin)

Run: python examples/tutorials/03_voltage_sources.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vac, Vdc, Vpulse, Vsin
from spicelab.core.net import GND, Net


def demo_dc_source():
    """Demonstrate DC voltage source."""
    print("\n1. DC Voltage Source (Vdc)")
    print("-" * 40)

    circuit = Circuit("dc_demo")

    # DC source with 5V
    v1 = Vdc("1", 5.0)
    r1 = Resistor("1", resistance=1000)

    circuit.add(v1, r1)

    vcc = Net("vcc")
    circuit.connect(v1.ports[0], vcc)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], vcc)
    circuit.connect(r1.ports[1], GND)

    print("""
   Vdc: Constant DC voltage

   Usage: Vdc(ref, value)  # value is DC voltage

   Parameters:
   - voltage: DC voltage in Volts

   Applications:
   - Power supply rails
   - Bias voltages
   - Reference voltages
""")
    print("   Netlist:")
    print(circuit.build_netlist())


def demo_ac_source():
    """Demonstrate AC voltage source."""
    print("\n2. AC Voltage Source (Vac)")
    print("-" * 40)

    circuit = Circuit("ac_demo")

    # AC source: 1V amplitude for AC analysis
    v1 = Vac("1", ac_mag=1.0)
    r1 = Resistor("1", resistance=1000)

    circuit.add(v1, r1)

    vin = Net("vin")
    circuit.connect(v1.ports[0], vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], GND)

    print("""
   Vac: AC source for frequency analysis

   Usage: Vac(ref, amplitude=1.0)

   Parameters:
   - amplitude: AC amplitude in Volts (default 1V)

   Applications:
   - AC analysis (frequency response)
   - Bode plots
   - Filter characterization
   - Small-signal analysis

   Note: Vac is used with .AC analysis command
""")
    print("   Netlist:")
    print(circuit.build_netlist())


def demo_pulse_source():
    """Demonstrate pulse voltage source."""
    print("\n3. Pulse Voltage Source (Vpulse)")
    print("-" * 40)

    circuit = Circuit("pulse_demo")

    # Pulse source: 0V to 5V, 1kHz, 50% duty cycle
    v1 = Vpulse(
        "1",
        v1=0,  # Initial voltage
        v2=5,  # Pulse voltage
        td=0,  # Delay time
        tr=1e-9,  # Rise time (1ns)
        tf=1e-9,  # Fall time (1ns)
        pw=500e-6,  # Pulse width (500us = 50% of 1ms period)
        per=1e-3,  # Period (1ms = 1kHz)
    )
    r1 = Resistor("1", resistance=1000)

    circuit.add(v1, r1)

    vclk = Net("vclk")
    circuit.connect(v1.ports[0], vclk)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], vclk)
    circuit.connect(r1.ports[1], GND)

    print("""
   Vpulse: Rectangular pulse waveform

   Usage: Vpulse(ref, v1=0, v2=5, td=0, tr=1n, tf=1n, pw=500u, per=1m)

   Parameters:
   - v1: Initial/low voltage
   - v2: Pulse/high voltage
   - td: Delay before first pulse
   - tr: Rise time
   - tf: Fall time
   - pw: Pulse width (high time)
   - per: Period

   Waveform:
        v2 ─────┐       ┌─────┐       ┌─────
               │       │     │       │
               │       │     │       │
        v1 ────┘       └─────┘       └─────
              ←─pw─→
              ←────per────→

   Applications:
   - Digital signals / clocks
   - PWM analysis
   - Step response testing
   - Switching power supplies
""")
    print("   Netlist:")
    print(circuit.build_netlist())


def demo_sin_source():
    """Demonstrate sinusoidal voltage source."""
    print("\n4. Sinusoidal Voltage Source (Vsin)")
    print("-" * 40)

    circuit = Circuit("sin_demo")

    # Sinusoidal source: 1kHz, 1V amplitude, 0V offset
    # Vsin takes raw SPICE SIN args: VO VA FREQ [TD THETA]
    v1 = Vsin("1", "0 1.0 1000")
    r1 = Resistor("1", resistance=1000)

    circuit.add(v1, r1)

    vsig = Net("vsig")
    circuit.connect(v1.ports[0], vsig)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], vsig)
    circuit.connect(r1.ports[1], GND)

    print("""
   Vsin: Sinusoidal waveform

   Usage: Vsin(ref, "VO VA FREQ")  # raw SPICE args

   Parameters:
   - vo: DC offset voltage
   - va: Amplitude (peak)
   - freq: Frequency in Hz

   Equation: V(t) = vo + va * sin(2π * freq * t)

   Waveform:
        vo+va ─     ╭───╮       ╭───╮
              │    ╱     ╲     ╱     ╲
        vo ───┼───╱───────╲───╱───────╲───
              │  ╱         ╲ ╱
        vo-va ─ ╱           ╲

   Applications:
   - Audio signals
   - Transient analysis
   - Signal processing
   - Oscillator testing
""")
    print("   Netlist:")
    print(circuit.build_netlist())


def main():
    """Demonstrate all voltage source types."""
    print("=" * 60)
    print("Tutorial 03: Voltage Sources")
    print("=" * 60)

    demo_dc_source()
    demo_ac_source()
    demo_pulse_source()
    demo_sin_source()

    print("\n" + "=" * 60)
    print("Summary: Choosing the Right Source")
    print("=" * 60)
    print("""
   | Source  | Use Case                          | Analysis Type |
   |---------|-----------------------------------|---------------|
   | Vdc     | Power supplies, bias              | DC, Transient |
   | Vac     | Frequency response                | AC            |
   | Vpulse  | Digital signals, step response    | Transient     |
   | Vsin    | Audio, continuous signals         | Transient     |
""")


if __name__ == "__main__":
    main()
