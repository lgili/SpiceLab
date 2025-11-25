"""Tutorial 02: RC Low-Pass Filter

This tutorial teaches you how to:
1. Build an RC low-pass filter
2. Calculate component values for a cutoff frequency
3. Use templates for common circuits
4. Compare manual vs template approaches

Run: python examples/tutorials/02_rc_filter.py
"""

import math

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.templates import rc_lowpass


def build_rc_filter_manual(fc: float, impedance: float = 10_000) -> Circuit:
    """Build an RC lowpass filter manually.

    Args:
        fc: Cutoff frequency in Hz
        impedance: Filter impedance (R value)

    Returns:
        Circuit with the RC filter
    """
    # Calculate capacitor value: fc = 1 / (2 * pi * R * C)
    # Therefore: C = 1 / (2 * pi * R * fc)
    R = impedance
    C = 1 / (2 * math.pi * R * fc)

    print("   Design equations:")
    print("   fc = 1 / (2π × R × C)")
    print("   C = 1 / (2π × R × fc)")
    print(f"   C = 1 / (2π × {R} × {fc})")
    print(f"   C = {C * 1e9:.2f} nF")

    # Create circuit
    circuit = Circuit("rc_lowpass_manual")

    # Create components
    v1 = Vdc("1", 1.0)  # 1V for easy gain calculation
    r1 = Resistor("1", resistance=R)
    c1 = Capacitor("1", capacitance=C)

    circuit.add(v1, r1, c1)

    # Create nets
    vin = Net("vin")
    vout = Net("vout")

    # Connect: V1 -> R1 -> C1 -> GND
    circuit.connect(v1.ports[0], vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)
    circuit.connect(c1.ports[0], vout)
    circuit.connect(c1.ports[1], GND)

    return circuit


def main():
    """Demonstrate RC filter design."""
    print("=" * 60)
    print("Tutorial 02: RC Low-Pass Filter")
    print("=" * 60)

    # Design parameters
    cutoff_freq = 1000  # 1 kHz
    impedance = 10_000  # 10k ohm

    # =========================================================================
    # Method 1: Build manually
    # =========================================================================
    print("\n1. Building RC filter manually:")
    print("-" * 40)
    manual_circuit = build_rc_filter_manual(cutoff_freq, impedance)
    print("\n   Netlist:")
    print(manual_circuit.build_netlist())

    # =========================================================================
    # Method 2: Use template
    # =========================================================================
    print("\n2. Using rc_lowpass template:")
    print("-" * 40)
    template_circuit = rc_lowpass(fc=cutoff_freq, impedance=impedance)
    print("   Template calculates same values automatically!")
    print("\n   Netlist:")
    print(template_circuit.build_netlist())

    # =========================================================================
    # Filter characteristics
    # =========================================================================
    print("\n3. RC Low-Pass Filter Characteristics:")
    print("-" * 40)
    print(f"""
   Circuit topology:

   Vin ─────┬──[R1]──┬───── Vout
            │        │
           [V1]     [C1]
            │        │
           GND      GND

   Transfer function: H(s) = 1 / (1 + sRC)

   At cutoff frequency (fc = {cutoff_freq} Hz):
   - Gain = -3 dB (≈ 0.707)
   - Phase = -45°

   Below fc: Signals pass with little attenuation
   Above fc: Signals are attenuated by 20 dB/decade
""")

    # =========================================================================
    # Key frequencies
    # =========================================================================
    print("4. Key frequencies:")
    print("-" * 40)

    print(f"   fc (cutoff):     {cutoff_freq} Hz")
    print(f"   fc/10:           {cutoff_freq/10} Hz (passband)")
    print(f"   fc*10:           {cutoff_freq*10} Hz (stopband, -20dB)")
    print(f"   fc*100:          {cutoff_freq*100} Hz (stopband, -40dB)")

    # =========================================================================
    # Validation
    # =========================================================================
    print("\n5. Circuit validation:")
    print("-" * 40)
    result = template_circuit.validate()
    print(f"   Status: {'VALID' if result.is_valid else 'INVALID'}")

    print("\n" + "=" * 60)
    print("Next: Try the rc_highpass template for high-pass filters!")
    print("=" * 60)


if __name__ == "__main__":
    main()
