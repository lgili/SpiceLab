"""Tutorial 09: Circuit Templates

This tutorial teaches you how to use built-in circuit templates:
1. RC filters (lowpass, highpass)
2. Voltage dividers
3. Template customization

Run: python examples/tutorials/09_circuit_templates.py
"""

import math

from spicelab.templates import rc_highpass, rc_lowpass, voltage_divider


def demo_rc_lowpass_template():
    """Demonstrate RC lowpass filter template."""
    print("\n1. RC Lowpass Filter Template")
    print("-" * 40)

    # Create a 1kHz lowpass filter
    circuit = rc_lowpass(fc=1000, impedance=10_000)

    # Calculate expected values
    R = 10_000
    C = 1 / (2 * math.pi * R * 1000)

    print(f"""
   rc_lowpass(fc, impedance) - Creates an RC lowpass filter

   Parameters:
   - fc: Cutoff frequency in Hz
   - impedance: Filter impedance (resistor value)

   Automatic calculation:
   - C = 1 / (2π × R × fc)
   - For fc={1000}Hz, R={R}Ω:
   - C = {C*1e9:.2f}nF

   Circuit topology:
   Vin ──[R]──┬── Vout
              │
             [C]
              │
             GND

   Frequency response:
   - At f << fc: Gain ≈ 1 (0 dB)
   - At f = fc: Gain = 0.707 (-3 dB)
   - At f >> fc: Gain ≈ fc/f (-20 dB/decade)
""")
    print("   Generated netlist:")
    print(circuit.build_netlist())


def demo_rc_highpass_template():
    """Demonstrate RC highpass filter template."""
    print("\n2. RC Highpass Filter Template")
    print("-" * 40)

    # Create a 100Hz highpass filter
    circuit = rc_highpass(fc=100, impedance=10_000)

    R = 10_000
    C = 1 / (2 * math.pi * R * 100)

    print(f"""
   rc_highpass(fc, impedance) - Creates an RC highpass filter

   Parameters:
   - fc: Cutoff frequency in Hz
   - impedance: Filter impedance (resistor value)

   Automatic calculation:
   - C = 1 / (2π × R × fc)
   - For fc={100}Hz, R={R}Ω:
   - C = {C*1e6:.2f}µF

   Circuit topology:
   Vin ──[C]──┬── Vout
              │
             [R]
              │
             GND

   Frequency response:
   - At f << fc: Gain ≈ f/fc (-20 dB/decade)
   - At f = fc: Gain = 0.707 (-3 dB)
   - At f >> fc: Gain ≈ 1 (0 dB)

   Applications:
   - DC blocking
   - AC coupling
   - Removing low-frequency noise
   - High-pass audio filters
""")
    print("   Generated netlist:")
    print(circuit.build_netlist())


def demo_voltage_divider_template():
    """Demonstrate voltage divider template."""
    print("\n3. Voltage Divider Template")
    print("-" * 40)

    # Create a 2:1 divider (50% ratio)
    circuit = voltage_divider(ratio=0.5, impedance=10_000)

    print("""
   voltage_divider(ratio, impedance) - Creates a resistive divider

   Parameters:
   - ratio: Vout/Vin ratio (0 to 1)
   - impedance: Total resistance (R1 + R2)

   Automatic calculation:
   - R2 = ratio × impedance
   - R1 = impedance - R2

   For ratio=0.5, impedance=10kΩ:
   - R2 = 0.5 × 10k = 5kΩ
   - R1 = 10k - 5k = 5kΩ

   Circuit topology:
   Vin ──[R1]──┬── Vout
               │
              [R2]
               │
              GND

   Formula: Vout = Vin × R2/(R1+R2)
""")
    print("   Generated netlist:")
    print(circuit.build_netlist())


def demo_template_combinations():
    """Demonstrate combining templates."""
    print("\n4. Combining Templates")
    print("-" * 40)

    # Different filter frequencies
    lp_100 = rc_lowpass(fc=100, impedance=10_000)
    lp_1k = rc_lowpass(fc=1000, impedance=10_000)
    lp_10k = rc_lowpass(fc=10_000, impedance=10_000)

    print("""
   Templates make it easy to create multiple variants:

   Lowpass filters at different frequencies:
""")
    print("   fc=100Hz:")
    print(lp_100.build_netlist())
    print("\n   fc=1kHz:")
    print(lp_1k.build_netlist())
    print("\n   fc=10kHz:")
    print(lp_10k.build_netlist())


def demo_template_validation():
    """Demonstrate template validation."""
    print("\n5. Template Validation")
    print("-" * 40)

    circuit = rc_lowpass(fc=1000, impedance=10_000)
    result = circuit.validate()

    print("""
   Templates create validated circuits automatically.
""")
    print(f"   Validation result: {'VALID' if result.is_valid else 'INVALID'}")
    if result.errors:
        for err in result.errors:
            print(f"   Error: {err.message}")
    if result.warnings:
        for warn in result.warnings:
            print(f"   Warning: {warn.message}")


def main():
    """Demonstrate circuit templates."""
    print("=" * 60)
    print("Tutorial 09: Circuit Templates")
    print("=" * 60)

    demo_rc_lowpass_template()
    demo_rc_highpass_template()
    demo_voltage_divider_template()
    demo_template_combinations()
    demo_template_validation()

    print("\n" + "=" * 60)
    print("Summary: Available Templates")
    print("=" * 60)
    print("""
   | Template         | Parameters         | Use Case            |
   |------------------|--------------------|--------------------|
   | rc_lowpass       | fc, impedance      | Audio/signal LP    |
   | rc_highpass      | fc, impedance      | DC blocking, HP    |
   | voltage_divider  | ratio, impedance   | Level shifting     |

   Template benefits:
   - Automatic component calculation
   - Proper net naming
   - Pre-validated circuits
   - Consistent topology

   Import: from spicelab.templates import rc_lowpass, rc_highpass
""")


if __name__ == "__main__":
    main()
