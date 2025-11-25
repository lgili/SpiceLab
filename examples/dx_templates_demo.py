"""DX Demo: Circuit Templates

Demonstrates how to use pre-built circuit templates for rapid prototyping.

Templates available:
- Filters: rc_lowpass, rc_highpass, rlc_bandpass, sallen_key_lowpass
- Amplifiers: inverting_amplifier, non_inverting_amplifier
- Power: voltage_divider

Run: python examples/dx_templates_demo.py
"""

from spicelab.templates import (
    FilterResult,
    inverting_amplifier,
    non_inverting_amplifier,
    rc_highpass,
    rc_lowpass,
    rlc_bandpass,
    sallen_key_lowpass,
    voltage_divider,
)


def demo_rc_filters():
    """Demonstrate RC filter templates."""
    print("=" * 60)
    print("RC Filter Templates")
    print("=" * 60)

    # 1kHz lowpass filter
    lpf = rc_lowpass(fc=1000, impedance=10_000)
    print("\n1. RC Lowpass (fc=1kHz, Z=10kΩ)")
    print(f"   Circuit: {lpf.name}")
    print(f"   Netlist:\n{lpf.build_netlist()}")

    # 100Hz highpass filter
    hpf = rc_highpass(fc=100, impedance=10_000)
    print("\n2. RC Highpass (fc=100Hz, Z=10kΩ)")
    print(f"   Circuit: {hpf.name}")
    print(f"   Netlist:\n{hpf.build_netlist()}")


def demo_rlc_bandpass():
    """Demonstrate RLC bandpass filter template."""
    print("\n" + "=" * 60)
    print("RLC Bandpass Filter Template")
    print("=" * 60)

    # 1kHz center, 100Hz bandwidth (Q=10)
    result: FilterResult = rlc_bandpass(fc=1000, bandwidth=100)

    print("\nRLC Bandpass (fc=1kHz, BW=100Hz)")
    print(f"   Q Factor: {result.q_factor}")
    print(f"   Cutoff: {result.cutoff_frequency} Hz")
    print(f"   Components: {list(result.components.keys())}")
    print(f"\n   Netlist:\n{result.circuit.build_netlist()}")


def demo_sallen_key():
    """Demonstrate Sallen-Key lowpass filter template."""
    print("\n" + "=" * 60)
    print("Sallen-Key Lowpass Filter Template")
    print("=" * 60)

    # Butterworth response (Q=0.707)
    butterworth = sallen_key_lowpass(fc=1000, q=0.707)
    print("\n1. Butterworth (Q=0.707)")
    print(f"   Q Factor: {butterworth.q_factor:.3f}")
    print("   Components: R1, R2, C1, C2")

    # Chebyshev-like response (Q=1.0)
    chebyshev = sallen_key_lowpass(fc=1000, q=1.0)
    print("\n2. Higher Q (Q=1.0)")
    print(f"   Q Factor: {chebyshev.q_factor:.3f}")


def demo_amplifiers():
    """Demonstrate amplifier templates."""
    print("\n" + "=" * 60)
    print("Amplifier Templates")
    print("=" * 60)

    # Inverting amplifier with gain of 10
    inv = inverting_amplifier(gain=10, input_impedance=10_000)
    print("\n1. Inverting Amplifier (Gain=-10)")
    print("   Input impedance: 10kΩ")
    print("   Feedback: 100kΩ (10 × 10kΩ)")
    print(f"   Netlist:\n{inv.build_netlist()}")

    # Non-inverting amplifier with gain of 11
    noninv = non_inverting_amplifier(gain=11, feedback_resistance=10_000)
    print("\n2. Non-Inverting Amplifier (Gain=+11)")
    print("   R1 = Rf/(gain-1) = 10kΩ/10 = 1kΩ")
    print(f"   Netlist:\n{noninv.build_netlist()}")


def demo_voltage_divider():
    """Demonstrate voltage divider template."""
    print("\n" + "=" * 60)
    print("Voltage Divider Template")
    print("=" * 60)

    # 50% divider
    div_50 = voltage_divider(ratio=0.5, total_resistance=10_000)
    print("\n1. 50% Divider")
    print("   R1 = R2 = 5kΩ")
    print(f"   Netlist:\n{div_50.build_netlist()}")

    # 25% divider (for 5V to 1.25V)
    _ = voltage_divider(ratio=0.25, total_resistance=10_000)
    print("\n2. 25% Divider (e.g., 5V → 1.25V)")
    print("   R1 = 7.5kΩ, R2 = 2.5kΩ")


def main():
    """Run all template demos."""
    print("\n" + "=" * 60)
    print("SpiceLab Circuit Templates Demo")
    print("=" * 60)
    print("\nTemplates provide pre-built circuits with correct component")
    print("values calculated from design specifications.")

    demo_rc_filters()
    demo_rlc_bandpass()
    demo_sallen_key()
    demo_amplifiers()
    demo_voltage_divider()

    print("\n" + "=" * 60)
    print("Templates make it easy to create common circuits!")
    print("=" * 60)


if __name__ == "__main__":
    main()
