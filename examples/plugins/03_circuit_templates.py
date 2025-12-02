"""Example: Circuit Templates Plugin

This example demonstrates how to use the CircuitTemplatesPlugin to
quickly create common circuit designs with customizable parameters.

Available template categories:
- Filters: RC/RL/RLC lowpass, highpass, bandpass
- Basic: Voltage dividers
- Amplifiers: Inverting, non-inverting, voltage follower
- Oscillators: RC phase-shift oscillator
- Test: Transient test circuits
"""

from spicelab.core.types import AnalysisSpec
from spicelab.plugins.examples import CircuitTemplatesPlugin


def example_list_templates():
    """List all available templates."""
    print("=" * 60)
    print("Example 1: List Available Templates")
    print("=" * 60)

    plugin = CircuitTemplatesPlugin()

    # List all categories
    categories = plugin.list_categories()
    print(f"\nAvailable categories: {categories}")

    # List templates by category
    for category in sorted(categories):
        templates = plugin.list_templates(category=category)
        print(f"\n{category.upper()}:")
        for name in templates:
            info = plugin.get_template_info(name)
            print(f"  - {name}: {info.description}")


def example_rc_filters():
    """Create RC low-pass and high-pass filters."""
    print("\n" + "=" * 60)
    print("Example 2: RC Filters")
    print("=" * 60)

    plugin = CircuitTemplatesPlugin()

    # Create RC low-pass filter with 1kHz cutoff
    print("\n1. RC Low-Pass Filter (1kHz cutoff)")
    lp_filter = plugin.create("rc_lowpass", cutoff_freq=1000)
    print(f"   Circuit: {lp_filter.name}")
    print(f"   Components: {len(lp_filter._components)}")
    print("\n   Netlist:")
    print("   " + lp_filter.build_netlist().replace("\n", "\n   "))

    # Create RC high-pass filter with 100Hz cutoff
    print("\n2. RC High-Pass Filter (100Hz cutoff)")
    hp_filter = plugin.create("rc_highpass", cutoff_freq=100)
    print(f"   Circuit: {hp_filter.name}")

    # Create filter with specific component values
    print("\n3. RC Low-Pass with custom values")
    custom_lp = plugin.create(
        "rc_lowpass",
        r_value="4.7k",
        c_value="33n",
        vin=3.3,
        name="custom_lpf",
    )
    print(f"   Circuit: {custom_lp.name}")
    print("   Custom R=4.7k, C=33nF, Vin=3.3V")


def example_voltage_dividers():
    """Create voltage divider circuits."""
    print("\n" + "=" * 60)
    print("Example 3: Voltage Dividers")
    print("=" * 60)

    plugin = CircuitTemplatesPlugin()

    # Simple voltage divider
    print("\n1. Simple Voltage Divider (10k/10k)")
    divider = plugin.create("voltage_divider", r1="10k", r2="10k", vin=12.0)
    print(f"   Vin = 12V, Vout = 6V (ratio 0.5)")

    # Voltage divider by ratio
    print("\n2. Voltage Divider by Ratio")
    print("   Creating 3.3V from 5V supply (ratio = 0.66)")
    ratio_divider = plugin.create(
        "voltage_divider_ratio",
        ratio=0.66,
        vin=5.0,
        total_r="20k",
    )
    print(f"   Circuit created with automatic R1/R2 calculation")

    # Different ratios
    print("\n3. Common Voltage Divider Ratios:")
    ratios = [0.5, 0.33, 0.25, 0.1]
    for ratio in ratios:
        circuit = plugin.create("voltage_divider_ratio", ratio=ratio, vin=10.0)
        vout = 10.0 * ratio
        print(f"   Ratio {ratio}: 10V → {vout}V")


def example_rlc_bandpass():
    """Create RLC band-pass filter."""
    print("\n" + "=" * 60)
    print("Example 4: RLC Band-Pass Filter")
    print("=" * 60)

    plugin = CircuitTemplatesPlugin()

    # Create band-pass filter centered at 10kHz
    print("\n1. Band-Pass Filter (10kHz center, Q=10)")
    bp_filter = plugin.create(
        "rlc_bandpass",
        center_freq=10000,
        q_factor=10,
        r_value="1k",
    )
    print(f"   Circuit: {bp_filter.name}")
    print(f"   Components: {len(bp_filter._components)}")

    # Show netlist
    print("\n   Netlist:")
    netlist = bp_filter.build_netlist()
    for line in netlist.split("\n")[:10]:
        print(f"   {line}")

    # Different Q factors
    print("\n2. Effect of Q Factor:")
    print("   Higher Q = narrower bandwidth, sharper response")
    for q in [1, 5, 10, 50]:
        circuit = plugin.create("rlc_bandpass", center_freq=1000, q_factor=q)
        print(f"   Q={q}: Created filter with name '{circuit.name}'")


def example_amplifier_circuits():
    """Create op-amp based amplifier circuits."""
    print("\n" + "=" * 60)
    print("Example 5: Op-Amp Amplifier Circuits")
    print("=" * 60)

    plugin = CircuitTemplatesPlugin()

    # Inverting amplifier
    print("\n1. Inverting Amplifier (Gain = -10)")
    inv_amp = plugin.create(
        "inverting_amplifier",
        gain=10,  # Will be -10 due to inverting config
        rin="10k",
        vcc=15,
        vee=-15,
    )
    print(f"   Rin = 10k, Rf = 100k (Rf = gain × Rin)")
    print(f"   Supply: ±15V")

    # Non-inverting amplifier
    print("\n2. Non-Inverting Amplifier (Gain = 5)")
    non_inv = plugin.create(
        "non_inverting_amplifier",
        gain=5,
        r1="10k",
        vcc=12,
        vee=-12,
    )
    print(f"   Gain = 1 + Rf/R1 = 5")
    print(f"   R1 = 10k, Rf = 40k")

    # Voltage follower (unity gain buffer)
    print("\n3. Voltage Follower (Unity Gain Buffer)")
    buffer = plugin.create(
        "voltage_follower",
        vcc=5,
        vee=-5,
    )
    print(f"   Gain = 1, High input impedance, Low output impedance")
    print(f"   Supply: ±5V")


def example_test_circuits():
    """Create test circuits for simulation validation."""
    print("\n" + "=" * 60)
    print("Example 6: Test Circuits")
    print("=" * 60)

    plugin = CircuitTemplatesPlugin()

    # RC transient test circuit
    print("\n1. RC Transient Test Circuit")
    print("   For testing step response and time constants")

    # Different time constants
    for tau_ms in [0.1, 1, 10]:
        tau = tau_ms * 1e-3
        circuit = plugin.create(
            "rc_transient_test",
            tau=tau,
            vpulse=5.0,
            name=f"rc_tau_{tau_ms}ms",
        )
        print(f"\n   τ = {tau_ms}ms:")
        print(f"   Circuit: {circuit.name}")
        print(f"   Expected 63.2% rise time: {tau_ms}ms")
        print(f"   Expected 99.3% settling: {5 * tau_ms}ms")


def example_custom_template():
    """Register and use a custom template."""
    print("\n" + "=" * 60)
    print("Example 7: Custom Templates")
    print("=" * 60)

    from spicelab.core.circuit import Circuit
    from spicelab.core.components import Capacitor, Resistor, Vdc
    from spicelab.core.net import GND, Net

    plugin = CircuitTemplatesPlugin()

    # Define a custom template function
    def create_pi_filter(
        r_value: str = "100",
        c_value: str = "10u",
        vin: float = 12.0,
        name: str = "pi_filter",
    ):
        """Create a Pi (π) filter for power supply filtering."""
        circuit = Circuit(name)

        v_in = Vdc("Vin", vin)
        c1 = Capacitor("C1", c_value)  # Input capacitor
        r1 = Resistor("R1", r_value)  # Series resistor
        c2 = Capacitor("C2", c_value)  # Output capacitor

        circuit.add(v_in, c1, r1, c2)

        circuit.connect(v_in.ports[0], Net("vin"))
        circuit.connect(v_in.ports[1], GND)
        circuit.connect(c1.ports[0], Net("vin"))
        circuit.connect(c1.ports[1], GND)
        circuit.connect(r1.ports[0], Net("vin"))
        circuit.connect(r1.ports[1], Net("vout"))
        circuit.connect(c2.ports[0], Net("vout"))
        circuit.connect(c2.ports[1], GND)

        return circuit

    # Register the custom template
    plugin.register_template(
        name="pi_filter",
        description="Pi filter for power supply filtering",
        category="power",
        parameters={
            "r_value": "Series resistor value",
            "c_value": "Capacitor value (both caps)",
            "vin": "Input voltage",
        },
        create_fn=create_pi_filter,
    )

    print("\n1. Registered custom template: pi_filter")
    print(f"   Categories now: {plugin.list_categories()}")

    # Use the custom template
    print("\n2. Creating Pi filter from custom template:")
    pi = plugin.create("pi_filter", r_value="47", c_value="100u", vin=24.0)
    print(f"   Circuit: {pi.name}")
    print(f"   Components: {len(pi._components)}")


def example_template_info():
    """Show detailed template information."""
    print("\n" + "=" * 60)
    print("Example 8: Template Information")
    print("=" * 60)

    plugin = CircuitTemplatesPlugin()

    templates_to_show = ["rc_lowpass", "voltage_divider", "inverting_amplifier"]

    for name in templates_to_show:
        info = plugin.get_template_info(name)
        print(f"\n{info.name}:")
        print(f"  Description: {info.description}")
        print(f"  Category: {info.category}")
        print(f"  Parameters:")
        for param, desc in info.parameters.items():
            print(f"    - {param}: {desc}")


if __name__ == "__main__":
    example_list_templates()
    example_rc_filters()
    example_voltage_dividers()
    example_rlc_bandpass()
    example_amplifier_circuits()
    example_test_circuits()
    example_custom_template()
    example_template_info()

    print("\n" + "=" * 60)
    print("Circuit Templates Plugin Examples Complete!")
    print("=" * 60)
