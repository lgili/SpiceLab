"""Tutorial 10: Component Library

This tutorial teaches you how to use the component library:
1. Listing available components
2. Getting component specifications
3. Creating components from the library
4. Using E-series resistor values

Run: python examples/tutorials/10_component_library.py
"""

from spicelab.library import create_component, get_component_spec, list_components
from spicelab.library.e_series import (
    e12_values,
    e24_values,
    nearest_e12,
    nearest_e24,
)


def demo_list_components():
    """Demonstrate listing components."""
    print("\n1. Listing Available Components")
    print("-" * 40)

    comps = list_components()

    # Group by category
    categories: dict[str, list] = {}
    for c in comps:
        if c.category not in categories:
            categories[c.category] = []
        categories[c.category].append(c.slug)

    print(f"   Total components: {len(comps)}")
    print("\n   By category:")
    for cat, items in sorted(categories.items()):
        print(f"   - {cat}: {len(items)} components")
        # Show first few
        preview = items[:3]
        if len(items) > 3:
            print(f"       {', '.join(preview)}, ...")
        else:
            print(f"       {', '.join(preview)}")


def demo_component_specs():
    """Demonstrate getting component specifications."""
    print("\n2. Component Specifications")
    print("-" * 40)

    # Look up a diode
    diode_spec = get_component_spec("diode.1n4148")
    print(f"""
   get_component_spec("diode.1n4148"):
   - Name: {diode_spec.name}
   - Category: {diode_spec.category}
   - Description: {diode_spec.metadata.get('description', 'N/A')}
""")

    # Look up a BJT
    bjt_spec = get_component_spec("bjt.2n2222")
    print(f"""
   get_component_spec("bjt.2n2222"):
   - Name: {bjt_spec.name}
   - Category: {bjt_spec.category}
   - Type: {bjt_spec.metadata.get('type', 'N/A')}
   - Description: {bjt_spec.metadata.get('description', 'N/A')}
""")

    # Look up a MOSFET
    mosfet_spec = get_component_spec("mosfet.irf540n")
    print(f"""
   get_component_spec("mosfet.irf540n"):
   - Name: {mosfet_spec.name}
   - Category: {mosfet_spec.category}
   - Polarity: {mosfet_spec.metadata.get('polarity', 'N/A')}
   - Description: {mosfet_spec.metadata.get('description', 'N/A')}
""")


def demo_create_components():
    """Demonstrate creating components from library."""
    print("\n3. Creating Components from Library")
    print("-" * 40)

    # Create various components
    d1 = create_component("diode.1n4148", "1")
    q1 = create_component("bjt.2n3904", "1")
    m1 = create_component("mosfet.2n7000", "1")

    print(f"""
   create_component(slug, ref) - Creates a component instance

   Examples:
   d1 = create_component("diode.1n4148", "1")
   - Component type: {type(d1).__name__}
   - Model: {d1.value}
   - Ports: {len(d1.ports)}

   q1 = create_component("bjt.2n3904", "1")
   - Component type: {type(q1).__name__}
   - Model: {q1.value}
   - Ports: {len(q1.ports)} (C, B, E)

   m1 = create_component("mosfet.2n7000", "1")
   - Component type: {type(m1).__name__}
   - Model: {m1.value}
   - Ports: {len(m1.ports)} (D, G, S)
""")


def demo_e_series():
    """Demonstrate E-series resistor values."""
    print("\n4. E-Series Resistor Values")
    print("-" * 40)

    print("""
   E-series: Standard resistor value sequences

   E12 (10% tolerance): 12 values per decade
   E24 (5% tolerance): 24 values per decade
   E48 (2% tolerance): 48 values per decade
   E96 (1% tolerance): 96 values per decade
""")

    # Show E12 values
    e12 = e12_values()
    print(f"   E12 base values: {[f'{v:.1f}' for v in e12]}")

    # Show E24 values
    e24 = e24_values()
    print(f"   E24 base values: {[f'{v:.2f}' for v in e24]}")

    print("\n   Finding nearest standard values:")

    # Examples of finding nearest values
    test_values = [3333, 4700, 9100, 15432]
    for val in test_values:
        nearest12 = nearest_e12(val)
        nearest24 = nearest_e24(val)
        err12 = abs(nearest12 - val) / val * 100
        err24 = abs(nearest24 - val) / val * 100
        print(f"   {val}Ω → E12: {nearest12}Ω ({err12:.1f}%), E24: {nearest24}Ω ({err24:.1f}%)")


def demo_library_in_circuit():
    """Demonstrate using library components in a circuit."""
    print("\n5. Using Library Components in Circuits")
    print("-" * 40)

    from spicelab.core.circuit import Circuit
    from spicelab.core.components import Resistor, Vdc
    from spicelab.core.net import GND, Net

    circuit = Circuit("library_demo")

    # Use library components
    vcc = Vdc("cc", 12.0)
    r1 = Resistor("1", resistance=nearest_e24(4700))  # Use standard value
    d1 = create_component("diode.led_red", "1")

    circuit.add(vcc, r1, d1)

    vcc_net = Net("vcc")
    vled = Net("vled")

    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)
    circuit.connect(r1.ports[0], vcc_net)
    circuit.connect(r1.ports[1], vled)
    circuit.connect(d1.ports[0], vled)
    circuit.connect(d1.ports[1], GND)

    print("""
   Example: LED driver with standard resistor value

   Components:
   - Vcc: 12V supply
   - R1: E24 nearest to 4700Ω
   - D1: Red LED from library (Vf ≈ 1.8V)

   Current: I = (12V - 1.8V) / 4700Ω ≈ 2.2mA
""")
    print("   Circuit netlist:")
    print(circuit.build_netlist())


def main():
    """Demonstrate component library usage."""
    print("=" * 60)
    print("Tutorial 10: Component Library")
    print("=" * 60)

    demo_list_components()
    demo_component_specs()
    demo_create_components()
    demo_e_series()
    demo_library_in_circuit()

    print("\n" + "=" * 60)
    print("Summary: Library API")
    print("=" * 60)
    print("""
   | Function                | Purpose                          |
   |-------------------------|----------------------------------|
   | list_components()       | List all available components    |
   | get_component_spec(slug)| Get component specification      |
   | create_component(slug, ref) | Create component instance    |
   | nearest_e12(value)      | Find nearest E12 resistor        |
   | nearest_e24(value)      | Find nearest E24 resistor        |

   Naming convention:
   - category.part_number (e.g., "diode.1n4148")
   - category.subcategory.name (e.g., "behavioral.switch.ideal")

   Import:
   from spicelab.library import list_components, get_component_spec, create_component
   from spicelab.library.e_series import nearest_e12, nearest_e24
""")


if __name__ == "__main__":
    main()
