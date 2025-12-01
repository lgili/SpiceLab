"""
Circuit Validation Demo

This script demonstrates the pre-simulation validation features of PyCircuitKit.
It shows how to use the `circuit.validate()` method to detect common
errors in circuit design before running a full simulation.

The following checks are demonstrated:
1.  Missing Ground Reference: Detects circuits that don't have a connection to GND.
2.  Floating Nodes: Finds nodes that are only connected to a single component pin.
3.  Voltage Source Loops: Detects when two or more voltage sources are in parallel.
4.  Current Source Loops: Detects when two or more current sources are in series
    without a load path.
5.  Unusual Component Values: Warns about component values that are physically
    unlikely (e.g., very low resistance).
6.  Strict Mode: Shows how warnings can be treated as errors.
"""

from spicelab import Circuit, Capacitor, Idc, Net, Resistor, Vdc, GND
from spicelab.validators.circuit_validation import ValidationResult


def print_validation_report(circuit: Circuit, result: ValidationResult):
    """Prints a formatted report for a circuit's validation result."""
    print("-" * 50)
    print(f"Validating Circuit: '{circuit.name}'")
    print("-" * 50)
    print(f"Description: {circuit.__doc__ or 'No description.'}")
    print("\nValidation Result:")
    if not result.has_issues():
        print("✓ Circuit validation passed with no errors or warnings.")
    else:
        # The __str__ method of ValidationResult provides a nice report
        print(result)
    print("\n")


def main():
    """Runs demonstrations for each validation check."""
    print("=" * 60)
    print("  PyCircuitKit Circuit Validation Demonstration")
    print("=" * 60)
    print(
        "This demo shows how `circuit.validate()` catches common circuit errors "
        "before simulation.\n"
    )

    # --- Demo 1: Correctly wired circuit ---
    c_valid = Circuit("Valid RC Filter")
    c_valid.__doc__ = "A simple, correctly wired RC low-pass filter."
    v1 = Vdc("1", 5)
    r1 = Resistor("1", resistance=1e3)
    c1 = Capacitor("1", capacitance=1e-6)
    c_valid.add(v1, r1, c1)
    vin, vout = Net("vin"), Net("vout")
    c_valid.connect(v1.ports[0], vin)
    c_valid.connect(v1.ports[1], GND)
    c_valid.connect(r1.ports[0], vin)
    c_valid.connect(r1.ports[1], vout)
    c_valid.connect(c1.ports[0], vout)
    c_valid.connect(c1.ports[1], GND)
    result_valid = c_valid.validate()
    print_validation_report(c_valid, result_valid)

    # --- Demo 2: Missing Ground ---
    c_no_gnd = Circuit("Missing Ground")
    c_no_gnd.__doc__ = "This circuit has no connection to the ground node (GND or '0')."
    r1_no_gnd = Resistor("1", resistance=1000)
    c_no_gnd.add(r1_no_gnd)
    c_no_gnd.connect(r1_no_gnd.ports[0], Net("A"))
    c_no_gnd.connect(r1_no_gnd.ports[1], Net("B"))
    result_no_gnd = c_no_gnd.validate()
    print_validation_report(c_no_gnd, result_no_gnd)

    # --- Demo 3: Floating Node ---
    c_floating = Circuit("Floating Node")
    c_floating.__doc__ = (
        "The resistor R1 is connected to VCC, but its other end is only connected to a single net."
    )
    v1_floating = Vdc("1", 5)
    r1_floating = Resistor("1", resistance=1000)
    c_floating.add(v1_floating, r1_floating)
    vcc = Net("VCC")
    c_floating.connect(v1_floating.ports[0], vcc)
    c_floating.connect(v1_floating.ports[1], GND)
    c_floating.connect(r1_floating.ports[0], vcc)
    c_floating.connect(r1_floating.ports[1], Net("FLOATING_NET"))
    result_floating = c_floating.validate()
    print_validation_report(c_floating, result_floating)

    # --- Demo 4: Parallel Voltage Sources ---
    c_parallel_v = Circuit("Parallel Voltage Sources")
    c_parallel_v.__doc__ = (
        "Two voltage sources, V1 and V2, are connected in parallel, creating a conflict."
    )
    v1_parallel = Vdc("1", 5)
    v2_parallel = Vdc("2", 3.3)
    c_parallel_v.add(v1_parallel, v2_parallel)
    vcc = Net("VCC")
    c_parallel_v.connect(v1_parallel.ports[0], vcc)
    c_parallel_v.connect(v2_parallel.ports[0], vcc)
    c_parallel_v.connect(v1_parallel.ports[1], GND)
    c_parallel_v.connect(v2_parallel.ports[1], GND)
    result_parallel_v = c_parallel_v.validate()
    print_validation_report(c_parallel_v, result_parallel_v)

    # --- Demo 5: Series Current Sources ---
    c_series_i = Circuit("Series Current Sources")
    c_series_i.__doc__ = (
        "Two current sources, I1 and I2, are in series with no other path for current to flow."
    )
    i1_series = Idc("1", "1m")
    i2_series = Idc("2", "2m")
    c_series_i.add(i1_series, i2_series)
    mid_node = Net("mid_node")
    c_series_i.connect(i1_series.ports[0], mid_node)
    c_series_i.connect(i2_series.ports[1], mid_node)
    c_series_i.connect(i1_series.ports[1], GND)
    # To demonstrate the validation, we need a ground reference.
    # The other end of I2 is connected to a different net to avoid a simple loop.
    c_series_i.connect(i2_series.ports[0], Net("another_node_floating")) # This will also be a floating node
    result_series_i = c_series_i.validate()
    print_validation_report(c_series_i, result_series_i)


    # --- Demo 6: Unusual Component Values (Warning) ---
    c_warning = Circuit("Unusual Resistor Value")
    c_warning.__doc__ = (
        "This circuit contains a resistor with a very low value, which triggers a warning."
    )
    v1_warn = Vdc("1", 1)
    r1_warn = Resistor("1", resistance=0.0005)  # 0.5 milliohm, unusually low
    c_warning.add(v1_warn, r1_warn)
    n1_warn = Net("N1")
    c_warning.connect(v1_warn.ports[0], n1_warn)
    c_warning.connect(r1_warn.ports[0], n1_warn)
    c_warning.connect(v1_warn.ports[1], GND)
    c_warning.connect(r1_warn.ports[1], GND)
    result_warning = c_warning.validate()
    print_validation_report(c_warning, result_warning)

    # --- Demo 7: Strict Mode (Treating Warning as Error) ---
    c_strict = Circuit("Strict Mode Validation")
    c_strict.__doc__ = (
        "This is the same circuit as the previous one, but validated with `strict=True`."
    )
    v1_strict = Vdc("1", 1)
    r1_strict = Resistor("1", resistance=1.5e9)  # 1.5 Gigaohm, unusually high
    c_strict.add(v1_strict, r1_strict)
    n1_strict = Net("N1")
    c_strict.connect(v1_strict.ports[0], n1_strict)
    c_strict.connect(r1_strict.ports[0], n1_strict)
    c_strict.connect(v1_strict.ports[1], GND)
    c_strict.connect(r1_strict.ports[1], GND)
    result_strict = c_strict.validate(strict=True)
    print_validation_report(c_strict, result_strict)


if __name__ == "__main__":
    main()
