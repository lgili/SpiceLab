"""Example: Report Generator Plugin

This example demonstrates how to use the ReportGeneratorPlugin to
automatically generate simulation reports in various formats.

The plugin hooks into POST_SIMULATION to auto-generate reports,
or you can manually generate reports for specific simulations.
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec
from spicelab.plugins.examples import ReportGeneratorPlugin


def create_rc_circuit() -> Circuit:
    """Create a simple RC low-pass filter for demonstration."""
    circuit = Circuit("rc_lowpass_demo")

    # Components
    vin = Vdc("Vin", 5.0)
    r1 = Resistor("R1", "1k")
    c1 = Capacitor("C1", "1u")

    # Add to circuit
    circuit.add(vin, r1, c1)

    # Connect
    circuit.connect(vin.ports[0], Net("input"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r1.ports[0], Net("input"))
    circuit.connect(r1.ports[1], Net("output"))
    circuit.connect(c1.ports[0], Net("output"))
    circuit.connect(c1.ports[1], GND)

    return circuit


def example_manual_report_generation():
    """Generate reports manually for a circuit."""
    print("=" * 60)
    print("Example 1: Manual Report Generation")
    print("=" * 60)

    # Create the plugin
    plugin = ReportGeneratorPlugin()

    # Configure it
    plugin.configure({
        "output_dir": "./reports",
        "format": "markdown",
        "include_netlist": True,
    })

    # Create circuit
    circuit = create_rc_circuit()

    # Create a mock result (in real usage, this comes from simulation)
    class MockResult:
        traces = {"v(input)": [5.0], "v(output)": [0.0, 2.5, 4.0, 4.8, 5.0]}

    result = MockResult()

    # Generate reports in different formats
    print("\nGenerating Markdown report...")
    md_path = plugin.generate_report(circuit, result, format="markdown")
    print(f"  -> {md_path}")

    print("\nGenerating HTML report...")
    html_path = plugin.generate_report(circuit, result, format="html")
    print(f"  -> {html_path}")

    print("\nGenerating JSON report...")
    json_path = plugin.generate_report(circuit, result, format="json")
    print(f"  -> {json_path}")

    # Show sample content
    print("\n" + "-" * 40)
    print("Sample Markdown content:")
    print("-" * 40)
    print(md_path.read_text()[:500] + "...")


def example_auto_report_with_hooks():
    """Demonstrate auto-report generation via hooks."""
    print("\n" + "=" * 60)
    print("Example 2: Auto-Report Generation with Hooks")
    print("=" * 60)

    from spicelab.plugins.hooks import HookManager, HookType

    # Create and configure plugin
    plugin = ReportGeneratorPlugin()
    plugin.configure({
        "output_dir": "./reports/auto",
        "format": "all",  # Generate all formats
        "include_netlist": True,
    })

    # Activate the plugin (registers hooks)
    plugin.activate()

    # Now simulate the hook being called (in real usage, this happens automatically)
    circuit = create_rc_circuit()

    class MockResult:
        traces = {"v(output)": [0, 1, 2, 3, 4, 5]}

    # Trigger hooks manually for demonstration
    hook_manager = HookManager.get_instance()

    print("\nSimulating PRE_SIMULATION hook...")
    hook_manager.trigger(HookType.PRE_SIMULATION, circuit=circuit, analyses=[])

    print("Simulating POST_SIMULATION hook (reports generated here)...")
    hook_manager.trigger(HookType.POST_SIMULATION, circuit=circuit, result=MockResult())

    print("\nReports have been auto-generated in ./reports/auto/")

    # Deactivate
    plugin.deactivate()


def example_custom_report_config():
    """Show various configuration options."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Report Configuration")
    print("=" * 60)

    plugin = ReportGeneratorPlugin()

    # Configuration options explained
    configs = {
        "Basic (Markdown only)": {
            "output_dir": "./reports/basic",
            "format": "markdown",
            "include_netlist": False,
        },
        "Full (All formats with netlist)": {
            "output_dir": "./reports/full",
            "format": "all",
            "include_netlist": True,
            "include_raw_data": True,
        },
        "HTML for documentation": {
            "output_dir": "./docs/reports",
            "format": "html",
            "include_netlist": True,
        },
        "JSON for automation": {
            "output_dir": "./data/reports",
            "format": "json",
            "include_raw_data": True,
        },
    }

    for name, config in configs.items():
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    example_manual_report_generation()
    example_auto_report_with_hooks()
    example_custom_report_config()

    print("\n" + "=" * 60)
    print("Report Generator Plugin Examples Complete!")
    print("=" * 60)
