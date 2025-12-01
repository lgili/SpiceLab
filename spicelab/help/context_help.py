"""Context-sensitive help system.

Provides interactive help for circuits, components, and results.
"""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.circuit import Circuit
    from ..core.components import Component


@dataclass
class HelpSection:
    """A section of help content."""

    title: str
    content: str
    examples: list[str] = field(default_factory=list)
    see_also: list[str] = field(default_factory=list)


@dataclass
class Help:
    """Base help class with common functionality.

    Provides context-sensitive help for SpiceLab objects.

    Example:
        >>> from spicelab.help import get_help
        >>> from spicelab.core.circuit import Circuit
        >>>
        >>> circuit = Circuit("test")
        >>> help_obj = get_help(circuit)
        >>> print(help_obj.summary())
    """

    obj: Any
    sections: list[HelpSection] = field(default_factory=list)

    def summary(self) -> str:
        """Get a brief summary of the object."""
        doc = inspect.getdoc(type(self.obj)) or "No documentation available."
        return doc.split("\n\n")[0]

    def full(self) -> str:
        """Get full help text."""
        lines = [
            f"Help for {type(self.obj).__name__}",
            "=" * 60,
            "",
            self.summary(),
            "",
        ]

        for section in self.sections:
            lines.append(f"\n{section.title}")
            lines.append("-" * len(section.title))
            lines.append(section.content)

            if section.examples:
                lines.append("\nExamples:")
                for ex in section.examples:
                    lines.append(textwrap.indent(ex, "    "))

            if section.see_also:
                lines.append(f"\nSee also: {', '.join(section.see_also)}")

        return "\n".join(lines)

    def methods(self) -> list[str]:
        """List available methods."""
        return [
            name
            for name in dir(self.obj)
            if not name.startswith("_") and callable(getattr(self.obj, name, None))
        ]

    def attributes(self) -> list[str]:
        """List available attributes."""
        return [
            name
            for name in dir(self.obj)
            if not name.startswith("_") and not callable(getattr(self.obj, name, None))
        ]

    def method_help(self, method_name: str) -> str:
        """Get help for a specific method."""
        method = getattr(self.obj, method_name, None)
        if method is None:
            return f"Method '{method_name}' not found."

        doc = inspect.getdoc(method) or "No documentation available."
        sig = ""
        try:
            sig = str(inspect.signature(method))
        except (ValueError, TypeError):
            pass

        return f"{method_name}{sig}\n\n{doc}"

    def __str__(self) -> str:
        return self.full()


class CircuitHelp(Help):
    """Context-sensitive help for Circuit objects.

    Provides help specific to circuit design and simulation.

    Example:
        >>> from spicelab.core.circuit import Circuit
        >>> from spicelab.help import CircuitHelp
        >>>
        >>> circuit = Circuit("my_circuit")
        >>> help_obj = CircuitHelp(circuit)
        >>> print(help_obj.quick_start())
    """

    def __init__(self, circuit: Circuit) -> None:
        """Initialize circuit help.

        Args:
            circuit: Circuit to provide help for
        """
        super().__init__(obj=circuit)
        self._build_sections()

    def _build_sections(self) -> None:
        """Build help sections for circuit."""
        self.sections = [
            HelpSection(
                title="Quick Start",
                content=self._quick_start_content(),
                examples=[
                    "# Add a component\n"
                    "from spicelab.core.components import Resistor\n"
                    "R1 = Resistor(ref='1', resistance=1000)\n"
                    "circuit.add(R1)",
                    "# Connect components\n"
                    "from spicelab.core.net import Net, GND\n"
                    "vin = Net('vin')\n"
                    "circuit.connect(R1.ports[0], vin)\n"
                    "circuit.connect(R1.ports[1], GND)",
                ],
                see_also=["Component", "Net", "simulate"],
            ),
            HelpSection(
                title="Common Operations",
                content=self._common_ops_content(),
                examples=[
                    "# Generate netlist\n" "print(circuit.build_netlist())",
                    "# Preview with formatting\n" "print(circuit.preview_netlist())",
                    "# Validate circuit\n" "result = circuit.validate()",
                ],
            ),
            HelpSection(
                title="Current State",
                content=self._state_content(),
            ),
        ]

    def _quick_start_content(self) -> str:
        """Generate quick start content."""
        return textwrap.dedent("""
            A Circuit is a container for components and their connections.

            Basic workflow:
            1. Create components (Resistor, Capacitor, etc.)
            2. Add components to circuit with circuit.add()
            3. Create nets for connection points
            4. Connect component ports to nets with circuit.connect()
            5. Generate netlist or run simulation
        """).strip()

    def _common_ops_content(self) -> str:
        """Generate common operations content."""
        return textwrap.dedent("""
            - add(*components): Add one or more components
            - connect(port, net): Connect a port to a net
            - build_netlist(): Generate SPICE netlist
            - preview_netlist(): Formatted netlist preview
            - validate(): Check for circuit errors
            - summary(): Human-readable summary
            - save_netlist(path): Save netlist to file
        """).strip()

    def _state_content(self) -> str:
        """Generate current state content."""
        circuit = self.obj
        return textwrap.dedent(f"""
            Circuit: {circuit.name}
            Components: {len(circuit._components)}
            Directives: {len(circuit._directives)}
        """).strip()

    def quick_start(self) -> str:
        """Get quick start guide."""
        section = self.sections[0]
        lines = [section.title, "-" * len(section.title), section.content]
        if section.examples:
            lines.append("\nExamples:")
            for ex in section.examples:
                lines.append(textwrap.indent(ex, "    "))
        return "\n".join(lines)

    def components_help(self) -> str:
        """Get help about current components."""
        circuit = self.obj
        if not circuit._components:
            return "No components in circuit. Use circuit.add() to add components."

        lines = ["Components in circuit:", ""]
        for comp in circuit._components:
            comp_type = type(comp).__name__
            lines.append(f"  {comp.ref} ({comp_type})")
            for port in comp.ports:
                net = circuit._port_to_net.get(port)
                net_name = getattr(net, "name", None) or "<unconnected>"
                lines.append(f"    - {port.name} -> {net_name}")

        return "\n".join(lines)

    def validation_help(self) -> str:
        """Get help about circuit validation."""
        circuit = self.obj
        result = circuit.validate()

        lines = ["Circuit Validation:", ""]

        if result.is_valid:
            lines.append("✓ Circuit is valid")
        else:
            lines.append("✗ Circuit has errors:")
            for error in result.errors:
                lines.append(f"  - {error}")

        if result.warnings:
            lines.append("\nWarnings:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")

        lines.extend(
            [
                "",
                "Common issues:",
                "  - Floating nodes: Connect all component terminals",
                "  - No ground: Add at least one connection to GND",
                "  - Voltage source loop: Don't connect voltage sources in parallel",
            ]
        )

        return "\n".join(lines)


class ResultHelp(Help):
    """Context-sensitive help for simulation results.

    Example:
        >>> from spicelab.help import ResultHelp
        >>> help_obj = ResultHelp(result)
        >>> print(help_obj.quick_start())
    """

    def __init__(self, result: Any) -> None:
        """Initialize result help.

        Args:
            result: Simulation result object
        """
        super().__init__(obj=result)
        self._build_sections()

    def _build_sections(self) -> None:
        """Build help sections for result."""
        self.sections = [
            HelpSection(
                title="Accessing Data",
                content=self._data_access_content(),
                examples=[
                    "# Get as xarray Dataset\n" "ds = result.dataset()\n" "print(ds.data_vars)",
                    "# Get specific variable\n" "vout = ds['V(vout)']",
                    "# Convert to pandas\n" "df = ds.to_dataframe()",
                ],
            ),
            HelpSection(
                title="Measurements",
                content=self._measurements_content(),
                examples=[
                    "# Measure bandwidth\n" "bw = result.bw('V(vout)')",
                    "# Measure phase margin\n" "pm = result.pm('V(vout)')",
                ],
                see_also=["Measure", "plot"],
            ),
            HelpSection(
                title="Plotting",
                content=self._plotting_content(),
                examples=[
                    "# Auto-plot based on analysis type\n" "result.plot()",
                    "# Bode plot\n" "result.bode('V(vout)')",
                ],
            ),
        ]

    def _data_access_content(self) -> str:
        """Generate data access content."""
        return textwrap.dedent("""
            Simulation results are stored as xarray Dataset.

            Methods:
            - dataset(): Get the xarray Dataset
            - to_dataframe(): Convert to pandas DataFrame
            - variables: List of available variables
        """).strip()

    def _measurements_content(self) -> str:
        """Generate measurements content."""
        return textwrap.dedent("""
            Built-in measurement functions:

            AC Analysis:
            - bw(signal): -3dB bandwidth
            - pm(signal): Phase margin
            - gm(signal): Gain margin
            - gain(signal): DC gain

            Transient Analysis:
            - rise_time(signal): 10%-90% rise time
            - fall_time(signal): 90%-10% fall time
            - overshoot(signal): Percent overshoot
            - settling_time(signal): Time to settle
        """).strip()

    def _plotting_content(self) -> str:
        """Generate plotting content."""
        return textwrap.dedent("""
            Plotting methods:

            - plot(): Auto-detect best plot type
            - bode(signal): Magnitude and phase vs frequency
            - transient(signal): Time-domain plot
            - nyquist(signal): Nyquist diagram
        """).strip()

    def variables_help(self) -> str:
        """Get help about available variables."""
        try:
            ds = self.obj.dataset()
            lines = ["Available variables:", ""]
            for var in ds.data_vars:
                lines.append(f"  - {var}")
            lines.extend(
                [
                    "",
                    "Coordinates:",
                ]
            )
            for coord in ds.coords:
                lines.append(f"  - {coord}: {len(ds.coords[coord])} points")
            return "\n".join(lines)
        except Exception:
            return "Unable to access result data."


class ComponentHelp(Help):
    """Context-sensitive help for components.

    Example:
        >>> from spicelab.core.components import Resistor
        >>> from spicelab.help import ComponentHelp
        >>>
        >>> R1 = Resistor(ref='1', resistance=1000)
        >>> help_obj = ComponentHelp(R1)
        >>> print(help_obj.quick_start())
    """

    def __init__(self, component: Component) -> None:
        """Initialize component help.

        Args:
            component: Component to provide help for
        """
        super().__init__(obj=component)
        self._build_sections()

    def _build_sections(self) -> None:
        """Build help sections for component."""
        comp = self.obj

        self.sections = [
            HelpSection(
                title="Component Info",
                content=self._info_content(),
            ),
            HelpSection(
                title="Ports",
                content=self._ports_content(),
                examples=[
                    f"# Access ports\n"
                    f"port_a = {comp.ref}.ports[0]\n"
                    f"port_b = {comp.ref}.ports[1]",
                    f"# Connect to net\n" f"circuit.connect({comp.ref}.ports[0], my_net)",
                ],
            ),
            HelpSection(
                title="SPICE Syntax",
                content=self._spice_content(),
            ),
        ]

    def _info_content(self) -> str:
        """Generate info content."""
        comp = self.obj
        comp_type = type(comp).__name__

        lines = [f"Type: {comp_type}", f"Reference: {comp.ref}"]

        # Add component-specific info
        if hasattr(comp, "resistance"):
            lines.append(f"Resistance: {comp.resistance}")
        if hasattr(comp, "capacitance"):
            lines.append(f"Capacitance: {comp.capacitance}")
        if hasattr(comp, "inductance"):
            lines.append(f"Inductance: {comp.inductance}")
        if hasattr(comp, "value"):
            lines.append(f"Value: {comp.value}")

        return "\n".join(lines)

    def _ports_content(self) -> str:
        """Generate ports content."""
        comp = self.obj
        lines = [f"Number of ports: {len(comp.ports)}", ""]

        for i, port in enumerate(comp.ports):
            lines.append(f"  [{i}] {port.name}")

        return "\n".join(lines)

    def _spice_content(self) -> str:
        """Generate SPICE syntax content."""
        comp = self.obj
        comp_type = type(comp).__name__

        syntax_map = {
            "Resistor": "Rname node1 node2 value",
            "Capacitor": "Cname node1 node2 value",
            "Inductor": "Lname node1 node2 value",
            "Vdc": "Vname node+ node- DC value",
            "Vac": "Vname node+ node- AC magnitude [phase]",
            "Diode": "Dname anode cathode model",
        }

        syntax = syntax_map.get(comp_type, "See SPICE documentation")
        return f"SPICE syntax: {syntax}"


def get_help(obj: Any) -> Help:
    """Get context-sensitive help for an object.

    Automatically determines the appropriate help class based on
    the object type.

    Args:
        obj: Object to get help for

    Returns:
        Appropriate Help subclass instance

    Example:
        >>> from spicelab.help import get_help
        >>> from spicelab.core.circuit import Circuit
        >>>
        >>> circuit = Circuit("test")
        >>> help_obj = get_help(circuit)
        >>> print(help_obj.summary())
    """
    # Import here to avoid circular imports
    from ..core.circuit import Circuit
    from ..core.components import Component

    if isinstance(obj, Circuit):
        return CircuitHelp(obj)
    elif isinstance(obj, Component):
        return ComponentHelp(obj)
    elif hasattr(obj, "dataset"):  # Result-like object
        return ResultHelp(obj)
    else:
        return Help(obj)


def show_help(obj: Any, section: str | None = None) -> None:
    """Print help for an object.

    Args:
        obj: Object to show help for
        section: Specific section to show (optional)

    Example:
        >>> from spicelab.help import show_help
        >>> from spicelab.core.circuit import Circuit
        >>>
        >>> circuit = Circuit("test")
        >>> show_help(circuit)
        >>> show_help(circuit, "quick_start")
    """
    help_obj = get_help(obj)

    if section:
        method = getattr(help_obj, section, None)
        if method and callable(method):
            print(method())
        else:
            print(f"Unknown section: {section}")
            print(f"Available sections: {help_obj.methods()}")
    else:
        print(help_obj.full())


__all__ = [
    "Help",
    "HelpSection",
    "CircuitHelp",
    "ResultHelp",
    "ComponentHelp",
    "get_help",
    "show_help",
]
