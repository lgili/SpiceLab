"""Interactive tutorial mode for SpiceLab.

Provides step-by-step tutorials for learning SpiceLab.
"""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, TextIO


class TutorialLevel(Enum):
    """Tutorial difficulty level."""

    BEGINNER = auto()
    INTERMEDIATE = auto()
    ADVANCED = auto()


@dataclass
class TutorialStep:
    """A single step in a tutorial.

    Attributes:
        title: Step title
        description: What this step teaches
        code: Code to demonstrate
        explanation: Explanation of the code
        exercise: Optional exercise for the user
        hints: Hints for the exercise
        validation: Function to validate user's solution
    """

    title: str
    description: str
    code: str
    explanation: str
    exercise: str = ""
    hints: list[str] = field(default_factory=list)
    validation: Callable[[Any], bool] | None = None

    def __str__(self) -> str:
        return f"{self.title}: {self.description}"


@dataclass
class Tutorial:
    """An interactive tutorial.

    Example:
        >>> from spicelab.help import Tutorial, run_tutorial
        >>>
        >>> # List available tutorials
        >>> for t in list_tutorials():
        ...     print(t)
        >>>
        >>> # Run a specific tutorial
        >>> run_tutorial("basics")
    """

    name: str
    title: str
    description: str
    level: TutorialLevel
    steps: list[TutorialStep] = field(default_factory=list)
    estimated_time: str = "10 minutes"

    def __str__(self) -> str:
        return f"[{self.level.name}] {self.title} ({self.estimated_time})"


# Built-in tutorials

TUTORIAL_BASICS = Tutorial(
    name="basics",
    title="SpiceLab Basics",
    description="Learn the fundamentals of circuit simulation with SpiceLab",
    level=TutorialLevel.BEGINNER,
    estimated_time="15 minutes",
    steps=[
        TutorialStep(
            title="Welcome to SpiceLab",
            description="Introduction to circuit simulation",
            code="",
            explanation=textwrap.dedent("""
                SpiceLab is a Python library for circuit simulation.

                In this tutorial, you'll learn:
                - How to create circuits
                - How to add components
                - How to connect components
                - How to run simulations

                Let's get started!
            """).strip(),
        ),
        TutorialStep(
            title="Creating a Circuit",
            description="Learn how to create a new circuit",
            code=textwrap.dedent("""
                from spicelab.core.circuit import Circuit

                # Create a new circuit
                circuit = Circuit("my_first_circuit")
                print(f"Created circuit: {circuit.name}")
            """).strip(),
            explanation=textwrap.dedent("""
                A Circuit is the main container for your electronic design.
                Every circuit needs a name - this becomes the title in the SPICE netlist.

                The Circuit class is imported from spicelab.core.circuit.
            """).strip(),
            exercise="Create a circuit called 'voltage_divider'",
            hints=[
                "Use Circuit('voltage_divider')",
                "Don't forget to import Circuit first",
            ],
        ),
        TutorialStep(
            title="Adding Components",
            description="Learn how to add resistors and other components",
            code=textwrap.dedent("""
                from spicelab.core.components import Resistor, Capacitor

                # Create components
                R1 = Resistor(ref="1", resistance=10_000)  # 10kΩ
                R2 = Resistor(ref="2", resistance=20_000)  # 20kΩ
                C1 = Capacitor(ref="1", capacitance=100e-9)  # 100nF

                # Add to circuit
                circuit.add(R1, R2, C1)
                print(f"Components: {len(circuit._components)}")
            """).strip(),
            explanation=textwrap.dedent("""
                Components are created with a reference designator (ref) and value.

                - Resistor: ref and resistance in Ohms
                - Capacitor: ref and capacitance in Farads
                - You can use engineering notation: 10_000, 1e3, etc.

                The add() method accepts one or more components.
            """).strip(),
            exercise="Add a 4.7kΩ resistor and a 1µF capacitor to your circuit",
            hints=[
                "R = Resistor(ref='1', resistance=4700)",
                "C = Capacitor(ref='1', capacitance=1e-6)",
                "1µF = 1e-6 Farads",
            ],
        ),
        TutorialStep(
            title="Creating Nets",
            description="Learn about nets and the ground reference",
            code=textwrap.dedent("""
                from spicelab.core.net import Net, GND

                # Create named nets
                vin = Net("vin")    # Input voltage node
                vout = Net("vout")  # Output voltage node

                # GND is the global ground reference (node 0)
                print(f"Ground node: {GND.name}")
            """).strip(),
            explanation=textwrap.dedent("""
                Nets represent electrical connection points (nodes).

                - Net("name") creates a named node
                - GND is the global ground reference (always node 0)
                - Every circuit needs at least one connection to GND

                Nets are used to connect component ports together.
            """).strip(),
        ),
        TutorialStep(
            title="Connecting Components",
            description="Learn how to wire up your circuit",
            code=textwrap.dedent("""
                # Connect R1 between vin and vout
                circuit.connect(R1.ports[0], vin)
                circuit.connect(R1.ports[1], vout)

                # Connect R2 between vout and ground
                circuit.connect(R2.ports[0], vout)
                circuit.connect(R2.ports[1], GND)

                # View the circuit
                print(circuit.summary())
            """).strip(),
            explanation=textwrap.dedent("""
                Components have ports that represent their terminals.

                - Resistors have 2 ports: ports[0] and ports[1]
                - connect(port, net) links a port to a net
                - Two ports connected to the same net are electrically connected

                This creates a voltage divider: vin -> R1 -> vout -> R2 -> GND
            """).strip(),
            exercise="Create a complete RC lowpass filter circuit",
            hints=[
                "R between vin and vout",
                "C between vout and GND",
                "Don't forget the voltage source!",
            ],
        ),
        TutorialStep(
            title="Generating a Netlist",
            description="Convert your circuit to SPICE format",
            code=textwrap.dedent("""
                # Generate the SPICE netlist
                netlist = circuit.build_netlist()
                print(netlist)

                # Or use preview for formatted output
                print(circuit.preview_netlist())
            """).strip(),
            explanation=textwrap.dedent("""
                build_netlist() generates a standard SPICE netlist.
                preview_netlist() adds formatting and syntax highlighting.

                The netlist contains:
                - Circuit title
                - Component definitions
                - Control directives
                - .end statement
            """).strip(),
        ),
        TutorialStep(
            title="Validating Your Circuit",
            description="Check for common errors before simulation",
            code=textwrap.dedent("""
                # Validate the circuit
                result = circuit.validate()

                if result.is_valid:
                    print("✓ Circuit is valid!")
                else:
                    print("✗ Errors found:")
                    for error in result.errors:
                        print(f"  - {error}")

                if result.warnings:
                    print("Warnings:")
                    for warning in result.warnings:
                        print(f"  - {warning}")
            """).strip(),
            explanation=textwrap.dedent("""
                validate() checks for common circuit errors:

                - Floating nodes (unconnected terminals)
                - Missing ground reference
                - Voltage source loops
                - Component value issues

                Always validate before running a simulation!
            """).strip(),
        ),
        TutorialStep(
            title="Congratulations!",
            description="You've completed the basics tutorial",
            code="",
            explanation=textwrap.dedent("""
                You've learned the fundamentals of SpiceLab:

                ✓ Creating circuits
                ✓ Adding components
                ✓ Creating and connecting nets
                ✓ Generating netlists
                ✓ Validating circuits

                Next steps:
                - Try the 'simulation' tutorial for running analyses
                - Explore the 'templates' tutorial for pre-built circuits
                - Check out the API cheat sheet for quick reference
            """).strip(),
        ),
    ],
)

TUTORIAL_SIMULATION = Tutorial(
    name="simulation",
    title="Running Simulations",
    description="Learn how to run AC, DC, and transient analyses",
    level=TutorialLevel.BEGINNER,
    estimated_time="20 minutes",
    steps=[
        TutorialStep(
            title="Simulation Overview",
            description="Introduction to SPICE analyses",
            code="",
            explanation=textwrap.dedent("""
                SpiceLab supports several analysis types:

                - OP: DC operating point
                - AC: Frequency response (Bode plots)
                - TRAN: Time-domain (transient) analysis
                - DC: DC sweep analysis
                - NOISE: Noise analysis

                Each analysis type reveals different circuit behaviors.
            """).strip(),
        ),
        TutorialStep(
            title="Using Templates",
            description="Start with a pre-built circuit",
            code=textwrap.dedent("""
                from spicelab.templates import rc_lowpass

                # Create an RC lowpass filter with 1kHz cutoff
                circuit = rc_lowpass(fc=1000)
                print(circuit.summary())
            """).strip(),
            explanation=textwrap.dedent("""
                Templates provide ready-to-use circuits.

                rc_lowpass(fc) creates a lowpass filter with:
                - Cutoff frequency fc
                - Automatically calculated R and C values
                - Proper connections and voltage source

                Templates are great for learning and prototyping.
            """).strip(),
        ),
        TutorialStep(
            title="Quick AC Analysis",
            description="Run a frequency sweep",
            code=textwrap.dedent("""
                from spicelab.shortcuts.simulation import quick_ac

                # Run AC analysis from 1Hz to 1MHz
                # result = quick_ac(circuit, start=1, stop=1e6)

                # The result contains frequency response data
                # ds = result.dataset()
                # print(ds.data_vars)
            """).strip(),
            explanation=textwrap.dedent("""
                quick_ac() runs a frequency sweep analysis.

                Parameters:
                - start: Start frequency in Hz
                - stop: Stop frequency in Hz
                - points: Points per decade (default: 20)

                Results include magnitude and phase vs frequency.
            """).strip(),
        ),
        TutorialStep(
            title="Quick Transient Analysis",
            description="Run a time-domain simulation",
            code=textwrap.dedent("""
                from spicelab.shortcuts.simulation import quick_tran

                # Run transient analysis for 10ms
                # result = quick_tran(circuit, duration="10ms")

                # Or with explicit timestep
                # result = quick_tran(circuit, duration="10ms", step="10us")
            """).strip(),
            explanation=textwrap.dedent("""
                quick_tran() runs a time-domain simulation.

                Parameters:
                - duration: Simulation duration (e.g., "1ms", "100us")
                - step: Maximum timestep (optional, auto-calculated)

                Great for seeing transient response, oscillations, etc.
            """).strip(),
        ),
        TutorialStep(
            title="Dry Run Mode",
            description="Validate without simulating",
            code=textwrap.dedent("""
                from spicelab.debug import dry_run

                # Validate simulation setup
                result = dry_run(circuit)
                print(result)

                if result.valid:
                    print("Ready to simulate!")
                else:
                    print("Fix these issues first:")
                    for error in result.errors:
                        print(f"  - {error}")
            """).strip(),
            explanation=textwrap.dedent("""
                dry_run() validates the complete simulation setup:

                - Circuit structure
                - Analysis parameters
                - Engine compatibility

                Use this before long simulations to catch errors early.
            """).strip(),
        ),
        TutorialStep(
            title="Verbose Mode",
            description="See what's happening during simulation",
            code=textwrap.dedent("""
                from spicelab.debug import VerboseSimulation

                # Run with verbose output
                # with VerboseSimulation():
                #     result = quick_ac(circuit, start=1, stop=1e6)

                # Shows progress, timing, and intermediate steps
            """).strip(),
            explanation=textwrap.dedent("""
                VerboseSimulation shows detailed progress:

                - Circuit validation steps
                - Netlist generation
                - Analysis configuration
                - Engine output
                - Timing information

                Very helpful for debugging simulation issues.
            """).strip(),
        ),
    ],
)

TUTORIAL_TEMPLATES = Tutorial(
    name="templates",
    title="Using Circuit Templates",
    description="Learn to use pre-built circuit templates",
    level=TutorialLevel.BEGINNER,
    estimated_time="10 minutes",
    steps=[
        TutorialStep(
            title="Available Templates",
            description="Overview of built-in templates",
            code=textwrap.dedent("""
                from spicelab import templates

                # Filter templates
                # rc_lowpass(fc)      - RC lowpass filter
                # rc_highpass(fc)     - RC highpass filter
                # rl_lowpass(fc)      - RL lowpass filter

                # Amplifier templates
                # voltage_divider(ratio)
                # inverting_amp(gain)
                # non_inverting_amp(gain)
            """).strip(),
            explanation=textwrap.dedent("""
                Templates are factory functions that create complete circuits.

                Each template:
                - Creates all necessary components
                - Makes proper connections
                - Adds voltage/current sources
                - Returns a ready-to-simulate Circuit
            """).strip(),
        ),
        TutorialStep(
            title="Filter Templates",
            description="Using filter circuit templates",
            code=textwrap.dedent("""
                from spicelab.templates import rc_lowpass, rc_highpass

                # Create filters with 1kHz cutoff
                lpf = rc_lowpass(fc=1000)
                hpf = rc_highpass(fc=1000)

                print(f"Lowpass: {lpf.name}")
                print(f"Highpass: {hpf.name}")
            """).strip(),
            explanation=textwrap.dedent("""
                Filter templates calculate component values automatically.

                For rc_lowpass(fc=1000):
                - R and C are calculated so fc = 1/(2*pi*R*C) = 1000Hz
                - Includes AC voltage source
                - Output node is labeled 'vout'
            """).strip(),
        ),
        TutorialStep(
            title="Voltage Divider",
            description="Create a voltage divider",
            code=textwrap.dedent("""
                from spicelab.templates import voltage_divider

                # Create a 2:1 voltage divider
                vdiv = voltage_divider(ratio=0.5)
                print(vdiv.preview_netlist())
            """).strip(),
            explanation=textwrap.dedent("""
                voltage_divider(ratio) creates a resistive divider.

                ratio = Vout/Vin
                - ratio=0.5 gives equal resistors (Vout = Vin/2)
                - ratio=0.1 gives 9:1 ratio
            """).strip(),
        ),
    ],
)

# Registry of all tutorials
TUTORIALS: dict[str, Tutorial] = {
    "basics": TUTORIAL_BASICS,
    "simulation": TUTORIAL_SIMULATION,
    "templates": TUTORIAL_TEMPLATES,
}


def list_tutorials() -> list[Tutorial]:
    """List all available tutorials.

    Returns:
        List of Tutorial objects

    Example:
        >>> from spicelab.help import list_tutorials
        >>> for tutorial in list_tutorials():
        ...     print(tutorial)
    """
    return list(TUTORIALS.values())


def get_tutorial(name: str) -> Tutorial | None:
    """Get a tutorial by name.

    Args:
        name: Tutorial name

    Returns:
        Tutorial or None if not found
    """
    return TUTORIALS.get(name)


class TutorialRunner:
    """Interactive tutorial runner.

    Guides users through tutorial steps interactively.
    """

    def __init__(
        self,
        tutorial: Tutorial,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        """Initialize tutorial runner.

        Args:
            tutorial: Tutorial to run
            input_stream: Input stream (default: stdin)
            output_stream: Output stream (default: stdout)
        """
        self._tutorial = tutorial
        self._input = input_stream or sys.stdin
        self._output = output_stream or sys.stdout
        self._current_step = 0
        self._context: dict[str, Any] = {}

    def _print(self, text: str = "") -> None:
        """Print to output stream."""
        self._output.write(text + "\n")
        self._output.flush()

    def _wait_for_enter(self) -> None:
        """Wait for user to press Enter."""
        self._print("\nPress Enter to continue...")
        try:
            self._input.readline()
        except EOFError:
            pass

    def _show_step(self, step: TutorialStep) -> None:
        """Display a tutorial step."""
        self._print("\n" + "=" * 60)
        self._print(f"Step {self._current_step + 1}: {step.title}")
        self._print("=" * 60)

        self._print(f"\n{step.description}")
        self._print("-" * 40)

        if step.explanation:
            self._print(f"\n{step.explanation}")

        if step.code:
            self._print("\n📝 Code:")
            self._print("-" * 40)
            for line in step.code.split("\n"):
                self._print(f"  {line}")
            self._print("-" * 40)

        if step.exercise:
            self._print(f"\n💪 Exercise: {step.exercise}")
            if step.hints:
                self._print("\nHints:")
                for i, hint in enumerate(step.hints, 1):
                    self._print(f"  {i}. {hint}")

    def run(self) -> None:
        """Run the tutorial interactively."""
        tutorial = self._tutorial

        # Header
        self._print("\n" + "=" * 60)
        self._print(f"📚 Tutorial: {tutorial.title}")
        self._print("=" * 60)
        self._print(f"\n{tutorial.description}")
        self._print(f"\nLevel: {tutorial.level.name}")
        self._print(f"Estimated time: {tutorial.estimated_time}")
        self._print(f"Steps: {len(tutorial.steps)}")

        self._wait_for_enter()

        # Run each step
        for i, step in enumerate(tutorial.steps):
            self._current_step = i
            self._show_step(step)
            self._wait_for_enter()

        # Completion
        self._print("\n" + "=" * 60)
        self._print("🎉 Tutorial Complete!")
        self._print("=" * 60)
        self._print(f"\nYou've completed: {tutorial.title}")
        self._print("\nNext tutorials to try:")
        for t in list_tutorials():
            if t.name != tutorial.name:
                self._print(f"  - {t.name}: {t.title}")


def run_tutorial(
    name: str = "basics",
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> None:
    """Run an interactive tutorial.

    Args:
        name: Tutorial name (default: "basics")
        input_stream: Input stream (default: stdin)
        output_stream: Output stream (default: stdout)

    Example:
        >>> from spicelab.help import run_tutorial
        >>> run_tutorial("basics")  # Interactive tutorial
        >>> run_tutorial("simulation")  # Simulation tutorial
    """
    tutorial = get_tutorial(name)
    if tutorial is None:
        print(f"Tutorial '{name}' not found.")
        print("Available tutorials:")
        for t in list_tutorials():
            print(f"  - {t.name}: {t.title}")
        return

    runner = TutorialRunner(tutorial, input_stream, output_stream)
    runner.run()


__all__ = [
    "Tutorial",
    "TutorialStep",
    "TutorialLevel",
    "TutorialRunner",
    "list_tutorials",
    "get_tutorial",
    "run_tutorial",
    "TUTORIALS",
]
