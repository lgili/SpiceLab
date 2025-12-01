"""Docstring example validation and execution.

Provides tools to validate and run examples from docstrings.
"""

from __future__ import annotations

import doctest
import importlib
import inspect
import io
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExampleResult:
    """Result of running a docstring example.

    Attributes:
        name: Name of the function/class/module
        source: The example source code
        success: Whether it ran without errors
        output: Captured output
        error: Error message if failed
        line_number: Line number in the source file
    """

    name: str
    source: str
    success: bool
    output: str = ""
    error: str = ""
    line_number: int | None = None

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        result = f"{status} {self.name}"
        if self.line_number:
            result += f" (line {self.line_number})"
        if self.error:
            result += f"\n  Error: {self.error}"
        return result


@dataclass
class ValidationReport:
    """Report from validating docstring examples.

    Attributes:
        module_name: Name of the module validated
        results: List of example results
        total: Total number of examples
        passed: Number of passing examples
        failed: Number of failing examples
    """

    module_name: str
    results: list[ExampleResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of examples."""
        return len(self.results)

    @property
    def passed(self) -> int:
        """Number of passing examples."""
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        """Number of failing examples."""
        return sum(1 for r in self.results if not r.success)

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total == 0:
            return 100.0
        return (self.passed / self.total) * 100

    def summary(self) -> str:
        """Get a summary of the validation results."""
        return (
            f"Module: {self.module_name}\n"
            f"Total: {self.total}, Passed: {self.passed}, Failed: {self.failed}\n"
            f"Success rate: {self.success_rate:.1f}%"
        )

    def __str__(self) -> str:
        lines = [self.summary(), ""]
        for result in self.results:
            lines.append(str(result))
        return "\n".join(lines)


def validate_docstring_examples(
    module: str | Any,
    verbose: bool = False,
    skip_patterns: list[str] | None = None,
) -> ValidationReport:
    """Validate all docstring examples in a module.

    Uses doctest to run examples found in docstrings and reports
    which ones pass or fail.

    Args:
        module: Module name (str) or module object
        verbose: Print detailed output
        skip_patterns: List of patterns to skip (e.g., ["_private"])

    Returns:
        ValidationReport with results

    Example:
        >>> from spicelab.help import validate_docstring_examples
        >>> report = validate_docstring_examples("spicelab.core.circuit")
        >>> print(report.summary())  # doctest: +SKIP
    """
    if isinstance(module, str):
        module_str = module
        try:
            module = importlib.import_module(module)
        except ImportError as e:
            return ValidationReport(
                module_name=module_str,
                results=[
                    ExampleResult(
                        name=module_str,
                        source="",
                        success=False,
                        error=f"Could not import module: {e}",
                    )
                ],
            )

    module_name = getattr(module, "__name__", str(module))
    skip_patterns = skip_patterns or ["_"]

    results: list[ExampleResult] = []

    # Get all objects with docstrings
    for name, obj in inspect.getmembers(module):
        # Skip based on patterns
        if any(name.startswith(p) for p in skip_patterns):
            continue

        # Skip imported items
        if hasattr(obj, "__module__") and obj.__module__ != module_name:
            continue

        doc = inspect.getdoc(obj)
        if not doc or ">>>" not in doc:
            continue

        # Run doctest on this object
        result = _run_doctest(name, obj, doc, verbose)
        if result:
            results.append(result)

    return ValidationReport(module_name=module_name, results=results)


def _run_doctest(
    name: str,
    obj: Any,
    doc: str,
    verbose: bool,
) -> ExampleResult | None:
    """Run doctest on a single object."""
    # Create a test finder and runner
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(verbose=verbose, optionflags=doctest.ELLIPSIS)

    # Capture output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        # Get tests from the object
        tests = finder.find(obj, name)
        if not tests:
            return None

        # Run all tests
        total_failures = 0
        total_tests = 0
        for test in tests:
            if test.examples:
                runner.run(test)
                summary = runner.summarize(verbose=False)
                total_failures += summary if isinstance(summary, int) else 0
                total_tests += len(test.examples)

        if total_tests == 0:
            return None

        output = sys.stdout.getvalue()
        error_output = sys.stderr.getvalue()

        # Get line number
        try:
            source_lines, line_number = inspect.getsourcelines(obj)
        except (OSError, TypeError):
            line_number = None

        return ExampleResult(
            name=name,
            source=doc,
            success=(total_failures == 0),
            output=output,
            error=error_output if total_failures > 0 else "",
            line_number=line_number,
        )

    except Exception as e:
        return ExampleResult(
            name=name,
            source=doc,
            success=False,
            error=str(e),
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def run_example(
    example_code: str,
    globals_dict: dict[str, Any] | None = None,
    capture_output: bool = True,
) -> ExampleResult:
    """Run a code example and return the result.

    Args:
        example_code: Python code to execute
        globals_dict: Global namespace for execution
        capture_output: Whether to capture stdout/stderr

    Returns:
        ExampleResult with execution results

    Example:
        >>> from spicelab.help import run_example
        >>> result = run_example("print(1 + 1)")
        >>> result.success
        True
        >>> "2" in result.output
        True
    """
    if globals_dict is None:
        globals_dict = {}

    # Add common imports
    globals_dict.setdefault("__builtins__", __builtins__)

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    captured_stdout: io.StringIO | None = None
    if capture_output:
        captured_stdout = io.StringIO()
        sys.stdout = captured_stdout
        sys.stderr = io.StringIO()

    try:
        exec(example_code, globals_dict)
        output = captured_stdout.getvalue() if captured_stdout else ""
        return ExampleResult(
            name="example",
            source=example_code,
            success=True,
            output=output,
        )
    except Exception:
        error = traceback.format_exc()
        return ExampleResult(
            name="example",
            source=example_code,
            success=False,
            error=error,
        )
    finally:
        if capture_output:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@dataclass
class Example:
    """A runnable example.

    Attributes:
        name: Example name
        description: What the example demonstrates
        code: Python code
        category: Category for organization
        prerequisites: List of prerequisite examples
    """

    name: str
    description: str
    code: str
    category: str = "general"
    prerequisites: list[str] = field(default_factory=list)

    def run(self, globals_dict: dict[str, Any] | None = None) -> ExampleResult:
        """Run this example."""
        return run_example(self.code, globals_dict)


# Pre-defined examples
EXAMPLES = [
    Example(
        name="create_circuit",
        description="Create a simple circuit with a resistor",
        category="basics",
        code="""
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor
from spicelab.core.net import Net, GND

# Create a circuit
circuit = Circuit("simple_circuit")

# Create a resistor
R1 = Resistor(ref="1", resistance=1000)

# Add to circuit
circuit.add(R1)

# Create nets and connect
vin = Net("vin")
circuit.connect(R1.ports[0], vin)
circuit.connect(R1.ports[1], GND)

# Generate netlist
print(circuit.build_netlist())
""",
    ),
    Example(
        name="rc_filter",
        description="Create an RC lowpass filter",
        category="filters",
        code="""
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Capacitor
from spicelab.core.net import Net, GND

# Create circuit
circuit = Circuit("rc_lowpass")

# Components for 1kHz cutoff: fc = 1/(2*pi*R*C)
R1 = Resistor(ref="1", resistance=1000)  # 1k
C1 = Capacitor(ref="1", capacitance=159e-9)  # ~159nF

circuit.add(R1, C1)

# Create nets
vin = Net("vin")
vout = Net("vout")

# Connect: vin -> R1 -> vout -> C1 -> GND
circuit.connect(R1.ports[0], vin)
circuit.connect(R1.ports[1], vout)
circuit.connect(C1.ports[0], vout)
circuit.connect(C1.ports[1], GND)

print(circuit.build_netlist())
""",
    ),
    Example(
        name="voltage_divider",
        description="Create a voltage divider",
        category="basics",
        code="""
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import Net, GND

circuit = Circuit("voltage_divider")

# Components
V1 = Vdc(ref="1", value="10")  # 10V input
R1 = Resistor(ref="1", resistance=10000)  # 10k
R2 = Resistor(ref="2", resistance=10000)  # 10k

circuit.add(V1, R1, R2)

# Create nets
vin = Net("vin")
vout = Net("vout")

# Connect voltage source
circuit.connect(V1.ports[0], vin)
circuit.connect(V1.ports[1], GND)

# Connect divider
circuit.connect(R1.ports[0], vin)
circuit.connect(R1.ports[1], vout)
circuit.connect(R2.ports[0], vout)
circuit.connect(R2.ports[1], GND)

print(circuit.build_netlist())
# Output voltage = 10V * (10k / (10k + 10k)) = 5V
""",
    ),
    Example(
        name="use_templates",
        description="Using circuit templates",
        category="templates",
        code="""
from spicelab.templates import voltage_divider, rc_lowpass

# Create a voltage divider with 50% ratio
div_circuit = voltage_divider(ratio=0.5)
print("Voltage Divider:")
print(div_circuit.build_netlist())

# Create an RC lowpass filter with 1kHz cutoff
filter_circuit = rc_lowpass(fc=1000)
print("\\nRC Lowpass Filter (1kHz):")
print(filter_circuit.build_netlist())
""",
    ),
    Example(
        name="validate_circuit",
        description="Validate a circuit for errors",
        category="basics",
        code="""
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor
from spicelab.core.net import Net, GND

circuit = Circuit("validation_test")

R1 = Resistor(ref="1", resistance=1000)
circuit.add(R1)

# Only connect one port (incomplete circuit)
vin = Net("vin")
circuit.connect(R1.ports[0], vin)

# Validate
result = circuit.validate()
print(f"Is valid: {result.is_valid}")
for error in result.errors:
    print(f"Error: {error}")
for warning in result.warnings:
    print(f"Warning: {warning}")
""",
    ),
    Example(
        name="context_help",
        description="Using context-sensitive help",
        category="help",
        code="""
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor
from spicelab.help import get_help, show_help

# Create a circuit
circuit = Circuit("my_circuit")
circuit.add(Resistor(ref="1", resistance=1000))

# Get help for the circuit
help_obj = get_help(circuit)
print("Summary:")
print(help_obj.summary())

print("\\nAvailable methods:")
for method in help_obj.methods()[:5]:
    print(f"  - {method}")
""",
    ),
]


def list_examples(category: str | None = None) -> list[Example]:
    """List available examples.

    Args:
        category: Filter by category (None = all)

    Returns:
        List of Example objects

    Example:
        >>> from spicelab.help import list_examples
        >>> examples = list_examples()
        >>> len(examples) > 0
        True
        >>> examples = list_examples("basics")
        >>> all(e.category == "basics" for e in examples)
        True
    """
    if category:
        return [e for e in EXAMPLES if e.category == category]
    return list(EXAMPLES)


def get_example(name: str) -> Example | None:
    """Get an example by name.

    Args:
        name: Example name

    Returns:
        Example or None if not found

    Example:
        >>> from spicelab.help.examples import get_example
        >>> ex = get_example("create_circuit")
        >>> ex is not None
        True
        >>> ex.name
        'create_circuit'
    """
    for example in EXAMPLES:
        if example.name == name:
            return example
    return None


def run_example_by_name(
    name: str,
    print_output: bool = True,
) -> ExampleResult:
    """Run an example by name.

    Args:
        name: Example name
        print_output: Whether to print the output

    Returns:
        ExampleResult

    Raises:
        ValueError: If example not found

    Example:
        >>> from spicelab.help.examples import run_example_by_name
        >>> result = run_example_by_name("create_circuit", print_output=False)
        >>> result.success
        True
    """
    example = get_example(name)
    if example is None:
        raise ValueError(f"Example not found: {name}")

    result = example.run()

    if print_output:
        print(f"=== {example.name} ===")
        print(example.description)
        print()
        if result.success:
            print("Output:")
            print(result.output)
        else:
            print("Error:")
            print(result.error)

    return result


def get_categories() -> list[str]:
    """Get all example categories.

    Returns:
        List of category names

    Example:
        >>> from spicelab.help.examples import get_categories
        >>> categories = get_categories()
        >>> "basics" in categories
        True
    """
    return sorted(set(e.category for e in EXAMPLES))


__all__ = [
    "ExampleResult",
    "ValidationReport",
    "Example",
    "validate_docstring_examples",
    "run_example",
    "list_examples",
    "get_example",
    "run_example_by_name",
    "get_categories",
    "EXAMPLES",
]
