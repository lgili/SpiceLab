# M4: Developer Experience (DX)

**Status:** Proposed
**Priority:** 🟡 MEDIUM-HIGH
**Estimated Duration:** 4-6 weeks
**Dependencies:** M1 (error handling), M3 (component library for autocomplete)

## Problem Statement

SpiceLab currently provides cryptic error messages, difficult debugging workflows, and limited IDE support. This creates a steep learning curve for new users and frustrates experienced users during development, directly impacting adoption and productivity.

### Current Gaps
- ❌ Error messages are cryptic (stack traces instead of explanations)
- ❌ No correction suggestions (typos, invalid values go undetected)
- ❌ No netlist diff visualization (hard to debug changes)
- ❌ No pre-simulation validation (errors discovered late)
- ❌ No autocomplete metadata (no hints for values like "1k", "10k")
- ❌ No ready-to-use circuit templates
- ❌ No interactive troubleshooting

### Impact
- **User Frustration:** Cryptic errors cause abandonment
- **Debugging Time:** Hours wasted on trial-and-error debugging
- **Learning Curve:** Steep barrier for beginners
- **IDE Integration:** Poor developer experience in VSCode/PyCharm
- **Productivity:** Slow iteration cycles

## Objectives

1. **Humanize error messages** with Rust-style helpful diagnostics
2. **Add correction suggestions** for typos and invalid values
3. **Implement netlist diff** with visual highlighting (rich/textual)
4. **Create circuit validation** that runs before simulation
5. **Provide autocomplete metadata** for IDE integration
6. **Build circuit templates** ready to use (filters, amplifiers, PSU)
7. **Add interactive troubleshooting** with guided diagnostics
8. **Target:** 60% reduction in average debugging time

## Technical Design

### 1. Humanized Error Messages

**Strategy:** Replace stack traces with helpful, actionable error messages (inspired by Rust/Elm compilers).

#### Before (Cryptic)
```python
Traceback (most recent call last):
  File "circuit.py", line 42, in build_netlist
    assert self.components, "No components"
AssertionError: No components
```

#### After (Helpful)
```
❌ Circuit Build Error: Empty Circuit

Your circuit "my_circuit" has no components. Add at least one component
before building a netlist.

Example:
  circuit.add(Resistor("R1", "1k"))

📖 See: https://docs.spicelab.io/building-circuits
```

#### Implementation
```python
# spicelab/errors.py
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

class CircuitError(Exception):
    """Base exception with helpful formatting."""

    def __init__(
        self,
        message: str,
        hint: str | None = None,
        code_example: str | None = None,
        docs_url: str | None = None,
    ):
        self.message = message
        self.hint = hint
        self.code_example = code_example
        self.docs_url = docs_url
        super().__init__(message)

    def __str__(self) -> str:
        """Format error with rich formatting."""
        parts = [f"❌ {self.message}"]

        if self.hint:
            parts.append(f"\n💡 {self.hint}")

        if self.code_example:
            parts.append("\nExample:")
            syntax = Syntax(self.code_example, "python", theme="monokai")
            # Note: This is simplified; actual impl uses rich.console
            parts.append(f"  {self.code_example}")

        if self.docs_url:
            parts.append(f"\n📖 See: {self.docs_url}")

        return "\n".join(parts)

class EmptyCircuitError(CircuitError):
    """Circuit has no components."""

    def __init__(self, circuit_name: str):
        super().__init__(
            message=f'Circuit "{circuit_name}" has no components',
            hint="Add at least one component before building a netlist",
            code_example='circuit.add(Resistor("R1", "1k"))',
            docs_url="https://docs.spicelab.io/building-circuits"
        )

# Usage in circuit.py
def build_netlist(self) -> str:
    if not self.components:
        raise EmptyCircuitError(self.name)
    # ...
```

### 2. Correction Suggestions

**Strategy:** Use fuzzy matching to suggest corrections for typos.

#### Typo Detection
```python
# spicelab/validation/suggestions.py
from difflib import get_close_matches

class SuggestionEngine:
    """Suggest corrections for common mistakes."""

    def __init__(self):
        # Known component types
        self.component_types = [
            "Resistor", "Capacitor", "Inductor", "Diode",
            "BJT", "MOSFET", "JFET", "OpAmp", "VoltageSource"
        ]

        # Known unit suffixes
        self.unit_suffixes = ["k", "M", "G", "m", "u", "n", "p", "f"]

    def suggest_component(self, typo: str) -> list[str]:
        """Suggest component type corrections."""
        matches = get_close_matches(typo, self.component_types, n=3, cutoff=0.6)
        return matches

    def suggest_value(self, value_str: str) -> list[str]:
        """Suggest value corrections."""
        # Example: "1kO" -> "1k" (resistor value)
        # Example: "10uF" -> "10u" (capacitor value)
        suggestions = []

        # Check for common typos
        if value_str.endswith("kO"):
            suggestions.append(value_str[:-2] + "k")  # 1kO -> 1k

        if "uF" in value_str:
            suggestions.append(value_str.replace("uF", "u"))  # 10uF -> 10u

        return suggestions

# Usage
class ComponentNotFoundError(CircuitError):
    """Component type not found."""

    def __init__(self, typo: str):
        engine = SuggestionEngine()
        suggestions = engine.suggest_component(typo)

        hint = None
        if suggestions:
            hint = f"Did you mean: {', '.join(suggestions)}?"

        super().__init__(
            message=f'Unknown component type "{typo}"',
            hint=hint,
            docs_url="https://docs.spicelab.io/components"
        )
```

#### Invalid Value Detection
```python
# spicelab/validation/values.py

class InvalidValueError(CircuitError):
    """Invalid component value."""

    def __init__(self, component: str, value: str, reason: str):
        hint = None

        # Suggest corrections for common mistakes
        if "negative" in reason.lower():
            hint = "Resistance/capacitance must be positive. Use absolute value."
        elif "too small" in reason.lower():
            hint = "Value may be unrealistically small. Check units (use 'k', 'M', etc.)"
        elif "too large" in reason.lower():
            hint = "Value may be unrealistically large. Check units."

        super().__init__(
            message=f'{component} has invalid value "{value}": {reason}',
            hint=hint,
            code_example=f'{component}("R1", "1k")  # Use SI prefixes'
        )
```

### 3. Netlist Diff Visualization

**Strategy:** Show side-by-side diff of netlists with syntax highlighting.

#### Implementation
```python
# spicelab/visualization/netlist_diff.py
from rich.console import Console
from rich.syntax import Syntax
from rich.columns import Columns
import difflib

class NetlistDiff:
    """Visualize netlist differences."""

    def __init__(self):
        self.console = Console()

    def diff(self, netlist1: str, netlist2: str, name1: str = "Before", name2: str = "After"):
        """Show side-by-side diff of netlists."""

        # Compute line-by-line diff
        lines1 = netlist1.splitlines()
        lines2 = netlist2.splitlines()

        diff = difflib.unified_diff(lines1, lines2, lineterm="")

        # Format with rich
        self.console.print(f"\n[bold]Netlist Diff: {name1} → {name2}[/bold]\n")

        for line in diff:
            if line.startswith('+'):
                self.console.print(f"[green]{line}[/green]")
            elif line.startswith('-'):
                self.console.print(f"[red]{line}[/red]")
            elif line.startswith('@@'):
                self.console.print(f"[blue]{line}[/blue]")
            else:
                self.console.print(line)

    def side_by_side(self, netlist1: str, netlist2: str):
        """Show side-by-side comparison."""
        syntax1 = Syntax(netlist1, "spice", theme="monokai", line_numbers=True)
        syntax2 = Syntax(netlist2, "spice", theme="monokai", line_numbers=True)

        columns = Columns([syntax1, syntax2])
        self.console.print(columns)

# Usage
diff = NetlistDiff()
netlist_before = circuit.build_netlist()
# ... modify circuit ...
netlist_after = circuit.build_netlist()
diff.diff(netlist_before, netlist_after)
```

### 4. Pre-Simulation Validation

**Strategy:** Validate circuit before running simulation to catch errors early.

#### Validation Rules
```python
# spicelab/validation/circuit_checks.py
from dataclasses import dataclass
from typing import Callable

@dataclass
class ValidationResult:
    """Result of validation check."""
    passed: bool
    message: str
    severity: str  # "error", "warning", "info"
    suggestion: str | None = None

class CircuitValidator:
    """Validate circuit before simulation."""

    def __init__(self):
        self.checks: list[Callable] = [
            self._check_floating_nodes,
            self._check_no_dc_path,
            self._check_voltage_source_loop,
            self._check_unrealistic_values,
            self._check_missing_ground,
        ]

    def validate(self, circuit: Circuit) -> list[ValidationResult]:
        """Run all validation checks."""
        results = []
        for check in self.checks:
            result = check(circuit)
            if result:
                results.append(result)
        return results

    def _check_floating_nodes(self, circuit: Circuit) -> ValidationResult | None:
        """Check for disconnected nodes."""
        # Build connectivity graph
        graph = circuit.graph
        components = circuit.components

        # Find nodes with <2 connections (excluding GND)
        floating = []
        for net_name, net in circuit.nets.items():
            if net_name == "0":  # GND
                continue
            if len(net.ports) < 2:
                floating.append(net_name)

        if floating:
            return ValidationResult(
                passed=False,
                severity="error",
                message=f"Floating nodes detected: {', '.join(floating)}",
                suggestion="Ensure all nodes are connected to at least two components"
            )
        return None

    def _check_voltage_source_loop(self, circuit: Circuit) -> ValidationResult | None:
        """Check for voltage source loops (illegal in SPICE)."""
        # TODO: Detect cycles in voltage sources
        return None

    def _check_unrealistic_values(self, circuit: Circuit) -> ValidationResult | None:
        """Check for unrealistic component values."""
        warnings = []

        for comp in circuit.components.values():
            if isinstance(comp, Resistor):
                if comp.resistance < 1e-3:
                    warnings.append(f"{comp.ref} = {comp.resistance}Ω (very small, check units)")
                elif comp.resistance > 1e9:
                    warnings.append(f"{comp.ref} = {comp.resistance}Ω (very large, check units)")

            elif isinstance(comp, Capacitor):
                if comp.capacitance < 1e-15:
                    warnings.append(f"{comp.ref} = {comp.capacitance}F (very small, check units)")
                elif comp.capacitance > 1:
                    warnings.append(f"{comp.ref} = {comp.capacitance}F (very large, check units)")

        if warnings:
            return ValidationResult(
                passed=True,  # Warning, not error
                severity="warning",
                message="Unrealistic component values detected",
                suggestion="\n".join(warnings)
            )
        return None

# Usage
validator = CircuitValidator()
results = validator.validate(circuit)

for result in results:
    if result.severity == "error":
        print(f"❌ {result.message}")
        if result.suggestion:
            print(f"💡 {result.suggestion}")
    elif result.severity == "warning":
        print(f"⚠️  {result.message}")
```

### 5. Autocomplete Metadata

**Strategy:** Provide metadata for IDE autocomplete (VSCode, PyCharm).

#### Type Stubs with Value Hints
```python
# spicelab/core/components.pyi (type stub file)
from typing import Literal

class Resistor:
    """Resistor component.

    Common values:
        - 1k, 10k, 100k (E12 series)
        - Use SI prefixes: k (kilo), M (mega)
    """

    def __init__(
        self,
        ref: str,
        resistance: float | Literal["1k", "10k", "100k", "1M"] | str,
    ) -> None: ...

class Capacitor:
    """Capacitor component.

    Common values:
        - 100n, 1u, 10u (ceramic)
        - Use SI prefixes: n (nano), u (micro)
    """

    def __init__(
        self,
        ref: str,
        capacitance: float | Literal["100n", "1u", "10u", "100u"] | str,
    ) -> None: ...
```

#### VSCode Extension (Optional)
```json
// .vscode/spicelab.code-snippets
{
  "Resistor E12 1k": {
    "prefix": "R1k",
    "body": [
      "Resistor(\"${1:R1}\", \"1k\")"
    ],
    "description": "1kΩ resistor (E12 series)"
  },
  "Capacitor 100nF": {
    "prefix": "C100n",
    "body": [
      "Capacitor(\"${1:C1}\", \"100n\")"
    ],
    "description": "100nF ceramic capacitor"
  }
}
```

### 6. Circuit Templates

**Strategy:** Provide ready-to-use circuit templates for common designs.

#### Template Library
```python
# spicelab/templates/filters.py

class LowPassRCFilter:
    """RC low-pass filter template."""

    @staticmethod
    def create(
        name: str,
        cutoff_freq: float,
        input_impedance: float = 1e3,
    ) -> Circuit:
        """Create RC low-pass filter.

        Args:
            name: Circuit name
            cutoff_freq: -3dB cutoff frequency (Hz)
            input_impedance: Input resistor value (Ω)

        Returns:
            Configured circuit ready to simulate
        """
        circuit = Circuit(name)

        # Calculate component values
        R = input_impedance
        C = 1 / (2 * np.pi * cutoff_freq * R)

        # Build circuit
        vin = VoltageSource("Vin", dc=0, ac=1)
        r1 = Resistor("R1", R)
        c1 = Capacitor("C1", C)

        circuit.add(vin)
        circuit.add(r1)
        circuit.add(c1)

        # Connect: Vin -> R1 -> C1 -> GND
        circuit.connect(vin.ports["p"], "in")
        circuit.connect(vin.ports["n"], "0")
        circuit.connect(r1.ports[0], "in")
        circuit.connect(r1.ports[1], "out")
        circuit.connect(c1.ports[0], "out")
        circuit.connect(c1.ports[1], "0")

        return circuit

# Usage
lpf = LowPassRCFilter.create("lpf_1kHz", cutoff_freq=1e3)
```

#### Template Catalog
```python
# spicelab/templates/__init__.py

# Filters
from .filters import (
    LowPassRCFilter,
    HighPassRCFilter,
    BandPassFilter,
    NotchFilter,
)

# Amplifiers
from .amplifiers import (
    NonInvertingAmplifier,
    InvertingAmplifier,
    DifferentialAmplifier,
    InstrumentationAmplifier,
)

# Power supplies
from .power import (
    LinearRegulator,
    BuckConverter,
    BoostConverter,
    InverterCircuit,
)

# Oscillators
from .oscillators import (
    RCOscillator,
    CrystalOscillator,
    VCO,
)
```

### 7. Interactive Troubleshooting

**Strategy:** Provide guided diagnostics for common issues.

#### Troubleshooter
```python
# spicelab/troubleshooting/interactive.py
from rich.prompt import Prompt, Confirm
from rich.console import Console

class InteractiveTroubleshooter:
    """Interactive circuit troubleshooting."""

    def __init__(self):
        self.console = Console()

    def diagnose_convergence_failure(self, circuit: Circuit):
        """Diagnose why simulation didn't converge."""
        self.console.print("[bold yellow]⚠️  Convergence Failure Diagnostics[/bold yellow]\n")

        # Step 1: Check for common issues
        issues = []

        # Check for floating nodes
        validator = CircuitValidator()
        results = validator.validate(circuit)
        for result in results:
            if result.severity == "error":
                issues.append(result.message)

        # Check for unrealistic values
        for comp in circuit.components.values():
            if isinstance(comp, Resistor) and comp.resistance < 1e-6:
                issues.append(f"{comp.ref} has very small resistance ({comp.resistance}Ω)")

        # Present findings
        if issues:
            self.console.print("[red]Found potential issues:[/red]")
            for issue in issues:
                self.console.print(f"  • {issue}")
            self.console.print()

        # Step 2: Suggest fixes
        self.console.print("[bold]Suggested fixes:[/bold]")
        self.console.print("1. Add .OPTIONS RELTOL=1e-4 (relax tolerance)")
        self.console.print("2. Add initial conditions (.IC or .NODESET)")
        self.console.print("3. Check for shorted voltage sources")
        self.console.print("4. Increase GMIN (.OPTIONS GMIN=1e-10)")
        self.console.print()

        # Step 3: Interactive fix
        if Confirm.ask("Would you like to automatically add .OPTIONS RELTOL=1e-4?"):
            # TODO: Add option to circuit
            self.console.print("[green]✓ Added relaxed convergence tolerance[/green]")

# Usage
troubleshooter = InteractiveTroubleshooter()
try:
    result = run_simulation(circuit, analyses)
except ConvergenceError:
    troubleshooter.diagnose_convergence_failure(circuit)
```

## Implementation Plan

### Week 1: Error Messages & Suggestions
- [ ] Design error hierarchy with helpful messages
- [ ] Implement CircuitError base class with rich formatting
- [ ] Create specific error classes (EmptyCircuitError, etc.)
- [ ] Add SuggestionEngine for typo correction
- [ ] Replace all assertions with helpful errors
- [ ] Test error messages with users (feedback)

### Week 2: Validation & Diff
- [ ] Implement CircuitValidator with checks
- [ ] Add validation rules (floating nodes, loops, etc.)
- [ ] Create NetlistDiff visualization
- [ ] Add side-by-side comparison
- [ ] Integrate validation into simulation workflow
- [ ] Add validation CLI command

### Week 3: Autocomplete & Type Stubs
- [ ] Create type stub files (.pyi) for all components
- [ ] Add Literal type hints for common values
- [ ] Create VSCode snippets (optional)
- [ ] Test autocomplete in VSCode/PyCharm
- [ ] Document IDE setup

### Week 4: Circuit Templates
- [ ] Implement filter templates (LPF, HPF, BPF, Notch)
- [ ] Create amplifier templates (inverting, non-inverting, etc.)
- [ ] Add power supply templates (LDO, Buck, Boost)
- [ ] Implement oscillator templates (RC, Crystal, VCO)
- [ ] Document all templates with examples

### Week 5: Interactive Troubleshooting
- [ ] Create InteractiveTroubleshooter class
- [ ] Add convergence failure diagnostics
- [ ] Implement guided fixes (interactive prompts)
- [ ] Add common issue detection (floating nodes, etc.)
- [ ] Test with real user scenarios

### Week 6: Integration & UX Testing
- [ ] Integrate all DX improvements into library
- [ ] Create comprehensive examples
- [ ] Conduct user testing (5+ developers)
- [ ] Measure debugging time reduction
- [ ] Refine based on feedback
- [ ] Update documentation

## Success Metrics

### Error Messages
- [ ] 100% of errors have helpful messages (no raw stack traces)
- [ ] 80%+ of errors include suggestions
- [ ] Average time to understand error: <30 seconds

### Validation
- [ ] Catch 90%+ of common errors before simulation
- [ ] Validation runs in <100ms for typical circuits
- [ ] Zero false positives in validation

### Developer Productivity
- [ ] **60% reduction** in average debugging time (measured)
- [ ] Autocomplete works in VSCode/PyCharm
- [ ] 10+ ready-to-use circuit templates
- [ ] User satisfaction score: >4.5/5

### Documentation
- [ ] All error types documented
- [ ] All templates documented with examples
- [ ] Troubleshooting guide written
- [ ] IDE setup guide written

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Error messages too verbose | MEDIUM | Make verbose mode optional, default to concise |
| Validation too strict | MEDIUM | Separate errors from warnings, allow overrides |
| Autocomplete doesn't work | LOW | Provide fallback documentation, test with multiple IDEs |
| Templates too opinionated | LOW | Provide customization parameters, document assumptions |
| Rich formatting breaks CI | MEDIUM | Auto-detect terminal capabilities, fallback to plain text |

## Dependencies

**Required:**
- M1 (error handling infrastructure)
- M3 (component library for autocomplete/templates)
- rich (`pip install rich`)
- difflib (stdlib)

**Optional:**
- prompt_toolkit (for interactive features)
- VSCode extension development (optional)

## Future Enhancements

- **M11:** Web UI for visual circuit building
- **M17:** Schematic import/export (visual debugging)
- **M19:** AI-powered debugging suggestions

## References

- [Rust Compiler Error Messages](https://blog.rust-lang.org/2016/08/10/Shape-of-errors-to-come.html)
- [Elm Error Messages Philosophy](https://elm-lang.org/news/compiler-errors-for-humans)
- [Rich Python Library](https://rich.readthedocs.io/)
- [Python Type Stubs (PEP 484)](https://peps.python.org/pep-0484/)
