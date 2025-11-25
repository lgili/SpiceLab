"""Interactive troubleshooting for circuit simulation.

Provides a guided troubleshooting experience with:
- Automatic problem detection
- Interactive questionnaire
- Suggested fixes with explanations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .diagnostics import (
    DiagnosticResult,
    DiagnosticSeverity,
    diagnose_circuit,
    diagnose_convergence,
    diagnose_empty_results,
)

if TYPE_CHECKING:
    from ..core.circuit import Circuit

__all__ = ["Troubleshooter", "TroubleshootResult"]


@dataclass
class SuggestedFix:
    """A suggested fix for a problem.

    Attributes:
        title: Short description of the fix
        description: Detailed explanation
        code_example: Example code to implement the fix
        risk: How risky is this fix (low, medium, high)
    """

    title: str
    description: str
    code_example: str = ""
    risk: Literal["low", "medium", "high"] = "low"


@dataclass
class TroubleshootResult:
    """Result of troubleshooting session.

    Attributes:
        diagnostics: The diagnostic findings
        suggested_fixes: List of suggested fixes
        problem_type: Detected problem category
    """

    diagnostics: DiagnosticResult
    suggested_fixes: list[SuggestedFix]
    problem_type: str = "unknown"

    @property
    def has_fixes(self) -> bool:
        """Return True if there are suggested fixes."""
        return len(self.suggested_fixes) > 0


class Troubleshooter:
    """Interactive troubleshooter for circuit simulation issues.

    Provides automated diagnostics and guided troubleshooting.

    Args:
        circuit: The circuit to troubleshoot.
        console: Rich console for output (creates new one if None).

    Example:
        >>> from spicelab.troubleshooting import Troubleshooter
        >>> ts = Troubleshooter(my_circuit)
        >>> ts.diagnose()  # Quick auto-diagnosis
        >>> ts.interactive()  # Full interactive session
    """

    def __init__(
        self,
        circuit: Circuit,
        console: Console | None = None,
    ) -> None:
        self.circuit = circuit
        self.console = console or Console()
        self._diagnostics: DiagnosticResult | None = None

    def diagnose(self, verbose: bool = True) -> DiagnosticResult:
        """Run automatic diagnostics on the circuit.

        Args:
            verbose: If True, print findings to console.

        Returns:
            DiagnosticResult with all findings.
        """
        self._diagnostics = diagnose_circuit(self.circuit)

        if verbose:
            self._print_diagnostics(self._diagnostics)

        return self._diagnostics

    def diagnose_convergence(
        self,
        error_message: str = "",
        verbose: bool = True,
    ) -> DiagnosticResult:
        """Diagnose convergence failure.

        Args:
            error_message: Error message from the simulator.
            verbose: If True, print findings to console.

        Returns:
            DiagnosticResult focused on convergence issues.
        """
        result = diagnose_convergence(self.circuit, error_message)

        if verbose:
            self._print_diagnostics(result, title="Convergence Diagnostics")

        return result

    def diagnose_empty_results(
        self,
        analysis_type: str = "",
        probes: list[str] | None = None,
        verbose: bool = True,
    ) -> DiagnosticResult:
        """Diagnose why results are empty.

        Args:
            analysis_type: Type of analysis performed.
            probes: List of requested probes.
            verbose: If True, print findings to console.

        Returns:
            DiagnosticResult focused on empty results.
        """
        result = diagnose_empty_results(self.circuit, analysis_type, probes)

        if verbose:
            self._print_diagnostics(result, title="Empty Results Diagnostics")

        return result

    def interactive(self) -> TroubleshootResult:
        """Run interactive troubleshooting session.

        Guides the user through a series of questions to identify
        and fix simulation problems.

        Returns:
            TroubleshootResult with findings and suggested fixes.
        """
        self.console.print(
            Panel(
                "[bold]SpiceLab Interactive Troubleshooter[/bold]\n\n"
                "I'll help you diagnose and fix simulation issues.\n"
                "Answer a few questions to get started.",
                border_style="blue",
            )
        )

        # First, run auto-diagnostics
        self.console.print("\n[dim]Running automatic diagnostics...[/dim]\n")
        diagnostics = self.diagnose(verbose=False)

        if diagnostics.has_errors:
            self._print_diagnostics(diagnostics)
            self.console.print("\n[yellow]Found issues that should be fixed first.[/yellow]\n")

        # Ask about the problem
        problem_type = self._ask_problem_type()

        # Get more details based on problem type
        suggested_fixes: list[SuggestedFix] = []

        if problem_type == "convergence":
            error_msg = Prompt.ask(
                "Do you have an error message? (paste it or press Enter to skip)",
                default="",
            )
            conv_diag = diagnose_convergence(self.circuit, error_msg)
            diagnostics.findings.extend(conv_diag.findings)
            suggested_fixes.extend(self._get_convergence_fixes())

        elif problem_type == "empty":
            analysis = Prompt.ask(
                "What analysis type did you run?",
                choices=["tran", "ac", "dc", "op", "other"],
                default="tran",
            )
            empty_diag = diagnose_empty_results(self.circuit, analysis)
            diagnostics.findings.extend(empty_diag.findings)
            suggested_fixes.extend(self._get_empty_results_fixes(analysis))

        elif problem_type == "slow":
            suggested_fixes.extend(self._get_performance_fixes())

        elif problem_type == "wrong":
            suggested_fixes.extend(self._get_wrong_results_fixes())

        # Print suggested fixes
        if suggested_fixes:
            self._print_fixes(suggested_fixes)

        return TroubleshootResult(
            diagnostics=diagnostics,
            suggested_fixes=suggested_fixes,
            problem_type=problem_type,
        )

    def quick_fix(self) -> list[str]:
        """Get quick fix suggestions without interactive mode.

        Returns:
            List of fix suggestions as strings.
        """
        diagnostics = self.diagnose(verbose=False)
        fixes: list[str] = []

        for finding in diagnostics.findings:
            if finding.suggestion:
                fixes.append(f"{finding.message}: {finding.suggestion}")

        return fixes

    def _ask_problem_type(self) -> str:
        """Ask user what type of problem they're experiencing."""
        self.console.print("[bold]What problem are you experiencing?[/bold]\n")
        choices = {
            "1": ("convergence", "Simulation fails to converge"),
            "2": ("empty", "Simulation runs but results are empty"),
            "3": ("wrong", "Results don't match expected values"),
            "4": ("slow", "Simulation is too slow"),
            "5": ("other", "Something else"),
        }

        for key, (_, desc) in choices.items():
            self.console.print(f"  [{key}] {desc}")

        choice = Prompt.ask("\nEnter your choice", choices=list(choices.keys()))
        return choices[choice][0]

    def _get_convergence_fixes(self) -> list[SuggestedFix]:
        """Get fixes for convergence issues."""
        return [
            SuggestedFix(
                title="Relax convergence tolerance",
                description="Increase RELTOL to allow larger errors during iteration",
                code_example=".OPTIONS RELTOL=0.01  ; default is 0.001",
                risk="low",
            ),
            SuggestedFix(
                title="Increase iteration limit",
                description="Allow more iterations for DC operating point",
                code_example=".OPTIONS ITL1=200  ; default is 100",
                risk="low",
            ),
            SuggestedFix(
                title="Add initial conditions",
                description="Help the simulator find a starting point",
                code_example=".IC V(out)=0 V(vcc)=5",
                risk="low",
            ),
            SuggestedFix(
                title="Use GMIN stepping",
                description="Add small conductances to help matrix conditioning",
                code_example=".OPTIONS GMIN=1e-12  ; minimum conductance",
                risk="medium",
            ),
            SuggestedFix(
                title="Add node hints",
                description="Suggest approximate node voltages",
                code_example=".NODESET V(out)=2.5 V(bias)=1.2",
                risk="low",
            ),
        ]

    def _get_empty_results_fixes(self, analysis_type: str) -> list[SuggestedFix]:
        """Get fixes for empty results."""
        fixes = [
            SuggestedFix(
                title="Check probe names",
                description="Ensure probed nodes exist in the circuit",
                code_example='probes=["V(out)", "I(R1)"]  ; use actual node names',
                risk="low",
            ),
            SuggestedFix(
                title="Verify circuit connectivity",
                description="Run circuit.validate() to check for issues",
                code_example="result = circuit.validate()\nprint(result)",
                risk="low",
            ),
        ]

        if analysis_type == "ac":
            fixes.append(
                SuggestedFix(
                    title="Add AC source",
                    description="AC analysis requires an AC signal source",
                    code_example='Vac("1", ac_mag=1.0)  ; 1V AC source',
                    risk="low",
                )
            )

        return fixes

    def _get_performance_fixes(self) -> list[SuggestedFix]:
        """Get fixes for slow simulations."""
        return [
            SuggestedFix(
                title="Increase timestep",
                description="Use larger timestep for faster simulation",
                code_example=".TRAN 1u 10m  ; 1µs step instead of default",
                risk="medium",
            ),
            SuggestedFix(
                title="Reduce simulation time",
                description="Simulate only the portion you need",
                code_example="tran(stop_time=1e-3)  ; 1ms instead of 1s",
                risk="low",
            ),
            SuggestedFix(
                title="Use caching",
                description="Cache results to avoid re-running simulations",
                code_example="result = orchestrator.run(cache=True)",
                risk="low",
            ),
            SuggestedFix(
                title="Simplify circuit",
                description="Remove unnecessary components or use behavioral models",
                code_example="# Replace complex subcircuit with behavioral model",
                risk="high",
            ),
        ]

    def _get_wrong_results_fixes(self) -> list[SuggestedFix]:
        """Get fixes for incorrect results."""
        return [
            SuggestedFix(
                title="Check units",
                description="Verify component values use correct units",
                code_example='R = Resistor("1", resistance=1e3)  ; 1kΩ not 1Ω',
                risk="low",
            ),
            SuggestedFix(
                title="Verify connections",
                description="Check that components are connected correctly",
                code_example="circuit.summary()  ; print circuit topology",
                risk="low",
            ),
            SuggestedFix(
                title="Check probe polarity",
                description="Ensure you're measuring the right nodes",
                code_example="V(out, gnd)  ; voltage from out to ground",
                risk="low",
            ),
            SuggestedFix(
                title="Compare with schematic",
                description="Generate netlist and compare with expected",
                code_example="print(circuit.build_netlist())",
                risk="low",
            ),
        ]

    def _print_diagnostics(
        self,
        diagnostics: DiagnosticResult,
        title: str = "Diagnostics",
    ) -> None:
        """Print diagnostics in a formatted way."""
        if not diagnostics.findings:
            self.console.print(f"[green]✓ {title}: No issues found[/green]")
            return

        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column("Severity", width=10)
        table.add_column("Category", width=12)
        table.add_column("Message")
        table.add_column("Suggestion", style="dim")

        for finding in diagnostics.findings:
            severity_style = {
                DiagnosticSeverity.INFO: "blue",
                DiagnosticSeverity.WARNING: "yellow",
                DiagnosticSeverity.ERROR: "red",
                DiagnosticSeverity.CRITICAL: "bold red",
            }.get(finding.severity, "white")

            table.add_row(
                Text(finding.severity.value.upper(), style=severity_style),
                finding.category,
                finding.message,
                finding.suggestion,
            )

        self.console.print(table)
        self.console.print(f"\n[dim]Summary: {diagnostics.summary()}[/dim]")

    def _print_fixes(self, fixes: list[SuggestedFix]) -> None:
        """Print suggested fixes in a formatted way."""
        self.console.print("\n[bold]Suggested Fixes:[/bold]\n")

        for i, fix in enumerate(fixes, 1):
            risk_style = {
                "low": "green",
                "medium": "yellow",
                "high": "red",
            }.get(fix.risk, "white")

            self.console.print(f"[bold]{i}. {fix.title}[/bold]")
            self.console.print(f"   {fix.description}")
            self.console.print(f"   Risk: [{risk_style}]{fix.risk}[/{risk_style}]")
            if fix.code_example:
                self.console.print(f"   [dim]Example: {fix.code_example}[/dim]")
            self.console.print()
