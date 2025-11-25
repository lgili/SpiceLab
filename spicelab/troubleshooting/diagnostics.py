"""Diagnostic functions for circuit simulation issues.

Provides automated detection of common problems:
- Convergence issues (timestep too large, bad initial conditions)
- Empty results (no probes, wrong analysis type)
- Circuit topology problems (floating nodes, shorts)

Each diagnostic returns a DiagnosticResult with findings and suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.circuit import Circuit

__all__ = [
    "DiagnosticResult",
    "DiagnosticSeverity",
    "Finding",
    "diagnose_circuit",
    "diagnose_convergence",
    "diagnose_empty_results",
]


class DiagnosticSeverity(Enum):
    """Severity level for diagnostic findings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Finding:
    """A single diagnostic finding.

    Attributes:
        category: Type of issue (e.g., "convergence", "topology", "config")
        severity: How serious the issue is
        message: Description of the problem
        suggestion: How to fix it
        details: Additional technical details
    """

    category: str
    severity: DiagnosticSeverity
    message: str
    suggestion: str
    details: str = ""

    def __str__(self) -> str:
        icon = {
            DiagnosticSeverity.INFO: "ℹ",
            DiagnosticSeverity.WARNING: "⚠",
            DiagnosticSeverity.ERROR: "✗",
            DiagnosticSeverity.CRITICAL: "🔥",
        }.get(self.severity, "•")
        return f"{icon} [{self.category}] {self.message}"


@dataclass
class DiagnosticResult:
    """Result of running diagnostics on a circuit.

    Attributes:
        findings: List of issues found
        circuit_name: Name of the analyzed circuit
        component_count: Number of components
    """

    findings: list[Finding] = field(default_factory=list)
    circuit_name: str = ""
    component_count: int = 0

    @property
    def has_issues(self) -> bool:
        """Return True if any issues were found."""
        return len(self.findings) > 0

    @property
    def has_errors(self) -> bool:
        """Return True if any errors or critical issues were found."""
        return any(
            f.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL)
            for f in self.findings
        )

    @property
    def error_count(self) -> int:
        """Count of error-level findings."""
        return sum(
            1
            for f in self.findings
            if f.severity in (DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL)
        )

    @property
    def warning_count(self) -> int:
        """Count of warning-level findings."""
        return sum(1 for f in self.findings if f.severity == DiagnosticSeverity.WARNING)

    def add(self, finding: Finding) -> None:
        """Add a finding to the result."""
        self.findings.append(finding)

    def by_category(self, category: str) -> list[Finding]:
        """Get findings by category."""
        return [f for f in self.findings if f.category == category]

    def summary(self) -> str:
        """Return a brief summary of findings."""
        if not self.findings:
            return "No issues found"
        parts = []
        if self.error_count:
            parts.append(f"{self.error_count} error(s)")
        if self.warning_count:
            parts.append(f"{self.warning_count} warning(s)")
        info_count = len(self.findings) - self.error_count - self.warning_count
        if info_count:
            parts.append(f"{info_count} info")
        return ", ".join(parts)


def diagnose_circuit(circuit: Circuit) -> DiagnosticResult:
    """Run comprehensive diagnostics on a circuit.

    Checks for:
    - Topology issues (floating nodes, no ground)
    - Component value issues (extreme values, zeros)
    - Potential convergence problems

    Args:
        circuit: The circuit to diagnose.

    Returns:
        DiagnosticResult with all findings.
    """
    result = DiagnosticResult(
        circuit_name=circuit.name,
        component_count=len(circuit._components),
    )

    # Run validation first
    validation = circuit.validate()
    for error in validation.errors:
        result.add(
            Finding(
                category="topology",
                severity=DiagnosticSeverity.ERROR,
                message=error.message,
                suggestion=error.suggestion or "Check circuit connections",
                details=f"Component: {error.component_ref}" if error.component_ref else "",
            )
        )
    for warning in validation.warnings:
        result.add(
            Finding(
                category="topology",
                severity=DiagnosticSeverity.WARNING,
                message=warning.message,
                suggestion=warning.suggestion or "Review component values",
                details=f"Component: {warning.component_ref}" if warning.component_ref else "",
            )
        )

    # Check for common convergence risk factors
    _check_convergence_risks(circuit, result)

    # Check component count
    if len(circuit._components) == 0:
        result.add(
            Finding(
                category="topology",
                severity=DiagnosticSeverity.ERROR,
                message="Circuit has no components",
                suggestion="Add components before simulating",
            )
        )
    elif len(circuit._components) > 1000:
        result.add(
            Finding(
                category="performance",
                severity=DiagnosticSeverity.INFO,
                message=f"Large circuit ({len(circuit._components)} components)",
                suggestion="Consider using .OPTIONS to tune simulation parameters",
                details="Large circuits may benefit from RELTOL=0.01 or ITL1=200",
            )
        )

    return result


def _check_convergence_risks(circuit: Circuit, result: DiagnosticResult) -> None:
    """Check for factors that commonly cause convergence issues."""
    from ..core.components import Capacitor, Inductor, Resistor

    # Check for very small/large component values
    for comp in circuit._components:
        if isinstance(comp, Resistor):
            r = comp.resistance
            if r is not None:
                if r < 0.001:  # < 1mΩ
                    result.add(
                        Finding(
                            category="convergence",
                            severity=DiagnosticSeverity.WARNING,
                            message=f"Very small resistance R{comp.ref} = {r}Ω",
                            suggestion="Consider using at least 1mΩ to avoid numerical issues",
                            details="Very small resistances can cause matrix singularity",
                        )
                    )
                elif r > 1e12:  # > 1TΩ
                    result.add(
                        Finding(
                            category="convergence",
                            severity=DiagnosticSeverity.WARNING,
                            message=f"Very large resistance R{comp.ref} = {r}Ω",
                            suggestion="Consider using at most 1GΩ for better convergence",
                            details="Very large resistances can cause numerical precision issues",
                        )
                    )

        elif isinstance(comp, Capacitor):
            c = comp.capacitance
            if c is not None:
                if c < 1e-15:  # < 1fF
                    result.add(
                        Finding(
                            category="convergence",
                            severity=DiagnosticSeverity.WARNING,
                            message=f"Very small capacitance C{comp.ref} = {c}F",
                            suggestion="Consider using at least 1fF",
                            details="Very small capacitances can cause timestep issues",
                        )
                    )

        elif isinstance(comp, Inductor):
            l_val = comp.inductance
            if l_val is not None:
                if l_val < 1e-12:  # < 1pH
                    result.add(
                        Finding(
                            category="convergence",
                            severity=DiagnosticSeverity.WARNING,
                            message=f"Very small inductance L{comp.ref} = {l_val}H",
                            suggestion="Consider using at least 1pH",
                            details="Very small inductances can cause timestep issues",
                        )
                    )


def diagnose_convergence(
    circuit: Circuit,
    error_message: str = "",
) -> DiagnosticResult:
    """Diagnose convergence failure.

    Analyzes the circuit and error message to suggest fixes.

    Args:
        circuit: The circuit that failed to converge.
        error_message: The error message from the simulator (if available).

    Returns:
        DiagnosticResult with convergence-specific findings.
    """
    result = DiagnosticResult(
        circuit_name=circuit.name,
        component_count=len(circuit._components),
    )

    # Parse error message for clues
    error_lower = error_message.lower()

    if "timestep too small" in error_lower or "time step" in error_lower:
        result.add(
            Finding(
                category="convergence",
                severity=DiagnosticSeverity.ERROR,
                message="Timestep became too small",
                suggestion="Try increasing RELTOL (e.g., .OPTIONS RELTOL=0.01)",
                details="This usually indicates a stiff system or rapid transient",
            )
        )
        result.add(
            Finding(
                category="convergence",
                severity=DiagnosticSeverity.INFO,
                message="Consider adding initial conditions",
                suggestion="Use .IC to set initial node voltages",
                details="Initial conditions help the simulator find a stable starting point",
            )
        )

    if "singular matrix" in error_lower:
        result.add(
            Finding(
                category="convergence",
                severity=DiagnosticSeverity.CRITICAL,
                message="Singular matrix encountered",
                suggestion="Check for floating nodes or voltage source loops",
                details="A singular matrix means the circuit has no unique solution",
            )
        )

    if "no convergence" in error_lower or "failed to converge" in error_lower:
        result.add(
            Finding(
                category="convergence",
                severity=DiagnosticSeverity.ERROR,
                message="DC operating point did not converge",
                suggestion="Try .OPTIONS ITL1=200 or add .NODESET hints",
                details="The simulator could not find a stable DC solution",
            )
        )

    if "gmin" in error_lower:
        result.add(
            Finding(
                category="convergence",
                severity=DiagnosticSeverity.WARNING,
                message="GMIN stepping was required",
                suggestion="This is often okay, but check for floating nodes",
                details="GMIN adds small conductances to help convergence",
            )
        )

    # Run general circuit diagnostics too
    general = diagnose_circuit(circuit)
    for finding in general.findings:
        if finding.category == "convergence":
            result.add(finding)

    # Add general convergence tips if no specific issues found
    if not result.findings:
        result.add(
            Finding(
                category="convergence",
                severity=DiagnosticSeverity.INFO,
                message="No obvious convergence issues detected",
                suggestion="Try these general fixes",
                details=(
                    "1. Add .OPTIONS RELTOL=0.01\n"
                    "2. Use .IC for initial conditions\n"
                    "3. Reduce simulation time step\n"
                    "4. Check for very fast transients"
                ),
            )
        )

    return result


def diagnose_empty_results(
    circuit: Circuit,
    analysis_type: str = "",
    probes: list[str] | None = None,
) -> DiagnosticResult:
    """Diagnose why simulation returned empty results.

    Args:
        circuit: The circuit that was simulated.
        analysis_type: Type of analysis (e.g., "tran", "ac", "dc").
        probes: List of requested probe names.

    Returns:
        DiagnosticResult with findings about empty results.
    """
    result = DiagnosticResult(
        circuit_name=circuit.name,
        component_count=len(circuit._components),
    )

    # Check if circuit is empty
    if len(circuit._components) == 0:
        result.add(
            Finding(
                category="config",
                severity=DiagnosticSeverity.ERROR,
                message="Circuit has no components",
                suggestion="Add components before simulating",
            )
        )
        return result

    # Check probes
    if probes is not None:
        if len(probes) == 0:
            result.add(
                Finding(
                    category="config",
                    severity=DiagnosticSeverity.ERROR,
                    message="No probes specified",
                    suggestion="Add probes like V(out) or I(R1) to capture data",
                    details="Without probes, the simulator has nothing to record",
                )
            )
        else:
            # Check if probed nodes exist
            for probe in probes:
                # Simple check for V(node) format
                if probe.startswith("V(") or probe.startswith("v("):
                    # Could check if node exists in circuit
                    result.add(
                        Finding(
                            category="config",
                            severity=DiagnosticSeverity.INFO,
                            message=f"Probe '{probe}' requested",
                            suggestion="Verify this node exists in the circuit",
                        )
                    )

    # Check analysis type
    if analysis_type:
        analysis_lower = analysis_type.lower()
        if analysis_lower == "dc" or analysis_lower == "op":
            result.add(
                Finding(
                    category="config",
                    severity=DiagnosticSeverity.INFO,
                    message="DC/OP analysis produces single-point data",
                    suggestion="For waveforms, use transient (.tran) or AC (.ac) analysis",
                )
            )
        elif analysis_lower == "ac":
            result.add(
                Finding(
                    category="config",
                    severity=DiagnosticSeverity.INFO,
                    message="AC analysis requires an AC source",
                    suggestion="Ensure you have a VAC or IAC source in the circuit",
                )
            )

    # Run general validation
    general = diagnose_circuit(circuit)
    if general.has_errors:
        result.add(
            Finding(
                category="topology",
                severity=DiagnosticSeverity.ERROR,
                message="Circuit has validation errors",
                suggestion="Fix topology issues before simulating",
                details=f"Found {general.error_count} error(s)",
            )
        )

    return result
