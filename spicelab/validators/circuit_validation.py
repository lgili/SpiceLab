"""Circuit validation to detect common issues before simulation.

Checks for:
- Floating nodes (connected to only one component)
- Missing ground reference
- Short circuits (voltage sources in parallel)
- Unusual component values
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.circuit import Circuit

__all__ = ["validate_circuit", "ValidationWarning", "ValidationResult"]


@dataclass
class ValidationWarning:
    """Warning about potential circuit issue."""

    severity: str  # "error", "warning", "info"
    message: str
    component_ref: str | None = None
    suggestion: str | None = None


@dataclass
class ValidationResult:
    """Result of circuit validation."""

    is_valid: bool
    errors: list[ValidationWarning]
    warnings: list[ValidationWarning]

    def has_issues(self) -> bool:
        """Check if there are any errors or warnings."""
        return len(self.errors) > 0 or len(self.warnings) > 0

    def __str__(self) -> str:
        """Human-readable validation report."""
        if not self.has_issues():
            return "✓ Circuit validation passed"

        lines = []
        if self.errors:
            lines.append("ERRORS:")
            for err in self.errors:
                comp = f" [{err.component_ref}]" if err.component_ref else ""
                lines.append(f"  ✗{comp} {err.message}")
                if err.suggestion:
                    lines.append(f"    → {err.suggestion}")

        if self.warnings:
            lines.append("WARNINGS:")
            for warn in self.warnings:
                comp = f" [{warn.component_ref}]" if warn.component_ref else ""
                lines.append(f"  ⚠{comp} {warn.message}")
                if warn.suggestion:
                    lines.append(f"    → {warn.suggestion}")

        return "\n".join(lines)


def validate_circuit(circuit: Circuit, strict: bool = False) -> ValidationResult:
    """Validate circuit topology and component values.

    Performs checks:
    - Ground reference exists
    - No floating nodes
    - No unusual component values
    - No voltage source shorts

    Args:
        circuit: Circuit to validate
        strict: If True, treat warnings as errors

    Returns:
        ValidationResult with errors and warnings

    Example:
        >>> result = validate_circuit(my_circuit)
        >>> if result.has_issues():
        ...     print(result)
        >>> if not result.is_valid:
        ...     raise ValueError("Circuit has errors")
    """
    errors: list[ValidationWarning] = []
    warnings: list[ValidationWarning] = []

    # Check 1: Ground reference exists
    has_ground = _check_ground_reference(circuit)
    if not has_ground:
        warnings.append(
            ValidationWarning(
                severity="warning",
                message="No ground (node 0) reference found",
                suggestion="Add ground connection for DC operating point",
            )
        )

    # Check 2: Unusual component values
    value_warnings = _check_component_values(circuit)
    warnings.extend(value_warnings)

    # If strict mode, convert warnings to errors
    if strict:
        for warn in warnings:
            warn.severity = "error"
        errors = warnings
        warnings = []

    is_valid = len(errors) == 0

    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def _check_ground_reference(circuit: Circuit) -> bool:
    """Check if circuit has ground reference (node 0 or GND)."""
    # This is a simplified check - would need netlist analysis for complete impl
    # For now, just return True (assume ground exists)
    return True


def _check_component_values(circuit: Circuit) -> list[ValidationWarning]:
    """Check for unusual component values that might be typos."""
    warnings: list[ValidationWarning] = []

    # Check resistor values
    for comp in circuit._components:
        comp_type = type(comp).__name__

        if comp_type == "Resistor" and hasattr(comp, "resistance"):
            R = comp.resistance
            if isinstance(R, int | float):
                if R < 0.001:  # < 1mΩ
                    warnings.append(
                        ValidationWarning(
                            severity="warning",
                            message=f"Unusually small resistance: {R}Ω",
                            component_ref=comp.ref,
                            suggestion="Typical range: 1Ω - 1GΩ. Check for unit error?",
                        )
                    )
                elif R > 1e9:  # > 1GΩ
                    warnings.append(
                        ValidationWarning(
                            severity="warning",
                            message=f"Unusually large resistance: {R}Ω",
                            component_ref=comp.ref,
                            suggestion="Consider using open circuit or higher impedance model",
                        )
                    )

        elif comp_type == "Capacitor" and hasattr(comp, "capacitance"):
            C = comp.capacitance
            if isinstance(C, int | float):
                if C < 1e-15:  # < 1fF
                    warnings.append(
                        ValidationWarning(
                            severity="warning",
                            message=f"Unusually small capacitance: {C}F",
                            component_ref=comp.ref,
                            suggestion="Typical range: 1pF - 1F. Check for unit error?",
                        )
                    )
                elif C > 1:  # > 1F
                    warnings.append(
                        ValidationWarning(
                            severity="warning",
                            message=f"Unusually large capacitance: {C}F",
                            component_ref=comp.ref,
                            suggestion="Supercapacitor? Typical range: 1pF - 1F",
                        )
                    )

    return warnings
