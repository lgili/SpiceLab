"""Advanced Design Rule Checks (DRC) for circuit validation.

This module extends basic validation with advanced checks:
- Power budget validation (current limits, power dissipation)
- Signal integrity (impedance matching, fanout limits)
- Component ratings (voltage, current, power limits)
- Constraint templates for reusable validation rules

Part of Sprint 2 (M11) - Validation and Quality improvements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .circuit_validation import ValidationWarning

if TYPE_CHECKING:
    from ..core.circuit import Circuit


class Severity(Enum):
    """Severity levels for DRC violations."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class DRCRule:
    """A design rule check definition.

    Attributes:
        name: Unique rule identifier
        description: Human-readable description
        category: Category (power, signal_integrity, ratings, custom)
        check_fn: Function that performs the check
        severity: Default severity level
        enabled: Whether rule is active
    """

    name: str
    description: str
    category: str
    check_fn: Callable[["Circuit", "DRCContext"], list[ValidationWarning]]
    severity: Severity = Severity.WARNING
    enabled: bool = True


@dataclass
class DRCContext:
    """Context for DRC checks with constraints and limits.

    Attributes:
        max_current_ma: Maximum allowed current per net (mA)
        max_power_mw: Maximum allowed power dissipation per component (mW)
        max_voltage: Maximum allowed voltage on any node (V)
        min_resistance: Minimum allowed resistance (Ohm)
        max_fanout: Maximum number of loads per output
        target_impedance: Target impedance for matching (Ohm)
        impedance_tolerance: Allowed impedance mismatch (%)
        custom_limits: Dictionary of custom constraint values
    """

    max_current_ma: float = 100.0
    max_power_mw: float = 250.0
    max_voltage: float = 50.0
    min_resistance: float = 1.0
    max_fanout: int = 10
    target_impedance: float | None = None
    impedance_tolerance: float = 10.0
    custom_limits: dict[str, Any] = field(default_factory=dict)


@dataclass
class DRCReport:
    """Complete DRC report with all violations and metadata.

    Attributes:
        circuit_name: Name of the checked circuit
        timestamp: When the check was performed
        context: DRC context used
        violations: List of all violations found
        rules_checked: Number of rules that were run
        rules_passed: Number of rules that passed
        summary: Brief summary of results
    """

    circuit_name: str
    timestamp: datetime
    context: DRCContext
    violations: list[ValidationWarning]
    rules_checked: int = 0
    rules_passed: int = 0
    summary: str = ""

    @property
    def has_errors(self) -> bool:
        """Check if any violations are errors."""
        return any(v.severity == "error" for v in self.violations)

    @property
    def has_warnings(self) -> bool:
        """Check if any violations are warnings."""
        return any(v.severity == "warning" for v in self.violations)

    @property
    def passed(self) -> bool:
        """Check if all rules passed (no errors)."""
        return not self.has_errors

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "circuit_name": self.circuit_name,
            "timestamp": self.timestamp.isoformat(),
            "passed": self.passed,
            "summary": self.summary,
            "statistics": {
                "rules_checked": self.rules_checked,
                "rules_passed": self.rules_passed,
                "errors": sum(1 for v in self.violations if v.severity == "error"),
                "warnings": sum(1 for v in self.violations if v.severity == "warning"),
                "info": sum(1 for v in self.violations if v.severity == "info"),
            },
            "context": {
                "max_current_ma": self.context.max_current_ma,
                "max_power_mw": self.context.max_power_mw,
                "max_voltage": self.context.max_voltage,
                "min_resistance": self.context.min_resistance,
                "max_fanout": self.context.max_fanout,
            },
            "violations": [
                {
                    "severity": v.severity,
                    "message": v.message,
                    "component": v.component_ref,
                    "suggestion": v.suggestion,
                }
                for v in self.violations
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Export report as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_html(self) -> str:
        """Export report as HTML string."""
        status_color = "#4CAF50" if self.passed else "#f44336"
        status_text = "PASSED" if self.passed else "FAILED"

        violations_html = ""
        for v in self.violations:
            icon = "❌" if v.severity == "error" else "⚠️" if v.severity == "warning" else "ℹ️"
            color = "#f44336" if v.severity == "error" else "#ff9800" if v.severity == "warning" else "#2196F3"
            comp_str = f" <code>{v.component_ref}</code>" if v.component_ref else ""
            sugg_str = f"<br><em>Suggestion: {v.suggestion}</em>" if v.suggestion else ""
            violations_html += f"""
            <div style="padding: 10px; margin: 5px 0; border-left: 4px solid {color}; background: #f5f5f5;">
                <strong>{icon} {v.severity.upper()}</strong>{comp_str}<br>
                {v.message}{sugg_str}
            </div>
            """

        if not violations_html:
            violations_html = "<p style='color: #4CAF50;'>No violations found.</p>"

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>DRC Report - {self.circuit_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .status {{ color: {status_color}; font-size: 24px; font-weight: bold; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat {{ padding: 15px; background: #e3f2fd; border-radius: 8px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>Design Rule Check Report</h1>
    <p><strong>Circuit:</strong> {self.circuit_name}</p>
    <p><strong>Date:</strong> {self.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p class="status">Status: {status_text}</p>

    <div class="stats">
        <div class="stat">
            <div class="stat-value">{self.rules_checked}</div>
            <div>Rules Checked</div>
        </div>
        <div class="stat">
            <div class="stat-value">{self.rules_passed}</div>
            <div>Rules Passed</div>
        </div>
        <div class="stat">
            <div class="stat-value">{sum(1 for v in self.violations if v.severity == "error")}</div>
            <div>Errors</div>
        </div>
        <div class="stat">
            <div class="stat-value">{sum(1 for v in self.violations if v.severity == "warning")}</div>
            <div>Warnings</div>
        </div>
    </div>

    <h2>Violations</h2>
    {violations_html}

    <h2>Constraints Used</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Max Current</td><td>{self.context.max_current_ma} mA</td></tr>
        <tr><td>Max Power</td><td>{self.context.max_power_mw} mW</td></tr>
        <tr><td>Max Voltage</td><td>{self.context.max_voltage} V</td></tr>
        <tr><td>Min Resistance</td><td>{self.context.min_resistance} Ω</td></tr>
        <tr><td>Max Fanout</td><td>{self.context.max_fanout}</td></tr>
    </table>

    <p><em>Generated by SpiceLab DRC</em></p>
</body>
</html>
"""

    def save(self, path: str | Path, format: str = "json") -> Path:
        """Save report to file.

        Args:
            path: Output file path
            format: Output format ('json' or 'html')

        Returns:
            Path to saved file
        """
        path = Path(path)
        if format == "html":
            content = self.to_html()
            if not path.suffix:
                path = path.with_suffix(".html")
        else:
            content = self.to_json()
            if not path.suffix:
                path = path.with_suffix(".json")

        path.write_text(content)
        return path


class ConstraintTemplate:
    """Reusable constraint template for common design scenarios.

    Provides pre-configured DRC contexts for typical design rules.
    """

    @staticmethod
    def low_power() -> DRCContext:
        """Constraints for low-power designs (battery operated)."""
        return DRCContext(
            max_current_ma=10.0,
            max_power_mw=50.0,
            max_voltage=5.0,
            min_resistance=100.0,
            max_fanout=5,
        )

    @staticmethod
    def high_power() -> DRCContext:
        """Constraints for high-power designs (power supplies, motors)."""
        return DRCContext(
            max_current_ma=10000.0,  # 10A
            max_power_mw=100000.0,  # 100W
            max_voltage=100.0,
            min_resistance=0.01,
            max_fanout=20,
        )

    @staticmethod
    def precision_analog() -> DRCContext:
        """Constraints for precision analog designs."""
        return DRCContext(
            max_current_ma=50.0,
            max_power_mw=100.0,
            max_voltage=15.0,
            min_resistance=10.0,
            max_fanout=3,
            target_impedance=10000.0,  # 10k typical for precision
            impedance_tolerance=5.0,
        )

    @staticmethod
    def rf_design() -> DRCContext:
        """Constraints for RF designs (impedance matching critical)."""
        return DRCContext(
            max_current_ma=100.0,
            max_power_mw=500.0,
            max_voltage=12.0,
            min_resistance=0.1,
            max_fanout=2,
            target_impedance=50.0,  # Standard RF impedance
            impedance_tolerance=10.0,
        )

    @staticmethod
    def digital_logic() -> DRCContext:
        """Constraints for digital logic designs."""
        return DRCContext(
            max_current_ma=500.0,
            max_power_mw=1000.0,
            max_voltage=5.0,
            min_resistance=10.0,
            max_fanout=10,
        )

    @staticmethod
    def automotive() -> DRCContext:
        """Constraints for automotive applications."""
        return DRCContext(
            max_current_ma=5000.0,
            max_power_mw=25000.0,  # 25W
            max_voltage=60.0,  # Handles load dump
            min_resistance=0.1,
            max_fanout=15,
        )


class AdvancedDRC:
    """Advanced Design Rule Checker with extensible rules.

    Example:
        >>> drc = AdvancedDRC()
        >>> context = ConstraintTemplate.low_power()
        >>> report = drc.check(circuit, context)
        >>> if not report.passed:
        ...     print(report.to_json())
        ...     report.save("drc_report.html", format="html")
    """

    def __init__(self):
        """Initialize DRC with default rules."""
        self._rules: list[DRCRule] = []
        self._register_default_rules()

    def _register_default_rules(self):
        """Register built-in DRC rules."""
        self.add_rule(
            DRCRule(
                name="min_resistance",
                description="Check for resistances below minimum limit",
                category="power",
                check_fn=self._check_min_resistance,
                severity=Severity.WARNING,
            )
        )
        self.add_rule(
            DRCRule(
                name="power_dissipation",
                description="Check estimated power dissipation limits",
                category="power",
                check_fn=self._check_power_dissipation,
                severity=Severity.WARNING,
            )
        )
        self.add_rule(
            DRCRule(
                name="voltage_ratings",
                description="Check component voltage ratings",
                category="ratings",
                check_fn=self._check_voltage_ratings,
                severity=Severity.WARNING,
            )
        )
        self.add_rule(
            DRCRule(
                name="fanout_limit",
                description="Check output fanout limits",
                category="signal_integrity",
                check_fn=self._check_fanout,
                severity=Severity.WARNING,
            )
        )
        self.add_rule(
            DRCRule(
                name="impedance_matching",
                description="Check impedance matching for RF/precision",
                category="signal_integrity",
                check_fn=self._check_impedance_matching,
                severity=Severity.INFO,
            )
        )
        self.add_rule(
            DRCRule(
                name="decoupling_caps",
                description="Check for decoupling capacitors near power pins",
                category="signal_integrity",
                check_fn=self._check_decoupling_caps,
                severity=Severity.INFO,
            )
        )

    def add_rule(self, rule: DRCRule):
        """Add a custom DRC rule."""
        self._rules.append(rule)

    def remove_rule(self, name: str):
        """Remove a rule by name."""
        self._rules = [r for r in self._rules if r.name != name]

    def enable_rule(self, name: str, enabled: bool = True):
        """Enable or disable a rule."""
        for rule in self._rules:
            if rule.name == name:
                rule.enabled = enabled
                return
        raise ValueError(f"Rule '{name}' not found")

    def list_rules(self) -> list[dict[str, Any]]:
        """List all registered rules."""
        return [
            {
                "name": r.name,
                "description": r.description,
                "category": r.category,
                "severity": r.severity.value,
                "enabled": r.enabled,
            }
            for r in self._rules
        ]

    def check(
        self,
        circuit: "Circuit",
        context: DRCContext | None = None,
        categories: list[str] | None = None,
    ) -> DRCReport:
        """Run DRC checks on a circuit.

        Args:
            circuit: Circuit to check
            context: DRC context with constraints (default: standard limits)
            categories: List of categories to check (None = all)

        Returns:
            DRCReport with all violations
        """
        if context is None:
            context = DRCContext()

        violations: list[ValidationWarning] = []
        rules_checked = 0
        rules_passed = 0

        for rule in self._rules:
            if not rule.enabled:
                continue
            if categories and rule.category not in categories:
                continue

            rules_checked += 1
            try:
                rule_violations = rule.check_fn(circuit, context)
                if not rule_violations:
                    rules_passed += 1
                else:
                    violations.extend(rule_violations)
            except Exception as e:
                violations.append(
                    ValidationWarning(
                        severity="error",
                        message=f"Rule '{rule.name}' failed: {e}",
                        suggestion="Check rule implementation",
                    )
                )

        # Generate summary
        error_count = sum(1 for v in violations if v.severity == "error")
        warning_count = sum(1 for v in violations if v.severity == "warning")

        if error_count == 0 and warning_count == 0:
            summary = f"All {rules_checked} rules passed"
        else:
            summary = f"{error_count} errors, {warning_count} warnings in {rules_checked} rules"

        return DRCReport(
            circuit_name=circuit.name,
            timestamp=datetime.now(),
            context=context,
            violations=violations,
            rules_checked=rules_checked,
            rules_passed=rules_passed,
            summary=summary,
        )

    # Built-in rule implementations

    def _check_min_resistance(
        self, circuit: "Circuit", context: DRCContext
    ) -> list[ValidationWarning]:
        """Check for resistances below minimum limit."""
        warnings = []

        for comp in circuit._components:
            if type(comp).__name__ == "Resistor":
                # Try 'value' first (used by Resistor), then 'resistance'
                r_value = getattr(comp, "value", None) or getattr(comp, "resistance", None)
                if isinstance(r_value, (int, float)) and r_value < context.min_resistance:
                    warnings.append(
                        ValidationWarning(
                            severity="warning",
                            message=(
                                f"Resistance {r_value}Ω below minimum {context.min_resistance}Ω"
                            ),
                            component_ref=comp.ref,
                            suggestion="Check for short circuit or use higher resistance",
                        )
                    )

        return warnings

    def _check_power_dissipation(
        self, circuit: "Circuit", context: DRCContext
    ) -> list[ValidationWarning]:
        """Estimate and check power dissipation in resistors.

        Note: This is a rough estimate based on voltage sources in circuit.
        Actual power requires simulation results.
        """
        warnings = []

        # Find max voltage in circuit from voltage sources
        max_v = 0.0
        for comp in circuit._components:
            comp_type = type(comp).__name__
            if comp_type in ("Vdc", "Vac", "VoltageSource"):
                v_value = getattr(comp, "value", None) or getattr(comp, "dc_value", None) or 0
                if isinstance(v_value, (int, float)):
                    max_v = max(max_v, abs(v_value))

        if max_v == 0:
            return warnings  # Can't estimate without voltage

        # Check resistors for worst-case power
        for comp in circuit._components:
            if type(comp).__name__ == "Resistor":
                r_value = getattr(comp, "value", None) or getattr(comp, "resistance", None)
                if isinstance(r_value, (int, float)) and r_value > 0:
                    # Worst case: full voltage across resistor
                    power_mw = (max_v**2 / r_value) * 1000
                    if power_mw > context.max_power_mw:
                        warnings.append(
                            ValidationWarning(
                                severity="warning",
                                message=(
                                    f"Estimated max power {power_mw:.1f}mW "
                                    f"exceeds limit {context.max_power_mw}mW"
                                ),
                                component_ref=comp.ref,
                                suggestion="Use higher power rating or increase resistance",
                            )
                        )

        return warnings

    def _check_voltage_ratings(
        self, circuit: "Circuit", context: DRCContext
    ) -> list[ValidationWarning]:
        """Check if voltage sources exceed maximum voltage."""
        warnings = []

        for comp in circuit._components:
            comp_type = type(comp).__name__
            if comp_type in ("Vdc", "Vac", "VoltageSource"):
                v_value = getattr(comp, "value", None) or getattr(comp, "dc_value", None) or 0
                if isinstance(v_value, (int, float)) and abs(v_value) > context.max_voltage:
                    warnings.append(
                        ValidationWarning(
                            severity="warning",
                            message=(
                                f"Voltage {abs(v_value)}V exceeds limit {context.max_voltage}V"
                            ),
                            component_ref=comp.ref,
                            suggestion="Check component voltage ratings",
                        )
                    )

        return warnings

    def _check_fanout(
        self, circuit: "Circuit", context: DRCContext
    ) -> list[ValidationWarning]:
        """Check output fanout (number of loads per driving output)."""
        from collections import defaultdict

        warnings = []

        # Build net connection count
        net_connections: dict[Any, int] = defaultdict(int)

        for comp in circuit._components:
            for port in comp.ports:
                net = circuit._port_to_net.get(port)
                if net is not None:
                    canonical = circuit._get_canonical_net(net)
                    net_connections[canonical] += 1

        # Check for high fanout
        for net, count in net_connections.items():
            if count > context.max_fanout:
                net_name = getattr(net, "name", "unnamed")
                warnings.append(
                    ValidationWarning(
                        severity="warning",
                        message=(
                            f"High fanout on net '{net_name}': "
                            f"{count} connections (limit: {context.max_fanout})"
                        ),
                        suggestion="Consider adding a buffer or reducing loads",
                    )
                )

        return warnings

    def _check_impedance_matching(
        self, circuit: "Circuit", context: DRCContext
    ) -> list[ValidationWarning]:
        """Check impedance matching for RF/precision designs."""
        if context.target_impedance is None:
            return []  # No target impedance specified

        warnings = []
        target = context.target_impedance
        tolerance_pct = context.impedance_tolerance

        for comp in circuit._components:
            if type(comp).__name__ == "Resistor":
                r_value = getattr(comp, "value", None) or getattr(comp, "resistance", None)
                if isinstance(r_value, (int, float)):
                    # Check if resistor is close to target impedance
                    error_pct = abs(r_value - target) / target * 100
                    # Only warn if it looks like it should match (within 2x)
                    if 0.5 * target < r_value < 2 * target and error_pct > tolerance_pct:
                        warnings.append(
                            ValidationWarning(
                                severity="info",
                                message=(
                                    f"Impedance mismatch: {r_value}Ω vs target {target}Ω "
                                    f"({error_pct:.1f}% error, limit {tolerance_pct}%)"
                                ),
                                component_ref=comp.ref,
                                suggestion=f"Use {target}Ω for impedance matching",
                            )
                        )

        return warnings

    def _check_decoupling_caps(
        self, circuit: "Circuit", context: DRCContext  # noqa: ARG002
    ) -> list[ValidationWarning]:
        """Check for presence of decoupling capacitors."""
        warnings = []

        # Count capacitors and voltage sources
        cap_count = 0
        vsource_count = 0

        for comp in circuit._components:
            comp_type = type(comp).__name__
            if comp_type == "Capacitor":
                cap_count += 1
            elif comp_type in ("Vdc", "Vac", "VoltageSource"):
                vsource_count += 1

        # Simple heuristic: should have at least one cap per voltage source
        if vsource_count > 0 and cap_count < vsource_count:
            warnings.append(
                ValidationWarning(
                    severity="info",
                    message=(
                        f"Found {vsource_count} voltage sources but only "
                        f"{cap_count} capacitors"
                    ),
                    suggestion="Consider adding decoupling capacitors near power sources",
                )
            )

        return warnings


# Convenience function
def run_drc(
    circuit: "Circuit",
    template: str | None = None,
    context: DRCContext | None = None,
) -> DRCReport:
    """Run DRC checks on a circuit.

    Convenience function for quick DRC.

    Args:
        circuit: Circuit to check
        template: Constraint template name ('low_power', 'high_power',
                 'precision_analog', 'rf_design', 'digital_logic', 'automotive')
        context: Custom DRC context (overrides template)

    Returns:
        DRCReport with all violations

    Example:
        >>> report = run_drc(my_circuit, template="low_power")
        >>> if not report.passed:
        ...     print(report)
    """
    if context is None and template:
        templates = {
            "low_power": ConstraintTemplate.low_power,
            "high_power": ConstraintTemplate.high_power,
            "precision_analog": ConstraintTemplate.precision_analog,
            "rf_design": ConstraintTemplate.rf_design,
            "digital_logic": ConstraintTemplate.digital_logic,
            "automotive": ConstraintTemplate.automotive,
        }
        if template not in templates:
            raise ValueError(
                f"Unknown template '{template}'. "
                f"Available: {list(templates.keys())}"
            )
        context = templates[template]()

    drc = AdvancedDRC()
    return drc.check(circuit, context)


__all__ = [
    "AdvancedDRC",
    "ConstraintTemplate",
    "DRCContext",
    "DRCReport",
    "DRCRule",
    "Severity",
    "run_drc",
]
