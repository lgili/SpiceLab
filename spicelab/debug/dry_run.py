"""Dry-run mode for simulation validation without execution.

Validates the complete simulation setup without actually running the simulation:
- Circuit validity (connections, ground, floating nodes)
- Netlist generation
- Analysis configuration
- Engine compatibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..core.types import AnalysisSpec


@dataclass
class ValidationIssue:
    """A single validation issue found during dry-run."""

    level: str  # "error", "warning", "info"
    category: str  # "circuit", "netlist", "analysis", "engine"
    message: str
    suggestion: str | None = None

    def __str__(self) -> str:
        prefix = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(self.level, "•")
        result = f"{prefix} [{self.category}] {self.message}"
        if self.suggestion:
            result += f"\n  Suggestion: {self.suggestion}"
        return result


@dataclass
class DryRunResult:
    """Result of a dry-run validation.

    Contains validation status, generated netlist, and any issues found.
    """

    valid: bool
    circuit_name: str
    netlist: str
    analyses: list[dict[str, Any]]
    engine: str
    issues: list[ValidationIssue] = field(default_factory=list)
    component_count: int = 0
    node_count: int = 0
    estimated_runtime: str | None = None

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.level == "warning"]

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"Dry-Run Validation: {'PASSED ✓' if self.valid else 'FAILED ✗'}",
            "=" * 60,
            "",
            f"Circuit: {self.circuit_name}",
            f"Components: {self.component_count}",
            f"Nodes: {self.node_count}",
            f"Engine: {self.engine}",
            "",
            "Analyses:",
        ]

        for analysis in self.analyses:
            mode = analysis.get("mode", "unknown")
            args = {k: v for k, v in analysis.items() if k != "mode"}
            lines.append(f"  • {mode}: {args}")

        if self.issues:
            lines.append("")
            lines.append("Issues Found:")
            for issue in self.issues:
                lines.append(f"  {issue}")

        if self.estimated_runtime:
            lines.append("")
            lines.append(f"Estimated runtime: {self.estimated_runtime}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def raise_if_invalid(self) -> None:
        """Raise exception if validation failed.

        Raises:
            ValueError: If there are validation errors
        """
        if not self.valid:
            error_msgs = [str(e) for e in self.errors]
            raise ValueError(
                f"Dry-run validation failed with {len(self.errors)} error(s):\n"
                + "\n".join(error_msgs)
            )


def dry_run(
    circuit: Any,
    analyses: Sequence[AnalysisSpec] | None = None,
    *,
    engine: str = "ngspice",
    validate_circuit: bool = True,
) -> DryRunResult:
    """Validate simulation setup without running.

    Performs comprehensive validation of the simulation configuration:
    - Circuit structure (connections, ground, floating nodes)
    - Netlist generation
    - Analysis parameter validity
    - Engine compatibility

    Args:
        circuit: Circuit to validate
        analyses: Analysis specifications (optional)
        engine: Simulation engine to validate against (default "ngspice")
        validate_circuit: Run circuit validation checks (default True)

    Returns:
        DryRunResult with validation status and details

    Example:
        >>> from spicelab.debug import dry_run
        >>> from spicelab.templates import rc_lowpass
        >>> from spicelab.core.types import AnalysisSpec
        >>>
        >>> circuit = rc_lowpass(fc=1000)
        >>> analyses = [AnalysisSpec("ac", {"sweep_type": "dec", "n": 20,
        ...                                  "fstart": 1, "fstop": 1e6})]
        >>> result = dry_run(circuit, analyses)
        >>> print(result)
        ============================================================
        Dry-Run Validation: PASSED ✓
        ============================================================

        Circuit: RC_Filter
        Components: 2
        Nodes: 3
        Engine: ngspice

        Analyses:
          • ac: {'sweep_type': 'dec', 'n': 20, 'fstart': 1, 'fstop': 1000000.0}

        ============================================================

        >>> # Check validity programmatically
        >>> if result.valid:
        ...     print("Ready to simulate!")
        ... else:
        ...     for error in result.errors:
        ...         print(f"Error: {error.message}")
    """
    issues: list[ValidationIssue] = []

    # Get circuit info
    circuit_name = getattr(circuit, "name", "Unknown")
    components = getattr(circuit, "_components", [])
    component_count = len(components)

    # Try to generate netlist
    netlist = ""
    try:
        netlist = circuit.build_netlist()
    except Exception as e:
        issues.append(
            ValidationIssue(
                level="error",
                category="netlist",
                message=f"Failed to generate netlist: {e}",
                suggestion="Check circuit connections and component definitions",
            )
        )

    # Count nodes from netlist
    node_count = _count_nodes(netlist) if netlist else 0

    # Run circuit validation if requested
    if validate_circuit and netlist:
        circuit_issues = _validate_circuit(circuit, netlist)
        issues.extend(circuit_issues)

    # Validate analyses
    analyses_list: list[dict[str, Any]] = []
    if analyses:
        for analysis in analyses:
            mode = getattr(analysis, "mode", str(analysis))
            args = getattr(analysis, "args", {})
            analyses_list.append({"mode": mode, **args})

            analysis_issues = _validate_analysis(mode, args)
            issues.extend(analysis_issues)
    else:
        issues.append(
            ValidationIssue(
                level="warning",
                category="analysis",
                message="No analyses specified",
                suggestion="Add at least one analysis (op, ac, tran, dc, noise)",
            )
        )

    # Validate engine
    engine_issues = _validate_engine(engine)
    issues.extend(engine_issues)

    # Estimate runtime
    estimated_runtime = _estimate_runtime(analyses_list, component_count)

    # Determine overall validity (no errors)
    valid = not any(i.level == "error" for i in issues)

    return DryRunResult(
        valid=valid,
        circuit_name=circuit_name,
        netlist=netlist,
        analyses=analyses_list,
        engine=engine,
        issues=issues,
        component_count=component_count,
        node_count=node_count,
        estimated_runtime=estimated_runtime,
    )


def _count_nodes(netlist: str) -> int:
    """Count unique nodes in netlist."""
    nodes: set[str] = set()
    for line in netlist.split("\n"):
        line = line.strip()
        if not line or line.startswith("*") or line.startswith("."):
            continue
        parts = line.split()
        if len(parts) >= 3:
            # Nodes are typically in positions 1, 2 (and sometimes 3, 4)
            for part in parts[1:5]:
                if part and not part.startswith("{") and not _is_value(part):
                    nodes.add(part.lower())
    return len(nodes)


def _is_value(s: str) -> bool:
    """Check if string looks like a component value."""
    try:
        float(s.rstrip("fpnumkMGT"))
        return True
    except ValueError:
        return False


def _validate_circuit(circuit: Any, netlist: str) -> list[ValidationIssue]:
    """Validate circuit structure."""
    issues: list[ValidationIssue] = []

    # Check for ground
    if "0" not in netlist and "gnd" not in netlist.lower():
        issues.append(
            ValidationIssue(
                level="error",
                category="circuit",
                message="No ground reference found",
                suggestion="Connect at least one node to GND (node 0)",
            )
        )

    # Check for sources
    has_source = False
    for line in netlist.split("\n"):
        line_upper = line.strip().upper()
        if line_upper and line_upper[0] in "VIE":
            has_source = True
            break

    if not has_source:
        issues.append(
            ValidationIssue(
                level="warning",
                category="circuit",
                message="No voltage or current sources found",
                suggestion="Add a source (Vdc, Vac, Idc, etc.) for simulation",
            )
        )

    # Try to use circuit validator if available
    try:
        from ..validators.circuit_validation import validate_circuit

        validation_result = validate_circuit(circuit)
        if hasattr(validation_result, "errors"):
            for error in validation_result.errors:
                issues.append(
                    ValidationIssue(
                        level="error",
                        category="circuit",
                        message=str(error),
                    )
                )
        if hasattr(validation_result, "warnings"):
            for warning in validation_result.warnings:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        category="circuit",
                        message=str(warning),
                    )
                )
    except ImportError:
        pass
    except Exception:
        pass

    return issues


def _validate_analysis(mode: str, args: dict[str, Any]) -> list[ValidationIssue]:
    """Validate analysis parameters."""
    issues: list[ValidationIssue] = []

    valid_modes = {"op", "ac", "dc", "tran", "noise", "sens", "tf", "pz"}
    if mode not in valid_modes:
        issues.append(
            ValidationIssue(
                level="error",
                category="analysis",
                message=f"Unknown analysis type: {mode}",
                suggestion=f"Use one of: {', '.join(sorted(valid_modes))}",
            )
        )
        return issues

    if mode == "ac":
        if "fstart" not in args or "fstop" not in args:
            issues.append(
                ValidationIssue(
                    level="error",
                    category="analysis",
                    message="AC analysis requires fstart and fstop",
                )
            )
        elif args.get("fstart", 0) >= args.get("fstop", 0):
            issues.append(
                ValidationIssue(
                    level="error",
                    category="analysis",
                    message="AC fstart must be less than fstop",
                )
            )
        if args.get("fstart", 1) <= 0:
            issues.append(
                ValidationIssue(
                    level="warning",
                    category="analysis",
                    message="AC fstart should be > 0 for log sweep",
                )
            )

    elif mode == "tran":
        if "tstop" not in args:
            issues.append(
                ValidationIssue(
                    level="error",
                    category="analysis",
                    message="Transient analysis requires tstop",
                )
            )
        tstep = args.get("tstep", 0)
        tstop = args.get("tstop", 0)
        if tstep and tstop and tstep > tstop / 10:
            issues.append(
                ValidationIssue(
                    level="warning",
                    category="analysis",
                    message="Transient timestep is large relative to duration",
                    suggestion="Consider smaller timestep for better resolution",
                )
            )

    elif mode == "dc":
        if "src" not in args:
            issues.append(
                ValidationIssue(
                    level="error",
                    category="analysis",
                    message="DC sweep requires src (source name)",
                )
            )
        if "start" not in args or "stop" not in args:
            issues.append(
                ValidationIssue(
                    level="error",
                    category="analysis",
                    message="DC sweep requires start and stop values",
                )
            )

    elif mode == "noise":
        if "output" not in args:
            issues.append(
                ValidationIssue(
                    level="error",
                    category="analysis",
                    message="Noise analysis requires output node",
                )
            )
        if "src" not in args:
            issues.append(
                ValidationIssue(
                    level="error",
                    category="analysis",
                    message="Noise analysis requires input source (src)",
                )
            )

    return issues


def _validate_engine(engine: str) -> list[ValidationIssue]:
    """Validate engine availability and configuration."""
    issues: list[ValidationIssue] = []

    valid_engines = {
        "ngspice",
        "ngspice-cli",
        "ngspice-shared",
        "ltspice",
        "ltspice-cli",
        "xyce",
        "xyce-cli",
    }

    if engine not in valid_engines:
        issues.append(
            ValidationIssue(
                level="error",
                category="engine",
                message=f"Unknown engine: {engine}",
                suggestion=f"Use one of: {', '.join(sorted(valid_engines))}",
            )
        )
        return issues

    # Try to check if engine is available
    try:
        from ..engines.factory import create_simulator

        create_simulator(engine)
    except Exception as e:
        issues.append(
            ValidationIssue(
                level="warning",
                category="engine",
                message=f"Engine '{engine}' may not be available: {e}",
                suggestion="Install the engine or use a different one",
            )
        )

    return issues


def _estimate_runtime(analyses: list[dict[str, Any]], component_count: int) -> str | None:
    """Estimate simulation runtime based on analysis and circuit complexity."""
    if not analyses:
        return None

    total_points = 0
    for analysis in analyses:
        mode = analysis.get("mode", "")
        if mode == "op":
            total_points += 1
        elif mode == "ac":
            n = analysis.get("n", 20)
            fstart = analysis.get("fstart", 1)
            fstop = analysis.get("fstop", 1e9)
            if fstart > 0 and fstop > fstart:
                import math

                decades = math.log10(fstop / fstart)
                total_points += int(n * decades)
        elif mode == "tran":
            tstep = analysis.get("tstep", 1e-6)
            tstop = analysis.get("tstop", 1e-3)
            if tstep > 0:
                total_points += int(tstop / tstep)
        elif mode == "dc":
            step = analysis.get("step", 0.1)
            start = analysis.get("start", 0)
            stop = analysis.get("stop", 1)
            if step > 0:
                total_points += int((stop - start) / step)

    if total_points == 0:
        return None

    # Very rough estimate: ~1µs per point per component for simple circuits
    estimated_seconds = total_points * component_count * 1e-6

    if estimated_seconds < 0.1:
        return "< 0.1s (fast)"
    elif estimated_seconds < 1:
        return f"~{estimated_seconds:.1f}s"
    elif estimated_seconds < 60:
        return f"~{estimated_seconds:.0f}s"
    else:
        return f"~{estimated_seconds / 60:.1f} minutes"


__all__ = ["dry_run", "DryRunResult", "ValidationIssue"]
