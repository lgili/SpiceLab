"""CLI for circuit validation.

Validates circuit files and reports errors/warnings.

Usage:
    spicelab-validate circuit.py
    spicelab-validate circuit.py --strict
    spicelab-validate circuit.py --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..core.circuit import Circuit
from ..validators.circuit_validation import ValidationResult


def load_circuit_from_file(filepath: Path) -> Circuit | None:
    """Load a circuit from a Python file.

    The file should define a circuit in one of these ways:
    1. A variable named 'circuit'
    2. A function named 'create_circuit()' that returns a Circuit
    3. A function named 'build_circuit()' that returns a Circuit

    Args:
        filepath: Path to the Python file.

    Returns:
        Circuit object or None if not found.
    """
    spec = importlib.util.spec_from_file_location("circuit_module", filepath)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading {filepath}: {e}") from e

    # Try different ways to get the circuit
    if hasattr(module, "circuit"):
        obj = module.circuit
        if isinstance(obj, Circuit):
            return obj

    if hasattr(module, "create_circuit"):
        func = module.create_circuit
        if callable(func):
            result = func()
            if isinstance(result, Circuit):
                return result

    if hasattr(module, "build_circuit"):
        func = module.build_circuit
        if callable(func):
            result = func()
            if isinstance(result, Circuit):
                return result

    return None


def format_result_rich(result: ValidationResult, console: Console) -> None:
    """Format validation result with rich styling."""
    if result.is_valid and not result.has_issues():
        console.print("[green]✓ Circuit validation passed[/green]")
        return

    # Build output
    output = Text()

    if result.errors:
        output.append("ERRORS\n", style="bold red")
        for error in result.errors:
            output.append(f"  ✗ {error.message}\n", style="red")
            if error.component_ref:
                output.append(f"    Component: {error.component_ref}\n", style="dim")
            if error.suggestion:
                output.append(f"    Suggestion: {error.suggestion}\n", style="yellow")
        output.append("\n")

    if result.warnings:
        output.append("WARNINGS\n", style="bold yellow")
        for warning in result.warnings:
            output.append(f"  ⚠ {warning.message}\n", style="yellow")
            if warning.component_ref:
                output.append(f"    Component: {warning.component_ref}\n", style="dim")
            if warning.suggestion:
                output.append(f"    Suggestion: {warning.suggestion}\n", style="cyan")

    status = "FAILED" if not result.is_valid else "PASSED WITH WARNINGS"
    border_style = "red" if not result.is_valid else "yellow"

    panel = Panel(
        output,
        title=f"Validation Result: {status}",
        border_style=border_style,
    )
    console.print(panel)


def format_result_json(result: ValidationResult) -> dict[str, Any]:
    """Format validation result as JSON-serializable dict."""
    return {
        "is_valid": result.is_valid,
        "has_issues": result.has_issues(),
        "errors": [
            {
                "severity": e.severity,
                "message": e.message,
                "component_ref": e.component_ref,
                "suggestion": e.suggestion,
            }
            for e in result.errors
        ],
        "warnings": [
            {
                "severity": w.severity,
                "message": w.message,
                "component_ref": w.component_ref,
                "suggestion": w.suggestion,
            }
            for w in result.warnings
        ],
    }


def main() -> int:
    """Main entry point for spicelab-validate CLI."""
    parser = argparse.ArgumentParser(
        prog="spicelab-validate",
        description="Validate SPICE circuit files",
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to Python file containing circuit definition",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output on failure",
    )

    args = parser.parse_args()
    console = Console()

    # Check file exists
    if not args.file.exists():
        if args.json:
            print(json.dumps({"error": f"File not found: {args.file}"}))
        else:
            console.print(f"[red]Error: File not found: {args.file}[/red]")
        return 1

    # Load circuit
    try:
        circuit = load_circuit_from_file(args.file)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]Error loading circuit: {e}[/red]")
        return 1

    if circuit is None:
        msg = (
            f"No circuit found in {args.file}. "
            "Define 'circuit' variable or create_circuit()/build_circuit() function."
        )
        if args.json:
            print(json.dumps({"error": msg}))
        else:
            console.print(f"[red]Error: {msg}[/red]")
        return 1

    # Validate
    result = circuit.validate(strict=args.strict)

    # Output
    if args.json:
        print(json.dumps(format_result_json(result), indent=2))
    elif not args.quiet or not result.is_valid:
        format_result_rich(result, console)

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
