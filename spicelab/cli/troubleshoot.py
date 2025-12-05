"""CLI for circuit troubleshooting.

Diagnoses and provides fixes for simulation issues.

Usage:
    spicelab-troubleshoot circuit.py
    spicelab-troubleshoot circuit.py --convergence
    spicelab-troubleshoot circuit.py --interactive
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from ..core.circuit import Circuit
from ..troubleshooting import Troubleshooter


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


def format_result_json(diagnostics: Any) -> dict[str, Any]:
    """Format diagnostic result as JSON-serializable dict."""
    return {
        "circuit_name": diagnostics.circuit_name,
        "component_count": diagnostics.component_count,
        "has_issues": diagnostics.has_issues,
        "has_errors": diagnostics.has_errors,
        "error_count": diagnostics.error_count,
        "warning_count": diagnostics.warning_count,
        "summary": diagnostics.summary(),
        "findings": [
            {
                "category": f.category,
                "severity": f.severity.value,
                "message": f.message,
                "suggestion": f.suggestion,
                "details": f.details,
            }
            for f in diagnostics.findings
        ],
    }


def main() -> int:
    """Main entry point for spicelab-troubleshoot CLI."""
    parser = argparse.ArgumentParser(
        prog="spicelab-troubleshoot",
        description="Diagnose and fix SPICE circuit simulation issues",
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to Python file containing circuit definition",
    )
    parser.add_argument(
        "--convergence",
        "-c",
        action="store_true",
        help="Focus on convergence issues",
    )
    parser.add_argument(
        "--error-message",
        "-e",
        type=str,
        default="",
        help="Error message from simulator (for convergence diagnosis)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run interactive troubleshooting session",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--quick-fix",
        "-q",
        action="store_true",
        help="Just list quick fix suggestions",
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

    # Create troubleshooter
    ts = Troubleshooter(circuit, console)

    # Run appropriate mode
    if args.interactive:
        if args.json:
            console.print("[yellow]Warning: --json ignored in interactive mode[/yellow]")
        ts.interactive()
        return 0

    if args.quick_fix:
        fixes = ts.quick_fix()
        if args.json:
            print(json.dumps({"fixes": fixes}))
        else:
            if fixes:
                console.print("[bold]Quick Fix Suggestions:[/bold]\n")
                for fix in fixes:
                    console.print(f"  • {fix}")
            else:
                console.print("[green]No issues found[/green]")
        return 0

    if args.convergence:
        diagnostics = ts.diagnose_convergence(
            error_message=args.error_message,
            verbose=not args.json,
        )
    else:
        diagnostics = ts.diagnose(verbose=not args.json)

    if args.json:
        print(json.dumps(format_result_json(diagnostics), indent=2))

    return 0 if not diagnostics.has_errors else 1


if __name__ == "__main__":
    sys.exit(main())
