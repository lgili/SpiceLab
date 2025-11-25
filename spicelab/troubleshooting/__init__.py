"""Interactive troubleshooting tools for circuit simulation.

Provides diagnostics and guided fixes for common simulation issues:
- Convergence failures
- Empty results
- Simulation timeouts
- Unexpected output

Example:
    >>> from spicelab.troubleshooting import Troubleshooter
    >>> ts = Troubleshooter(circuit)
    >>> ts.diagnose()  # Auto-diagnose issues
    >>> ts.interactive()  # Interactive questionnaire
"""

from __future__ import annotations

from .diagnostics import (
    DiagnosticResult,
    diagnose_circuit,
    diagnose_convergence,
    diagnose_empty_results,
)
from .interactive import Troubleshooter

__all__ = [
    "Troubleshooter",
    "DiagnosticResult",
    "diagnose_circuit",
    "diagnose_convergence",
    "diagnose_empty_results",
]
