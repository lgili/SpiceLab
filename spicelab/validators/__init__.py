"""Circuit validation utilities to catch common errors before simulation.

This module provides:
- Basic validation: validate_circuit() for topology checks
- Advanced DRC: AdvancedDRC for power, signal integrity, and rating checks
- Constraint templates: Pre-built configurations for common design scenarios
- Report export: JSON and HTML report generation
"""

from __future__ import annotations

from .advanced_drc import (
    AdvancedDRC,
    ConstraintTemplate,
    DRCContext,
    DRCReport,
    DRCRule,
    Severity,
    run_drc,
)
from .circuit_validation import ValidationResult, ValidationWarning, validate_circuit

__all__ = [
    # Basic validation
    "validate_circuit",
    "ValidationWarning",
    "ValidationResult",
    # Advanced DRC
    "AdvancedDRC",
    "DRCContext",
    "DRCReport",
    "DRCRule",
    "Severity",
    "run_drc",
    # Constraint templates
    "ConstraintTemplate",
]
