"""Model Parameter Validation for SPICE Components.

This module provides validation utilities for SPICE model parameters,
ensuring that model cards are syntactically correct and parameters
are within reasonable ranges.

Part of Sprint 4 (M9) - Model Library improvements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelType(Enum):
    """SPICE model types."""

    DIODE = "D"
    NPN = "NPN"
    PNP = "PNP"
    NMOS = "NMOS"
    PMOS = "PMOS"
    NJF = "NJF"  # N-channel JFET
    PJF = "PJF"  # P-channel JFET
    RESISTOR = "R"
    CAPACITOR = "C"
    INDUCTOR = "L"


@dataclass
class ParameterSpec:
    """Specification for a model parameter."""

    name: str
    description: str
    default: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    unit: str = ""
    required: bool = False


@dataclass
class ValidationIssue:
    """A validation issue found in a model."""

    level: str  # "error", "warning", "info"
    message: str
    parameter: str | None = None
    suggestion: str | None = None


@dataclass
class ValidationResult:
    """Result of model validation."""

    valid: bool
    model_type: str | None = None
    model_name: str | None = None
    parameters: dict[str, float] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.level == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.level == "warning" for issue in self.issues)

    def __str__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        lines = [f"Model Validation: {status}"]
        if self.model_name:
            lines.append(f"  Model: {self.model_name} ({self.model_type})")
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues:
                prefix = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(
                    issue.level, "•"
                )
                lines.append(f"    {prefix} {issue.message}")
                if issue.suggestion:
                    lines.append(f"       → {issue.suggestion}")
        return "\n".join(lines)


# Parameter specifications for common model types
DIODE_PARAMS: dict[str, ParameterSpec] = {
    "Is": ParameterSpec("Is", "Saturation current", 1e-14, 1e-18, 1e-6, "A"),
    "Rs": ParameterSpec("Rs", "Ohmic resistance", 0, 0, 1000, "Ω"),
    "N": ParameterSpec("N", "Emission coefficient", 1.0, 0.5, 4.0),
    "Tt": ParameterSpec("Tt", "Transit time", 0, 0, 1e-6, "s"),
    "Cjo": ParameterSpec("Cjo", "Zero-bias junction capacitance", 0, 0, 1e-9, "F"),
    "Vj": ParameterSpec("Vj", "Junction potential", 1.0, 0.1, 2.0, "V"),
    "M": ParameterSpec("M", "Grading coefficient", 0.5, 0.1, 0.9),
    "Bv": ParameterSpec("Bv", "Reverse breakdown voltage", None, 1, 10000, "V"),
    "Ibv": ParameterSpec("Ibv", "Current at breakdown", 1e-10, 1e-15, 1e-3, "A"),
    "Eg": ParameterSpec("Eg", "Band gap energy", 1.11, 0.5, 4.0, "eV"),
}

BJT_PARAMS: dict[str, ParameterSpec] = {
    "Is": ParameterSpec("Is", "Transport saturation current", 1e-16, 1e-20, 1e-10, "A"),
    "Bf": ParameterSpec("Bf", "Ideal forward beta", 100, 10, 1000),
    "Br": ParameterSpec("Br", "Ideal reverse beta", 1, 0.1, 100),
    "Nf": ParameterSpec("Nf", "Forward emission coefficient", 1.0, 0.8, 2.0),
    "Nr": ParameterSpec("Nr", "Reverse emission coefficient", 1.0, 0.8, 2.0),
    "Vaf": ParameterSpec("Vaf", "Forward Early voltage", None, 10, 1000, "V"),
    "Var": ParameterSpec("Var", "Reverse Early voltage", None, 10, 1000, "V"),
    "Ikf": ParameterSpec("Ikf", "Forward knee current", None, 1e-6, 100, "A"),
    "Ikr": ParameterSpec("Ikr", "Reverse knee current", None, 1e-6, 100, "A"),
    "Rb": ParameterSpec("Rb", "Base resistance", 0, 0, 1000, "Ω"),
    "Rc": ParameterSpec("Rc", "Collector resistance", 0, 0, 100, "Ω"),
    "Re": ParameterSpec("Re", "Emitter resistance", 0, 0, 100, "Ω"),
    "Cje": ParameterSpec("Cje", "Base-emitter junction capacitance", 0, 0, 1e-9, "F"),
    "Cjc": ParameterSpec("Cjc", "Base-collector junction capacitance", 0, 0, 1e-9, "F"),
}

MOSFET_PARAMS: dict[str, ParameterSpec] = {
    "Vto": ParameterSpec("Vto", "Threshold voltage", 0, -10, 10, "V"),
    "Kp": ParameterSpec("Kp", "Transconductance parameter", 2e-5, 1e-7, 1, "A/V²"),
    "Lambda": ParameterSpec("Lambda", "Channel length modulation", 0, 0, 0.5, "1/V"),
    "Rd": ParameterSpec("Rd", "Drain ohmic resistance", 0, 0, 100, "Ω"),
    "Rs": ParameterSpec("Rs", "Source ohmic resistance", 0, 0, 100, "Ω"),
    "Cbd": ParameterSpec("Cbd", "Bulk-drain junction capacitance", 0, 0, 1e-9, "F"),
    "Cbs": ParameterSpec("Cbs", "Bulk-source junction capacitance", 0, 0, 1e-9, "F"),
    "Cgd": ParameterSpec("Cgd", "Gate-drain overlap capacitance", 0, 0, 1e-9, "F"),
    "Cgs": ParameterSpec("Cgs", "Gate-source overlap capacitance", 0, 0, 1e-9, "F"),
    "Pb": ParameterSpec("Pb", "Bulk junction potential", 0.8, 0.1, 2.0, "V"),
    "Is": ParameterSpec("Is", "Bulk junction saturation current", 1e-14, 1e-18, 1e-6, "A"),
    "Tox": ParameterSpec("Tox", "Oxide thickness", 1e-7, 1e-9, 1e-5, "m"),
}

MODEL_PARAM_SPECS: dict[str, dict[str, ParameterSpec]] = {
    "D": DIODE_PARAMS,
    "NPN": BJT_PARAMS,
    "PNP": BJT_PARAMS,
    "NMOS": MOSFET_PARAMS,
    "PMOS": MOSFET_PARAMS,
}


def parse_spice_value(value_str: str) -> float:
    """Parse a SPICE value string with suffix.

    Supports suffixes: T, G, MEG, K, M, U, N, P, F

    Args:
        value_str: SPICE value string (e.g., "10k", "100n", "1MEG")

    Returns:
        Float value

    Raises:
        ValueError: If value cannot be parsed
    """
    value_str = value_str.strip().upper()

    suffixes = {
        "T": 1e12,
        "G": 1e9,
        "MEG": 1e6,
        "K": 1e3,
        "M": 1e-3,
        "U": 1e-6,
        "N": 1e-9,
        "P": 1e-12,
        "F": 1e-15,
    }

    for suffix, multiplier in sorted(suffixes.items(), key=lambda x: -len(x[0])):
        if value_str.endswith(suffix):
            numeric = value_str[: -len(suffix)]
            return float(numeric) * multiplier

    # Handle scientific notation
    return float(value_str)


def parse_model_card(model_card: str) -> tuple[str | None, str | None, dict[str, Any]]:
    """Parse a SPICE model card.

    Args:
        model_card: SPICE model card string (e.g., ".model D1N4148 D(Is=2.52e-9 N=1.906)")

    Returns:
        Tuple of (model_name, model_type, parameters dict)
    """
    # Remove comments
    model_card = re.sub(r"\*.*$", "", model_card, flags=re.MULTILINE)
    model_card = model_card.strip()

    if not model_card.lower().startswith(".model"):
        return None, None, {}

    # Extract model name and type
    pattern = r"\.model\s+(\S+)\s+(\w+)\s*(?:\((.*)\))?"
    match = re.match(pattern, model_card, re.IGNORECASE)

    if not match:
        return None, None, {}

    model_name = match.group(1)
    model_type = match.group(2).upper()
    params_str = match.group(3) or ""

    # Parse parameters
    params: dict[str, Any] = {}
    param_pattern = r"(\w+)\s*=\s*([^\s,)]+)"

    for param_match in re.finditer(param_pattern, params_str, re.IGNORECASE):
        param_name = param_match.group(1)
        param_value_str = param_match.group(2)

        try:
            params[param_name] = parse_spice_value(param_value_str)
        except ValueError:
            params[param_name] = param_value_str

    return model_name, model_type, params


def validate_model_card(model_card: str) -> ValidationResult:
    """Validate a SPICE model card.

    Checks:
    - Syntax correctness
    - Parameter names are valid for model type
    - Parameter values are within reasonable ranges
    - Required parameters are present

    Args:
        model_card: SPICE model card string

    Returns:
        ValidationResult with details
    """
    issues: list[ValidationIssue] = []

    model_name, model_type, params = parse_model_card(model_card)

    if model_name is None:
        issues.append(
            ValidationIssue(
                level="error",
                message="Invalid model card syntax",
                suggestion="Model card should start with '.model NAME TYPE(params)'",
            )
        )
        return ValidationResult(valid=False, issues=issues)

    if model_type not in MODEL_PARAM_SPECS:
        issues.append(
            ValidationIssue(
                level="warning",
                message=f"Unknown model type: {model_type}",
                suggestion=f"Known types: {', '.join(MODEL_PARAM_SPECS.keys())}",
            )
        )
        return ValidationResult(
            valid=True,
            model_type=model_type,
            model_name=model_name,
            parameters=params,
            issues=issues,
        )

    param_specs = MODEL_PARAM_SPECS[model_type]

    # Check for unknown parameters
    for param_name in params:
        if param_name not in param_specs:
            # Could be valid but not in our specs
            issues.append(
                ValidationIssue(
                    level="info",
                    message=f"Unknown parameter: {param_name}",
                    parameter=param_name,
                )
            )

    # Validate known parameters
    for param_name, value in params.items():
        if param_name not in param_specs:
            continue

        spec = param_specs[param_name]

        if not isinstance(value, (int, float)):
            issues.append(
                ValidationIssue(
                    level="error",
                    message=f"Parameter {param_name} has non-numeric value: {value}",
                    parameter=param_name,
                )
            )
            continue

        # Check range
        if spec.min_value is not None and value < spec.min_value:
            issues.append(
                ValidationIssue(
                    level="warning",
                    message=f"{param_name}={value} is below typical minimum ({spec.min_value})",
                    parameter=param_name,
                    suggestion=f"{spec.description}: typical range [{spec.min_value}, {spec.max_value}] {spec.unit}",
                )
            )

        if spec.max_value is not None and value > spec.max_value:
            issues.append(
                ValidationIssue(
                    level="warning",
                    message=f"{param_name}={value} is above typical maximum ({spec.max_value})",
                    parameter=param_name,
                    suggestion=f"{spec.description}: typical range [{spec.min_value}, {spec.max_value}] {spec.unit}",
                )
            )

    # Check for required parameters
    for param_name, spec in param_specs.items():
        if spec.required and param_name not in params:
            issues.append(
                ValidationIssue(
                    level="error",
                    message=f"Required parameter missing: {param_name}",
                    parameter=param_name,
                    suggestion=f"{spec.description} (default: {spec.default})",
                )
            )

    has_errors = any(issue.level == "error" for issue in issues)

    return ValidationResult(
        valid=not has_errors,
        model_type=model_type,
        model_name=model_name,
        parameters=params,
        issues=issues,
    )


def validate_component_params(
    model_type: str,
    params: dict[str, float],
) -> ValidationResult:
    """Validate component parameters without a full model card.

    Args:
        model_type: SPICE model type (D, NPN, PNP, NMOS, PMOS)
        params: Parameter dictionary

    Returns:
        ValidationResult
    """
    issues: list[ValidationIssue] = []

    if model_type not in MODEL_PARAM_SPECS:
        issues.append(
            ValidationIssue(
                level="warning",
                message=f"Unknown model type: {model_type}",
            )
        )
        return ValidationResult(valid=True, model_type=model_type, issues=issues)

    param_specs = MODEL_PARAM_SPECS[model_type]

    for param_name, value in params.items():
        if param_name not in param_specs:
            continue

        spec = param_specs[param_name]

        if spec.min_value is not None and value < spec.min_value:
            issues.append(
                ValidationIssue(
                    level="warning",
                    message=f"{param_name}={value} below minimum ({spec.min_value})",
                    parameter=param_name,
                )
            )

        if spec.max_value is not None and value > spec.max_value:
            issues.append(
                ValidationIssue(
                    level="warning",
                    message=f"{param_name}={value} above maximum ({spec.max_value})",
                    parameter=param_name,
                )
            )

    has_errors = any(issue.level == "error" for issue in issues)

    return ValidationResult(
        valid=not has_errors,
        model_type=model_type,
        parameters=params,
        issues=issues,
    )


def get_parameter_info(model_type: str, param_name: str) -> ParameterSpec | None:
    """Get information about a model parameter.

    Args:
        model_type: SPICE model type
        param_name: Parameter name

    Returns:
        ParameterSpec or None if not found
    """
    if model_type not in MODEL_PARAM_SPECS:
        return None

    return MODEL_PARAM_SPECS[model_type].get(param_name)


def list_parameters(model_type: str) -> list[ParameterSpec]:
    """List all known parameters for a model type.

    Args:
        model_type: SPICE model type

    Returns:
        List of ParameterSpec
    """
    if model_type not in MODEL_PARAM_SPECS:
        return []

    return list(MODEL_PARAM_SPECS[model_type].values())


__all__ = [
    "ModelType",
    "ParameterSpec",
    "ValidationIssue",
    "ValidationResult",
    "parse_spice_value",
    "parse_model_card",
    "validate_model_card",
    "validate_component_params",
    "get_parameter_info",
    "list_parameters",
    "DIODE_PARAMS",
    "BJT_PARAMS",
    "MOSFET_PARAMS",
]
