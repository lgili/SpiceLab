"""Exception hierarchy for SpiceLab.

This module defines all custom exceptions used throughout SpiceLab,
organized hierarchically for easy catching and handling.

Exception Hierarchy:
    SpiceLabError (base)
    ├── CircuitError
    │   ├── FloatingNodeError
    │   ├── ShortCircuitError
    │   ├── InvalidConnectionError
    │   └── ComponentNotFoundError
    ├── SimulationError
    │   ├── EngineNotFoundError
    │   ├── ConvergenceError
    │   ├── SimulationFailedError
    │   └── AnalysisError
    ├── ParseError
    │   ├── NetlistParseError
    │   ├── ResultParseError
    │   └── ModelParseError
    ├── ValidationError
    │   ├── ComponentValidationError
    │   ├── ParameterValidationError
    │   └── CircuitValidationError
    └── ConfigurationError
        ├── EngineConfigurationError
        └── PathNotFoundError

Features:
    - Error codes for programmatic handling (e.g., E1001, E2001)
    - Context information (what was being attempted)
    - Recovery suggestions
    - "Did you mean?" suggestions for typos

Examples:
    >>> from spicelab.exceptions import CircuitError, FloatingNodeError
    >>> try:
    ...     # Some circuit operation
    ...     pass
    ... except FloatingNodeError as e:
    ...     print(f"Error {e.code}: {e}")
    ...     print(f"Suggestion: {e.suggestion}")
    ... except CircuitError as e:
    ...     print(f"General circuit error: {e}")
"""

from __future__ import annotations

from difflib import get_close_matches
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


__all__ = [
    # Error codes
    "ErrorCode",
    "ERROR_CATALOG",
    # Helpers
    "suggest_similar",
    "format_suggestions",
    # Base exception
    "SpiceLabError",
    # Circuit errors
    "CircuitError",
    "FloatingNodeError",
    "ShortCircuitError",
    "InvalidConnectionError",
    "ComponentNotFoundError",
    # Simulation errors
    "SimulationError",
    "EngineNotFoundError",
    "ConvergenceError",
    "SimulationFailedError",
    "AnalysisError",
    # Parse errors
    "ParseError",
    "NetlistParseError",
    "ResultParseError",
    "ModelParseError",
    # Validation errors
    "ValidationError",
    "ComponentValidationError",
    "ParameterValidationError",
    "CircuitValidationError",
    # Configuration errors
    "ConfigurationError",
    "EngineConfigurationError",
    "PathNotFoundError",
    # Attribute errors
    "AttributeNotFoundError",
]


# =============================================================================
# Error Codes
# =============================================================================


class ErrorCode(Enum):
    """Error codes for programmatic handling.

    Format: EXYYZ where:
    - X = category (1=Circuit, 2=Simulation, 3=Parse, 4=Validation, 5=Config)
    - YY = specific error within category
    - Z = variant (0=base)

    Example:
        >>> from spicelab.exceptions import ErrorCode
        >>> if error.code == ErrorCode.FLOATING_NODE:
        ...     handle_floating_node(error)
    """

    # Circuit errors (1xxx)
    CIRCUIT_ERROR = "E1000"
    FLOATING_NODE = "E1001"
    SHORT_CIRCUIT = "E1002"
    INVALID_CONNECTION = "E1003"
    COMPONENT_NOT_FOUND = "E1004"
    MISSING_GROUND = "E1005"

    # Simulation errors (2xxx)
    SIMULATION_ERROR = "E2000"
    ENGINE_NOT_FOUND = "E2001"
    CONVERGENCE_FAILURE = "E2002"
    SIMULATION_FAILED = "E2003"
    ANALYSIS_ERROR = "E2004"
    TIMEOUT = "E2005"

    # Parse errors (3xxx)
    PARSE_ERROR = "E3000"
    NETLIST_PARSE = "E3001"
    RESULT_PARSE = "E3002"
    MODEL_PARSE = "E3003"
    SYNTAX_ERROR = "E3004"

    # Validation errors (4xxx)
    VALIDATION_ERROR = "E4000"
    COMPONENT_VALIDATION = "E4001"
    PARAMETER_VALIDATION = "E4002"
    CIRCUIT_VALIDATION = "E4003"
    VALUE_OUT_OF_RANGE = "E4004"
    INVALID_UNIT = "E4005"

    # Configuration errors (5xxx)
    CONFIG_ERROR = "E5000"
    ENGINE_CONFIG = "E5001"
    PATH_NOT_FOUND = "E5002"
    MISSING_DEPENDENCY = "E5003"


# =============================================================================
# Error Catalog with Solutions
# =============================================================================


ERROR_CATALOG: dict[ErrorCode, dict[str, str]] = {
    ErrorCode.FLOATING_NODE: {
        "title": "Floating Node Detected",
        "description": "A node in the circuit is not properly connected.",
        "common_causes": (
            "- Node connected to only one component terminal\n"
            "- Missing connection to ground or power rail\n"
            "- Typo in net name causing unintended separate nets"
        ),
        "solutions": (
            "1. Connect the node to at least two component terminals\n"
            "2. Add a high-value resistor to ground if intentionally floating\n"
            "3. Check net names for typos"
        ),
    },
    ErrorCode.SHORT_CIRCUIT: {
        "title": "Short Circuit Detected",
        "description": "Voltage sources are connected in parallel or shorted.",
        "common_causes": (
            "- Two voltage sources with same nodes\n"
            "- Wire connecting both terminals of a voltage source\n"
            "- Missing series resistance"
        ),
        "solutions": (
            "1. Remove one of the parallel voltage sources\n"
            "2. Add a small series resistance (e.g., 1mΩ)\n"
            "3. Check for unintended wire connections"
        ),
    },
    ErrorCode.CONVERGENCE_FAILURE: {
        "title": "Simulation Convergence Failure",
        "description": "The simulator could not find a stable solution.",
        "common_causes": (
            "- Unrealistic component values (too large/small)\n"
            "- Missing DC path to ground\n"
            "- Positive feedback without limiting\n"
            "- Discontinuities in behavioral sources"
        ),
        "solutions": (
            "1. Add .options reltol=0.01 or similar to relax tolerance\n"
            "2. Ensure all nodes have a DC path to ground\n"
            "3. Add initial conditions with .ic\n"
            "4. Check for realistic component values\n"
            "5. Try .options method=gear for stiff circuits"
        ),
    },
    ErrorCode.ENGINE_NOT_FOUND: {
        "title": "Simulation Engine Not Found",
        "description": "The SPICE engine binary could not be located.",
        "common_causes": (
            "- Engine not installed\n"
            "- Engine not on system PATH\n"
            "- Wrong engine name specified"
        ),
        "solutions": (
            "1. Install the engine (e.g., 'brew install ngspice' on macOS)\n"
            "2. Add engine directory to PATH\n"
            "3. Set SPICELAB_NGSPICE environment variable to binary path\n"
            "4. Use 'spicelab doctor' to diagnose installation"
        ),
    },
    ErrorCode.COMPONENT_NOT_FOUND: {
        "title": "Component Not Found",
        "description": "The requested component reference does not exist.",
        "common_causes": (
            "- Typo in component reference\n"
            "- Component not added to circuit\n"
            "- Case sensitivity issue"
        ),
        "solutions": (
            "1. Check spelling of component reference\n"
            "2. Use circuit.components to list all components\n"
            "3. Ensure component was added with circuit.add()"
        ),
    },
    ErrorCode.VALUE_OUT_OF_RANGE: {
        "title": "Value Out of Valid Range",
        "description": "A parameter value is outside acceptable limits.",
        "common_causes": (
            "- Unit prefix error (e.g., 'u' vs 'µ' vs 'm')\n"
            "- Missing unit prefix\n"
            "- Negative value where positive required"
        ),
        "solutions": (
            "1. Check unit prefixes: p=1e-12, n=1e-9, u=1e-6, m=1e-3, k=1e3, M=1e6\n"
            "2. Use explicit scientific notation: 1e-9 instead of 1n\n"
            "3. Verify value is appropriate for component type"
        ),
    },
    ErrorCode.INVALID_UNIT: {
        "title": "Invalid Unit or Prefix",
        "description": "The unit or SI prefix could not be parsed.",
        "common_causes": (
            "- Misspelled unit suffix\n" "- Unknown SI prefix\n" "- Mixed case issues"
        ),
        "solutions": (
            "1. Valid prefixes: f, p, n, u/µ, m, k, M/Meg, G, T\n"
            "2. Valid units: Ohm, F, H, V, A, Hz, s\n"
            "3. Examples: '10k', '100nF', '1.5Meg', '22pF'"
        ),
    },
}


# =============================================================================
# Helper Functions
# =============================================================================


def suggest_similar(name: str, candidates: list[str], n: int = 3, cutoff: float = 0.6) -> list[str]:
    """Find similar strings for 'did you mean?' suggestions.

    Args:
        name: The misspelled/unknown name
        candidates: List of valid names to match against
        n: Maximum number of suggestions
        cutoff: Minimum similarity ratio (0-1)

    Returns:
        List of similar strings, ordered by similarity

    Example:
        >>> suggest_similar("resitor", ["resistor", "capacitor", "inductor"])
        ['resistor']
    """
    return get_close_matches(name, candidates, n=n, cutoff=cutoff)


def format_suggestions(suggestions: list[str]) -> str:
    """Format 'did you mean?' suggestions for display.

    Args:
        suggestions: List of suggested alternatives

    Returns:
        Formatted string with suggestions
    """
    if not suggestions:
        return ""
    if len(suggestions) == 1:
        return f"Did you mean '{suggestions[0]}'?"
    quoted = [f"'{s}'" for s in suggestions]
    return f"Did you mean one of: {', '.join(quoted)}?"


# =============================================================================
# Base Exception
# =============================================================================


class SpiceLabError(Exception):
    """Base exception for all SpiceLab errors.

    All exceptions raised by SpiceLab inherit from this class,
    making it easy to catch all library-specific errors.

    Attributes:
        message: Human-readable error message
        code: ErrorCode enum for programmatic handling
        context: What was being attempted when error occurred
        suggestion: Recovery suggestion
        details: Optional dictionary with additional error context

    Example:
        >>> try:
        ...     # some operation
        ...     pass
        ... except SpiceLabError as e:
        ...     print(f"[{e.code.value}] {e.message}")
        ...     if e.suggestion:
        ...         print(f"Try: {e.suggestion}")
    """

    # Default error code for base class
    _default_code: ErrorCode = ErrorCode.CIRCUIT_ERROR

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        *,
        code: ErrorCode | None = None,
        context: str | None = None,
        suggestion: str | None = None,
    ):
        """Initialize error with message and optional details.

        Args:
            message: Human-readable error description
            details: Optional dictionary with additional context
            code: Error code for programmatic handling
            context: What was being attempted
            suggestion: Recovery suggestion
        """
        self.message = message
        self.details = details or {}
        self.code = code or self._default_code
        self.context = context
        self.suggestion = suggestion
        super().__init__(message)

    def __str__(self) -> str:
        """Return formatted error message with context and suggestions."""
        parts = []

        # Error code prefix
        parts.append(f"[{self.code.value}] {self.message}")

        # Context (what was being attempted)
        if self.context:
            parts.append(f"\nContext: {self.context}")

        # Details
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"\nDetails: {details_str}")

        # Suggestion
        if self.suggestion:
            parts.append(f"\nSuggestion: {self.suggestion}")

        return "".join(parts)

    def with_context(self, context: str) -> SpiceLabError:
        """Return a copy of this error with added context.

        Args:
            context: Description of what was being attempted

        Returns:
            Self (for chaining)
        """
        self.context = context
        return self

    def with_suggestion(self, suggestion: str) -> SpiceLabError:
        """Return a copy of this error with added suggestion.

        Args:
            suggestion: Recovery suggestion

        Returns:
            Self (for chaining)
        """
        self.suggestion = suggestion
        return self

    def full_help(self) -> str:
        """Get detailed help from error catalog if available.

        Returns:
            Detailed help text with causes and solutions
        """
        if self.code not in ERROR_CATALOG:
            return str(self)

        info = ERROR_CATALOG[self.code]
        parts = [
            f"[{self.code.value}] {info['title']}",
            f"\n{info['description']}",
            f"\n\nCommon Causes:\n{info['common_causes']}",
            f"\n\nSolutions:\n{info['solutions']}",
        ]

        if self.context:
            parts.insert(1, f"\nContext: {self.context}")

        return "".join(parts)


# ============================================================================
# Circuit Errors
# ============================================================================


class CircuitError(SpiceLabError):
    """Errors related to circuit construction and topology.

    Raised when there are issues with circuit structure, connections,
    or component relationships.
    """

    _default_code = ErrorCode.CIRCUIT_ERROR


class FloatingNodeError(CircuitError):
    """Circuit contains disconnected or floating nodes.

    A floating node is a net that is not connected to ground or has
    insufficient connections to determine its voltage.

    Attributes:
        nodes: List of Net objects that are floating
        message: Error description with node names
    """

    _default_code = ErrorCode.FLOATING_NODE

    def __init__(self, nodes: list[Any], suggestion: str | None = None):
        """Initialize with list of floating nodes.

        Args:
            nodes: List of Net objects that are floating
            suggestion: Optional suggestion for fixing the issue
        """
        self.nodes = nodes
        node_names = [getattr(n, "name", str(n)) for n in nodes]

        message = f"Floating nodes detected: {', '.join(node_names)}"

        default_suggestion = (
            "Connect each node to at least two component terminals, "
            "or add a high-value resistor to ground"
        )

        super().__init__(
            message,
            {"node_count": len(nodes), "nodes": node_names},
            suggestion=suggestion or default_suggestion,
        )


class ShortCircuitError(CircuitError):
    """Circuit contains a short circuit (invalid topology).

    Raised when voltage sources are shorted or there are zero-resistance
    loops that would cause numerical issues.

    Attributes:
        components: Components involved in the short circuit
    """

    _default_code = ErrorCode.SHORT_CIRCUIT

    def __init__(self, components: list[Any], message: str | None = None):
        """Initialize with components involved in short.

        Args:
            components: List of components creating the short
            message: Optional custom message
        """
        self.components = components
        comp_refs = [getattr(c, "ref", str(c)) for c in components]

        if message is None:
            message = f"Short circuit detected involving: {', '.join(comp_refs)}"

        super().__init__(
            message,
            {"components": comp_refs},
            suggestion="Remove one voltage source or add a series resistance",
        )


class InvalidConnectionError(CircuitError):
    """Invalid connection between components or ports.

    Raised when attempting to connect incompatible ports or
    create invalid topologies.

    Attributes:
        port1: First port in invalid connection
        port2: Second port in invalid connection
    """

    _default_code = ErrorCode.INVALID_CONNECTION

    def __init__(self, port1: Any, port2: Any, reason: str | None = None):
        """Initialize with invalid port pair.

        Args:
            port1: First port
            port2: Second port
            reason: Explanation of why connection is invalid
        """
        self.port1 = port1
        self.port2 = port2

        message = f"Invalid connection between {port1} and {port2}"
        if reason:
            message += f": {reason}"

        super().__init__(
            message,
            suggestion="Check port compatibility and connection rules",
        )


class ComponentNotFoundError(CircuitError):
    """Requested component does not exist in circuit.

    Attributes:
        ref: Component reference that was not found
        circuit_name: Name of circuit that was searched
        similar: List of similar component names (for "did you mean?")
    """

    _default_code = ErrorCode.COMPONENT_NOT_FOUND

    def __init__(
        self,
        ref: str,
        circuit_name: str | None = None,
        available: list[str] | None = None,
    ):
        """Initialize with component reference.

        Args:
            ref: Component reference that was not found
            circuit_name: Optional circuit name for context
            available: List of available component refs for suggestions
        """
        self.ref = ref
        self.circuit_name = circuit_name

        message = f"Component '{ref}' not found"
        if circuit_name:
            message += f" in circuit '{circuit_name}'"

        # Generate "did you mean?" suggestions
        suggestion = None
        self.similar: list[str] = []
        if available:
            self.similar = suggest_similar(ref, available)
            if self.similar:
                suggestion = format_suggestions(self.similar)

        super().__init__(message, {"ref": ref}, suggestion=suggestion)


# ============================================================================
# Simulation Errors
# ============================================================================


class SimulationError(SpiceLabError):
    """Errors during simulation execution.

    Base class for all simulation-related errors including engine
    problems, convergence failures, and analysis issues.
    """

    _default_code = ErrorCode.SIMULATION_ERROR


class EngineNotFoundError(SimulationError):
    """SPICE engine binary not found or not executable.

    Raised when the requested simulation engine (ngspice, ltspice, xyce)
    cannot be located or executed.

    Attributes:
        engine: Name of engine that wasn't found
        path: Path that was searched (if specified)
    """

    _default_code = ErrorCode.ENGINE_NOT_FOUND

    def __init__(self, engine: str, path: str | None = None):
        """Initialize with engine name and optional path.

        Args:
            engine: Name of simulation engine (e.g., 'ngspice')
            path: Optional path that was searched
        """
        self.engine = engine
        self.path = path

        message = f"Engine '{engine}' not found"
        if path:
            message += f" at '{path}'"

        suggestion = (
            f"Install {engine} and ensure it's on PATH, "
            f"or set SPICELAB_{engine.upper()} environment variable. "
            "Run 'spicelab doctor' to diagnose."
        )

        super().__init__(message, {"engine": engine, "path": path}, suggestion=suggestion)


class ConvergenceError(SimulationError):
    """Simulation failed to converge.

    Raised when the SPICE simulator cannot find a solution due to
    convergence issues. Common causes include unrealistic component
    values, numerical instability, or missing initial conditions.

    Attributes:
        analysis: Type of analysis that failed
        iteration: Iteration count when convergence failed (if available)
    """

    _default_code = ErrorCode.CONVERGENCE_FAILURE

    def __init__(
        self, analysis: str | None = None, iteration: int | None = None, log: str | None = None
    ):
        """Initialize with convergence failure details.

        Args:
            analysis: Analysis type (e.g., 'dc', 'tran')
            iteration: Iteration count at failure
            log: Relevant portion of simulation log
        """
        self.analysis = analysis
        self.iteration = iteration
        self.log = log

        message = "Simulation failed to converge"
        if analysis:
            message += f" during {analysis} analysis"
        if iteration:
            message += f" at iteration {iteration}"

        details: dict[str, Any] = {}
        if analysis:
            details["analysis"] = analysis
        if iteration:
            details["iteration"] = iteration

        suggestion = (
            "Try: (1) Add .options reltol=0.01, "
            "(2) Check for missing DC path to ground, "
            "(3) Add initial conditions with .ic, "
            "(4) Verify realistic component values"
        )

        super().__init__(message, details, suggestion=suggestion)


class SimulationFailedError(SimulationError):
    """General simulation failure.

    Raised when simulation fails for reasons other than convergence,
    such as syntax errors in netlist or engine crashes.

    Attributes:
        stderr: Standard error output from engine
        returncode: Exit code from simulation process
    """

    _default_code = ErrorCode.SIMULATION_FAILED

    def __init__(self, stderr: str | None = None, returncode: int | None = None):
        """Initialize with simulation failure details.

        Args:
            stderr: Error output from simulation engine
            returncode: Exit code from process
        """
        self.stderr = stderr
        self.returncode = returncode

        message = "Simulation failed"
        if returncode:
            message += f" with exit code {returncode}"
        if stderr:
            # Truncate long stderr
            max_len = 500
            truncated = stderr[:max_len] + "..." if len(stderr) > max_len else stderr
            message += f"\n\nEngine output:\n{truncated}"

        suggestion = (
            "Check netlist syntax and component models. " "Run 'spicelab doctor' to verify setup."
        )

        super().__init__(message, {"returncode": returncode}, suggestion=suggestion)


class AnalysisError(SimulationError):
    """Invalid or unsupported analysis specification.

    Raised when analysis parameters are invalid or the requested
    analysis type is not supported by the engine.

    Attributes:
        analysis_type: Type of analysis that failed
        params: Analysis parameters
    """

    _default_code = ErrorCode.ANALYSIS_ERROR

    def __init__(
        self, analysis_type: str, params: dict[str, Any] | None = None, reason: str | None = None
    ):
        """Initialize with analysis details.

        Args:
            analysis_type: Analysis type (e.g., 'dc', 'ac', 'tran')
            params: Analysis parameters
            reason: Explanation of why analysis is invalid
        """
        self.analysis_type = analysis_type
        self.params = params or {}

        message = f"Invalid {analysis_type} analysis"
        if reason:
            message += f": {reason}"
        if params:
            message += f"\nParameters: {params}"

        valid_types = ["op", "dc", "ac", "tran", "noise"]
        suggestion = f"Valid analysis types: {', '.join(valid_types)}"

        super().__init__(
            message, {"analysis_type": analysis_type, "params": params}, suggestion=suggestion
        )


# ============================================================================
# Parse Errors
# ============================================================================


class ParseError(SpiceLabError):
    """Errors parsing netlists, results, or models.

    Base class for all parsing-related errors.
    """

    _default_code = ErrorCode.PARSE_ERROR


class NetlistParseError(ParseError):
    """Failed to parse SPICE netlist.

    Raised when netlist syntax is invalid or unsupported.

    Attributes:
        line_number: Line number where parsing failed
        line_content: Content of the problematic line
    """

    _default_code = ErrorCode.NETLIST_PARSE

    def __init__(
        self, message: str, line_number: int | None = None, line_content: str | None = None
    ):
        """Initialize with parse error details.

        Args:
            message: Error description
            line_number: Line number in netlist
            line_content: Content of problematic line
        """
        self.line_number = line_number
        self.line_content = line_content

        full_message = f"Netlist parse error: {message}"
        if line_number:
            full_message += f" at line {line_number}"
        if line_content:
            full_message += f"\n  {line_content}"

        super().__init__(
            full_message,
            {"line_number": line_number},
            suggestion="Check SPICE netlist syntax. Common issues: missing node names, invalid values",  # noqa: E501
        )


class ResultParseError(ParseError):
    """Failed to parse simulation results.

    Raised when result file (e.g., .raw file) is corrupted or in
    an unexpected format.

    Attributes:
        file_path: Path to result file
        format: Expected format
    """

    _default_code = ErrorCode.RESULT_PARSE

    def __init__(self, message: str, file_path: str | None = None, format: str | None = None):
        """Initialize with result parse error.

        Args:
            message: Error description
            file_path: Path to result file
            format: Expected file format
        """
        self.file_path = file_path
        self.format = format

        full_message = f"Result parse error: {message}"
        if file_path:
            full_message += f"\nFile: {file_path}"
        if format:
            full_message += f"\nExpected format: {format}"

        super().__init__(
            full_message,
            {"file_path": file_path, "format": format},
            suggestion="Verify simulation completed successfully and output file is not corrupted",
        )


class ModelParseError(ParseError):
    """Failed to parse device model.

    Raised when .model or .subckt definition is invalid or
    model file cannot be loaded.

    Attributes:
        model_name: Name of model that failed to parse
        file_path: Path to model file (if applicable)
    """

    _default_code = ErrorCode.MODEL_PARSE

    def __init__(self, message: str, model_name: str | None = None, file_path: str | None = None):
        """Initialize with model parse error.

        Args:
            message: Error description
            model_name: Model name
            file_path: Model file path
        """
        self.model_name = model_name
        self.file_path = file_path

        full_message = f"Model parse error: {message}"
        if model_name:
            full_message += f"\nModel: {model_name}"
        if file_path:
            full_message += f"\nFile: {file_path}"

        super().__init__(
            full_message,
            {"model_name": model_name},
            suggestion="Check .model syntax and ensure all required parameters are present",
        )


# ============================================================================
# Validation Errors
# ============================================================================


class ValidationError(SpiceLabError):
    """Input validation errors.

    Raised when user-provided data fails validation checks.
    """

    _default_code = ErrorCode.VALIDATION_ERROR


class ComponentValidationError(ValidationError):
    """Component parameters are invalid.

    Raised when component values are outside valid ranges or
    incompatible with component type.

    Attributes:
        component_ref: Reference of invalid component
        parameter: Name of invalid parameter
        value: Invalid value
    """

    _default_code = ErrorCode.COMPONENT_VALIDATION

    def __init__(self, component_ref: str, parameter: str, value: Any, reason: str):
        """Initialize with validation failure details.

        Args:
            component_ref: Component reference
            parameter: Parameter name
            value: Invalid value
            reason: Why validation failed
        """
        self.component_ref = component_ref
        self.parameter = parameter
        self.value = value

        message = f"Invalid {parameter} for {component_ref}: {value}\n{reason}"

        super().__init__(
            message,
            {"component_ref": component_ref, "parameter": parameter, "value": value},
            suggestion="Check value and units. Use SI prefixes: p, n, u, m, k, M, G",
        )


class ParameterValidationError(ValidationError):
    """Analysis parameter validation failed.

    Raised when analysis parameters are outside valid ranges or
    mutually incompatible.

    Attributes:
        parameter: Name of invalid parameter
        value: Invalid value
    """

    _default_code = ErrorCode.PARAMETER_VALIDATION

    def __init__(self, parameter: str, value: Any, reason: str):
        """Initialize with parameter validation error.

        Args:
            parameter: Parameter name
            value: Invalid value
            reason: Why validation failed
        """
        self.parameter = parameter
        self.value = value

        message = f"Invalid parameter '{parameter}': {value}\n{reason}"

        super().__init__(
            message,
            {"parameter": parameter, "value": value},
            suggestion="Check parameter name and value type",
        )


class CircuitValidationError(ValidationError):
    """Circuit validation failed.

    Raised when circuit as a whole fails validation checks
    (e.g., missing ground, disconnected components).

    Attributes:
        circuit_name: Name of circuit
        failures: List of validation failures
    """

    _default_code = ErrorCode.CIRCUIT_VALIDATION

    def __init__(self, circuit_name: str, failures: list[str]):
        """Initialize with validation failures.

        Args:
            circuit_name: Circuit name
            failures: List of failure descriptions
        """
        self.circuit_name = circuit_name
        self.failures = failures

        message = f"Circuit '{circuit_name}' validation failed:\n"
        message += "\n".join(f"  - {f}" for f in failures)

        super().__init__(
            message,
            {"circuit_name": circuit_name, "failure_count": len(failures)},
            suggestion="Use validate_circuit(circuit) to get detailed diagnostics",
        )


# ============================================================================
# Configuration Errors
# ============================================================================


class ConfigurationError(SpiceLabError):
    """Configuration or environment errors.

    Raised when SpiceLab configuration is invalid or required
    resources cannot be found.
    """

    _default_code = ErrorCode.CONFIG_ERROR


class EngineConfigurationError(ConfigurationError):
    """Engine configuration is invalid.

    Raised when engine-specific configuration (paths, options) is
    incorrect or incompatible.

    Attributes:
        engine: Engine name
        setting: Configuration setting that is invalid
    """

    _default_code = ErrorCode.ENGINE_CONFIG

    def __init__(self, engine: str, setting: str, reason: str):
        """Initialize with configuration error.

        Args:
            engine: Engine name
            setting: Configuration setting
            reason: Why configuration is invalid
        """
        self.engine = engine
        self.setting = setting

        message = f"Invalid configuration for {engine}: {setting}\n{reason}"

        super().__init__(
            message,
            {"engine": engine, "setting": setting},
            suggestion="Check engine documentation for valid configuration options",
        )


class PathNotFoundError(ConfigurationError):
    """Required file or directory not found.

    Raised when a required resource (model file, library, etc.)
    cannot be located.

    Attributes:
        path: Path that was not found
        resource_type: Type of resource (e.g., 'model', 'library')
    """

    _default_code = ErrorCode.PATH_NOT_FOUND

    def __init__(self, path: str, resource_type: str | None = None):
        """Initialize with path error.

        Args:
            path: Path that was not found
            resource_type: Type of resource
        """
        self.path = path
        self.resource_type = resource_type

        message = f"Path not found: {path}"
        if resource_type:
            message += f" ({resource_type})"

        super().__init__(
            message,
            {"path": path, "resource_type": resource_type},
            suggestion="Check file path and ensure the file exists",
        )


# ============================================================================
# Attribute Error with "Did You Mean?"
# ============================================================================


class AttributeNotFoundError(SpiceLabError):
    """Attribute or method not found with 'did you mean?' suggestions.

    Use this to provide helpful suggestions when users mistype
    attribute or method names.

    Attributes:
        obj_type: Type of object
        attr_name: Attribute that was not found
        similar: List of similar attribute names
    """

    _default_code = ErrorCode.COMPONENT_NOT_FOUND

    def __init__(self, obj_type: str, attr_name: str, available: list[str]):
        """Initialize with attribute error details.

        Args:
            obj_type: Type of object (e.g., "Circuit", "Resistor")
            attr_name: Attribute name that was not found
            available: List of available attributes for suggestions
        """
        self.obj_type = obj_type
        self.attr_name = attr_name
        self.similar = suggest_similar(attr_name, available)

        message = f"'{obj_type}' has no attribute '{attr_name}'"

        suggestion = None
        if self.similar:
            suggestion = format_suggestions(self.similar)

        super().__init__(message, suggestion=suggestion)
