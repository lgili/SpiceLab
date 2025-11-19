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

Examples:
    >>> from spicelab.exceptions import CircuitError, FloatingNodeError
    >>> try:
    ...     # Some circuit operation
    ...     pass
    ... except FloatingNodeError as e:
    ...     print(f"Floating node detected: {e}")
    ... except CircuitError as e:
    ...     print(f"General circuit error: {e}")
"""

from __future__ import annotations

from typing import Any


class SpiceLabError(Exception):
    """Base exception for all SpiceLab errors.

    All exceptions raised by SpiceLab inherit from this class,
    making it easy to catch all library-specific errors.

    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional error context
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Initialize error with message and optional details.

        Args:
            message: Human-readable error description
            details: Optional dictionary with additional context
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# ============================================================================
# Circuit Errors
# ============================================================================


class CircuitError(SpiceLabError):
    """Errors related to circuit construction and topology.

    Raised when there are issues with circuit structure, connections,
    or component relationships.
    """


class FloatingNodeError(CircuitError):
    """Circuit contains disconnected or floating nodes.

    A floating node is a net that is not connected to ground or has
    insufficient connections to determine its voltage.

    Attributes:
        nodes: List of Net objects that are floating
        message: Error description with node names
    """

    def __init__(self, nodes: list[Any], suggestion: str | None = None):
        """Initialize with list of floating nodes.

        Args:
            nodes: List of Net objects that are floating
            suggestion: Optional suggestion for fixing the issue
        """
        self.nodes = nodes
        node_names = [getattr(n, "name", str(n)) for n in nodes]

        message = f"Floating nodes detected: {', '.join(node_names)}"
        if suggestion:
            message += f"\nSuggestion: {suggestion}"

        super().__init__(message, {"node_count": len(nodes), "nodes": node_names})


class ShortCircuitError(CircuitError):
    """Circuit contains a short circuit (invalid topology).

    Raised when voltage sources are shorted or there are zero-resistance
    loops that would cause numerical issues.

    Attributes:
        components: Components involved in the short circuit
    """

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

        super().__init__(message, {"components": comp_refs})


class InvalidConnectionError(CircuitError):
    """Invalid connection between components or ports.

    Raised when attempting to connect incompatible ports or
    create invalid topologies.

    Attributes:
        port1: First port in invalid connection
        port2: Second port in invalid connection
    """

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

        super().__init__(message)


class ComponentNotFoundError(CircuitError):
    """Requested component does not exist in circuit.

    Attributes:
        ref: Component reference that was not found
        circuit_name: Name of circuit that was searched
    """

    def __init__(self, ref: str, circuit_name: str | None = None):
        """Initialize with component reference.

        Args:
            ref: Component reference that was not found
            circuit_name: Optional circuit name for context
        """
        self.ref = ref
        self.circuit_name = circuit_name

        message = f"Component '{ref}' not found"
        if circuit_name:
            message += f" in circuit '{circuit_name}'"

        super().__init__(message, {"ref": ref})


# ============================================================================
# Simulation Errors
# ============================================================================


class SimulationError(SpiceLabError):
    """Errors during simulation execution.

    Base class for all simulation-related errors including engine
    problems, convergence failures, and analysis issues.
    """


class EngineNotFoundError(SimulationError):
    """SPICE engine binary not found or not executable.

    Raised when the requested simulation engine (ngspice, ltspice, xyce)
    cannot be located or executed.

    Attributes:
        engine: Name of engine that wasn't found
        path: Path that was searched (if specified)
    """

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
        message += (
            "\n\nInstall the engine and ensure it's on PATH, "
            f"or set SPICELAB_{engine.upper()} environment variable."
        )

        super().__init__(message, {"engine": engine, "path": path})


class ConvergenceError(SimulationError):
    """Simulation failed to converge.

    Raised when the SPICE simulator cannot find a solution due to
    convergence issues. Common causes include unrealistic component
    values, numerical instability, or missing initial conditions.

    Attributes:
        analysis: Type of analysis that failed
        iteration: Iteration count when convergence failed (if available)
    """

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

        message += "\n\nTry:\n"
        message += "- Adjusting component values\n"
        message += "- Adding initial conditions (.ic)\n"
        message += "- Increasing simulation tolerance\n"
        message += "- Simplifying the circuit"

        details: dict[str, Any] = {}
        if analysis:
            details["analysis"] = analysis
        if iteration:
            details["iteration"] = iteration

        super().__init__(message, details)


class SimulationFailedError(SimulationError):
    """General simulation failure.

    Raised when simulation fails for reasons other than convergence,
    such as syntax errors in netlist or engine crashes.

    Attributes:
        stderr: Standard error output from engine
        returncode: Exit code from simulation process
    """

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
            message += f"\n\nEngine output:\n{stderr}"

        super().__init__(message, {"returncode": returncode})


class AnalysisError(SimulationError):
    """Invalid or unsupported analysis specification.

    Raised when analysis parameters are invalid or the requested
    analysis type is not supported by the engine.

    Attributes:
        analysis_type: Type of analysis that failed
        params: Analysis parameters
    """

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

        super().__init__(message, {"analysis_type": analysis_type, "params": params})


# ============================================================================
# Parse Errors
# ============================================================================


class ParseError(SpiceLabError):
    """Errors parsing netlists, results, or models.

    Base class for all parsing-related errors.
    """


class NetlistParseError(ParseError):
    """Failed to parse SPICE netlist.

    Raised when netlist syntax is invalid or unsupported.

    Attributes:
        line_number: Line number where parsing failed
        line_content: Content of the problematic line
    """

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

        super().__init__(full_message, {"line_number": line_number})


class ResultParseError(ParseError):
    """Failed to parse simulation results.

    Raised when result file (e.g., .raw file) is corrupted or in
    an unexpected format.

    Attributes:
        file_path: Path to result file
        format: Expected format
    """

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

        super().__init__(full_message, {"file_path": file_path, "format": format})


class ModelParseError(ParseError):
    """Failed to parse device model.

    Raised when .model or .subckt definition is invalid or
    model file cannot be loaded.

    Attributes:
        model_name: Name of model that failed to parse
        file_path: Path to model file (if applicable)
    """

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

        super().__init__(full_message, {"model_name": model_name})


# ============================================================================
# Validation Errors
# ============================================================================


class ValidationError(SpiceLabError):
    """Input validation errors.

    Raised when user-provided data fails validation checks.
    """


class ComponentValidationError(ValidationError):
    """Component parameters are invalid.

    Raised when component values are outside valid ranges or
    incompatible with component type.

    Attributes:
        component_ref: Reference of invalid component
        parameter: Name of invalid parameter
        value: Invalid value
    """

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
            message, {"component_ref": component_ref, "parameter": parameter, "value": value}
        )


class ParameterValidationError(ValidationError):
    """Analysis parameter validation failed.

    Raised when analysis parameters are outside valid ranges or
    mutually incompatible.

    Attributes:
        parameter: Name of invalid parameter
        value: Invalid value
    """

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

        super().__init__(message, {"parameter": parameter, "value": value})


class CircuitValidationError(ValidationError):
    """Circuit validation failed.

    Raised when circuit as a whole fails validation checks
    (e.g., missing ground, disconnected components).

    Attributes:
        circuit_name: Name of circuit
        failures: List of validation failures
    """

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

        super().__init__(message, {"circuit_name": circuit_name, "failure_count": len(failures)})


# ============================================================================
# Configuration Errors
# ============================================================================


class ConfigurationError(SpiceLabError):
    """Configuration or environment errors.

    Raised when SpiceLab configuration is invalid or required
    resources cannot be found.
    """


class EngineConfigurationError(ConfigurationError):
    """Engine configuration is invalid.

    Raised when engine-specific configuration (paths, options) is
    incorrect or incompatible.

    Attributes:
        engine: Engine name
        setting: Configuration setting that is invalid
    """

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

        super().__init__(message, {"engine": engine, "setting": setting})


class PathNotFoundError(ConfigurationError):
    """Required file or directory not found.

    Raised when a required resource (model file, library, etc.)
    cannot be located.

    Attributes:
        path: Path that was not found
        resource_type: Type of resource (e.g., 'model', 'library')
    """

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

        super().__init__(message, {"path": path, "resource_type": resource_type})
