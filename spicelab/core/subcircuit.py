"""Hierarchical subcircuit support (M8).

This module provides:
- SubcircuitDefinition: Define reusable subcircuit blocks
- SubcircuitLibrary: Manage and cache subcircuit definitions
- Hierarchical circuit composition with parameter passing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .circuit import Circuit


@dataclass(frozen=True)
class SubcircuitPort:
    """Port definition for a subcircuit.

    Args:
        name: Port name (e.g., "in", "out", "vdd")
        description: Optional description for documentation
    """

    name: str
    description: str = ""


@dataclass(frozen=True)
class SubcircuitParameter:
    """Parameter definition for a subcircuit.

    Args:
        name: Parameter name (e.g., "R", "C", "GAIN")
        default: Default value (required)
        description: Optional description
        min_value: Optional minimum allowed value
        max_value: Optional maximum allowed value
    """

    name: str
    default: float | str
    description: str = ""
    min_value: float | None = None
    max_value: float | None = None

    def validate(self, value: float | str) -> bool:
        """Check if a value is within allowed range."""
        if isinstance(value, str):
            # Can't validate string expressions
            return True
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


@dataclass
class SubcircuitDefinition:
    """A reusable subcircuit definition.

    This represents a .SUBCKT block that can be instantiated multiple times
    with different parameter values.

    Args:
        name: Subcircuit name (e.g., "LM741", "RC_FILTER")
        ports: List of port definitions
        body: SPICE netlist body (lines between .SUBCKT and .ENDS)
        parameters: Optional list of parameter definitions
        description: Optional description for documentation
        category: Optional category for library organization

    Example:
        # Define a simple RC filter subcircuit
        rc_filter = SubcircuitDefinition(
            name="RC_LOWPASS",
            ports=[
                SubcircuitPort("in", "Input signal"),
                SubcircuitPort("out", "Filtered output"),
                SubcircuitPort("gnd", "Ground reference"),
            ],
            body=\"\"\"
R1 in out {R}
C1 out gnd {C}
\"\"\",
            parameters=[
                SubcircuitParameter("R", 1000, "Resistance in ohms", min_value=1),
                SubcircuitParameter("C", 1e-9, "Capacitance in farads", min_value=1e-15),
            ],
            description="First-order RC lowpass filter",
            category="filters",
        )
    """

    name: str
    ports: list[SubcircuitPort]
    body: str
    parameters: list[SubcircuitParameter] = field(default_factory=list)
    description: str = ""
    category: str = "general"
    # Optional source file path
    source_file: Path | None = None

    @property
    def port_names(self) -> list[str]:
        """Get list of port names in order."""
        return [p.name for p in self.ports]

    @property
    def parameter_names(self) -> list[str]:
        """Get list of parameter names."""
        return [p.name for p in self.parameters]

    @property
    def default_params(self) -> dict[str, float | str]:
        """Get default parameter values as a dict."""
        return {p.name: p.default for p in self.parameters}

    def to_spice(self, include_params: bool = True) -> str:
        """Generate SPICE .SUBCKT block.

        Args:
            include_params: If True, include parameter defaults in header

        Returns:
            Complete .SUBCKT ... .ENDS block as string
        """
        ports_str = " ".join(self.port_names)

        # Build parameter string with defaults
        params_str = ""
        if include_params and self.parameters:
            param_parts = [f"{p.name}={p.default}" for p in self.parameters]
            params_str = " params: " + " ".join(param_parts)

        header = f".SUBCKT {self.name} {ports_str}{params_str}"

        # Clean up body - ensure no leading/trailing whitespace issues
        body_lines = self.body.strip().split("\n")
        body = "\n".join(line for line in body_lines if line.strip())

        return f"{header}\n{body}\n.ENDS {self.name}"

    def validate_params(self, params: dict[str, float | str]) -> list[str]:
        """Validate parameter values against definitions.

        Args:
            params: Dict of parameter name -> value

        Returns:
            List of error messages (empty if all valid)
        """
        errors: list[str] = []

        # Check for unknown parameters
        known = set(self.parameter_names)
        for name in params:
            if name not in known:
                errors.append(f"Unknown parameter '{name}' for subcircuit {self.name}")

        # Validate each known parameter
        for pdef in self.parameters:
            if pdef.name in params:
                value = params[pdef.name]
                if not pdef.validate(value):
                    errors.append(
                        f"Parameter '{pdef.name}' value {value} out of range "
                        f"[{pdef.min_value}, {pdef.max_value}]"
                    )

        return errors

    def instantiate(
        self,
        ref: str,
        nodes: list[str],
        params: dict[str, float | str] | None = None,
    ) -> str:
        """Generate SPICE X-element instance.

        Args:
            ref: Instance reference (e.g., "1" for X1)
            nodes: List of node names to connect to ports
            params: Optional parameter overrides

        Returns:
            SPICE X-element card

        Raises:
            ValueError: If node count doesn't match port count
        """
        if len(nodes) != len(self.ports):
            raise ValueError(
                f"Subcircuit {self.name} has {len(self.ports)} ports, "
                f"but {len(nodes)} nodes provided"
            )

        nodes_str = " ".join(nodes)

        # Build parameter string
        params_str = ""
        if params:
            param_parts = [f"{k}={v}" for k, v in params.items()]
            params_str = " " + " ".join(param_parts)

        return f"X{ref} {nodes_str} {self.name}{params_str}"

    @classmethod
    def from_spice(cls, text: str, *, category: str = "imported") -> SubcircuitDefinition:
        """Parse a SubcircuitDefinition from SPICE .SUBCKT block.

        Args:
            text: SPICE text containing .SUBCKT ... .ENDS block
            category: Category to assign

        Returns:
            Parsed SubcircuitDefinition

        Raises:
            ValueError: If text doesn't contain valid .SUBCKT block
        """
        lines = text.strip().split("\n")

        # Find .SUBCKT line
        subckt_line = None
        subckt_idx = -1
        for i, line in enumerate(lines):
            if line.strip().upper().startswith(".SUBCKT"):
                subckt_line = line
                subckt_idx = i
                break

        if subckt_line is None:
            raise ValueError("No .SUBCKT directive found")

        # Find .ENDS line
        ends_idx = -1
        for i, line in enumerate(lines):
            if line.strip().upper().startswith(".ENDS"):
                ends_idx = i
                break

        if ends_idx == -1:
            raise ValueError("No .ENDS directive found")

        # Parse .SUBCKT header
        # Format: .SUBCKT name port1 port2 ... [params: p1=v1 p2=v2 ...]
        parts = subckt_line.split()
        name = parts[1]

        # Find where params start (if any)
        params_idx = None
        for i, part in enumerate(parts):
            if part.lower() == "params:":
                params_idx = i
                break

        # Extract ports
        if params_idx:
            port_names = parts[2:params_idx]
        else:
            port_names = parts[2:]

        ports = [SubcircuitPort(name=pn) for pn in port_names]

        # Extract parameters
        parameters: list[SubcircuitParameter] = []
        if params_idx:
            for part in parts[params_idx + 1 :]:
                if "=" in part:
                    pname, pval = part.split("=", 1)
                    # Try to parse as float
                    try:
                        default: float | str = float(pval)
                    except ValueError:
                        default = pval
                    parameters.append(SubcircuitParameter(name=pname, default=default))

        # Extract body
        body_lines = lines[subckt_idx + 1 : ends_idx]
        body = "\n".join(body_lines)

        return cls(
            name=name,
            ports=ports,
            body=body,
            parameters=parameters,
            category=category,
        )


class SubcircuitLibrary:
    """Registry for subcircuit definitions.

    Manages subcircuit definitions with:
    - Registration and lookup by name
    - Category-based organization
    - File loading and caching
    - Dependency tracking

    Example:
        lib = SubcircuitLibrary()

        # Register a definition
        lib.register(rc_filter_def)

        # Load from file
        lib.load_file(Path("opamps.sub"))

        # Get definition
        lm741 = lib.get("LM741")

        # List by category
        filters = lib.list_by_category("filters")
    """

    def __init__(self) -> None:
        self._definitions: dict[str, SubcircuitDefinition] = {}
        self._by_category: dict[str, list[str]] = {}
        self._loaded_files: set[Path] = set()

    def register(self, definition: SubcircuitDefinition) -> None:
        """Register a subcircuit definition.

        Args:
            definition: SubcircuitDefinition to register

        Raises:
            ValueError: If a definition with same name already exists
        """
        if definition.name in self._definitions:
            raise ValueError(f"Subcircuit '{definition.name}' already registered")

        self._definitions[definition.name] = definition

        # Update category index
        if definition.category not in self._by_category:
            self._by_category[definition.category] = []
        self._by_category[definition.category].append(definition.name)

    def register_or_replace(self, definition: SubcircuitDefinition) -> None:
        """Register a subcircuit, replacing any existing definition.

        Args:
            definition: SubcircuitDefinition to register
        """
        # Remove from old category if exists
        if definition.name in self._definitions:
            old_def = self._definitions[definition.name]
            if old_def.category in self._by_category:
                self._by_category[old_def.category] = [
                    n for n in self._by_category[old_def.category] if n != definition.name
                ]

        self._definitions[definition.name] = definition

        # Update category index
        if definition.category not in self._by_category:
            self._by_category[definition.category] = []
        if definition.name not in self._by_category[definition.category]:
            self._by_category[definition.category].append(definition.name)

    def get(self, name: str) -> SubcircuitDefinition | None:
        """Get a subcircuit definition by name.

        Args:
            name: Subcircuit name (case-sensitive)

        Returns:
            SubcircuitDefinition or None if not found
        """
        return self._definitions.get(name)

    def __getitem__(self, name: str) -> SubcircuitDefinition:
        """Get a subcircuit definition by name.

        Args:
            name: Subcircuit name

        Returns:
            SubcircuitDefinition

        Raises:
            KeyError: If not found
        """
        if name not in self._definitions:
            raise KeyError(f"Subcircuit '{name}' not found. Available: {list(self._definitions.keys())}")
        return self._definitions[name]

    def __contains__(self, name: str) -> bool:
        """Check if a subcircuit is registered."""
        return name in self._definitions

    @property
    def names(self) -> list[str]:
        """Get all registered subcircuit names."""
        return list(self._definitions.keys())

    @property
    def categories(self) -> list[str]:
        """Get all categories."""
        return list(self._by_category.keys())

    def list_by_category(self, category: str) -> list[str]:
        """Get subcircuit names in a category.

        Args:
            category: Category name

        Returns:
            List of subcircuit names in category
        """
        return self._by_category.get(category, [])

    def load_file(self, path: Path, *, category: str | None = None) -> list[str]:
        """Load subcircuit definitions from a SPICE file.

        Args:
            path: Path to .sub or .lib file
            category: Optional category override (defaults to filename)

        Returns:
            List of subcircuit names loaded

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path).resolve()

        if path in self._loaded_files:
            # Already loaded - return existing names from this file
            return [n for n, d in self._definitions.items() if d.source_file == path]

        if not path.exists():
            raise FileNotFoundError(f"Subcircuit file not found: {path}")

        text = path.read_text()

        # Use filename as default category
        if category is None:
            category = path.stem

        loaded: list[str] = []

        # Parse multiple .SUBCKT blocks
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.upper().startswith(".SUBCKT"):
                # Find matching .ENDS
                start_idx = i
                end_idx = i + 1
                while end_idx < len(lines):
                    if lines[end_idx].strip().upper().startswith(".ENDS"):
                        break
                    end_idx += 1

                # Extract block
                block = "\n".join(lines[start_idx : end_idx + 1])

                try:
                    definition = SubcircuitDefinition.from_spice(block, category=category)
                    definition = SubcircuitDefinition(
                        name=definition.name,
                        ports=definition.ports,
                        body=definition.body,
                        parameters=definition.parameters,
                        description=definition.description,
                        category=definition.category,
                        source_file=path,
                    )
                    self.register_or_replace(definition)
                    loaded.append(definition.name)
                except ValueError:
                    pass  # Skip invalid blocks

                i = end_idx + 1
            else:
                i += 1

        self._loaded_files.add(path)
        return loaded

    def load_directory(self, directory: Path, *, pattern: str = "*.sub") -> dict[str, list[str]]:
        """Load all subcircuit files from a directory.

        Args:
            directory: Directory to scan
            pattern: Glob pattern for files (default: "*.sub")

        Returns:
            Dict mapping filename to list of loaded subcircuit names
        """
        directory = Path(directory)
        results: dict[str, list[str]] = {}

        for path in directory.glob(pattern):
            loaded = self.load_file(path)
            results[path.name] = loaded

        return results

    def to_spice(self, names: list[str] | None = None) -> str:
        """Generate SPICE text for subcircuit definitions.

        Args:
            names: List of subcircuit names to include (None = all)

        Returns:
            Combined SPICE text with all subcircuit definitions
        """
        if names is None:
            names = self.names

        blocks: list[str] = []
        for name in names:
            if name in self._definitions:
                blocks.append(self._definitions[name].to_spice())

        return "\n\n".join(blocks)

    def get_dependencies(self, name: str) -> set[str]:
        """Find subcircuits that this subcircuit depends on.

        Scans the body for X-element instances referencing other subcircuits.

        Args:
            name: Subcircuit name to analyze

        Returns:
            Set of subcircuit names that this one depends on
        """
        if name not in self._definitions:
            return set()

        definition = self._definitions[name]
        deps: set[str] = set()

        for line in definition.body.split("\n"):
            line = line.strip()
            if line.upper().startswith("X"):
                # Parse X-element: X<ref> <nodes...> <subckt_name> [params...]
                parts = line.split()
                if len(parts) >= 3:
                    # Find subcircuit name (last non-parameter token)
                    for i in range(len(parts) - 1, 0, -1):
                        if "=" not in parts[i]:
                            subckt_name = parts[i]
                            if subckt_name in self._definitions and subckt_name != name:
                                deps.add(subckt_name)
                            break

        return deps

    def get_all_dependencies(self, name: str) -> set[str]:
        """Get transitive closure of dependencies.

        Args:
            name: Subcircuit name to analyze

        Returns:
            Set of all subcircuits that this one depends on (directly or indirectly)
        """
        all_deps: set[str] = set()
        to_process = [name]
        processed: set[str] = set()

        while to_process:
            current = to_process.pop()
            if current in processed:
                continue
            processed.add(current)

            deps = self.get_dependencies(current)
            all_deps.update(deps)
            to_process.extend(deps)

        return all_deps

    def clear(self) -> None:
        """Clear all registered definitions."""
        self._definitions.clear()
        self._by_category.clear()
        self._loaded_files.clear()


# Global default library instance
_default_library: SubcircuitLibrary | None = None


def get_subcircuit_library() -> SubcircuitLibrary:
    """Get the default global subcircuit library."""
    global _default_library
    if _default_library is None:
        _default_library = SubcircuitLibrary()
    return _default_library


def register_subcircuit(definition: SubcircuitDefinition) -> None:
    """Register a subcircuit in the default library."""
    get_subcircuit_library().register(definition)


def get_subcircuit(name: str) -> SubcircuitDefinition | None:
    """Get a subcircuit from the default library."""
    return get_subcircuit_library().get(name)
