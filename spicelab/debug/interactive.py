"""Interactive mode for handling ambiguous choices.

Provides prompts for user decisions when multiple valid options exist:
- Analysis type selection
- Parameter suggestions
- Component value validation
- Engine selection
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TextIO, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


class InteractiveMode(Enum):
    """Mode for handling interactive prompts."""

    NEVER = auto()  # Never prompt, use defaults
    ALWAYS = auto()  # Always prompt for choices
    AMBIGUOUS = auto()  # Only prompt when ambiguous (default)


@dataclass
class Choice:
    """A single choice option.

    Attributes:
        value: The actual value to use if selected
        label: Display label for the choice
        description: Optional description explaining the choice
        is_default: Whether this is the default choice
    """

    value: Any
    label: str
    description: str = ""
    is_default: bool = False

    def __str__(self) -> str:
        result = self.label
        if self.description:
            result += f" - {self.description}"
        if self.is_default:
            result += " (default)"
        return result


@dataclass
class InteractivePrompt:
    """A prompt for user input.

    Attributes:
        question: The question to ask
        choices: Available choices
        category: Category of the prompt (analysis, parameter, etc.)
        context: Additional context information
    """

    question: str
    choices: list[Choice]
    category: str = "general"
    context: dict[str, Any] = field(default_factory=dict)

    def get_default(self) -> Choice | None:
        """Get the default choice, if any."""
        for choice in self.choices:
            if choice.is_default:
                return choice
        return self.choices[0] if self.choices else None


@dataclass
class InteractiveConfig:
    """Configuration for interactive mode."""

    mode: InteractiveMode = InteractiveMode.AMBIGUOUS
    input_stream: TextIO = field(default_factory=lambda: sys.stdin)
    output_stream: TextIO = field(default_factory=lambda: sys.stdout)
    timeout: float | None = None  # Timeout in seconds (None = no timeout)
    callback: Callable[[InteractivePrompt], Any] | None = None


# Thread-local storage for interactive context
_interactive_context = threading.local()


def get_interactive_context() -> InteractiveConfig | None:
    """Get the current interactive configuration, if any."""
    return getattr(_interactive_context, "config", None)


def set_interactive_mode(mode: InteractiveMode | str = InteractiveMode.AMBIGUOUS) -> None:
    """Set the interactive mode globally.

    Args:
        mode: Interactive mode (NEVER, ALWAYS, AMBIGUOUS) or string

    Example:
        >>> from spicelab.debug import set_interactive_mode, InteractiveMode
        >>> set_interactive_mode(InteractiveMode.ALWAYS)  # Always prompt
        >>> set_interactive_mode("never")  # Never prompt, use defaults
    """
    if isinstance(mode, str):
        mode = InteractiveMode[mode.upper()]

    if mode == InteractiveMode.NEVER:
        _interactive_context.config = None
    else:
        _interactive_context.config = InteractiveConfig(mode=mode)


class InteractiveSession:
    """Context manager for interactive prompting.

    Allows prompting users for choices when ambiguous situations arise.

    Example:
        >>> from spicelab.debug import InteractiveSession
        >>> with InteractiveSession():
        ...     # Ambiguous choices will prompt for user input
        ...     result = quick_ac(circuit)  # May prompt for frequency range
        ...
        >>> # Non-interactive mode (use defaults)
        >>> with InteractiveSession(mode=InteractiveMode.NEVER):
        ...     result = quick_ac(circuit)  # Uses default frequency range

        >>> # Custom callback for programmatic handling
        >>> def my_handler(prompt):
        ...     return prompt.choices[0].value  # Always pick first option
        >>> with InteractiveSession(callback=my_handler):
        ...     result = quick_ac(circuit)
    """

    def __init__(
        self,
        *,
        mode: InteractiveMode = InteractiveMode.AMBIGUOUS,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
        timeout: float | None = None,
        callback: Callable[[InteractivePrompt], Any] | None = None,
    ) -> None:
        """Initialize interactive session.

        Args:
            mode: When to prompt (NEVER, ALWAYS, AMBIGUOUS)
            input_stream: Input stream for reading responses (default stdin)
            output_stream: Output stream for prompts (default stdout)
            timeout: Timeout for user input in seconds (None = no timeout)
            callback: Optional callback to handle prompts programmatically
        """
        self._config = InteractiveConfig(
            mode=mode,
            input_stream=input_stream or sys.stdin,
            output_stream=output_stream or sys.stdout,
            timeout=timeout,
            callback=callback,
        )
        self._previous_config: InteractiveConfig | None = None

    def __enter__(self) -> InteractiveSession:
        """Enter interactive context."""
        self._previous_config = get_interactive_context()
        _interactive_context.config = self._config
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit interactive context."""
        if self._previous_config is not None:
            _interactive_context.config = self._previous_config
        else:
            _interactive_context.config = None

    def prompt(self, prompt: InteractivePrompt) -> Any:
        """Prompt for user choice.

        Args:
            prompt: The prompt to display

        Returns:
            The selected value
        """
        return _prompt_user(prompt, self._config)


def _prompt_user(prompt: InteractivePrompt, config: InteractiveConfig) -> Any:
    """Internal function to prompt user for choice."""
    # If callback is provided, use it
    if config.callback:
        return config.callback(prompt)

    # Check mode
    if config.mode == InteractiveMode.NEVER:
        default = prompt.get_default()
        if default:
            return default.value
        raise ValueError(f"No default choice available for: {prompt.question}")

    # Display prompt
    out = config.output_stream
    out.write(f"\n{prompt.question}\n")

    if prompt.context:
        out.write("Context:\n")
        for key, value in prompt.context.items():
            out.write(f"  {key}: {value}\n")

    out.write("\nOptions:\n")
    for i, choice in enumerate(prompt.choices, 1):
        marker = "*" if choice.is_default else " "
        out.write(f"  {marker}[{i}] {choice}\n")

    default = prompt.get_default()
    default_num = None
    if default:
        default_num = prompt.choices.index(default) + 1
        out.write(f"\nPress Enter for default [{default_num}], or enter choice: ")
    else:
        out.write("\nEnter choice: ")
    out.flush()

    # Read input
    try:
        response = config.input_stream.readline().strip()
    except Exception:
        # Fallback to default on read error
        if default:
            return default.value
        raise

    # Handle empty input (use default)
    if not response:
        if default:
            return default.value
        raise ValueError("No input provided and no default available")

    # Parse response
    try:
        choice_num = int(response)
        if 1 <= choice_num <= len(prompt.choices):
            return prompt.choices[choice_num - 1].value
        out.write(f"Invalid choice: {choice_num}. Using default.\n")
        if default:
            return default.value
        return prompt.choices[0].value
    except ValueError:
        # Try matching by label
        response_lower = response.lower()
        for choice in prompt.choices:
            if choice.label.lower() == response_lower:
                return choice.value
        out.write(f"Unknown choice: {response}. Using default.\n")
        if default:
            return default.value
        return prompt.choices[0].value


def prompt_choice(
    question: str,
    choices: Sequence[tuple[Any, str] | tuple[Any, str, str]],
    *,
    default: Any | None = None,
    category: str = "general",
    context: dict[str, Any] | None = None,
) -> Any:
    """Prompt user to select from choices.

    Convenience function for creating and executing a prompt.

    Args:
        question: Question to ask
        choices: List of (value, label) or (value, label, description) tuples
        default: Default value (must match a choice value)
        category: Category for the prompt
        context: Additional context information

    Returns:
        Selected value

    Example:
        >>> from spicelab.debug import prompt_choice, InteractiveSession
        >>> with InteractiveSession():
        ...     engine = prompt_choice(
        ...         "Which simulation engine?",
        ...         [
        ...             ("ngspice", "ngspice", "Open-source SPICE"),
        ...             ("ltspice", "LTspice", "Linear Technology SPICE"),
        ...         ],
        ...         default="ngspice"
        ...     )
    """
    config = get_interactive_context()
    if not config:
        # Not in interactive mode, return default
        if default is not None:
            return default
        if choices:
            return choices[0][0]
        raise ValueError("No choices provided")

    choice_list: list[Choice] = []
    for item in choices:
        value = item[0]
        label = item[1]
        desc = item[2] if len(item) > 2 else ""
        is_default = value == default
        choice_list.append(
            Choice(value=value, label=label, description=desc, is_default=is_default)
        )

    prompt = InteractivePrompt(
        question=question,
        choices=choice_list,
        category=category,
        context=context or {},
    )

    return _prompt_user(prompt, config)


def prompt_confirm(
    question: str,
    *,
    default: bool = True,
    context: dict[str, Any] | None = None,
) -> bool:
    """Prompt for yes/no confirmation.

    Args:
        question: Question to ask
        default: Default value
        context: Additional context

    Returns:
        True for yes, False for no

    Example:
        >>> from spicelab.debug import prompt_confirm, InteractiveSession
        >>> with InteractiveSession():
        ...     if prompt_confirm("Run with default parameters?"):
        ...         result = run_simulation()
    """
    return prompt_choice(
        question,
        [
            (True, "Yes", "Proceed"),
            (False, "No", "Cancel"),
        ],
        default=default,
        category="confirmation",
        context=context,
    )


def prompt_value(
    question: str,
    *,
    default: Any | None = None,
    validator: Callable[[str], Any] | None = None,
    suggestions: Sequence[Any] | None = None,
    context: dict[str, Any] | None = None,
) -> Any:
    """Prompt for a value input with optional validation.

    Args:
        question: Question to ask
        default: Default value
        validator: Function to validate/convert input
        suggestions: Suggested values to show
        context: Additional context

    Returns:
        Validated input value

    Example:
        >>> from spicelab.debug import prompt_value, InteractiveSession
        >>> with InteractiveSession():
        ...     freq = prompt_value(
        ...         "Enter cutoff frequency:",
        ...         default=1000,
        ...         validator=float,
        ...         suggestions=[100, 1000, 10000]
        ...     )
    """
    config = get_interactive_context()
    if not config:
        # Not in interactive mode
        if default is not None:
            return default
        raise ValueError("No default value provided")

    if config.mode == InteractiveMode.NEVER:
        if default is not None:
            return default
        raise ValueError("No default value and interactive mode is NEVER")

    # If callback provided, create a prompt
    if config.callback:
        choices = []
        if default is not None:
            choices.append(Choice(value=default, label=str(default), is_default=True))
        if suggestions:
            for s in suggestions:
                if s != default:
                    choices.append(Choice(value=s, label=str(s)))
        prompt = InteractivePrompt(
            question=question,
            choices=choices,
            category="value_input",
            context=context or {},
        )
        return config.callback(prompt)

    out = config.output_stream
    out.write(f"\n{question}\n")

    if context:
        out.write("Context:\n")
        for key, value in context.items():
            out.write(f"  {key}: {value}\n")

    if suggestions:
        out.write(f"Suggestions: {', '.join(str(s) for s in suggestions)}\n")

    if default is not None:
        out.write(f"Default: {default}\n")
        out.write("Enter value (or press Enter for default): ")
    else:
        out.write("Enter value: ")
    out.flush()

    try:
        response = config.input_stream.readline().strip()
    except Exception:
        if default is not None:
            return default
        raise

    if not response:
        if default is not None:
            return default
        raise ValueError("No input provided")

    if validator:
        try:
            return validator(response)
        except Exception as e:
            out.write(f"Invalid input: {e}\n")
            if default is not None:
                out.write(f"Using default: {default}\n")
                return default
            raise
    return response


# Pre-built prompts for common scenarios


def prompt_analysis_type(circuit_info: dict[str, Any] | None = None) -> str:
    """Prompt for analysis type selection.

    Args:
        circuit_info: Optional circuit information for context

    Returns:
        Selected analysis type
    """
    return prompt_choice(
        "What type of analysis would you like to run?",
        [
            ("op", "Operating Point", "DC operating point analysis"),
            ("ac", "AC Analysis", "Frequency response (Bode plot)"),
            ("tran", "Transient", "Time-domain simulation"),
            ("dc", "DC Sweep", "Sweep a DC source"),
            ("noise", "Noise", "Noise analysis"),
        ],
        default="op",
        category="analysis",
        context=circuit_info or {},
    )


def prompt_frequency_range(*, suggested_fc: float | None = None) -> tuple[float, float]:
    """Prompt for frequency range selection.

    Args:
        suggested_fc: Suggested center/cutoff frequency

    Returns:
        Tuple of (fstart, fstop)
    """
    context = {}
    if suggested_fc:
        context["suggested_fc"] = f"{suggested_fc:.2e} Hz"

    # Suggest ranges based on fc
    if suggested_fc:
        decades_below = 2
        decades_above = 2
        fstart = suggested_fc / (10**decades_below)
        fstop = suggested_fc * (10**decades_above)
    else:
        fstart = 1.0
        fstop = 1e9

    result = prompt_choice(
        "Select frequency range for AC analysis:",
        [
            ((fstart, fstop), f"{fstart:.0e} - {fstop:.0e} Hz", "Recommended range"),
            ((1, 1e6), "1 Hz - 1 MHz", "Audio range"),
            ((1e3, 1e9), "1 kHz - 1 GHz", "RF range"),
            ((0.1, 1e3), "0.1 Hz - 1 kHz", "Low frequency"),
        ],
        default=(fstart, fstop),
        category="frequency",
        context=context,
    )
    return result


def prompt_simulation_engine() -> str:
    """Prompt for simulation engine selection.

    Returns:
        Selected engine name
    """
    return prompt_choice(
        "Which simulation engine would you like to use?",
        [
            ("ngspice", "ngspice", "Open-source SPICE (recommended)"),
            ("ltspice", "LTspice", "Linear Technology SPICE"),
            ("xyce", "Xyce", "Sandia National Labs SPICE"),
        ],
        default="ngspice",
        category="engine",
    )


__all__ = [
    "InteractiveMode",
    "InteractiveSession",
    "InteractivePrompt",
    "InteractiveConfig",
    "Choice",
    "set_interactive_mode",
    "get_interactive_context",
    "prompt_choice",
    "prompt_confirm",
    "prompt_value",
    "prompt_analysis_type",
    "prompt_frequency_range",
    "prompt_simulation_engine",
]
