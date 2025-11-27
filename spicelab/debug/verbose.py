"""Verbose mode for simulation progress logging.

Provides detailed logging during simulation execution including:
- Circuit validation steps
- Netlist generation
- Analysis configuration
- Simulation progress
- Result processing
"""

from __future__ import annotations

import contextlib
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from collections.abc import Generator

# Thread-local storage for verbose context
import threading

_verbose_context = threading.local()


@dataclass
class VerboseConfig:
    """Configuration for verbose output."""

    enabled: bool = False
    show_netlist: bool = True
    show_analysis: bool = True
    show_timing: bool = True
    show_validation: bool = True
    indent: int = 2
    output: TextIO = field(default_factory=lambda: sys.stderr)


def get_verbose_context() -> VerboseConfig | None:
    """Get the current verbose configuration, if any."""
    return getattr(_verbose_context, "config", None)


def set_verbose(enabled: bool = True) -> None:
    """Enable or disable verbose mode globally.

    Args:
        enabled: Whether to enable verbose mode

    Example:
        >>> from spicelab.debug import set_verbose
        >>> set_verbose(True)  # Enable verbose mode globally
        >>> result = quick_ac(circuit, start=1, stop=1e6)  # Will print progress
        >>> set_verbose(False)  # Disable verbose mode
    """
    if enabled:
        _verbose_context.config = VerboseConfig(enabled=True)
    else:
        _verbose_context.config = None


class VerboseSimulation:
    """Context manager for verbose simulation output.

    Provides detailed progress logging during simulation execution.

    Example:
        >>> from spicelab.debug import VerboseSimulation
        >>> with VerboseSimulation():
        ...     result = quick_ac(circuit, start=1, stop=1e6)
        [verbose] Circuit: RC_Filter (2 components)
        [verbose] Generating netlist for ngspice...
        [verbose] Analysis: AC sweep dec 20 pts, 1.0Hz - 1000000000.0Hz
        [verbose] Running simulation...
        [verbose] Simulation completed in 0.23s
        [verbose] Processing results: 201 data points

        >>> # With custom options
        >>> with VerboseSimulation(show_netlist=True, show_timing=True):
        ...     result = quick_tran(circuit, duration="1ms")
    """

    def __init__(
        self,
        *,
        show_netlist: bool = False,
        show_analysis: bool = True,
        show_timing: bool = True,
        show_validation: bool = True,
        output: TextIO | None = None,
    ) -> None:
        """Initialize verbose context.

        Args:
            show_netlist: Print the full SPICE netlist (default False)
            show_analysis: Print analysis configuration (default True)
            show_timing: Print timing information (default True)
            show_validation: Print validation steps (default True)
            output: Output stream (default stderr)
        """
        self._config = VerboseConfig(
            enabled=True,
            show_netlist=show_netlist,
            show_analysis=show_analysis,
            show_timing=show_timing,
            show_validation=show_validation,
            output=output or sys.stderr,
        )
        self._previous_config: VerboseConfig | None = None
        self._start_time: float | None = None

    def __enter__(self) -> VerboseSimulation:
        """Enter verbose context."""
        self._previous_config = get_verbose_context()
        _verbose_context.config = self._config
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit verbose context."""
        if self._previous_config is not None:
            _verbose_context.config = self._previous_config
        else:
            _verbose_context.config = None

    def log(self, message: str, level: str = "info") -> None:
        """Log a message if verbose mode is enabled.

        Args:
            message: Message to log
            level: Log level (info, step, detail, timing)
        """
        if not self._config.enabled:
            return

        prefix = "[verbose]"
        if level == "step":
            prefix = "[verbose] →"
        elif level == "detail":
            prefix = "[verbose]   "
        elif level == "timing":
            prefix = "[verbose] ⏱"
        elif level == "warning":
            prefix = "[verbose] ⚠"
        elif level == "error":
            prefix = "[verbose] ✗"
        elif level == "success":
            prefix = "[verbose] ✓"

        print(f"{prefix} {message}", file=self._config.output)


# Convenience functions for logging from anywhere


def verbose_log(message: str, level: str = "info") -> None:
    """Log a message if verbose mode is active.

    Args:
        message: Message to log
        level: Log level
    """
    config = get_verbose_context()
    if config and config.enabled:
        prefix = "[verbose]"
        if level == "step":
            prefix = "[verbose] →"
        elif level == "detail":
            prefix = "[verbose]   "
        elif level == "timing":
            prefix = "[verbose] ⏱"
        elif level == "warning":
            prefix = "[verbose] ⚠"
        elif level == "error":
            prefix = "[verbose] ✗"
        elif level == "success":
            prefix = "[verbose] ✓"

        print(f"{prefix} {message}", file=config.output)


def verbose_circuit_info(circuit: Any) -> None:
    """Log circuit information if verbose mode is active."""
    config = get_verbose_context()
    if not config or not config.enabled:
        return

    name = getattr(circuit, "name", "Unknown")
    components = getattr(circuit, "_components", [])
    verbose_log(f"Circuit: {name} ({len(components)} components)", "step")


def verbose_netlist(netlist: str) -> None:
    """Log netlist if verbose mode is active and show_netlist enabled."""
    config = get_verbose_context()
    if not config or not config.enabled or not config.show_netlist:
        return

    verbose_log("Generated netlist:", "step")
    for line in netlist.split("\n"):
        verbose_log(f"  {line}", "detail")


def verbose_analysis(analyses: Any) -> None:
    """Log analysis configuration if verbose mode is active."""
    config = get_verbose_context()
    if not config or not config.enabled or not config.show_analysis:
        return

    for analysis in analyses:
        mode = getattr(analysis, "mode", str(analysis))
        args = getattr(analysis, "args", {})

        if mode == "ac":
            sweep_type = args.get("sweep_type", "dec")
            n = args.get("n", 20)
            fstart = args.get("fstart", 1)
            fstop = args.get("fstop", 1e9)
            verbose_log(
                f"Analysis: AC sweep {sweep_type} {n} pts, {fstart}Hz - {fstop}Hz",
                "step",
            )
        elif mode == "tran":
            tstep = args.get("tstep", 0)
            tstop = args.get("tstop", 0)
            verbose_log(
                f"Analysis: Transient, step={_format_time(tstep)}, "
                f"duration={_format_time(tstop)}",
                "step",
            )
        elif mode == "dc":
            src = args.get("src", "?")
            start = args.get("start", 0)
            stop = args.get("stop", 0)
            verbose_log(f"Analysis: DC sweep {src} from {start}V to {stop}V", "step")
        elif mode == "op":
            verbose_log("Analysis: DC operating point", "step")
        elif mode == "noise":
            output = args.get("output", "?")
            src = args.get("src", "?")
            verbose_log(f"Analysis: Noise at {output} referred to {src}", "step")
        else:
            verbose_log(f"Analysis: {mode} {args}", "step")


def verbose_timing(operation: str, duration: float) -> None:
    """Log timing information if verbose mode is active."""
    config = get_verbose_context()
    if not config or not config.enabled or not config.show_timing:
        return

    verbose_log(f"{operation} completed in {duration:.3f}s", "timing")


def verbose_validation(message: str, passed: bool = True) -> None:
    """Log validation step if verbose mode is active."""
    config = get_verbose_context()
    if not config or not config.enabled or not config.show_validation:
        return

    if passed:
        verbose_log(f"Validation: {message}", "success")
    else:
        verbose_log(f"Validation: {message}", "warning")


def verbose_result(result: Any) -> None:
    """Log result summary if verbose mode is active."""
    config = get_verbose_context()
    if not config or not config.enabled:
        return

    try:
        ds = result.dataset()
        num_vars = len(ds.data_vars)
        # Get first coordinate size
        coord_size = 0
        for coord in ds.coords:
            coord_size = len(ds.coords[coord])
            break

        verbose_log(f"Results: {num_vars} variables, {coord_size} data points", "success")
    except Exception:
        verbose_log("Results: available", "success")


@contextlib.contextmanager
def verbose_section(name: str) -> Generator[None, None, None]:
    """Context manager for timing a section of code.

    Example:
        >>> with verbose_section("Netlist generation"):
        ...     netlist = circuit.build_netlist()
        [verbose] → Starting: Netlist generation
        [verbose] ⏱ Netlist generation completed in 0.002s
    """
    config = get_verbose_context()
    if not config or not config.enabled:
        yield
        return

    verbose_log(f"Starting: {name}", "step")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        verbose_timing(name, elapsed)


def _format_time(seconds: float) -> str:
    """Format time value with appropriate SI prefix."""
    if seconds >= 1:
        return f"{seconds:.3g}s"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.3g}ms"
    elif seconds >= 1e-6:
        return f"{seconds * 1e6:.3g}µs"
    elif seconds >= 1e-9:
        return f"{seconds * 1e9:.3g}ns"
    else:
        return f"{seconds * 1e12:.3g}ps"


__all__ = [
    "VerboseSimulation",
    "VerboseConfig",
    "set_verbose",
    "get_verbose_context",
    "verbose_log",
    "verbose_circuit_info",
    "verbose_netlist",
    "verbose_analysis",
    "verbose_timing",
    "verbose_validation",
    "verbose_result",
    "verbose_section",
]
