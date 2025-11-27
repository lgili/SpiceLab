"""Progress bars for long operations with ETA.

Provides visual feedback during simulation and analysis operations.
"""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TextIO, TypeVar

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

T = TypeVar("T")


class ProgressStyle(Enum):
    """Style options for progress bar display."""

    BAR = auto()  # [████████░░░░░░░░░░░░] 40%
    SPINNER = auto()  # ⠋ Processing...
    DOTS = auto()  # Processing... ...
    MINIMAL = auto()  # 40% (12/30)
    DETAILED = auto()  # [████████░░░░░░░░░░░░] 40% | 12/30 | ETA: 5s


@dataclass
class ProgressConfig:
    """Configuration for progress display."""

    style: ProgressStyle = ProgressStyle.BAR
    width: int = 40
    show_eta: bool = True
    show_count: bool = True
    show_percentage: bool = True
    show_elapsed: bool = False
    output: TextIO = field(default_factory=lambda: sys.stderr)
    refresh_rate: float = 0.1  # seconds between updates
    fill_char: str = "█"
    empty_char: str = "░"
    spinner_chars: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class ProgressBar:
    """Progress bar with ETA estimation.

    Provides visual feedback for long-running operations with
    automatic ETA calculation based on observed progress rate.

    Example:
        >>> from spicelab.ux import ProgressBar
        >>> with ProgressBar(total=100, desc="Simulating") as pbar:
        ...     for i in range(100):
        ...         do_work()
        ...         pbar.update(1)
        Simulating [████████████████████████████████████████] 100% | ETA: 0s

        >>> # Or iterate directly
        >>> for item in ProgressBar.iter(items, desc="Processing"):
        ...     process(item)

        >>> # Custom style
        >>> with ProgressBar(total=50, style=ProgressStyle.DETAILED) as pbar:
        ...     for i in range(50):
        ...         pbar.update(1)
    """

    def __init__(
        self,
        total: int | None = None,
        desc: str = "",
        *,
        style: ProgressStyle = ProgressStyle.BAR,
        config: ProgressConfig | None = None,
        leave: bool = True,
        disable: bool = False,
    ) -> None:
        """Initialize progress bar.

        Args:
            total: Total number of items (None for indeterminate)
            desc: Description text
            style: Display style
            config: Custom configuration
            leave: Leave progress bar visible after completion
            disable: Disable progress display entirely
        """
        self._total = total
        self._desc = desc
        self._style = style
        self._config = config or ProgressConfig(style=style)
        self._leave = leave
        self._disable = disable

        self._current = 0
        self._start_time: float | None = None
        self._last_update: float = 0
        self._spinner_idx = 0
        self._finished = False

    def __enter__(self) -> ProgressBar:
        """Start the progress bar."""
        self._start_time = time.perf_counter()
        self._last_update = 0
        self._current = 0
        self._finished = False
        self._render()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Finish the progress bar."""
        self._finished = True
        if not self._disable:
            self._render(force=True)
            if self._leave:
                self._config.output.write("\n")
            else:
                self._clear_line()
            self._config.output.flush()

    def update(self, n: int = 1) -> None:
        """Update progress by n steps.

        Args:
            n: Number of steps to advance
        """
        self._current += n
        self._render()

    def set_description(self, desc: str) -> None:
        """Update the description text.

        Args:
            desc: New description
        """
        self._desc = desc
        self._render(force=True)

    def set_total(self, total: int) -> None:
        """Set or update the total count.

        Args:
            total: New total
        """
        self._total = total
        self._render(force=True)

    def _render(self, force: bool = False) -> None:
        """Render the progress bar."""
        if self._disable:
            return

        now = time.perf_counter()
        if not force and (now - self._last_update) < self._config.refresh_rate:
            return
        self._last_update = now

        self._clear_line()
        line = self._format_line()
        self._config.output.write(line)
        self._config.output.flush()

    def _clear_line(self) -> None:
        """Clear the current line."""
        self._config.output.write("\r\033[K")

    def _format_line(self) -> str:
        """Format the progress line based on style."""
        if self._style == ProgressStyle.SPINNER:
            return self._format_spinner()
        elif self._style == ProgressStyle.DOTS:
            return self._format_dots()
        elif self._style == ProgressStyle.MINIMAL:
            return self._format_minimal()
        elif self._style == ProgressStyle.DETAILED:
            return self._format_detailed()
        else:  # BAR
            return self._format_bar()

    def _format_bar(self) -> str:
        """Format standard progress bar."""
        parts = []

        if self._desc:
            parts.append(self._desc)

        bar = self._make_bar()
        parts.append(bar)

        if self._config.show_percentage and self._total:
            pct = (self._current / self._total) * 100
            parts.append(f"{pct:.0f}%")

        if self._config.show_eta and self._total and self._start_time:
            eta = self._calculate_eta()
            parts.append(f"ETA: {eta}")

        return " ".join(parts)

    def _format_detailed(self) -> str:
        """Format detailed progress bar."""
        parts = []

        if self._desc:
            parts.append(self._desc)

        bar = self._make_bar()
        parts.append(bar)

        if self._config.show_percentage and self._total:
            pct = (self._current / self._total) * 100
            parts.append(f"{pct:.0f}%")

        parts.append("|")

        if self._config.show_count and self._total:
            parts.append(f"{self._current}/{self._total}")

        if self._config.show_elapsed and self._start_time:
            elapsed = time.perf_counter() - self._start_time
            parts.append(f"Elapsed: {self._format_time(elapsed)}")

        if self._config.show_eta and self._total and self._start_time:
            eta = self._calculate_eta()
            parts.append(f"ETA: {eta}")

        return " | ".join(parts)

    def _format_spinner(self) -> str:
        """Format spinner style."""
        chars = self._config.spinner_chars
        char = chars[self._spinner_idx % len(chars)]
        self._spinner_idx += 1

        parts = [char]
        if self._desc:
            parts.append(self._desc)
        if self._total and self._config.show_count:
            parts.append(f"({self._current}/{self._total})")

        return " ".join(parts)

    def _format_dots(self) -> str:
        """Format dots style."""
        dots = "." * ((self._spinner_idx % 3) + 1)
        self._spinner_idx += 1

        parts = []
        if self._desc:
            parts.append(f"{self._desc}{dots}")
        else:
            parts.append(f"Processing{dots}")

        if self._total and self._config.show_count:
            parts.append(f"({self._current}/{self._total})")

        return " ".join(parts)

    def _format_minimal(self) -> str:
        """Format minimal style."""
        parts = []

        if self._desc:
            parts.append(self._desc)

        if self._total:
            pct = (self._current / self._total) * 100
            parts.append(f"{pct:.0f}%")
            if self._config.show_count:
                parts.append(f"({self._current}/{self._total})")
        else:
            parts.append(f"{self._current}")

        return " ".join(parts)

    def _make_bar(self) -> str:
        """Create the visual progress bar."""
        width = self._config.width
        fill = self._config.fill_char
        empty = self._config.empty_char

        if self._total and self._total > 0:
            filled = int(width * self._current / self._total)
            bar = fill * filled + empty * (width - filled)
        else:
            # Indeterminate: show moving pattern
            pos = self._spinner_idx % width
            self._spinner_idx += 1
            bar = empty * pos + fill * 3 + empty * (width - pos - 3)
            bar = bar[:width]

        return f"[{bar}]"

    def _calculate_eta(self) -> str:
        """Calculate estimated time remaining."""
        if not self._start_time or not self._total or self._current == 0:
            return "?"

        elapsed = time.perf_counter() - self._start_time
        rate = self._current / elapsed if elapsed > 0 else 0

        if rate == 0:
            return "?"

        remaining = self._total - self._current
        eta_seconds = remaining / rate

        return self._format_time(eta_seconds)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 1:
            return "<1s"
        elif seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m{secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h{mins}m"

    @classmethod
    def iter(
        cls,
        iterable: Iterator[T] | list[T],
        desc: str = "",
        total: int | None = None,
        **kwargs: Any,
    ) -> Generator[T, None, None]:
        """Iterate with progress bar.

        Args:
            iterable: Items to iterate
            desc: Description text
            total: Total count (auto-detected if possible)
            **kwargs: Additional ProgressBar arguments

        Yields:
            Items from iterable

        Example:
            >>> for item in ProgressBar.iter(items, "Processing"):
            ...     process(item)
        """
        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)  # type: ignore

        with cls(total=total, desc=desc, **kwargs) as pbar:
            for item in iterable:
                yield item
                pbar.update(1)


@contextlib.contextmanager
def progress_context(
    desc: str = "",
    total: int | None = None,
    **kwargs: Any,
) -> Generator[ProgressBar, None, None]:
    """Context manager for progress bar.

    Args:
        desc: Description text
        total: Total count
        **kwargs: Additional ProgressBar arguments

    Yields:
        ProgressBar instance

    Example:
        >>> with progress_context("Analyzing", total=100) as pbar:
        ...     for i in range(100):
        ...         analyze(data[i])
        ...         pbar.update(1)
    """
    with ProgressBar(total=total, desc=desc, **kwargs) as pbar:
        yield pbar


def with_progress(
    desc: str = "",
    total_attr: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add progress bar to a function.

    Args:
        desc: Description text
        total_attr: Name of argument containing total count

    Returns:
        Decorated function

    Example:
        >>> @with_progress("Simulating")
        ... def simulate(circuit, iterations=100):
        ...     for i in range(iterations):
        ...         run_iteration()
        ...         yield i  # Progress update
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            total = None
            if total_attr and total_attr in kwargs:
                total = kwargs[total_attr]

            # Check if function is a generator
            result = func(*args, **kwargs)

            if hasattr(result, "__iter__") and hasattr(result, "__next__"):
                # Wrap generator with progress
                def gen_wrapper() -> Generator[Any, None, None]:
                    with ProgressBar(total=total, desc=desc) as pbar:
                        for item in result:
                            yield item
                            pbar.update(1)

                return gen_wrapper()  # type: ignore

            return result

        return wrapper

    return decorator


# Simulation-specific progress helpers


class SimulationProgress(ProgressBar):
    """Progress bar specialized for simulations.

    Provides phase-aware progress tracking for multi-step simulations.

    Example:
        >>> with SimulationProgress(phases=["init", "run", "post"]) as pbar:
        ...     pbar.start_phase("init")
        ...     initialize()
        ...     pbar.start_phase("run")
        ...     for i in range(100):
        ...         step()
        ...         pbar.update(1)
        ...     pbar.start_phase("post")
        ...     postprocess()
    """

    def __init__(
        self,
        phases: list[str] | None = None,
        total: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize simulation progress.

        Args:
            phases: List of phase names
            total: Total steps within phases
            **kwargs: Additional ProgressBar arguments
        """
        super().__init__(total=total, **kwargs)
        self._phases = phases or ["initializing", "running", "processing"]
        self._current_phase = 0
        self._phase_progress: dict[str, float] = {}

    def start_phase(self, phase: str) -> None:
        """Start a new simulation phase.

        Args:
            phase: Phase name
        """
        if phase in self._phases:
            self._current_phase = self._phases.index(phase)
        self.set_description(f"[{phase}]")
        self._current = 0

    def complete_phase(self) -> None:
        """Mark current phase as complete."""
        if self._current_phase < len(self._phases):
            phase = self._phases[self._current_phase]
            self._phase_progress[phase] = 1.0
            self._current_phase += 1


__all__ = [
    "ProgressBar",
    "ProgressConfig",
    "ProgressStyle",
    "SimulationProgress",
    "progress_context",
    "with_progress",
]
