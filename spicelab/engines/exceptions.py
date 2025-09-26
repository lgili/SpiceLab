from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["EngineBinaryNotFound", "EngineSharedLibraryNotFound"]


@dataclass
class EngineBinaryNotFound(RuntimeError):
    """Raised when an external SPICE engine binary cannot be located.

    Attributes
    ----------
    engine: canonical engine name (ngspice|ltspice|xyce)
    attempted: optional list of paths or strategies tried
    hints: human readable installation hints
    """

    engine: str
    hints: Iterable[str]
    attempted: Iterable[str] | None = None

    def __post_init__(self) -> None:
        hints_txt = "\n".join(f"  - {h}" for h in self.hints)
        tried_txt = "\nTried: " + ", ".join(self.attempted) if self.attempted else ""
        msg = (
            f"{self.engine} binary not found.{tried_txt}\n"
            f"Install or configure it. Hints:\n{hints_txt}"
        )
        super().__init__(msg)


@dataclass
class EngineSharedLibraryNotFound(RuntimeError):
    """Raised when a required shared library (e.g. libngspice) is missing."""

    engine: str
    hints: Iterable[str]
    attempted: Iterable[str] | None = None

    def __post_init__(self) -> None:
        hints_txt = "\n".join(f"  - {h}" for h in self.hints)
        tried_txt = "\nTried: " + ", ".join(self.attempted) if self.attempted else ""
        msg = (
            f"{self.engine} shared library not found.{tried_txt}\n"
            f"Install or configure it. Hints:\n{hints_txt}"
        )
        super().__init__(msg)
