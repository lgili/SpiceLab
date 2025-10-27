from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..core.types import AnalysisSpec, Probe, ResultHandle, SweepSpec


@dataclass(frozen=True)
class EngineFeatures:
    """Capabilities advertised by an engine backend."""

    name: str
    supports_callbacks: bool = False
    supports_shared_lib: bool = False
    supports_noise: bool = False
    supports_verilog_a: bool = False
    supports_parallel: bool = False

    def satisfies(self, required: EngineFeatures) -> bool:
        """Check if this engine satisfies required features.

        Args:
            required: Features that must be supported

        Returns:
            True if all required features are supported

        Example:
            >>> engine_features = EngineFeatures("ngspice-shared", supports_callbacks=True)
            >>> required = EngineFeatures("", supports_callbacks=True)
            >>> engine_features.satisfies(required)
            True
        """
        if required.supports_callbacks and not self.supports_callbacks:
            return False
        if required.supports_shared_lib and not self.supports_shared_lib:
            return False
        if required.supports_noise and not self.supports_noise:
            return False
        if required.supports_verilog_a and not self.supports_verilog_a:
            return False
        if required.supports_parallel and not self.supports_parallel:
            return False
        return True


@runtime_checkable
class Simulator(Protocol):
    """Common engine interface. Concrete backends should implement this Protocol.

    Phase 3: Added context manager support for resource cleanup (shared libs, temp dirs).
    """

    def features(self) -> EngineFeatures:  # pragma: no cover - trivial accessors
        ...

    def run(
        self,
        circuit: object,
        analyses: Sequence[AnalysisSpec],
        sweep: SweepSpec | None = None,
        probes: list[Probe] | None = None,
    ) -> ResultHandle: ...

    def __enter__(self) -> Simulator:
        """Initialize engine resources (load shared lib, create temp dirs).

        Returns:
            Self for context manager protocol
        """
        ...

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Cleanup engine resources (unload shared lib, remove temp dirs)."""
        ...


__all__ = ["EngineFeatures", "Simulator"]
