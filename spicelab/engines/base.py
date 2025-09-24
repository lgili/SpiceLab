from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

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


class Simulator(Protocol):
    """Common engine interface. Concrete backends should implement this Protocol."""

    def features(self) -> EngineFeatures:  # pragma: no cover - trivial accessors
        ...

    def run(
        self,
        circuit: object,
        analyses: Sequence[AnalysisSpec],
        sweep: SweepSpec | None = None,
        probes: list[Probe] | None = None,
    ) -> ResultHandle: ...


__all__ = ["EngineFeatures", "Simulator"]
