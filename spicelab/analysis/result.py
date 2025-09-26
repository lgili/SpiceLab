from __future__ import annotations

from dataclasses import dataclass

from ..io.raw_reader import TraceSet
from ..spice.base import RunResult


@dataclass(frozen=True)
class AnalysisResult:
    """Lightweight container used by Monte Carlo helpers for legacy traces."""

    run: RunResult
    traces: TraceSet


__all__ = ["AnalysisResult"]
