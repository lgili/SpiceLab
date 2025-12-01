"""Workflow shortcuts for common simulation tasks.

Provides convenience functions and helpers to reduce boilerplate.

Analysis presets:
- quick_op: DC operating point analysis
- quick_ac: AC frequency sweep with sensible defaults
- quick_tran: Transient analysis with auto timestep
- quick_noise: Noise analysis with sensible defaults
- detailed_ac: AC sweep with high point density
- detailed_tran: Transient with fine timestep

Auto-detection:
- suggest_analysis: Suggest suitable analysis based on circuit topology

Method chaining:
- ChainableResult: Fluent wrapper for result.pm().bw().plot() workflows
- wrap_result: Helper to wrap ResultHandle for chaining
"""

from __future__ import annotations

from .chainable import ChainableResult, MeasurementResult, wrap_result
from .simulation import (
    AnalysisSuggestion,
    detailed_ac,
    detailed_tran,
    quick_ac,
    quick_noise,
    quick_op,
    quick_tran,
    suggest_analysis,
)

__all__ = [
    # Analysis presets
    "quick_op",
    "quick_ac",
    "quick_tran",
    "quick_noise",
    "detailed_ac",
    "detailed_tran",
    # Auto-detection
    "suggest_analysis",
    "AnalysisSuggestion",
    # Method chaining
    "ChainableResult",
    "MeasurementResult",
    "wrap_result",
]
