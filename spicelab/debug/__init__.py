"""Debugging and interactive features for SpiceLab.

This module provides tools for debugging simulations:
- VerboseSimulation: Context manager for verbose logging
- dry_run: Validate simulation setup without running
- SimulationDebugger: Step-by-step simulation control
- InteractiveSession: Context manager for interactive prompts
"""

from __future__ import annotations

from .debugger import DebugStep, SimulationDebugger
from .dry_run import DryRunResult, dry_run
from .interactive import (
    Choice,
    InteractiveMode,
    InteractivePrompt,
    InteractiveSession,
    get_interactive_context,
    prompt_analysis_type,
    prompt_choice,
    prompt_confirm,
    prompt_frequency_range,
    prompt_simulation_engine,
    prompt_value,
    set_interactive_mode,
)
from .verbose import VerboseSimulation, get_verbose_context, set_verbose

__all__ = [
    # Verbose mode
    "VerboseSimulation",
    "set_verbose",
    "get_verbose_context",
    # Dry-run
    "dry_run",
    "DryRunResult",
    # Debugger
    "SimulationDebugger",
    "DebugStep",
    # Interactive mode
    "InteractiveMode",
    "InteractiveSession",
    "InteractivePrompt",
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
