"""Help and documentation utilities for SpiceLab.

This module provides:
- Context-sensitive help for circuits and results
- Interactive tutorial mode
- API cheat sheet generation
- Docstring example validation
"""

from __future__ import annotations

from .cheatsheet import (
    CheatsheetFormat,
    generate_cheatsheet,
)
from .context_help import (
    CircuitHelp,
    ComponentHelp,
    Help,
    ResultHelp,
    get_help,
    show_help,
)
from .examples import (
    list_examples,
    run_example,
    validate_docstring_examples,
)
from .tutorial import (
    Tutorial,
    TutorialStep,
    list_tutorials,
    run_tutorial,
)

__all__ = [
    # Context help
    "Help",
    "CircuitHelp",
    "ResultHelp",
    "ComponentHelp",
    "get_help",
    "show_help",
    # Tutorial
    "Tutorial",
    "TutorialStep",
    "run_tutorial",
    "list_tutorials",
    # Cheatsheet
    "generate_cheatsheet",
    "CheatsheetFormat",
    # Examples
    "validate_docstring_examples",
    "run_example",
    "list_examples",
]
