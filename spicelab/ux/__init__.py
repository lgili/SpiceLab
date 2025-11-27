"""User experience utilities for SpiceLab.

This module provides tools for enhanced user experience:
- Progress bars for long operations
- Undo/redo for circuit modifications
- Clipboard support for circuit snippets
- Circuit diff visualization
- Bookmarks/favorites for configurations
"""

from __future__ import annotations

from .bookmarks import (
    Bookmark,
    BookmarkManager,
    list_bookmarks,
    load_bookmark,
    save_bookmark,
)
from .clipboard import (
    CircuitClipboard,
    copy_circuit,
    copy_component,
    paste_circuit,
    paste_component,
)
from .diff import (
    CircuitDiff,
    DiffChange,
    diff_circuits,
)
from .history import (
    CircuitHistory,
    HistoryManager,
    Snapshot,
)
from .progress import (
    ProgressBar,
    progress_context,
    with_progress,
)

__all__ = [
    # Progress
    "ProgressBar",
    "progress_context",
    "with_progress",
    # History (undo/redo)
    "CircuitHistory",
    "HistoryManager",
    "Snapshot",
    # Clipboard
    "CircuitClipboard",
    "copy_circuit",
    "paste_circuit",
    "copy_component",
    "paste_component",
    # Diff
    "CircuitDiff",
    "diff_circuits",
    "DiffChange",
    # Bookmarks
    "Bookmark",
    "BookmarkManager",
    "save_bookmark",
    "load_bookmark",
    "list_bookmarks",
]
