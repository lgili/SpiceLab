"""Workflow shortcuts for common simulation tasks.

Provides convenience functions and helpers to reduce boilerplate.
"""

from __future__ import annotations

from .simulation import quick_ac, quick_tran

__all__ = [
    "quick_ac",
    "quick_tran",
]
