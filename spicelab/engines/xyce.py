"""Compatibility wrapper around the Xyce CLI simulator."""

from __future__ import annotations

from .xyce_cli import XyceCLISimulator as XyceSimulator

__all__ = ["XyceSimulator"]
