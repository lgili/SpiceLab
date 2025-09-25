"""Compatibility wrapper around the LTspice CLI simulator."""

from __future__ import annotations

from .ltspice_cli import LtSpiceCLISimulator as LtSpiceSimulator

__all__ = ["LtSpiceSimulator"]
