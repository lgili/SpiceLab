"""Backwards-compatibility wrapper for the NGSpice process simulator."""

from __future__ import annotations

from .ngspice_proc import NgSpiceProcSimulator as NgSpiceSimulator

__all__ = ["NgSpiceSimulator"]
