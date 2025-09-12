"""Public exports for :mod:`cat.io` used by docs tooling.

Keep explicit exports to avoid star-imports which trigger linter errors.
"""

from .raw_reader import (
    Trace,
    TraceSet,
    parse_ngspice_ascii_raw,
    parse_ngspice_ascii_raw_multi,
    parse_ngspice_raw,
)

__all__ = [
    "Trace",
    "TraceSet",
    "parse_ngspice_raw",
    "parse_ngspice_ascii_raw",
    "parse_ngspice_ascii_raw_multi",
]
