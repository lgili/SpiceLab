"""Public exports for :mod:`spicelab.io` used by docs tooling.

Keep explicit exports to avoid star-imports which trigger linter errors.
"""

from .raw_reader import (
    Trace,
    TraceSet,
    parse_ngspice_ascii_raw,
    parse_ngspice_ascii_raw_multi,
    parse_ngspice_raw,
)
from .readers import (
    dataset_to_long_polars,
    list_signals,
    load_dataset,
    load_saved_dataset,
    read,
    read_ltspice,
    read_ltspice_raw,
    read_ngspice_log,
    read_ngspice_raw,
    read_ngspice_raw_multi,
    read_waveform,
    read_xyce_csv,
    read_xyce_prn,
    read_xyce_table,
    save_dataset,
    to_pandas,
    to_polars,
)

__all__ = [
    "Trace",
    "TraceSet",
    "parse_ngspice_raw",
    "parse_ngspice_ascii_raw",
    "parse_ngspice_ascii_raw_multi",
    "read_ltspice_raw",
    "read",
    "read_ltspice",
    "read_xyce_table",
    "read_waveform",
    "read_ngspice_raw",
    "read_ngspice_raw_multi",
    "read_ngspice_log",
    "read_xyce_prn",
    "read_xyce_csv",
    "to_polars",
    "to_pandas",
    "load_dataset",
    "list_signals",
    "dataset_to_long_polars",
    "save_dataset",
    "load_saved_dataset",
]
