import os
import tempfile

import pytest
from spicelab.io.raw_reader import parse_ngspice_ascii_raw, parse_ngspice_raw


def test_parse_ascii_missing_values_raises() -> None:
    ascii_bad = """Title:  rc
Date:   Thu Sep  1 12:00:00 2025
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 1
Variables:
        0       time    time
        1       v(n1)   voltage
"""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.raw")
        with open(p, "w", encoding="utf-8") as f:
            f.write(ascii_bad)
        with pytest.raises(ValueError):
            parse_ngspice_ascii_raw(p)


def test_parse_raw_binary_raises_not_implemented() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.raw")
        # put 'Binary:' in the header to trigger detection
        with open(p, "wb") as f:
            f.write(b"Title: test\nBinary: yes\n")
        with pytest.raises(NotImplementedError):
            parse_ngspice_raw(p)
