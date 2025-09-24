import os
import tempfile

from spicelab.io.raw_reader import parse_ngspice_ascii_raw_multi

ASCII_MULTI = """Title:  plot1
Date:   Thu Sep  1 12:00:00 2025
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 2
Variables:
        0       time    time
        1       v(n1)   voltage
Values:
        0       0.0     0.0
        1       1e-3    1.0

Title:  plot2
Date:   Thu Sep  1 12:00:01 2025
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 3
Variables:
        0       time    time
        1       v(n2)   voltage
Values:
        0       0.0     0.0
        1       1e-3    2.0
        2       2e-3    3.0
"""


def test_parse_ascii_raw_multi_blocks() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.raw")
        with open(p, "w") as f:
            f.write(ASCII_MULTI)
        sets = parse_ngspice_ascii_raw_multi(p)
        assert len(sets) >= 2
        assert "time" in sets[0].names
        assert any(n.startswith("v(") for n in sets[1].names)
