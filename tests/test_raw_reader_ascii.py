import os
import tempfile

import numpy as np
from spicelab.io.raw_reader import parse_ngspice_ascii_raw

ASCII = """Title:  rc
Date:   Thu Sep  1 12:00:00 2025
Plotname: Transient Analysis
Flags: real
No. Variables: 3
No. Points: 4
Variables:
        0       time    time
        1       v(in)   voltage
        2       v(out)  voltage
Values:
        0       0.0     0.0     0.0
        1       1e-3    5.0     3.0
        2       2e-3    5.0     4.0
        3       3e-3    5.0     4.5
"""


def test_parse_ascii_raw_minimal() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.raw")
        with open(p, "w") as f:
            f.write(ASCII)
        ts = parse_ngspice_ascii_raw(p)
        assert ts.names == ["time", "v(in)", "v(out)"]
        assert np.isclose(ts.x.values[1], 1e-3)
        assert np.isclose(ts["v(out)"].values[-1], 4.5)
        df = ts.to_dataframe()
        assert list(df.columns) == ["time", "v(in)", "v(out)"]
        assert df.shape == (4, 3)
