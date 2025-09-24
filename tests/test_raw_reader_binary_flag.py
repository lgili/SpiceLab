import os
import tempfile

import pytest
from spicelab.io.raw_reader import parse_ngspice_raw


def test_parse_ngspice_raw_binary_not_supported() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "b.raw")
        with open(p, "wb") as f:
            f.write(b"Title: x\nBinary: yes\n")
        with pytest.raises(NotImplementedError):
            parse_ngspice_raw(p)
