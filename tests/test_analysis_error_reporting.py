import os
import tempfile
from collections.abc import Sequence

import pytest
from spicelab.analysis import AC
from spicelab.core.circuit import Circuit
from spicelab.core.components import VA, Resistor
from spicelab.core.net import GND
from spicelab.spice.base import RunArtifacts, RunResult
from spicelab.spice.registry import get_run_directives, set_run_directives


def _make_ascii_raw(path: str) -> None:
    content = """Title:  ac
Date:   Thu Sep  1 12:00:00 2025
Plotname: AC Analysis
Flags: complex
No. Variables: 2
No. Points: 2
Variables:
        0       frequency       frequency
        1       v(n1)  voltage
Values:
        0       1e3     1.0,0.0
        1       2e3     0.5,0.0
"""
    with open(path, "w") as f:
        f.write(content)


def test_analysis_includes_log_errors_in_exception() -> None:
    old = get_run_directives()

    def fake_runner(netlist: str, directives: Sequence[str]) -> RunResult:
        td = tempfile.mkdtemp()
        log = os.path.join(td, "ngspice.log")
        with open(log, "w") as f:
            f.write("%Error: some failure\n")
        art = RunArtifacts(netlist_path=os.path.join(td, "deck.cir"), log_path=log, raw_path=None)
        return RunResult(artifacts=art, returncode=1, stdout="", stderr="boom")

    try:
        set_run_directives(fake_runner)
        c = Circuit("err")
        v = VA()
        r = Resistor("1", "1k")
        c.add(v, r)
        c.connect(v.ports[0], r.ports[0])
        c.connect(v.ports[1], GND)
        c.connect(r.ports[1], GND)
        with pytest.raises(RuntimeError) as ei:
            AC("dec", 2, 1e3, 2e3).run(c)
        assert "Errors:" in str(ei.value)
    finally:
        set_run_directives(old)


def test_analysis_runs_with_fake_ascii_raw() -> None:
    old = get_run_directives()

    def fake_runner2(netlist: str, directives: Sequence[str]) -> RunResult:
        td = tempfile.mkdtemp()
        raw = os.path.join(td, "sim.raw")
        _make_ascii_raw(raw)
        log = os.path.join(td, "ngspice.log")
        with open(log, "w") as f:
            f.write("ok\n")
        art = RunArtifacts(netlist_path=os.path.join(td, "deck.cir"), log_path=log, raw_path=raw)
        return RunResult(artifacts=art, returncode=0, stdout="", stderr="")

    try:
        set_run_directives(fake_runner2)
        c = Circuit("ok")
        v = VA()
        r = Resistor("1", "1k")
        c.add(v, r)
        c.connect(v.ports[0], r.ports[0])
        c.connect(v.ports[1], GND)
        c.connect(r.ports[1], GND)
        res = AC("dec", 2, 1e3, 2e3).run(c)
        assert "frequency" in res.traces.names
    finally:
        set_run_directives(old)
