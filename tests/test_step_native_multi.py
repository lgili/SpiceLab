import os
import tempfile
from collections.abc import Sequence

from spicelab.analysis import StepNativeResult, run_step_native
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND
from spicelab.spice.base import RunArtifacts, RunResult
from spicelab.spice.registry import get_run_directives, set_run_directives

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
No. Points: 2
Variables:
        0       time    time
        1       v(n1)   voltage
Values:
        0       0.0     0.0
        1       1e-3    2.0
"""


def test_step_native_parses_multiple_plots() -> None:
    old = get_run_directives()

    def fake_runner(net: str, dirs: Sequence[str]) -> RunResult:
        td = tempfile.mkdtemp()
        raw = os.path.join(td, "sim.raw")
        with open(raw, "w") as f:
            f.write(ASCII_MULTI)
        log = os.path.join(td, "ngspice.log")
        with open(log, "w") as f:
            f.write("ok\n")
        art = RunArtifacts(netlist_path=os.path.join(td, "deck.cir"), log_path=log, raw_path=raw)
        return RunResult(artifacts=art, returncode=0, stdout="", stderr="")

    try:
        set_run_directives(fake_runner)
        c = Circuit("native_step")
        V1 = Vdc("1", 1.0)
        R1 = Resistor("1", "1k")
        c.add(V1, R1)
        c.connect(V1.ports[0], R1.ports[0])
        c.connect(R1.ports[1], GND)
        c.connect(V1.ports[1], GND)
        res = run_step_native(c, directives=[".tran 1ms 2ms", ".step param R 1k 2k 1k"])
        assert isinstance(res, StepNativeResult)
        assert len(res.tracesets) == 2
    finally:
        set_run_directives(old)
