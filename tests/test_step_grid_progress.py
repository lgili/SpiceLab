import os
import tempfile
from collections.abc import Sequence

import pytest
from spicelab.analysis.sweep_grid import run_param_grid
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec
from spicelab.spice.base import RunArtifacts, RunResult
from spicelab.spice.registry import get_run_directives, set_run_directives

ASCII_TEMPLATE = """Title:  op
Date:   Thu Sep  1 12:00:00 2025
Plotname: Operating Point
Flags: real
No. Variables: 2
No. Points: 1
Variables:
        0       time    time
        1       v(n1)   voltage
Values:
        0       0.0     {value}
"""


def test_step_grid_progress_and_order_workers2() -> None:
    old = get_run_directives()
    calls = {"n": 0}

    def fake_runner(net: str, dirs: Sequence[str]) -> RunResult:
        # emit increasing values to detect reordering
        calls["n"] += 1
        val = float(calls["n"])  # 1.0, 2.0, 3.0 ...
        td = tempfile.mkdtemp()
        raw = os.path.join(td, "sim.raw")
        with open(raw, "w") as f:
            f.write(ASCII_TEMPLATE.format(value=val))
        log = os.path.join(td, "ngspice.log")
        with open(log, "w") as f:
            f.write("ok\n")
        art = RunArtifacts(netlist_path=os.path.join(td, "deck.cir"), log_path=log, raw_path=raw)
        return RunResult(artifacts=art, returncode=0, stdout="", stderr="")

    try:
        set_run_directives(fake_runner)
        c = Circuit("step_grid")
        V1 = Vdc("VIN", 1.0)
        R1 = Resistor("R", "1k")
        R2 = Resistor("K", 1.0)
        c.add(V1, R1, R2)
        c.connect(V1.ports[0], R1.ports[0])
        c.connect(R1.ports[1], GND)
        c.connect(V1.ports[1], GND)
        # R2 apenas para ter mais um parâmetro de step no grid (não afeta V(n1))
        c.connect(R2.ports[0], GND)
        c.connect(R2.ports[1], GND)

        result = run_param_grid(
            c,
            variables=[(R1, ["1k", "2k"]), (R2, [1, 2])],
            analyses=[AnalysisSpec("op", {})],
            engine="ngspice",
            workers=2,
            progress=True,
        )
        # Ordem deve preservar o grid na sequência [(1k,1), (1k,2), (2k,1), (2k,2)]
        vals = []
        combos = []
        for run in result.runs:
            ds = run.handle.dataset()
            vals.append(float(ds["V(n1)"].values[-1]))
            combos.append(run.combo)
        assert vals == [1.0, 2.0, 3.0, 4.0]
        # params da última execução devem refletir o último ponto do grid
        last = combos[-1]
        assert pytest.approx(float(last["R"])) == 2000.0
        assert pytest.approx(float(last["K"])) == 2.0
        assert len(result.runs) == 4
    finally:
        set_run_directives(old)
