import os
import re
import tempfile
from collections.abc import Sequence

from cat.analysis import OP, worst_case
from cat.analysis.core import AnalysisResult
from cat.core.circuit import Circuit
from cat.core.components import Resistor, Vdc
from cat.core.net import GND
from cat.spice.base import RunArtifacts, RunResult
from cat.spice.registry import get_run_directives, set_run_directives


def _parse_params(directives: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in directives:
        m = re.match(r"\.param\s+(\w+)=([-+\w\.]+)", d)
        if m:
            k, v = m.group(1), m.group(2)
            try:
                out[k] = float(v)
            except Exception:
                pass
    return out


def test_worstcase_min_uses_metric() -> None:
    old = get_run_directives()

    def fake_runner(net: str, dirs: Sequence[str]) -> RunResult:
        p = _parse_params(dirs)
        # Metric will be simply value of R
        val = p.get("R", 0.0)
        td = tempfile.mkdtemp()
        raw = os.path.join(td, "sim.raw")
        content = f"""Title:  op
Date:   Thu Sep  1 12:00:00 2025
Plotname: Operating Point
Flags: real
No. Variables: 2
No. Points: 1
Variables:
        0       time    time
        1       v(n1)   voltage
Values:
        0       0.0     {val}
"""
        with open(raw, "w") as f:
            f.write(content)
        log = os.path.join(td, "ngspice.log")
        with open(log, "w") as f:
            f.write("ok\n")
        art = RunArtifacts(netlist_path=os.path.join(td, "deck.cir"), log_path=log, raw_path=raw)
        return RunResult(artifacts=art, returncode=0, stdout="", stderr="")

    try:
        set_run_directives(fake_runner)
        c = Circuit("wc")
        V1 = Vdc("1", 1.0)
        R1 = Resistor("1", "{R}")
        c.add(V1, R1)
        c.connect(V1.ports[0], R1.ports[0])
        c.connect(R1.ports[1], GND)
        c.connect(V1.ports[1], GND)

        def metric(res: AnalysisResult) -> float:
            return float(res.traces["v(n1)"].values[-1])

        space = {"R": [1.0, 2.0, 3.0]}
        res = worst_case(
            c,
            analysis_factory=lambda: OP(),
            metric=metric,
            space=space,
            mode="min",
            n_random=3,
            n_refine=1,
        )
        assert res.best_params["R"] == 1.0
    finally:
        set_run_directives(old)
