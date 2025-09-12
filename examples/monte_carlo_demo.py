import os
import tempfile
from collections.abc import Callable, Sequence
from typing import Any

from cat.analysis import OP, UniformAbs, monte_carlo
from cat.core.circuit import Circuit
from cat.core.components import Resistor, Vdc
from cat.core.net import GND
from cat.spice.base import RunArtifacts, RunResult
from cat.spice.registry import get_run_directives, set_run_directives

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


def _fake_runner_value(val: float) -> Callable[[str, Sequence[str]], RunResult]:
    def _runner(netlist: str, directives: Sequence[str]) -> RunResult:
        td = tempfile.mkdtemp()
        raw = os.path.join(td, "sim.raw")
        with open(raw, "w") as f:
            f.write(ASCII_TEMPLATE.format(value=val))
        log = os.path.join(td, "ngspice.log")
        with open(log, "w") as f:
            f.write("ok\n")
        art = RunArtifacts(netlist_path=os.path.join(td, "deck.cir"), log_path=log, raw_path=raw)
        return RunResult(artifacts=art, returncode=0, stdout="", stderr="")

    return _runner


def build_demo_circuit() -> tuple[Circuit, Resistor]:
    c = Circuit("mc_demo")
    V1 = Vdc("1", 1.0)
    R1 = Resistor("1", 1000.0)
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(R1.ports[1], GND)
    c.connect(V1.ports[1], GND)
    return c, R1


def run_demo(n: int = 3) -> Any:
    """Run a monte_carlo with a fake runner and return the resulting dataframe."""
    old = get_run_directives()
    try:
        vals = [0.1 * (i + 1) for i in range(n)]
        runners = [_fake_runner_value(v) for v in vals]
        idx = {"i": 0}

        def mux(net: str, dirs: Sequence[str]) -> RunResult:
            i = idx["i"]
            idx["i"] = (i + 1) % len(runners)
            return runners[i](net, dirs)

        set_run_directives(mux)

        c, r = build_demo_circuit()
        mc = monte_carlo(
            c, {r: UniformAbs(0.0)}, n=n, analysis_factory=lambda: OP(), seed=1, workers=1
        )
        df = mc.to_dataframe(metric=None)
        return df
    finally:
        set_run_directives(old)


if __name__ == "__main__":
    df = run_demo(4)
    print(df.head())
