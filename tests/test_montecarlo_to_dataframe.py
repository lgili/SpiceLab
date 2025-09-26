import os
import tempfile
from collections.abc import Callable, Sequence

from spicelab.analysis import UniformAbs, monte_carlo
from spicelab.analysis.result import AnalysisResult
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


def test_montecarlo_dataframe_metrics() -> None:
    old = get_run_directives()
    try:
        # Make runner cycle values deterministically per trial
        runners = [_fake_runner_value(v) for v in (1.0, 2.0, 3.0, 4.0)]
        idx = {0: 0}

        def mux(net: str, dirs: Sequence[str]) -> RunResult:
            i = idx[0]
            idx[0] = (i + 1) % len(runners)
            return runners[i](net, dirs)

        set_run_directives(mux)

        c = Circuit("mc_df")
        V1 = Vdc("1", 1.0)
        R1 = Resistor("1", 1000.0)
        c.add(V1, R1)
        c.connect(V1.ports[0], R1.ports[0])
        c.connect(R1.ports[1], GND)
        c.connect(V1.ports[1], GND)

        def metr(res: AnalysisResult) -> dict[str, float]:
            return {"vout": float(res.traces["V(n1)"].values[-1])}

        mc = monte_carlo(
            c,
            {R1: UniformAbs(0.0)},
            n=4,
            analyses=[AnalysisSpec("op", {})],
            engine="ngspice",
            seed=1,
            workers=1,
        )
        df = mc.to_dataframe(metric=metr, param_prefix="param_")

        assert set(["trial"]) <= set(df.columns)
        # Param column prefixed
        assert "param_Resistor.1" in df.columns
        assert "vout" in df.columns
        assert list(df["vout"]) == [1.0, 2.0, 3.0, 4.0]
    finally:
        set_run_directives(old)


def test_montecarlo_dataframe_sample_at() -> None:
    old = get_run_directives()
    try:
        # produce two points per run: (0,0) and (1e-3, value)
        def runner_factory(val: float) -> Callable[[str, Sequence[str]], RunResult]:
            def _runner(netlist: str, directives: Sequence[str]) -> RunResult:
                td = tempfile.mkdtemp()
                raw = os.path.join(td, "sim.raw")
                content = f"""Title:  tran
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
        1       1e-3    {val}
"""
                with open(raw, "w") as f:
                    f.write(content)
                log = os.path.join(td, "ngspice.log")
                with open(log, "w") as f:
                    f.write("ok\n")
                art = RunArtifacts(
                    netlist_path=os.path.join(td, "deck.cir"), log_path=log, raw_path=raw
                )
                return RunResult(artifacts=art, returncode=0, stdout="", stderr="")

            return _runner

        vals = [0.5, 1.0, 1.5]
        runners = [runner_factory(v) for v in vals]
        idx = {0: 0}

        def mux(net: str, dirs: Sequence[str]) -> RunResult:
            i = idx[0]
            idx[0] = (i + 1) % len(runners)
            return runners[i](net, dirs)

        set_run_directives(mux)

        c = Circuit("mc_df2")
        V1 = Vdc("1", 1.0)
        R1 = Resistor("1", 1000.0)
        c.add(V1, R1)
        c.connect(V1.ports[0], R1.ports[0])
        c.connect(R1.ports[1], GND)
        c.connect(V1.ports[1], GND)

        mc = monte_carlo(
            c,
            {R1: UniformAbs(0.0)},
            n=3,
            analyses=[AnalysisSpec("op", {})],
            engine="ngspice",
            seed=1,
            workers=1,
        )
        df = mc.to_dataframe(metric=None, y=["V(n1)"], sample_at=1e-3)
        assert "V(n1)" in df.columns
        assert list(df["V(n1)"]) == vals
    finally:
        set_run_directives(old)
