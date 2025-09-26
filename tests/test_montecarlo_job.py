from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from spicelab.analysis import UniformAbs, monte_carlo
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


def _runner_with_values(values: Sequence[float]) -> Callable[[str, Sequence[str]], RunResult]:
    vals = list(values)

    def _runner(netlist: str, directives: Sequence[str]) -> RunResult:
        if not vals:
            raise AssertionError("runner invoked more times than expected")
        val = vals.pop(0)
        td = tempfile.mkdtemp()
        raw = os.path.join(td, "sim.raw")
        with open(raw, "w", encoding="utf-8") as fh:
            fh.write(ASCII_TEMPLATE.format(value=val))
        log = os.path.join(td, "ngspice.log")
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("ok\n")
        art = RunArtifacts(netlist_path=os.path.join(td, "deck.cir"), log_path=log, raw_path=raw)
        return RunResult(artifacts=art, returncode=0, stdout="", stderr="")

    return _runner


def _rc_circuit() -> tuple[Circuit, Resistor]:
    c = Circuit("mc_job")
    v1 = Vdc("1", 1.0)
    r1 = Resistor("1", "1k")
    c.add(v1, r1)
    c.connect(v1.ports[0], r1.ports[0])
    c.connect(r1.ports[1], GND)
    c.connect(v1.ports[1], GND)
    return c, r1


def test_montecarlo_job_handles_and_cache(tmp_path: Path) -> None:
    old = get_run_directives()
    try:
        set_run_directives(_runner_with_values([1.0, 2.0, 3.0]))
        circuit, r1 = _rc_circuit()
        analyses = [AnalysisSpec("op", {})]
        mc = monte_carlo(
            circuit,
            mapping={r1: UniformAbs(0.0)},
            n=3,
            seed=123,
            analyses=analyses,
            cache_dir=tmp_path,
            engine="ngspice",
            workers=2,
        )
        assert mc.handles is not None
        assert len(mc.handles) == 3
        assert mc.job is not None
        assert mc.job.cache_dir is not None
        assert [run.from_cache for run in mc.job.runs] == [False, False, False]
        vals = [float(run.traces["V(n1)"].values[-1]) for run in mc.runs]
        assert vals == [1.0, 2.0, 3.0]

        def _should_not_run(netlist: str, directives: Sequence[str]) -> RunResult:
            raise AssertionError("cached run attempted to execute")

        set_run_directives(_should_not_run)
        circuit2, r2 = _rc_circuit()
        mc_cached = monte_carlo(
            circuit2,
            mapping={r2: UniformAbs(0.0)},
            n=3,
            seed=123,
            analyses=analyses,
            cache_dir=tmp_path,
            engine="ngspice",
            reuse_cache=True,
        )
        assert mc_cached.job is not None
        assert [run.from_cache for run in mc_cached.job.runs] == [True, True, True]
        vals_cached = [float(run.traces["V(n1)"].values[-1]) for run in mc_cached.runs]
        assert vals_cached == vals
    finally:
        set_run_directives(old)
